"""propella-1 (ellamind, arXiv 2602.12414): 18 document properties per doc.

Adapted from steered-dense-retrieval/scripts/23_annotate_propella.py,
reduced to one GPU per process.

  --backend transformers  (default) plain transformers, no server. Slower
      (roughly 1-3 docs/s on an A100), and without grammar constraints a few
      generations may be invalid JSON -> stored with ok=False.
  --backend sglang        starts an SGLang server on a free port for this
      process (or uses --base-url), which enforces the JSON schema and is
      several times faster. Defaults: --concurrency 240 (just below the
      server's 256 running requests), --batch-size 2048.

Prompts and schema come from the model repo's own propella.py.
Output per document: the 18 properties + ok (parsed and schema-valid).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from loguru import logger

SINGLE = ["content_integrity", "content_ratio", "content_length",
          "information_density", "content_quality", "audience_level",
          "commercial_bias", "time_sensitivity", "content_safety",
          "educational_value", "reasoning_indicators", "pii_presence"]
MULTI = ["content_type", "business_sector", "technical_content",
         "regional_relevance", "country_relevance"]
FREE_TEXT = "one_sentence_description"
FIELDS = [FREE_TEXT] + SINGLE + MULTI
SGLANG_CONTEXT = 65536  # server --context-length (model maximum)


def add_args(ap) -> None:
    ap.add_argument("--model", default="ellamind/propella-1-4b")
    ap.add_argument("--backend", default="transformers", choices=["transformers", "sglang"])
    ap.add_argument("--batch-size", type=int, default=None,
                    help="docs per generate() call (transformers, default 8) or per checkpoint (sglang, default 2048)")
    ap.add_argument("--max-chars", type=int, default=50000, help="model card's recommended truncation")
    # transformers backend (--max-new-tokens applies to both)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-input-tokens", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=640,
                    help="output limit; the JSON answer is ~150-300 tokens. Also sent to SGLang, whose "
                         "scheduler otherwise reserves room for context-length outputs and admits "
                         "new requests one at a time")
    # sglang backend
    ap.add_argument("--base-url", default=None, help="use a running server instead of starting one")
    ap.add_argument("--fp8", action="store_true", help="fp8 weights + KV cache (H100, L40S; not A100)")
    # Must stay below the server's --max-running-requests (256): at 256 running
    # requests, a mixed prefill+decode step builds a batch of 257 and SGLang
    # crashes ("size of tensor a (257) must match ... (256)"), seen on cn19.
    ap.add_argument("--concurrency", type=int, default=240,
                    help="parallel requests; keep below 256 (server limit)")
    ap.add_argument("--sglang-args", default="",
                    help='extra server flags, e.g. "--attention-backend triton --sampling-backend pytorch"')


def load_propella_module(model: str):
    """Import the model repo's own propella.py (prompts + JSON schema).

    propella.py reads property_descriptions.md via a relative path at import
    time, so we fetch it into the same snapshot dir and import from there."""
    import importlib.util
    import os

    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(model, "propella.py"))
    hf_hub_download(model, "property_descriptions.md")
    cwd = os.getcwd()
    try:
        os.chdir(path.parent)
        spec = importlib.util.spec_from_file_location("propella", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def to_row(ann: dict | None, propella) -> dict:
    """Fixed set of columns, whatever the model produced."""
    if not isinstance(ann, dict):
        return {**{k: None for k in FIELDS}, "ok": False}
    row = {}
    for k in FIELDS:
        v = ann.get(k)
        if k in MULTI:
            v = [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v is not None else None)
        elif v is not None:
            v = str(v)
        row[k] = v
    try:
        propella.AnnotationResponse.model_validate(ann)
        row["ok"] = True
    except Exception:
        row["ok"] = False
    return row


def load(args):
    if args.batch_size is None:  # backend-dependent default (run.py reads it after load)
        args.batch_size = 2048 if args.backend == "sglang" else 8
    propella = load_propella_module(args.model)
    if args.backend == "transformers":
        return load_transformers(args, propella)
    return load_sglang(args, propella)


def load_transformers(args, propella):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched generation
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(device).eval()

    # Budget tokens for the document, not the formatted prompt: truncating the
    # prompt would cut the generation marker and the model emits garbage.
    probe = tok.apply_chat_template(propella.create_messages(""), tokenize=False, add_generation_prompt=True)
    overhead = len(tok(probe, add_special_tokens=False)["input_ids"])
    doc_budget = max(256, args.max_input_tokens - overhead - 8)
    logger.info(f"loaded {args.model} on {device}, {doc_budget} tokens per document")

    @torch.no_grad()
    def annotate(texts: list[str]) -> list[dict]:
        prompts = []
        for text in texts:
            text = text[: args.max_chars]
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > doc_budget:
                text = tok.decode(ids[:doc_budget], skip_special_tokens=True)
            prompts.append(tok.apply_chat_template(propella.create_messages(text),
                                                   tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [to_row(parse_json(tok.decode(g, skip_special_tokens=True)), propella) for g in gen]

    return annotate


def start_sglang(args) -> str:
    """Start an SGLang server for this process; it is killed when we exit."""
    import atexit
    import os
    import socket
    import subprocess
    import time
    import urllib.request

    from pipeline.common import DATA

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", args.model,
           "--host", "127.0.0.1", "--port", str(port), "--context-length", str(SGLANG_CONTEXT),
           "--max-running-requests", "256", "--chunked-prefill-size", "8192",
           "--enable-mixed-chunk", "--num-continuous-decode-steps", "8",
           "--grammar-backend", "llguidance", "--mem-fraction-static", "0.7"]
    if args.fp8:
        cmd += ["--quantization", "w8a8_fp8", "--kv-cache-dtype", "fp8_e4m3"]
    cmd += args.sglang_args.split()
    log_path = DATA / "logs" / f"sglang-{socket.gethostname().strip()}-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The venv's bin dir on PATH even when the venv isn't activated: flashinfer
    # calls `ninja` from there to compile kernels on first start.
    # PYTHONUNBUFFERED: the server log is written as it happens, so the
    # "waiting for sglang" lines below can show its latest line.
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env)
    atexit.register(proc.terminate)
    logger.info(f"starting sglang on port {port}, log: {log_path}")

    t0 = time.time()
    for k in range(240):  # up to 20 min (first run downloads the model, compiles kernels)
        if proc.poll() is not None:
            raise RuntimeError(f"sglang server exited, see {log_path}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            logger.info(f"sglang server ready after {time.time() - t0:.0f}s")
            return f"http://127.0.0.1:{port}/v1"
        except Exception:
            log_text = log_path.read_text(errors="replace")
            # A crashed server does not always exit (stray threads), so poll()
            # alone can miss it.
            if "Traceback (most recent call last)" in log_text:
                proc.terminate()
                last = log_text.strip().splitlines()[-1]
                raise RuntimeError(f"sglang server crashed: {last}  (full log: {log_path})")
            if k % 6 == 5:  # every ~30s: show what the server is doing
                lines = log_text.strip().splitlines()
                last = lines[-1][-150:] if lines else ""
                logger.info(f"waiting for sglang ({time.time() - t0:.0f}s): {last}")
            time.sleep(5)
    raise RuntimeError(f"sglang server did not come up, see {log_path}")


def load_sglang(args, propella):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from openai import BadRequestError, OpenAI
    from transformers import AutoTokenizer

    client = OpenAI(base_url=args.base_url or start_sglang(args), api_key="EMPTY", timeout=600)
    schema = propella.get_annotation_response_schema(flatten=True, compact_whitespace=True)
    pool = ThreadPoolExecutor(max_workers=args.concurrency)

    # The model card truncates by characters (50k), but token-dense text
    # (symbols, emoji, base64, some scripts) can exceed the 65,536-token context
    # even then; seen on stackexchange with 99k tokens. Cap the document's
    # tokens too, leaving room for the prompt and the answer.
    tok = AutoTokenizer.from_pretrained(args.model)
    tok_lock = threading.Lock()  # fast tokenizers are not safe across threads
    doc_budget = SGLANG_CONTEXT - 2048 - args.max_new_tokens

    def fit(text: str) -> str:
        text = text[: args.max_chars]
        if len(text) * 4 <= doc_budget:  # a character is at most 4 tokens (byte-level BPE)
            return text
        with tok_lock:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) <= doc_budget:
                return text
            return tok.decode(ids[:doc_budget]) + "\n" + propella.TRUNCATION_TAG

    def one(text: str) -> dict:
        # Server/connection errors propagate and stop the run (the shard stays
        # partial and resumes); a request rejected for this document alone, or
        # an unusable output, is stored as ok=False.
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=propella.create_messages(fit(text), max_content_chars=args.max_chars),
                max_tokens=args.max_new_tokens,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "AnnotationResponse", "schema": schema, "strict": True}},
            )
        except BadRequestError as e:
            logger.warning(f"document rejected by the server, stored as ok=False: {e}")
            return to_row(None, propella)
        return to_row(parse_json(resp.choices[0].message.content or ""), propella)

    def annotate(texts: list[str]):
        return pool.map(one, texts)  # iterator, in order: lets the progress bar move per document

    return annotate
