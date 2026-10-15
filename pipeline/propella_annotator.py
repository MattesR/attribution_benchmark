"""propella-1 (ellamind, arXiv 2602.12414): 18 document properties per doc.

Adapted from steered-dense-retrieval/scripts/23_annotate_propella.py,
reduced to one GPU per process.

  --backend transformers  (default) plain transformers, no server. Slower
      (roughly 1-3 docs/s on an A100), and without grammar constraints a few
      generations may be invalid JSON -> stored with ok=False.
  --backend sglang        starts an SGLang server on a free port for this
      process (or uses --base-url), which enforces the JSON schema and is
      several times faster. Use a large --batch-size (e.g. 512) so the
      server has enough concurrent requests.

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


def add_args(ap) -> None:
    ap.add_argument("--model", default="ellamind/propella-1-4b")
    ap.add_argument("--backend", default="transformers", choices=["transformers", "sglang"])
    ap.add_argument("--batch-size", type=int, default=8,
                    help="docs per generate() call (transformers) or per checkpoint (sglang)")
    ap.add_argument("--max-chars", type=int, default=50000, help="model card's recommended truncation")
    # transformers backend
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-input-tokens", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    # sglang backend
    ap.add_argument("--base-url", default=None, help="use a running server instead of starting one")
    ap.add_argument("--fp8", action="store_true", help="fp8 weights + KV cache (H100, L40S; not A100)")
    ap.add_argument("--concurrency", type=int, default=128)


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
    import socket
    import subprocess
    import time
    import urllib.request

    from pipeline.common import DATA

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", args.model,
           "--host", "127.0.0.1", "--port", str(port), "--context-length", "65536",
           "--max-running-requests", "256", "--chunked-prefill-size", "8192",
           "--enable-mixed-chunk", "--num-continuous-decode-steps", "8",
           "--grammar-backend", "llguidance", "--mem-fraction-static", "0.7"]
    if args.fp8:
        cmd += ["--quantization", "w8a8_fp8", "--kv-cache-dtype", "fp8_e4m3"]
    log_path = DATA / "logs" / f"sglang-{socket.gethostname()}-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    atexit.register(proc.terminate)
    logger.info(f"starting sglang on port {port}, log: {log_path}")

    for _ in range(240):  # up to 20 min (first run downloads the model)
        if proc.poll() is not None:
            raise RuntimeError(f"sglang server exited, see {log_path}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return f"http://127.0.0.1:{port}/v1"
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"sglang server did not come up, see {log_path}")


def load_sglang(args, propella):
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url or start_sglang(args), api_key="EMPTY", timeout=600)
    schema = propella.get_annotation_response_schema(flatten=True, compact_whitespace=True)
    pool = ThreadPoolExecutor(max_workers=args.concurrency)

    def one(text: str) -> dict:
        # Server/connection errors propagate and stop the run (the shard stays
        # partial and resumes); only unusable outputs are stored as ok=False.
        resp = client.chat.completions.create(
            model=args.model,
            messages=propella.create_messages(text, max_content_chars=args.max_chars),
            response_format={"type": "json_schema", "json_schema": {
                "name": "AnnotationResponse", "schema": schema, "strict": True}},
        )
        return to_row(parse_json(resp.choices[0].message.content or ""), propella)

    def annotate(texts: list[str]) -> list[dict]:
        return list(pool.map(one, texts))

    return annotate
