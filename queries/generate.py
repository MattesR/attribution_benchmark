#!/usr/bin/env python3
"""Generate Comma continuations for the selected prompts: the benchmark queries.

    python -m queries.generate                                   # starts SGLang on the visible GPU(s)
    CUDA_VISIBLE_DEVICES=1,2,3 python -m queries.generate --sglang-args "--dp-size 3"
    python -m queries.generate --base-url http://127.0.0.1:30000/v1   # running server (SGLang or vLLM)

Input data/queries/prompts.parquet (queries.wildchat); output
data/queries/generations/<run>.parquet, one row per (prompt_id, sample), plus
<run>.manifest.json with model revision, frame and sampling parameters.

Comma v0.1 is a base model with a 4,096-token context. Each request is put
into a dialogue frame (queries/templates.py) and the model continues it until
it starts the next user turn (stop string), hits EOS or --max-tokens. If frame
+ request leave less than --max-tokens, the answer gets the remaining room
(at least --min-tokens; otherwise the prompt is skipped and counted in the
manifest).

Checkpoint: <run>.partial.jsonl, appended per prompt; a killed run resumes
(only with the same settings, kept in <run>.params.json). With --limit the
checkpoint stays and the output is <run>.preview.parquet.
Decoding is sampled (--temperature, --top-p), so --n > 1 gives several
continuations per prompt. Seeds are not fixed: the stored outputs are the
benchmark, not the procedure.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from tqdm import tqdm

from queries import templates
from queries.store import PROMPTS, generations_path, open_append, read_jsonl, write_manifest, write_parquet

MODEL = "common-pile/comma-v0.1-2t"
CONTEXT = 4096


def repetition(text: str, n: int = 4) -> float:
    """Share of repeated word n-grams (0 = none; degenerate loops approach 1)."""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return round(1 - len(set(grams)) / len(grams), 4) if grams else 0.0


def resolve_revision(model: str, revision: str | None) -> str:
    if revision:
        return revision
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model).sha
    except Exception as e:  # offline, or a server-side model name
        logger.warning(f"could not resolve the revision of {model}: {e}")
        return "unknown"


def token_counter(name: str | None):
    if not name or name == "none":
        logger.warning("no tokenizer: estimating tokens as chars / 3")
        return lambda s: len(s) // 3 + 1
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    return lambda s: len(tok(s, add_special_tokens=True)["input_ids"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=None, help="model commit; resolved from the hub if omitted")
    ap.add_argument("--tokenizer", default=None, help="for the length check; default: --model; 'none' estimates")
    ap.add_argument("--run", default=None, help="output name; default comma2t-<template>")
    ap.add_argument("--template", default="fewshot", choices=list(templates.TEMPLATES))
    ap.add_argument("--tasks", default=None, help="comma-separated tasks to generate for (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="prototype: first N prompts only")
    ap.add_argument("--n", type=int, default=1, help="continuations per prompt")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--min-tokens", type=int, default=128, help="skip prompts that leave less room than this")
    ap.add_argument("--context", type=int, default=CONTEXT)
    ap.add_argument("--base-url", default=None, help="use a running OpenAI-compatible server")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--sglang-args", default="", help='extra server flags, e.g. "--dp-size 3"')
    args = ap.parse_args()
    run = args.run or f"comma2t-{args.template}"
    out = generations_path(run)
    partial = out.with_suffix(".partial.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    prompts = pq.read_table(PROMPTS).to_pylist()
    if args.tasks:
        prompts = [p for p in prompts if p["task"] in args.tasks.split(",")]
    if args.limit:
        prompts = prompts[: args.limit]

    count = token_counter(args.tokenizer or args.model)
    jobs, skipped = [], []
    for p in prompts:
        text = templates.render(args.template, p["prompt"])
        room = args.context - count(text) - 8
        if room < args.min_tokens:
            skipped.append(p["prompt_id"])
            continue
        jobs.append((p, text, min(args.max_tokens, room)))
    params = {k: getattr(args, k) for k in ["model", "revision", "template", "n", "temperature", "top_p",
                                            "max_tokens", "context"]}
    params_file = out.with_suffix(".params.json")
    if partial.exists() and params_file.exists() and json.loads(params_file.read_text()) != params:
        raise SystemExit(f"{partial} was generated with other settings ({params_file}); "
                         "use another --run name or delete the checkpoint")
    params_file.write_text(json.dumps(params, indent=2) + "\n")
    done = {r["prompt_id"] for r in read_jsonl(partial)}
    todo = [j for j in jobs if j[0]["prompt_id"] not in done]
    logger.info(f"{len(prompts)} prompts: {len(skipped)} too long for the context, {len(done)} done, "
                f"{len(todo)} to generate (n={args.n})")

    if todo:
        from openai import BadRequestError, OpenAI
        if args.base_url:
            base_url = args.base_url
        else:
            from queries.server import start_sglang
            base_url = start_sglang(args.model, args.context, args.sglang_args.split(), args.revision)
        client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=900)
        stop = templates.stops(args.template)

        def one(job):
            p, text, max_tokens = job
            try:
                resp = client.completions.create(model=args.model, prompt=text, n=args.n, max_tokens=max_tokens,
                                                 temperature=args.temperature, top_p=args.top_p, stop=stop)
            except BadRequestError as e:
                logger.warning(f"{p['prompt_id']}: rejected by the server: {e}")
                return [{"prompt_id": p["prompt_id"], "sample": i, "text": None, "finish_reason": "rejected",
                         "max_tokens": max_tokens} for i in range(args.n)]
            return [{"prompt_id": p["prompt_id"], "sample": c.index, "text": c.text,
                     "finish_reason": c.finish_reason, "max_tokens": max_tokens}
                    for c in sorted(resp.choices, key=lambda c: c.index)]

        with open_append(partial) as sink, ThreadPoolExecutor(args.concurrency) as pool:
            for rows in tqdm(pool.map(one, todo), total=len(todo), desc=run, unit="prompt"):
                for r in rows:
                    sink.write(json.dumps(r) + "\n")
                sink.flush()

    rows = {(r["prompt_id"], r["sample"]): r for r in read_jsonl(partial)}
    if len({k[0] for k in rows}) < len(jobs):
        logger.info(f"{len(rows)} generations so far, run stays partial")
        return
    by_id = {p["prompt_id"]: p for p in prompts}
    final = []
    for (pid, sample), r in sorted(rows.items()):
        if pid not in by_id:
            continue
        text = (r["text"] or "").strip()
        # A frame-style continuation the stop string missed (e.g. "User :")
        text = re.split(r"\n\s*User\s*:", text)[0].rstrip() if args.template == "fewshot" else text
        final.append({"query_id": f"{pid}:{sample}", "prompt_id": pid, "sample": sample,
                      "task": by_id[pid]["task"], "stratum": by_id[pid].get("stratum"),
                      "task_prob": by_id[pid].get("task_prob"), "openness": by_id[pid].get("openness"),
                      "openness_prob": by_id[pid].get("openness_prob"), "prompt": by_id[pid]["prompt"],
                      "generation": text, "finish_reason": r["finish_reason"], "max_tokens": r["max_tokens"],
                      "n_chars": len(text), "empty": not text, "repetition": repetition(text)})
    if args.limit:  # prototype: the checkpoint stays, so the full run continues from it
        out = out.with_name(f"{run}.preview.parquet")
    write_parquet(final, out)
    fin = {}
    for r in final:
        fin[r["finish_reason"]] = fin.get(r["finish_reason"], 0) + 1
    write_manifest(out, model=args.model, revision=resolve_revision(args.model, args.revision),
                   template=args.template, frame=templates.TEMPLATES[args.template][0],
                   stop=templates.stops(args.template), n=args.n, temperature=args.temperature,
                   top_p=args.top_p, max_tokens=args.max_tokens, context=args.context,
                   n_prompts=len(jobs), skipped_too_long=skipped, finish_reasons=fin,
                   n_empty=sum(r["empty"] for r in final),
                   n_repetitive=sum(r["repetition"] > 0.5 for r in final))
    if not args.limit:
        partial.unlink()
    logger.success(f"{len(final)} generations -> {out}  (finish reasons: {fin})")


if __name__ == "__main__":
    main()
