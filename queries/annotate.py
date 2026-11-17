#!/usr/bin/env python3
"""Annotate the generations with the corpus annotators, so queries and documents
share one label space.

    python -m queries.annotate propella --run comma2t-fewshot --backend sglang
    VENV=.venv-weborganizer ... python -m queries.annotate weborganizer --run comma2t-fewshot

Same annotator modules and options as pipeline.run. Input
data/queries/generations/<run>.parquet; output
data/queries/annotations/<annotator>/<run>.parquet, one row per query_id.
By default the generation alone is annotated (it is what gets attributed);
--with-prompt annotates "request + answer" instead (output name gets -wp).
"""

from __future__ import annotations

import argparse
import importlib
import json

import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from pipeline.run import ANNOTATORS
from queries.store import annotations_path, generations_path, open_append, read_jsonl, write_manifest, write_parquet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False)
    ap.add_argument("annotator", choices=ANNOTATORS)
    ap.add_argument("--run", default="comma2t-fewshot")
    ap.add_argument("--with-prompt", action="store_true")
    annotator = ap.parse_known_args()[0].annotator
    module = importlib.import_module(ANNOTATORS[annotator])
    module.add_args(ap)
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()
    name = annotator + ("-wp" if args.with_prompt else "")

    gens = [g for g in pq.read_table(generations_path(args.run)).to_pylist() if not g["empty"]]
    texts = {g["query_id"]: (f"{g['prompt']}\n\n{g['generation']}" if args.with_prompt else g["generation"])
             for g in gens}
    out = annotations_path(name, args.run)
    partial = out.with_suffix(".partial.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["query_id"] for r in read_jsonl(partial)}
    todo = [q for q in texts if q not in done]
    logger.info(f"{args.run}: {len(texts)} non-empty generations, {len(todo)} to annotate with {name}")

    if todo:
        annotate = module.load(args)  # sets args.batch_size if the module has a backend default
        with open_append(partial) as sink, tqdm(total=len(todo), desc=name, unit="query") as bar:
            for b in range(0, len(todo), args.batch_size):
                batch = todo[b: b + args.batch_size]
                for qid, ann in zip(batch, annotate([texts[q] for q in batch])):
                    sink.write(json.dumps({"query_id": qid, **ann}) + "\n")
                    bar.update(1)
                sink.flush()

    rows = {r["query_id"]: r for r in read_jsonl(partial)}
    if len(rows.keys() & texts.keys()) < len(texts):
        logger.info("run stays partial")
        return
    write_parquet([rows[q] for q in texts], out)
    write_manifest(out, annotator=annotator, run=args.run, with_prompt=args.with_prompt,
                   options={k: v for k, v in vars(args).items() if k not in ("annotator", "run")})
    partial.unlink()
    logger.success(f"{len(texts)} annotations -> {out}")


if __name__ == "__main__":
    main()
