#!/usr/bin/env python3
"""Annotate Comma shards with one annotator, one GPU per process.

    python -m pipeline.run weborganizer
    python -m pipeline.run propella [--backend sglang --batch-size 512]

Loop over shards in priority order, skipping those already marked
`annotated` in progress.tsv or locked by another process:

    download -> annotate in batches, appending to <shard>.partial.jsonl
             -> all rows done: write <shard>.parquet, log `annotated`

The partial file is the checkpoint: a killed process resumes where it
stopped. With --limit-docs the shard stays partial (for prototyping).
"""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from pipeline import common

ANNOTATORS = {
    "weborganizer": "pipeline.weborganizer_annotator",
    "propella": "pipeline.propella_annotator",
}


def read_partial(path) -> dict[int, dict]:
    rows = {}
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # last line cut off by a kill
                continue
            rows[r["row"]] = r
    return rows


def float32(table: pa.Table) -> pa.Table:
    """float64 -> float32 (also inside lists): halves the size of probability columns."""
    def f32(t):
        if pa.types.is_floating(t):
            return pa.float32()
        if pa.types.is_list(t):
            return pa.list_(f32(t.value_type))
        return t
    return table.cast(pa.schema([f.with_type(f32(f.type)) for f in table.schema]))


def annotate_shard(name: str, annotate, shard: str, args) -> None:
    texts = common.read_texts(common.download(shard))
    partial = common.output_path(name, shard, partial=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    done = read_partial(partial)

    # Length-sorted so batches pad efficiently.
    todo = sorted((i for i in range(len(texts)) if i not in done), key=lambda i: len(texts[i]))
    if args.limit_docs:
        todo = todo[: args.limit_docs]

    t0 = time.time()
    with open(partial, "a") as sink, tqdm(total=len(texts), initial=len(done), desc=shard, unit="doc") as bar:
        for b in range(0, len(todo), args.batch_size):
            rows = todo[b : b + args.batch_size]
            for i, ann in zip(rows, annotate([texts[i] for i in rows])):
                sink.write(json.dumps({"row": i, **ann}) + "\n")
            sink.flush()
            bar.update(len(rows))

    done = read_partial(partial)
    if len(done) < len(texts):
        logger.info(f"{shard}: {len(done)}/{len(texts)} rows annotated, shard stays partial")
        return

    rows = [done[i] for i in range(len(texts))]
    out = common.output_path(name, shard)
    pq.write_table(float32(pa.Table.from_pylist(rows)), out, compression="zstd")
    partial.unlink()
    n_failed = sum(r.get("ok") is False for r in rows)
    common.log("annotated", annotator=name, shard=shard, n_docs=len(rows),
               n_failed=n_failed, seconds=round(time.time() - t0))
    logger.success(f"{shard}: {len(rows)} docs -> {out} ({n_failed} failed)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False)  # -h added below, after the annotator options
    ap.add_argument("annotator", choices=ANNOTATORS)
    ap.add_argument("--sources", help="comma-separated, overrides the priority list in common.py")
    ap.add_argument("--part", help="i/n or i,j/n: only shards whose position mod n is in {i,j}, "
                                   "e.g. 0,1/3 on the cluster and 2/3 on cir, so machines without "
                                   "shared locks don't overlap")
    ap.add_argument("--max-shards", type=int, default=0, help="stop after this many shards")
    ap.add_argument("--limit-docs", type=int, default=0, help="prototype: annotate only N docs per shard")
    name = ap.parse_known_args()[0].annotator
    module = importlib.import_module(ANNOTATORS[name])
    module.add_args(ap)  # adds --batch-size and annotator-specific options
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()

    # SIGTERM (kill, SLURM time limit) -> normal exit, so the lock is released.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))

    shards = common.ordered_shards(args.sources.split(",") if args.sources else None)
    if args.part:
        # Every n-th shard of the (pinned, identical everywhere) list: disjoint
        # work for machines that don't share locks, in the same priority order.
        mine, k = args.part.split("/")
        mine, k = {int(i) for i in mine.split(",")}, int(k)
        shards = [s for j, s in enumerate(shards) if j % k in mine]

    annotate = None  # model is loaded only once there is work
    n = 0
    for shard in shards:
        if shard in common.annotated(name):
            continue
        with common.shard_lock(name, shard) as got:
            if not got or shard in common.annotated(name):
                continue
            if annotate is None:
                annotate = module.load(args)
            annotate_shard(name, annotate, shard, args)
        n += 1
        if args.max_shards and n >= args.max_shards:
            break
    logger.info(f"done: processed {n} shard(s)")


if __name__ == "__main__":
    main()
