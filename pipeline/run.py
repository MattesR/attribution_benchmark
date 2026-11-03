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

--windows N annotates windows of at most N characters instead of whole
records (for long texts such as books; see pipeline/windows.py). Output, locks
and log then use the name <annotator>-w<N>, and every row carries
row, window, start, end.
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
from pipeline.windows import split_windows

ANNOTATORS = {
    "weborganizer": "pipeline.weborganizer_annotator",
    "propella": "pipeline.propella_annotator",
}


def key(r: dict):
    """A unit is a record (row) or, with --windows, a window of a record."""
    return (r["row"], r["window"]) if "window" in r else r["row"]


def read_partial(path) -> dict:
    rows = {}
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # last line cut off by a kill
                continue
            rows[key(r)] = r
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
    records = common.read_texts(common.download(shard))
    # Units to annotate: whole records, or windows of them (--windows).
    if args.windows:
        units = [{"row": i, "window": w, "start": s, "end": e}
                 for i, t in enumerate(records) for w, (s, e) in enumerate(split_windows(t, args.windows))]
        logger.info(f"{shard}: {len(records):,} records -> {len(units):,} windows of <= {args.windows:,} chars")
    else:
        units = [{"row": i} for i in range(len(records))]
    texts = {key(u): records[u["row"]][u["start"]:u["end"]] if "window" in u else records[u["row"]]
             for u in units}
    partial = common.output_path(name, shard, partial=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    done = read_partial(partial)

    # Length-sorted so batches pad efficiently.
    todo = sorted((u for u in units if key(u) not in done), key=lambda u: len(texts[key(u)]))
    if args.limit_docs:
        todo = todo[: args.limit_docs]
    if done:
        logger.info(f"{shard}: resuming, {len(done):,} units already annotated")
    logger.info(f"{shard}: annotating {len(todo):,} units in batches of {args.batch_size} "
                "(shortest first, so the rate drops over time)")

    t0 = time.time()
    chars = 0
    # In a batch job (log file, not a terminal) update once a minute, so the
    # log stays readable and its last line is the current progress.
    interactive = sys.stderr.isatty()
    with open(partial, "a") as sink, tqdm(total=len(units), initial=len(done), desc=shard, unit="doc",
                                          dynamic_ncols=True, mininterval=0.1 if interactive else 60,
                                          ascii=not interactive) as bar:
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b : b + args.batch_size]
            # annotate() may return an iterator (sglang): the bar moves per document.
            for u, ann in zip(batch, annotate([texts[key(u)] for u in batch])):
                sink.write(json.dumps({**u, **ann}) + "\n")
                chars += len(texts[key(u)])
                bar.update(1)
            sink.flush()
            bar.set_postfix(chars_per_s=f"{chars / (time.time() - t0):,.0f}",
                            doc_len=f"{len(texts[key(batch[-1])]):,}")
    if todo:
        # Rate of this run only (a resumed shard was partly done before).
        sec = time.time() - t0
        chars = sum(len(texts[key(u)]) for u in todo)
        logger.info(f"{shard}: {len(todo)} units in {sec:.0f}s = {len(todo) / sec:.2f} units/s, "
                    f"{chars / sec:,.0f} chars/s (avg {chars / len(todo):,.0f} chars/unit)")

    done = read_partial(partial)
    if len(done) < len(units):
        logger.info(f"{shard}: {len(done)}/{len(units)} units annotated, shard stays partial")
        return

    rows = [done[key(u)] for u in units]
    out = common.output_path(name, shard)
    logger.info(f"{shard}: complete, writing {out.name}")
    pq.write_table(float32(pa.Table.from_pylist(rows)), out, compression="zstd")
    partial.unlink()
    n_failed = sum(r.get("ok") is False for r in rows)
    common.log("annotated", annotator=name, shard=shard, n_docs=len(rows),
               n_failed=n_failed, seconds=round(time.time() - t0))
    logger.success(f"{shard}: {len(rows)} units -> {out} ({n_failed} failed)")


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
    ap.add_argument("--windows", type=int, default=0,
                    help="annotate windows of at most N characters instead of whole records "
                         "(e.g. 32000 for books); results go to <annotator>-w<N>")
    annotator = ap.parse_known_args()[0].annotator
    module = importlib.import_module(ANNOTATORS[annotator])
    module.add_args(ap)  # adds --batch-size and annotator-specific options
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()
    # Window runs are kept apart from record-level runs: own output dir, locks, log name.
    name = f"{annotator}-w{args.windows}" if args.windows else annotator

    # SIGTERM (kill, SLURM time limit) -> normal exit, so the lock is released.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))

    shards = common.ordered_shards(args.sources.split(",") if args.sources else None)
    if args.part:
        # Every n-th shard of the (pinned, identical everywhere) list: disjoint
        # work for machines that don't share locks, in the same priority order.
        mine, k = args.part.split("/")
        mine, k = {int(i) for i in mine.split(",")}, int(k)
        shards = [s for j, s in enumerate(shards) if j % k in mine]
    n_done = len(set(shards) & common.annotated(name))
    logger.info(f"{name}: {len(shards)} shards selected, {n_done} already annotated, "
                f"{len(shards) - n_done} to go")

    annotate = None  # model is loaded only once there is work
    n = 0
    for shard in shards:
        if shard in common.annotated(name):
            continue
        with common.shard_lock(name, shard) as got:
            if not got or shard in common.annotated(name):
                if not got:
                    logger.info(f"{shard}: locked by another process, skipping")
                continue
            logger.info(f"{shard}: starting")
            if annotate is None:
                annotate = module.load(args)
            annotate_shard(name, annotate, shard, args)
        n += 1
        if args.max_shards and n >= args.max_shards:
            break
    logger.info(f"done: processed {n} shard(s)")


if __name__ == "__main__":
    main()
