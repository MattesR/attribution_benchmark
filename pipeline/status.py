#!/usr/bin/env python3
"""Overview of what has been downloaded and annotated.

    python -m pipeline.status          # sources in the priority list + any with activity
    python -m pipeline.status --all    # every source

Reads progress.tsv (merge logs from other machines into it to see everything)
and lists running/stale locks and partial shards on this machine.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from pipeline import common
from pipeline.run import ANNOTATORS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    by_source = common.list_shards()
    log = common.read_log()
    downloaded = {r["shard"] for r in log if r["event"] == "downloaded"}
    done = defaultdict(dict)  # annotator -> shard -> n_docs
    for r in log:
        if r["event"] == "annotated":
            done[r["annotator"]][r["shard"]] = int(r["n_docs"] or 0)

    priority = [s for group in common.PRIORITY for s in group]
    active = {r["shard"].split("/")[0] for r in log}
    sources = list(by_source) if args.all else priority + sorted(active - set(priority))

    names = list(ANNOTATORS)
    header = f"{'source':<30}{'shards':>7}{'downl.':>8}" + "".join(f"{n:>14}{'docs':>13}" for n in names)
    print(header)
    print("-" * len(header))
    totals = defaultdict(int)
    for s in sources:
        shards = set(by_source.get(s, []))
        line = f"{s:<30}{len(shards):>7}{len(shards & downloaded):>8}"
        totals["shards"] += len(shards)
        totals["downl"] += len(shards & downloaded)
        for n in names:
            finished = shards & done[n].keys()
            docs = sum(done[n][x] for x in finished)
            line += f"{f'{len(finished)}/{len(shards)}':>14}{docs:>13,}"
            totals[n] += len(finished)
            totals[n + "_docs"] += docs
        print(line)
    print("-" * len(header))
    print(f"{'total':<30}{totals['shards']:>7}{totals['downl']:>8}"
          + "".join(f"{totals[n]:>14}{totals[n + '_docs']:>13,}" for n in names))

    locks = sorted((common.DATA / "locks").rglob("*.lock"))
    partials = sorted((common.DATA / "annotations").rglob("*.partial.jsonl"))
    if locks:
        print("\nlocks on this machine (running, or stale after a crash -> delete):")
        for p in locks:
            print(f"  {p.relative_to(common.DATA / 'locks')}: {p.read_text().strip()}")
    if partials:
        print("\npartial shards on this machine (resume automatically):")
        for p in partials:
            n = sum(1 for _ in open(p))
            print(f"  {p.relative_to(common.DATA / 'annotations')}: {n:,} rows")


if __name__ == "__main__":
    main()
