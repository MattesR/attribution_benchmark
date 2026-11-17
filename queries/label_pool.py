#!/usr/bin/env python3
"""Label-matched candidates: corpus documents that share a query's silver labels.

    python -m queries.label_pool --run comma2t-fewshot --match topic,format,content_type --k 100
    python -m queries.label_pool ... --with-text        # also attach the document texts

One of the candidate sources for the relevance-annotation pool (the others
are retrieval systems over an index). It finds documents that are similar
to a query in topic, genre and register without needing any lexical
overlap, and it gives an estimate of how many such documents exist.

For every query (a generation, labelled by queries.annotate), a document
matches if
  - single-valued fields are equal: WebOrganizer `topic`, `format` (argmax),
    or propella fields such as `audience_level`, `educational_value`;
  - multi-valued propella fields (`content_type`, `technical_content`,
    `business_sector`, ...) share at least one value.

Among the matching documents, --k are drawn uniformly at random by bottom-k
hashing: each (query, document) pair gets a fixed pseudo-random priority and
the k smallest are kept. The sample is reproducible, independent of the order
in which shards are scanned, and samples from different machines (e.g.
WebOrganizer labels on cir, propella labels on the cluster) can be merged by
keeping the k smallest priorities over their union. `bucket_size`, the number
of matching documents, is written per query.

Output: data/queries/pools/<run>.labels.<match>[.<corpus annotators>].parquet
        (query_id, shard, row[, window, start, end], priority, rank[, text])
        and <...>.stats.parquet (query_id, bucket_size, query labels).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from pipeline.common import DATA
from queries.store import QDATA, annotations_path, write_manifest, write_parquet

WEBORGANIZER_FIELDS = {"topic", "format"}
PROPELLA_MULTI = {"content_type", "business_sector", "technical_content", "regional_relevance",
                  "country_relevance"}
UNIT_COLS = ["row", "window", "start", "end"]
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def hash64(s: str) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little"))


def needed(match: list[str]) -> dict[str, list[str]]:
    """Fields per annotator for the given match spec."""
    by = {"weborganizer": [f for f in match if f in WEBORGANIZER_FIELDS],
          "propella": [f for f in match if f not in WEBORGANIZER_FIELDS]}
    return {a: f for a, f in by.items() if f}


def query_labels(run: str, by: dict[str, list[str]], suffix: str) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for ann, fields in by.items():
        path = annotations_path(ann + suffix, run)
        if not path.exists():
            raise SystemExit(f"{path} missing: run `python -m queries.annotate {ann} --run {run}"
                             f"{' --with-prompt' if suffix else ''}` first")
        t = pq.read_table(path, columns=["query_id", *fields] + (["ok"] if ann == "propella" else []))
        for r in t.to_pylist():
            if r.pop("ok", True) is False:
                continue
            labels.setdefault(r["query_id"], {}).update(r)
    want = {f for fs in by.values() for f in fs}
    complete = {q: l for q, l in labels.items() if want <= l.keys() and all(l[f] for f in want)}
    if len(complete) < len(labels):
        logger.warning(f"{len(labels) - len(complete)} queries lack some labels (failed annotation), skipped")
    return complete


def corpus_shards(by: dict[str, list[str]], names: dict[str, str], sources: list[str] | None) -> list[str]:
    """Shard stems that are annotated by every needed annotator."""
    sets = []
    for ann in by:
        root = DATA / "annotations" / names[ann]
        stems = {str(p.relative_to(root))[: -len(".parquet")] for p in root.rglob("*.parquet")}
        sets.append(stems)
        logger.info(f"{names[ann]}: {len(stems)} annotated shards in {root}")
    stems = sorted(set.intersection(*sets)) if sets else []
    if len(sets) > 1 and len(stems) < max(map(len, sets)):
        logger.warning(f"only {len(stems)} shards are annotated by all of {[names[a] for a in by]}")
    if sources:
        stems = [s for s in stems if s.split("/")[0] in sources]
    return stems


def read_shard(stem: str, by: dict[str, list[str]], names: dict[str, str]) -> pa.Table:
    cols, units = {}, None
    for ann, fields in by.items():
        path = DATA / "annotations" / names[ann] / f"{stem}.parquet"
        schema = pq.read_schema(path)
        unit = [c for c in UNIT_COLS if c in schema.names]
        extra = ["ok"] if ann == "propella" else []
        t = pq.read_table(path, columns=unit + fields + extra)
        if extra:  # failed propella rows have null labels: never match
            t = t.filter(pc.fill_null(t["ok"], False)).drop_columns(["ok"])
        if units is None:
            units = t.select(unit)
            cols.update({f: t[f] for f in fields})
            continue
        if t.select(unit).equals(units):  # annotators write units in the same order
            cols.update({f: t[f] for f in fields})
        else:  # different order or failed rows dropped: join on the unit key
            key = lambda tab: list(zip(*[tab[c].to_pylist() for c in unit]))
            pos = {k: i for i, k in enumerate(key(t))}
            idx = [pos.get(k) for k in key(units)]
            keep = [i for i, j in enumerate(idx) if j is not None]
            units = units.take(keep)
            cols = {f: c.take(keep) for f, c in cols.items()}
            cols.update({f: t[f].take([idx[i] for i in keep]) for f in fields})
    return pa.table({**{c: units[c] for c in units.column_names}, **cols})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="comma2t-fewshot")
    ap.add_argument("--match", default="topic,format,content_type",
                    help="label fields that must agree (WebOrganizer: topic, format; any propella field)")
    ap.add_argument("--k", type=int, default=100, help="documents sampled per query")
    ap.add_argument("--sources", default=None, help="comma-separated corpus sources (default: all annotated)")
    ap.add_argument("--weborganizer", default="weborganizer", help="corpus annotation dir (e.g. weborganizer-w32000)")
    ap.add_argument("--propella", default="propella")
    ap.add_argument("--with-prompt", action="store_true",
                    help="use the query annotations of request + answer (queries.annotate --with-prompt)")
    ap.add_argument("--with-text", action="store_true", help="attach document texts from data/raw")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.query_suffix = "-wp" if args.with_prompt else ""
    match = args.match.split(",")
    by = needed(match)
    names = {"weborganizer": args.weborganizer, "propella": args.propella}
    single = [f for f in match if f not in PROPELLA_MULTI]
    multi = [f for f in match if f in PROPELLA_MULTI]
    queries = query_labels(args.run, by, args.query_suffix)
    qids = sorted(queries)
    logger.info(f"{len(qids)} queries, match on {match}")
    qhash = {q: hash64(f"{args.seed}:{q}") for q in qids}
    qmulti = {q: {f: set(queries[q][f]) for f in multi} for q in qids}
    # Queries grouped by their single-valued labels: one lookup per group and shard.
    groups: dict[tuple, list[str]] = {}
    for q in qids:
        groups.setdefault(tuple(queries[q][f] for f in single), []).append(q)

    stems = corpus_shards(by, names, args.sources.split(",") if args.sources else None)
    logger.info(f"scanning {len(stems)} shards")
    # Per query: the current bottom-k as parallel arrays (priority, shard index, row, window, start, end).
    empty = (np.empty(0, np.uint64),) + tuple(np.empty(0, np.int64) for _ in range(5))
    best = dict.fromkeys(qids, empty)
    bucket = dict.fromkeys(qids, 0)
    has_window = False
    t0 = time.time()
    for si, stem in enumerate(tqdm(stems, unit="shard")):
        t = read_shard(stem, by, names)
        n = len(t)
        has_window = has_window or "window" in t.column_names
        col = lambda c: t[c].to_numpy().astype(np.int64) if c in t.column_names else np.full(n, -1, np.int64)
        rows, window, start, end = col("row"), col("window"), col("start"), col("end")
        doc_id = splitmix64(hash64(stem) ^ ((rows.astype(np.uint64) << np.uint64(20))
                                            | np.maximum(window, 0).astype(np.uint64)))
        # Documents per single-valued label combination.
        if single:
            index: dict[tuple, list[int]] = {}
            for i, k in enumerate(zip(*[t[f].to_pylist() for f in single])):
                index.setdefault(k, []).append(i)
        else:
            index = {(): list(range(n))}
        for key, qs in groups.items():
            cand = index.get(key)
            if not cand:
                continue
            cand = np.asarray(cand)
            dsets = {f: [set(v or ()) for v in t[f].take(pa.array(cand)).to_pylist()] for f in multi}
            for q in qs:
                ok = np.ones(len(cand), bool)
                for f in multi:
                    want = qmulti[q][f]
                    ok &= np.fromiter((bool(want & d) for d in dsets[f]), bool, len(cand))
                hit = cand[ok]
                if not len(hit):
                    continue
                bucket[q] += len(hit)
                new = (splitmix64(doc_id[hit] ^ qhash[q]), np.full(len(hit), si, np.int64),
                       rows[hit], window[hit], start[hit], end[hit])
                cur = tuple(np.concatenate([a, b]) for a, b in zip(best[q], new))
                if len(cur[0]) > args.k:
                    keep = np.argpartition(cur[0], args.k)[: args.k]
                    cur = tuple(a[keep] for a in cur)
                best[q] = cur
    logger.info(f"scan took {time.time() - t0:.0f}s")

    out_rows = []
    for q in qids:
        p, s, r, w, a, b = best[q]
        for rank, j in enumerate(np.argsort(p)):
            row = {"query_id": q, "shard": stems[s[j]], "row": int(r[j])}
            if has_window:
                row.update(window=int(w[j]), start=int(a[j]), end=int(b[j]))
            out_rows.append({**row, "priority": int(p[j]), "rank": rank})
    if args.with_text:
        attach_text(out_rows)
    tag = args.match.replace(",", "+")
    corpus = "+".join(names[a] for a in by)  # e.g. weborganizer-w32000 for book windows
    if corpus not in ("weborganizer", "propella", "weborganizer+propella"):
        tag += f".{corpus}"
    out = QDATA / "pools" / f"{args.run}{args.query_suffix}.labels.{tag}.parquet"
    write_parquet(out_rows, out)
    stats = [{"query_id": q, "bucket_size": bucket[q], "sampled": len(best[q][0]),
              **{f: (list(v) if isinstance(v, (list, set)) else v) for f, v in queries[q].items() if f != "query_id"}}
             for q in qids]
    write_parquet(stats, out.with_suffix(".stats.parquet"))
    sizes = np.array([bucket[q] for q in qids])
    write_manifest(out, run=args.run, match=match, k=args.k, seed=args.seed, n_shards=len(stems),
                   annotators={a: names[a] for a in by}, query_suffix=args.query_suffix,
                   bucket_size={"min": int(sizes.min()), "median": float(np.median(sizes)),
                                "max": int(sizes.max()), "n_below_k": int((sizes < args.k).sum())}
                   if len(sizes) else {})
    logger.success(f"{len(out_rows)} candidates for {len(qids)} queries -> {out}; "
                   f"bucket size median {np.median(sizes) if len(sizes) else 0:,.0f}, "
                   f"{int((sizes < args.k).sum())} queries with fewer than k matches")


def attach_text(rows: list[dict]) -> None:
    by_shard: dict[str, list[dict]] = {}
    for r in rows:
        by_shard.setdefault(r["shard"], []).append(r)
    for stem, rs in tqdm(by_shard.items(), desc="texts", unit="shard"):
        path = next((p for p in [DATA / "raw" / f"{stem}.jsonl.gz", DATA / "raw" / f"{stem}.json.gz"]
                     if p.exists()), None)
        if path is None:
            logger.warning(f"{stem}: raw file not in {DATA / 'raw'}, texts left empty")
            continue
        wanted = {r["row"] for r in rs}
        texts = {}
        with gzip.open(path, "rt") as f:
            for i, line in enumerate(f):
                if i in wanted:
                    texts[i] = json.loads(line)["text"]
        for r in rs:
            t = texts.get(r["row"])
            if t is not None and r.get("end") is not None and r["end"] >= 0:
                t = t[r["start"]:r["end"]]
            r["text"] = t


if __name__ == "__main__":
    main()
