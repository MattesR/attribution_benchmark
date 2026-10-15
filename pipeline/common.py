"""Shared bookkeeping: which shards exist, where files live, what has been done.

A *shard* is one file of the Comma dataset, named by its repo path, e.g.
"cccc/cccc.chunk.00.jsonl.gz". Documents are addressed by (shard, row) — the
line number in that file — which is stable because the dataset revision is
pinned below.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
import socket
import time
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

REPO = "common-pile/comma_v0.1_training_dataset"
REVISION = "5afc546db324e7f39f297ba757c9a60547151e7c"  # pinned: row numbers must not shift

# Processing order. Web-like sources first; sources not listed are not
# processed unless named explicitly with --sources.
PRIORITY = [
    ["cccc", "news", "foodista", "public_domain_review"],
    ["wikimedia", "wikiteam", "stackexchange", "libretexts", "pressbooks", "oercommons"],
]

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ATTRIB_DATA", ROOT / "data"))
LOG = ROOT / "progress.tsv"
HOST = socket.gethostname().strip()
LOG_FIELDS = ["time", "host", "event", "annotator", "shard", "n_docs", "n_failed", "seconds"]


# ---- shards -----------------------------------------------------------------

def list_shards() -> dict[str, list[str]]:
    """All shards in the dataset, grouped by source."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
    by_source: dict[str, list[str]] = {}
    for f in sorted(files):
        # Most sources are *.chunk.NN.jsonl.gz; wikiteam also has NNNNN_wikiteam.json.gz.
        if f.endswith((".jsonl.gz", ".json.gz")):
            by_source.setdefault(f.split("/")[0], []).append(f)
    return by_source


def ordered_shards(sources: list[str] | None = None) -> list[str]:
    by_source = list_shards()
    sources = sources or [s for group in PRIORITY for s in group]
    return [shard for s in sources for shard in by_source.get(s, [])]


def stem(shard: str) -> str:
    """cccc/cccc.chunk.00.jsonl.gz -> cccc/cccc.chunk.00"""
    return re.sub(r"\.jsonl?\.gz$", "", shard)


def raw_path(shard: str) -> Path:
    return DATA / "raw" / shard


def download(shard: str) -> Path:
    path = raw_path(shard)
    if path.exists():
        logger.info(f"{shard}: raw file already here ({path.stat().st_size / 1e6:,.0f} MB)")
        return path
    from huggingface_hub import hf_hub_download
    logger.info(f"{shard}: downloading")
    t0 = time.time()
    hf_hub_download(REPO, shard, repo_type="dataset", revision=REVISION, local_dir=DATA / "raw")
    sec = time.time() - t0
    logger.info(f"{shard}: downloaded {path.stat().st_size / 1e6:,.0f} MB in {sec:.0f}s")
    log("downloaded", shard=shard, seconds=round(sec))
    return path


def read_texts(path: Path) -> list[str]:
    t0 = time.time()
    with gzip.open(path, "rt") as f:
        texts = [json.loads(line)["text"] for line in f]
    chars = sum(map(len, texts))
    logger.info(f"read {len(texts):,} docs, {chars / 1e6:,.0f}M chars "
                f"(avg {chars / max(len(texts), 1):,.0f} chars/doc) in {time.time() - t0:.0f}s")
    return texts


def output_path(annotator: str, shard: str, partial: bool = False) -> Path:
    suffix = ".partial.jsonl" if partial else ".parquet"
    return DATA / "annotations" / annotator / f"{stem(shard)}{suffix}"


# ---- progress log -----------------------------------------------------------
# progress.tsv is append-only, one line per event, so logs from different
# machines can be merged by concatenation (git does this via merge=union).

def log(event: str, annotator: str = "", shard: str = "", n_docs="", n_failed="", seconds="") -> None:
    new = not LOG.exists()
    row = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), host=HOST, event=event,
               annotator=annotator, shard=shard, n_docs=n_docs, n_failed=n_failed, seconds=seconds)
    line = ("\t".join(LOG_FIELDS) + "\n" if new else "") + "\t".join(str(row[k]) for k in LOG_FIELDS) + "\n"
    with open(LOG, "a") as f:
        f.write(line)


def read_log() -> list[dict]:
    if not LOG.exists():
        return []
    with open(LOG) as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["event"] != "event"]


def annotated(annotator: str) -> set[str]:
    return {r["shard"] for r in read_log() if r["event"] == "annotated" and r["annotator"] == annotator}


# ---- locks ------------------------------------------------------------------
# Keeps several processes on the same machine (or the same shared filesystem)
# off the same shard. A lock left behind by a hard crash shows up in
# `python -m pipeline.status` and can simply be deleted.

def lock_path(annotator: str, shard: str) -> Path:
    return DATA / "locks" / annotator / f"{stem(shard)}.lock"


def _stale(path: Path) -> bool:
    """Lock left by a dead process on this host (hard kill, crash)."""
    try:
        host, pid = path.read_text().split()
        pid = int(pid.removeprefix("pid="))
    except (OSError, ValueError):
        return False
    if host != HOST:
        return False  # can't check processes on other machines
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass  # pid exists (reused by another user's process)
    return False


@contextmanager
def shard_lock(annotator: str, shard: str):
    path = lock_path(annotator, shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and _stale(path):
        logger.warning(f"{shard}: removing stale lock ({path.read_text().strip()})")
        path.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        yield False
        return
    os.write(fd, f"{HOST} pid={os.getpid()}\n".encode())
    os.close(fd)
    try:
        yield True
    finally:
        path.unlink(missing_ok=True)
