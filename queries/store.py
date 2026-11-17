"""Small helpers shared by the stage-2 scripts: paths, manifests, checkpoints."""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.common import DATA

QDATA = DATA / "queries"
PROMPTS = QDATA / "prompts.parquet"


def generations_path(run: str) -> Path:
    return QDATA / "generations" / f"{run}.parquet"


def annotations_path(annotator: str, run: str) -> Path:
    return QDATA / "annotations" / annotator / f"{run}.parquet"


def write_manifest(path: Path, **info) -> None:
    """<output>.manifest.json next to every output: how it was made."""
    info = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": socket.gethostname().strip(),
            "argv": sys.argv, **info}
    path.with_suffix(".manifest.json").write_text(json.dumps(info, indent=2, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # last line cut off by a kill
                pass
    return rows


def open_append(path: Path):
    """Append to a checkpoint; a line cut off by a kill is closed first, so the next record isn't lost."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            cut = f.read(1) != b"\n"
        if cut:
            with open(path, "a") as f:
                f.write("\n")
    return open(path, "a")


def write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
