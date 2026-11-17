"""Start an SGLang server for this process (killed when the process exits).

Same procedure as pipeline/propella_annotator.start_sglang, but for any model
and without the propella-specific flags (grammar backend, 64k context).
"""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from loguru import logger

from pipeline.common import DATA


def start_sglang(model: str, context_length: int, extra: list[str] = (), revision: str | None = None) -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", model,
           "--host", "127.0.0.1", "--port", str(port), "--context-length", str(context_length),
           "--mem-fraction-static", "0.8", *extra]
    if revision:
        cmd += ["--revision", revision]
    log_path = DATA / "logs" / f"sglang-{socket.gethostname().strip()}-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env)
    atexit.register(proc.terminate)
    logger.info(f"starting sglang ({model}) on port {port}, log: {log_path}")

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
            if "Traceback (most recent call last)" in log_text:
                proc.terminate()
                raise RuntimeError(f"sglang server crashed: {log_text.strip().splitlines()[-1]}  "
                                   f"(full log: {log_path})")
            if k % 6 == 5:
                lines = log_text.strip().splitlines()
                logger.info(f"waiting for sglang ({time.time() - t0:.0f}s): {lines[-1][-150:] if lines else ''}")
            time.sleep(5)
    raise RuntimeError(f"sglang server did not come up, see {log_path}")
