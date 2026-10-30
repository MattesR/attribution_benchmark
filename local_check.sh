#!/bin/bash
# Overview of the workers started with local_run.sh (machines without SLURM).
#
#     ./local_check.sh
#
# 1. each worker (running / stopped) with its latest progress line,
# 2. GPU usage, 3. which shards are done (pipeline.status, from progress.tsv).

cd "$(dirname "$0")" || exit 1
python="${VENV:-${VIRTUAL_ENV:-.venv}}/bin/python"

echo "== workers =="
for pidfile in logs/*-gpu*.pid; do
    [ -f "$pidfile" ] || { echo "(none started)"; break; }
    name=$(basename "$pidfile" .pid)
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then state="RUNNING"; else state="stopped"; fi
    # tqdm separates updates with \r; keep the last non-empty line.
    last=$(tr '\r' '\n' < "logs/$name.log" 2>/dev/null | grep -v '^\s*$' | tail -1)
    printf "%-22s %-8s pid %-8s %s\n" "$name" "$state" "$pid" "${last:0:130}"
done

echo
echo "== GPUs =="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | awk -F', ' '{printf "GPU %s: %4s util, %s / %s\n", $1, $2, $3, $4}'

echo
echo "== shards done (progress.tsv) =="
"$python" -m pipeline.status
