#!/bin/bash
# Daily overview of the annotation jobs. Run on the head node:
#
#     bash slurm/check.sh
#
# 1. your running jobs, 2. the latest progress line of each job log,
# 3. which shards are done (pipeline.status, from progress.tsv).

cd "$(dirname "$0")/.." || exit 1

echo "== jobs =="
squeue -u "$USER" -o "%.12i %.10j %.8T %.12M %.10l %R"

echo
echo "== latest progress per job log =="
for f in $(ls -t logs/annotate-*.out 2>/dev/null | head -20); do
    # tqdm separates updates with \r; keep the last non-empty one.
    last=$(tr '\r' '\n' < "$f" | grep -v '^\s*$' | tail -1)
    printf "%-28s %s\n" "$(basename "$f")" "${last:0:160}"
done

echo
echo "== shards done (progress.tsv) =="
source .venv/bin/activate && python -m pipeline.status
