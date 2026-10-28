#!/bin/bash
# Start one background worker per GPU on a machine without SLURM (e.g. cir).
#
#   bash local_run.sh weborganizer                    # GPUs 3 2 1 (from the back; GPU 0 stays free)
#   GPUS="3 2" bash local_run.sh propella --backend sglang
#   VENV=.venv-weborganizer bash local_run.sh weborganizer
#
# Logs:   logs/<annotator>-gpu<N>.log      (tail -f to watch)
# Stop:   kill $(cat logs/<annotator>-gpu<N>.pid)   (one GPU; its file resumes later)
# Status: .venv/bin/python -m pipeline.status

cd "$(dirname "$0")" || exit 1
annotator=${1:?usage: bash local_run.sh <annotator> [pipeline.run options]}
shift
python="${VENV:-.venv}/bin/python"
[ -x "$python" ] || { echo "no python at $python (set VENV?)"; exit 1; }

for g in ${GPUS:-3 2 1}; do
    log="logs/$annotator-gpu$g.log"
    CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 nohup "$python" -m pipeline.run "$annotator" "$@" > "$log" 2>&1 &
    echo $! > "logs/$annotator-gpu$g.pid"
    echo "GPU $g: pid $!, log $log"
    sleep 5  # stagger starts, so the workers don't all load models at the same moment
done
