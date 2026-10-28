#!/bin/bash
# One GPU per array task; each task works through shards until none are left.
#
#   sbatch -w cn19 --array=0-3 slurm/annotate.sbatch propella --part 0,1/3 --backend sglang --fp8
#   sbatch -w cn11 --array=0-1 slurm/annotate.sbatch propella --part 0,1/3 --backend sglang --fp8
#   VENV=.venv-weborganizer sbatch slurm/annotate.sbatch weborganizer
#
# On the time limit SLURM sends SIGTERM: the lock is released and the
# partial shard resumes in the next job.
#SBATCH -J annotate
#SBATCH --partition=NLPNorth
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH --signal=B:TERM@300   # SIGTERM 5 min before the limit: time to release locks
#SBATCH -o logs/%x-%A_%a.out

cd "$SLURM_SUBMIT_DIR"
source slurm_startup.sh || exit 1   # modules (GCC, CUDA) + venv
export PYTHONUNBUFFERED=1           # progress bar reaches the log file in whole updates
# No srun: one process per task needs no job step, and srun's CPU binding
# can fail when SLURM variables leak in from another session. exec makes
# python the batch shell itself, so it receives the SIGTERM above.
exec python -m pipeline.run "$@"
