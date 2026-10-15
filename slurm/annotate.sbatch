#!/bin/bash
# One GPU per array task; each task works through shards until none are left.
#
#   sbatch -w cn19 --array=0-3 slurm/annotate.sbatch propella --backend sglang --fp8 --batch-size 512
#   sbatch -w cn11 --array=0-1 slurm/annotate.sbatch propella --backend sglang --fp8 --batch-size 512
#   sbatch slurm/annotate.sbatch weborganizer
#
# On the time limit SLURM sends SIGTERM: the lock is released and the
# partial shard resumes in the next job.
#SBATCH -J annotate
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH -o logs/%x-%A_%a.out

cd "$SLURM_SUBMIT_DIR"
# CUDA toolkit (CUDA_HOME, nvcc): SGLang compiles kernels at startup and its
# DeepGEMM import on H100 fails without it. 12.6 matches the nodes' driver.
module load CUDA/12.6.0
source .venv/bin/activate
srun python -m pipeline.run "$@"
