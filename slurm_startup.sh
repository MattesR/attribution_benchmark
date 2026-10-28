# Environment for running the pipeline on the ITU cluster (interactive srun
# sessions and slurm/annotate.sbatch). Must be *sourced*, not executed:
#
#     source slurm_startup.sh
#
# - GCC 13: SGLang/flashinfer compile GPU kernels at startup; the system
#   compiler is too old for them (C++20). 13 is the newest GCC CUDA 12.6 supports.
# - CUDA 12.6: nvcc + CUDA_HOME for those compiles (and DeepGEMM on H100);
#   12.6 matches the nodes' driver (12.9 would be too new).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Batch shells don't always have the `module` command yet.
if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
        [ -f "$f" ] && source "$f" && break
    done
fi

module load GCC/13.3.0 CUDA/12.6.0 || return 1

# EasyBuild sets CUDA_HOME; fall back to deriving it from nvcc.
if [ -z "$CUDA_HOME" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

# VENV selects the environment: .venv (default; propella + SGLang, which needs
# transformers 5) or .venv-weborganizer (WebOrganizer, needs transformers<5).
source "$ROOT/${VENV:-.venv}/bin/activate" || return 1

echo "gcc:       $(gcc --version | head -1)"
echo "nvcc:      $(nvcc --version | tail -1)"
echo "CUDA_HOME: ${CUDA_HOME:-NOT SET}"
echo "python:    $(command -v python) ($(python --version 2>&1))"
echo "transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "ninja:     $(command -v ninja || echo 'NOT FOUND')"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
fi
python -c "import torch; print(f'torch:     {torch.__version__} (built for CUDA {torch.version.cuda}), cuda available: {torch.cuda.is_available()}')"
