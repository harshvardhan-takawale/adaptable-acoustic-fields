#!/bin/bash
#SBATCH --job-name=aaf_hello
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields"
CONDA_ROOT="/fs/nexus-scratch/htakawal/miniconda3"

cd "${PROJECT_ROOT}"
mkdir -p logs/slurm

echo "=== aaf_hello @ $(hostname) ==="
echo "JOB_ID=${SLURM_JOB_ID}"
echo "NODE=${SLURMD_NODENAME:-unknown}"
echo "DATE=$(date -Iseconds)"
echo "CWD=$(pwd)"
echo

echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi failed (no GPU?)"
echo

echo "=== conda env ==="
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate aaf
# Make conda's libstdc++ (with GLIBCXX_3.4.29) take precedence over /lib64's older copy.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "PYTHON=$(which python)"
python --version
echo

echo "=== torch CUDA ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("device count:", torch.cuda.device_count())
PY
echo

echo "=== done ==="
