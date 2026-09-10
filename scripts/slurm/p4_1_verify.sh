#!/bin/bash
# P4-1 GPU verification: Arm C bit-identity regression, the DC loss-contribution diagnostic,
# and the full test suite. Runs alongside Stage 0 training on a separate GPU.
#SBATCH --job-name=aaf_p4_1_verify
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
echo "host=$(hostname); job=${SLURM_JOB_ID}"
nvidia-smi -L || true

echo; echo "=============== 1. Arm C bit-identity regression (D64 guard) ==============="
python scripts/p4_1_armc_regression.py

echo; echo "=============== 2. DC loss-contribution diagnostic ==============="
python scripts/p4_1_dc_diagnostic.py

echo; echo "=============== 3. full test suite ==============="
pytest -q
