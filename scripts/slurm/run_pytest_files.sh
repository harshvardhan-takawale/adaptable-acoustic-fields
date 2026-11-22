#!/bin/bash
# Run selected test files on a GPU node (the full suite is scripts/slurm/run_pytest.sh).
# Usage: sbatch scripts/slurm/run_pytest_files.sh tests/test_a.py tests/test_b.py
#SBATCH --job-name=aaf_pytest_sel
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --exclude=legacy43
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; files=$*"
nvidia-smi -L || true
pytest -q "$@"
