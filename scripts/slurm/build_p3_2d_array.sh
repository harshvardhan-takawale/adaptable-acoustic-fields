#!/bin/bash
# P3-2 dataset build: one config per array task (690 total).
# Usage:  sbatch --array=0-68%40 scripts/slurm/build_2d_mat_array.sh   (10 configs/task)
# Idempotent: tasks with a .done sentinel exit immediately, so re-running fills gaps.
#SBATCH --job-name=aaf_p3_2d_build
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
# Required: conda's libstdc++ (GLIBCXX_3.4.29) must precede /lib64's older copy.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}"
python scripts/build_p3_2d_dataset.py --idx "${SLURM_ARRAY_TASK_ID}" --chunk "${CHUNK:-20}"
