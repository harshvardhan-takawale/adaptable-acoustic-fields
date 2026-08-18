#!/bin/bash
# FT-B aperture sweep: one FDTD config per array task (13 total = 11 sweep + 2 replicates).
# Usage:  sbatch --array=0-12 scripts/slurm/p3_3fast_ftb.sh
# Then:   python scripts/p3_3fast_ftb.py --aggregate
#SBATCH --job-name=aaf_ftb
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
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
# One BLAS thread per worker: the FDTD loop is numpy element-wise, threads only add churn.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}"
python scripts/p3_3fast_ftb.py --task "${SLURM_ARRAY_TASK_ID}"
