#!/bin/bash
# P4-2 Stage 2 dataset: 20 sweep rooms (fixed L,W,w; d swept), dense receiver grid. CPU only.
# Usage:  sbatch --array=0-3 scripts/slurm/build_p4_2_sweep_array.sh
#SBATCH --job-name=aaf_p4_2_sweep
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
# The FDTD loop is numpy element-wise; extra BLAS threads only add churn.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}"
python scripts/build_p4_2_sweep.py --idx "${SLURM_ARRAY_TASK_ID}" --chunk 5
