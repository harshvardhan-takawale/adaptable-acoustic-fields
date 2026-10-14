#!/bin/bash
# P4-5: FDTD ground truth for B4 (rx4x), A2 (testdense) and A3 (a3). CPU only.
# Usage:  sbatch --export=ALL,MODE=rx4x,CHUNK=5 --array=0-49 scripts/slurm/build_p4_5_array.sh
#SBATCH --job-name=aaf_p4_5_build
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --requeue
# legacy43 kills every batch step at launch with no log written (D88).
#SBATCH --exclude=legacy43
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
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}; mode=${MODE}"
python scripts/build_p4_5_rooms.py --mode "${MODE}" --idx "${SLURM_ARRAY_TASK_ID}" --chunk "${CHUNK:-1}"
