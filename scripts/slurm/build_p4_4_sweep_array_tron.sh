#!/bin/bash
# P4-4 demo v2: FDTD ground truth for the morph-expansion sweeps. 36 unique rooms, CPU only.
# Usage:  sbatch --array=0-17 scripts/slurm/build_p4_4_sweep_array.sh
#SBATCH --job-name=aaf_v2fdtd_tron
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --requeue
# legacy43 kills the batch step at launch -- 24/24 tasks that landed there died in 3 s
# with signal 53 and NO log file written, while every task on every other node passed.
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
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}"
python scripts/build_p4_4_sweep_rooms.py --idx "${SLURM_ARRAY_TASK_ID}" --chunk 2
