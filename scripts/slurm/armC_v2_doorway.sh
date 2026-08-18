#!/bin/bash
# Arm C v2 stage 0: dense-field doorway simulation (3 apertures, ground truth only).
# Usage:  sbatch scripts/slurm/armC_v2_doorway.sh
#
# CPU only -- no GPU, no model. Memory is the binding constraint, not time: `simulate`
# allocates ir_t, ir, H_complex and H_deconv all at n_rx x n (8192 x 30720), and H_deconv
# has no skip flag. ~10 GB peak; 48G leaves headroom for the transposed copy.
#SBATCH --job-name=aaf_v2door
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
# Required: conda's libstdc++ (GLIBCXX_3.4.29) must precede /lib64's older copy.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
# One BLAS thread: the FDTD loop is numpy element-wise, threads only add churn.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host=$(hostname); job=${SLURM_JOB_ID}"
python scripts/armC_v2_doorway.py
