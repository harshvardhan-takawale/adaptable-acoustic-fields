#!/bin/bash
# P3-2b ground-truth m-response sweep (CPU only -- pyroomacoustics ISM, no GPU).
#   sbatch --array=0-3 scripts/slurm/p3_2b_mresponse_gt.sh        # 3 cells / task
#   python scripts/p3_2b_mresponse_gt.py --merge --verify         # after all tasks finish
# Deliberately GPU-free so it cannot contend with the four P3-2b training arms.
#SBATCH --job-name=aaf_p3_2b_mresp_gt
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:40:00
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

N_SHARDS="${N_SHARDS:-4}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; shard=${SLURM_ARRAY_TASK_ID}/${N_SHARDS}"
python -W ignore scripts/p3_2b_mresponse_gt.py \
    --shard "${SLURM_ARRAY_TASK_ID}" --n-shards "${N_SHARDS}"
