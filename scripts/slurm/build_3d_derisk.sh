#!/bin/bash
# Array task: build one de-risk room. Submit with --array=0-4%4 to cap at 4
# concurrent tasks on tron (per manager directive).
#SBATCH --job-name=aaf_build_3d_derisk
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

ROOMS_YAML="configs/sweeps_3d/derisk_rooms.yaml"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}; idx=${IDX}"

python scripts/build_3d_dataset.py --rooms-yaml "${ROOMS_YAML}" --idx "${IDX}"
