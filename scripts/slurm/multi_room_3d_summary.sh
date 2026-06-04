#!/bin/bash
# Multi-room 3D zero-shot summary. Usage:
#   sbatch scripts/slurm/multi_room_3d_summary.sh <RUN_ID>
#SBATCH --job-name=aaf_summary_3d
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

RUN_ID="${1:?usage: $0 <RUN_ID>}"
TRAIN_OUT="outputs/multi_room_3d/${RUN_ID}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; run=${RUN_ID}"

python scripts/multi_room_3d_summary.py --train-output-dir "${TRAIN_OUT}"
