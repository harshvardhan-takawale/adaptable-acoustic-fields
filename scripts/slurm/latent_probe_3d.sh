#!/bin/bash
# Latent probe 3D. Usage:
#   sbatch scripts/slurm/latent_probe_3d.sh <RUN_ID>
#SBATCH --job-name=aaf_probe_3d
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

RUN_ID="${1:?usage: $0 <RUN_ID>}"
TRAIN_OUT="outputs/multi_room_3d/${RUN_ID}"
OUT="${TRAIN_OUT}/latent_probe"
mkdir -p "${OUT}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; run=${RUN_ID}; out=${OUT}"

python -m aaf.eval.latent_probe_3d \
    --train-output-dir "${TRAIN_OUT}" \
    --output_dir "${OUT}"
