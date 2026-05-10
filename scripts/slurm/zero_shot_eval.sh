#!/bin/bash
#SBATCH --job-name=aaf_zs_eval
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

L="${1:?usage: $0 <L>}"
TRAIN_OUT="${2:-outputs/multi_room/dense}"
OUT="${TRAIN_OUT}/zero_shot/L${L}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; L=${L}; out=${OUT}"
nvidia-smi -L || true

mkdir -p "${OUT}"
python -m aaf.eval.zero_shot \
    --L "${L}" \
    --train_output_dir "${TRAIN_OUT}" \
    --output_dir "${OUT}"
