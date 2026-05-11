#!/bin/bash
#SBATCH --job-name=aaf_zs_variant
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --exclude=legacygpu06,legacygpu07
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

VARIANT="${1:?usage: $0 <variant B1..B6> <L> <run>}"
L="${2:?usage: $0 <variant> <L> <run>}"
RUN="${3:?usage: $0 <variant> <L> <run>}"

cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; variant=${VARIANT}; L=${L}; run=${RUN}"
nvidia-smi -L

python -m scripts.zero_shot_variant --variant "$VARIANT" --L "$L" --run "$RUN"
