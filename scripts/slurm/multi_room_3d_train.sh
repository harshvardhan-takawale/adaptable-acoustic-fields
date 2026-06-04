#!/bin/bash
# Multi-room 3D training. Usage:
#   sbatch scripts/slurm/multi_room_3d_train.sh configs/sweep_3d/M1_45rooms.yaml
#SBATCH --job-name=aaf_train_multi_3d
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

CONFIG="${1:?usage: $0 <config-yaml>}"
RUN_ID=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['run_id'])")
OUT="outputs/multi_room_3d/${RUN_ID}"
mkdir -p "${OUT}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; config=${CONFIG}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.train.multi_room_3d --config "${CONFIG}" --output_dir "${OUT}"
