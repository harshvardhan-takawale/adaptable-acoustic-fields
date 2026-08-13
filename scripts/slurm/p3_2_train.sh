#!/bin/bash
# P3-2 training (single GPU). Usage:
#   sbatch scripts/slurm/p3_2_train.sh configs/sweep_2d_mat/P3_2_main.yaml
#SBATCH --job-name=aaf_p3_2_train
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"

CONFIG="${1:?usage: $0 <config-yaml>}"
RUN_ID=$(python -c "import yaml,sys; print(yaml.safe_load(open('${CONFIG}'))['run_id'])")
OUT="outputs/p3_2/${RUN_ID}"
mkdir -p "${OUT}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; config=${CONFIG}; out=${OUT}"
nvidia-smi -L || true
python -m aaf.train.multi_room_2d_mat --config "${CONFIG}" --output_dir "${OUT}"
