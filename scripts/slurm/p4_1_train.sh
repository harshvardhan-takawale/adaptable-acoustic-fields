#!/bin/bash
# P4-1 training (Stage 0 Arm T-geo, Stage 1 per-scene, DC-masked A2 retrain).
# Usage:  sbatch scripts/slurm/p4_1_train.sh configs/sweep_2d_mat/P4_1_Tgeo.yaml
#         OUT_ROOT=outputs/p4_1/stage0 sbatch scripts/slurm/p4_1_train.sh <yaml>
#
# Copied from p3_3fast_train.sh, NOT from p3_2_train.sh -- the latter is the frozen
# reproducibility record for P3-2/P3-2b and must stay byte-stable.
#
# 24 h, not 12: 60K iters at the measured ~0.73 s/it is ~12.1 h, which overran the old 12 h cap.
# P3-2b only survived because scavenger --requeue resumed it mid-flight.
#SBATCH --job-name=aaf_p4_1
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=24:00:00
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
# Required: conda's libstdc++ (GLIBCXX_3.4.29) must precede /lib64's older copy.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"

CONFIG="${1:?usage: $0 <config-yaml>}"
RUN_ID=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['run_id'])")
OUT="${OUT_ROOT:-outputs/p4_1/stage0}/${RUN_ID}"
mkdir -p "${OUT}" logs/slurm
echo "host=$(hostname); job=${SLURM_JOB_ID}; config=${CONFIG}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.train.multi_room_2d_mat --config "${CONFIG}" --output_dir "${OUT}"
