#!/bin/bash
# P4-3 training: the four mechanism arms (X1 attn-residual, X2 attn, S sigma-sup, D data).
# Usage:  sbatch scripts/slurm/p4_3_train.sh configs/sweep_2d_mat/P4_3_X1_ATTN_RESIDUAL.yaml
#
# All four arms share the P4-2 recipe EXACTLY and differ in one variable each, so the
# baseline they are read against (p4_2_s2_mean_unmasked) needs no re-run -- its corpus was
# verified built AFTER the D67c grid fix.
#
# Copied from p3_3fast_train.sh, NOT from p3_2_train.sh -- the latter is the frozen
# reproducibility record for P3-2/P3-2b and must stay byte-stable.
#
# 24 h, not 12: 60K iters at the measured ~0.73 s/it is ~12.1 h, which overran the old 12 h cap.
# P3-2b only survived because scavenger --requeue resumed it mid-flight.
#SBATCH --job-name=aaf_p4_3
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
OUT="${OUT_ROOT:-outputs/p4_3}/${RUN_ID}"
mkdir -p "${OUT}" logs/slurm
echo "host=$(hostname); job=${SLURM_JOB_ID}; config=${CONFIG}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.train.multi_room_2d_mat --config "${CONFIG}" --output_dir "${OUT}"
