#!/bin/bash
# P2-2.5 Run B: full 45 rooms, batch=32, n_pts=32, tron qos=high.
# GPU: rtxa6000 (48 GB). MUST name the GPU type — qos=high sets QoS limits,
# NOT GPU type; a bare `--gres=gpu:1` lands on an 11 GB 2080 Ti and OOMs.
# Relaxed early-stop (0.5% / 5K) and 60K iter target.
#SBATCH --job-name=aaf_diag_B
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

CONFIG="configs/sweep_3d/B_diag.yaml"
OUT="outputs/diag_p2_2_5/B_45rm_b32"
mkdir -p "${OUT}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; config=${CONFIG}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.train.multi_room_3d --config "${CONFIG}" --output_dir "${OUT}"
