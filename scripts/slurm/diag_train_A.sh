#!/bin/bash
# P2-2.5 Run A: 10-room subset, batch=16, n_pts=16, tron qos=default.
# GPU: rtxa5000 (24 GB) — the "modest GPU" per spec. MUST name the GPU type;
# a bare `--gres=gpu:1` lands on whatever is free (an 11 GB 2080 Ti) and OOMs.
#SBATCH --job-name=aaf_diag_A
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

CONFIG="configs/sweep_3d/A_diag.yaml"
OUT="outputs/diag_p2_2_5/A_10rm_b16"
mkdir -p "${OUT}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; config=${CONFIG}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.train.multi_room_3d --config "${CONFIG}" --output_dir "${OUT}"
