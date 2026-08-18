#!/bin/bash
# P3-3-FAST Track A localization diagnostic (single GPU, ~10 min). Usage:
#   sbatch scripts/slurm/p3_3fast_trackA_diag.sh [extra args to the diag script]
# e.g.
#   sbatch scripts/slurm/p3_3fast_trackA_diag.sh --checkpoint outputs/p3_3fast/p3_3fast_trackA/ckpt_iter0012000.pt
#
# Track A training may still be RUNNING. Pass --checkpoint explicitly when the newest file
# could be mid-write; without it the driver picks the newest ckpt_iter*.pt at job start,
# which races with the trainer's ckpt_every window.
#SBATCH --job-name=aaf_p33A_diag
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
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

mkdir -p outputs/p3_3fast/trackA logs/slurm
echo "host=$(hostname); job=${SLURM_JOB_ID:-none}"
nvidia-smi -L || true

python scripts/p3_3fast_trackA_diag.py "$@"
