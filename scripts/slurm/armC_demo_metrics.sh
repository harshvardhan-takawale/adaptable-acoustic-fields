#!/bin/bash
# Arm C demo pack, stage 1: the metrics (single GPU, ~4 min). Usage:
#   sbatch scripts/slurm/armC_demo_metrics.sh [extra args to armC_demo_metrics.py]
# e.g.
#   sbatch scripts/slurm/armC_demo_metrics.sh --dense 64
#
# A GPU is required even though nothing trains: the renderer is tinycudann and has no CPU path.
# Writes outputs/armC_demo/metrics.json plus the cached field dumps under fields/. The script
# EXITS NON-ZERO if the pre-registered spatial-Pearson abort rule trips (D62a), so stage 2
# (armC_demo_figures.py) refuses to draw. Do not paper over a non-zero exit here.
#SBATCH --job-name=aaf_armC
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

mkdir -p outputs/armC_demo logs/slurm
echo "host=$(hostname); job=${SLURM_JOB_ID:-none}"
nvidia-smi -L || true

python scripts/armC_demo_metrics.py "$@"
