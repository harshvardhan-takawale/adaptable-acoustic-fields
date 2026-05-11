#!/bin/bash
#SBATCH --job-name=aaf_track_a_bandlim
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --exclude=legacygpu06,legacygpu07
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname); job=${SLURM_JOB_ID}"
nvidia-smi -L

# Stage 1: re-forward all (run, L) and write band_limited_metrics.json next to existing outputs.
python -m scripts.band_limited_recompute

# Stage 2: aggregate into outputs/multi_room/sweep/band_limited_summary.md + figure.
python -m scripts.track_a_summary

echo "TRACK A done."
