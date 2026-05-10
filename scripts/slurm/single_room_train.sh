#!/bin/bash
#SBATCH --job-name=aaf_train
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

L="${1:?usage: $0 <L>}"
OUT="outputs/single_room/L${L}"
mkdir -p "${OUT}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; L=${L}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.train.single_room \
    --L "${L}" \
    --sweep configs/sweeps/dense.yaml \
    --output_dir "${OUT}" \
    --n_pts_per_ray 32  # GTX TITAN X (12 GB) needs this; see outputs/memory_check/REPORT.md
