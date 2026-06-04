#!/bin/bash
# Single-room 3D eval (signal-level + traditional). Runs on scavenger (GPU
# forward-only, no autograd → fits the 12 GB TITAN X if scheduled there).
# Usage:
#   sbatch scripts/slurm/single_room_3d_eval.sh <L> <W> <H>
#SBATCH --job-name=aaf_eval_3d
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
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

L="${1:?usage: $0 <L> <W> <H>}"
W="${2:?usage: $0 <L> <W> <H>}"
H="${3:?usage: $0 <L> <W> <H>}"
OUT="outputs/single_room_3d/L${L}_W${W}_H${H}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; L=${L} W=${W} H=${H}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.eval.single_room_3d_eval \
    --L "${L}" --W "${W}" --H "${H}" \
    --rooms-yaml configs/sweeps_3d/derisk_rooms.yaml \
    --output_dir "${OUT}"
