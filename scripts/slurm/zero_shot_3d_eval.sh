#!/bin/bash
# Zero-shot 3D eval for one test room. Usage:
#   sbatch scripts/slurm/zero_shot_3d_eval.sh <RUN_ID> <L> <W> <H>
#SBATCH --job-name=aaf_zs_3d
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
# n_pts=32 models (P2-2.5 Run C, P3) need ~12 GB for the renderer — an 11 GB
# legacy scavenger card OOMs. Name a 24 GB rtxa5000 (still scavenger qos).
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

RUN_ID="${1:?usage: $0 <RUN_ID> <L> <W> <H>}"
L="${2:?usage: $0 <RUN_ID> <L> <W> <H>}"
W="${3:?usage: $0 <RUN_ID> <L> <W> <H>}"
H="${4:?usage: $0 <RUN_ID> <L> <W> <H>}"

TARGET_H5="data/track_a_3d/L${L}_W${W}_H${H}.h5"
TRAIN_OUT="outputs/multi_room_3d/${RUN_ID}"
OUT="${TRAIN_OUT}/zero_shot/L${L}_W${W}_H${H}"
mkdir -p "${OUT}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; run=${RUN_ID}; target=${TARGET_H5}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.eval.zero_shot_3d \
    --target-h5 "${TARGET_H5}" \
    --train-output-dir "${TRAIN_OUT}" \
    --output_dir "${OUT}"
