#!/bin/bash
# Single-room 3D training. Pinned to tron RTX 2080 Ti (24 GB) per
# DECISIONS.md D13 — TITAN X (12 GB) is too small for 3D activations.
# Usage:
#   sbatch scripts/slurm/single_room_3d_train.sh <L> <W> <H>
#SBATCH --job-name=aaf_train_3d
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
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
mkdir -p "${OUT}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; L=${L} W=${W} H=${H}; out=${OUT}"
nvidia-smi -L || true

# Read the memory check's chosen config so we honor the cascade (D12).
# Falls back to D8/D12 defaults if no memory-check result is on disk.
MEM_RESULT="outputs/memory_check_3d/result.json"
if [[ -f "${MEM_RESULT}" ]]; then
    eval "$(python -c "
import json
r = json.load(open('${MEM_RESULT}'))
if r.get('status') == 'pass':
    c = r['chosen']
    print(f'CHOSEN_N_AZI={c[\"n_azi\"]}; CHOSEN_N_ELE={c[\"n_ele\"]}; CHOSEN_N_PTS={c[\"n_pts_per_ray\"]}; CHOSEN_BATCH={c[\"batch\"]}')
" 2>/dev/null)"
fi
N_AZI="${N_AZI:-${CHOSEN_N_AZI:-16}}"
N_ELE="${N_ELE:-${CHOSEN_N_ELE:-16}}"
N_PTS="${N_PTS:-${CHOSEN_N_PTS:-32}}"
BATCH="${BATCH:-${CHOSEN_BATCH:-8}}"
N_ITERS="${N_ITERS:-15000}"
echo "renderer config: n_azi=${N_AZI} n_ele=${N_ELE} n_pts=${N_PTS} batch=${BATCH} (mem-check chosen)"

python -m aaf.train.single_room_3d \
    --L "${L}" --W "${W}" --H "${H}" \
    --rooms-yaml configs/sweeps_3d/derisk_rooms.yaml \
    --output_dir "${OUT}" \
    --n_iters "${N_ITERS}" --batch_size "${BATCH}" \
    --n_azi "${N_AZI}" --n_ele "${N_ELE}" --n_pts_per_ray "${N_PTS}"
