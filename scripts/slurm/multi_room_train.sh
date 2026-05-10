#!/bin/bash
#SBATCH --job-name=aaf_mr_train
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Multi-room auto-decoder training. Defaults to tron (non-preemptible). Falls back
# to scavenger by changing the three SBATCH lines above to:
#   #SBATCH --partition=scavenger
#   #SBATCH --account=scavenger
#   #SBATCH --qos=scavenger
# CLUSTER_INFO.md documents the fallback policy.

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

OUT="${1:-outputs/multi_room/dense}"
SWEEP="${2:-configs/sweeps/dense.yaml}"

# Read the chosen batch + grad-accum from the memory-check report (default fallback
# values are batch=8, grad_accum=2 — Chunk-2 single-room budget).
RESULT_JSON="outputs/memory_check/multi_room_result.json"
if [ -f "${RESULT_JSON}" ]; then
    BATCH=$(python -c "import json; print(json.load(open('${RESULT_JSON}'))['chosen']['batch'])")
    GA=$(python -c "import json; print(json.load(open('${RESULT_JSON}'))['grad_accum_steps'])")
else
    BATCH=8
    GA=2
fi
echo "host=$(hostname); job=${SLURM_JOB_ID}; out=${OUT}; batch=${BATCH}; grad_accum=${GA}"
nvidia-smi -L || true

mkdir -p "${OUT}"
python -m aaf.train.multi_room \
    --sweep "${SWEEP}" \
    --output_dir "${OUT}" \
    --batch_size "${BATCH}" \
    --grad_accum_steps "${GA}"
