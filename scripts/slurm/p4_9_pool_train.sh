#!/bin/bash
# P4-9: train ONE arm from a POOL of submissions on different GPU pools (whichever starts first trains). Exclusivity is
# an atomic lock directory (mkdir) in the arm's output dir: a copy that finds it held by a RUNNING job exits 0 at once;
# a stale lock (its owner job no longer running) is taken over. The trainer resumes from the newest full checkpoint,
# so any copy continues where the last one stopped (a resume always wins over init_ckpt, D181 A8).
# Usage: sbatch [pool options] --job-name=aaf_s2x --export=ALL,OUT_ROOT=outputs/p4_9/arms scripts/slurm/p4_9_pool_train.sh CONFIG
#SBATCH --cpus-per-task=4
#SBATCH --exclude=legacy43
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
CONFIG="${1:?usage: $0 <config-yaml>}"
RUN_ID=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['run_id'])")
OUT="${OUT_ROOT:?set OUT_ROOT}/${RUN_ID#p4_[0-9]_}"
mkdir -p "${OUT}" logs/slurm
LOCK="${OUT}/.pool_lock"
take() { mkdir "$LOCK" 2>/dev/null && echo "${SLURM_JOB_ID}" > "$LOCK/owner"; }
if ! take; then
  OWNER=$(cat "$LOCK/owner" 2>/dev/null || echo none)
  if squeue -j "$OWNER" -h -o %T 2>/dev/null | grep -q RUNNING; then
    echo "$(date) pool: ${RUN_ID} is already training in job ${OWNER}; this copy (${SLURM_JOB_ID}) exits"; exit 0
  fi
  echo "$(date) pool: stale lock (owner ${OWNER} not running) -> taking over"
  rm -rf "$LOCK"; take || { echo "lost the lock race"; exit 0; }
fi
trap 'rm -rf "$LOCK"' EXIT
echo "host=$(hostname); job=${SLURM_JOB_ID}; config=${CONFIG}; out=${OUT}"
nvidia-smi -L || true
python -m aaf.train.multi_room_2d_mat --config "${CONFIG}" --output_dir "${OUT}"
