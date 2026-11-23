#!/bin/bash
# P4-10: train ONE arm. Stages the memmap corpora it needs from /fs/nexus-scratch to the node's local disk
# (/scratch0/htakawal/aaf_p410/<name>; the configs point there, so a resumed job on another node finds the same path)
# before starting. Exclusivity and resume as in p4_9_pool_train.sh (atomic lock dir in the arm's output dir).
# Usage: sbatch [opts] --export=ALL,OUT_ROOT=outputs/p4_10/arms,STAGE="ms_mm" scripts/slurm/p4_10_train.sh CONFIG
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
OUT="${OUT_ROOT:?set OUT_ROOT}/${RUN_ID#p4_10_}"
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
echo "host=$(hostname); job=${SLURM_JOB_ID}; config=${CONFIG}; out=${OUT}; stage=${STAGE:-}"
nvidia-smi -L || true
for NAME in ${STAGE:-}; do                                  # stage each memmap directory to the node's local disk
  SRC=/fs/nexus-scratch/htakawal/aaf_p410/$NAME; DST=/scratch0/htakawal/aaf_p410/$NAME
  if [ ! -w /scratch0 ]; then echo "no writable /scratch0 on this node: reading $NAME from $SRC"; DST=$SRC; continue; fi
  mkdir -p "$(dirname "$DST")"
  if [ -f "$DST/index.json" ] && [ "$(stat -c %s "$DST/H.npy" 2>/dev/null)" = "$(stat -c %s "$SRC/H.npy")" ]; then
    echo "$(date '+%H:%M:%S') $NAME already staged at $DST"
  else
    rm -rf "$DST.tmp"; mkdir -p "$DST.tmp"
    S0=$(date +%s)
    if cp "$SRC"/* "$DST.tmp"/ && [ "$(stat -c %s "$DST.tmp/H.npy")" = "$(stat -c %s "$SRC/H.npy")" ] && rm -rf "$DST" && mv "$DST.tmp" "$DST"; then
      echo "$(date '+%H:%M:%S') staged $NAME to $DST in $(( $(date +%s) - S0 )) s"
    else
      rm -rf "$DST.tmp"; echo "FAILED to stage $NAME (disk full?); see the .err file"; df -h /scratch0 | tail -1; exit 1
    fi
  fi
done
python -m aaf.train.multi_room_2d_mat --config "${CONFIG}" --output_dir "${OUT}"
