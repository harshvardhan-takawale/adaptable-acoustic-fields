#!/bin/bash
# P4-10 CPU jobs (FDTD arrays, assembly). Usage: sbatch [--array=...] --export=ALL,CMD="scripts/x.py --idx @" scripts/slurm/p4_10_cpu.sh
# ('@' in CMD is replaced by the array task id; no commas in CMD -- sbatch --export splits on them)
#SBATCH --job-name=aaf_p410
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --requeue
#SBATCH --exclude=legacy43
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail
# SCRATCH GUARD (2026-11-24: some nodes lost /fs/nexus-scratch): wait up to 2 min for the automount, else requeue this task
for _i in 1 2 3 4 5 6; do [ -f /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh ] && break; sleep 20; done
if [ ! -f /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh ]; then echo "no scratch on $(hostname); requeueing"; scontrol requeue "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}" || true; sleep 5; exit 1; fi
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
export OMP_NUM_THREADS=${THREADS:-1} MKL_NUM_THREADS=${THREADS:-1} OPENBLAS_NUM_THREADS=${THREADS:-1} NUMEXPR_NUM_THREADS=1
C="${CMD//@/${SLURM_ARRAY_TASK_ID:-}}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID:-}; cmd=${C}"
python ${C}
