#!/bin/bash
# P4-9 X1 (D189): network-prior KRR fits (scripts/p4_9_netprior.py). CPU only; requeue-safe (one JSON per fit).
# Usage:  C=$(sbatch --parsable --export=ALL,MODE=cal scripts/slurm/p4_9_netprior_array.sh)
#         R=$(sbatch --parsable --dependency=afterok:$C --array=0-30 --export=ALL,MODE=room scripts/slurm/p4_9_netprior_array.sh)
#         sbatch --dependency=afterok:$R --export=ALL,MODE=agg scripts/slurm/p4_9_netprior_array.sh
#SBATCH --job-name=aaf_p49_netprior
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --exclude=legacy43
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID:-}; mode=${MODE}"
case "${MODE}" in
  cal)  python scripts/p4_9_netprior.py --selftest --calibrate ;;
  room) python scripts/p4_9_netprior.py --room "${SLURM_ARRAY_TASK_ID}" ;;
  iii)  python scripts/p4_9_netprior.py --room-iii "${SLURM_ARRAY_TASK_ID}" ;;
  agg)  python scripts/p4_9_netprior.py --aggregate ;;
esac
