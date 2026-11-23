#!/bin/bash
# P4-9 X2 (D190): fast-physics baseline (scripts/p4_9_fast_physics.py). CPU only; requeue-safe per room.
# Usage:  sbatch --array=0-30 --export=ALL,MODE=room scripts/slurm/p4_9_fastphys.sh
#         sbatch --constraint=E5-2680 --exclusive=user --export=ALL,MODE=timing scripts/slurm/p4_9_fastphys.sh
#         sbatch --mem=96G --time=03:00:00 --export=ALL,MODE=scaling scripts/slurm/p4_9_fastphys.sh
#         sbatch --export=ALL,MODE=agg scripts/slurm/p4_9_fastphys.sh
#SBATCH --job-name=aaf_p49_fastphys
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
  room)    python scripts/p4_9_fast_physics.py --room "${SLURM_ARRAY_TASK_ID}" ;;
  timing)  python scripts/p4_9_fast_physics.py --timing ;;
  scaling) python scripts/p4_9_fast_physics.py --scaling ;;
  agg)     python scripts/p4_9_fast_physics.py --aggregate ;;
esac
