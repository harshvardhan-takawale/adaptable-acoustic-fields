#!/bin/bash
# P4-9 modal solves (scripts/p4_9_modal.py): one (room, geometry variant) per task, m1's solver to 600 Hz.
# Usage:  sbatch --array=0-108 scripts/slurm/p4_9_modal_array.sh
#SBATCH --job-name=aaf_p49_modal
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
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}"
python scripts/p4_9_modal.py --idx "${SLURM_ARRAY_TASK_ID}" ${MODAL_EXTRA:-}
