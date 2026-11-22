#!/bin/bash
# P4-9 (D181 E8): the confirmation set (S-5 exception) and the second-source unseen rooms (S-3).
# Usage:  sbatch --array=0-7  --export=ALL,MODE=confirm scripts/slurm/p4_9_extra_rooms_array.sh
#         sbatch --array=0-15 --export=ALL,MODE=src2    scripts/slurm/p4_9_extra_rooms_array.sh
#SBATCH --job-name=aaf_p49_rooms
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}; mode=${MODE}"
python scripts/p4_9_build_extra_rooms.py "--${MODE}" --idx "${SLURM_ARRAY_TASK_ID}"
