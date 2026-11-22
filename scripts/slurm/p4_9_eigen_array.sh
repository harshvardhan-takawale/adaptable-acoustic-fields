#!/bin/bash
# P4-9 2c eigen-labels (scripts/p4_9_eigen_labels.py). Usage:
#   sbatch --array=0-124 --export=ALL,SET=gx scripts/slurm/p4_9_eigen_array.sh      (8 GX rooms per task)
#   sbatch --array=0-30  --export=ALL,SET=dense scripts/slurm/p4_9_eigen_array.sh
#SBATCH --job-name=aaf_p49_eig
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
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}; set=${SET}"
python scripts/p4_9_eigen_labels.py --set "${SET}" --idx "${SLURM_ARRAY_TASK_ID}"
