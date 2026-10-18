#!/bin/bash
# P4-8 Step 0b (D174(b)): dense FDTD for the frozen unseen-family set (2 rooms per task; D111 rooms are symlinked).
# Usage:  sbatch --array=0-7 scripts/slurm/build_p4_8_unseen_array.sh
#SBATCH --job-name=aaf_unseen
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}"
python scripts/build_p4_8_unseen.py --idx "${SLURM_ARRAY_TASK_ID}" --chunk 2
