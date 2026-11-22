#!/bin/bash
# P4-9 (D180 0b / 0c / 2b): FDTD reruns recording gradient / wall stencils (scripts/p4_9_fdtd_stencils.py).
# Usage:  sbatch --array=0-30 --export=ALL,SET=dense scripts/slurm/p4_9_fdtd_stencils_array.sh
#         sbatch --array=0-124 --export=ALL,SET=gx,CHUNK=8 scripts/slurm/p4_9_fdtd_stencils_array.sh
# <= 5000 recorded nodes per run (~2 MB each) -> 16G.
#SBATCH --job-name=aaf_p49_sten
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=02:00:00
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
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}; set=${SET}"
python scripts/p4_9_fdtd_stencils.py --set "${SET}" --idx "${SLURM_ARRAY_TASK_ID}" --chunk "${CHUNK:-8}"
