#!/bin/bash
# P4-7 m3 (D170(g)): modal prior + features per room. SETS env var = comma list (see the script).
# Usage:  sbatch --array=0-59 --export=ALL,SETS=famtrain+testdense+shapestest+famdense+famoffl scripts/slurm/p4_7_m3_precompute_array.sh
# ('+' separates sets: sbatch --export splits its value on commas.)
#SBATCH --job-name=aaf_m3pre
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=03:00:00
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
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}; sets=${SETS}"
python scripts/p4_7_m3_precompute.py --sets "${SETS}" --idx "${SLURM_ARRAY_TASK_ID}" --chunk 5
