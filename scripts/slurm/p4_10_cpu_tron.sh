#!/bin/bash
# P4-10 CPU jobs on tron (nexus, qos high: better priority than scavenger; no GPU). Usage: sbatch --export=ALL,CMD="scripts/x.py --flag" scripts/slurm/p4_10_cpu_tron.sh
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="${PWD}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "host=$(hostname); cmd=${CMD}"
eval "${CMD}"
