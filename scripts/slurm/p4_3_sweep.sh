#!/bin/bash
# P4-3: the shape-edit sweep curve for one arm. Usage: sbatch p4_3_sweep.sh <run-dir> <tag>
#SBATCH --job-name=aaf_p4_3_sweep
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
RUN="${1:?usage: $0 <run-dir> <tag>}"
TAG="${2:?usage: $0 <run-dir> <tag>}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; run=${RUN}"
python scripts/p4_2_sweep_figures.py --run-dir "${RUN}" --out "outputs/p4_3/sweep/${TAG}"
