#!/bin/bash
# Dump GT + predicted spectra for the P3-2 meeting-pack Figure E. Usage:
#   sbatch scripts/slurm/p3_2_dump_fields.sh
#SBATCH --job-name=aaf_p3_2_fields
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"

echo "host=$(hostname); job=${SLURM_JOB_ID}"
nvidia-smi -L || true
python scripts/p3_2_dump_fields.py "$@"
