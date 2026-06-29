#!/bin/bash
# P2-4 dataset gen — parameterized array. Usage:
#   sbatch --array=0-249%32 scripts/slurm/build_3d_array.sh <rooms_yaml>
# Idempotent (.done sentinel) — already-simulated rooms exit 0 immediately.
#SBATCH --job-name=aaf_build3d
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
ROOMS_YAML="${1:?usage: sbatch --array=A-B $0 <rooms_yaml>}"
python scripts/build_3d_dataset.py --rooms-yaml "${ROOMS_YAML}" --idx "${SLURM_ARRAY_TASK_ID}"
