#!/bin/bash
#SBATCH --job-name=aaf_p4_1_gate0
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
echo "host=$(hostname); job=${SLURM_JOB_ID}"; nvidia-smi -L || true
echo; echo "=============== GATE 0: Arm T-geo vs Arm C ==============="
python scripts/p4_1_gate0.py
echo "[gate0 exit] $?"
