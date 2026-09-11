#!/bin/bash
# P4-1 evaluation: Gate 1 (L-room LOS/NLOS), its figures, and the DC before/after comparison.
#SBATCH --job-name=aaf_p4_1_eval
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
echo "host=$(hostname); job=${SLURM_JOB_ID}"
nvidia-smi -L || true

echo; echo "=============== GATE 1: L-room LOS/NLOS ==============="
python scripts/p4_1_stage1_eval.py
G1=$?
echo "[gate1 exit] ${G1}   (0=pass, 2=fail -- both are results; figures are drawn either way)"

echo; echo "=============== Stage 1 figures ==============="
python scripts/p4_1_stage1_figures.py

echo; echo "=============== DC mask: fair before/after ==============="
python scripts/p4_1_dc_compare.py
