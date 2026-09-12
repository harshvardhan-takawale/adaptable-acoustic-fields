#!/bin/bash
# P4-2 Gate 2: evaluate all four Stage 2 arms. One job, four arms, sequential on one GPU.
#SBATCH --job-name=aaf_p4_2_g2
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=04:00:00
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
for arm in mean_masked extent_masked mean_unmasked extent_unmasked; do
  echo; echo "=================== ${arm} ==================="
  python scripts/p4_2_stage2_eval.py --run-dir "outputs/p4_2/stage2/p4_2_s2_${arm}"
  echo "[gate2 exit ${arm}] $?"
done
