#!/bin/bash
# P4-3 guard: the P4-2 baseline must be BIT-IDENTICAL after the inr_2d/trainer changes.
# Re-runs Gate 2 on mean_unmasked; every number must reproduce or the baseline every P4-3 arm
# is read against has silently moved.
#SBATCH --job-name=aaf_p4_3_regress
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --time=01:00:00
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
echo "host=$(hostname); job=${SLURM_JOB_ID}"
python scripts/p4_2_stage2_eval.py --run-dir outputs/p4_2/stage2/p4_2_s2_mean_unmasked \
       --out outputs/p4_2/stage2/p4_2_s2_mean_unmasked_eval_regress
