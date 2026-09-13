#!/bin/bash
# P4-2 CORRECTION: re-evaluate the two extent_sum arms with the right pooling.
# load_model never passed token_pool, and extent_sum adds no parameters, so both arms were
# silently evaluated as masked_mean. D69 rests on those numbers.
#SBATCH --job-name=aaf_extent_reeval
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
echo "host=$(hostname); job=${SLURM_JOB_ID}"
for arm in extent_masked extent_unmasked mean_unmasked; do
  echo; echo "=================== ${arm} ==================="
  python scripts/p4_2_stage2_eval.py --run-dir "outputs/p4_2/stage2/p4_2_s2_${arm}" \
      --out "outputs/p4_2/stage2/p4_2_s2_${arm}_eval_fixed"
  echo "[exit ${arm}] $?"
done
