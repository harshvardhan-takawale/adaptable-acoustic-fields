#!/bin/bash
# P4-2 Task A evaluation: demo protocol + the frozen P3-2b splits for geom_token_m.
#
# Writes to outputs/p4_2/taskA/, NOT outputs/p3_2b/eval/. The p3_2b_eval wrapper hardcodes the
# latter, and in P4-1 that put a P4-x arm inside the P3-2b tree where the publication guard
# demanded a results-doc row for it and failed the suite. Explicit --out avoids repeating that.
#SBATCH --job-name=aaf_p4_2_tA
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

CK=outputs/p4_2/taskA/p4_2_TgeoM/ckpt_iter0060000.pt
echo; echo "=========== demo protocol (spatial Pearson / band LSD) ==========="
python scripts/armC_demo_metrics.py --checkpoint "${CK}" \
  --out outputs/p4_2/taskA/metrics.json --npz-dir outputs/p4_2/taskA/fields

echo; echo "=========== frozen P3-2b splits (edit slope -- the Task A question) ==========="
python -m aaf.eval.p3_2b_eval --arm-dir outputs/p4_2/taskA/p4_2_TgeoM \
  --out outputs/p4_2/taskA/splits_eval
