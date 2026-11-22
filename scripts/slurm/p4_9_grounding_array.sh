#!/bin/bash
# P4-9 Step 1 fits (scripts/p4_9_grounding.py; D183). Requeue-safe: every fit writes its own JSON and is skipped if present.
# Usage:  sbatch --array=0-61 --export=ALL,MODE=net  --gres=gpu:rtxa5000:1 scripts/slurm/p4_9_grounding_array.sh
#         sbatch --array=0-30 --export=ALL,MODE=phys scripts/slurm/p4_9_grounding_array.sh
#SBATCH --job-name=aaf_p49_step1
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --exclude=legacy43
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
I=${SLURM_ARRAY_TASK_ID}
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${I}; mode=${MODE}"
if [ "${MODE}" = net ]; then
  MODELS=(R0_120K S2_120K)
  python scripts/p4_9_grounding.py --net "${MODELS[$((I / 31))]}" --room $((I % 31))
else
  python scripts/p4_9_grounding.py --phys --room "${I}"
fi
