#!/bin/bash
# P3-2b zero-shot material-edit evaluation for ONE arm (single GPU). Usage:
#   sbatch scripts/slurm/p3_2b_eval.sh <arm-dir> [extra args to aaf.eval.p3_2b_eval]
# e.g.
#   sbatch scripts/slurm/p3_2b_eval.sh outputs/p3_2/p3_2b_C_cont_mlinear
#   sbatch scripts/slurm/p3_2b_eval.sh outputs/p3_2/p3_2b_C_cont_mlinear --limit 4
#
# Training for all four arms may still be RUNNING. Pass --checkpoint explicitly when the
# newest file could be mid-write; without it the driver picks the newest ckpt_iter*.pt at
# job start, which races with the trainer's ckpt_every window.
#SBATCH --job-name=aaf_p3_2b_eval
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

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"

ARM_DIR="${1:?usage: $0 <arm-dir> [extra args]}"
shift || true
ARM=$(basename "${ARM_DIR}")
OUT="outputs/p3_2b/eval/${ARM}"
mkdir -p "${OUT}" logs/slurm
echo "host=$(hostname); job=${SLURM_JOB_ID:-none}; arm=${ARM}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.eval.p3_2b_eval --arm-dir "${ARM_DIR}" --out "${OUT}" "$@"
