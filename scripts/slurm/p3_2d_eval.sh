#!/bin/bash
# P3-2d density-sweep evaluation for ONE arm. Usage:
#   sbatch scripts/slurm/p3_2d_eval.sh <arm-dir> <ARM> [extra args]
# e.g.
#   sbatch scripts/slurm/p3_2d_eval.sh outputs/p3_2d/p3_2d_W030_mlinear W030
#   sbatch scripts/slurm/p3_2d_eval.sh outputs/p3_2/p3_2b_C_cont_mlinear W015
#
# The estimator, kappa, controls C1-C4 and the hashed thresholds are the FROZEN P3-2b ones --
# this script only selects which manifest and which expected split counts apply, via
# --arm-spec. W015 is the already-trained P3-2b arm C and lives under outputs/p3_2/.
#SBATCH --job-name=aaf_p3_2d_eval
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

ARM_DIR="${1:?usage: $0 <arm-dir> <ARM> [extra args]}"
ARM="${2:?usage: $0 <arm-dir> <ARM> [extra args]}"
shift 2 || true
OUT="${OUT_ROOT:-outputs/p3_2d/eval}/${ARM}"
mkdir -p "${OUT}" logs/slurm
echo "host=$(hostname); job=${SLURM_JOB_ID:-none}; arm=${ARM}; dir=${ARM_DIR}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.eval.p3_2b_eval --arm-dir "${ARM_DIR}" --out "${OUT}" \
       --arm-spec "p3_2d:${ARM}" "$@"
