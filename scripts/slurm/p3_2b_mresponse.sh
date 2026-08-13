#!/bin/bash
# P3-2b model-side m-response for one arm (GPU; ~250 renders).
#   sbatch scripts/slurm/p3_2b_mresponse.sh p3_2b_C_cont_mlinear [checkpoint.pt]
# Uses the newest ckpt_iter*.pt in the arm's output dir unless one is given, so it can be
# run against a still-training arm. Reads outputs/p3_2b/mresponse_gt.json; writes
# outputs/p3_2b/eval/<arm>/m_response.json.
#SBATCH --job-name=aaf_p3_2b_mresp
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
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

ARM="${1:?usage: $0 <arm-run-id> [checkpoint]}"
CKPT="${2:-}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; arm=${ARM}; ckpt=${CKPT:-newest}"
nvidia-smi -L || true
if [ -n "${CKPT}" ]; then
  python -W ignore -m aaf.eval.p3_2b_mresponse --arm "${ARM}" --checkpoint "${CKPT}" \
      --rx-chunk "${RX_CHUNK:-4}"
else
  python -W ignore -m aaf.eval.p3_2b_mresponse --arm "${ARM}" --rx-chunk "${RX_CHUNK:-4}"
fi
