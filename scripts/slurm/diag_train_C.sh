#!/bin/bash
# P2-2.5 Run C: 10-room subset, micro_batch=8 + grad_accum=8 → effective
# batch=64, n_pts=32, tron qos=high. High-coverage ceiling test.
# GPU: rtxa6000 (48 GB). MUST name the GPU type — a bare `--gres=gpu:1`
# lands on an 11 GB 2080 Ti and OOMs (it did, in validate() at 15 min).
#SBATCH --job-name=aaf_diag_C
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

CONFIG="configs/sweep_3d/C_diag.yaml"
OUT="outputs/diag_p2_2_5/C_10rm_b64"
mkdir -p "${OUT}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; config=${CONFIG}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.train.multi_room_3d --config "${CONFIG}" --output_dir "${OUT}"
