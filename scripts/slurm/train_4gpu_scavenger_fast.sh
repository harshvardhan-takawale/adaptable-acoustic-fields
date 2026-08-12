#!/bin/bash
# P2-4 density training — parameterized 4-GPU DDP (clone of p3_train_4gpu.sh).
# Usage: sbatch density_train_4gpu.sh <config.yaml> <output_dir>
# Auto-resumes from the latest ckpt in <output_dir>; re-submit (afterany) to
# continue past the 24h cap.
#SBATCH --job-name=aaf_p3_fast
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
CONFIG="${1:?usage: $0 <config.yaml> <output_dir>}"
OUTDIR="${2:?usage: $0 <config.yaml> <output_dir>}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 1500))
export NCCL_DEBUG=WARN
echo "host=$(hostname); job=${SLURM_JOB_ID}; cfg=${CONFIG}; out=${OUTDIR}; MASTER=${MASTER_ADDR}:${MASTER_PORT}"
nvidia-smi -L || true
srun python -m aaf.train.multi_room_3d --config "${CONFIG}" --output_dir "${OUTDIR}" --ddp
echo "# density training step done: ${OUTDIR}"
