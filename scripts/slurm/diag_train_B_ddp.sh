#!/bin/bash
# P2-2.5 Run B — 2-GPU DDP speedup. ~2x the single-GPU rate → B to 60K in ~16h.
# Resumes from single-B's checkpoint (copied into the ddp output dir first).
# qos=medium (2 GPU / 8 CPU) so it fits alongside the single-GPU B+C insurance
# (qos=high, 2 GPU / 16 CPU) under the 4-GPU / 32-CPU partition cap.
#SBATCH --job-name=aaf_diag_B_ddp
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29542
export NCCL_DEBUG=WARN

echo "host=$(hostname); job=${SLURM_JOB_ID}; MASTER=${MASTER_ADDR}:${MASTER_PORT}"
nvidia-smi -L || true

srun python -m aaf.train.multi_room_3d \
    --config configs/sweep_3d/B_ddp.yaml \
    --output_dir outputs/diag_p2_2_5/B_45rm_ddp \
    --ddp
