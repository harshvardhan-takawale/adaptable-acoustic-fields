#!/bin/bash
# P2-3 converged training — 4-GPU DDP, eff-batch 64, 60K iters (D36 user cap).
# Auto-resumes from the latest ckpt_iter*.pt in the output dir, so re-submitting
# the SAME command (afterany dependency) continues past the 24h qos cap to 60K.
#SBATCH --job-name=aaf_p3_train
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29545
export NCCL_DEBUG=WARN

echo "host=$(hostname); job=${SLURM_JOB_ID}; ntasks=${SLURM_NTASKS}; MASTER=${MASTER_ADDR}:${MASTER_PORT}"
nvidia-smi -L || true

srun python -m aaf.train.multi_room_3d \
    --config configs/sweep_3d/P3_45rooms_4gpu.yaml \
    --output_dir outputs/multi_room_3d/P3_45rooms_4gpu \
    --ddp
