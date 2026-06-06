#!/bin/bash
# 2-GPU DDP smoke test (~5-8 min): confirm NCCL + manual all-reduce + rank-0
# val/ckpt/broadcast all work before committing the full DDP-B run.
#SBATCH --job-name=aaf_ddp_smoke
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29541
# NCCL: single node, be permissive about transport.
export NCCL_DEBUG=WARN

rm -rf outputs/diag_p2_2_5/_ddp_smoke
echo "host=$(hostname); job=${SLURM_JOB_ID}; MASTER=${MASTER_ADDR}:${MASTER_PORT}"
nvidia-smi -L || true

srun python -m aaf.train.multi_room_3d \
    --config configs/sweep_3d/_ddp_smoke.yaml \
    --output_dir outputs/diag_p2_2_5/_ddp_smoke \
    --ddp
