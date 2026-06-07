#!/bin/bash
# P2-3 DDP sanity gate — 4-GPU, 2000 iters. Confirms the 4-rank NCCL all-reduce
# works (no hang/desync), throughput scales (~0.6 it/s vs single-GPU eff-64's
# 0.157), and val@2000 descends like the 45-room early curve. BLOCKING: inspect
# its result before launching the long run (p3_train_4gpu.sh).
#SBATCH --job-name=aaf_p3_sanity
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29544
export NCCL_DEBUG=WARN

echo "host=$(hostname); job=${SLURM_JOB_ID}; ntasks=${SLURM_NTASKS}; MASTER=${MASTER_ADDR}:${MASTER_PORT}"
nvidia-smi -L || true

srun python -m aaf.train.multi_room_3d \
    --config configs/sweep_3d/P3_sanity.yaml \
    --output_dir outputs/multi_room_3d/P3_sanity \
    --ddp
