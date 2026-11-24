#!/bin/bash
# P4-5: a single-GPU diagnostic or evaluation job. CMD is the python command line to run.
# Usage:  sbatch --export=ALL,CMD="scripts/p4_5_a1_bin_gradient.py --smoke" scripts/slurm/p4_5_gpu.sh
#SBATCH --job-name=aaf_p4_5_gpu
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --exclude=legacy43
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -uo pipefail
# SCRATCH GUARD (2026-11-24: some nodes lost /fs/nexus-scratch): wait up to 2 min for the automount, else requeue
for _i in 1 2 3 4 5 6; do [ -f /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh ] && break; sleep 20; done
if [ ! -f /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh ]; then echo "no scratch on $(hostname); requeueing"; scontrol requeue "$SLURM_JOB_ID" || true; sleep 5; exit 1; fi
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; cmd=${CMD}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# max_split_size_mb: the renderer graph sits near the 24 GB ceiling; limit fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python ${CMD}
