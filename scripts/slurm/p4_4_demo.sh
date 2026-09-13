#!/bin/bash
# P4-4 Task 1: the shape-edit demo pack. Panels from X1, full-range morph from D.
#SBATCH --job-name=aaf_p4_4_demo
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
echo "host=$(hostname); job=${SLURM_JOB_ID}"
python scripts/p4_3_demo_pack.py \
    --run-dir outputs/p4_3/p4_3_X1_attn_residual \
    --morph-run-dir outputs/p4_3/p4_3_D_data \
    --morph-full-range \
    --out outputs/p4_4/demo
