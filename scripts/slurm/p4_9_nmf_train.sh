#!/bin/bash
# P4-9 2c NMF / NMF-P (aaf/train/nmf_2d.py; D184). Resumes from its newest checkpoint after a pre-emption.
# Usage: sbatch scripts/slurm/p4_9_nmf_train.sh configs/sweep_2d_mat/P4_9_NMF.yaml
#SBATCH --job-name=aaf_p49_nmf
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --exclude=legacy43
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
CONFIG="${1:?config}"
RUN_ID=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['run_id'])")
OUT="outputs/p4_9/arms/${RUN_ID#p4_9_}"
mkdir -p "${OUT}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; config=${CONFIG}; out=${OUT}"
nvidia-smi -L || true
python -m aaf.train.nmf_2d --config "${CONFIG}" --output_dir "${OUT}"
