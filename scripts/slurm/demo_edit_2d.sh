#!/bin/bash
# P3-2 zero-shot material-edit demo. Every argument is forwarded to the script, e.g.
#   sbatch scripts/slurm/demo_edit_2d.sh --L 4.51 --W 4.00 --wall west --material curtain \
#          --gt data/track_c_2d
# A GPU is required: the model imports tinycudann, which raises "Unknown compute
# capability" on a login node.
#SBATCH --job-name=aaf_demo_edit_2d
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; args=$*"
nvidia-smi -L || true
python scripts/demo_edit_2d.py "$@"
