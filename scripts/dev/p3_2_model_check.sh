#!/bin/bash
#SBATCH --job-name=aaf_p3_2_modelchk
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
python scripts/dev/p3_2_model_check.py
echo "--- existing 2D model tests (back-compat) ---"
python -m pytest -q tests/test_autodecoder_2d.py tests/test_film_conditioning.py \
  tests/test_latent_jitter.py tests/test_l_head.py tests/test_model_2d.py 2>&1 | tail -4
