#!/bin/bash
#SBATCH --job-name=aaf_p3_2b_final
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
for A in p3_2b_C_cont_mlinear p3_2b_B_cont_fourier p3_2b_A_preset_fourier; do
  echo "### m_response $A"
  python -u -m aaf.eval.p3_2b_mresponse --arm "$A" 2>&1 | tail -6
done
echo "### figures"
python -u scripts/make_p3_2b_figures.py 2>&1 | tail -20
