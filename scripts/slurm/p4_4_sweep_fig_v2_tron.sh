#!/bin/bash
# P4-4 demo v2: render one sweep strip. One job per sweep, so they run CONCURRENTLY -- the
# wall-clock is one sweep (~8 rooms x ~2 min), not the sum over five.
# Usage:  sbatch --export=ALL,SWEEP=s1_notch_width scripts/slurm/p4_4_sweep_fig_v2.sh
#SBATCH --job-name=aaf_v2fig_tron
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --exclude=legacy43
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; sweep=${SWEEP}"
python scripts/p4_4_sweep_figures_v2.py --sweep "${SWEEP}" ${EXTRA:-}
