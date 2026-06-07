#!/bin/bash
# P2-3 closeout — aggregate the 8 zero-shot evals into SUMMARY.md (3-way verdict)
# and build the meeting plots. CPU-only work; tiny scavenger allocation.
#SBATCH --job-name=aaf_p3_closeout
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

RUN_ID="${1:-P3_45rooms_4gpu}"
TRAIN_OUT="outputs/multi_room_3d/${RUN_ID}"

echo "host=$(hostname); closeout for ${TRAIN_OUT}"
python scripts/multi_room_3d_summary.py --train-output-dir "${TRAIN_OUT}"
python scripts/make_p3_meeting_plots.py --train-output-dir "${TRAIN_OUT}" --top-k 3
echo "# closeout done"
