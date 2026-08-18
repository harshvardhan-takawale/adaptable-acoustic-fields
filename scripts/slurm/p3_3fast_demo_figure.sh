#!/bin/bash
# FIG 5 of the P3-3-FAST meeting pack: "edit the room, zero-shot".
#
# Renders 5 zero-shot forward passes from the converged Track A2 checkpoint and writes
#   outputs/p3_3fast/meeting_assets/fig5_topological_edits.png (+ .json sidecar)
# and appends/refreshes the FIG 5 section of FIGURE_MANIFEST.md.
#
# Needs a GPU: tinycudann has no CPU path. Also (re)computes the corpus-wide raw-vs-floored
# LSD at the SAME checkpoint the figure uses, so the number quoted in the figure note is not
# borrowed from a different iteration.
#
# Usage:  sbatch scripts/slurm/p3_3fast_demo_figure.sh
#SBATCH --job-name=aaf_p33_demofig
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
# Required: conda's libstdc++ (GLIBCXX_3.4.29) must precede /lib64's older copy.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"

CKPT="${CKPT:-outputs/p3_3fast/p3_3fast_trackA2/ckpt_iter0030000.pt}"
echo "host=$(hostname); job=${SLURM_JOB_ID:-none}; ckpt=${CKPT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# 1. corpus-wide raw vs floored LSD at the figure's own checkpoint, over ALL 120 test
#    configs (the script's default --limit 24 covers only test geometries 0-1, which is not
#    a corpus-wide number). Idempotent; ~8 min.
if [ ! -f outputs/p3_3fast/floored_lsd_30k.json ]; then
  python scripts/p3_3fast_floored_lsd.py \
    --checkpoint "${CKPT}" --limit 120 \
    --out outputs/p3_3fast/floored_lsd_30k.json
fi

# 2. the figure itself
python scripts/p3_3fast_demo_figure.py --checkpoint "${CKPT}" "$@"
