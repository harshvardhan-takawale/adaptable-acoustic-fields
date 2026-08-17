#!/bin/bash
# P3-3-FAST Track 2b aperture evaluation (single GPU). Usage:
#   sbatch scripts/slurm/p3_3fast_trackB_eval.sh [extra args to the eval script]
# e.g. pin the checkpoint (recommended while training is live):
#   sbatch scripts/slurm/p3_3fast_trackB_eval.sh \
#       --checkpoint outputs/p3_3fast/p3_3fast_trackB/ckpt_iter0030000.pt
#
# Track 2b training may still be RUNNING. Pass --checkpoint explicitly when the newest file
# could be mid-write; without it the driver picks the newest ckpt_iter*.pt at job start,
# which races with the trainer's ckpt_every window and can load a half-written file.
#
# Cost: 72 test configs x 64 receivers. Rendering dominates; the GT-side measurement half
# takes ~20 s total on CPU. Budget well under the hour requested below.
#
# Writes outputs/p3_3fast/trackB/EVAL.json and outputs/p3_3fast/trackB/EVAL.md.
# --gt-only writes EVAL_GT_ONLY.{json,md} instead and needs no GPU at all.
#SBATCH --job-name=aaf_p33B_eval
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"

mkdir -p outputs/p3_3fast/trackB logs/slurm
echo "host=$(hostname); job=${SLURM_JOB_ID:-none}"
nvidia-smi -L || true

python scripts/p3_3fast_trackB_eval.py --out outputs/p3_3fast/trackB "$@"

# Demo figure: predicted vs GT spectra at a = 0 (sealed / topological), 0.3, 1.0 (HELD OUT)
# and 2.0, both sub-rooms. Same --checkpoint is forwarded when one was given.
python scripts/p3_3fast_trackB_demo_fig.py --out outputs/p3_3fast/trackB "$@" || \
    echo "[warn] demo figure failed; EVAL.json/EVAL.md are already written"
