#!/bin/bash
# P2-3.5 Exp1 — known-geometry lookup (HEADLINE) + LOO sanity. 24GB rtxa5000 scavenger.
#SBATCH --job-name=aaf_kg_lookup
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "host=$(hostname)"; nvidia-smi -L || true
# simulate the 3 interior rooms if absent
for i in 0 1 2; do python scripts/build_3d_dataset.py --rooms-yaml configs/sweeps_3d/interior_probe_rooms.yaml --idx $i; done
python -m aaf.eval.known_geometry --mode lookup --loo --n-loo 45 \
  --train-output-dir outputs/multi_room_3d/P3_45rooms_4gpu \
  --output-dir outputs/known_geometry \
  --rooms "4.50 4.00 3.25; 4.10 3.01 3.93; 5.94 4.93 2.51; 5.92 3.06 2.55; 5.91 4.17 3.72; 3.17 3.00 3.49; 5.99 3.96 2.54; 3.14 3.08 2.51; 4.40 4.09 3.26; 3.52 4.31 3.40; 4.82 3.81 2.92" \
  --plot-rooms "4.50 4.00 3.25; 4.40 4.09 3.26; 3.52 4.31 3.40; 4.82 3.81 2.92"
echo "# lookup done"
