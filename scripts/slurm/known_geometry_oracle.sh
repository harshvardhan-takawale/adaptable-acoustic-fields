#!/bin/bash
# P2-3.5 Exp2 — oracle latent (best the decoder can represent). 24GB rtxa5000 scavenger.
# Usage: sbatch known_geometry_oracle.sh "L W H; L W H; ..."
#SBATCH --job-name=aaf_kg_oracle
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=02:00:00
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
python -m aaf.eval.known_geometry --mode oracle --n-oracle-recv 32 --n-adapt-iters 1200 \
  --train-output-dir outputs/multi_room_3d/P3_45rooms_4gpu \
  --output-dir outputs/known_geometry \
  --rooms "${1:?usage: $0 \"L W H; ...\"}"
echo "# oracle done"
