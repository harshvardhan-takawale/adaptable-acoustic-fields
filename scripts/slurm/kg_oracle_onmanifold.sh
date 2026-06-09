#!/bin/bash
# P2-3.5 on-manifold oracle: best latent with ||z|| clipped to the training shell (8.4),
# 48 receivers, 1200 iters. Settles whether ANY on-manifold latent renders the room.
#SBATCH --job-name=aaf_kg_orcM
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
echo "host=$(hostname)"
python -m aaf.eval.known_geometry --mode oracle --out-sub oracle_onmanifold \
  --n-oracle-recv 48 --n-adapt-iters 1200 --lambda-latent 1e-3 --z-max-norm 8.4 \
  --train-output-dir outputs/multi_room_3d/P3_45rooms_4gpu \
  --output-dir outputs/known_geometry --rooms "${1:?need room}"
echo "# on-manifold oracle done"
