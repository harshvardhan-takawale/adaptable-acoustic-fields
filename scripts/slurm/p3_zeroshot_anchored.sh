#!/bin/bash
# P2-3 disambiguation: does manifold-anchored adaptation (z_init=mean) recover P3's
# zero-shot? If yes -> P2-4 = fix the z* search (procedure). If no -> P2-4 = more
# rooms (coverage). 24GB rtxa5000 scavenger card (n_pts=32). Brackets lambda on the
# box center; lambda=1e-2 on a small + large room. Compare to stock (mag 0.27/0.28/0.28).
#SBATCH --job-name=aaf_p3_zsanchor
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
P3=outputs/multi_room_3d/P3_45rooms_4gpu
OUT=outputs/p3_zeroshot_anchored; mkdir -p "${OUT}"; echo "host=$(hostname)"
run() {  # L W H lambda tag
  local L=$1 W=$2 H=$3 LAM=$4 TAG=$5
  local RDIR="${OUT}/L${L}_W${W}_H${H}__${TAG}"; mkdir -p "${RDIR}"
  echo "=== anchored zero-shot P3 -> L${L}_W${W}_H${H} (z_init=mean lambda=${LAM}) ==="
  python -m aaf.eval.zero_shot_3d --target-h5 "data/track_a_3d/L${L}_W${W}_H${H}.h5" \
    --train-output-dir "${P3}" --output_dir "${RDIR}" --z_init mean --lambda_latent "${LAM}"
}
run 4.50 4.00 3.25 1e-2 lam1e-2     # box center, two lambda
run 4.50 4.00 3.25 1e-1 lam1e-1
run 3.17 3.00 3.49 1e-2 lam1e-2     # small room
run 5.91 4.17 3.72 1e-2 lam1e-2     # large room
echo "# p3 anchored disambiguation done"
