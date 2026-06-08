#!/bin/bash
# Spec A disambiguation: init z* on the manifold (z_init=mean) but WEAK lambda=1e-4
# (free to move from a good start). Separates "lambda over-pins" from "no good
# latent exists (coverage)". Box center + L3.65; 2h walltime to avoid TIMEOUT.
#SBATCH --job-name=aaf_runC_initonly
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
RUNC=outputs/diag_p2_2_5/C_10rm_b64
OUT=outputs/runC_zeroshot_initonly
mkdir -p "${OUT}"; echo "host=$(hostname)"
for r in "4.50 4.00 3.25" "3.65 4.04 3.24"; do
  read -r L W H <<< "$r"
  RDIR="${OUT}/L${L}_W${W}_H${H}"; mkdir -p "${RDIR}"
  echo "=== init-only zero-shot Run C -> L${L}_W${W}_H${H} (z_init=mean, lambda=1e-4) ==="
  python -m aaf.eval.zero_shot_3d \
    --target-h5 "data/track_a_3d/L${L}_W${W}_H${H}.h5" \
    --train-output-dir "${RUNC}" --output_dir "${RDIR}" \
    --z_init mean --lambda_latent 1e-4
done
echo "# init-only done"
