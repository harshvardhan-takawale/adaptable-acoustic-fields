#!/bin/bash
# Spec A validation — does manifold-anchored test-time adaptation (z_init=mean +
# stronger lambda) recover Run C zero-shot? Same 3 interior rooms, 24 GB rtxa5000
# scavenger card. Opt-in flags only; the default zero-shot path is unchanged.
#SBATCH --job-name=aaf_runC_zsfix
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
RUNC=outputs/diag_p2_2_5/C_10rm_b64
OUT=outputs/runC_zeroshot_probe_fix
mkdir -p "${OUT}"
echo "host=$(hostname)"; nvidia-smi -L || true
ROOMS=("4.50 4.00 3.25" "3.65 4.04 3.24" "4.36 4.14 3.96")
for r in "${ROOMS[@]}"; do
  read -r L W H <<< "$r"
  RDIR="${OUT}/L${L}_W${W}_H${H}"; mkdir -p "${RDIR}"
  echo "=== anchored zero-shot Run C -> L${L}_W${W}_H${H} (z_init=mean, lambda=1e-2) ==="
  python -m aaf.eval.zero_shot_3d \
    --target-h5 "data/track_a_3d/L${L}_W${W}_H${H}.h5" \
    --train-output-dir "${RUNC}" --output_dir "${RDIR}" \
    --z_init mean --lambda_latent 1e-2
done
echo "# fix-probe done"
