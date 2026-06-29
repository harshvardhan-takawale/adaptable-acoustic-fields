#!/bin/bash
# P2-4 per-density known-geometry eval on the FROZEN interior test set.
# Usage: sbatch density_eval.sh <train_output_dir> <label>
# Writes outputs/coverage_curve/eval_<label>/lookup/<room>__{rbf,linear}/metrics.json
#SBATCH --job-name=aaf_density_eval
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
TRAIN_DIR="${1:?usage: $0 <train_output_dir> <label>}"
LABEL="${2:?usage: $0 <train_output_dir> <label>}"
echo "host=$(hostname); density eval ${LABEL} <- ${TRAIN_DIR}"
# Build the "L W H; ..." rooms string from the frozen interior test yaml.
ROOMS=$(python - <<'PY'
import yaml
r=yaml.safe_load(open("configs/sweeps_3d/test_rooms_interior_frozen.yaml"))["rooms"]
print("; ".join(f"{x['L']:.2f} {x['W']:.2f} {x['H']:.2f}" for x in r))
PY
)
python -m aaf.eval.known_geometry --mode lookup \
  --train-output-dir "${TRAIN_DIR}" \
  --output-dir "outputs/coverage_curve/eval_${LABEL}" \
  --rooms "${ROOMS}"
echo "# density eval done: ${LABEL}"
