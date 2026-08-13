#!/bin/bash
#SBATCH --job-name=aaf_p3_2b_gate
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
echo "=== 1. CPU tests (must pass before training) ==="
python -m pytest -q tests/test_conditioning_encoder.py tests/test_baseline_dedup.py \
  tests/test_band_mask_2d.py tests/test_modal_bandwidth.py tests/test_modal_decay.py \
  tests/test_wall_convention.py 2>&1 | tail -3
echo "=== 2. DATASET GATE ==="
python scripts/p3_2b_dataset_gate.py
echo "=== 3. SMOKE per arm (300 iters, peak GPU memory) ==="
for A in A B C D; do
  echo "--- arm $A"
  timeout 1800 python -u -m aaf.train.multi_room_2d_mat \
    --config configs/sweep_2d_mat/P3_2b_${A}_smoke.yaml \
    --output_dir outputs/p3_2b/smoke_${A} 2>&1 | tail -25
done
