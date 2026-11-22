#!/bin/bash
# P4-9 S-4 parity (D181 E8, c5): both envs on ONE node -- R0 @120K dense renders + 300 training steps of R0's recipe each.
#SBATCH --job-name=aaf_s4parity
#SBATCH --partition=scavenger --account=scavenger --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1 --cpus-per-task=4 --mem=64G --time=03:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -eo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH="${PWD}"
TP=/fs/nexus-scratch/htakawal/aaf_debug/s4_tp; mkdir -p "$TP"
python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/sweep_2d_mat/P4_8_R0.yaml"))
c.update(run_id="p4_9_S4tp", n_iters=300, early_stop_warmup=300, ckpt_every=100000, val_every=100000)
yaml.safe_dump(c, open("/fs/nexus-scratch/htakawal/aaf_debug/s4_tp/R0_tp.yaml", "w"))
PY
for ENV in aaf aaf_sm89; do
  conda activate $ENV
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  TAG=$([ $ENV = aaf ] && echo aaf || echo sm89)
  python scripts/p4_9_s4_parity.py --render $TAG
  rm -rf "$TP/out_$TAG"
  python -m aaf.train.multi_room_2d_mat --config "$TP/R0_tp.yaml" --output_dir "$TP/out_$TAG" > "$TP/train_$TAG.log" 2>&1 || true
  tail -3 "$TP/train_$TAG.log"
  conda deactivate
done
conda activate aaf; export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python scripts/p4_9_s4_parity.py --compare "$TP/train_aaf.log" "$TP/train_sm89.log"
