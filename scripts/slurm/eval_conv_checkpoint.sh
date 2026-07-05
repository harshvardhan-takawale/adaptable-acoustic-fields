#!/bin/bash
# P2-4b: known-geometry full-suite eval of ONE checkpoint from a training run,
# selected by its in-distribution val LSD (for matched-convergence comparison).
# Usage: sbatch eval_conv_checkpoint.sh <source_train_dir> <target_lsd> <label>
#   e.g. sbatch eval_conv_checkpoint.sh outputs/multi_room_3d/density_45_conv 4.30 conv45_lsd43
# Stages the ckpt nearest <target_lsd> (+ train_meta) into a temp dir so
# _load_trained_model (which takes the highest-iter ckpt) loads exactly it, then
# runs the same known_geometry lookup on the FROZEN interior test set.
#SBATCH --job-name=aaf_conv_eval
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

SRC="${1:?usage: $0 <source_train_dir> <target_lsd> <label>}"
TARGET_LSD="${2:?usage: $0 <source_train_dir> <target_lsd> <label>}"
LABEL="${3:?usage: $0 <source_train_dir> <target_lsd> <label>}"

# Pick the checkpoint whose in-dist val LSD is nearest TARGET_LSD.
read -r CKPT_ITER CKPT_LSD < <(python - "$SRC" "$TARGET_LSD" <<'PY'
import json, sys, glob, os
src, target = sys.argv[1], float(sys.argv[2])
val = [x for x in json.load(open(f"{src}/scalars.json"))
       if x.get("phase") == "val" and x.get("lsd_db") is not None]
vmap = {int(x["iter"]): float(x["lsd_db"]) for x in val}
ckpts = []
for p in glob.glob(f"{src}/ckpt_iter*.pt"):
    it = int(os.path.basename(p).split("ckpt_iter")[-1].split(".")[0])
    # nearest val LSD to this ckpt iter (val cadence may be finer)
    if it in vmap:
        ckpts.append((it, vmap[it]))
    elif vmap:
        near = min(vmap, key=lambda k: abs(k - it))
        ckpts.append((it, vmap[near]))
if not ckpts:
    raise SystemExit("no checkpoints with a matching val LSD")
best = min(ckpts, key=lambda c: abs(c[1] - target))
print(best[0], f"{best[1]:.4f}")
PY
)
echo "host=$(hostname); label=${LABEL}; picked ckpt_iter${CKPT_ITER} (val LSD ${CKPT_LSD}, target ${TARGET_LSD}) from ${SRC}"

# Stage the single checkpoint + train_meta into an isolated dir.
STAGE="outputs/multi_room_3d/_conv_stage/${LABEL}"
rm -rf "${STAGE}"; mkdir -p "${STAGE}"
CKPT_FILE=$(printf "ckpt_iter%07d.pt" "${CKPT_ITER}")
ln -sf "$(readlink -f "${SRC}/${CKPT_FILE}")" "${STAGE}/${CKPT_FILE}"
ln -sf "$(readlink -f "${SRC}/train_meta.json")" "${STAGE}/train_meta.json"

ROOMS=$(python - <<'PY'
import yaml
r=yaml.safe_load(open("configs/sweeps_3d/test_rooms_interior_frozen.yaml"))["rooms"]
print("; ".join(f"{x['L']:.2f} {x['W']:.2f} {x['H']:.2f}" for x in r))
PY
)
python -m aaf.eval.known_geometry --mode lookup \
  --train-output-dir "${STAGE}" \
  --output-dir "outputs/coverage_curve/eval_${LABEL}" \
  --rooms "${ROOMS}"

# Record which checkpoint/LSD this eval used (provenance for CONFOUND_CHECK.md).
python - "$LABEL" "$CKPT_ITER" "$CKPT_LSD" "$TARGET_LSD" "$SRC" <<'PY'
import json, sys
label, it, lsd, target, src = sys.argv[1:6]
open(f"outputs/coverage_curve/eval_{label}/provenance.json", "w").write(json.dumps(
    {"label": label, "source": src, "ckpt_iter": int(it),
     "indist_val_lsd_db": float(lsd), "target_lsd_db": float(target)}, indent=2))
PY
echo "# conv eval done: ${LABEL} (ckpt ${CKPT_ITER} @ ${CKPT_LSD} dB)"
