#!/bin/bash
# P3-2 zero-shot material-edit eval (single GPU). Usage:
#   sbatch scripts/slurm/p3_2_eval.sh [train_dir] [out_dir] [extra args...]
# Defaults to the newest checkpoint in outputs/p3_2/p3_2_main; training may still be
# running -- a mid-flight checkpoint is a valid input, the eval just reports its iter.
#SBATCH --job-name=aaf_p3_2_eval
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"

TRAIN_DIR="${1:-outputs/p3_2/p3_2_main}"; { [ $# -ge 1 ] && shift; } || true
OUT="${1:-outputs/p3_2/eval}";            { [ $# -ge 1 ] && shift; } || true
mkdir -p "${OUT}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; train_dir=${TRAIN_DIR}; out=${OUT}"
nvidia-smi -L || true

python -m aaf.eval.p3_2_eval --train-dir "${TRAIN_DIR}" --out "${OUT}" "$@"

# Fail the job if the schema the figure script consumes is not intact.
python - "${OUT}/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
need_top = ("checkpoint", "iter", "in_dist_val_lsd_db", "splits", "selectivity_matrix",
            "selectivity_index", "controls", "heldout_combos", "unseen_alpha")
missing = [k for k in need_top if k not in s]
splits = ("i_unseen_geom_seen_combo", "ii_seen_geom_heldout_combo",
          "iii_unseen_geom_heldout_combo", "iv_unseen_alpha")
missing += [k for k in splits if k not in s["splits"]]
for k in splits:
    v = s["splits"].get(k, {})
    missing += ["{}.{}".format(k, f) for f in ("n_configs", "fidelity", "edit", "by_family")
                if f not in v]
    for f in ("mag_corr", "band_lsd_db", "phase_corr_mw", "rir_pearson", "t20_rel_err"):
        if f not in v.get("fidelity", {}):
            missing.append("{}.fidelity.{}".format(k, f))
    for f in ("E_BW_hz", "edit_bw_pearson", "edit_bw_slope", "E_LVL_db", "edit_gain"):
        if f not in v.get("edit", {}):
            missing.append("{}.edit.{}".format(k, f))
    for f in ("x_axial", "y_axial", "tangential"):
        if f not in v.get("by_family", {}):
            missing.append("{}.by_family.{}".format(k, f))
missing += ["controls." + k for k in ("C1_null_model", "C2_floor_hz",
                                      "C3_conditioning_identity", "C4_wall_identity")
            if k not in s["controls"]]
if missing:
    print("SCHEMA FAIL: missing " + ", ".join(missing))
    sys.exit(1)
print("SCHEMA OK: iter={} splits={}".format(s["iter"], [s["splits"][k]["n_configs"]
                                                        for k in splits]))
PY
