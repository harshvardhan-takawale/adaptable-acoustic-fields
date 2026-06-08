#!/bin/bash
# Spec A — Run C zero-shot probe. ONE scavenger GPU; must NOT touch P3's tron
# qos=high slots. (1) Simulate the 2 new interpolative interior rooms if absent
# (~23s each), then (2) run zero-shot adaptation against the converged Run C
# (10-room, 0.98 dB) checkpoint on 3 interior rooms: box center (pre-simulated)
# + the 2 new ones. Self-diagnosis (geometry head on z*) + signal-level suite
# come for free from zero_shot_3d.py.
#SBATCH --job-name=aaf_runC_zsprobe
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
# Run C renders at n_pts=32 → the zero-shot render needs ~12 GB; an 11 GB legacy
# scavenger card OOMs. Name a 24 GB rtxa5000 (still scavenger qos — preemptible,
# does NOT touch P3's tron qos=high A6000 reservation).
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

RUNC=outputs/diag_p2_2_5/C_10rm_b64
PROBE_YAML=configs/sweeps_3d/runC_probe_rooms.yaml
OUT=outputs/runC_zeroshot_probe
mkdir -p "${OUT}"
echo "host=$(hostname); job=${SLURM_JOB_ID}"; nvidia-smi -L || true

# --- (1) simulate the 2 new interior rooms (idempotent via .done sentinel) ---
for i in 0 1; do
    python scripts/build_3d_dataset.py --rooms-yaml "${PROBE_YAML}" --idx "${i}"
done

# --- (2) zero-shot on Run C for the 3 interior rooms ---
#   box center (existing maximin test room) + the 2 new interpolative rooms.
ROOMS=("4.50 4.00 3.25" "3.65 4.04 3.24" "4.36 4.14 3.96")
for r in "${ROOMS[@]}"; do
    read -r L W H <<< "$r"
    H5="data/track_a_3d/L${L}_W${W}_H${H}.h5"
    RDIR="${OUT}/L${L}_W${W}_H${H}"
    mkdir -p "${RDIR}"
    echo "=== zero-shot Run C -> L${L}_W${W}_H${H} ==="
    python -m aaf.eval.zero_shot_3d \
        --target-h5 "${H5}" \
        --train-output-dir "${RUNC}" \
        --output_dir "${RDIR}"
done
echo "# probe done; metrics under ${OUT}/L*/metrics.json"
