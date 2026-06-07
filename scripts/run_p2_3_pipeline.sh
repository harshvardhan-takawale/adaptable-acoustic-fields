#!/bin/bash
# P2-3 pipeline (post-sanity-gate). Submits the dependency chain:
#   train (4-GPU, 60K) → continuation (afterany; auto-resumes past 24h cap)
#     → 8 zero-shot evals (afterok continuation, parallel on scavenger)
#       → closeout (afterok all zero-shot: SUMMARY.md 3-way verdict + meeting plots)
#
# The DDP sanity gate (scripts/slurm/p3_sanity_4gpu.sh) is launched + INSPECTED
# separately first — afterok only checks exit code, not the trajectory, so the
# gate is a human decision. Run THIS only after the gate passes.
#
# Usage: bash scripts/run_p2_3_pipeline.sh
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields

RUN_ID=P3_45rooms_4gpu
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf 2>/dev/null || true

echo "== submit main training =="
JID_TRAIN=$(sbatch --parsable scripts/slurm/p3_train_4gpu.sh)
echo "  train       = ${JID_TRAIN}"

echo "== submit continuation (afterany: auto-resumes to 60K past the 24h cap) =="
JID_CONT=$(sbatch --parsable --dependency=afterany:${JID_TRAIN} scripts/slurm/p3_train_4gpu.sh)
echo "  continuation= ${JID_CONT}"

echo "== submit 8 zero-shot evals (afterok continuation) =="
# Emit the 8 test rooms as %.2f L W H (matches the H5 filenames).
ROOMS=$(python - <<'PY'
import yaml
for r in yaml.safe_load(open('configs/sweeps_3d/test_rooms.yaml'))['rooms']:
    print(f"{r['L']:.2f} {r['W']:.2f} {r['H']:.2f}")
PY
)
ZS_JIDS=()
while read -r L W H; do
    [ -z "$L" ] && continue
    JID=$(sbatch --parsable --dependency=afterok:${JID_CONT} \
        scripts/slurm/zero_shot_3d_eval.sh "${RUN_ID}" "${L}" "${W}" "${H}")
    ZS_JIDS+=("${JID}")
    echo "  zs L${L}_W${W}_H${H} = ${JID}"
done <<< "${ROOMS}"

ZS_DEP=$(IFS=:; echo "${ZS_JIDS[*]}")
echo "== submit closeout (afterok all zero-shot) =="
JID_CLOSE=$(sbatch --parsable --dependency=afterok:${ZS_DEP} scripts/slurm/p3_closeout.sh "${RUN_ID}")
echo "  closeout    = ${JID_CLOSE}"

echo
echo "chain: ${JID_TRAIN} -> ${JID_CONT} -> [${ZS_JIDS[*]}] -> ${JID_CLOSE}"
echo "watch: squeue -u \$USER ; results: outputs/multi_room_3d/${RUN_ID}/SUMMARY.md"
