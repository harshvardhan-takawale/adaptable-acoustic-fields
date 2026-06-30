#!/bin/bash
# P2-4 SECONDARY column: few-shot 8-measurement zero-shot route on the FROZEN
# interior test set (15 rooms) for one density model. "For completeness" — the
# headline is the known-geometry route (scripts/density_eval.sh). Idempotent:
# skips rooms whose band_limited_metrics.json already exists.
#
#   bash scripts/run_fewshot_frozen.sh <RUN_ID>
# e.g. P3_45rooms_4gpu (density-45) | density_90 | density_150 | density_250
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
RUN_ID="${1:?usage: $0 <RUN_ID>}"
TRAIN_OUT="outputs/multi_room_3d/${RUN_ID}"
[ -d "${TRAIN_OUT}" ] || { echo "ERROR: ${TRAIN_OUT} does not exist (train not done?)"; exit 1; }

# Read the 15 frozen test rooms as 'L W H' lines (2-decimal, matching h5 names).
mapfile -t ROWS < <(python3 -c "
import json
for r in json.loads(open('outputs/coverage_curve/test_nn_distances.json').read())['test_rooms']:
    print(f\"{r['L']:.2f} {r['W']:.2f} {r['H']:.2f}\")
")

sub=0; skip=0
for row in "${ROWS[@]}"; do
    read -r L W H <<< "${row}"
    out="${TRAIN_OUT}/zero_shot/L${L}_W${W}_H${H}/band_limited_metrics.json"
    if [ -f "${out}" ]; then
        skip=$((skip+1)); continue
    fi
    sbatch scripts/slurm/zero_shot_3d_eval.sh "${RUN_ID}" "${L}" "${W}" "${H}" >/dev/null
    sub=$((sub+1))
done
echo "[fewshot ${RUN_ID}] submitted=${sub} skipped(done)=${skip} of 15"
