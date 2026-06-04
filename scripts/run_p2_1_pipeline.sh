#!/bin/bash
# Chunk P2-1 orchestrator. Submits the pipeline as a DAG of SLURM jobs:
#
#   pytest_gate
#     → sample_rooms_3d            (writes configs/sweeps_3d/*.yaml)
#       → budget_check_3d          (1 job; HARD-blocks the rest)
#         → build_3d_derisk array  (5 tasks on tron, 4-concurrent cap)
#           → build_3d_manifest    (refreshes data/track_a_3d/manifest.json)
#             → memory_check_3d    (chooses n_pts/batch cascade per D12)
#               → 5× single_room_3d_train (parallel on tron)
#                 → 5× single_room_3d_eval (sequential or scavenger parallel)
#                   → single_room_3d_summary
#         → build_3d_train array   (45 tasks on scavenger, no cap;
#                                   runs in parallel with single-room train)
#           → build_3d_manifest    (idempotent refresh)
#
# All deps via --dependency=afterok:JOBID. Each task is idempotent (sentinels)
# so partial completions auto-resume on re-submit.
#
# Per CLUSTER_INFO.md + DECISIONS.md:
#   - de-risk dataset on tron (qos=default, 4-concurrent cap)
#   - training dataset on scavenger (wide array, preempt-safe)
#   - single-room training on tron RTX 2080 Ti (D13: 24 GB needed)
#   - single-room eval on scavenger (forward-only fits 12 GB)

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
mkdir -p logs/slurm

DERISK_ROOMS=(
    "4.50 4.00 3.25"
    "3.00 3.00 2.50"
    "6.00 5.00 4.00"
    "3.00 5.00 2.50"
    "6.00 3.00 4.00"
)

echo "# Submitting P2-1 pipeline at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Stage 0: pytest gate.
PY=$(sbatch --parsable scripts/slurm/run_pytest.sh)
echo "stage 0 (pytest)            : $PY"

# Stage 1: write configs/sweeps_3d/*.yaml.
SAMPLE=$(sbatch --parsable --dependency=afterok:$PY scripts/slurm/sample_rooms_3d.sh)
echo "stage 1 (sample rooms 3d)   : $SAMPLE"

# Stage 2: budget check. HARD-blocks the rest. If this fails, the dependent
# jobs sit in 'DependencyNeverSatisfied' and you can inspect
# outputs/budget_check_3d/REPORT.md + OPEN_QUESTIONS.md before re-submitting.
BUDGET=$(sbatch --parsable --dependency=afterok:$SAMPLE scripts/slurm/budget_check_3d.sh)
echo "stage 2 (budget check)      : $BUDGET"

# Stage 3a: de-risk dataset (5 rooms, tron, 4-concurrent cap).
DERISK_ARRAY=$(sbatch --parsable --dependency=afterok:$BUDGET \
    --array=0-4%4 scripts/slurm/build_3d_derisk.sh)
echo "stage 3a (build derisk x5)  : $DERISK_ARRAY"

# Stage 3b: training dataset (45 rooms, scavenger, no concurrency cap).
# Independent of stage 3a — runs in parallel.
TRAIN_ARRAY=$(sbatch --parsable --dependency=afterok:$BUDGET \
    --array=0-44 scripts/slurm/build_3d_train.sh)
echo "stage 3b (build train x45)  : $TRAIN_ARRAY"

# Stage 4a: refresh manifest after de-risk completes.
MANIFEST_DERISK=$(sbatch --parsable --dependency=afterok:$DERISK_ARRAY \
    scripts/slurm/build_3d_manifest.sh)
echo "stage 4a (manifest derisk)  : $MANIFEST_DERISK"

# Stage 4b: refresh manifest after train completes (separate job).
MANIFEST_TRAIN=$(sbatch --parsable --dependency=afterok:$TRAIN_ARRAY \
    scripts/slurm/build_3d_manifest.sh)
echo "stage 4b (manifest train)   : $MANIFEST_TRAIN"

# Stage 5: GPU memory check (after derisk so we have at least one room to
# load if we want to extend the smoke check, but actually it's standalone).
MEMCHECK=$(sbatch --parsable --dependency=afterok:$MANIFEST_DERISK \
    scripts/slurm/memory_check_3d.sh)
echo "stage 5 (memory check 3d)   : $MEMCHECK"

# Stage 6: single-room 3D training (5 parallel jobs on tron).
TRAIN_JOBS=()
for ROOM in "${DERISK_ROOMS[@]}"; do
    read -r L W H <<< "$ROOM"
    JID=$(sbatch --parsable --dependency=afterok:$MEMCHECK \
        scripts/slurm/single_room_3d_train.sh "$L" "$W" "$H")
    TRAIN_JOBS+=("$JID")
    echo "stage 6 (train L=$L W=$W H=$H)  : $JID"
done

# Stage 7: single-room eval — chained per-room after the matching train.
EVAL_JOBS=()
for i in "${!DERISK_ROOMS[@]}"; do
    read -r L W H <<< "${DERISK_ROOMS[$i]}"
    TRAIN_JID="${TRAIN_JOBS[$i]}"
    JID=$(sbatch --parsable --dependency=afterok:$TRAIN_JID \
        scripts/slurm/single_room_3d_eval.sh "$L" "$W" "$H")
    EVAL_JOBS+=("$JID")
    echo "stage 7 (eval L=$L W=$W H=$H)   : $JID"
done

# Stage 8: final summary (after all evals).
EVAL_DEPS=$(IFS=:; echo "${EVAL_JOBS[*]}")
SUMMARY=$(sbatch --parsable --dependency=afterok:$EVAL_DEPS \
    scripts/slurm/single_room_3d_summary.sh)
echo "stage 8 (summary)           : $SUMMARY"

echo
echo "# All jobs submitted. Monitor:"
echo "#   squeue -u \$USER"
echo "#   tail -f logs/slurm/aaf_*.out"
echo "# Final summary lands at outputs/single_room_3d/SUMMARY.md"
