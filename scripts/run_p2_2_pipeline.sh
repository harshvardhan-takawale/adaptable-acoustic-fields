#!/bin/bash
# Chunk P2-2 orchestrator: multi-room 3D conditioning + zero-shot adaptation.
#
# DAG:
#   pytest_gate
#     ├─ build_3d_test array (8 rooms; parallel with pytest)
#     │     ↓ afterok
#     │   manifest_refresh
#     │     ↓ afterok
#     │   memory_check_multi_3d
#     │     ↓ afterok
#     │   ┌─ M1 train (d=16)  ─────────┬─ M2 train (d=32)
#     │   │     ↓ afterok              │     ↓ afterok
#     │   │  8× zs_eval (parallel)     │  8× zs_eval (parallel)
#     │   │     ↓ afterok (all)        │     ↓ afterok (all)
#     │   └─ probe + summary           └─ probe + summary

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
mkdir -p logs/slurm

echo "# Submitting P2-2 pipeline at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Stage 0: pytest gate.
PY=$(sbatch --parsable scripts/slurm/run_pytest.sh)
echo "stage 0 (pytest)            : $PY"

# Stage 1: build 8 test rooms (parallel with pytest; idempotent).
TEST_ARRAY=$(sbatch --parsable --array=0-7 scripts/slurm/build_3d_test.sh)
echo "stage 1 (build test x8)     : $TEST_ARRAY"

# Stage 2: refresh manifest after test rooms are on disk.
MANIFEST=$(sbatch --parsable --dependency=afterok:$TEST_ARRAY \
    scripts/slurm/build_3d_manifest.sh)
echo "stage 2 (manifest refresh)  : $MANIFEST"

# Stage 3: memory check for the auto-decoder.
MEMCHECK=$(sbatch --parsable --dependency=afterok:$PY:$MANIFEST \
    scripts/slurm/memory_check_multi_3d.sh)
echo "stage 3 (memory check)      : $MEMCHECK"

# ----------------------------- M1 path (d=16) -----------------------------
M1_TRAIN=$(sbatch --parsable --dependency=afterok:$MEMCHECK \
    scripts/slurm/multi_room_3d_train.sh configs/sweep_3d/M1_45rooms.yaml)
echo "M1 train (d=16)             : $M1_TRAIN"

TEST_ROOMS=(
    "$(python -c "import yaml; rs=yaml.safe_load(open('configs/sweeps_3d/test_rooms.yaml'))['rooms']
for r in rs: print(f\"{r['L']:.2f} {r['W']:.2f} {r['H']:.2f}\")")"
)
# Re-parse so each line is a separate element.
mapfile -t TEST_ROOMS < <(python -c "
import yaml
rs = yaml.safe_load(open('configs/sweeps_3d/test_rooms.yaml'))['rooms']
for r in rs:
    print(f\"{r['L']:.2f} {r['W']:.2f} {r['H']:.2f}\")
")

M1_ZS_JOBS=()
for ROOM in "${TEST_ROOMS[@]}"; do
    read -r L W H <<< "$ROOM"
    JID=$(sbatch --parsable --dependency=afterok:$M1_TRAIN \
        scripts/slurm/zero_shot_3d_eval.sh M1_45rooms "$L" "$W" "$H")
    M1_ZS_JOBS+=("$JID")
    echo "M1 zs L=$L W=$W H=$H          : $JID"
done
M1_ZS_DEPS=$(IFS=:; echo "${M1_ZS_JOBS[*]}")
M1_PROBE=$(sbatch --parsable --dependency=afterok:$M1_ZS_DEPS \
    scripts/slurm/latent_probe_3d.sh M1_45rooms)
M1_SUMMARY=$(sbatch --parsable --dependency=afterok:$M1_ZS_DEPS \
    scripts/slurm/multi_room_3d_summary.sh M1_45rooms)
echo "M1 probe                    : $M1_PROBE"
echo "M1 summary                  : $M1_SUMMARY"

# ----------------------------- M2 path (d=32) -----------------------------
M2_TRAIN=$(sbatch --parsable --dependency=afterok:$MEMCHECK \
    scripts/slurm/multi_room_3d_train.sh configs/sweep_3d/M2_45rooms_d32.yaml)
echo "M2 train (d=32)             : $M2_TRAIN"

M2_ZS_JOBS=()
for ROOM in "${TEST_ROOMS[@]}"; do
    read -r L W H <<< "$ROOM"
    JID=$(sbatch --parsable --dependency=afterok:$M2_TRAIN \
        scripts/slurm/zero_shot_3d_eval.sh M2_45rooms_d32 "$L" "$W" "$H")
    M2_ZS_JOBS+=("$JID")
    echo "M2 zs L=$L W=$W H=$H          : $JID"
done
M2_ZS_DEPS=$(IFS=:; echo "${M2_ZS_JOBS[*]}")
M2_PROBE=$(sbatch --parsable --dependency=afterok:$M2_ZS_DEPS \
    scripts/slurm/latent_probe_3d.sh M2_45rooms_d32)
M2_SUMMARY=$(sbatch --parsable --dependency=afterok:$M2_ZS_DEPS \
    scripts/slurm/multi_room_3d_summary.sh M2_45rooms_d32)
echo "M2 probe                    : $M2_PROBE"
echo "M2 summary                  : $M2_SUMMARY"

echo
echo "# All jobs submitted. Final summaries land at:"
echo "#   outputs/multi_room_3d/M1_45rooms/SUMMARY.md"
echo "#   outputs/multi_room_3d/M2_45rooms_d32/SUMMARY.md"
