#!/usr/bin/env bash
# Chunk 3.6 orchestrator — band-limited eval + 6 inner-loop variants + FiLM/jitter retrains.
#
# Stages:
#   0  pytest gate                           ( 1 job   )
#   1  Track A band-limited recompute        ( 1 job   )
#   2  Track B 6 variants × 6 unseen L       (36 jobs  )
#   3  Track B summary                       ( 1 job   )
#   4  Track C training (C1, C2 on tron)     ( 2 jobs  )
#   5a C1/C2 B1-baseline ZS                  (12 jobs  )
#   5b C1/C2 Track-B-winner ZS               (12 jobs  )
#   6  C1/C2 latent probes                   ( 2 jobs  )
#   7  Final summary                         ( 1 job   )
# Total: ~68 jobs.
#
# Track A is purposefully chained behind pytest only so its results land first
# (~15 min after pytest passes). Tracks B and C run in parallel.

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
mkdir -p logs/slurm

VARIANTS=(B1 B2 B3 B4 B5 B6)
LS=(3.25 3.75 4.25 4.75 5.25 5.75)
TRACK_C_RUNS=(C1_film C2_latent_jitter)

# Stage 0: pytest gate.
PY_JOB=$(sbatch --parsable scripts/slurm/run_pytest.sh)
echo "stage 0 (pytest)        : $PY_JOB"

# Stage 1: Track A band-limited recompute (depends on pytest).
TRACK_A_JOB=$(sbatch --parsable --dependency=afterok:${PY_JOB} \
    scripts/slurm/track_a_band_limited.sh)
echo "stage 1 (Track A)       : $TRACK_A_JOB"

# Stage 2: Track B 36 jobs.
B_JOBS=()
for V in "${VARIANTS[@]}"; do
    for L in "${LS[@]}"; do
        J=$(sbatch --parsable --dependency=afterok:${PY_JOB} \
            scripts/slurm/zero_shot_variant.sh "$V" "$L" R6_tiny_lhead)
        B_JOBS+=("$J")
    done
done
B_DEPS=$(IFS=,; echo "${B_JOBS[*]}")
echo "stage 2 (Track B 36)    : ${B_JOBS[0]}..${B_JOBS[-1]} ($(echo $B_DEPS | tr ',' '\n' | wc -l) jobs)"

# Stage 3: Track B summary.
TRACK_B_SUM_JOB=$(sbatch --parsable --dependency=afterok:${B_DEPS} \
    scripts/slurm/track_b_summary.sh)
echo "stage 3 (Track B sum)   : $TRACK_B_SUM_JOB"

# Stage 4: Track C training on tron (parallel).
C_TRAIN_JOBS=()
for RUN in "${TRACK_C_RUNS[@]}"; do
    YAML="configs/sweep/${RUN}.yaml"
    J=$(sbatch --parsable --dependency=afterok:${PY_JOB} \
        --partition=tron --account=nexus --qos=default \
        scripts/slurm/sweep_train.sh "$YAML")
    C_TRAIN_JOBS+=("$J")
    echo "stage 4 (train $RUN)    : $J"
done
C1_TRAIN_JOB=${C_TRAIN_JOBS[0]}
C2_TRAIN_JOB=${C_TRAIN_JOBS[1]}

# Stage 5a: C1/C2 B1-baseline ZS (depends on per-run train only).
C_B1_JOBS=()
for IDX in 0 1; do
    RUN=${TRACK_C_RUNS[$IDX]}
    DEP=${C_TRAIN_JOBS[$IDX]}
    for L in "${LS[@]}"; do
        J=$(sbatch --parsable --dependency=afterok:${DEP} \
            scripts/slurm/zero_shot_variant.sh B1 "$L" "$RUN")
        C_B1_JOBS+=("$J")
    done
done
echo "stage 5a (C1/C2 B1 ZS)  : ${C_B1_JOBS[0]}..${C_B1_JOBS[-1]} (12 jobs)"

# Stage 5b: C1/C2 Track-B-winner ZS (depends on per-run train AND Track B summary).
C_BEST_JOBS=()
for IDX in 0 1; do
    RUN=${TRACK_C_RUNS[$IDX]}
    DEP=${C_TRAIN_JOBS[$IDX]}
    for L in "${LS[@]}"; do
        J=$(sbatch --parsable --dependency=afterok:${DEP}:${TRACK_B_SUM_JOB} \
            scripts/slurm/zero_shot_with_best_variant.sh "$RUN" "$L")
        C_BEST_JOBS+=("$J")
    done
done
echo "stage 5b (C1/C2 best)   : ${C_BEST_JOBS[0]}..${C_BEST_JOBS[-1]} (12 jobs)"

# Stage 6: C1/C2 latent probes.
C_PROBE_JOBS=()
for IDX in 0 1; do
    RUN=${TRACK_C_RUNS[$IDX]}
    DEP=${C_TRAIN_JOBS[$IDX]}
    J=$(sbatch --parsable --dependency=afterok:${DEP} \
        scripts/slurm/latent_probing.sh "outputs/multi_room/sweep/${RUN}")
    C_PROBE_JOBS+=("$J")
done
echo "stage 6 (probes)        : ${C_PROBE_JOBS[*]}"

# Stage 7: Final summary (depends on EVERYTHING above).
ALL_DEPS_LIST=( "$TRACK_A_JOB" "$TRACK_B_SUM_JOB" "${C_B1_JOBS[@]}" "${C_BEST_JOBS[@]}" "${C_PROBE_JOBS[@]}" )
ALL_DEPS=$(IFS=,; echo "${ALL_DEPS_LIST[*]}")
FINAL_JOB=$(sbatch --parsable --dependency=afterok:${ALL_DEPS} \
    scripts/slurm/chunk_3_6_final_summary.sh)
echo "stage 7 (final)         : $FINAL_JOB"

echo ""
echo "All jobs queued. Tail with:"
echo "    squeue -u \$USER -t PD,R --noheader -o '%.10i %.10P %.20j %.10T %.10M %R'"
echo "Track A summary will land at:"
echo "    outputs/multi_room/sweep/band_limited_summary.md"
