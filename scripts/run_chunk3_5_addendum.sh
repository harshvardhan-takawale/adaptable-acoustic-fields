#!/usr/bin/env bash
# Chunk-3.5+ ADDENDUM orchestrator: 3 new sweep runs (R6/R7/R8) layered on top
# of the running R0-R5 sweep. Does not interrupt R0-R5.
#
# Pipeline:
#   1. Linear-L-head smoke check on R6 (200 iters, scavenger)
#   2. R6/R7/R8 trainings (default to tron; fall back to scavenger w/ TITAN-X
#      exclusion if tron is congested)
#   3. 18 zero-shot evals (3 runs × 6 unseen L), scavenger, dependent on respective trainings
#   4. 3 latent probes, scavenger, dependent on respective ZS evals
#   5. Re-run sweep summary AFTER all 9 runs (R0-R8) have probed.
#      Hard-codes the existing chunk-3.5 R0-R5 probe IDs so the re-summary waits
#      for them too. (If those have already produced SWEEP_SUMMARY.md, the
#      re-summary just overwrites with the 9-run version.)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/slurm

# Prior R0-R5 probe job IDs (from scripts/run_chunk3_5_sweep.sh launch). Used
# as additional dependencies for the re-summary so we always include R0-R5.
PRIOR_PROBE_JOBS="6813101 6813102 6813103 6813104 6813105 6813106"

# Stage 1: smoke check (linear L-head code path)
SMOKE_JOB=$(sbatch --parsable scripts/slurm/addendum_smoke.sh)
echo "addendum smoke: ${SMOKE_JOB}"

# Stage 2: 3 trainings on tron (RTX, ~3× faster than TITAN X)
declare -A TRAIN_JOBS
ORDER=(R6_tiny_lhead R7_medium_hash R8_tiny_latent)
for RUN in "${ORDER[@]}"; do
    JOB=$(sbatch --parsable --dependency=afterok:${SMOKE_JOB} \
                 --partition=tron --account=nexus --qos=default \
                 scripts/slurm/sweep_train.sh configs/sweep/${RUN}.yaml)
    TRAIN_JOBS[${RUN}]=${JOB}
    echo "train ${RUN} (tron): ${JOB} (deps on smoke ${SMOKE_JOB})"
done

# Stage 3: 18 zero-shot evals
declare -A ZS_DEPS
for RUN in "${ORDER[@]}"; do
    DEPS=""
    for L in 3.25 3.75 4.25 4.75 5.25 5.75; do
        JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOBS[${RUN}]} \
                     scripts/slurm/zero_shot_eval.sh ${L} outputs/multi_room/sweep/${RUN})
        DEPS+="${DEPS:+,}${JOB}"
    done
    ZS_DEPS[${RUN}]=${DEPS}
    echo "zero_shot ${RUN}: 6 jobs (deps on ${TRAIN_JOBS[${RUN}]})"
done

# Stage 4: 3 latent probes
declare -A PROBE_JOBS
for RUN in "${ORDER[@]}"; do
    JOB=$(sbatch --parsable --dependency=afterok:${ZS_DEPS[${RUN}]} \
                 scripts/slurm/latent_probing.sh outputs/multi_room/sweep/${RUN})
    PROBE_JOBS[${RUN}]=${JOB}
    echo "probe ${RUN}: ${JOB} (deps on ${ZS_DEPS[${RUN}]})"
done

# Stage 5: re-run sweep summary covering all 9 runs.
# Depends on all 6 prior R0-R5 probes + the 3 new R6-R8 probes.
NEW_PROBE_DEP=$(IFS=,; echo "${PROBE_JOBS[*]}")
PRIOR_PROBE_DEP=$(echo $PRIOR_PROBE_JOBS | tr ' ' ',')
ALL_DEP="${PRIOR_PROBE_DEP},${NEW_PROBE_DEP}"
RESUMMARY_JOB=$(sbatch --parsable --dependency=afterok:${ALL_DEP} \
                       scripts/slurm/sweep_summary.sh)
echo "re-summary (R0-R8): ${RESUMMARY_JOB} (deps on ${ALL_DEP})"

echo
echo "watch:  watch 'squeue --me -o \"%.10i %.20j %.2t %.10M %R\"'"
echo "logs:   tail -f logs/slurm/aaf_addendum_smoke-${SMOKE_JOB}.out"
