#!/usr/bin/env bash
# Chunk-3.5 orchestrator: smoke → 6 trains → 36 zero-shots → 6 latent probes → 1 summary.
# R0 trains on tron (non-preemptible); R1-R5 default to scavenger.
# All chained via afterok dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/slurm

# Stage 1: smoke check
SMOKE_JOB=$(sbatch --parsable scripts/slurm/sweep_smoke_check.sh)
echo "smoke check: ${SMOKE_JOB}"

# Stage 2: 6 trainings, depending on smoke. Submission order = priority order
# (R0 → R3 → R1 → R2/R4/R5) so scavenger FIFO favours the most-important runs.
declare -A TRAIN_JOBS
ORDER=(R0_central R3_no_lhead R1_smaller_hash R2_larger_latent R4_strong_lhead R5_strong_l2)
for RUN in "${ORDER[@]}"; do
    if [ "${RUN}" = "R0_central" ]; then
        # R0 → tron (non-preemptible).
        JOB=$(sbatch --parsable --dependency=afterok:${SMOKE_JOB} \
                     --partition=tron --account=nexus --qos=default \
                     scripts/slurm/sweep_train.sh configs/sweep/${RUN}.yaml)
    else
        # R1-R5 → scavenger (default in sweep_train.sh).
        JOB=$(sbatch --parsable --dependency=afterok:${SMOKE_JOB} \
                     scripts/slurm/sweep_train.sh configs/sweep/${RUN}.yaml)
    fi
    TRAIN_JOBS[${RUN}]=${JOB}
    echo "train ${RUN}: ${JOB} (deps on ${SMOKE_JOB})"
done

# Stage 3: 36 zero-shot eval jobs (6 per run × 6 runs), each dependent on its training.
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

# Stage 4: 6 latent probes, each dependent on its 6 zero-shot evals.
declare -A PROBE_JOBS
for RUN in "${ORDER[@]}"; do
    JOB=$(sbatch --parsable --dependency=afterok:${ZS_DEPS[${RUN}]} \
                 scripts/slurm/latent_probing.sh outputs/multi_room/sweep/${RUN})
    PROBE_JOBS[${RUN}]=${JOB}
    echo "probe ${RUN}: ${JOB} (deps on ${ZS_DEPS[${RUN}]})"
done

# Stage 5: sweep summary, dependent on all probes.
PROBE_DEP=$(IFS=,; echo "${PROBE_JOBS[*]}")
SUMMARY_JOB=$(sbatch --parsable --dependency=afterok:${PROBE_DEP} \
                     scripts/slurm/sweep_summary.sh)
echo "summary: ${SUMMARY_JOB} (deps on ${PROBE_DEP})"

echo
echo "watch:  watch 'squeue --me -o \"%.10i %.20j %.2t %.10M %R\"'"
echo "logs:   tail -f logs/slurm/aaf_sweep_smoke-${SMOKE_JOB}.out"
