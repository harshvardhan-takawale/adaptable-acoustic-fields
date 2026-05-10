#!/usr/bin/env bash
# Chunk-3 orchestrator: memory_check → multi-room training (1 job) → 6× zero-shot
# eval (parallel) → latent probing. All chained via afterok dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/slurm

MEM_JOB=$(sbatch --parsable scripts/slurm/multi_room_memory_check.sh)
echo "memory_check: ${MEM_JOB}"

TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${MEM_JOB} \
                  scripts/slurm/multi_room_train.sh)
echo "multi_room train: ${TRAIN_JOB} (deps on ${MEM_JOB})"

declare -a ZS_JOBS=()
for L in 3.25 3.75 4.25 4.75 5.25 5.75; do
    JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} \
                 scripts/slurm/zero_shot_eval.sh ${L})
    ZS_JOBS+=("${JOB}")
    echo "zero_shot L=${L}: ${JOB} (deps on ${TRAIN_JOB})"
done

ZS_DEP=$(IFS=,; echo "${ZS_JOBS[*]}")
LAT_JOB=$(sbatch --parsable --dependency=afterok:${ZS_DEP} \
                 scripts/slurm/latent_probing.sh)
echo "latent probe: ${LAT_JOB} (deps on ${ZS_DEP})"

echo
echo "watch:  watch 'squeue --me -o \"%.10i %.20j %.2t %.10M %R\"'"
echo "logs:   tail -f logs/slurm/aaf_mr_train-${TRAIN_JOB}.out"
