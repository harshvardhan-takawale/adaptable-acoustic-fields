#!/usr/bin/env bash
# Chunk-2 orchestrator: memory_check → 3× train (parallel) → 3× eval (parallel).
# Each downstream step depends on the previous via afterok.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/slurm

MEM_JOB=$(sbatch --parsable scripts/slurm/memory_check.sh)
echo "memory_check: ${MEM_JOB}"

declare -A TRAIN
for L in 3.0 4.5 6.0; do
    JOB=$(sbatch --parsable --dependency=afterok:${MEM_JOB} \
                 scripts/slurm/single_room_train.sh ${L})
    TRAIN[${L}]=${JOB}
    echo "train L=${L}: ${JOB} (deps on ${MEM_JOB})"
done

for L in 3.0 4.5 6.0; do
    JOB=$(sbatch --parsable --dependency=afterok:${TRAIN[${L}]} \
                 scripts/slurm/single_room_eval.sh ${L})
    echo "eval  L=${L}: ${JOB} (deps on ${TRAIN[${L}]})"
done

echo
echo "watch:  watch 'squeue --me -o \"%.10i %.20j %.2t %.10M %R\"'"
echo "logs:   tail -f logs/slurm/aaf_train-${TRAIN[3.0]}.out  # etc."
