#!/bin/bash
# P2-4 orchestrator: sequential density trainings (90->150->250) + per-density
# known-geometry eval on the FROZEN test set + scaling-curve build.
# Prereq: dataset arrays already submitted (their JIDs in /tmp/p2_4_dataset_jids.txt,
# or pass as: run_p2_4_pipeline.sh <train250_arr_jid> <test_arr_jid>).
# density-45 = reuse P3 (no retrain). Each density = main + 2 afterany resume jobs.
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
if [ $# -ge 2 ]; then TRAIN_ARR=$1; TEST_ARR=$2; else read TRAIN_ARR TEST_ARR < /tmp/p2_4_dataset_jids.txt; fi
echo "dataset deps: train250=${TRAIN_ARR}  test=${TEST_ARR}"
S() { sbatch --parsable "$@"; }
TR=scripts/slurm/density_train_4gpu.sh
EV=scripts/slurm/density_eval.sh

# density-45 eval (P3 already trained) — waits only on the test rooms being simulated.
E45=$(S --dependency=afterok:${TEST_ARR} ${EV} outputs/multi_room_3d/P3_45rooms_4gpu density_45)
echo "eval density_45 = ${E45}"

# Train one density (main + 2 afterany continuations); prints jids; returns last via file.
train_chain() {  # $1=config $2=outdir $3=afterok-dep
  local cfg=$1 out=$2 dep=$3
  local m c1 c2
  m=$(S --dependency=afterok:${dep} ${TR} ${cfg} ${out})
  c1=$(S --dependency=afterany:${m} ${TR} ${cfg} ${out})
  c2=$(S --dependency=afterany:${c1} ${TR} ${cfg} ${out})
  echo "  train ${out}: main=${m} c1=${c1} c2=${c2}" >&2
  echo "${c2}"
}

D90=$(train_chain configs/sweep_3d/density_90.yaml  outputs/multi_room_3d/density_90  ${TRAIN_ARR})
E90=$(S --dependency=afterok:${D90}:${TEST_ARR} ${EV} outputs/multi_room_3d/density_90 density_90); echo "eval density_90 = ${E90}"
D150=$(train_chain configs/sweep_3d/density_150.yaml outputs/multi_room_3d/density_150 ${D90})
E150=$(S --dependency=afterok:${D150}:${TEST_ARR} ${EV} outputs/multi_room_3d/density_150 density_150); echo "eval density_150 = ${E150}"
D250=$(train_chain configs/sweep_3d/density_250.yaml outputs/multi_room_3d/density_250 ${D150})
E250=$(S --dependency=afterok:${D250}:${TEST_ARR} ${EV} outputs/multi_room_3d/density_250 density_250); echo "eval density_250 = ${E250}"

# Scaling curve, after all four evals.
SCALE=$(S --dependency=afterok:${E45}:${E90}:${E150}:${E250} scripts/slurm/p2_4_scaling.sh)
echo "scaling build = ${SCALE}"
echo "chain: [dataset ${TRAIN_ARR}/${TEST_ARR}] -> 90 -> 150 -> 250 (sequential) ; evals -> scaling ${SCALE}"
