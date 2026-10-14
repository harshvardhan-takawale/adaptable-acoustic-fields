#!/bin/bash
# P4-5: launch each arm's evaluation the moment its checkpoint for ITER exists, and exit when
# every arm's evaluation for ITER has produced its JSON -- or when an arm's training job has died
# without reaching ITER, so a failure surfaces instead of the watcher waiting forever.
# Usage: bash scripts/p4_5_auto_eval.sh 30000
set -u
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
ITER=$1; K=$((ITER / 1000))K
ARMS="B1 B1b B2 B3 B4"
MARK=outputs/p4_5/eval/.launched; mkdir -p "$MARK"
declare -A JOB=([B1]=7816487 [B4]=7816488 [B2]=7816489 [B3]=7816490 [B1b]=7816701)
while true; do
  done_n=0
  for A in $ARMS; do
    CK=$(printf "ckpt_iter%07d.pt" "$ITER")
    if [ -f "outputs/p4_5/arms/$A/$CK" ] && [ ! -f "$MARK/${A}_${K}" ]; then
      J=$(sbatch --parsable --export=ALL,RUN=outputs/p4_5/arms/$A,CK=$CK,NAME=${A}_${K} \
          scripts/slurm/p4_5_eval_ckpt.sh) && touch "$MARK/${A}_${K}" && \
          echo "$(date +%H:%M) launched eval ${A}_${K} -> job $J"
    fi
    [ -f "outputs/p4_5/eval/${A}_${K}.json" ] && done_n=$((done_n + 1))
    # an arm whose training job is gone before its checkpoint exists has FAILED
    if [ ! -f "outputs/p4_5/arms/$A/$CK" ] && ! squeue -j "${JOB[$A]}" -h 2>/dev/null | grep -q .; then
      echo "$(date +%H:%M) ARM $A: training job ${JOB[$A]} is no longer running and has no $CK"
      exit 2
    fi
  done
  if [ "$done_n" -ge 5 ]; then echo "$(date +%H:%M) all five ${K} evaluations complete"; exit 0; fi
  sleep 120
done
