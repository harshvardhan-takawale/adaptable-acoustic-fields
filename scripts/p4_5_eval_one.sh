#!/bin/bash
# P4-5: watch ONE arm -- launch its evaluation when its checkpoint for ITER exists, then exit as
# soon as the evaluation (curve + lattice split + energy diagnostic) has written its JSONs, so
# each arm reports the moment it lands. Exits 2 if the arm's training job dies first.
# Usage: bash scripts/p4_5_eval_one.sh B1 30000 7816487
set -u
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
A=$1; ITER=$2; JOB=$3; K=$((ITER / 1000))K
CK=$(printf "ckpt_iter%07d.pt" "$ITER"); MARK=outputs/p4_5/eval/.launched; mkdir -p "$MARK"
while true; do
  if [ -f "outputs/p4_5/arms/$A/$CK" ] && [ ! -f "$MARK/${A}_${K}" ]; then
    E=$(sbatch --parsable --export=ALL,RUN=outputs/p4_5/arms/$A,CK=$CK,NAME=${A}_${K} \
        scripts/slurm/p4_5_eval_ckpt.sh) && touch "$MARK/${A}_${K}" && echo "$(date +%H:%M) launched eval ${A}_${K} -> $E"
  fi
  if [ -f "outputs/p4_5/diagnostics/energy_${A}_${K}.json" ]; then
    echo "$(date +%H:%M) ${A}_${K} COMPLETE"; exit 0
  fi
  if [ ! -f "outputs/p4_5/arms/$A/$CK" ] && ! squeue -j "$JOB" -h 2>/dev/null | grep -q .; then
    echo "$(date +%H:%M) ARM $A: training job $JOB gone before $CK"; exit 2
  fi
  sleep 90
done
