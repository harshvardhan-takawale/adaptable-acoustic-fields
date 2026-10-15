#!/bin/bash
# P4-5: watch ONE arm -- launch its evaluation when its checkpoint for ITER exists, then exit as
# soon as the evaluation has written its JSONs, so each arm reports the moment it lands. Exits 2
# if the arm's training job dies first.
#
# Two jobs per checkpoint: `p4_5_eval_ckpt.sh` (3-shape dense curve, frozen 800-rx Gate 2,
# lattice split, energy, k-space) and -- since D105 / Rule 11 -- the dense 15-shape Gate-2
# re-score split on / off the lattice (`p4_5_gate2_dense.py`), which the headline off-lattice
# number and its bootstrap CI come from. COMPLETE needs both.
# Usage: bash scripts/p4_5_eval_one.sh B1 30000 7816487
set -u
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
A=$1; ITER=$2; JOB=$3; K=$((ITER / 1000))K
ARMS=${ARMS:-outputs/p4_5/arms}          # P4-6: arms under outputs/p4_6/arms
CK=$(printf "ckpt_iter%07d.pt" "$ITER"); MARK=outputs/p4_5/eval/.launched; mkdir -p "$MARK"
G2=outputs/p4_5/gate2_dense/${A}_${K}__track_p4_5_testdense.json
while true; do
  if [ -f "$ARMS/$A/$CK" ] && [ ! -f "$MARK/${A}_${K}" ]; then
    E=$(sbatch --parsable --export=ALL,RUN=$ARMS/$A,CK=$CK,NAME=${A}_${K} \
        scripts/slurm/p4_5_eval_ckpt.sh) && touch "$MARK/${A}_${K}" && echo "$(date +%H:%M) launched eval ${A}_${K} -> $E"
  fi
  if [ -f "$ARMS/$A/$CK" ] && [ ! -f "$MARK/${A}_${K}_dense" ]; then
    E=$(sbatch --parsable --partition=scavenger --account=scavenger --qos=scavenger --requeue \
        --gres=gpu:rtxa5000:1 --exclude=legacy43 --job-name=aaf_g2dense --time=03:00:00 \
        --export=ALL,CMD="scripts/p4_5_gate2_dense.py --checkpoint $ARMS/$A/$CK --name ${A}_${K}" \
        scripts/slurm/p4_5_gpu.sh) && touch "$MARK/${A}_${K}_dense" && echo "$(date +%H:%M) launched dense Gate-2 ${A}_${K} -> $E"
  fi
  if [ -f "outputs/p4_5/diagnostics/energy_${A}_${K}.json" ] && [ -f "$G2" ]; then
    echo "$(date +%H:%M) ${A}_${K} COMPLETE"; exit 0
  fi
  if [ ! -f "$ARMS/$A/$CK" ] && ! squeue -j "$JOB" -h 2>/dev/null | grep -q .; then
    # grace period: a job that has just FINISHED may leave the queue a moment before its final
    # checkpoint is visible on the shared filesystem (B1b @60K, 2026-10-14) -- re-check first
    sleep 120
    if [ ! -f "$ARMS/$A/$CK" ]; then
      echo "$(date +%H:%M) ARM $A: training job $JOB gone before $CK"; exit 2
    fi
  fi
  sleep 90
done
