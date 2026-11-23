#!/bin/bash
# P4-9 X3 (D191): when each RCP checkpoint lands (130K / 140K / 150K), score it at the second source on both sets
# (in-family 15: --set test; unseen 16) -- one GPU job per set. The standard report runs from p4_9_watch_report.sh.
#   setsid nohup bash scripts/p4_9_x3_watch.sh > logs/p4_9_x3_watch.log 2>&1 &
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
GPU="--partition=scavenger --account=scavenger --qos=scavenger --requeue --gres=gpu:rtxa5000:1 --exclude=legacy43"
for ITER in 130000 140000 150000; do
  K=$((ITER / 1000))K
  CK=outputs/p4_9/arms/RCP/ckpt_iter$(printf %07d "$ITER").pt
  until [ -f "$CK" ]; do sleep 300; done
  s1=$(stat -c %s "$CK"); sleep 60
  until [ "$s1" = "$(stat -c %s "$CK")" ]; do s1=$(stat -c %s "$CK"); sleep 60; done
  for SET in test unseen; do
    J=$(sbatch --parsable $GPU --job-name=aaf_x3_${SET}_$K --time=04:00:00 --mem=32G \
        --export=ALL,CMD="scripts/p4_9_src2_eval.py --set $SET --items RCP_${K}=$CK" scripts/slurm/p4_5_gpu.sh)
    echo "$(date) RCP_$K $SET -> $J"
  done
done
