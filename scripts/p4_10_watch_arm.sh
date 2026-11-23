#!/bin/bash
# P4-10: watch one arm's checkpoints (+5K ... +30K every 5K: 125K 130K 135K 140K 145K 150K, so no gap exceeds ~2 h) and launch a scored read at each: the
# standard report (HF / UF) and the scorecard-lite (full at +30K). bash scripts/p4_10_watch_arm.sh ARM FAMILIES_LITE FAMILIES_FULL
#   setsid nohup bash scripts/p4_10_watch_arm.sh R1 Source "StretchU+StretchA+Wall+Notch+Source+Material" > logs/p4_10_watch_R1.log 2>&1 &
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
ARM=$1; LF=$2; FF=$3
for ITER in 125000 130000 135000 140000 145000 150000; do
  K=$((ITER / 1000))K; NAME=${ARM}_$K; CK=outputs/p4_10/arms/$ARM/$(printf 'ckpt_iter%07d.pt' "$ITER")
  until [ -f "$CK" ]; do
    squeue -h -n aaf_${ARM,,} 2>/dev/null | grep -q . || { echo "$(date '+%H:%M') FAILED $ARM: no training job and no $CK"; sleep 600; }
    sleep 60
  done
  a=$(stat -c %s "$CK"); sleep 60; until [ "$a" = "$(stat -c %s "$CK")" ]; do a=$(stat -c %s "$CK"); sleep 30; done
  echo "$(date '+%H:%M') $NAME checkpoint ready"
  if [ "$ITER" = 150000 ]; then bash scripts/p4_10_read.sh $NAME $CK "$FF" full 16; else bash scripts/p4_10_read.sh $NAME $CK "$LF" lite 12; fi
  setsid nohup bash scripts/p4_9_watch_report.sh $ARM $ITER aaf_${ARM,,} outputs/p4_10/arms c > logs/p4_10_watch_report_${ARM}_$ITER.log 2>&1 < /dev/null &
done
