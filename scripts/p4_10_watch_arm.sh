#!/bin/bash
# P4-10: watch one arm's checkpoints (+5K ... +30K every 5K: 125K 130K 135K 140K 145K 150K, so no gap exceeds ~2 h) and launch a scored read at each: the
# standard report (HF / UF) and the scorecard-lite (full at +30K). bash scripts/p4_10_watch_arm.sh ARM FAM_SMALL FAM_ALL_LITE FAMILIES_FULL
#   ARM's small (cheap) families at +5K / +15K / +25K, all lite families at +10K / +20K (the arms' paired reads against R1 at the
#   matched step), and the full card at +30K. setsid nohup bash scripts/p4_10_watch_arm.sh R1 Source ALL "StretchU+...+Material" ...
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
ARM=$1; LF=$2; LA=$3; FF=$4
ALL_LITE="StretchU+StretchA+Wall+Notch+Source+Material"; [ "$LA" = ALL ] && LA=$ALL_LITE
for ITER in 125000 130000 135000 140000 145000 150000; do
  [ "$ITER" -lt "${START:-125000}" ] && continue
  K=$((ITER / 1000))K; NAME=${ARM}_$K; CK=outputs/p4_10/arms/$ARM/$(printf 'ckpt_iter%07d.pt' "$ITER")
  until [ -f "$CK" ]; do
    squeue -h -n aaf_${ARM,,} 2>/dev/null | grep -q . || { echo "$(date '+%H:%M') FAILED $ARM: no training job and no $CK"; sleep 600; }
    sleep 60
  done
  a=$(stat -c %s "$CK"); sleep 60; until [ "$a" = "$(stat -c %s "$CK")" ]; do a=$(stat -c %s "$CK"); sleep 30; done
  echo "$(date '+%H:%M') $NAME checkpoint ready"
  if [ "$ITER" = 150000 ]; then bash scripts/p4_10_read.sh $NAME $CK "$FF" full 16
  elif [ "$ITER" = 130000 ] || [ "$ITER" = 140000 ]; then bash scripts/p4_10_read.sh $NAME $CK "$LA" lite 12
  else bash scripts/p4_10_read.sh $NAME $CK "$LF" lite 12; fi
  GRES=${GRES:-gpu:p6000:1} setsid nohup bash scripts/p4_9_watch_report.sh $ARM $ITER aaf_${ARM,,} outputs/p4_10/arms c > logs/p4_10_watch_report_${ARM}_$ITER.log 2>&1 < /dev/null &
done
