#!/bin/bash
# P4-10: backstop for the login-node watchers (runs as a 1-CPU Slurm job, independent of any login session).
# Every INTERVAL seconds, for each arm's scheduled checkpoint older than AGE seconds whose read was NOT launched (no watcher or
# backstop log line, no queued render, no scores), launch the same read the watcher would have (p4_10_read.sh) and the standard
# report. It never acts on a checkpoint younger than AGE, so a live watcher always goes first. Exits when every arm has its 150K read.
# sbatch --job-name=aaf_backstop --time=14:00:00 --mem=2G --cpus-per-task=1 --partition=scavenger --account=scavenger \
#        --qos=scavenger --requeue --output=logs/slurm/%x-%j.out --wrap="bash scripts/p4_10_backstop.sh"     (DRY=1 prints only)
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
AGE=${AGE:-2700}; INTERVAL=${INTERVAL:-600}
ALL="StretchU+StretchA+Wall+Notch+Source+Material"
LOG=logs/p4_10_backstop.log; mkdir -p logs; touch $LOG
say() { echo "$(date '+%H:%M') $*" | tee -a $LOG; }
launched() {  # NAME ARM -> 0 if a read of NAME was already launched by anyone
  grep -qE " $1 (lite|full) " logs/p4_10_watch_$2.log 2>/dev/null && return 0
  grep -qE "BACKSTOP $1 " $LOG 2>/dev/null && return 0
  squeue -h -u "$USER" -n aaf_sc_$1 2>/dev/null | grep -q . && return 0
  [ -d /fs/nexus-scratch/htakawal/aaf_p410/score/$1 ] && [ -n "$(ls -A /fs/nexus-scratch/htakawal/aaf_p410/score/$1 2>/dev/null)" ] && return 0
  return 1
}
scan() {
  for spec in "R1:Source" "R1S:StretchU+StretchA" "R1M:Material"; do
    ARM=${spec%%:*}; LF=${spec#*:}
    for ITER in 125000 130000 135000 140000 145000 150000; do
      NAME=${ARM}_$((ITER/1000))K; CK=outputs/p4_10/arms/$ARM/$(printf 'ckpt_iter%07d.pt' $ITER)
      [ -f "$CK" ] || continue
      [ $(( $(date +%s) - $(stat -c %Y "$CK") )) -lt $AGE ] && continue
      launched $NAME $ARM && continue
      if [ "$ITER" = 150000 ]; then FAM=$ALL; MODE=full; CH=16
      elif [ "$ITER" = 130000 ] || [ "$ITER" = 140000 ]; then FAM=$ALL; MODE=lite; CH=12
      else FAM=$LF; MODE=lite; CH=12; fi
      if [ "${DRY:-0}" = 1 ]; then say "DRY would launch $NAME $MODE $FAM"; continue; fi
      say "BACKSTOP $NAME $MODE $FAM (watcher did not launch it within $((AGE/60)) min of the checkpoint)"
      bash scripts/p4_10_read.sh $NAME "$CK" "$FAM" $MODE $CH 2>&1 | tee -a $LOG
      ( GRES=gpu:p6000:1 bash scripts/p4_9_watch_report.sh $ARM $ITER aaf_${ARM,,} outputs/p4_10/arms c > logs/p4_10_watch_report_${ARM}_$ITER.backstop.log 2>&1 < /dev/null ) &
    done
  done
}
done_all() { for a in R1 R1S R1M; do launched ${a}_150K $a || return 1; done; return 0; }
[ "${DRY:-0}" = 1 ] && { scan; exit 0; }
say "backstop started on $(hostname), job ${SLURM_JOB_ID:-none}"
while true; do scan; if done_all; then say "every arm has its 150K read launched; exiting"; wait; exit 0; fi; sleep $INTERVAL; done
