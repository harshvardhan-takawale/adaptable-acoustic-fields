#!/bin/bash
# P4-10 meeting day: dead-man publisher (1-CPU Slurm job). While the agent is active (logs/p4_10_heartbeat younger than STALE seconds) it
# does nothing. When the session has gone silent it, every INTERVAL s: rebuilds MEETING_PACK.md with --auto (marked unreviewed) from
# the JSON on disk, rebuilds the board, commits ONLY the allowlisted result paths, and pushes fast-forward only. It never edits the
# status report, DECISIONS, OPEN_QUESTIONS or CONTEXT, never force-pushes, and stops at END (local HH:MM).
#   sbatch --job-name=aaf_autopub --time=08:00:00 --mem=8G --cpus-per-task=1 ... --wrap="bash scripts/p4_10_autopublish.sh"   (DRY=1: no commit)
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh; conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH=$PWD OMP_NUM_THREADS=1
HB=logs/p4_10_heartbeat; STALE=${STALE:-3000}; INTERVAL=${INTERVAL:-600}; END=${END:-17:00}; LOG=logs/p4_10_autopublish.log
say() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
ALLOW="outputs/p4_10/meeting outputs/p4_10/eb outputs/p4_10/ec outputs/p4_10/p2 outputs/p4_10/pt outputs/p4_10/ee outputs/p4_10/v7 outputs/p4_10/progress configs/p4_10 scripts tests aaf/train/p410_pt.py"
publish() {
  python -m scripts.p4_10_board >/dev/null 2>&1 || say "board rebuild failed"
  python scripts/p4_10_meeting_pack.py --auto >/dev/null 2>&1 || say "pack build failed"
  local f p="";                                       # only small text / figure files; never npz, ckpt, pt
  while IFS= read -r f; do case "$f" in */arms/*) continue;; esac; case "$f" in *.md|*.json|*.png|*.pdf|*.mp4|*.gif|*.py|*.sh|*.yaml) [ -f "$f" ] && [ "$(stat -c %s "$f")" -lt 12000000 ] && p="$p $f";; esac; done < <(git ls-files -m -o --exclude-standard -- $ALLOW 2>/dev/null)
  [ -z "$p" ] && { say "nothing to publish"; return; }
  if [ "${DRY:-0}" = 1 ]; then say "DRY would commit:"; echo "$p" | tr ' ' '\n' | sed '/^$/d' | sed 's/^/   /' | head -60; return; fi
  for n in 1 2 3; do git add -- $p 2>/dev/null && git commit -q -m "P4-10 automatic publish: results on disk while the session was silent (MEETING_PACK built without review)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- $p 2>/dev/null && break; sleep 20; done
  timeout 120 git push -q origin main && say "pushed $(git rev-parse --short HEAD)" || say "FAILED push (not fast-forward or network)"
}
[ "${DRY:-0}" = 1 ] && { publish; exit 0; }
say "autopublish started on $(hostname) job ${SLURM_JOB_ID:-none}; stale after ${STALE}s; until $END"
while [ "$(date +%H:%M)" \< "$END" ]; do
  age=$(( $(date +%s) - $(stat -c %Y $HB 2>/dev/null || echo 0) ))
  if [ "$age" -gt "$STALE" ]; then say "heartbeat ${age}s old: publishing"; publish; fi
  sleep $INTERVAL
done
say "end of day $END reached"
