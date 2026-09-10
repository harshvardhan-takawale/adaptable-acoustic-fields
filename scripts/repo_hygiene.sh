#!/bin/bash
# Chunk-end repo hygiene check. Run this before declaring a chunk done.
#
# Why it exists: the manager and other agents read this repo through raw GitHub URLs, so
# anything not pushed is invisible to them. Chunk-end commits have historically covered the
# reports and figures and left run telemetry, eval output and the occasional SLURM wrapper
# behind -- 37 files across five chunks by the time anyone looked. None of it was gitignored;
# it was simply never `git add`ed, which is invisible in a normal `git status` once the
# untracked directories collapse.
#
# Also reports loose-object bloat: accidentally `git add`ing a checkpoint writes a multi-hundred-MB
# blob that survives even if never committed. That is how .git reached 11 GB against 0.31 GiB of
# real content.
#
# Usage:  bash scripts/repo_hygiene.sh
# Exits 1 if anything needs attention, so it can gate a chunk-end script.
set -uo pipefail
cd "$(dirname "$0")/.."

rc=0

echo "=== remote sync ==="
git fetch --quiet origin 2>/dev/null || echo "  (fetch failed -- offline? showing local view)"
git status -sb | head -1
ahead=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)
behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
[ "$ahead" != "0" ]  && { echo "  ! $ahead local commit(s) NOT pushed"; rc=1; }
[ "$behind" != "0" ] && { echo "  ! $behind remote commit(s) not pulled"; rc=1; }
[ "$ahead" = "0" ] && [ "$behind" = "0" ] && echo "  in sync"

echo
echo "=== uncommitted changes to TRACKED files ==="
if git diff --quiet && git diff --cached --quiet; then
  echo "  none"
else
  git status --porcelain | grep -vE '^\?\?' | sed 's/^/  /'
  rc=1
fi

echo
echo "=== untracked, NOT ignored (would be invisible to the manager) ==="
untracked=$(git status --porcelain -uall | grep '^??' | sed 's/^?? //')
if [ -z "$untracked" ]; then
  echo "  none"
else
  echo "$untracked" | while read -r f; do
    printf "  %8s  %s\n" "$(du -h "$f" 2>/dev/null | cut -f1)" "$f"
  done
  echo "  ---"
  echo "  $(echo "$untracked" | wc -l) file(s), $(echo "$untracked" | tr '\n' '\0' | du -ch --files0-from=- 2>/dev/null | tail -1 | cut -f1) total"
  rc=1
fi

echo
echo "=== object store ==="
git count-objects -vH | grep -E '^(count|size|size-pack):' | sed 's/^/  /'
loose=$(git count-objects -v | awk '/^count:/{print $2}')
if [ "${loose:-0}" -gt 2000 ]; then
  echo "  ! $loose loose objects -- run 'git gc --prune=now' (verify no stashes/dangling commits first)"
  rc=1
fi

echo
[ "$rc" -eq 0 ] && echo "CLEAN" || echo "ATTENTION NEEDED (see above)"
exit "$rc"
