#!/bin/bash
# P4-9 I11 (E2 / D188): when each arm's 150K checkpoint lands, render its directional readouts on the 31 dense rooms
# (one GPU job per arm, resumable from the per-room cache). Scores once CTL and DIR are rendered (DIR's E2 read), and
# again once DIR-G is (DIR-G runs ~2x slower per step); each score job lists every rendered arm (afterok).
#   setsid nohup bash scripts/p4_9_dir_watch.sh > logs/p4_9_dir_watch.log 2>&1 &
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
GPU="--partition=scavenger --account=scavenger --qos=scavenger --requeue --gres=gpu:rtxa5000:1 --exclude=legacy43"
ITER=${ITER:-150000}
K=$((ITER / 1000))K
JOBS=(); LAUNCHED=" "; N=0
while [ $N -lt 3 ]; do                 # poll all three arms; launch each as soon as its checkpoint is complete
  for A in CTL DIR DIRG; do
    case "$LAUNCHED" in *" $A "*) continue;; esac
    CK=outputs/p4_9/arms/$A/ckpt_iter$(printf %07d "$ITER").pt
    [ -f "$CK" ] || continue
    s1=$(stat -c %s "$CK"); sleep 60; [ "$s1" = "$(stat -c %s "$CK")" ] || continue   # still being written
    J=$(sbatch --parsable $GPU --job-name=aaf_dirzs_$A --time=04:00:00 --mem=32G \
        --export=ALL,CMD="scripts/p4_9_dir_zeroshot.py --render ${A}_${K}=$CK" scripts/slurm/p4_5_gpu.sh)
    echo "$(date) $A $CK -> render job $J"
    JOBS+=("$J"); LAUNCHED="$LAUNCHED$A "; N=$((N + 1))
    if [[ "$LAUNCHED" == *" CTL "* && "$LAUNCHED" == *" DIR "* ]]; then
      if [ -z "${SCORED_CD:-}" ]; then
        D=$(IFS=:; echo "${JOBS[*]}"); NAMES="R0_120K+S2_120K+CTL_${K}+DIR_${K}"
        case "$LAUNCHED" in *" DIRG "*) NAMES="$NAMES+DIRG_${K}"; SCORED_ALL=1;; esac
        S=$(sbatch --parsable $GPU --job-name=aaf_dirzs_score --time=03:00:00 --mem=32G --dependency=afterok:$D \
            --export=ALL,CMD="scripts/p4_9_dir_zeroshot.py --score $NAMES" scripts/slurm/p4_5_gpu.sh)
        echo "$(date) score job $S ($NAMES; after $D)"; SCORED_CD=1
      fi
    fi
  done
  [ $N -lt 3 ] && sleep 300
done
[ -n "${SCORED_ALL:-}" ] && exit 0
DEP=$(IFS=:; echo "${JOBS[*]}")
[ -n "$DEP" ] || { echo "no render jobs; not scoring"; exit 1; }
S=$(sbatch --parsable $GPU --job-name=aaf_dirzs_score --time=03:00:00 --mem=32G --dependency=afterok:$DEP \
    --export=ALL,CMD="scripts/p4_9_dir_zeroshot.py --score R0_120K+S2_120K+CTL_${K}+DIR_${K}+DIRG_${K}" \
    scripts/slurm/p4_5_gpu.sh)
echo "$(date) score job $S (all three; after $DEP)"
