#!/bin/bash
# P4-8 (D174(h)): watch ONE arm's checkpoint and launch its whole standard report (D160) the moment it lands, then exit
# when every output exists. Replaces the P4-6 inline watcher loops, which died with their session.
#   eval_ckpt + dense Gate-2 (p4_5_eval_one.sh's two jobs) -> speckle analysis (needs the dense cache)
#   training-room fit (FIT=c: C250 off-lattice rooms; FIT=fam: the 20 family rooms) and out-of-family
#   then GAP_CI + RISK_METRICS (CPU) once the dense cache exists.
# At 4K only the fit runs (D135's kill signal). Exits 2 if the training job dies before the checkpoint.
# Usage: bash scripts/p4_8_watch_report.sh R0 30000 7886225 outputs/p4_8/arms c   (KILLSIG=2000 for LS2)
# Adds the frozen unseen-family set (scripts/p4_8_unseen_eval.py, UF) to P4-7's standard report.
set -u
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
A=$1; ITER=$2; JOB=$3; ARMS=$4; FIT=$5; K=$((ITER / 1000))K; N=${A}_${K}
CK=$ARMS/$A/$(printf "ckpt_iter%07d.pt" "$ITER"); MARK=outputs/p4_5/eval/.launched; mkdir -p "$MARK"
G2J=outputs/p4_5/gate2_dense/${N}__track_p4_5_testdense.json
EVJ=outputs/p4_5/diagnostics/energy_${N}.json
if [ "$FIT" = fam ]; then FITPY=scripts/p4_6_train_fit_family.py; FITJ=outputs/p4_6/train_fit/${N}.json
else FITPY=scripts/p4_5_train_offlat_fit.py; FITJ=outputs/p4_5/train_offlat_fit/${N}.json; fi
OOFJ=outputs/p4_6/family_oof/${N}.json; SPJ=outputs/p4_5/speckle/${N}.json
CIJ=outputs/p4_8/GAP_CI_${N}.json; RKJ=outputs/p4_8/RISK_METRICS_${N}.json; UNJ=outputs/p4_8/unseen/${N}.json
GPU="--partition=scavenger --account=scavenger --qos=scavenger --requeue --gres=gpu:rtxa5000:1 --exclude=legacy43"
launch() {  # tag, job-name, time, mem, cmd
  [ -f "$MARK/${N}_$1" ] && return
  E=$(sbatch --parsable $GPU --job-name=$2 --time=$3 --mem=$4 --export=ALL,CMD="$5" scripts/slurm/p4_5_gpu.sh) \
    && touch "$MARK/${N}_$1" && echo "$(date +%H:%M) launched $1 $N -> $E"
}
while true; do
  if [ -f "$CK" ]; then
    launch fit aaf_fit 02:00:00 32G "$FITPY --items $N=$CK"
    if [ "$ITER" -gt "${KILLSIG:-4000}" ]; then
      if [ ! -f "$MARK/${N}" ]; then
        E=$(sbatch --parsable --export=ALL,RUN=$ARMS/$A,CK=$(basename "$CK"),NAME=$N scripts/slurm/p4_5_eval_ckpt.sh) \
          && touch "$MARK/${N}" && echo "$(date +%H:%M) launched eval $N -> $E"
      fi
      launch dense aaf_g2dense 03:00:00 32G "scripts/p4_5_gate2_dense.py --checkpoint $CK --name $N"
      launch oof aaf_oof 03:00:00 64G "scripts/p4_6_family_oof.py --items $N=$CK"
      launch unseen aaf_unseen_ev 03:00:00 32G "scripts/p4_8_unseen_eval.py --items $N=$CK"
      [ -f "$G2J" ] && launch speckle aaf_speckle_an 03:00:00 64G "scripts/p4_5_speckle.py --names $N --project"
      if [ -f "$G2J" ] && [ -f "$EVJ" ] && [ ! -f "$MARK/${N}_ci" ]; then
        E=$(sbatch --parsable --partition=scavenger --account=scavenger --qos=scavenger --requeue --cpus-per-task=2 \
            --mem=32G --time=01:00:00 --job-name=aaf_ci --output=logs/slurm/%x-%j.out --error=logs/slurm/%x-%j.err \
            --wrap="cd $PWD && source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh && conda activate aaf && export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-} && export PYTHONPATH=$PWD && python scripts/p4_5_gate2_ci.py --names $N --out $CIJ && python scripts/p4_5_risk_metrics.py --names $N --evals $N --out $RKJ") \
          && touch "$MARK/${N}_ci" && echo "$(date +%H:%M) launched ci+risk $N -> $E"
      fi
      if [ -f "$FITJ" ] && [ -f "$G2J" ] && [ -f "$EVJ" ] && [ -f "$OOFJ" ] && [ -f "$SPJ" ] && [ -f "$CIJ" ] && [ -f "$RKJ" ] && [ -f "$UNJ" ]; then
        echo "$(date +%H:%M) $N REPORT COMPLETE"; exit 0
      fi
    elif [ -f "$FITJ" ]; then
      echo "$(date +%H:%M) $N FIT (kill signal) COMPLETE"; exit 0
    fi
  elif ! squeue -j "$JOB" -h 2>/dev/null | grep -q .; then
    sleep 120
    [ -f "$CK" ] || { echo "$(date +%H:%M) ARM $A: training job $JOB gone before $(basename "$CK")"; exit 2; }
  fi
  sleep 120
done
