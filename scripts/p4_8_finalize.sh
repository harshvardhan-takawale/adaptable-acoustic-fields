#!/bin/bash
# P4-8 (D174(h)): when every wave-1 FINAL standard report exists, build the primary table and the meeting figures
# (CPU job). Also writes the 60K-read table (wave-2 decision) as soon as the 60K finals exist, before R0 @120K.
# Usage: setsid nohup bash scripts/p4_8_finalize.sh >> logs/watch_p4_8_finalize.log 2>&1 &
set -u
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
have() { for n in "$@"; do [ -f outputs/p4_8/unseen/$n.json ] && [ -f outputs/p4_5/speckle/$n.json ] && \
         [ -f outputs/p4_5/train_offlat_fit/$n.json ] || return 1; done; }
job() {  # name, command
  sbatch --parsable --partition=scavenger --account=scavenger --qos=scavenger --requeue --cpus-per-task=2 --mem=48G \
    --time=02:00:00 --job-name=$1 --output=logs/slurm/%x-%j.out --error=logs/slurm/%x-%j.err \
    --wrap="cd $PWD && source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh && conda activate aaf && export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-} && export PYTHONPATH=$PWD && $2"
}
W1="MB1_60K MB2_60K LS1A_60K LS1B_60K LS3_60K LS2A_30K LS2B_30K"   # LS2: end point 30K (D176)
until have $W1; do sleep 300; done
echo "$(date +%m-%d_%H:%M) wave-1 60K finals in -> table $(job aaf_p48_table 'python scripts/p4_8_primary_table.py')"
until have R0_120K; do sleep 300; done
# D176(e): figures carry S2 at 120K if it lands; if S2's job is gone without it, its 60K read (labelled)
until have S2_120K || ! squeue -n aaf_p4_8 -h -o %j 2>/dev/null | grep -q . && ! ls outputs/p4_8/arms/S2/ckpt_iter0120000.pt >/dev/null 2>&1; do sleep 300; done
S2ARG="--s2 S2_60K"; have S2_120K && S2ARG="--s2 S2_120K"
echo "$(date +%m-%d_%H:%M) finals in -> table + figures ($S2ARG) $(job aaf_p48_figs "python scripts/p4_8_primary_table.py && python scripts/p4_8_meeting_figs.py $S2ARG")"
