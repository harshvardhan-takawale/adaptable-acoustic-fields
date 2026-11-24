#!/bin/bash
# P4-10: one scored lite read when GPUs are scarce. bash scripts/p4_10_read_pool.sh NAME CKPT FAMILIES
# The render array is submitted once per GPU type (cells already rendered are skipped); one CPU job waits until every prediction exists (or
# its time is up), scores all cells with 8 processes, cancels the leftover render arrays and rebuilds the board. No afterok chain on a GPU type.
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
NAME=$1; CK=$2; FAM=$3; CHUNK=12
[ -f "$CK" ] || { echo "FAILED $NAME: no checkpoint"; exit 1; }
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh; conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH=$PWD
N=$(python scripts/p4_10_scorecard.py --plan --families "$FAM" --lite --chunk $CHUNK 2>/dev/null | grep -oE '^[0-9]+ cells' | cut -d' ' -f1)
T=$(( (N - 1) / CHUNK + 1 ))
EX=legacy43,tron29,tron36,gammagpu02,gammagpu03,gammagpu04,gammagpu05,gammagpu06,gammagpu09,cml23
for spec in "p6000:16" "rtxa4000:16" "rtxa5000:16" "rtxa6000:24" "l40s:24" "rtx2080ti:6"; do
  G=${spec%%:*}; P=${spec##*:}
  sbatch --parsable --array=0-$((T-1)) --exclude=$EX --partition=scavenger --account=scavenger --qos=scavenger --requeue --gres=gpu:$G:1 --time=00:30:00 --cpus-per-task=1 --mem=10G --job-name=aaf_pool_$NAME --export=ALL,P410_CHUNK=$P,CMD="scripts/p4_10_scorecard.py --render --model $NAME=$CK --families $FAM --lite --chunk $CHUNK --idx @" scripts/slurm/p4_10_gpu.sh >/dev/null || echo "submit failed for $G"
done
LG=$(sinfo -h -N -o "%N" | grep -E "^legacy" | sort -u | paste -sd,)
S=$(sbatch --parsable --exclude=$LG,quics00,cml23 --mem=24G --cpus-per-task=8 --time=02:30:00 --job-name=aaf_poolscore_$NAME --export=ALL,NAME=$NAME,FAM=$FAM,T=$T,N=$N,CHUNK=$CHUNK --partition=scavenger --account=scavenger --qos=scavenger --requeue --output=logs/slurm/%x-%j.out --error=logs/slurm/%x-%j.err --wrap='cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields; source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh; conda activate aaf; export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-} PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; D=/fs/nexus-scratch/htakawal/aaf_p410/pred/$NAME; for i in $(seq 1 130); do n=$(ls $D/*.npz 2>/dev/null | grep -v tmp | wc -l); [ "$n" -ge "$N" ] && break; sleep 60; done; echo "$(date +%H:%M) $n of $N predictions"; seq 0 $((T-1)) | xargs -P 8 -I{} python scripts/p4_10_scorecard.py --score --model $NAME --families "$FAM" --lite --chunk $CHUNK --idx {}; scancel -u $USER -n aaf_pool_$NAME 2>/dev/null; python -m scripts.p4_10_board >/dev/null 2>&1; echo "$(date +%H:%M) scored"')
echo "$(date '+%H:%M') $NAME POOL $FAM ($N cells): gather+score $S"
