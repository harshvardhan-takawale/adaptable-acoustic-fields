#!/bin/bash
# P4-10: one scored read of one arm checkpoint. bash scripts/p4_10_read.sh NAME CKPT FAMILIES lite|full [CHUNK]
# GPU render array -> CPU score array (dependent) -> board rebuild (dependent). Every sbatch exit code is checked: a failed
# submission prints FAILED (tracked waits grep for it). Lite = the 16-room scorecard-lite subset of FAMILIES.
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
NAME=$1; CK=$2; FAM=$3; MODE=$4; CHUNK=${5:-12}
[ -f "$CK" ] || { echo "FAILED $NAME: no checkpoint $CK"; exit 1; }
L=""; [ "$MODE" = lite ] && L="--lite"
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh; conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH=$PWD
N=$(python scripts/p4_10_scorecard.py --plan --families "$FAM" $L --chunk $CHUNK 2>/dev/null | grep -oE '^[0-9]+ cells' | cut -d' ' -f1)
[ "${N:-0}" -gt 0 ] || { echo "FAILED $NAME: no cells"; exit 1; }
T=$(( (N - 1) / CHUNK + 1 ))
sub() { local J; J=$(sbatch --parsable "$@") && echo "$J" || { echo "FAILED submit: $*" >&2; return 1; }; }
R=$(sub --array=0-$((T-1))%16 --job-name=aaf_sc_$NAME --export=ALL,CMD="scripts/p4_10_scorecard.py --render --model $NAME=$CK --families $FAM $L --chunk $CHUNK --idx @" scripts/slurm/p4_10_gpu.sh) || { echo "FAILED $NAME render"; exit 1; }
S=$(sub --array=0-$((T-1))%32 --dependency=afterok:$R --kill-on-invalid-dep=yes --mem=16G --time=02:00:00 --job-name=aaf_ss_$NAME --export=ALL,CMD="scripts/p4_10_scorecard.py --score --model $NAME --families $FAM $L --chunk $CHUNK --idx @" scripts/slurm/p4_10_cpu.sh) || { echo "FAILED $NAME score"; exit 1; }
G=$(sub --dependency=afterany:$S --time=00:30:00 --mem=8G --job-name=aaf_board --export=ALL,CMD="scripts/p4_10_board.py" scripts/slurm/p4_10_cpu.sh) || { echo "FAILED $NAME board"; exit 1; }
echo "$(date '+%H:%M') $NAME $MODE $FAM ($N cells): render $R score $S board $G"
