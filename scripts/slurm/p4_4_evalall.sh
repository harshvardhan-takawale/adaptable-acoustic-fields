#!/bin/bash
# P4-4 Task 2: fire Gate 2 + the sweep curve for one arm. Usage: sbatch p4_4_evalall.sh <tag>
#SBATCH --job-name=aaf_p4_4_eval
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
TAG="${1:?usage: $0 <C92|C250|C500>}"
RUN="outputs/p4_4/p4_4_${TAG}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; run=${RUN}"
# Gate 2 is always on the p4_2 manifest: its 15 test shapes are byte-identical in every
# corpus, so this keeps the headline number comparable across P4-2, P4-3 and P4-4.
python scripts/p4_2_stage2_eval.py --run-dir "${RUN}" \
       --manifest configs/sweeps_2d_mat/p4_2_shapes_manifest.json
echo "[gate2 exit] $?"
python scripts/p4_2_sweep_figures.py --run-dir "${RUN}" --out "outputs/p4_4/sweep/${TAG}"
echo "[sweep exit] $?"
