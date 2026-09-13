#!/bin/bash
# P4-3: Gate-2 metrics + sigma probe for one arm. Usage: sbatch p4_3_gate2.sh <run-dir>
# The p4_2 manifest is used for EVERY arm, including D: the 15 test shapes are byte-identical
# in both manifests (asserted at freeze), so this keeps the evaluation literally the same.
#SBATCH --job-name=aaf_p4_3_gate2
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --time=01:00:00
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
RUN="${1:?usage: $0 <run-dir>}"
echo "host=$(hostname); job=${SLURM_JOB_ID}; run=${RUN}"
python scripts/p4_2_stage2_eval.py --run-dir "${RUN}" \
    --manifest configs/sweeps_2d_mat/p4_2_shapes_manifest.json
