#!/bin/bash
# P4-3 Arm D dataset: the 32 shallow-band rooms added to P4-2's 75. CPU only.
# The other 75 are skipped by their .done sentinels -- the worklist is the FULL stable list,
# never filtered on .done (filtering races the array index against the config mapping).
# Usage:  sbatch --array=0-64 scripts/slurm/build_p4_3_shapes_array.sh
#SBATCH --job-name=aaf_p4_4_fam
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}"
python scripts/build_p4_4_family_rooms.py \
    --manifest configs/sweeps_2d_mat/p4_4_family_manifest.json \
    --idx "${SLURM_ARRAY_TASK_ID}" --chunk 8
