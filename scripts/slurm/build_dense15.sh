#!/bin/bash
#SBATCH --job-name=aaf_build_dense15
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Build the 12 new ISM HDF5 rooms for the Chunk-3.7 I1 dense_15 sweep.
# scripts/build_datasets.py is idempotent — it skips any L file already on disk
# (the existing 3 of 15 are reused) and only regenerates the missing ones.

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "host=$(hostname); job=${SLURM_JOB_ID}"
python -m scripts.build_datasets
echo "# build_dense15 done."
