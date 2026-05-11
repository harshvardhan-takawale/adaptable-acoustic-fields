#!/bin/bash
#SBATCH --job-name=aaf_spatial_nodes
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --exclude=legacygpu06,legacygpu07
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Args: $1=run (e.g., C2_latent_jitter), $2=L (e.g., 4.25), $3=inner_loop (B1|B6)
# Exit code 0 if verdict is GREEN/YELLOW; 1 if RED. The orchestrator's afterok
# dependency uses this to gate the V1-V4 presentation chain.

set -euo pipefail

RUN="${1:?usage: $0 <run> <L> <inner_loop>}"
L="${2:?usage: $0 <run> <L> <inner_loop>}"
INNER="${3:?usage: $0 <run> <L> <inner_loop>}"

cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; run=${RUN}; L=${L}; inner_loop=${INNER}"

python -m scripts.spatial_nodes_check --run "$RUN" --L "$L" --inner_loop "$INNER"
