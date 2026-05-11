#!/bin/bash
#SBATCH --job-name=aaf_sweep_train
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Generic Chunk-3.5 sweep training wrapper. Takes a hyperparameter YAML as $1.
# Output dir is derived from the YAML's `run_id` field. The orchestrator
# overrides --partition/--account/--qos at submit time for R0 (tron).

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

CONFIG="${1:?usage: $0 <configs/sweep/R*.yaml>}"
RUN_ID=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "$CONFIG")
OUT="outputs/multi_room/sweep/${RUN_ID}"

echo "host=$(hostname); job=${SLURM_JOB_ID}; cfg=${CONFIG}; out=${OUT}"
nvidia-smi -L || true
mkdir -p "${OUT}"

python -m aaf.train.multi_room --config "${CONFIG}" --output_dir "${OUT}"
