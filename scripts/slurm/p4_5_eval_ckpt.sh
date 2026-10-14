#!/bin/bash
# P4-5: evaluate one checkpoint -- the frequency curve (p4_5_eval.py) AND the frozen Gate-2
# evaluator, run unchanged. Usage:
#   sbatch --export=ALL,RUN=outputs/p4_5/arms/B1,CK=ckpt_iter0030000.pt,NAME=B1_30K scripts/slurm/p4_5_eval_ckpt.sh
#SBATCH --job-name=aaf_p4_5_eval
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --exclude=legacy43
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -uo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo "host=$(hostname); job=${SLURM_JOB_ID}; run=${RUN}; ck=${CK}; name=${NAME}"
python scripts/p4_5_eval.py --checkpoint "${RUN}/${CK}" --name "${NAME}"
python scripts/p4_2_stage2_eval.py --run-dir "${RUN}" --checkpoint "${RUN}/${CK}" \
    --manifest configs/sweeps_2d_mat/p4_2_shapes_manifest.json --data-dir data/track_p4_2_shapes \
    --out "outputs/p4_5/eval/gate2_${NAME}"
# D99: every evaluation also splits the dense set by training-lattice membership, and reports
# D98's energy/contrast diagnostic -- both CPU-only, from the render cache just written.
python scripts/p4_5_lattice_diag.py --cache "outputs/p4_5/eval/cache/render_${NAME}.npz" --name "${NAME}"
python scripts/p4_5_energy_diag.py --cache "outputs/p4_5/eval/cache/render_${NAME}.npz" --name "${NAME}"
# D108: the supra-physical fraction is a standing metric for every checkpoint.
python scripts/p4_5_kspace_diag.py --names "${NAME}" --out "outputs/p4_5/diagnostics/kspace_${NAME}" 2>&1 | tail -3 || true
