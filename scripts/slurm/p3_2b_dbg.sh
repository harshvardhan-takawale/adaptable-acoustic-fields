#!/bin/bash
#SBATCH --job-name=aaf_p3_2b_dbg
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
echo "### arm A smoke, FULL output, no pipe ###"
python -u -m aaf.train.multi_room_2d_mat --config configs/sweep_2d_mat/P3_2b_A_smoke.yaml \
  --output_dir outputs/p3_2b/smoke_A
echo "### exit=$? ###"
