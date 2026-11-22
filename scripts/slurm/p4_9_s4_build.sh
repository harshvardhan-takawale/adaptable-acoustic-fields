#!/bin/bash
# P4-9 S-4 (D181 E8): build tinycudann for sm_86 / sm_89 in a SEPARATE env (aaf_sm89, cloned from aaf). The aaf env and
# environment.yml are NOT touched (CLAUDE.md: do not rebuild tinycudann in aaf). Same tinycudann commit as aaf's wheel
# (c91138bc, v1.7; aaf's wheel reports compute capability 52, i.e. FullyFusedMLP falls back). Time-boxed (4 h).
#SBATCH --job-name=aaf_s4build
#SBATCH --partition=scavenger --account=scavenger --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1 --cpus-per-task=16 --mem=48G --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -eo pipefail
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
[ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
ENV=/fs/nexus-scratch/htakawal/miniconda3/envs/aaf_sm89
if [ ! -f "$ENV/.clone_done" ]; then
  rm -rf "$ENV"
  conda create -y --clone aaf -n aaf_sm89
  touch "$ENV/.clone_done"
fi
conda activate aaf_sm89
module load cuda/11.8.0
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export TCNN_CUDA_ARCHITECTURES="86;89"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=16
nvcc --version | tail -2; gcc --version | head -1; nvidia-smi --query-gpu=name,compute_cap --format=csv
cd /fs/nexus-scratch/htakawal/aaf_sm89_build
[ -d tiny-cuda-nn ] || git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git
cd tiny-cuda-nn && git fetch -q && git checkout -q c91138bcd4c6877c8d5e60e483c0581aafc70cce && git submodule update --init --recursive
grep -n "TCNN_HALF_PRECISION\|TCNN_CUDA_ARCHITECTURES\|min_compute" bindings/torch/setup.py | head
cd bindings/torch
time pip install --no-deps --force-reinstall --no-build-isolation .
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
python -c "import torch, tinycudann as t; print('import ok', torch.cuda.get_device_capability())"
echo BUILD_OK
