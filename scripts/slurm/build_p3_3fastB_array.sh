#!/bin/bash
# P3-3-FAST Track 2b dataset build: 8 configs per array task (472 total = 59 tasks).
# Usage:  sbatch --array=0-58%60 scripts/slurm/build_p3_3fastB_array.sh
# Idempotent: build_one short-circuits on the .done sentinel, so re-running fills gaps.
# The WORKLIST is never filtered on .done -- doing so would shrink it as the build progresses
# and race the array index against the config mapping (79 of 479 P3-2c configs went missing
# that way).
#
# WALL TIME: the FDTD loop is memory-bandwidth bound, so the per-room cost depends on how many
# array tasks the scheduler packs onto one node. Measured on the 2026-08-17 build: ~225 s/room
# with a node to spare, but ~25-30 min/room where 10 tasks landed on one legacygpu node -- an
# 8x spread, and 8 rooms x 30 min would blow a 3 h limit. Hence 8 h. If contention recurs,
# throttle to %30 rather than raising this further.
#SBATCH --job-name=aaf_p33B_build
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
source /fs/nexus-scratch/htakawal/miniconda3/etc/profile.d/conda.sh
conda activate aaf
# Required: conda's libstdc++ (GLIBCXX_3.4.29) must precede /lib64's older copy.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}"
# One BLAS thread per worker: the FDTD loop is numpy element-wise, threads only add churn.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host=$(hostname); job=${SLURM_JOB_ID}; task=${SLURM_ARRAY_TASK_ID}"
python scripts/build_p3_3fast_trackB.py --idx "${SLURM_ARRAY_TASK_ID}" --chunk "${CHUNK:-8}"
