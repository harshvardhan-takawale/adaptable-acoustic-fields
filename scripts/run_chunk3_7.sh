#!/usr/bin/env bash
# Chunk 3.7 orchestrator — meeting visual story + parallel improvement experiments.
#
# Stages:
#   0  pytest gate                                  ( 1 job )
#   1  V0 spatial nodes check at L=4.25 (BLOCKING)  ( 1 job )
#   2a V1 cross-L spatial check at the other 5 L    ( 5 jobs, afterok V0 )
#   2b V1 cross-L summary                           ( 1 job )
#   2c V2 modal-tracking polished plot              ( 1 job, afterok V0 )
#   2d V3 audio morph demo                          ( 1 job, afterok V0 )
#   2e V4 assemble meeting assets                   ( 1 job, after V1/V2/V3 )
#   3  Track I — I1 dataset build + train + ZS      ( 1 + 1 + 12 jobs )
#   4  Track I — I2 LoRA train + ZS                 ( 1 + 12 jobs )
#   5  Track I — I3 B7 chunked ZS on C2             ( 6 jobs )
#   6  Final summary                                ( 1 job, depends on all above )
#
# V0 exit code 0 (GREEN/YELLOW) → V1-V4 run. Exit 1 (RED) → V1-V4 stay PENDING
# under afterok and SLURM cancels them with reason=DependencyNeverSatisfied.
# Track I is independent of V0 — those jobs run regardless of the visual verdict.

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
mkdir -p logs/slurm

# Stage 0: pytest gate.
PY=$(sbatch --parsable scripts/slurm/run_pytest.sh)
echo "stage 0 (pytest)        : $PY"

# Stage 1: V0 — BLOCKING. C2_latent_jitter + B6 inner loop at L=4.25.
V0=$(sbatch --parsable --dependency=afterok:$PY \
    scripts/slurm/spatial_nodes_check.sh C2_latent_jitter 4.25 B6)
echo "stage 1 (V0)            : $V0"

# Stage 2a: V1 cross-L spatial checks at the other 5 L values.
V1_JOBS=()
for L in 3.25 3.75 4.75 5.25 5.75; do
    J=$(sbatch --parsable --dependency=afterok:$V0 \
        scripts/slurm/spatial_nodes_check.sh C2_latent_jitter $L B6)
    V1_JOBS+=("$J")
done
V1_DEPS=$(IFS=,; echo "${V1_JOBS[*]}")
echo "stage 2a (V1 5×L)       : ${V1_JOBS[*]}"

# Stage 2b: V1 cross-L summary.
V1_SUM=$(sbatch --parsable --dependency=afterok:$V0,$V1_DEPS \
    scripts/slurm/spatial_nodes_summary.sh)
echo "stage 2b (V1 summary)   : $V1_SUM"

# Stage 2c: V2 modal-tracking polished plot.
V2=$(sbatch --parsable --dependency=afterok:$V0 scripts/slurm/modal_tracking_plot.sh)
echo "stage 2c (V2)           : $V2"

# Stage 2d: V3 audio morph demo.
V3=$(sbatch --parsable --dependency=afterok:$V0 scripts/slurm/audio_morph_demo.sh)
echo "stage 2d (V3)           : $V3"

# Stage 2e: V4 assemble — depends on V1 summary + V2 + V3.
V4=$(sbatch --parsable --dependency=afterok:$V1_SUM,$V2,$V3 \
    scripts/slurm/assemble_meeting_assets.sh)
echo "stage 2e (V4 assemble)  : $V4"

# Stage 3: I1 — dataset build + train + ZS.
BUILD=$(sbatch --parsable --dependency=afterok:$PY scripts/slurm/build_dense15.sh)
echo "stage 3 (build dense15) : $BUILD"
I1_TRAIN=$(sbatch --parsable --dependency=afterok:$BUILD --partition=tron \
    --account=nexus --qos=default scripts/slurm/sweep_train.sh configs/sweep/D1_dense15.yaml)
echo "stage 3 (train D1)      : $I1_TRAIN"

# Stage 4: I2 — LoRA train + ZS (no dataset dep; uses existing dense.yaml).
I2_TRAIN=$(sbatch --parsable --dependency=afterok:$PY --partition=tron \
    --account=nexus --qos=default scripts/slurm/sweep_train.sh configs/sweep/D2_filmlora.yaml)
echo "stage 4 (train D2)      : $I2_TRAIN"

# I1/I2 evaluation: B1 + B6 at each unseen L (12 ZS jobs each).
I1_ZS_JOBS=()
I2_ZS_JOBS=()
for L in 3.25 3.75 4.25 4.75 5.25 5.75; do
    # I1
    I1_ZS_JOBS+=($(sbatch --parsable --dependency=afterok:$I1_TRAIN \
        scripts/slurm/zero_shot_variant.sh B1 $L D1_dense15))
    I1_ZS_JOBS+=($(sbatch --parsable --dependency=afterok:$I1_TRAIN \
        scripts/slurm/zero_shot_with_best_variant.sh D1_dense15 $L))
    # I2
    I2_ZS_JOBS+=($(sbatch --parsable --dependency=afterok:$I2_TRAIN \
        scripts/slurm/zero_shot_variant.sh B1 $L D2_filmlora))
    I2_ZS_JOBS+=($(sbatch --parsable --dependency=afterok:$I2_TRAIN \
        scripts/slurm/zero_shot_with_best_variant.sh D2_filmlora $L))
done
I1_ZS_DEPS=$(IFS=,; echo "${I1_ZS_JOBS[*]}")
I2_ZS_DEPS=$(IFS=,; echo "${I2_ZS_JOBS[*]}")
echo "stage 3 (I1 ZS x12)     : ${I1_ZS_JOBS[0]}..${I1_ZS_JOBS[-1]}"
echo "stage 4 (I2 ZS x12)     : ${I2_ZS_JOBS[0]}..${I2_ZS_JOBS[-1]}"

# Stage 5: I3 — B7 chunked-receiver inner loop on existing C2_latent_jitter.
I3_JOBS=()
for L in 3.25 3.75 4.25 4.75 5.25 5.75; do
    I3_JOBS+=($(sbatch --parsable --dependency=afterok:$PY \
        scripts/slurm/zero_shot_variant.sh B7 $L C2_latent_jitter))
done
I3_DEPS=$(IFS=,; echo "${I3_JOBS[*]}")
echo "stage 5 (I3 B7 x6)      : ${I3_JOBS[*]}"

# Stage 6: final summary — depends on V4 + Track I evaluations.
# Use afterany on the V chain so the final summary still fires if V0 was RED
# (and V1-V4 got cancelled). The summary script handles missing artifacts.
FINAL_DEPS_OK="$V4,$I1_ZS_DEPS,$I2_ZS_DEPS,$I3_DEPS"
FINAL=$(sbatch --parsable --dependency=afterany:$V0:$V4:$I1_TRAIN:$I2_TRAIN \
    --kill-on-invalid-dep=no scripts/slurm/chunk_3_7_final_summary.sh)
echo "stage 6 (final)         : $FINAL"

echo ""
echo "All jobs queued. Tail with:"
echo "    squeue -u \$USER -t PD,R --noheader -o '%.10i %.20j %.10T %.10M %R'"
echo ""
echo "V0 verdict will land at:"
echo "    outputs/spatial_nodes_check/L4.25/nodes_check_report.md"
echo "If RED, V1/V2/V3/V4 stay PENDING with reason=DependencyNeverSatisfied; the"
echo "final summary still fires and notes the RED verdict."
