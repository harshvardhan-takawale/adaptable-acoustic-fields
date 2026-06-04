#!/bin/bash
# Chunk P2-2.5 orchestrator: 3 diagnostic training runs + summary.
#
# DAG:
#   pytest_gate
#     ↓
#   sample_diag_rooms  (writes configs/sweeps_3d/diag_10rooms.yaml)
#     ↓
#     ├─ diag_train_A (tron qos=default)
#     ├─ diag_train_B (tron qos=high)
#     └─ diag_train_C (tron qos=high)
#         ↓ afterok (all 3)
#       diag_summary  (writes DIAGNOSIS.md)

set -euo pipefail
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
mkdir -p logs/slurm

echo "# Submitting P2-2.5 pipeline at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Stage 0: pytest gate.
PY=$(sbatch --parsable scripts/slurm/run_pytest.sh)
echo "stage 0 (pytest)        : $PY"

# Stage 1: generate the 10-room maximin subset YAML.
SAMPLE=$(sbatch --parsable --dependency=afterok:$PY scripts/slurm/sample_diag_rooms.sh)
echo "stage 1 (sample diag)   : $SAMPLE"

# Stage 2: 3 trainings in parallel after the YAML lands.
A=$(sbatch --parsable --dependency=afterok:$SAMPLE scripts/slurm/diag_train_A.sh)
echo "stage 2 (A 10rm b=16)   : $A"
B=$(sbatch --parsable --dependency=afterok:$SAMPLE scripts/slurm/diag_train_B.sh)
echo "stage 2 (B 45rm b=32)   : $B"
C=$(sbatch --parsable --dependency=afterok:$SAMPLE scripts/slurm/diag_train_C.sh)
echo "stage 2 (C 10rm b=64)   : $C"

# Stage 3: summary depends on all three.
SUMMARY=$(sbatch --parsable --dependency=afterok:$A:$B:$C scripts/slurm/diag_summary.sh)
echo "stage 3 (DIAGNOSIS.md)  : $SUMMARY"

echo
echo "# All jobs submitted. Final DIAGNOSIS lands at:"
echo "#   outputs/diag_p2_2_5/DIAGNOSIS.md"
