# Chunk P4-9 — physics as structure, directional output, sparse grounding: RESULTS (live)

**Status (2026-11-22, evening): pre-registered (D180-D181); Day 0 done (D182); I2 in, I1/I3/I4/I5 running.** Live log: [P4_9_STATUS_REPORT.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_9_STATUS_REPORT.md).

**Questions.** Q-A: where does the network beat physics (update speed, approximate inputs, sparse measurements)? Q-B: does physics help as structure (learned modes, NMF)? Q-C: can the model output direction, and does directional supervision change omni accuracy? Q-D: does any field-level physics residual target the unseen-family failure?

**Protocol** (unchanged): off-lattice everywhere (Rule 11); frozen sets = the 15 in-family test shapes and the 16 unseen rooms; paired bootstrap 10k, seed 20261013; the D160 standard report for every trained arm.

Sections below are filled as each step lands.

## Step 0 — free diagnostics

**0b(i) interior Helmholtz (I2):** the pre-registered reading DOES NOT HOLD (projected R0 ~23x FDTD's raw residual in every band). Confound flagged (D182(f)): the projection raises TRUTH's residual 21x; against projected truth, projected R0 is 1.08x / 1.13x / 1.11x. [physics_gates.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/physics_gates.md).

**0c unit test (E3):** passes ([unit.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/dirzs/unit.json)).
## Step 1 — sparse-measurement grounding
## Step 2 — training arms (S2X, CTL, DIR, DIR-G, NMF, NMF-P)
## Suggestions S-1 to S-4
## Engineering

All default-off, Rule-8 tested (D182(e)): warm start + closed-form LR restart; the P4-9 step with a dedicated batch generator (CTL / DIR / DIR-G draw identical batches and jitters); DIR (per-ray readout) and DIR-G (finite-difference omni gradient) terms with a calibrated, checkpointed weight; the 601-bin inference path; per-ray output; the per-token override (M1); the projection's padded-grid return + exact band-limited evaluation; FDTD stencil reruns (exact); m1 modal solves on perturbed geometries; pooled training launches.
