# Chunk P4-9 — physics as structure, directional output, sparse grounding: RESULTS (live)

**Status (2026-11-22, night): pre-registered (D180-D181); Day 0 done (D182); I2, I9, I4, I1, I13 (kill read) in; I3/I5/I7 running.** Live log: [P4_9_STATUS_REPORT.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_9_STATUS_REPORT.md).

**Questions.** Q-A: where does the network beat physics (update speed, approximate inputs, sparse measurements)? Q-B: does physics help as structure (learned modes, NMF)? Q-C: can the model output direction, and does directional supervision change omni accuracy? Q-D: does any field-level physics residual target the unseen-family failure?

**Protocol** (unchanged): off-lattice everywhere (Rule 11); frozen sets = the 15 in-family test shapes and the 16 unseen rooms; paired bootstrap 10k, seed 20261013; the D160 standard report for every trained arm.

Sections below are filled as each step lands.

## Step 0 — free diagnostics

**0b(i) interior Helmholtz (I2):** the pre-registered reading DOES NOT HOLD (projected R0 ~23x FDTD's raw residual in every band). Confound flagged (D182(f)): the projection raises TRUTH's residual 21x; against projected truth, projected R0 is 1.08x / 1.13x / 1.11x. [physics_gates.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/physics_gates.md).

**0b(ii) Robin + gate G-BC (I4): FAILS on (2)** (unseen / in-family 1.25 projected, 0.92 raw; truth 0.004; R0 raw 6.95 = ~1,700x) -> PL1 not launched. Q-D: neither residual targets the unseen-family failure.

**0a latency by edit type (I1, F2):** cached-mode physics answers listener / source / material edits faster than the network (2 ms / 49 ms vs 10 ms / 32 s per listener / 64x64 grid); the network's case is the geometry edit (12 ms per listener vs 55.6 s modal re-solve, 49.4 s CPU FDTD). Update rate: network 101 Hz (601-bin path), cached physics 496 Hz. GPU FDTD ~0.1-0.3 s is an estimate. [latency.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/latency.md) · [F2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F2_latency.png).

**0c unit test (E3):** passes ([unit.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/dirzs/unit.json)).
## Step 1 — sparse-measurement grounding
## Step 2 — training arms (S2X, CTL, DIR, DIR-G, NMF, NMF-P)

**2c NMF kill read (I13): passes** after the NaN fix (D186): in-family off-lattice r at 25-100 Hz 0.796 at 10K (>= 0.70), 0.840 at 20K; R0 0.918 on the same rooms. Unseen r pending.

**2b interim @130K:** CTL HF 0.575, UF 0.464; DIR HF 0.576, UF 0.471 (R0 @120K: 0.582 / 0.468).

**2c feasibility (I9): GO.** S-b holds (24 modes: 0.984 in family at 25-100 Hz). Error budget: in-family >= 0.90 needs eigenfrequencies within ~2% and mode shapes within delta ~0.1; unseen >= 0.60 tolerates 5% / delta 0.4. [feasibility.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/nmf/feasibility.md).
## Suggestions S-1 to S-4
## Engineering

All default-off, Rule-8 tested (D182(e)): warm start + closed-form LR restart; the P4-9 step with a dedicated batch generator (CTL / DIR / DIR-G draw identical batches and jitters); DIR (per-ray readout) and DIR-G (finite-difference omni gradient) terms with a calibrated, checkpointed weight; the 601-bin inference path; per-ray output; the per-token override (M1); the projection's padded-grid return + exact band-limited evaluation; FDTD stencil reruns (exact); m1 modal solves on perturbed geometries; pooled training launches.
