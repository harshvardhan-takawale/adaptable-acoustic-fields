# Chunk P4-9 — physics as structure, directional output, sparse grounding: RESULTS (live)

**Status (2026-11-23, 00:50): pre-registered (D180-D181); Day 0 done (D182); I2, I9, I4, I1, I13, I3, I7, S-3, S-4, S-1, I14, I5, 150K reads (S2X interim, CTL final) in; extensions X1 (D189) and X1b (D192) in, X2 / X3 (D190, D191, D193) running; DIR E2 / C2 / C3 next.** Live log: [P4_9_STATUS_REPORT.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_9_STATUS_REPORT.md).

**Questions.** Q-A: where does the network beat physics (update speed, approximate inputs, sparse measurements)? Q-B: does physics help as structure (learned modes, NMF)? Q-C: can the model output direction, and does directional supervision change omni accuracy? Q-D: does any field-level physics residual target the unseen-family failure?

**Protocol** (unchanged): off-lattice everywhere (Rule 11); frozen sets = the 15 in-family test shapes and the 16 unseen rooms; paired bootstrap 10k, seed 20261013; the D160 standard report for every trained arm.

Sections below are filled as each step lands.

## Step 0 — free diagnostics

**0b(i) interior Helmholtz (I2):** the pre-registered reading DOES NOT HOLD (projected R0 ~23x FDTD's raw residual in every band). Confound flagged (D182(f)): the projection raises TRUTH's residual 21x; against projected truth, projected R0 is 1.08x / 1.13x / 1.11x. [physics_gates.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/physics_gates.md).

**0b(ii) Robin + gate G-BC (I4): FAILS on (2)** (unseen / in-family 1.25 projected, 0.92 raw; truth 0.004; R0 raw 6.95 = ~1,700x) -> PL1 not launched. Q-D: neither residual targets the unseen-family failure.

**0a latency by edit type (I1, F2):** cached-mode physics answers listener / source / material edits faster than the network (2 ms / 49 ms vs 10 ms / 32 s per listener / 64x64 grid); the network's case is the geometry edit (12 ms per listener vs 55.6 s modal re-solve, 49.4 s CPU FDTD). Update rate: network 101 Hz (601-bin path), cached physics 496 Hz. GPU FDTD ~0.1-0.3 s is an estimate. [latency.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/latency.md) · [F2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F2_latency.png).

**0d physics-input sensitivity (I3, F3):** under the same wrong inputs physics stays far ahead for small errors (unseen 25-100 Hz: exact 0.978, snapped 0.968, notch removed 0.760, beta x0.5 / x2 0.813 / 0.899 vs R0 0.465-0.469, S2 0.495-0.501); bounding box only: physics 0.400 = R0 0.408. [physics_sensitivity.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/physics_sensitivity.md) · [F3](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F3_sensitivity.png).

**0c unit test (E3):** passes ([unit.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/dirzs/unit.json)).
**0c directional zero-shot (I5):** E3 unit test passes (closed form exact; within 0.086 of 1/(2kr) over kr 1-30). R0 @120K, Re rho vs FDTD grad p / (jk), receivers >= 0.5 m from the source: projected-field gradient 0.943 / 0.844 / 0.803 in-family, 0.651 / 0.625 / 0.624 unseen; per-ray D 0.835 / 0.795 / 0.780 and 0.553 / 0.567 / 0.596; raw-FD 0.848 / 0.739 / 0.747 and 0.580 / 0.549 / 0.580. The expected < 0.3 is refuted. S2 vs R0 (projected): in-family -0.009 / -0.037 / -0.061, unseen +0.020 / +0.027 / +0.035. D188 fixes E2's bar as CTL's better omni readout per band (the literal reading lets R0 pass against itself by +0.095 / +0.106 / +0.056). [directional_zero_shot.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/directional_zero_shot.md).

## Step 1 — sparse-measurement grounding

**I7 (D183 readings):** P1 FAILS (M1 at K = 32: unseen 0.597 R0 / 0.635 S2 < 0.65; M2 0.807 / 0.839). P2 HOLDS (R0 given only the bounding box beats physics given it: M2-iii - M4-iii +0.392 / +0.274 / +0.203, CIs > 0). P3 FAILS (M1 - M5 at 200-300 Hz: +0.142 at K = 8, -0.008 at 16, -0.313 at 32). Geometry-free Helmholtz kernel ridge (M5) at K = 32: 0.952 / 0.868 / 0.796, above every network method; physics with exact geometry 0.977 / 0.971 / 0.959 at any K. Realism (20 dB + 2 cm) leaves the network fits unchanged and costs M5 0.07 at 200-300 Hz. [grounding.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/step1/grounding.md) · [F1](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F1_grounding.png).
**X1 (extension, D189): the network as a PRIOR for few-microphone reconstruction.** M6s = zero-shot network field as the mean of Step 1's Helmholtz KRR, with its scale fitted (0.8 s CPU). Unseen 16, pooled, R0:
* K = 8: 0.806 / 0.527 / 0.443, against KRR alone 0.753 / 0.389 / 0.311 and the network fine-tuned on the mics (M2, 58 s GPU) 0.735 / 0.452 / 0.439.
* K = 32: 0.967 / 0.903 / 0.826 (KRR 0.951 / 0.872 / 0.805).
* R1 HOLDS. R2 (beats both parts at every K) HOLDS on unseen, every band. In-family it ties the network alone at K = 8 (does not hold) and leads from K = 16.
* The prior is worth 10-14 microphones at K = 8 unseen, 20-25 in-family.
* Physics with the exact shape needs no microphones (0.977 / 0.970 / 0.959); the partial-geometry comparison is X1b (D192).

[netprior.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/netprior/netprior.md).
**X1b (D192): Q does NOT hold -- with only the bounding box, physics is as good a prior as the network.** Unseen, both priors + the same KRR:
* K = 8: M6-iii (network on box) 0.790 / 0.482 / 0.434 vs M7-iii (physics on box) 0.790 / 0.477 / 0.435 (tie).
* K = 16-64: physics ahead by 0.007-0.029 (CIs < 0 in most bands).
* In-family: physics ahead at every K (by up to 0.07).
* Even the exact-shape network prior leads the box physics prior only at K = 8.
* "Learned prior for reconstruction" is therefore not a role physics cannot fill (Q24).

[F8_netprior.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F8_netprior.png).
## Step 2 — training arms (S2X, CTL, DIR, DIR-G, NMF, NMF-P)

**2a S2X @150K (interim):** HF 0.510, UF 0.496 / 0.417 / 0.402 (S2 @120K 0.502 / 0.501; HF +0.007 [+0.002, +0.012], UF -0.005 [-0.009, -0.000]). The 240K test needs HF >= 0.56 AND UF >= 0.49; UF holds now, HF is 0.050 short. **2b CTL @150K (final):** HF 0.583, UF 0.467 / 0.376 / 0.373, flat vs R0 (HF +0.002 [-0.003, +0.006]); train-room fit 0.670 (R0 0.645).

**2c NMF N1 / N2 (I14): both FAIL.** NMF @30K 0.848 in-family / 0.451 unseen; NMF-P 0.828 / 0.438 (N1 bars 0.90 / 0.60). NMF-P - NMF on unseen -0.013 [-0.034, +0.007] (N2 needs >= +0.03). NMF - R0 on unseen -0.016 [-0.061, +0.031]; NMF - S2 -0.050 [-0.093, -0.002]. References: R0 0.918 / 0.468, S2 0.501 unseen, m1 0.978. Oracle decomposition (D187): with exact shapes + NMF k 0.544 / 0.484, NMF shapes + exact k 0.475 / 0.413, all exact 0.983 / 0.974 -> errors interact, neither half named. F5: in both median unseen rooms NMF has no counterpart of true mode 2 (the first width mode). [readings.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/nmf/readings.json) · [oracle.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/nmf/oracle.md) · [F5](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F5_nmf_modes.png).

**2c NMF kill read (I13): passes** after the NaN fix (D186): in-family off-lattice r at 25-100 Hz 0.796 at 10K (>= 0.70), 0.840 at 20K; R0 0.918 on the same rooms. Unseen r pending.

**2b interim @130K:** CTL HF 0.575, UF 0.464; DIR HF 0.576, UF 0.471 (R0 @120K: 0.582 / 0.468). **@140K:** CTL HF 0.582, UF 0.466; DIR HF 0.585, UF 0.471; DIR-G @130K HF 0.578, UF 0.473. The directional terms cost nothing on the omni field so far; E2 / C2 / C3 are read at 150K.

**2c feasibility (I9): GO.** S-b holds (24 modes: 0.984 in family at 25-100 Hz). Error budget: in-family >= 0.90 needs eigenfrequencies within ~2% and mode shapes within delta ~0.1; unseen >= 0.60 tolerates 5% / delta 0.4. [feasibility.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/nmf/feasibility.md).
## Suggestions S-1 to S-4

**S-3 (second source, 16 unseen rooms; descriptive):** m1 0.978 / 0.972 / 0.960 -> 0.978 / 0.969 / 0.960; R0 0.468 / 0.380 / 0.371 -> 0.172 / 0.077 / 0.055 (deltas -0.30 / -0.30 / -0.32, CIs well below 0); S2 0.501 / 0.419 / 0.404 -> 0.386 / 0.193 / 0.155. References vs the second-source truth: R0's original-source render 0.174 / 0.031 / 0.052; original-source truth 0.357 / 0.204 / 0.210. The network does not use its source input off the single trained position. [src2.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/src2.md).

**S-4 (sm_86/89 tinycudann, env `aaf_sm89`): parity HOLDS** -- R0 @120K dense in-family 0.9185 / 0.7009 / 0.5818 in both envs (delta 0.0000; max relative field difference 3.9e-6); training 0.761 -> 0.573 s/it (1.33x). Adoption from P4-10 (E8). [s4_parity.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/s4_parity.md).

**S-1 (D185) final @10K: NOT the renderer cap.** One-room training, 200-300 Hz on that room: 0.931 (lowest-fit room; R0 0.581) and 0.953 (median room; R0 0.645), +0.350 / +0.308 against the +0.10 bar; LSD 1.3 / 1.0 dB vs R0 4.0 / 3.8; converged (4K -> 10K +0.020 / +0.016 <= 0.03). The one-room models read 0.15 / 0.17 on the other 22 fit rooms. Capacity vs per-room exposure is open (Q25); R0's own fit trajectory (0.443 / 0.556 / 0.612 / 0.645 at 30-120K, CTL 0.659 at 140K) says exposure helps slowly.
## Engineering

All default-off, Rule-8 tested (D182(e)): warm start + closed-form LR restart; the P4-9 step with a dedicated batch generator (CTL / DIR / DIR-G draw identical batches and jitters); DIR (per-ray readout) and DIR-G (finite-difference omni gradient) terms with a calibrated, checkpointed weight; the 601-bin inference path; per-ray output; the per-token override (M1); the projection's padded-grid return + exact band-limited evaluation; FDTD stencil reruns (exact); m1 modal solves on perturbed geometries; pooled training launches.
