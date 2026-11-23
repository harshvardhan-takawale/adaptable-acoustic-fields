# P4-9 status report (live log) — physics as structure, directional output, sparse grounding

**Pre-registration:** D180 (spec, verbatim) + D181 (A1–A15, the manager's E1–E8, clarifications c1–c6) in [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md).
**Write-up:** [CHUNK_P4_9_RESULTS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_P4_9_RESULTS.md).

**How to read this log.** One dated entry per increment, newest first. Each entry opens with a plain-language headline that can be forwarded to collaborators as is, then the numbers (off-lattice, paired bootstrap 10k, seed 20261013) and raw links (each checked for HTTP 200).

**Planned increments** (targets; pre-emption can shift a read by up to a day, the order does not change):

| # | Increment | Target |
|---|---|---|
| I0 | D180 + D181 recorded; S2 @120K final recorded | Day 0 |
| I2 | Interior Helmholtz gate, raw vs projected (0b-i) | Day 0 |
| I1 | Latency by edit type (0a, E5) → F2 | Day 1 |
| I3 | Physics vs network under the same wrong inputs (0d, A1) → F3 | Day 1 |
| I4 | Robin residual + gate G-BC (0b-ii) | Day 1 |
| I5 | Directional zero-shot: omni-gradient vs per-ray (0c, E2/E3) | Day 1 |
| I7 | Sparse-measurement grounding at K = 32 (Step 1) | Day 2 |
| I8 | S2X @180K (pre-registered read 1); CTL @30K; S-3 (R0, m1) | Day 2 |
| I9 | NMF feasibility (S-a, S-b, error budget); S-1 single-room ceiling | Day 2 |
| I10 | Step 1 full sweep → F1; realism condition (S-2) | Day 3 |
| I11 | DIR / DIR-G @30K → E2 pass rule, C2, C3 → F4 | Day 3 |
| I12 | S2X @240K final + confirmation set → meeting model; F6 | Day 3 |
| I13–I14 | NMF / NMF-P: 10K kill read, N1 / N2 → F5 | Day 4 |
| I15 | Meeting pack (F1–F6 + manifest), primary table, results doc | Day 5 |

A training-trajectory figure (S2X, CTL, DIR, DIR-G vs step, with R0 / S2 references) is redrawn at every read.

---

## 2026-11-23 (02:30) — X2 (D190): after a geometry edit, a coarse physics re-solve beats the network's full-room render on BOTH speed and accuracy; the network keeps only the single-listener case -> F7

**Headline:** The network's one clear speed advantage was the geometry edit: change the room, and it renders a listener in 12 ms where a full physics re-solve took ~55 s. But that re-solve was built to reproduce the simulator exactly, not to be fast. Re-solved on a 10 cm grid with modes to 350 Hz, physics recomputes the WHOLE room in 0.4 s on one CPU, scoring r 0.97 / 0.91 / 0.76 on unseen rooms. The network takes 32 s on a GPU to render the room at 0.47 / 0.38 / 0.37. A 6 cm grid gives 0.98 / 0.95 / 0.89 in 1 s. So for full-room rendering in 2-D at <= 300 Hz, physics is faster AND more accurate. The network keeps one case: a single listener needed within ~0.3 s of an edit. Physics cost climbs steeply with frequency (0.3 s at 300 Hz, 3 s at 600 Hz, 62 s at 1 kHz, > 30 min at 2 kHz), so the network's speed case lies at high frequency and in 3-D, which it does not cover yet.

m1 (the FDTD's operator in modal form, exact beta) re-solved on a grid of spacing h with modes to f_max; polygon snapped by the grid. Mode shapes are bilinearly interpolated to the dense receivers. Off-lattice r (frozen scorer), mean over rooms. Timing = operator build + eigen-solve + field, median, 4 threads, one CPU node (legacy01, Xeon E5-2620 v3), shape 1000 (3,753 points). Selected rows (all 15 in [fastphys.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/fastphys/fastphys.md)):

| method after a geometry edit | 1 listener | full room | unseen 16 r | in-family 15 r |
|---|---|---|---|---|
| physics, 2 cm, modes to 600 Hz (= m1; I1's re-solve) | 46.9 s | 47.0 s | 0.978 / 0.972 / 0.960 | 0.988 / 0.976 / 0.960 |
| physics, 4 cm, 350 Hz | 2.6 s | 2.7 s | 0.978 / 0.959 / 0.918 | 0.988 / 0.967 / 0.933 |
| physics, 6 cm, 350 Hz | 0.89 s | 0.96 s | 0.977 / 0.945 / 0.886 | 0.987 / 0.949 / 0.891 |
| physics, 8 cm, 350 Hz | 0.46 s | 0.54 s | 0.975 / 0.924 / 0.826 | 0.986 / 0.945 / 0.855 |
| **physics, 10 cm, 350 Hz** | **0.31 s** | **0.38 s** | **0.972 / 0.913 / 0.759** | **0.984 / 0.931 / 0.774** |
| **network R0 @120K (A5000 GPU, I1)** | **0.012 s** | **32.1 s** | **0.468 / 0.380 / 0.371** | **0.918 / 0.701 / 0.582** |

- **Parity gate passes.** At h = 2 cm and 600 Hz the new path reproduces m1's per-room r to 8.8e-9 on all 31 rooms. The level offset is the same at every h, so the FDTD constant with coarse-grid normalisation is right.
- **D190(c) verdict:**
  - Full room: 14 of 15 configurations dominate the network (faster AND higher r in all 3 bands), unseen and in-family alike; the fastest is 10 cm / 350 Hz at 0.38 s. The exception is 2 cm / 600 Hz (47 s).
  - Single listener: no configuration dominates. 12 ms is below every solve.
  - Prediction D190(d) held.
- **Frequency scaling** (timing only, largest unseen room, h = c / (10 f_max), EPYC node): 300 Hz 0.31 s (88 modes); 600 Hz 3.2 s (336); 1 kHz 62 s (904); 2 kHz > 30 min (cap). Mode count grows as f^2 in 2-D (f^3 in 3-D).
- **Hardware is not matched, and the mismatch favours the network.** Physics ran on an old CPU (this node's 2 cm re-solve is 46.9 s against I1's 55.6 s on legacy42); the network ran on a GPU. The timing node differs from I1's because that node type was fully booked (deviation noted).
- **What this changes:** F2's sentence "the network's case is the geometry edit" is amended to "the single-listener geometry edit, under ~0.3 s". Combined with X1b, the 2-D case for the network at <= 300 Hz is narrow. The project's stated band (0-2 kHz) and 3-D are where per-edit physics stops being cheap.

Links: [F7_fast_physics.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F7_fast_physics.png) · [fastphys.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/fastphys/fastphys.md) · [fastphys.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/fastphys/fastphys.json) · [p4_9_fast_physics.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_9_fast_physics.py)

## 2026-11-23 (01:10) — X3a (D191, D193): the network cannot move the source, in known shapes too — but source/listener swaps in training largely fix it, with no new simulation

**Headline:** Moving the sound source breaks the network even in room shapes it knows well: accuracy at 25-100 Hz falls from 0.92 to 0.23. That is worse than not moving the source at all (0.31). The network is also not "reciprocal". Swapping source and listener leaves the true sound field unchanged, yet it collapses the network's prediction (0.92 -> 0.17); it predicts almost entirely from the listener's position. An older model trained with source/listener swaps and mirror images (AFAM) keeps 0.50 / 0.40 / 0.41 at the moved source, against 0.23 / 0.06 / 0.05 for its otherwise identical twin trained without them (G2), and beats "not moving" in every band. This needs no new simulation. RCP (the meeting model R0 + swaps, CTL's exact batches) is training; its reads are at 130K / 140K / 150K (~02:40 / ~04:50 / ~07:00).

**New set (D191(b)(ii)):** the 15 in-family test shapes re-simulated at a second source (S-3's rule; geometry proven by the same rerun check), same dense receivers. Raw off-lattice r per band (S-3's scorer); original source -> second source; drop [95% CI]:

| model | in-family 15: 25-100 / 100-200 / 200-300 Hz | unseen 16: 25-100 / 100-200 / 200-300 Hz |
|---|---|---|
| R0 @120K | 0.918 -> 0.227 / 0.701 -> 0.067 / 0.582 -> 0.040 | 0.468 -> 0.172 / 0.380 -> 0.077 / 0.371 -> 0.055 |
| CTL @150K | 0.920 -> 0.221 / 0.708 -> 0.068 / 0.583 -> 0.048 | 0.467 -> 0.176 / 0.376 -> 0.068 / 0.373 -> 0.052 |
| S2 @120K | 0.908 -> 0.487 / 0.652 -> 0.182 / 0.502 -> 0.141 | 0.501 -> 0.386 / 0.419 -> 0.193 / 0.404 -> 0.155 |
| G2 @60K (no augmentation) | 0.774 -> 0.226 / 0.475 -> 0.056 / 0.309 -> 0.053 | 0.458 -> 0.165 / 0.312 -> -0.001 / 0.262 -> 0.022 |
| **AFAM @60K (= G2 + D4 flips + swaps)** | **0.806 -> 0.499 / 0.521 -> 0.398 / 0.405 -> 0.412** | **0.444 -> 0.329 / 0.361 -> 0.213 / 0.351 -> 0.221** |
| "ignore the move" (original-source truth vs second-source truth) | 0.311 / 0.211 / 0.247 | 0.357 / 0.204 / 0.210 |
| physics m1 (exact geometry) | -- | 0.978 -> 0.978 / 0.972 -> 0.969 / 0.960 -> 0.960 |

- **X3a, AFAM - G2 at the second source (paired):**
  - in-family **+0.273 [+0.213, +0.330] / +0.342 [+0.309, +0.376] / +0.359 [+0.327, +0.391]**;
  - unseen +0.163 [+0.075, +0.250] / +0.213 [+0.138, +0.287] / +0.199 [+0.111, +0.282].
  - At the original source the gap is small: in-family +0.032 / +0.046 / +0.096; unseen -0.014 / +0.049 / +0.089.
- **Drops:**
  - AFAM loses -0.307 / -0.123 / +0.007 in-family, against G2's -0.548 / -0.419 / -0.256.
  - The prediction "at most half of G2's drop" holds in 5 of 6 band-sets; in-family 25-100 Hz it is 56%.
- **Caveat:** AFAM's augmentation is D4 flips (which also move the source, mirrored) plus swaps, so X3a cannot attribute the gain to swaps alone. RCP isolates the swaps (p 0.25, nothing else changed against CTL).
- **Reciprocity gap (D193):** the swapped render scored against the SAME truth, off-lattice interior receivers.

  | model | set | normal | swapped | swapped vs normal |
  |---|---|---|---|---|
  | R0 @120K | in-family | 0.918 / 0.701 / 0.582 | 0.167 / 0.050 / 0.027 | 0.172 / 0.070 / 0.030 |
  | R0 @120K | unseen | 0.468 / 0.380 / 0.371 | 0.122 / 0.054 / 0.027 | 0.189 / 0.082 / 0.038 |
  | CTL @150K | in-family | 0.920 / 0.708 / 0.583 | 0.170 / 0.050 / 0.034 | 0.173 / 0.069 / 0.033 |

  R0 is far from reciprocal: exchanging source and listener changes its field almost completely. Together with S-3 (R0's moved-source render is no better than its original-source one), this says R0 has learned the field as a function of the listener position for the one source each room was trained with.
- **Pre-registered RCP readings (D191(e)), at 150K vs CTL @150K:**
  - S1: in-family second source, 25-100 Hz, >= +0.10 with CI > 0;
  - S2: unseen, >= +0.05;
  - C2: no harm at the original source.
  - RCP runs at 0.78 s/it on tron; every training row is swappable (no wall nodes).

Links: [x3_read.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/x3_read.md) · [src2_test.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/src2_test.md) · [recip_gap.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/recip_gap.md) · [P4_9_RCP.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_9_RCP.yaml)

## 2026-11-23 (01:00) — X1b (D192): when physics gets the same partial information, it is as good a prior as the network — Q does NOT hold -> F8

**Headline:** X1 showed the network is a useful starting guess for reconstructing a room from a few microphones. The fair question is whether physics can do the same job. We gave both the same partial information, only the room's bounding box. A quick physics solve on that box, used as the starting guess, is as good as the network at 8 microphones and slightly better from 16 on (unseen rooms); in known room shapes it is better at every count. Even the network given the EXACT shape beats the box-only physics guess only at 8 microphones. So "a learned prior for reconstruction" is a role cheap physics already fills. This does not answer Q24 in the network's favour.

Unseen 16, pooled, r 25-100 / 100-200 / 200-300 Hz (both priors combined with the same KRR, M6s's form and lam):

| method (bounding box only) | K = 8 | K = 16 | K = 32 | K = 64 |
|---|---|---|---|---|
| network on the box, no mics (M0-iii) | 0.420 / 0.336 / 0.389 | same | same | same |
| physics on the box, beta fitted (M4-iii) | 0.412 / 0.321 / 0.370 | same | same | same |
| mics only (M5) | 0.753 / 0.389 / 0.311 | 0.910 / 0.688 / 0.490 | 0.951 / 0.872 / 0.805 | 0.963 / 0.924 / 0.917 |
| **network-on-box prior + mics (M6-iii)** | 0.790 / 0.482 / 0.434 | 0.931 / 0.739 / 0.593 | 0.967 / 0.900 / 0.827 | 0.977 / 0.944 / 0.924 |
| **physics-on-box prior + mics (M7-iii)** | 0.790 / 0.477 / 0.435 | 0.940 / 0.754 / 0.600 | 0.975 / 0.923 / 0.856 | 0.985 / 0.963 / 0.942 |

- **Q (M6-iii - M7-iii, unseen, CI > 0 in >= 2 bands at K = 8 and 16): does NOT hold.**
  - K = 8: +0.000 [-0.004, +0.005] / +0.004 [-0.008, +0.016] / -0.001 [-0.023, +0.018] (tie).
  - K = 16: -0.009 [-0.011, -0.007] / -0.016 [-0.024, -0.007] / -0.007 [-0.020, +0.005].
  - K = 32: -0.008 / -0.023 / -0.029, all CIs < 0.
- **In-family 15: physics-on-box is the better prior at every K,** -0.018 to -0.070 at K = 8 (CIs < 0); M7-iii 0.837 / 0.644 / 0.571 vs M6-iii 0.819 / 0.587 / 0.501.
- **Descriptive: the network given the EXACT shape (X1's M6s) against physics given only the box (M7-iii).** The network leads only at K = 8:
  - unseen: +0.015 [-0.004, +0.035] / +0.050 [+0.022, +0.080] / +0.008 [-0.019, +0.032];
  - in-family: +0.089 / +0.087 / +0.037.
  - From K = 32 the box physics is ahead: unseen -0.008 / -0.019 / -0.030; in-family -0.003 / -0.028 / -0.042.
- **Prediction (D192(d)) was wrong.** I expected the network to win at K = 8 at 100-300 Hz (Step 1's P2 had the network well ahead when each method fitted alone). Zero-shot the two are equally poor (0.42 vs 0.41 unseen). Once measurements correct them, physics's errors are the easier kind for the Helmholtz kernel to fix.
- **Reading for Q24:** X1 + X1b together say the network can act as a reconstruction prior, but not one that does better than a crude modal solve given the same geometry knowledge. The remaining cases for the network are those physics cannot reach cheaply: interactive geometry editing (X2 tests even that) and settings where the physics model itself is wrong (measured rooms, unknown materials), which the simulated data here cannot test.

Links: [F8_netprior.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F8_netprior.png) · [netprior.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/netprior/netprior.md) · [netprior.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/netprior/netprior.json)

## 2026-11-23 (00:50) — X1 (extension, D189): the network as a PRIOR for few-microphone reconstruction — it cuts the microphones needed, and beats fine-tuning the network

**Headline:** We tested a new use for the network. Some microphones are measured in a room, and the sound field everywhere else is reconstructed from them. The network's prediction serves as the starting guess and the measurements correct it with a standard physics-based interpolation (kernel ridge regression with Helmholtz kernels). In unseen rooms, 8 microphones with the network's guess are worth about 10-14 microphones without it; in known room shapes about 20-25. With 32 microphones, unseen-room accuracy reaches r 0.97 / 0.90 / 0.83 (25-100 / 100-200 / 200-300 Hz). It takes 0.8 s on a CPU, and it beats fine-tuning the network on the same microphones (58 s on a GPU) at every microphone count. **Caveat:** if the exact room shape is known, a physics solve needs no microphones at all (0.98 / 0.97 / 0.96). The fair test gives physics and the network the same partial information (only the bounding box); it is running (X1b, D192).

Unseen 16 rooms, off-lattice, mean over rooms (seeds 0-2), pooled strata. M0 = network alone; M5 = microphones only (Step 1's Helmholtz KRR); M2 = network fine-tuned on the microphones (Step 1); **M6s = network prior + KRR** (primary). r 25-100 / 100-200 / 200-300 Hz:

| method | K = 8 | K = 16 | K = 32 | K = 64 |
|---|---|---|---|---|
| R0 alone (M0) | 0.485 / 0.408 / 0.408 | same | same | same |
| mics only (M5) | 0.753 / 0.389 / 0.311 | 0.910 / 0.688 / 0.490 | 0.951 / 0.872 / 0.805 | 0.963 / 0.924 / 0.917 |
| R0 fine-tuned on mics (M2, 58 s GPU) | 0.735 / 0.452 / 0.439 | 0.782 / 0.559 / 0.519 | 0.811 / 0.605 / 0.578 | 0.829 / 0.630 / 0.605 |
| **R0 prior + mics (M6s, 0.8 s CPU)** | **0.806 / 0.527 / 0.443** | **0.933 / 0.763 / 0.600** | **0.967 / 0.903 / 0.826** | **0.977 / 0.943 / 0.924** |
| physics, exact shape, no mics (M3) | 0.977 / 0.970 / 0.959 | same | same | same |

- **R1 (the prior adds to KRR), M6s - M5: HOLDS.** K = 8: +0.053 [+0.030, +0.079] / +0.138 [+0.107, +0.171] / +0.132 [+0.109, +0.158]. K = 16: +0.023 [+0.015, +0.030] / +0.074 [+0.064, +0.086] / +0.111 [+0.093, +0.129].
- **R2 (beats BOTH the network alone and KRR alone at every K): HOLDS on unseen, every band at every K.**
  - K = 8, against the better part: +0.053 [+0.030, +0.079] vs M5 / +0.119 [+0.102, +0.135] vs M0 / +0.035 [+0.025, +0.046] vs M0.
  - K = 64, against M5: +0.014 / +0.019 / +0.006, all CIs > 0.
  - S2 as the prior gives the same picture (R2 holds on unseen).
- **R2 does NOT hold in-family.** At K = 8 the combination only ties the network alone (in-family R0 0.923 / 0.720 / 0.615): +0.003 [-0.005, +0.012] / +0.010 [+0.002, +0.020] / -0.007 [-0.014, +0.002]. From K = 16 it is ahead: +0.036 / +0.101 / +0.087 vs M0; at K = 32 +0.035 / +0.052 / +0.047 vs M5.
- **Microphones the prior is worth** (the K at which mics-only matches M6s at K = 8): unseen 10.1 / 11.0 / 13.4; in-family 25.1 / 20.1 / 21.5.
- **Against the pre-registered prediction (D189(d)):**
  - R1 held, as predicted.
  - R2 on unseen at K >= 32 was predicted to FAIL at 200-300 Hz; it holds (+0.021 [+0.016, +0.027] at K = 32). The fitted prior scale lets KRR down-weight a weak prior without paying for it.
  - R2 in-family at K = 8 was predicted to hold; it ties.
  - The fixed-mean variant (M6) falls below KRR alone on unseen at K = 64 at 100-300 Hz (0.909 / 0.865 vs 0.924 / 0.917), as predicted. This is why the scaled form is primary.
- **lam** (E7 calibration on the 4 T / DN rooms): 0.01 for M6 and M6s, the same value M5 chose.

Running now (extensions, all pre-registered before any number):
- **X1b (D192):** network vs physics as the prior when only the bounding box is known.
- **X2 (D190):** a coarse, fast physics solve as the geometry-edit baseline (the network's one latency case).
- **X3 (D191, D193):** moving the source without new simulation, via reciprocity swaps. Zero-training pre-check AFAM vs G2; arm RCP queued on tron; a new in-family second-source set (15 rooms) built.

Links: [netprior.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/netprior/netprior.md) · [netprior.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/netprior/netprior.json) · [calibration.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/netprior/calibration.json) · [p4_9_netprior.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_9_netprior.py)

## 2026-11-22 (late night, 5) — 150K reads: S2X @150K (interim) and CTL @150K (final) -> F0

**Headline:** The unseen-room specialist S2 is being trained longer (S2X). After 30K extra steps it is slightly better on known shapes (0.510, up from 0.502) and unchanged on unseen rooms (0.496). To become the meeting model at 240K it needs 0.56 on known shapes. On its current pace that looks out of reach, but the first pre-registered read is at 180K. The plain continuation of R0 (CTL) finished flat: more steps on the same corpus do not move either number.

| model (read) | in-family test r 25-100 / 100-200 / 200-300 Hz (HF = 200-300) | unseen UF 25-100 / 100-200 / 200-300 Hz | train-room fit 200-300 Hz |
|---|---|---|---|
| R0 @120K (meeting model) | 0.918 / 0.701 / **0.582** | **0.468** / 0.380 / 0.371 | 0.645 |
| S2 @120K | 0.908 / 0.652 / **0.502** | **0.501** / 0.419 / 0.404 | 0.523 |
| **S2X @150K** | 0.906 / 0.656 / **0.510** | **0.496** / 0.417 / 0.402 | 0.533 |
| **CTL @150K** (final) | 0.920 / 0.708 / **0.583** | **0.467** / 0.376 / 0.373 | 0.670 |

- **S2X @150K - S2 @120K:** HF +0.007 [+0.002, +0.012]; UF -0.005 [-0.009, -0.000] / -0.003 / -0.002.
- **S2X @150K - R0 @120K:** HF -0.072 [-0.082, -0.062]; UF +0.029 [+0.016, +0.043] / +0.037 / +0.031.
- **S2X's pre-registered test at 240K** (2a): HF >= 0.56 AND UF >= 0.49.
  - UF holds now (0.496).
  - HF needs +0.050 more. The 120K -> 150K slope is +0.008 per 30K, restart dip included. At that slope HF would reach about 0.53 at 240K. This is an extrapolation, labelled, and not a verdict.
  - Read 1 at 180K shows whether the slope steepens once the warm-up / restart transient is past.
- **CTL @150K - R0 @120K:** HF +0.002 [-0.003, +0.006]; UF -0.001 [-0.003, +0.002] / -0.004 / +0.002. CTL is the reference for DIR / DIR-G, whose E2 / C2 / C3 reads follow.
- Train-room fit at 200-300 Hz keeps rising (CTL 0.645 -> 0.670 over 30K) while held-out HF does not. More steps fit the training rooms better without transferring. This is relevant to Q25: exposure buys fit, not generalisation.

Links: [F0_trajectory.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F0_trajectory.png) · [S2X_150K unseen](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/unseen/S2X_150K.json) · [CTL_150K unseen](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/unseen/CTL_150K.json)

## 2026-11-22 (night, 4) — I5: directional output zero-shot — R0 already predicts sound direction (0.94 / 0.84 / 0.80 in family), far above the spec's expected < 0.3; D188 fixes E2's bar before the 150K read

**Headline:** Without ever being trained on direction, the current model already predicts which way sound travels through each point of the room. Its directional estimate correlates 0.94 (below 100 Hz) to 0.80 (200-300 Hz) with the true acoustic gradient in known room shapes, and 0.62-0.65 in unseen rooms. The spec expected below 0.3. Direction mostly comes for free from a good omnidirectional field. The directional training arms, read at 150K, must beat this, not zero.

- **Unit test (E3): passes.** The per-ray readout's error on an analytic point source equals the closed form |H1/(jH0) - 1| and stays within 0.086 of 1/(2kr) over kr 1-30. The finite-difference gradient reproduces the analytic one to 9.3e-4.
- **Score (c2):** Re of the normalised complex correlation with FDTD grad p / (jk), per bin, over off-lattice receivers >= 0.5 m from the source, both components, band means. 1.0 is perfect by construction.

  | R0 @120K readout | in-family (15) 25-100 / 100-200 / 200-300 Hz [95% CI] | unseen (16) |
  |---|---|---|
  | per-ray D | 0.835 [0.824, 0.846] / 0.795 / 0.780 | 0.553 [0.489, 0.621] / 0.567 / 0.596 |
  | raw-field gradient (FD, h = 4 cm) | 0.848 [0.832, 0.862] / 0.739 / 0.747 | 0.580 / 0.549 / 0.580 |
  | **projected-field gradient (spectral)** | **0.943** [0.935, 0.950] / **0.844** / **0.803** | **0.651** [0.591, 0.715] / **0.625** / **0.624** |
  | S2 @120K, projected-field gradient | 0.935 / 0.808 / 0.742 | 0.670 / 0.652 / 0.659 |

- **Contrasts (paired bootstrap):**
  - Projected minus raw gradient, R0 in-family: +0.095 [+0.084, +0.108] / +0.106 [+0.089, +0.124] / +0.056 [+0.050, +0.063]. Unseen: +0.071 / +0.076 / +0.044, all CIs > 0. The projection cleans the field's fine-scale errors (W4), and the gradient is where they show.
  - Per-ray minus raw gradient, in-family: -0.013 [-0.028, +0.004] / +0.056 / +0.033.
  - S2 - R0, projected gradient: in-family -0.009 / -0.037 / -0.061; unseen +0.020 / +0.027 / +0.035 (CIs > 0). This is the same in-family / unseen trade-off as the omni scores.
- **The spec's labelled expectation (< 0.3) is refuted.** Omni supervision constrains the field's spatial structure well enough that its gradient carries first-order direction.
- **D188 (recorded before any 150K directional number).** E2's rule compares DIR / DIR-G's best readout with "CTL's omni-gradient readout". Read literally (bar = raw gradient only), R0 itself passes against its own bar in all 3 bands, through the projection alone. The bar is therefore CTL's BETTER omni-field readout per band (raw or projected), and the literal reading is reported beside it. The manager may override before the read.
- Ops: the first render job hit its 5 h limit with 61 / 62 room renders done. It resumed from cache.

Links: [directional_zero_shot.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/directional_zero_shot.md) · [directional_zero_shot.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/directional_zero_shot.json) · [contrasts](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/directional_zero_shot_contrasts.json) · [DECISIONS.md (D188)](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md)

## 2026-11-22 (night, 3) — trajectory read: CTL / DIR @140K, DIR-G @130K (interim; no verdict) -> F0

**Headline:** Teaching the network to output sound direction so far costs nothing on its ordinary (omnidirectional) field. After 20K extra steps, the directional arms match the plain continuation (CTL) and the starting model R0 to within 0.01 everywhere. Whether they gain DIRECTION accuracy is read at 150K.

| arm (read) | in-family test r 25-100 / 100-200 / 200-300 Hz (HF = 200-300) | unseen UF 25-100 / 100-200 / 200-300 Hz | train-room fit 200-300 Hz |
|---|---|---|---|
| R0 @120K (start) | 0.918 / 0.701 / **0.582** | **0.468** / 0.380 / 0.371 | 0.645 |
| CTL @140K | 0.918 / 0.706 / **0.582** | **0.466** / 0.377 / 0.370 | 0.659 |
| DIR @140K | 0.916 / 0.709 / **0.585** | **0.471** / 0.370 / 0.365 | 0.665 |
| DIR-G @130K | 0.918 / 0.700 / **0.578** | **0.473** / 0.384 / 0.377 | 0.641 |
| S2 @120K (reference) | 0.908 / 0.652 / 0.502 | 0.501 / 0.419 / 0.404 | 0.523 |

- Interim reads, no pass rules. E2's pass rule, C2 and C3 are read on the 150K checkpoints (I11).
- The restart dip at 130K has recovered by 140K for CTL and DIR.
- Train-room fit at 200-300 Hz keeps rising slowly (0.645 -> 0.659 / 0.665). This is the exposure curve in Q25.
- S2X (S2 continued to 240K) is at about 146K; its first pre-registered read is at 180K, with an interim at 150K.
- The duplicate CTL 150K watcher was stopped (PID 846444), so each 150K eval launches once.

Links: [F0_trajectory.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F0_trajectory.png)

## 2026-11-22 (late night, 2) — I14: NMF / NMF-P fail N1 and N2 -> F5; S-1 final: sharing one model across rooms, not the renderer, caps 200-300 Hz

**Headline:** The physics-structured network (NMF) predicts a room's resonant modes and builds the sound field from them. It does not beat the plain network: 0.85 in known room shapes (plain network 0.92) and 0.45 in unseen ones (0.47). Adding physics losses on its modes changes nothing. The pictures show one way it fails: in both unseen rooms shown, it gets the first mode right but has no counterpart for the second (the first mode across the room's width). Each later predicted mode resembles the true mode one place further on. Separately, the plain network trained on ONE room fits it at 0.93-0.95 at 200-300 Hz, against 0.58-0.65 for the model shared across 1000 rooms. The renderer can represent these fields; sharing is what limits them.

- **N1 / N2 (D180, D184; 25-100 Hz, off-lattice; 15 in-family / 16 unseen dense rooms):**
  - **N1 fails** (needs in-family >= 0.90 AND unseen >= 0.60): NMF @30K **0.848 / 0.451**; NMF-P @30K **0.828 / 0.438**.
  - **N2 fails:** NMF-P - NMF on unseen **-0.013 [-0.034, +0.007]** (needs >= +0.03, CI > 0). The physics losses on the modes do not help.
  - Against the networks on unseen rooms: NMF - R0 -0.016 [-0.061, +0.031] (NMF ahead in 4 / 16 rooms); NMF - S2 -0.050 [-0.093, -0.002].
  - Reference levels (Rule 12): R0 0.918 in-family / 0.468 unseen; S2 0.501 unseen; m1 with exact inputs 0.978.
  - Trajectory (NMF; in-family / unseen): 10K 0.796 / 0.424 -> 20K 0.840 / 0.450 -> 30K 0.848 / 0.451. Unseen has plateaued.
  - Eigenfrequency MAE, first 10 modes (in-family / unseen): NMF 5.6 / 6.1 Hz, NMF-P 6.9 / 6.8 Hz. Subspace overlap: NMF 0.91 / 0.86, NMF-P 0.88 / 0.86.
- **Oracle decomposition (D187; recorded after N1 / N2, before computing; descriptive).** Each half of the NMF is replaced in turn by m1's exact eigenpairs, 24 modes paired by sort order.

  | rooms | NMF | true shapes + NMF k | NMF shapes + true k | all true (24 modes) |
  |---|---|---|---|---|
  | in-family | 0.848 | 0.544 | 0.475 | 0.983 |
  | unseen | 0.451 | 0.484 | 0.413 | 0.974 |

  - Either substitution LOWERS the in-family field.
  - **D187(c) reading: the errors interact; neither half is named as the bottleneck.** The NMF's eigenfrequencies and shapes are co-adapted: they work as a pair but not with the true other half. NMF-P reads the same way (oracle.md).
- **F5 (each family's median-r unseen room, NMF @30K).** True modes 1-6 sit above predicted modes 1-6, both sorted by frequency. Each predicted mode's title also names its best-correlated true mode among the first 10.
  - Mode 1 is right in both rooms (|r| 0.97 / 0.98).
  - True mode 2 (U 32.9 Hz, C3 38.9 Hz; the first mode across the width) has no counterpart.
  - U: predicted modes 2-4 match true modes 3-5 at |r| 0.95 / 0.86 / 0.82, one place on.
  - C3: the shapes past mode 1 are weaker (|r| 0.62-0.78).
  - Seen in both rooms shown; whether it is systematic across rooms is not yet tested.
- **S-1 final (D185; read at 10K): NOT the renderer cap.**

  | room | R0 @120K (1000 rooms) | S-1 @10K (this room only) | S-1 - R0 at 200-300 Hz |
  |---|---|---|---|
  | lowest 200-300 Hz fit | 0.952 / 0.712 / **0.581**; LSD 4.0 dB | 0.996 / 0.964 / **0.931**; 1.3 dB | **+0.350** |
  | median fit | 0.955 / 0.779 / **0.645**; 3.8 dB | 0.997 / 0.971 / **0.953**; 1.0 dB | **+0.308** |

  - Both clear the +0.10 bar.
  - Converged: the 4K -> 10K rise is +0.020 / +0.016, within 0.03, so there is no extension.
  - The one-room models read 0.15 / 0.17 at 200-300 Hz on the other 22 fit rooms: they specialise, as they should.
  - S-1 cannot separate shared capacity from per-room exposure (about 83x R0's per room).
  - R0's own fit trajectory (0.443 / 0.556 / 0.612 / 0.645 at 30-120K; CTL 0.659 at 140K) says exposure helps but slowly. This is Q25 for the manager: OPEN_QUESTIONS.md.

Links: [readings.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/nmf/readings.json) · [oracle.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/nmf/oracle.md) · [F5_nmf_modes.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F5_nmf_modes.png) · [S1lo_10K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/train_offlat_fit/S1lo_10K.json) · [S1med_10K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/train_offlat_fit/S1med_10K.json) · [OPEN_QUESTIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md)

## 2026-11-22 (late night) — S-3: the network does not generalise to a moved source; S-4: the faster GPU build is exact and 1.33x faster; S-1 interim: one-room training reaches 0.91-0.94 at 200-300 Hz

**Headline:** Every room the network has trained on had its sound source at the same spot. When we move the source in the unseen test rooms, the network's accuracy below 100 Hz drops from 0.47 to 0.17. That is no better than leaving its old-source answer unchanged, and worse than reusing the true field from the old position (0.36). Physics does not care where the source is (0.98 at both positions). Separately, a recompiled GPU library gives identical results and trains 1.33x faster. In an early read (pre-registered read pending), the same network trained on one room alone reaches 0.91-0.94 at 200-300 Hz on that room, against 0.58-0.65 for the shared 1000-room model. So the renderer is not what limits 200-300 Hz.

- **S-3 (D181 E8, D182(b); descriptive, no pass rule): a second source position, 16 frozen unseen rooms.**
  - Setup: same dense receivers; off-lattice (Rule 11); raw field; second minus original source, paired bootstrap over rooms.
  - The second source is (L - 0.5, 1.6), or (L/2, 1.0) when that is blocked; the original is (0.5, 1.6).

  | field | 25-100 Hz | 100-200 Hz | 200-300 Hz |
  |---|---|---|---|
  | m1 physics | 0.978 -> 0.978; +0.001 [-0.001, +0.003] | 0.972 -> 0.969; -0.003 [-0.006, -0.000] | 0.960 -> 0.960; -0.000 [-0.003, +0.003] |
  | R0 @120K | 0.468 -> **0.172**; -0.295 [-0.390, -0.205] | 0.380 -> **0.077**; -0.303 [-0.347, -0.259] | 0.371 -> **0.055**; -0.316 [-0.363, -0.270] |
  | S2 @120K | 0.501 -> **0.386**; -0.115 [-0.182, -0.041] | 0.419 -> **0.193**; -0.226 [-0.264, -0.186] | 0.404 -> **0.155**; -0.249 [-0.275, -0.220] |

  - LSD (dB): R0 5.3 / 5.7 / 5.4 -> 8.5 / 7.7 / 6.9; S2 5.1 / 5.5 / 5.2 -> 7.1 / 7.3 / 6.0; m1 0.9-1.1 at both sources. The projected field reads the same (in `src2.md`).
  - **Reference fields (Rule 12), scored against the second-source truth:**
    - R0's *original*-source render: 0.174 / 0.031 / 0.052.
    - The original-source *truth*, a field that ignores the move: 0.357 / 0.204 / 0.210.
    - R0 at the new source is no better than not moving the source at 25-100 and 200-300 Hz. It is below the "ignore the move" truth in every band.
    - S2 is the only network field to reach that reference, at 25-100 Hz (0.386 vs 0.357).
  - By family (25-100 Hz, original -> second): R0 U 0.481 -> 0.158, C3 0.454 -> 0.186; S2 U 0.510 -> 0.417, C3 0.491 -> 0.355.
  - **Reading:**
    - Every corpus fixes the source at (0.5, 1.6), so the network's source input has never varied in training.
    - F2's source-move latency row is now hatched: latency only, accuracy failing.
    - Caveat: these rooms are already out of family, so two shifts compound. An in-family second-source test would isolate the source shift.
    - The remedy is source-diverse corpora (S-5, next round).
- **S-4 (D181 E8, c5): tinycudann built for sm_86/89 (separate env `aaf_sm89`; `aaf` untouched). Parity HOLDS.**
  - R0 @120K on the 15 in-family dense rooms reads 0.9185 / 0.7009 / 0.5818 in both envs. Delta +0.0000 in every band (bar 0.003); max relative field difference 3.9e-6.
  - Training throughput (R0's recipe, 300 steps, same A5000): 0.761 -> 0.573 s/it (**1.33x**).
  - Adoption is a P4-10 decision (E8). No P4-9 arm switches environments.
- **S-1 interim (D185; the pre-registered read is at 10K, and its fit jobs are queued):** R0's recipe trained from scratch on ONE training room, 4K steps, scored on that room's off-lattice receivers.

  | room (D185) | R0 @120K, shared over 1000 rooms | S-1 @4K, this room only |
  |---|---|---|
  | lowest 200-300 Hz fit (L6.16_W4.56_d0.34_w0.48) | 0.952 / 0.712 / **0.581**; LSD 200-300 4.0 dB | 0.993 / 0.952 / **0.911**; 1.5 dB |
  | median fit (L5.74_W4.78_d0.62_w2.32) | 0.955 / 0.779 / **0.645**; 3.8 dB | 0.995 / 0.957 / **0.937**; 1.3 dB |

  - Already **+0.33 / +0.29** at 200-300 Hz, against D185's +0.10 bar for "not the renderer cap". The 10K read and the convergence check (a rise of more than 0.03 from 4K -> extend to 20K) follow.

Links: [src2.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/src2.md) · [F2_latency.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F2_latency.png) · [latency.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/latency.md) · [s4_parity.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/s4_parity.md) · [S1lo_4K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/train_offlat_fit/S1lo_4K.json) · [S1med_4K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/train_offlat_fit/S1med_4K.json)

## 2026-11-22 (night) — I7: sparse-measurement grounding (Step 1) -> F1; P1 fails, P2 holds, P3 fails; NMF-P passes its kill read

**Headline:** With 32 microphone measurements in an unseen room, briefly fine-tuning part of the network lifts its accuracy from 0.49 to 0.81 below 100 Hz. Even when the network only knows the room's bounding box it reaches 0.80, where physics given the same box gets 0.41. But a simple geometry-free method, wave-equation interpolation between the measurements, does better still once it has 16 or more measurements (0.95 at 32). The network's edge over it exists only with very few measurements (8) and above 100 Hz.

- **Pre-registered readings (D183(e); unseen 16, clean):**
  - **P1 does not hold.** M1 (fit the room code) at K = 32 reaches **0.597** (R0) and **0.635** (S2), below the 0.65 bar. M2 (code + FiLM heads), the method P1 did not name, reaches **0.807** / **0.839**.
  - **P2 holds.** Given only the bounding box, R0 with 32 measurements beats physics fitted to the same box.
    - M2-iii minus M4-iii: +0.392 [+0.356, +0.428] / +0.274 [+0.241, +0.307] / +0.203 [+0.174, +0.232].
    - M1-iii minus M4-iii: +0.158 / +0.109 / +0.086. All CIs are above 0.
  - **P3 does not hold.** M1 minus M5 at 200-300 Hz:

    | K | R0 | S2 |
    |---|---|---|
    | 8 | **+0.142** [+0.113, +0.169] | +0.112 [+0.091, +0.131] |
    | 16 | -0.008 [-0.044, +0.026] | -0.036 [-0.066, -0.010] |
    | 32 | **-0.313** [-0.345, -0.284] | -0.344 [-0.368, -0.322] |

    Absolute levels at 200-300 Hz (Rule 12): M1 0.453 / 0.482 / 0.492 and M5 0.311 / 0.490 / 0.805 at K = 8 / 16 / 32.
- **Headline table (K = 32, all non-drawn off-lattice receivers, 25-100 / 100-200 / 200-300 Hz; median fit time per room):**

  | method | R0 | S2 | fit time |
  |---|---|---|---|
  | M0 zero-shot | 0.468 / 0.380 / 0.371 | 0.501 / 0.419 / 0.404 | -- |
  | M1 fit room code | 0.597 / 0.468 / 0.459 | 0.635 / 0.455 / 0.426 | 57 s (GPU) |
  | M2 fit code + FiLM heads | **0.807 / 0.588 / 0.552** | **0.839 / 0.566 / 0.490** | 58 s (GPU) |
  | M2-iii (bounding-box tokens) | 0.792 / 0.575 / 0.548 | -- | 58 s |
  | M3 physics, exact geometry + beta fit | 0.977 / 0.971 / 0.959 | | 2.1 s (CPU) |
  | M4-ii physics, snapped geometry | 0.967 / 0.923 / 0.834 | | 2.1 s |
  | M4-iii physics, bounding box | 0.400 / 0.301 / 0.345 | | 2.2 s |
  | M5 wave-equation kernel ridge + source term, no geometry | **0.952 / 0.868 / 0.796** | | 0.7 s |

- **M5's K-curve (pooled strata):** 0.753 / 0.910 / 0.951 / 0.963 at K = 8 / 16 / 32 / 64 (25-100 Hz) and 0.311 / 0.490 / 0.805 / 0.917 (200-300 Hz). R0's M2 at 200-300 Hz: 0.439 / 0.519 / 0.578 / 0.605.
- **Strata (E1; K = 32, 25-100 Hz; near < 0.3 m from the source / far):** R0 zero-shot 0.661 / 0.465; R0 M2 0.846 / 0.806; physics M3 0.909 / 0.977; M5 0.904 / 0.952. Near the source every method is lower except the network.
- **Realism (S-2; 20 dB SNR + 2 cm position error):** the network fits do not move (M1 0.610, M2 0.809 vs 0.608 / 0.811 clean). M5 degrades, most at 200-300 Hz: 0.805 clean, 0.738 real2, 0.619 real5. Physics M3 does not move. M1 / M2 real5 was not run (E8: GPU budget).
- **Calibration (E7, T + DN dense rooms):** M2 lr 1e-3, 400 steps; settings frozen before any reported fit (D183(c)). Complete: 3,844 / 3,844 fits.
- **Reading (Q-A):** the network's measurement-grounding case is narrow. It wins when geometry is unknown AND measurements are very few (K <= 8, above 100 Hz), or against physics given a wrong geometry. With 16+ measurements a geometry-free wave interpolator is better, and with good geometry physics is better at any K.
- **NMF-P passes its kill read:** 0.780 at 10K (>= 0.70). NMF 30K read pending.
- [grounding.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/step1/grounding.md) · [F1_grounding.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F1_grounding.png)

## 2026-11-22 (night) — I3: physics vs network under the same wrong inputs (0d, A1) -> F3; NMF unseen interim

**Headline:** Given the same imperfect room description, physics stays far ahead when the errors are small. With wall corners off by up to 5 cm, it scores 0.97 on unseen rooms against the network's 0.47. With a small alcove missing it scores 0.76, and with the wall absorption wrong by 2x it scores 0.90. Only when physics gets nothing but the room's bounding box does it fall to the network's level (0.40 vs 0.41). On unseen rooms the network barely uses fine geometry: snapping corners or deleting an alcove leaves its score unchanged.

- **Off-lattice r against FDTD truth (A1, primary), 25-100 / 200-300 Hz:**

  | input | physics, unseen 16 | R0 @120K, unseen 16 | S2 @120K, unseen 16 | physics, in-family 15 | R0 @120K, in-family 15 |
  |---|---|---|---|---|---|
  | exact geometry | 0.978 / 0.960 | 0.468 / 0.371 | 0.501 / 0.404 | 0.988 / 0.960 | 0.918 / 0.582 |
  | vertices snapped to 10 cm | 0.968 / 0.836 | 0.465 / 0.365 | 0.495 / 0.400 | 0.974 / 0.838 | 0.910 / 0.561 |
  | smallest notch removed (A7) | 0.760 / 0.636 | 0.469 / 0.335 | 0.495 / 0.369 | n/a | n/a |
  | bounding rectangle only | 0.400 / 0.345 | 0.408 / 0.358 | 0.414 / 0.371 | 0.533 / 0.507 | 0.526 / 0.438 |
  | wall absorption x0.5 (physics only, A2) | 0.813 / 0.821 | -- | -- | 0.774 / 0.805 | -- |
  | wall absorption x2 (physics only, A2) | 0.899 / 0.845 | -- | -- | 0.898 / 0.853 | -- |

- **Reading (0d is descriptive; Q-A):** the approximate-input case for the network does not exist at these error sizes. Physics degrades gracefully: snapping costs 0.01 at 25-100 Hz and 0.12 at 200-300 Hz, and a 2x absorption error costs 0.08-0.17. It ties the network only when the geometry is gone (bounding box). In family, the network does use geometry: the bounding box costs R0 0.39 at 25-100 Hz, the same as physics. On unseen rooms, R0's 0.468 is already close to its own bounding-box score (0.408), so it extracts little from unseen geometry (W2).
- **NMF interim (unseen, 25-100 Hz):** 10K **0.424**, 20K **0.450**; in-family 0.796, 0.840.
  - Reference: R0 0.468 / 0.918. N1 needs unseen >= 0.60 and in-family >= 0.90.
  - Eigenfrequency MAE (first 10 modes, Hungarian-matched) is ~5.5 Hz in family and ~6.0 Hz unseen; subspace overlap is 0.91 / 0.85. The eigenfrequencies are the bottleneck: I9's budget needs ~2%.
  - 30K reads for NMF and NMF-P are pending.
- [physics_sensitivity.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/physics_sensitivity.md) · [F3_sensitivity.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F3_sensitivity.png)

## 2026-11-22 (night) — I13 (kill read): NMF passes its 10K kill rule after the NaN fix; CTL / DIR @130K interim

**Headline:** The learned-modes model (NMF) works after a bug fix. It passes its pre-registered 10K check: on familiar room shapes it reproduces the sound field below 100 Hz with r 0.80 (bar 0.70), still climbing (0.84 at 20K). The main network scores 0.92 on the same rooms. Whether NMF transfers to unseen room shapes, its reason for existing, is being scored now. The two fine-tunes from the main network are unchanged 10K steps in, as expected.

- **NaN fix (D186):** the first NMF / NMF-P runs were void. The subspace loss took `sqrt` of a cutoff weight that float32 rounds to exactly 0 just below the cutoff. The forward pass was finite; the gradient was inf, then NaN. Fixed with an identical loss value and regression-tested (8 / 8 NMF tests pass on a GPU). The kill rule is now NaN-proof and the trainer stops on any non-finite value. Both arms were relaunched from step 0.
- **NMF kill read (D180), in-family 15 shapes, off-lattice r at 25-100 Hz:** 2K 0.598, 4K 0.683, 6K 0.735, 8K 0.761, **10K 0.796 >= 0.70: passes**. 20K 0.840.
  - Reference (Rule 12): R0 @120K reads 0.918 on the same rooms, and N1's in-family bar is 0.90.
  - Training losses at 21.6K: field 0.80, subspace 0.13, relative k error 4.6%, orthonormality 0.11. I9's error budget needs eigenfrequency error <= ~2% for in-family >= 0.90.
- **NMF-P:** 6K 0.735; its kill read is due in ~15 min.
- **Next for NMF:** unseen r (N1: >= 0.60, where R0 reads 0.468) at 10K / 20K is being scored now, and at 30K for both arms when they land (`p4_9_nmf_eval.py`: r and LSD, eigenfrequency MAE, subspace overlap).
- **CTL @130K / DIR @130K (interim, 10K after the restart, D181 A8):**

  | arm | HF (200-300 Hz) | UF 25-100 / 100-200 / 200-300 |
  |---|---|---|
  | R0 @120K (start) | 0.582 | 0.468 / 0.380 / 0.371 |
  | CTL @130K | 0.575 | 0.464 / 0.385 / 0.380 |
  | DIR @130K | 0.576 | 0.471 / 0.378 / 0.376 |

  The directional loss has not cost omni accuracy so far. E2's directional pass rule is read at 150K.
- [F0_trajectory.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F0_trajectory.png)

## 2026-11-22 (night) — I1: latency by edit type (E5) -> F2; NMF / NMF-P runs went NaN (engineering, no NMF number)

**Headline:** For moving the listener, moving the source or changing a wall material, physics with precomputed room modes is faster than the network (2 ms vs 10 ms per listener position). The network's speed case is the geometry edit: 12 ms per listener position, against about a minute for physics or the simulator to recompute. A full 64x64 map of the room takes the network 32 s, only ~1.5x faster than re-running the simulator on a CPU, and a GPU simulator would likely be faster still (estimate, not measured).

- **Per edit type** (one room, 3753 grid points; network R0 @120K fp32 on an RTX A5000; physics and FDTD on one CPU node, Xeon E5-2680 v4; medians of 20 runs):

  | edit | network, 1 listener | network, 64x64 grid | physics, cached modes (1 listener / grid) | physics, re-solve to 600 Hz | FDTD re-run |
  |---|---|---|---|---|---|
  | listener move | 9.9 ms | 32.4 s | 2 ms / 49 ms | 55.6 s | 49.4 s |
  | source move | 9.9 ms | 32.4 s | 2 ms / 49 ms | 55.6 s | 49.4 s |
  | material change | 12 ms (accuracy UNTESTED) | 32.4 s | 2 ms / 49 ms | 55.6 s | 49.4 s |
  | geometry change | 12 ms | 32.1 s | n/a (modes change) | 55.6 s | 49.4 s |

- **Single-listener update rate:** network 101 Hz on the 601-bin path (61 Hz on the current 4097-bin path); cached-mode physics 496 Hz.
- The 601-bin path cuts the network's grid render from 57 s to 32 s. Largest dense room (4056 points): network 35 s, FDTD 60 s, modal re-solve 111 s.
- **A 2D FDTD on a GPU** would take ~0.1-0.3 s per run. This is an ESTIMATE (4.5e9 node updates at 15-50 G/s), not measured. If it holds, the network's whole-room latency case rests on the geometry edit at a single listener position and on 3D, where FDTD cost grows by orders of magnitude.
- The reading stated in advance (D181 E5) holds as written. This is weakness W3, now quantified.
- **NMF / NMF-P: no number yet.** Both runs went NaN between steps 200 and 400. The 10K kill rule did not fire because `nan < 0.70` is False. NMF-P is cancelled. NMF ran to 30K on NaN weights; that run is void. I am debugging on a GPU, and both will relaunch from step 0 once fixed, with a NaN-proof kill rule and a stop on any non-finite loss. I13-I14 slip by the relaunch time (~1 h for NMF to 10K, ~0.5 h more for NMF-P).
- [latency.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/latency.md) · [F2_latency.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/meeting/F2_latency.png)

## 2026-11-22 (evening) — I4: Robin boundary residual + gate G-BC -> PL1 NOT launched; NMF / NMF-P launched

**Headline:** The network's fields break the wall boundary condition badly (~1,700x the simulator's error), but equally badly in familiar and unseen rooms. So a physics penalty at the walls would not target what goes wrong in unseen rooms, and the pre-registered physics-loss arm is not run. The learned-modes route (NMF) passed its feasibility check and is now training.

- **Gate G-BC (R0 @120K, D180):**
  - (1) FDTD truth residual 0.004 <= 0.15: holds.
  - (2) projected residual unseen / in-family 1.25 < 1.5: **fails** (raw: 0.92).
  - (3) Spearman(per-room residual, 25-100 Hz r) = -0.81 <= -0.4: holds.
  - **PL1 is not launched.**
- **Absolute levels (25-100 / 100-200 / 200-300 Hz):** FDTD 0.004 / 0.003 / 0.002; R0 raw 6.95 / 7.49 / 4.49; S2 raw 9.00 / 9.00 / 6.70.
  - The projection's extension to the wall fails its validity check (projected truth 0.61 > 0.15), so the projected values are not trusted. The gate fails on raw values anyway.
- **Q-D:** neither residual separates unseen from in-family rooms (interior ~1.1x, Robin 0.9-1.25x). The free gates are closed.
- **NMF and NMF-P launched** (D184; 2 scavenger A5000s): eigen-labels for 1031 rooms, S-a verified (the formula with the true modes reproduces m1 to 1e-6), 6 tests pass. The kill read at 10K is in the trainer.
- [physics_gates.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/physics_gates.md)

## 2026-11-22 (evening) — I9: NMF feasibility (physics only) -> GO

**Headline:** A modal model with just 24 room modes is accurate enough below 100 Hz (r 0.98), so learning room modes is a viable route. To clear the bar on unseen rooms it can afford sizeable errors. To stay strong on familiar rooms it must place each mode's frequency within about 2%.

- **S-b (pre-registered): HOLDS.** The zero mode + 24 modes reach **0.984** in family at 25-100 Hz (unseen 0.974; every mode to 600 Hz: 0.988 / 0.978). The 24th mode lies at ~165 Hz. D180's fallback (a static correction for the omitted modes) is not needed: with the true higher modes it would only recover 24 -> 0.988.
- **Error budget** (24 modes, r at 25-100 Hz, in family / unseen):

  | eigenfrequency error | 0.5% | 1% | 2% | 5% |
  |---|---|---|---|---|
  | r | 0.978 / 0.970 | 0.965 / 0.956 | 0.920 / 0.923 | 0.768 / 0.786 |

  | mode-shape mixing delta | 0.05 | 0.1 | 0.2 | 0.4 |
  |---|---|---|---|---|
  | r | 0.969 / 0.962 | 0.934 / 0.933 | 0.836 / 0.841 | 0.617 / 0.641 |

- **Reading against N1's bars** (unseen >= 0.60, in-family >= 0.90): the unseen bar tolerates 5% eigenfrequency error or delta 0.4 (each alone); the in-family bar needs <= ~2% and delta <= ~0.1. Errors compound when combined.
- [feasibility.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/nmf/feasibility.md)

## 2026-11-22 (evening) — I2: interior Helmholtz gate; Day-0 launches

**Headline:** A projection that makes the network's fields physically smooth also distorts the TRUE field by about the same amount. So "projected network vs raw truth" cannot tell us whether the network still breaks the wave equation. The fair comparison says the projected network is within ~10% of projected truth.

- **0b(i), pre-registered reading: DOES NOT HOLD.** Projected R0 @120K is 23x / 26x / 21x FDTD's raw residual (2.62 / 1.14 / 0.59 against 0.11 / 0.04 / 0.03, bands 25-100 / 100-200 / 200-300 Hz, 31 rooms), not <= 2x. So the interior Helmholtz loss is NOT closed by the reading as written.
- **Confound, flagged for the manager (D182(f)):** the same projection applied to FDTD truth raises truth's residual 21x (2.43 / 1.01 / 0.54). Against projected truth, projected R0 is 1.08x / 1.13x / 1.11x, S2 1.09x / 1.19x / 1.17x. Raw (unprojected): R0 is 89x / 89x / 58x FDTD's residual, S2 150x / 150x / 100x.
- **Table:** [physics_gates.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/physics_gates.md); per room: [interior.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/gates/interior.json).

**Day-0 launches** (D182):
- **CTL** is training on tron (A5000, 0.77 s/it; restart verified in its log).
- **S2X** (the meeting-model candidate) is pooled across the L40S and A6000 queues: the L40S queue alone estimated 6 days; every pool estimates about a day.
- **DIR and DIR-G** are pooled across tron and scavenger A5000s (labels built: 1000 rooms, FDTD reruns exact to 6.8e-9).
- **Running now:** latency (I1), physics-vs-network sensitivity (I3), Robin gate (I4), directional zero-shot (I5); modal solves (109), confirmation set (16 rooms, frozen) and second-source rooms (16) are built.
- **Directional unit test (E3) passes:** the readout's error equals its closed form and tracks 1/(2kr) within 8.6% for kr 1-30.

## 2026-11-22 — I0: P4-9 pre-registered; P4-8 closed

**Headline:** The next round is locked in before any result exists: four questions about where the learned model earns its keep against physics, all pre-registered with their pass/fail rules.

- **D180** records the manager's spec verbatim; **D181** records 15 agent amendments (approved), the manager's 8 edits and 6 clarifications. The most consequential: every network-vs-physics comparison now gives both methods the same information (A1–A3); the S2 restart avoids a scheduler trap that would have silently never restarted (A8); the directional work makes the gradient of the omni field the primary readout and adds an arm, DIR-G (E2).
- **P4-8 closed: S2 @120K FAILS** its pre-registered test on UF by 0.009 (UF 0.501, HF 0.502). It is under-exposed (200–300 Hz fit 0.523 vs R0's 0.645); P4-9's S2X continues it to 240K at matched per-room exposure.
