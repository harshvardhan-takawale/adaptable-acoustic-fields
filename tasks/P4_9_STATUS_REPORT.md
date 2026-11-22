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
