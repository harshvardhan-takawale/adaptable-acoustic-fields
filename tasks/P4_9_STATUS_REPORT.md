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
