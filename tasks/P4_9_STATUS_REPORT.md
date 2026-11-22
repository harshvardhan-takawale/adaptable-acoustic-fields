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
