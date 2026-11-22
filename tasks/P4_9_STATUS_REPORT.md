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

## 2026-11-22 — I0: P4-9 pre-registered; P4-8 closed

**Headline:** The next round is locked in before any result exists: four questions about where the learned model earns its keep against physics, all pre-registered with their pass/fail rules.

- **D180** records the manager's spec verbatim; **D181** records 15 agent amendments (approved), the manager's 8 edits and 6 clarifications. The most consequential: every network-vs-physics comparison now gives both methods the same information (A1–A3); the S2 restart avoids a scheduler trap that would have silently never restarted (A8); the directional work makes the gradient of the omni field the primary readout and adds an arm, DIR-G (E2).
- **P4-8 closed: S2 @120K FAILS** its pre-registered test on UF by 0.009 (UF 0.501, HF 0.502). It is under-exposed (200–300 Hz fit 0.523 vs R0's 0.645); P4-9's S2X continues it to 240K at matched per-room exposure.
