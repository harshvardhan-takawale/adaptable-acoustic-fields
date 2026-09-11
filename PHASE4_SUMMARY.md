# Phase 4 — Status Summary for Manager Review

**Scope**: shape editing. The phase goal is to train on rectangles and L-rooms and render a
Z-shaped corridor zero-shot. It is deliberately staged, each stage behind a gate, so that when
something breaks we know *which* thing broke.

**Status as of 2026-09-11**: **P4-1 complete. Both gates pass.** Stages 2+ not started.

---

## TL;DR

**The conditioning is ready for shapes; the renderer's occlusion mechanism is not demonstrated.**

* **Gate 0 PASS.** Boundary tokens in absolute world coordinates replace global `(L, W)` — and
  beat it: spatial Pearson **0.982 vs 0.951**, band LSD **1.722 vs 2.268 dB**, in-distribution
  val LSD **0.9914 vs 1.0132**, on an otherwise identical arm. The last shoebox-specific
  component is gone at a net gain.
* **Gate 1 PASS.** A per-scene fit of a non-convex L-room reaches NLOS spatial Pearson
  **+0.974** with a LOS−NLOS gap of **+0.020** (+0.890 / +0.074 on held-out receivers). The
  architecture *can* represent non-convex acoustics.
* **But Gate 1 is about capacity, not mechanism.** The learned σ field is statistically elevated
  inside the wall (p = 2.5e-05) yet physically negligible — transmittance across the notch is
  0.068 against 0.082 for the same path in air. σ does generic distance attenuation; the shape
  lives in the `signal` field. **No geometric occlusion mechanism has been demonstrated**, and
  that is what Stage 2 would need to transfer.
* **One counter-result.** Tokens reconstruct better but respond to edits *less linearly*:
  `edit_bw_slope` is below Arm C in every split, worst on S4 (0.464 vs 0.789).

---

## The three numbers that matter (keep these regimes distinct)

| regime | number | what it is |
|---|---|---|
| **zero-shot, unseen rectangle** | spatial Pearson **0.982**, LSD **1.722 dB** | Gate 0, demo protocol, 12 scenarios |
| **per-scene fit, non-convex** | NLOS Pearson **+0.974** (fit) / **+0.890** (held-out) | Gate 1 — *not* a generalization claim |
| **held-out receivers, same room** | LSD **3.25 dB** vs 1.00 dB trained | the generalization number inside Gate 1 |

Conflating row 2 with row 3 is the single easiest mistake to make with this chunk.

---

## What P4-1 built (all reusable)

* **`geom_token` arm** — polygon-edge tokens `[cx, cy, nx, ny, extent, m̂]` in absolute metres,
  12 padded slots with an explicit validity mask, masked-mean pooled. One module covers
  rectangles (4 edges), L-rooms (6) and the planned Z-corridor (8). No global geometry prefix.
* **`WORLD_SCALE = 10.0`** (D64) with `normalize_position` in `aaf/walls.py`, opt-in per
  checkpoint so every pre-P4-1 model is bit-identical (verified 12/12, worst |delta| 0.000e+00).
* **`PolyConfig`** — the first config class describing a non-rectangular room.
* **The L-room corpus recipe** — `mask`-based non-convex FDTD geometry with an analytic
  node-for-node self-check, plus first-ever test coverage for the `mask` spec type.
* **Gate machinery** — `p4_1_gate0.py`, `p4_1_stage1_eval.py`, both computing the verdict before
  any figure is drawn.

---

## Honesty notes (so nothing gets conflated)

1. **Gate 1's headline is a FIT metric.** 87.5% of its receivers were supervised.
2. **σ ratio 1.075 is not "the model found the wall".** It is a medium-effect statistical bump
   with negligible physical consequence.
3. **`mode_shape_invariance` (0.9898) is not the spatial Pearson (0.982).** Different quantities;
   the first is agreement with the analytic cosine shape.
4. **The DC hypothesis is a quarter-effect (23.7%), not a majority**, because `L_amp` (log10) and
   `L_phase` (cosine) are structurally immune to the DC term. Masking is a **trade-off**:
   spatial Pearson +0.132, modal-peak LSD +0.473 worse.
5. **Arm T-geo started behind and finished ahead.** Any matched-iteration comparison before
   ~22000 iterations reverses the conclusion.

---

## Risks carried into Stage 2

1. **No demonstrated occlusion mechanism** (above). Track B (D60a) failed at exactly this point
   when asked to generalize, and nothing in P4-1 shows that has been solved.
2. **Edit linearity regressed** with tokens (S4 slope 0.464). If the phase needs both editing and
   shape transfer, this is unresolved.
3. **`tx` has been `(0.5, 0.5)` m in 100% of the 2D corpus**, so the source-position encoders
   have seen exactly one point ever. Any Stage-2 claim that moves the source is unconstrained by
   training.
4. **`wall_segments`/`patch` silently mis-describe non-rectangular rooms** (D65). Safe only
   because Stage 1 used uniform α.

---

## Artifacts (raw GitHub, `…/main/<path>`)

`tasks/CHUNK_P4_1_RESULTS.md` · `outputs/p4_1/stage0/GATE0.json` ·
`outputs/p4_1/stage1/GATE1.json` · `outputs/p4_1/stage1/SIGMA_ANALYSIS.json` ·
`outputs/p4_1/stage1/DATASET_GATE.json` · `outputs/p4_1/dc_fix/DC_COMPARISON.json` ·
`outputs/p4_1/dc_fix/loss_contribution.json` · `outputs/p4_1/stage0/ARMC_REGRESSION.json` ·
`outputs/p4_1/stage0/splits_eval/{summary,verdict}.json` ·
`outputs/p4_1/stage1/fig{G_lroom_fields,H_sigma_profile}.png` · `DECISIONS.md` (D64–D65)

---

## Recommended next step

**Stage 2 as specified (multi-shape generalization), with one addition**: carry the σ probe from
Stage 1 forward as a standing diagnostic. If a model trained across shapes still shows σ ratio
≈ 1, the representation is memorizing each room in `signal` rather than learning geometry, and
the Z-corridor will not transfer no matter how good the in-distribution numbers look. That
measurement costs nothing and is the earliest available warning.
