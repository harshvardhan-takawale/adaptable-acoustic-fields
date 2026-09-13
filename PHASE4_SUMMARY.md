# Phase 4 — Status Summary for Manager Review

**Scope**: shape editing. The phase goal is to train on rectangles and L-rooms and render a
Z-shaped corridor zero-shot. It is deliberately staged, each stage behind a gate, so that when
something breaks we know *which* thing broke.

**Status as of 2026-09-12**: **P4-1 complete (both gates pass). P4-2 complete — GATE 2 FAILS.**
The staging did its job: we know which thing broke.

---

## TL;DR

**The conditioning is finished and good. The renderer is the blocker, and we now have the
measurement that says so.**

* **Gate 0 PASS** (P4-1). Boundary tokens in absolute world coordinates replace global `(L, W)`
  and beat it: spatial Pearson **0.982 vs 0.951**, band LSD **1.722 vs 2.268 dB**.
* **Gate 1 PASS** (P4-1). A per-scene fit of one non-convex L-room reaches NLOS spatial Pearson
  **+0.974**. The architecture *can* represent non-convex acoustics.
* **GATE 2 FAIL** (P4-2), all four arms. One model across a 60-shape family reaches in-slab
  spatial Pearson **0.7978** against a 0.80 threshold. It can fit one room and cannot generalize
  across the family.
* **σ ratio ≈ 1.00 on every arm** (0.9987 / 0.9896 / 1.0361 / 0.9820), **two of four below 1.0**.
  There is no occlusion mechanism. This is the number P4-1 recommended carrying forward as an
  early warning, and it fired.
* **Edit linearity is RESTORED and the conditioning question is closed.** `geom_token_m` recovers
  `edit_bw_slope` 0.871 → **0.957** against Arm C's 0.959 on the gated split, at the best
  reconstruction of any arm (val LSD **0.9165**). **Adopted** (D68).

---

## The four numbers that matter (keep these regimes distinct)

| regime | number | what it is |
|---|---|---|
| **zero-shot, unseen rectangle** | spatial Pearson **0.982** | Gate 0, demo protocol |
| **per-scene fit, non-convex** | NLOS Pearson **+0.974** | Gate 1 — *not* a generalization claim |
| **zero-shot, unseen SHAPE** | spatial Pearson **+0.798** | **Gate 2 — this is the generalization number** |
| **mechanism** | σ solid/air **≈ 1.00** | no learned occluder, on any arm |

Row 2 is the one most easily mistaken for row 3. Row 3 is the honest state of the phase.

---

## Why Gate 2 failed, in one paragraph

`FreqRenderer2D` fans rays **outward from the receiver** and feeds the source in as a network
input; transmittance accumulates only along the receiver→point leg (**D66**, now a standing
entry because three separate analyses have had to re-derive it). So σ is structurally able to
represent a wall between the RECEIVER and a sample point, and structurally **unable** to
represent one between the SOURCE and that point — the latter can only be absorbed into `signal`,
i.e. memorized per room. A per-scene fit can afford that memorization; a model asked to
generalize across shapes cannot. P4-1 measured σ at 1.075 on a single fitted room. P4-2 measures
it at ~1.00 across a family, with the NLOS deficit **negative** on all four arms (NLOS predicted
*better* than LOS — the opposite of shadow modelling). Gate 2's 0.7978 is what memorization
looks like when the corpus stops fitting in the weights.

**The 0.0022 miss is a fail, not an "almost".** The threshold was frozen before the run. And a
checkpoint with σ ≈ 1.0 is not 0.3% away from a mechanism it does not have.

### And the sweep says the difficulty is not where the hold-out looked

20 unseen notch depths at a fixed bounding box. The curve is neither flat nor slab-dipping: it is
a **U**, worst at **shallow** notches (+0.741 at d-hat 0.21), best at **deep** ones (+0.906 at
0.90), with the held-out band costing just **+0.034**. Deep notches are the strongly non-convex,
high-NLOS rooms and they are the EASY case — the opposite of an occlusion story, and consistent
with sigma ~ 1.0. Band LSD reproduces the U independently. Training density correlates only
r = +0.45 and the zero-density slab decile beats three populated ones, so density is a confound
but not the explanation (D72).

---

## What P4-2 built (all reusable)

* **`geom_token_m` conditioning** (D68) — geometry-only tokens plus Arm C's proven per-wall
  material channel. The adopted conditioning for everything downstream.
* **`aaf/data/shape_configs.py`** — the notch family, grid-snapped, with the hold-out enforced in
  four independent layers.
* **`aaf/sim/polygon_geom.py`** — general point-in-polygon, line-of-sight, and the **D65 per-wall
  extent guard** that makes a silent `tiles_exactly: True` impossible.
* **`aaf/eval/sigma_probe.py`** (Task C) — the mechanism diagnostic as a standing, tested library,
  reproducing P4-1's ad-hoc numbers exactly (same probe receiver, means to ten decimals).
* **`token_pool`** as a plumbed, resume-guarded config key.
* **The sweep axis** — 20 unseen notch depths at fixed `L, W, w` with probe receivers pinned
  across rooms, so a shape edit cannot be confounded with a change of listening position. The
  render caches to disk, so re-drawing a figure is 17 s rather than 37 min.

---

## Honesty notes (so nothing gets conflated)

1. **Gate 1's headline is a FIT metric** (87.5% of receivers supervised). Gate 2's is not — every
   test shape is unseen, so all 800 of its receivers are unseen too.
2. **The NLOS-deficit criterion PASSED and means nothing good.** All four deficits are negative.
   Reported with its power: 354 pooled in-slab receiver-modes, 1.2% mean NLOS census, 5 of 15
   test shapes with ZERO NLOS receivers.
3. **The held-out slab is EASIER than the rest of the test set** on every arm (Q22). It is not
   isolating the difficulty it was designed to isolate.
4. **One of the spec's two hypotheses came back negative; the other was a BUG.** DC-masking
   destroys the impulse response — RIR Pearson 0.9989 → 0.0630 (D70) — and that now holds on
   four correctly-evaluated arms. The pooling claim (D69) is **RETRACTED**: `load_model` never
   passed `token_pool`, so both extent arms were rendered with mean pooling. Corrected gaps are
   +0.014 and **−0.043**, i.e. extent *beats* mean under the unmasked loss, and the pooling axis
   is **not resolved** (D73).
5. **Task A's S4 split did NOT recover** (0.491 vs Arm C's 0.789). The entanglement explanation
   holds for moderate absorption edits, not for edits at the range edge.
6. **`tx` has been `(0.5, 0.5)` m in 100% of the prior 2D corpus**; P4-2's family uses a single
   fixed source at `(0.5, 1.6)`. The source-position encoders remain essentially unconstrained.
7. **Δ\* ≈ 0.275 from P3-2d is RETRACTED** — it does not replicate at a second seed. See D67(d)
   and the corrections now in `outputs/p3_2d/SAMPLING_LAW.md`.

---

## Artifacts (raw GitHub, `…/main/<path>`)

`tasks/CHUNK_P4_2_RESULTS.md` · `outputs/p4_2/stage2/DATASET_GATE.json` ·
`outputs/p4_2/stage2/p4_2_s2_{mean,extent}_{masked,unmasked}_eval/GATE2.json` ·
`outputs/p4_2/stage2/sweep/sweep_metrics.json` ·
`outputs/p4_2/stage2/sweep/fig{I_sweep_accuracy,J_sweep_waterfall,K_sweep_field_strip}.png` ·
`outputs/p4_2/taskA/metrics.json` · `outputs/p4_2/taskA/splits_eval/{summary,verdict}.json` ·
`configs/sweeps_2d_mat/p4_2_shapes_manifest.json` · `DECISIONS.md` (D66–D73) ·
`OPEN_QUESTIONS.md` (Q21–Q22)

---

## Recommended next step

**Do not build the Z-corridor, and do not start the renderer rewrite yet.** Two cheap experiments
decide between "the architecture cannot represent occlusion" and "it has no reason to" (Q21):

1. **More shapes / longer training.** The best arm missed by 0.002 on a 60-shape corpus. This is
   the only way to establish whether the ceiling is architectural at all, and it needs no new
   code. σ ≈ 1.0 predicts it plateaus — but that prediction should be tested, not assumed.
2. **An auxiliary loss on σ inside known-solid regions.** The dataset already stores the solid
   masks. If σ *can* be pushed into the wall and Gate 2 improves, the renderer is fine and the
   loss was the problem. If it cannot, the renderer change is justified and scoped.

Only then, if needed: **source-side occlusion in the renderer** (a second transmittance leg from
`tx`, or geometry-conditioned ray termination). It changes the renderer for every arm and
invalidates cross-phase comparisons, so it needs its own gate and a reference-arm re-run — which
is exactly why it should not be spent before (1) and (2) have run.
