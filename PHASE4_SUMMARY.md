# Phase 4 — Status Summary for Manager Review

**Scope**: shape editing. The phase goal is to train on rectangles and L-rooms and render a
Z-shaped corridor zero-shot. It is deliberately staged, each stage behind a gate, so that when
something breaks we know *which* thing broke.

**Status as of 2026-09-13**: **P4-1 complete (both gates pass). P4-2 complete (Gate 2 failed).
P4-3 complete — GATE 2 NOW PASSES on two arms, and the mechanism question answers negatively.**

---

## TL;DR

**Shape generalization is a DATA problem, not a renderer problem. That is the phase's main
result and it reverses the direction P4-2 pointed.**

* **Gate 0 PASS** (P4-1). Boundary tokens beat global `(L, W)`: 0.982 vs 0.951.
* **Gate 1 PASS** (P4-1). A per-scene fit of one non-convex L-room reaches NLOS Pearson +0.974.
* **Gate 2 FAIL** (P4-2), all four arms; best 0.7978. σ ≈ 1.0 everywhere — no occlusion mechanism.
* **Gate 2 PASS** (P4-3), two arms: **X1 attention-residual 0.8438**, **D more-data 0.8021**.
* **The U-shaped accuracy curve is eliminated by DATA alone** — Arm D, amplitude 0.2185 → 0.0752,
  shallow half +0.7017 → +0.8893, with σ **unchanged at 0.93–1.04**.
* **Injecting the mechanism does not help.** Arm S learns a real, generalizing **8×** σ contrast
  inside the wall and is the **worst** arm in the phase.

---

## The five numbers that matter (keep these regimes distinct)

| regime | number | what it is |
|---|---|---|
| zero-shot, unseen rectangle | spatial Pearson **0.982** | P4-1 Gate 0, demo protocol |
| per-scene fit, non-convex | NLOS Pearson **+0.974** | P4-1 Gate 1 — *not* a generalization claim |
| zero-shot, unseen SHAPE (P4-2 best) | **+0.798** | Gate 2, failed |
| zero-shot, unseen SHAPE (P4-3 best) | **+0.844** | Gate 2, **passed** (X1); D 0.802 |
| mechanism | σ **≈ 1.0** on every arm that works | no learned occluder, and none needed |

---

## Why Gate 2 now passes, in one paragraph

P4-2 diagnosed the failure as a missing occlusion mechanism, reasoning from D66 (rays fan outward
from the receiver, so a source-side wall can only be memorized) and from σ ≈ 1.0. P4-3 tested that
diagnosis three ways and it did not hold. **Adding 32 training shapes in the thin `d_hat` band —
five minutes of CPU — eliminated the U-shaped accuracy curve entirely and passed the gate, with
σ untouched.** Meanwhile **forcing** a genuine occluder into the wall (Arm S), which succeeded on
its own terms and generalized to unseen shapes at 8×, made every field metric worse. So the
occluder is neither necessary nor sufficient, and P4-2's U-shape was a symptom of corpus density
along the shape parameter (D72, D75), not of the renderer.

---

## Honesty notes (so nothing gets conflated)

1. **Attention helps for a reason that is not the stated one.** X1 posts the best Gate-2 number
   but produces **0.7%** position-dependence in the pooled conditioning and no σ contrast. The
   pre-registered condition ("σ meaningfully above 1.0 AND flattens the U") is **not met** (D76).
2. **X2, the literal attention replacement, is WORSE than the baseline at all 20 swept depths**,
   while having the second-flattest curve. Flatness without level is not a result.
3. **Arm D's win is conservative**: 92 shapes at a matched 60K means ~35% fewer gradient steps per
   shape than the baseline, a handicap chosen so data was the only variable.
4. **D69 is RETRACTED** (D73). `load_model` ignored `token_pool`, so P4-2's extent arms were
   evaluated as mean-pooled — silently, because extent weighting adds no parameters. Corrected
   gaps are +0.014 and **−0.043**; the pooling axis is unresolved.
5. **The σ probe was counting out-of-room samples as solid** — 82% of them (D74). P4-1/P4-2's
   σ ≈ 1.0 conclusions survive (verified against an independent volume probe), but a *localized*
   contrast was being diluted toward 1.0.
6. **The ray probe and the volume probe answer different questions** — "is there attenuation along
   the occlusion path" vs "is there attenuation in the wall". Arm S reads 1.09 and 8.08. Report
   both or neither.
7. **A hold-out band is not a substitute for a parameter sweep** (D78). Gate 2's slab criterion
   scored a +0.006 effect while the same curve varied by 0.22 end to end.
8. **`tx` has been a single fixed source in 100% of the 2D corpus.** The source-position encoders
   remain essentially unconstrained.

---

## Artifacts (raw GitHub, `…/main/<path>`)

`tasks/CHUNK_P4_3_RESULTS.md` · `tasks/CHUNK_P4_2_RESULTS.md` ·
`outputs/p4_3/p4_3_{X1_attn_residual,X2_attn,S_sigma,D_data}_eval/GATE2.json` ·
`outputs/p4_3/sweep/{baseline_mean_unmasked,X1_attn_residual,X2_attn,S_sigma,D_data}/sweep_metrics.json` ·
`outputs/p4_3/demo/demo_metrics.json` + `fig{L_A,L_B,M_A,M_B,N}*.png` ·
`outputs/p4_2/stage2/p4_2_s2_extent_*_eval_fixed/GATE2.json` (the D73 correction) ·
`configs/sweeps_2d_mat/p4_3_shapes_manifest.json` · `DECISIONS.md` (D66–D78) ·
`OPEN_QUESTIONS.md` (Q21)

---

## Recommended next step

**Scale the corpus along the shape parameters, not the renderer.** Arm D is the generalizable
lesson of this phase: density along a shape axis is a first-order driver of accuracy along that
axis, and FDTD rooms cost seconds while training runs cost half a day. The natural next chunk
adds shapes along `w_hat` and the `(L, W)` box as well as `d_hat`, and re-reads the sweep.

**Do not rewrite the renderer.** It was predicated on occlusion being the binding constraint, and
Arm S falsifies that: a model with a real, generalizing occluder is the worst one measured.

**One cheap experiment remains on the mechanism question** (Q21): Arm S supervised σ in the notch
*volume*, and the clipped ray probe shows the contrast is absent on the grazing path the renderer
actually integrates along (1.09 vs 8.08). A σ supervision targeted **on** the occlusion path is
one training run, needs no new code beyond the sampling, and is the only branch of the mechanism
hypothesis not yet falsified.

**Still do not build the Z-corridor.** Nothing in this phase has demonstrated transfer to a shape
family the model has not seen the parameters of.
