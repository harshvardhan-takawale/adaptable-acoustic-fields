# Chunk results — P4-3: minimal shape-edit demo, and the mechanism question

**VERIFY FIRST — Gate 2 was NOT contaminated. The P4-2 corpus was built AFTER the D67c
grid-alignment fix, not before.** All 75 `.h5` files and all 75 `.done` sentinels carry mtimes
inside a single 83-second window beginning 13:04:28 on 2026-09-11 — seven seconds *before* the
fix commit landed at 13:04:35, i.e. the rebuild was already running when the fix was committed —
and all four Stage-2 arms record `manifest_sha 50e3bb164fff`, the post-fix manifest. Re-derived
independently rather than taken from the P4-2 gate: across all 75 rooms, **0** stored-attr
mismatches against the manifest, **0** parameters off the 0.02 m grid, **0** node-for-node
polygon disagreements, worst notch-face displacement **8.882e-16 m**. No rebuild and no re-run
were needed; the "66 of 75 rooms contaminated" scenario did not occur.

---

**GATE 2 PASSES ON TWO ARMS — the first passes in this phase after four P4-2 failures.** Arm X1
(attention, residual form) reaches in-slab spatial Pearson **0.8438** and Arm D (more data)
**0.8021**, against the baseline's 0.7483 and a 0.80 threshold. **The shallow-notch weakness was
a DATA problem**: Arm D flattens P4-2's U-shaped accuracy curve completely — amplitude
**0.2185 → 0.0752**, shallow half **+0.7017 → +0.8893** — while its σ ratio stays at **0.93–1.04,
indistinguishable from the baseline**. And **injecting the mechanism did not help**: Arm S learns
a real, generalizing **8×** σ contrast inside the wall and is the **worst** arm in the phase.

**Date**: 2026-09-13 · **Branch**: `main` · Tests: **524 passed, 0 failed** (506 before)
**Decisions**: D73–D77 · **Open questions**: Q21 substantially answered, Q22 resolved.

---

## 1. The headline table

Gate 2 — 15 frozen test shapes, identical for every arm, threshold 0.80 with NLOS deficit ≤ 0.15:

| arm | in-slab R | out-slab R | band LSD | RIR r | NLOS def | σ ray | σ volume | gate |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| BASELINE `mean_unmasked` | 0.7483 | 0.7369 | 4.84 | 0.9989 | −0.120 | 1.036 | 1.041 | **FAIL** |
| **X1 `attn_residual`** | **0.8438** | 0.7612 | 4.61 | 0.9988 | −0.002 | 1.130 | 1.180 | **PASS** |
| X2 `attn` | 0.7789 | 0.7427 | 4.70 | 0.9987 | +0.070 | 1.053 | — | FAIL |
| S σ-supervision | 0.7329 | 0.6650 | 4.67 | 0.9989 | +0.020 | 1.089 | **8.077** | FAIL |
| **D data(92)** | **0.8021** | **0.7641** | **4.42** | **0.9992** | −0.088 | 0.933 | 1.042 | **PASS** |

The shape-edit sweep — 20 unseen depths at a fixed bounding box, so `L`, `W`, `w` cannot confound
it. This is where the U lives:

| arm | shallow half (d̂ ≤ 0.32) | deep half (d̂ ≥ 0.74) | mean | U amplitude |
|---|---:|---:|---:|---:|
| BASELINE | +0.7017 | +0.8375 | +0.7615 | 0.2185 |
| X1 | +0.7503 | +0.8491 | +0.7836 | 0.1633 |
| X2 | +0.7286 | +0.6971 | +0.7011 | 0.0819 |
| S | +0.5331 | +0.6815 | +0.5906 | 0.2273 |
| **D** | **+0.8893** | **+0.8878** | **+0.8811** | **0.0752** |

> **The baseline sweep had to be run for this chunk.** P4-2's U-curve was measured on
> `mean_masked`; every P4-3 arm uses the unmasked loss (D70). Comparing against the old curve
> would have conflated the loss change with the architecture change.

---

## 2. The three pre-registered branches, and how each resolved

**(a) Arm D — "if D alone flattens the U with σ ≈ 1.0, the shallow-notch weakness was a data
problem and the mechanism question is separable from it."** *This is what happened* (D75). The U
is not reduced, it is **gone**, and the ordering has **inverted**: D's best point is now the pure
rectangle (+0.9262) and its worst is maximum depth (+0.8510). σ is 0.933 / 1.042 — the baseline's
value. Gate 2 passes at 0.8021 with the best band LSD (4.42 dB) and RIR (0.9992) in the phase.
The caveat runs in D's favour: 92 shapes at a matched 60K is ~35% fewer gradient steps per shape,
so D wins *despite* a handicap chosen deliberately to isolate data as the only variable.

**(b) Arm X — "if X raises σ meaningfully above 1.0 AND flattens the U, the diagnosis is
confirmed and attention becomes the adopted aggregator."** *The condition is NOT met* (D76). X1
flattens the U (0.2185 → 0.1633) and posts the phase's best Gate-2 number (0.8438), but σ reads
**1.130 / 1.180** — not meaningfully above 1. Attention helps for some reason **other than**
representing occlusion, and the hypothesis that mean pooling's position-independence was the
bottleneck is **not confirmed**.

Measured directly, the attention is barely used: the position-dependent fraction of the pooled
conditioning, `‖z − mean(z)‖ / ‖z‖` over an interior grid, is **0.007 (X1)** and **0.014 (X2)**.
The conditioning stays ~99% uniform across the room. The mechanism was made available and
training did not take it up. The `1/√d` score scaling (d = 64) keeping logits small and the
softmax near-uniform is a plausible cause; a sharper temperature was **not** tested.

**X2 is the cautionary half.** It FAILS at 0.7789 and is worse than the baseline at every one of
the 20 swept depths — yet it has the second-flattest U in the phase (0.0819), because it
flattened by **uniform degradation**. A flatness statistic read without the level would have
scored it a success. Running both forms was worth it: they differ in sign, not just in speed.

**(c) Arm S — "if S improves fields while σ rises only because it was forced, note that mechanism
was injected rather than learned."** *The mechanism was injected — and the fields got worse*
(D77). The hinge was active in only **3 of 300** logged checkpoints, i.e. satisfied almost from
the start, and the contrast **generalizes**: volume-sampling the removed corner on **unseen** test
shapes reads **8.08** (6.56 on training shapes) against the baseline's 1.04. This is the only arm
in the phase with a genuine occluder in the wall. It is also the **worst** arm: in-slab 0.7329
(below baseline), sweep mean +0.5906, worse at all 20 depths.

**A learned occluder is not sufficient for shape generalization, and on this evidence it is not
even helpful.**

---

## 3. Two bugs found, both of which would have published wrong numbers

**1. `load_model` ignored `token_pool` — P4-2's extent arms were evaluated as mean-pooled (D73).**
Found while diagnosing why the attention arms trailed: their checkpoints would not load at all.
`load_model` rebuilds the model from the checkpoint config and never passed `token_pool`, so every
polygon arm was reconstructed as `masked_mean`. For attention that fails loudly (extra
parameters). **For `extent_sum` it is silent** — extent weighting adds no parameters, so its
`state_dict` is key-identical, the load succeeds without a word, and the checkpoint renders
through a pooling it was never trained with.

| arm | published | corrected |
|---|---:|---:|
| extent_masked | 0.5763 | **0.7835** |
| extent_unmasked | 0.5681 | **0.7916** |
| mean_unmasked (control) | 0.7483 | 0.7483 — **Δ +0.0000** |

**D69 is retracted.** It claimed extent pooling was worse "consistently, by 0.222 and 0.180", and
argued that two loss arms agreeing made it more than a seed artifact. It was not a seed artifact;
it was an evaluation artifact, identical in both arms because both were mis-evaluated the same
way. Corrected gaps: **+0.014** and **−0.043** — under the unmasked loss, extent *beats* mean.
The pooling axis is **not resolved** by that data. Gate 2's verdict, the σ ≈ 1.0 finding, D70 and
D71 are all unaffected; D70 is now supported on four correctly-evaluated arms instead of two.

**2. The σ probe counted everything outside the polygon as solid — 82% of its samples had left
the room (D74).** The ray runs to the bbox diagonal, passes the source, and exits through the far
wall; `solid = ~polygon_contains` swept all of that in. On P4-2's ten probeable test shapes,
**1707 of 2091 "solid" samples (82%) were out-of-room, 99% on one**. Harmless for a near-uniform
field — P4-1 and P4-2 read ~1.0 either way, and an independent volume probe confirms the baseline
at 1.049/1.041 — so **D71's finding and Q21 stand**. But it diluted Arm S's real contrast from
8.08 to 1.31, which is exactly the mechanism this diagnostic exists to detect. Now clipped to the
bounding box, with `n_samples_beyond_room` reported so the clipping is auditable.

---

## 4. Task 1 — the scoped shape-edit demo (COMPLETE)

Five figures from the `mean_unmasked` checkpoint, no optimisation at demo time, two unseen rooms
× four notch depths plus a five-frame morph. Probe receiver drift across depths: **0.000 m**.

| | spatial R | band LSD | modal RIR r |
|---|---|---|---|
| room A | 0.739 → 0.805, then **0.674** at the deepest | 4.2–5.3 dB | 0.83–0.93 |
| room B | 0.843 → **0.906** monotone | 4.5–5.2 dB | 0.82–0.90 |
| morph | 0.877 → **0.950** monotone | 5.34 → 3.62 dB | 0.83–0.95 |

**The RIR number needed fixing rather than shipping.** The first pass reported RIR Pearson +0.999
from a raw 0–300 Hz inverse transform. That number is real and nearly meaningless: the transform
is dominated by a slow near-DC ramp both curves share, so it read +1.000 while the traces sat at
visibly different levels and the panel showed no impulse structure — Q18 resurfacing in the time
domain. The pack now reports **both**, using the repo's own `band_limited_rir`: full-band +0.999
(kept for comparability with Gate 2, computed the same way) and **modal 20–300 Hz, +0.82 to
+0.95**, which is what the figure draws.

Every caption states the d̂ ≥ 0.50 scope and carries the shallow number (+0.741 at 0.21 against
+0.906 at 0.90). The pack does not claim the half that does not work.

---

## 5. What the manager should take from this

1. **Add data before adding architecture.** Arm D is the cheapest intervention in the phase — 32
   FDTD rooms, ~5 minutes of CPU — and it produced the largest effect: the U eliminated, Gate 2
   passed, best LSD and RIR. D72's density warning was first-order, not methodological.
2. **The mechanism question is now substantially answered, negatively (Q21).** Option (c) has been
   run: a model with a genuine, generalizing 8× occluder in the wall is the **worst** arm. That
   points away from "the renderer cannot represent occlusion" as the binding constraint, and it
   means a renderer rewrite (option a) should **not** be the next move.
3. **Attention is worth keeping, narrowly.** `attn_residual` posts the best Gate-2 number, but it
   produces ~1% position-dependence and no σ contrast, so it must not be described as having
   solved occlusion. The residual form, not the replacement — X2 is worse than the baseline.
4. **Do not read flatness without level.** X2 has a flat sweep and is worse everywhere.
5. **Two published results were wrong for the same underlying reason**: a quantity that differs
   only in *forward behaviour*, not in parameters, cannot be validated by a checkpoint round-trip.
   Both are corrected and both now have tests.
