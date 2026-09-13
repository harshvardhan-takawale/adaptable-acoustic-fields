# Chunk results — P4-2: minimal shape editing, the notch family

**GATE 2 FAILS on all four arms — best in-slab spatial Pearson 0.7978 against a 0.80 threshold,
a miss of 0.0022. Edit linearity WAS restored: Task A's `edit_bw_slope` recovers from 0.871 to
0.957 against Arm C's 0.959 on the gated split, so Q20's entanglement hypothesis was right and
`geom_token_m` is adopted. And the σ ratio, the number that says whether any of this can
transfer, reads 0.9987 / 0.9896 / 1.0361 / 0.9820 across the four arms — indistinguishable from
1.0, with two of the four BELOW 1.0.** The near-miss is not a near-pass: a model whose σ field
shows no occluder at all did not almost solve shape generalization, and the 0.3% shortfall is not
the kind of gap that scaling closes. Gate 2's second criterion (NLOS deficit ≤ 0.15) passes
everywhere, but every deficit is *negative* — NLOS receivers are predicted slightly BETTER than
LOS ones, which is the opposite of what a model representing shadow geometry would do.

**Three findings the spec did not anticipate**, each detailed below: extent-weighted pooling —
the spec's own hypothesis — is **consistently worse** than masked mean (−0.22 and −0.18 in-slab
across two independent loss arms); DC-masking the loss **destroys the impulse response** (RIR
Pearson 0.9989 → 0.0630) for a +0.05 modal gain; and the held-out slab turns out to be **easier**
than the rest of the test set on every arm, so it is not isolating the difficulty it was designed
to isolate.

**Date**: 2026-09-12 · **Branch**: `main` · Tests: **506 passed, 0 failed** (471 before)
**Decisions**: D66–D72 · **Open questions**: Q19 and Q20 resolved and removed; Q21–Q22 added.

---

## 1. TASK A — edit linearity restored (Q20 answered: the entanglement hypothesis was right)

The question: did tokenizing `m̂` into the shared token MLP cost the linear-in-`m` calibration?
The test was Q20 option (c) exactly as written — a token arm carrying **geometry only**
(`[cx, cy, nx, ny, extent]`, `D_TOK_GEO = 19`) with material re-entering on Arm C's proven
per-wall channel (`m̂` identity + Fourier k=0..2, 28 dims, **verified byte-identical to
`m_linear_features_2d[32:60]`**), concatenated with the pooled token embedding before FiLM.
Stored 268, reduced 92. Identical corpus, manifest sha, recipe, optimizer and 60K budget; the
only change from T-geo is *where `m̂` lives*.

### The frozen P3-2b splits (thresholds unchanged, `thr a8479c5e1dcc`)

| split | **T-geo+m** | T-geo | Arm C | | LSD T+m | T-geo | Arm C |
|---|---:|---:|---:|---|---:|---:|---:|
| S1 unseen_geom 1wall | +0.850 | +0.784 | +0.997 | | 2.45 | 2.31 | 2.98 |
| **S2 unseen_geom_slab** *(gated)* | **+0.957** | +0.871 | +0.959 | | 2.18 | 2.05 | 2.67 |
| S3 seen_geom_slab | +0.742 | +0.720 | +0.720 | | 0.59 | 0.66 | 0.62 |
| S4 unseen_geom α=0.30 | +0.491 | +0.464 | +0.789 | | 2.50 | 2.37 | 3.00 |
| S5 unseen_geom 2wall | +0.997 | +0.913 | +1.010 | | 2.08 | 1.95 | 2.63 |

S2 **PASSES** all four gate criteria: slope 0.957, pearson 0.871, gain 1.274, and
`|ρ − 1| = 0.009` — the best of the three arms (T-geo 0.054, Arm C 0.053).

### It costs nothing in reconstruction — it is the best arm on that too

In-distribution val LSD **0.9165** vs T-geo 0.9914 vs Arm C 1.0132 (best checkpoint 0.9052),
ahead of both at every matched iteration from 20K. Demo protocol spatial Pearson mean **0.982**,
worst 0.964, clearing the ≥ 0.97 adoption bar. Band LSD beats Arm C in every unseen-geometry
split. The spec's success condition was "T-geo's reconstruction with Arm C's edit slope"; that is
what this is, and **`geom_token_m` is adopted for all downstream work** (D68).

### The caveat that is not optional to report

**S4 does not recover**: 0.491 against Arm C's 0.789, an improvement of 0.027 on T-geo's 0.464 —
about 9% of the gap. So disentangling `m̂` recovers edit linearity where the edit is a *moderate*
absorption change and does **not** recover it where the edit pushes absorption to the edge of the
sampled range. The entanglement claim is established for S2/S5 and **not** for S4. S4 is not a
gated split, so nothing is blocked, but whatever is happening there is untouched by this change.

---

## 2. TASK B — GATE 2: does one model generalize across shapes? (FAIL, all four arms)

**Family (D67)**: base rectangle `L ∈ [5.0, 7.0]`, `W ∈ [4.0, 5.5]`, rectangular notch of depth
`d ∈ [0, 0.45W]` and width `w ∈ [0, 0.45L]` removed from the top-left corner. 60 training shapes
(≥ 8 pure rectangles, none in the slab), 15 frozen test shapes (6 in-slab). FDTD `dx = 0.02`,
`fs = 30720`, 0–300 Hz, 800 receivers per room. Dataset gate **11/11 PASS**.

`d̂ = d/(0.45W)`, **not** `d/W` — under the other reading the hold-out band lies entirely outside
the sampled range and the slab would have been **empty**, excluding nothing while the chunk
reported generalization it never tested.

### GATE 2 verdict

> Requires in-slab test shapes to reach **spatial Pearson ≥ 0.80 mean** with an **NLOS deficit
> ≤ 0.15**. Verdict computed and printed before any figure was drawn (D62a).

| arm | pool | loss | **in-slab R** | out-slab R | band LSD | RIR r | NLOS deficit | **σ solid/air** | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `mean_masked` | masked-mean | DC-masked | **+0.7978** | +0.7020 | 5.76 | 0.063 | −0.056 | **0.9987** | **FAIL** |
| `mean_unmasked` | masked-mean | unmasked | +0.7483 | +0.7369 | 4.84 | 0.9989 | −0.120 | **1.0361** | **FAIL** |
| `extent_masked` | extent-sum | DC-masked | +0.5763 | +0.3975 | 6.42 | 0.070 | −0.126 | **0.9896** | **FAIL** |
| `extent_unmasked` | extent-sum | unmasked | +0.5681 | +0.5284 | 8.77 | 0.9967 | −0.065 | **0.9820** | **FAIL** |

> **CORRECTION (P4-3, 2026-09-12): the two extent rows below were computed with the WRONG
> POOLING and are superseded.** `load_model` never passed `token_pool`, and because
> `extent_sum` adds no parameters its `state_dict` is key-identical to `masked_mean`'s, so both
> extent checkpoints loaded silently and rendered through mean pooling. Corrected in-slab
> spatial Pearson: **extent_masked 0.7835** (was 0.5763) and **extent_unmasked 0.7916** (was
> 0.5681); `extent_unmasked` band LSD 8.77 -> 4.81 dB. The control is exact — `mean_unmasked`
> re-evaluates to 0.7483, delta +0.0000. **The Gate 2 verdict is unchanged** (all four FAIL,
> best still mean_masked 0.7978) and **the sigma ratios are unchanged**. See **D73**, and note
> that **section 3(a) below is RETRACTED**.


The best arm misses by **0.0022**. That is reported as a fail, not a near-pass, and the threshold
frozen before the run is not revisited after seeing the result (D71c).

### Why the NLOS pass must not be cited as shadow modelling

All four deficits are **negative** — NLOS receivers score *better* than LOS ones. If the model
represented shadow geometry, NLOS would be the hard case. The likeliest explanation is that NLOS
receivers here sit deep in a corner where the field is smooth and low-variance, so they are easy
for reasons unrelated to the occluder. The criterion is reported with its power: **354 pooled
in-slab receiver-modes**, a **1.2% mean NLOS census**, and **5 of 15 test shapes with ZERO NLOS
receivers**. Per-shape NLOS averaging would have averaged several undefined values and a handful
computed on 3–12 points; the pooled estimate is the only one the data supports.

### The σ probe — the number that predicts transfer

Run on all 10 probeable test shapes per arm: solid/air ratio **0.9987, 0.9896, 1.0361, 0.9820**;
mean transmittance ratio 1.003, 1.027, 1.087, 0.917. **There is no occlusion mechanism.** Two of
the four arms show *less* attenuation inside the wall than in air. P4-1's per-scene fit at least
reached 1.075 with a real effect size (p = 2.5e-05, d = 0.736); the multi-shape model has nothing.

Read with D66 — the renderer fans rays *outward from the receiver*, so a wall between the SOURCE
and a sample point has **no structural representation at all** and can only be memorized in
`signal` — this is the predicted failure mode from Q19, observed. The architecture can **fit** one
non-convex room (P4-1 Gate 1, +0.974) and cannot **generalize** across a shape family (+0.798).
The gap between those two numbers is exactly the memorization D66 makes structurally available.

### The shape-edit sweep — the money figure, and it does not show what was predicted

`L, W, w` pinned; `d` swept across **20 unseen depths** (three inside the held-out slab), same
model, same source, same fixed probe receivers. Both pre-registered outcomes were wrong: the
curve is neither **flat** (shape interpolated) nor **dipping over the slab** (shape memorized).

| d̂ | 0.00 | 0.21 | 0.27 | 0.47\* | 0.52\* | 0.58\* | 0.68 | 0.90 | 1.00 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial R | +0.866 | **+0.741** | +0.743 | +0.767 | +0.784 | +0.797 | +0.848 | **+0.906** | +0.867 |
| band LSD | 4.67 | 6.12 | 6.15 | 6.10 | 6.19 | 5.75 | 5.46 | 4.86 | 4.92 |
| NLOS % | 0.0 | 0.0 | 0.1 | 0.9 | 1.2 | 1.6 | 2.9 | 7.7 | 11.8 |

\* = inside the held-out slab.

**Accuracy is worst at SHALLOW notches and best at DEEP ones** — a U-shape, minimum +0.741 at
d̂ ≈ 0.21, maximum +0.906 at d̂ ≈ 0.90. **Band LSD shows the same U independently** (6.15 worst,
4.17 best), which matters because LSD is not variance-normalized and spatial Pearson is: the two
metrics are confounded in different directions, so their agreement is not a Pearson artifact.

**This is the opposite of an occlusion story.** If the model had to represent shadow, the deep
notches — 11.8% NLOS, a strongly non-convex room — would be the hard case. They are the *easy*
case. What is hard is the **small perturbation of a rectangle**, where the tokens change slightly
and the field changes slightly. That is consistent with σ ≈ 1.0: the model is not doing geometry,
it is doing something closer to "large, low-frequency shape change → large, easily-fit field
change", and it loses exactly the fine discrimination that a near-rectangle demands.

**The density control** (`train_density` in `sweep_metrics.json`, grey bars on figI). The training
corpus is strongly **bimodal** in d̂ — 15 shapes in [0, 0.1) (8 of them the mandated pure
rectangles), a trough of 2–5 per bin through [0.1, 0.5), then 12/4/6/10 across the upper half.
Accuracy correlates only **r = +0.45** with local training density, and the decisive detail is
that the **zero-density slab bin scores +0.791, beating three bins that DO contain training
shapes**. So sampling density explains part of the curve and cannot explain it on its own.

**A methodological finding worth carrying forward** (D72): the sampler satisfied its
`max_normalized_gap ≤ 0.125` invariant (realized 0.070) while producing a **7.5× density spread**
across d̂ deciles. A maximum-gap constraint bounds the largest *hole*; it says nothing about
*uniformity*. Any future chunk that wants even coverage must constrain density directly.

**And it confirms Q22 on an independent axis.** In-slab +0.783 vs out-of-slab +0.817 — a deficit
of just **+0.034** across 20 unseen depths at a *fixed* bounding box, where the Gate 2 test set
varies `L, W, w` as well. The hold-out band is simply not where the difficulty lives.

**One thing the sweep shows working**: the predicted waterfall **does migrate with** the FDTD
waterfall (figJ) — the model's shape dependence is continuous and qualitatively correct; it is
the amplitude that is wrong. And figJ makes D70 visible: below ~20 Hz the predicted traces
oscillate violently while FDTD is smooth, because this arm's loss excluded those bins and the
model is simply unconstrained there. That is the RIR collapse, drawn.

---

---

## 3. Three findings the spec did not anticipate

### (a) The spec's pooling hypothesis is empirically WRONG (D69)

> **RETRACTED (P4-3).** This section rests on the mis-evaluated extent numbers corrected above.
> The corrected gaps are **+0.014 (masked)** and **−0.043 (unmasked)** — under the unmasked loss
> that D70 makes primary, **extent_sum BEATS masked_mean**. The four arms lie within 0.05 of one
> another and the pooling axis is **not resolved** by this data. D69 is retracted in **D73**;
> the text below is kept only so the retracted claim and its evidence stay legible.


The spec proposed `Σ extentᵢ · φ(tokenᵢ)` on the reasoning that a longer boundary should have a
larger effect. Implemented **raw** exactly as specified (normalizing would destroy the very
property that is its rationale) and unit-tested to reduce to masked mean when extents are equal.

It **loses in both loss arms**: in-slab spatial Pearson 0.798 vs 0.576 (DC-masked, gap 0.222) and
0.748 vs 0.568 (unmasked, gap 0.180). Two independent loss configurations, same ordering — not a
seed artifact. Likely mechanism (stated as hypothesis, not finding): the raw sum's magnitude
scales with total boundary length, so the pooled embedding's norm is confounded with room *size*
as well as *shape*. A **normalized** extent weighting was not tested and is the obvious rescue.

### (b) DC-masking destroys the impulse response (D70)

| | in-slab R | out-slab R | band LSD | **RIR Pearson** |
|---|---:|---:|---:|---:|
| masked-mean, DC-masked | 0.798 | 0.702 | 5.76 | **0.0630** |
| masked-mean, unmasked | 0.748 | 0.737 | 4.84 | **0.9989** |

A +0.050 in-slab modal gain costs the entire time-domain signal. A spatial-audio representation
whose impulse response is uncorrelated with the truth (r = 0.06) has failed at the thing it is
for, whatever its modal correlation says. **The unmasked loss is primary downstream**; any future
report of a DC-masked LSD must carry the RIR correlation beside it.

### (c) The held-out slab is EASIER than the rest of the test set (Q22)

In-slab exceeds out-of-slab on **every** arm: +0.096, +0.011, +0.179, +0.040. The binding
difficulty is not the 0.15-wide gap in one parameter — it is shape generalization in general, and
the slab hold-out is not isolating it. Either the slab is too narrow to bite (0.15 of a normalized
range against a 0.125 sampling target — barely one sample spacing), or in-slab shapes are easier
for an unrelated reason. Checkable from `per_shape` at zero compute cost, and it should be checked
before any future chunk reuses this hold-out design.

---

## 4. TASK C — the σ probe as a reusable diagnostic

`aaf/eval/sigma_probe.py`. P4-1's σ analysis — the single number that says whether shape transfer
can work — had **no producer script**; it was done ad hoc in-session and only its JSON survived.
It is now a standing diagnostic callable on any `(model, polygon, source)`, returning the keys
`GATE1.json` already carried plus the statistics that had no code behind them (Mann–Whitney,
Cohen's *d*, transmittance across solid vs the same path in air), with `reach` derived from the
bbox diagonal instead of a hard-wired 6.5 and a world-scale consistency assert.

**Anchored against P4-1's own recorded values** and reproducing them exactly: same probe receiver
`[3.12, 4.82]`, same 226 solid samples, mean σ solid/air matching to **ten decimals**
(0.7292405963 / 0.6782693267), ratio 1.0751, p = 2.543e-05, Cohen's *d* = 0.736, transmittance
0.068 vs 0.082. It also ships the general point-in-polygon and line-of-sight helpers that did not
exist in-repo (`build_p4_1_lroom.inside_polygon` was a hand-written boolean for one notch).

---

## 5. Bugs and corrections found during the chunk

**1. The grid-alignment bug — the serious one (D67c).** `_fit_axis` picks `dx` *per axis* so `L`
and `W` divide exactly (`dxx = 0.020040` for `L = 5.05`), so independently-drawn 2-dp notch
dimensions do not land on nodes: the notch face was displaced **8.4 mm**. The token would have
carried `extent` for a wall the solver put elsewhere — **the conditioning and the ground truth
would have described different rooms, silently.** Caught only by the builder's node-for-node
assert, which failed on all 75 rooms. Fixed by snapping `L, W, d, w` to multiples of 0.02; worst
displacement now **8.88e-16 m**. **9 of the 75 had built cleanly beforehand purely by coincidence**
— a partial pass is the worst possible signal, and is what a weaker assert would have produced.

**2. Missing trainer schema branch.** `p4_2.shape` had no dispatch case, so it fell through to
`mat_configs_cont`, which parses the rows happily and yields configs **with no `verts`** — every
L-room would have been conditioned as a rectangle with no error at all.

**3. The D65 extent guard passed the exact bug it exists to catch.** My first version compared
reported extents against *any* polygon edge length; for a top-left notch the east wall genuinely
*is* the full 5.0 m, so a west wall wrongly reported as 5.0 sailed through. Rewritten to check
**per wall** via `wall_extent()`; it now fires on west-as-5.0 and north-as-6.0.

**4. A sampling-gap metric artifact.** Raw `d̂`/`ŵ` gaps read 0.166/0.175 and flagged "too coarse"
— but those *are* the deliberate hold-out band and the sliver cutoff. Added an `exclude`
parameter; real gaps are 0.070/0.067.

**5. Stage-2 OOM.** `batch 16 / rx 4 / accum 1` puts 16 rows in one backward (accumulation steps
do not share memory) against P4-1's proven 8. My own config comment claimed it matched P4-1 and
was wrong. Fixed to `batch 8 / rx 2 / accum 2` — 8 rows per backward, 4 distinct shapes.

**6. `outputs/p3_2d/SAMPLING_LAW.md` was publicly wrong.** It still read "a second seed is
running". It landed and **did not replicate**: G030 — the sole evidence for Δ\* ≈ 0.275 — flips
ρ 1.3083 FAIL → 1.1753 PASS between seeds. Corrections added at both stale passages. The spec's
"interval ≤ 0.2 of normalized range" citation lands at 0.2 × 1.59 = **0.318**, which is exactly
that G030 arm; this chunk used **0.125** (G020, passing under both ρ definitions at both seeds).
Δ\* is **retracted**, not merely unreported.

---

## 7. What the manager should take from this

1. **Task A is a clean win and is adopted.** `geom_token_m` gives T-geo's reconstruction *and*
   Arm C's edit slope on the gated split. Use it for all downstream conditioning. The S4 caveat
   is real and is about absorption edits at the range edge, not about geometry.
2. **Gate 2 failed, and the σ ratio says why.** Do not read 0.7978-vs-0.80 as "almost". A model
   with σ ≈ 1.0 has no occlusion mechanism, and D66 says the renderer gives it no structural way
   to acquire one on the source side. This is the strongest evidence the project has that
   in-distribution accuracy will keep failing to transfer.
3. **Do not build the Z-corridor on this.** The spec already said not to; the σ result is the
   independent confirmation of why.
4. **Two cheap experiments should precede any renderer rewrite** (Q21): more shapes / longer
   training to test whether 0.002 is a capacity ceiling at all, and a direct auxiliary loss on σ
   inside known-solid regions to separate "the architecture cannot represent occlusion" from "it
   has no reason to". The dataset stores the geometry (`verts`, `d`, `w`) from which the solid mask is RECONSTRUCTIBLE in closed form -- `x < w and y > W - d` for this axis-aligned notch. It does not store a mask array; only a scalar `notch_solid_nodes` count (corrected in P4-3, which implements the term).
5. **The hardest case is the SHALLOW notch, not the deep one.** The sweep's U-shape is the most
   actionable single fact in this chunk: a model that handles a strongly non-convex room better
   than a near-rectangle is not doing geometry. If the next chunk adds shapes, weight them toward
   `d̂ ∈ [0.1, 0.5]`, where the corpus is thin (D72) and the model is worst.
6. **Two of the chunk's arms were the spec's own hypotheses, and both came back negative**
   (extent pooling, DC-masking). Both are recorded with their controls so they do not have to be
   re-run.
