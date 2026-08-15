# CHUNK P3-2c — Results

**The interior-gap sweep does not measure what it was designed to measure, and the reason is
identifiable rather than statistical.** The within-run control (north — same slab, same
byte-identical draws in all four arms) moves *perfectly monotonically* with the manipulation
(Spearman ρ_rank = **1.000**, spread **0.316** against a pre-registered tolerance of 0.15),
while the manipulated wall (west) does not (Spearman **−0.400**, spread 0.350). Widening the
west hold-out band changes 0/31/120/236 training configs and therefore shifts the whole
training distribution, so its effect is global, not west-local. The pre-registered
interpretability gate **FAILS** and the cross-arm density curve is **not reportable**.

**The extrapolation arm is clean and is the chunk's usable result.** Within a *single* trained
model — immune to cross-arm realization noise by construction — the edit slope decays
monotonically with distance beyond the training edge and crosses the 0.80 acceptance threshold
at **+0.173 in m**:

| beyond training edge (m) | +0.106 | +0.288 | +0.511 |
|---|---:|---:|---:|
| west edit slope | 0.917 | 0.597 | 0.313 |

### The extrapolation curve is not a selection artifact

About 40% of candidate modes are dropped as unmeasurable at these absorptions, so if the
measurable-mode population shifted between the three points, the decay could be bookkeeping
rather than model behaviour. Recomputing each point over the intersection of modes measurable
at *all three*:

| alpha | n (full pool) | slope (full) | n (always-valid) | slope (always-valid) | delta |
|---|---:|---:|---:|---:|---:|
| 0.70 | 93 | 0.942 | 93 | 0.942 | +0.000 |
| 0.75 | 93 | 0.641 | 93 | 0.641 | +0.000 |
| 0.80 | 94 | 0.356 | 93 | 0.356 | -0.000 |

93 of 93-94 modes are always-valid and the two curves agree to three decimals. Per-point drop
rates are 0.3961 / 0.3961 / 0.3896. Verdict: **NO_SELECTION_BIAS**
(`outputs/p3_2c/selection_bias.json`). Ground-truth effect size rises monotonically across the
three points (6.162 -> 7.141 -> 8.204 Hz) while the model's slope falls, so the model is
under-responding progressively -- the signal is growing while the response shrinks.

## The five arms

All four interior-gap arms PASS the frozen P3-2b acceptance gate. XTRAP fails it, correctly:
its pooled S2 mixes a *trained* west with a held-out north (see below).

| arm | realized gap (m) | d_support (m) | west slope | west ρ | north slope | north ρ | frac dropped | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| W015 | 0.1613 | 0.0752 | 0.981 | 0.928 | 0.950 | 0.965 | 0.263 | PASS |
| W030 | 0.3060 | 0.1501 | 0.837 | 0.913 | 1.076 | 1.072 | 0.257 | PASS |
| W060 | 0.6027 | 0.3013 | 0.978 | 1.168 | 1.075 | 1.265 | 0.227 | PASS |
| W100 | 1.0033 | 0.5015 | 0.508 | 0.818 | 1.105 | 1.281 | 0.299 | PASS |
| XTRAP | — (beyond-edge) | 0.0001 | 0.904 | 0.444 | 1.046 | 1.182 | 0.208 | FAIL |

`d_support` is the distance from the test point to the nearest *actual* training draw, and
lands at almost exactly half the realized gap — as it must for a test point at the slab centre.

**W015 reproduces P3-2b arm C exactly** (slope 0.959, ρ 0.947, 18 cells, dropped 0.2627 —
byte-for-byte the stored values), confirming that reusing the trained arm as the sweep's first
point is exact rather than approximate.

## The two gated metrics disagree about whether a breakpoint exists

The acceptance gate scores both `edit_bw_slope ≥ 0.80` and `|ρ−1| ≤ 0.25`. On this data they
give different answers:

- **ρ**: never crosses 0.80. Minimum 0.818 at gap 1.0033 → breakpoint bounded at **> 1.0033**.
- **slope**: crosses → breakpoint at gap **0.7542**.
- Paired geometry bootstrap (4000 resamples, one index set applied to every arm):
  **49.5% of resamples show no crossing at all**.

Both are reported. Neither is "the" answer, and the disagreement is itself evidence that the
interior-gap sweep is under-powered — which the north control then explains.

## Why the control failure is a confound, not noise

north's slab is identical in every arm and its draws are byte-identical, so it was designed as
the run-to-run noise floor. Instead it tracks the manipulation monotonically:

| arm | configs changed vs W015 | north ρ |
|---|---:|---:|
| W015 | 0 | 0.965 |
| W030 | 31 | 1.072 |
| W060 | 120 | 1.265 |
| W100 | 236 | 1.281 |

The mechanism is the repair stream itself. Repairing a rejected west draw pushes it *out* of
the slab and toward the extremes of the m range, so a wider slab does not merely remove a band
— it re-shapes the marginal distribution of west absorptions the model trains on, and with it
the model's global allocation. North's rooms are the same rooms, but their west walls are not.

Consequence: **no west-specific gap effect is identifiable from this design.** A future sweep
must hold the training marginal fixed by construction (resample so the out-of-slab m
distribution matches across arms) *and* run replicate seeds, since one seed per arm cannot
separate the two explanations even in principle.

## The in-distribution confound is clear

Final validation LSD: W015 1.0132, W030 0.9560, W060 1.0217, W100 0.9517, XTRAP 0.9820. The
spread is 0.070 dB and is **not** monotone in gap width — W100, the widest gap, has the best
fit. So none of the above is attributable to the wide-gap arms simply training worse.

## XTRAP's S2 is an interpolation control, not a hold-out

west@0.50 sits at m = 0.6931, *below* XTRAP's exclusion threshold of 1.10, so it is **trained**
in that arm — `d_support = 0.0001`. Its S2-west entry is therefore an interpolation control.
Reported unflagged it would have read as a spectacular pass at the sweep's widest exclusion.
The annotation layer (`arm_holdout`) catches this mechanically; the extrapolation measurement
is the separate S2X split.

XTRAP's low west ρ (0.444) is consistent: the west cell fit spans all five alpha points
including the three extrapolation values, so it is dominated by the extrapolation decay.

## Dataset rule

No breakpoint was established on the interior-gap axis, so the sweep yields only a **bound**,
and the control failure means even that bound is not attributable to gap width. The usable rule
comes from the within-run extrapolation curve:

> Keep test material within **Δm ≈ 0.17** of the training edge. Beyond that the edit slope falls
> below the 0.80 acceptance threshold, reaching 0.313 by Δm = 0.511.

This is an *extrapolation* rule. The interior-interpolation requirement is looser — every
interior arm passed, up to a realized gap of 1.0033 — but by how much is not established here.

## Relation to P3-2's failure

P3-2 failed at an effective gap of ≈1.04 with slope 0.133, using discrete presets and raw-alpha
conditioning. W100 sits at a realized gap of 1.0033 — essentially the same hole — and scores
0.508 (slope) / 0.818 (ρ), passing the gate. The two differ in parameterization and sampling,
not in gap width, which is consistent with P3-2b's conclusion that continuous sampling in
`m = −ln(1−α)` was the operative change. P3-2c does not strengthen that claim beyond P3-2b,
because its own cross-arm axis is confounded.

## Dataset and gate

479 new simulations (92% reuse of the P3-2b corpus); 1649/1649 configs built across four arms.
Dataset gate G0–G5 PASS on every arm, with manifest deltas matching the design predictions
exactly (31/120/236/156). G4 measured per-wall selectivity of **49.7× (x) / 33.5× (y)** in the
simulations, independently reproducing P3-2's separately-measured 49.2×.

## Defect found and fixed during this chunk

The P3-2c audit **A1** fix (commit `ee6ead0`) made the entire per-cell slope regression
unreachable: the publication-policy block was inserted into `fit_cell()` instead of
`slope_fit()`, with its `return out` at function-body indentation. **Every ρ computed between
`ee6ead0` and `ad91b3a` was NaN.** It was caught because W015 is a re-evaluation of an
already-published checkpoint and reproduced every number except ρ. The A1 guard tests could not
have caught it — they asserted over already-stored `summary.json` files produced by the pre-A1
code, validating documents rather than the code that generates them.
`tests/test_p3_2b_slopefit_regression.py` now fits synthetic data with a known slope
end-to-end. No published number changed: the P3-2b corrections were made from the stored
pre-regression summaries.

## Artifacts

- `outputs/p3_2c/density.json` — the full curve, both breakpoint metrics, the bootstrap, the
  control analysis, and the reportability verdict
- `outputs/p3_2c/dataset_gate.json` — G0–G5 per arm
- `outputs/p3_2c/eval/{W015,W030,W060,W100,XTRAP}/summary.json`
- `configs/sweeps_2d_mat/p3_2c_{W030,W060,W100,XTRAP}_manifest.json`
- `aaf/data/mat_configs_p3_2c.py`, `aaf/eval/p3_2c_splits.py`, `scripts/p3_2c_density.py`
