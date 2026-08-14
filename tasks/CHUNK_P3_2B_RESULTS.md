# Chunk P3-2b — material-editable field, corrected conditioning — RESULTS

**S2 PASSES, and the cause is the training distribution — not the fit, and not (mainly) the coordinate.** On unseen geometry with a **never-seen (wall, absorption) combination**, arms B, C and D clear all four frozen acceptance thresholds (slope ≥ 0.80, Pearson ≥ 0.80, edit_gain > 1, |ρ−1| ≤ 0.25); arm A fails. The attribution is unambiguous because **A → B changes only the training distribution** — identical encoder, identical renderer, identical everything else — and the S2 edit slope jumps **0.153 → 1.147**. P3-2's failure was caused by sampling α at 3 preset values on single walls, giving 11 distinct α-vectors in a 4-D space; sampling continuously in m = −ln(1−α) fixes it.

| arm | training data | encoder | in-dist LSD | S2 slope | S2 r | edit_gain | ρ (slab-local, κ-scaled) | **S2** |
|---|---|---|---:|---:|---:|---:|---:|:---:|
| **A** | P3-2's 440 presets | α-Fourier | **0.931** | 0.153 | 0.499 | 0.868 | **0.509** | **FAIL** |
| **B** | 960 continuous | α-Fourier | 0.998 | 1.147 | 0.871 | 1.084 | 1.039 | **PASS** |
| **C** | 960 continuous | **m_linear** | 1.013 | **0.959** | 0.868 | 1.087 | **0.947** | **PASS** |
| **D** | single-wall only | m_linear | 1.464 | 1.038 | 0.871 | 1.082 | 1.031 | **PASS** |

## 1. What the ablation settles

**(a) Continuous sampling is the fix (A → B).** Same 64-d α-Fourier encoder, same `n_pts_per_ray=64` renderer; only the data differs. Slope 0.153 → 1.147, Pearson 0.499 → 0.871, edit_gain 0.868 → 1.084.

**(b) The m-coordinate is not necessary, but it is materially more accurate (B → C).** Both pass; C lands closer to unity on both calibration measures — slope **0.959 vs 1.147** and ρ (slab-local) **0.947 vs 1.039**. B systematically **overshoots** the edit by ~15%. The clearest number in the chunk: **outside the held-out slab, arm C's fitted slope is ρ = 0.99991 of κ-scaled theory** — it recovers the physical law to within 0.01%. So the coordinate buys calibration, not capability.

**(c) Multi-wall training is not necessary (D).** D trains on single-wall configs only and still passes; it does cost in-distribution fit (1.464 vs 1.013).

**(d) Fit quality is again ANTI-correlated with edit transfer.** Arm A has the **best** in-distribution val LSD of all four (0.931) and is the **only** failure. This is the second independent refutation of P3-2's stated recommendation ("a sharper recipe, not a different conditioning design") — a recommendation this agent wrote and which was wrong. The renderer fix (`n_pts_per_ray` 32 → 64) is present in *every* arm here, including the failing one.

## 2. The sharpest evidence: ρ inside vs outside the held-out slab

| arm | ρ outside slab | ρ **inside** slab |
|---|---:|---:|
| A | 1.060 | **0.509** |
| B | 1.083 | 1.039 |
| C | **0.99991** | 0.947 |
| D | 1.018 | 1.031 |

Arm A **learns the law correctly everywhere except where it is tested** — full strength outside the held-out band, half strength inside it. That is compositional failure localized to the exact region the holdout defines, and it is precisely what B/C/D pass straight through. No arm tripped the measurability blocker (`frac_modes_dropped` 0.42–0.46, threshold 0.50).

## 3. Why the P3-2 holdouts were not comparable tests (verified)

P3-2 held out `(west, 0.50)` and `(north, 0.70)`. Per-wall trained absorptions were west {0.05, 0.15, **0.70**} and north {0.05, 0.15, **0.50**} — so `(west, 0.50)` was an **interpolation** while `(north, 0.70)` was an **extrapolation** beyond that wall's maximum. P3-2b's slabs are both strictly interior to the sampled range, making S2 a pure composition test on each wall's own axis.

## 4. Method integrity

- **The κ correction was decided before any result existed and is independently confirmed.** The estimator measures a *calibrated* −3 dB width, so the theoretical slope of a measured Δbandwidth is `κ·c/(4πD)`, not the raw `c/(4πD)`. Fitting the 240 simulated ground-truth m-response points across **12 (geometry, wall) cells** gives mean ρ = **1.0050**, range [0.947, 1.084], every **r² ≥ 0.9996**; against raw theory the same GT fits read ρ = 1.669. Using the spec's literal raw formula would have made a *perfect* model score 0.602 — and on a mid-training checkpoint it made an unconverged model read ρ = 0.980 (a **false pass**). Both values are in the JSON (`rho_vs_raw_theory_median`).
- **Thresholds were frozen and hashed** (`a8479c5e1dcc…`) with a test pinning the hash, so they could not be softened after seeing results. The verdict is emitted before any figure is drawn.
- **The estimator and controls C1–C4 are reused verbatim** from P3-2, asserted by function-identity tests, so the numbers are comparable across chunks.
- **Q15 was bounded, not fixed**: the edit estimator imports only `bin_index_for_freq` and `spatial_correlation_complex`, neither of which touches the buggy grid reconstruction. No P3-2 number moved.
- **Dataset gate PASS**: 0 slab violations, 0 preset collisions, uniform m coverage outside the slabs and empty inside, and the block-diagonal physics signature reproduced on new-generator configs (selectivity 33.3 and 18.7).

## 5. Scoping (corrects D48)

The block-diagonal structure is **genuine physics**: an axial family's damping is dominated by the two walls perpendicular to it, and grazing-incidence absorption on a locally-reacting surface tends to zero as θ→90°. What is set by the simulator is the **magnitude** of the selectivity ratio (~29:1 here). The supportable claim is therefore: **the representation learns whatever per-wall absorption law it is trained on** — demonstrated here by recovering that law to within 0.01% outside the held-out band and passing through the band itself.

## 6. Recommendation

Adopt **arm C** (continuous sampling + m-coordinate) as the design: it is the best-calibrated of the passing arms and the only one that recovers the theoretical slope essentially exactly. Note for planning that **the sampling distribution, not the conditioning parameterization, was the binding constraint** — the same lesson generalizes to the 3D pipeline, where P3-1's geometry conditioning was trained on a similarly sparse grid.

---

## Correction (2026-08-14, P3-2c audit A1)

**What was wrong.** The arm table above and the §1(b) narrative originally printed
`slope_fit.aggregate.own_family.**all**.rho_median` — a diagnostic that pools slab and non-slab
cells — while the acceptance gate correctly used
`slope_fit.aggregate.own_family.**slab_local**.rho_median`. The two differ:

| arm | printed (rho_all) | used by the gate (rho_slab_local) |
|---|---:|---:|
| A | 0.887 | **0.509** |
| B | 1.045 | 1.039 |
| C | 0.971 | 0.947 |
| D | 1.031 | 1.031 |

**Why it mattered.** Arm A's row showed ρ = 0.887 — *inside* the ±0.25 acceptance band — directly
beside a **FAIL** verdict, so the flagship results doc contradicted its own gate. A reader checking
the ρ criterion against that row would have concluded arm A passed it.

**What did NOT change.** No verdict moves. The gate always consumed `rho_slab_local` (recorded in
each `summary.json` as `verdict.rho_used`), and `outputs/p3_2b/EVAL.md`, `ablation.json` and all five
figures were already correct. Only this document was affected.

**Guard.** `slope_fit` now emits `rho_published` (aliased to `slab_local`) plus a
`publication_policy` block naming `all` as diagnostic-only, and `tests/test_p3_2c_publication.py`
asserts (a) `rho_published == verdict.rho_used` for every summary on disk and (b) every ρ printed in
a human-facing document matches `slab_local` to 3 dp. The `all` aggregate is no longer published.

