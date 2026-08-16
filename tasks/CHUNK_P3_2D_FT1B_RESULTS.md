# CHUNK P3-2d / FT-1b — Results (IN PROGRESS)

**Δ\* ≈ 0.275 in the linearizing coordinate, provisional and held for a second seed.** The edit
slope rises monotonically with the sampling interval (Spearman **+1.000** over five runs) and
the failure mode is systematic **over**-prediction, not loss of the edit — so the frozen
one-sided criterion `edit_bw_slope ≥ 0.80` never fires and would certify every interval up to
Δ = 0.53 as passing. Only ρ detects the degradation. **FT-C: NO-GO**, but on the antinode
criterion alone — the position effect itself is real at 10.8× the estimator floor. **The solver
is not yet cleared for training use**: A0a/A0b/A3-fix pass, but A2b is PARTIAL and A0c shows the
aperture axis needs a finer grid than planned. **FT-B has not run.**

## Part 3 — the sampling law

| run | realized Δ (m) | slope | Pearson | gain | ρ_slab_local | ρ_all | frac dropped | in-dist LSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G010 | 0.1060 | 1.020 | 0.874 | 1.088 | 1.122 | **0.985** | 0.457 | 1.0610 |
| G015 | 0.1590 | 1.062 | 0.875 | 1.080 | 1.118 | 1.088 | 0.460 | 1.0314 |
| G020 | 0.1988 | 1.092 | 0.889 | 1.077 | 1.148 | 1.062 | 0.450 | 0.9871 |
| G030 | 0.3180 | 1.160 | 0.875 | 1.102 | 1.308 | 1.202 | 0.405 | 1.0066 |
| G050 | 0.5300 | 1.182 | 0.887 | 1.062 | 1.274 | 1.183 | 0.411 | 0.9792 |

x is the **realized** interval, not the nominal label (D53(c) on a new axis). At the finest
interval the model recovers the physical law almost exactly: ρ_all = 0.985.

**Both confounds lean against the effect.** In-distribution fit anti-correlates with Δ
(Spearman −0.900; coarsest fits *best*), and mode-drop rate anti-correlates too (−0.800; coarse
runs drop *fewer* modes). Coarse arms enter with an advantage on both axes and still calibrate
worse on the hold-out, so the degradation is conservative.

**Δ\* is not final.** Under `slab_local` the ±0.25 band is crossed at 0.2746 (bracket
G020↔G030); under `rho_all` there is no crossing (max |ρ−1| = 0.202). Both ρ curves are
non-monotone at the coarse end, adjacent to the crossing — the signature of single-seed
realization noise, which is what destroyed P3-2c. Second seed running at that pair.

**Which ρ gates this chunk is escalated, not decided.** `slab_local` here means "west and north
cells", inherited from the frozen P3-2b `HOLDOUT_SLABS`; P3-2d has no slab. The choice flips
the verdict (Δ\* ≈ 0.275 vs no bound). See `outputs/p3_2d/rho_definition_question.json`.

**G050 ↔ P3-2:** four values at Δ = 0.530 score 1.182 and do **not** reproduce P3-2's failure
regime (0.133). P3-2's failure was therefore not a sampling-interval failure — which sharpens
D52 toward the **coordinate** being the operative variable, with interval a looser secondary
constraint.

## Part 1 — solver corrections

| gate | verdict | evidence |
|---|---|---|
| A0a per-axis exact fit | **PASS** | L=3.68×4.03 frequency error −0.54% (snap) → **1e-14%** (fit); 4.5×4.0 bit-unchanged |
| A0b patch extent | **PASS** | 0.40 m request realized 0.40 m (was 0.45 m, +dx) |
| A3-fix | **PASS** | vs exact `2·artanh(ξ)/ξ`: +0.09% (α=0.30), +3.24% (α=0.70) |
| A2b κ-free | **PARTIAL** | α=0.15 validated (2.81% median); α=0.70 grazing unresolved |
| A0c quantization | **B3 NOT closed** | aperture moves **10.4×** the floor between dx 0.02→0.01; patch 0.31× |
| A0d off-grid gates | not run | — |

`fs` stays frozen at 12288 rather than recomputed per room (spec said otherwise): exact fitting
moves dx by only 0.656%, worst anisotropic CFL 0.794 vs bound 1.0, and a per-room `dt` would
destroy the exactly-0.5 Hz bin grid `fs=12288` exists to provide.

**A2b detail.** Rebuilt κ-free by construction (T1 frequencies, T2 Schroeder EDC decay in dB/s,
T3 within-solver ratio). Stratified by incidence on the absorbing wall: uniform α=0.15 agrees to
2.81% median; α=0.70 oblique (n_x≥1) T2 14.55% / T3 11.05%; α=0.70 grazing (n_x=0) T2 61.55% /
T3 66.98%. FDTD is more damped at grazing — the correct direction — but the quantitative test
fails: observed FDTD/ISM 1.06–1.62 vs predicted Kuttruff/ism_ray 1.34–2.09 (r=0.58). So "it's
just grazing physics" is **not** established and a scoped PASS would be rationalizing.

**B4 severity corrected downward.** The limit value 2 is off by only −0.26% (α=0.30) and −2.91%
(α=0.70), both inside the original ±10% band; it would not break that band until α > ~0.9. The
original gate would *not* have failed a correct solver — it passed for a slightly wrong reason.

## Part 2 — FT-C only

**NO-GO, on the antinode criterion alone.**

| | measured | threshold |
|---|---:|---|
| position residual | 0.43 Hz = **10.8×** the P3-2b floor | ≥5× — **passes** |
| antinode Pearson r | **0.507** (1seg 0.427, 4seg 0.345) | ≥0.70 — **fails** |

Position carries substantial signal beyond area-weighted mean α: at *fixed extent*, moving the
patch changes per-mode bandwidth by ~0.43 Hz against modal widths of 3–7 Hz. What fails is the
first-order *model* — mean pressure² over the patch predicts it poorly. A conditioning design
built on antinode overlap would be built on the wrong feature.

**FT-B not run.** Gated on A2b (aperture coupling is exactly where a mis-calibrated admittance
would distort the result) and now additionally on A0c, which shows it needs dx ≤ 0.01.

## Dataset

5145 unique configs built; gate H1–H6 PASS on all five grids. The collision-aware sampler was
required, not optional: a naive grid substitution produces **27 duplicate filenames of 960**,
and the trainer does not check uniqueness. Two hazards surfaced rather than hidden — G010/G030
each place a midpoint 0.0165 from the always-trained baseline (excluded from the headline as a
trained-value control), and G020 has a grid value 0.0085 from the α=0.70 preset.

## Still open

1. **A2b α=0.70 grazing shortfall** — research call; three candidates (KvW grazing
   under-absorption, poor analytic predictor, ISM truncation).
2. **Which ρ gates P3-2d** — flips the headline.
3. Second seeds at G020/G030 (~9 h remaining), then final Δ\*.
4. A0d, the A3 drop diagnostic, FT-B at dx ≤ 0.01.
