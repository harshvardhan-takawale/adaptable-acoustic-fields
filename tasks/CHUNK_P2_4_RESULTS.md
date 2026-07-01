# Chunk P2-4 — Coverage-density scaling curve — RESULTS (⏳ IN PROGRESS)

> **STATUS: PARTIAL — 2 of 4 densities measured (45, 90 done; 150, 250 training).**
> This file exists so the SCALING.md / CONTEXT_FOR_MANAGER pointers resolve and so the
> pre-publication verification findings are recorded. **Do not draw a scaling/saturation
> conclusion from it yet.** It will be finalized (all 4 points + full analysis) when the
> autonomous SLURM chain completes. Live numbers: `outputs/coverage_curve/SCALING.md`.

## Goal (recap)

Turn the two P2-3.5 anchors (sparse-45 → 0.27; LOO/training-density → 0.89) into a *measured*
curve: train the frozen P3 recipe at **45 / 90 / 150 / 250 rooms** (room count the ONLY
independent variable, nested 45⊂90⊂150⊂250) and at each density evaluate **known-geometry
zero-shot** rendering (predict latent from (L,W,H), render, no measurements) on a **single FROZEN
interior test set** (15 rooms strictly inside the 45-hull → interpolative at every density; reused
by P3-1). Decisions **D39** (nested maximin augmentation), **D40** (frozen interior test set),
**D41** (frozen recipe + lean budget 90/150/250 = 60K/70K/85K, user-approved) in DECISIONS.md.

## Measured so far (2/4) — headline known-geometry route

| rooms | mean NN-dist (m) | in-dist val LSD | mag corr full | mag corr modal (0–250) |
|---:|---:|---:|---:|---:|
| 45  | 0.260 | 2.17 dB @60K | 0.273 | 0.409 |
| 90  | 0.236 | 3.31 dB @60K | 0.345 | 0.625 |
| *LOO ceiling* | — | — | *0.894* | *0.938* |

**Direction is up**: doubling rooms moved full-band +0.07 and the **modal band +0.22**
(0.409 → 0.625) — the strongest response in the hardest (sub-Schroeder modal) band.

## Load-bearing caveat — the whole curve is a fixed-budget LOWER-BOUND envelope

The iteration budget is lean and **fixed** (45/90→60K, 150→70K, 250→85K) while room count rises,
so per-room sample exposure **falls** as density grows. Consequently in-distribution convergence is
NOT held constant:
- **Neither the 45 nor the 90 point is converged.** Both val-LSD trajectories are still strictly
  descending at the final iteration (last-10K relative improvement **5.3%** at 45, **3.9%** at 90 —
  both ~13× above the 0.003 early-stop threshold; early-stop never fired, both hit the iter cap
  mid-descent). So the 45 baseline is *not* a converged reference — it is itself a (tighter) lower
  bound.
- Every zero-shot point is therefore a **lower bound**; the higher a point's in-dist LSD, the looser
  the bound. 150/250 (70K/85K over more rooms) will be looser still and not directly iso-convergence
  comparable to the 60K 45/90 points.
- **This makes the coverage signal conservative, not inflated**: fidelity climbs as rooms rise *despite*
  falling per-room exposure. Per D41 we do NOT re-tune — the confound is reported, not fixed.

## Pre-publication verification (adversarial, 7 dimensions + completeness critic)

Verdict: **PASS-WITH-WARNINGS**. Mechanical integrity confirmed by independent re-derivation from
source: frozen test set intact (15 rooms, monotone NN 0.260→0.178, distinct from all 250 train,
`frozen_note` present); nesting 45⊂90⊂150⊂250 holds on disk and train_rooms_45 == the pre-existing
45; recipe frozen (only `run_id`, `rooms_yaml`, `n_iters` differ across density configs — 21 other
hyperparameters byte-identical); both measured points reproduce to 3 decimals; scaling builder's
15-frozen-room few-shot filter correctly excludes the 8 legacy maximin rooms. Open items carried
into finalization:
- **W-1 (metric comparability, footnoted in SCALING.md):** the secondary few-shot mag corr is over
  the 504 held-out receivers; the headline known-geometry mag corr is over all 512. Same function,
  band, pooling — only the receiver population differs. Near-parity (few-shot 0.355 slightly *above*
  known-geometry 0.345 at 90 rooms) must NOT be read as "few-shot wins" — it is the harder unfitted set.
- **W-2 (chain fragility, low-probability):** the 150/250 evals + few-shot jobs gate `afterok` on the
  final `afterany` resume of each training; if that resume TIMEOUTs (non-zero exit) the eval + few-shot
  cascade to `DependencyNeverSatisfied`. Quantified as low-risk: observed throughput ≈2000 iters/hr, each
  density gets main + 2 resumes ≈ 72h of walltime, and 85K iters needs only ~43h — so the last resume is a
  clean no-op that COMPLETES (as 7047710 did for the 90-chain: 1-min exit-0). Failure needs throughput to
  fall below ~1180 iters/hr (a ~1.7× margin). Further mitigated by the live chain monitor (detects a
  stalled chain → resubmit). Not hardened by re-pointing deps — that would touch 33 pending jobs and add
  more fat-finger risk than the low-probability failure it prevents.

## Pending before finalization
- density_150 (70K) + density_250 (85K) trainings → known-geometry + few-shot evals on the frozen set.
- Final `outputs/coverage_curve/SCALING.md` + `scaling_curve.png` (4 points).
- Saturation analysis (is 250 enough / extrapolate toward the 0.89 LOO ceiling); per-room NN-distance
  view (does fidelity track each room's distance to its training set); explicit P3-1 setup (reuse this
  frozen test set + the 4 density baselines); recommendation on geometry conditioning (P3-1) vs
  densification data-cost; manager actions.
