# CHUNK FT-1 — Results

**FT-A: GO-WITH-CHANGES.** A 2D FDTD wave solver was built and passes all ten correctness
gates, and the cost probe clears the budget by 52×. But adversarial review found four blocking
issues, none of them in the solver's numerics and all of them in what the validation *covers*.
FT-B and FT-C were **not run**: three of the four blockers bear directly on them, so running
them now would produce results whose meaning is already known to be in question.

All three independent reviewers (numerics / physics / measurement lenses) refuted the reported
GO. Their agreement on the solver itself is worth quoting:

> Everything I tried to break in the solver held. The GO on correctness and cost is sound; the
> refutation is of the report's physics reasoning and two gate thresholds, not of the solver.

## The gates that stand

| gate | result | measured |
|---|---|---|
| A0 | PASS | max per-step energy rise 3.51e-16 relative; drift −2.63e-15 over 24263 source-free steps |
| A1a | PASS | worst 0.0233% vs continuous analytic (tol 1%) |
| A1b | PASS | worst 0.0198% vs exact SLF discrete dispersion (tol 0.05%) |
| A1c | PASS | min mode-shape Pearson 0.999925 (tol 0.99) |
| A2 | PASS | worst +8.37% vs Kuttruff 2.843/2.957/3.867 Hz (tol 10%) |
| A4a | PASS | max Δf 0.0236%, ΔBW 0.0361% under grid refinement (tol 2%) |
| A4b | PASS* | max Δf 0.243%, ΔBW 0.557% (tol 2%) — *but see M4* |

**Cost probe (measured, not estimated):** 0.83 s per 2 s room on one pinned CPU core →
**0.231 CPU-h per 1000 configs** against a 12 CPU-h budget (**52× headroom**). Interior
structure — a doorway divider, an absorber patch, or both — costs **nothing measurable** (2.8%
spread across three geometries), because it runs through the same dense code path as an empty
box. That is the load-bearing cost fact for the topological edit axes.

## Blocking issues

**B1 — The validation covers one geometry, and it is the only on-grid one.** Every gate ran at
L=4.5, W=4.0. Verified independently: **39 of 40 training and 9 of 10 test geometries are not
integer multiples of dx=0.05**, and `_grid_count` merely `warnings.warn`s before snapping them.
L=3.68 becomes 73.6 cells → snaps to 3.65 or 3.70, a 0.5–0.8% dimension error and a
proportional mode-frequency shift — 10–16× the A1b tolerance the solver was validated to. The
solver is accurate to 0.02% on the one room that fits the grid and would be ~30× worse on
almost every room we actually have.
*Fix:* define the wave-track geometry family **on** the dx grid from the start (FT-B/FT-C need
new datasets regardless), and make `_grid_count` raise by default. dx=0.01 would divide every
2-dp dimension but costs 125× (28.8 CPU-h/1000), breaking the budget — so snapping the rooms,
not the grid, is the affordable direction.

**B2 — The absorber patch is dx wider than requested**, and `_apply_patch` reports the nominal
span rather than the realized one. This is the FT-C independent variable, mis-reported at
source.

**B3 — Both new edit parameters are quantized to dx = 0.05 m** and therefore cannot be sampled
continuously. This is the one that matters strategically: **P3-2b's headline finding was that
continuous sampling in the linearizing coordinate — not the conditioning design — was the
operative variable.** A dx-quantized aperture or patch axis forces preset sampling, which is
precisely the regime P3-2b showed fails. This is a design constraint on the next chunk, not a
bug, and it needs an answer before either axis is trained.

**B4 — A3's invariance claim is false.** The gate rested on "BW(1,0)/BW(0,1) is exactly 2,
independent of L, W, α and c". The exact ratio is `2·artanh(ξ)/ξ`, which is α-dependent. The
gate passed (2.1267 at α=0.7) against an exact value of 2.0516 — i.e. **it passed for the wrong
reason**, and at higher α it would fail a correct solver.

This is the third time in this project that a proposed gate would have mis-scored correct
physics, and the first where a gate *passed* on a wrong target. The pattern is consistent
enough to be worth a standing rule: derive the gate's expected value from the governing law at
the exact configuration being run, and check its invariances symbolically rather than assuming
them.

## Major issues

- **M1** — A2b is not a cross-validation of two solvers. No ISM simulation is run anywhere in
  FT-A; `BW_ISM` is the analytic `ism_ray` formula, so A2b is an algebraic restatement of A2.
  *This was an error in the task specification I wrote, not in its execution.*
- **M2** — `kappa_fdtd = 1.0208` vs `kappa_ism = 1.6608` compares slopes fitted against
  **different regressors**. It is a change of damping law, not a property of the estimator. The
  constant downstream actually consumes, `d(BW)/d(m_wall)`, measures ≈1.005 on FDTD. The
  headline "factor of 1.63" should not be carried forward as an estimator recalibration.
- **M3** — Neither edit axis FT-A exists to authorize has a physics gate. The absorber patch is
  exercised only by the cost probe.
- **M4** — A4b normalizes the grid-refinement change by absolute frequency (~89 Hz) instead of
  by the aperture-induced shift (~2–3 Hz), diluting by ~30× the very quantity it exists to
  probe. Its PASS is therefore much weaker than it reads.
- **M5** — The cost headline is quoted for a configuration validated only to 120 Hz; the repo's
  evaluation protocol scores 0–300 Hz.

## What FT-A does establish

The wave solver is real, correct on the physics it was tested against, and cheap — 52× inside
budget with topological structure free. That was the question blocking the doorway and
absorber-patch axes, and the answer is yes. What is *not* established is that it is ready to
run on this project's existing room family, or that either new edit axis can be sampled the way
P3-2b showed is necessary.

## Recommended sequence before FT-B/FT-C

1. Snap the wave-track geometry family to the dx grid; make off-grid dimensions raise.
2. Fix `_apply_patch` to report realized extent, and re-derive A3's target as `2·artanh(ξ)/ξ`.
3. Answer B3: either accept a quantized edit axis and test whether the P3-2b finding transfers
   to it, or find a formulation (sub-cell boundary interpolation) that restores continuity.
4. Add a physics gate for each edit axis before either is trained.

## Artifacts

- `aaf/sim/fdtd_2d.py` — the solver (SLF leapfrog, Kowalczyk–van Walstijn locally-reacting
  boundary, one code path for outer walls / dividers / aperture edges / corners)
- `tests/test_fdtd_2d.py` — CPU unit tests (CFL assertion, zero-DC source, energy, reciprocity,
  rigid-limit boundary)
- `scripts/ft1_a_validate.py`, `outputs/ft1/solver_validation.json` — the ten gates, the cost
  probe, and the recorded adversarial review with the corrected verdict
