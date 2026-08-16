# P3-2d — The edit-axis sampling law

**The edit slope rises monotonically with the sampling interval — Spearman +1.000 across five
runs — and the failure mode is systematic OVER-prediction, not loss of the edit.** The frozen
acceptance criterion is one-sided (`edit_bw_slope ≥ 0.80`) and therefore never fires in that
direction: read alone, it certifies every interval up to Δ = 0.53 as passing. Only ρ detects
the degradation. Under `rho_slab_local` the crossing of the ±0.25 calibration band sits at
**Δ\* ≈ 0.275**, bracketed by G020↔G030; under `rho_all` there is no crossing in range. That
definitional choice is unresolved and is escalated rather than made (see below).

Δ\* is **not** reported as a final number: a second seed is running at the bracketing pair.

## The curve

| run | realized Δ (m) | slope | Pearson | edit_gain | ρ_slab_local | ρ_all | frac dropped | in-dist LSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G010 | 0.1060 | 1.020 | 0.874 | 1.088 | 1.122 | **0.985** | 0.457 | 1.0610 |
| G015 | 0.1590 | 1.062 | 0.875 | 1.080 | 1.118 | 1.088 | 0.460 | 1.0314 |
| G020 | 0.1988 | 1.092 | 0.889 | 1.077 | 1.148 | 1.062 | 0.450 | 0.9871 |
| G030 | 0.3180 | 1.160 | 0.875 | 1.102 | 1.308 | 1.202 | 0.405 | 1.0066 |
| G050 | 0.5300 | 1.182 | 0.887 | 1.062 | 1.274 | 1.183 | 0.411 | 0.9792 |

x is the **realized** interval, not the nominal run label: anchoring n points inclusively on
m ∈ [0.02, 1.61] is what reproduces the intended counts 16/11/9/6/4, and it makes the realized
interval 0.1060 / 0.1590 / 0.1988 / 0.3180 / 0.5300 (D53(c) applied to a new axis).

At the finest interval the model recovers the physical law almost exactly: **ρ_all = 0.985**.

## Confound check — both confounds lean against the effect

This is the section that makes Δ\* defensible against "the coarse arms simply trained better."

**In-distribution fit quality anti-correlates with Δ (Spearman −0.900).** Final validation LSD
runs 1.0610 / 1.0314 / 0.9871 / 1.0066 / 0.9792 across increasing Δ — spread 0.082 dB, with the
**coarsest** grid fitting **best** and the finest fitting **worst**. So the coarse arms enter
the comparison with a small in-distribution *advantage* and still calibrate worse on the
hold-out. Any observed degradation is therefore conservative.

**Mode-drop rate also anti-correlates with Δ (Spearman −0.800).** Dropped fractions run
0.457 / 0.460 / 0.450 / 0.405 / 0.411: the coarse runs drop *fewer* modes, so they are scored
over a slightly larger and easier mode population — again favouring them, and again making the
degradation conservative.

Neither confound can manufacture the observed trend; both would suppress it.

## Which ρ gates this chunk — unresolved, escalated

| ρ definition | G010 | G015 | G020 | G030 | G050 | outcome |
|---|---:|---:|---:|---:|---:|---|
| `slab_local` | 1.122 | 1.118 | 1.148 | **1.308** | **1.274** | crossing at **Δ\* ≈ 0.275** (G020↔G030) |
| `all` | 0.985 | 1.088 | 1.062 | 1.202 | 1.183 | no crossing (max \|ρ−1\| = 0.202) |

`rho_slab_local` means "own-family cells on a wall that has a hold-out slab", and
`wall_has_slab` is decided by the **frozen P3-2b** `HOLDOUT_SLABS = {west, north}`. P3-2d has
**no slab** — its hold-out is the midpoint set — so in this chunk that key is simply "west and
north cells", an arbitrary half of the walls inherited from a different experiment's design.
`all` is every own-family cell, which is what "no distinguished wall" ought to mean.

The A1 publication policy forbids publishing `all`. That policy was written for P3-2b/P3-2c,
where a slab existed and `slab_local` *was* the gated quantity. Applying it unchanged here
gates on an arbitrary wall subset; overriding it silently would breach a guard that exists to
stop exactly this kind of ρ-shopping. Both are reported; the verdict is given under both.

## Second seed

`rho_slab_local` and `rho_all` are both **non-monotone at the coarse end** (G050 below G030 in
each: 1.274 vs 1.308, 1.183 vs 1.202). One inversion adjacent to the crossing is what
single-seed realization noise looks like, and P3-2c's collapse was caused by exactly this class
of unmodelled run-to-run variation. A second seed is therefore running at the bracketing pair
(G020_s2, G030_s2 — same manifests, same data, seed 20260816). Δ\* stays bracketed until it
lands.

Note that the slope curve is monotone with Spearman **+1.000** while both ρ curves are not,
which suggests the slope is the lower-variance observable here even though it is the one whose
threshold cannot fire.

## G050 ↔ P3-2 consistency

G050 samples the axis at Δ = 0.530 in m with only **four** distinct values, and scores slope
1.182 / ρ_all 1.183 — it does **not** reproduce P3-2's failure regime (slope 0.133 at an
effective gap ≈ 1.04). These are not the same quantity, so this is not a contradiction; but it
does mean P3-2's failure cannot be attributed to sampling interval alone. Four well-placed
values in the linearizing coordinate transfer to midpoints, where P3-2's comparable count of
raw-α presets did not. That sharpens D52: the operative variable looks like the **coordinate**,
with interval a secondary and much looser constraint than the chunk was designed to find.

## Deliverable sentence (provisional)

> Sample any physical edit axis at intervals no coarser than **Δ\* ≈ 0.275** in its linearizing
> coordinate — for absorption, **≥ 6 materials** spanning α ∈ [0.02, 0.80].

Provisional because it rests on the `slab_local` reading and on one seed at the crossing. Under
`rho_all` the data support no upper bound within Δ ≤ 0.53, i.e. **4 materials suffice**.

## Artifacts

`outputs/p3_2d/{sampling_law.json, dataset_gate.json, rho_definition_question.json}` ·
`outputs/p3_2d/eval/{G010,G015,G020,G030,G050}/summary.json` ·
`configs/sweeps_2d_mat/p3_2d_*_manifest.json` · `aaf/data/mat_configs_grid.py` ·
`aaf/eval/p3_2d_splits.py` · `scripts/p3_2d_sampling_law.py`
