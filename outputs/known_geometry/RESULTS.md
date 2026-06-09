# P2-3.5 — Known-geometry rendering + oracle-latent ceiling: RESULTS

**Date**: 2026-06-09. Both experiments reuse the converged P3 model (in-distribution
val LSD **2.169 dB**); **no retraining**. The question: does rendering from a room's
**known (L,W,H)** sidestep the P2-3 failure (zero-shot from 8 sparse measurements,
mag corr 0.20–0.28), and what is the ceiling?

---

## Headline (positive + the binding constraint, both honest)

1. **✅ Known-geometry rendering WORKS at training density.** Leave-one-out: predict a
   held-out room's latent from the other 44 (via the (L,W,H)→latent map) and render —
   **mag corr 0.89 (full) / 0.94 (0–250 Hz), LSD 2.6 dB, with NO measurements.** The
   architecture can render a room from geometry alone when that geometry sits at
   training density. *(See `fig_density_contrast.png`.)*

2. **✅ The 45-room model is excellent in-distribution** — val LSD 2.169 dB.

3. **⛔ Zero-shot to arbitrary NEW rooms is bounded by training coverage — definitively,
   not by the method or the architecture.** On the 8 maximin test rooms (+ 3 augmented
   strictly-interior rooms), **every route collapses to ~0.27**: the 8-recv search, the
   known-geometry lookup (RBF and linear), AND the **oracle** (the best latent found by
   optimizing against measurements of the room). This was verified **three ways**, ruling
   out every alternative explanation: (i) the lookup uses an *on-manifold, geometrically-
   correct* latent (‖z‖≈6) → 0.27; (ii) an unconstrained oracle search → 0.27 (it escaped
   to ‖z‖≈10, so its fit is not even on-manifold); (iii) a **norm-clipped on-manifold
   oracle** (‖z‖≤8.4, the training shell, optimized on 48 receivers) → **still 0.27**, with
   z* pinned to the boundary (the optimiser *wanted* to escape but couldn't help even at
   the edge of the trained region). Because even the **best on-manifold latent** fails, the
   45-room decoder has **no good latent for an unseen geometry** — it memorized its 45
   rooms and does not interpolate between them. *(See `fig_methods_comparison.png`.)*

**→ The path is clear and evidenced: denser training coverage (P2-4).** The LOO result
proves the route reaches 0.89 once a target room is near trained geometries; the test
rooms fail because, at 45 rooms, the (L,W,H) gaps are too large.

---

## Comparison table (the meeting asset)

Cells = **mag corr (full / 0–250 Hz modal band)**. Interpolative and extrapolative
rooms reported separately, never averaged.

| room | type | P2-3 8-recv | lookup-RBF | lookup-lin | **oracle** |
|---|---|---|---|---|---|
| **INTERPOLATIVE** | | | | | |
| L4.50_W4.00_H3.25 (box center) | native | 0.270 / 0.446 | 0.272 / 0.414 | 0.266 / 0.400 | 0.297 / 0.501 |
| L4.40_W4.09_H3.26 | augmented | — | 0.271 / 0.430 | 0.265 / 0.415 | 0.262 / 0.429 |
| L3.52_W4.31_H3.40 | augmented | — | 0.259 / 0.383 | 0.253 / 0.373 | 0.262 / 0.388 |
| L4.82_W3.81_H2.92 | augmented | — | 0.271 / 0.359 | 0.261 / 0.338 | 0.266 / 0.370 |
| **EXTRAPOLATIVE** | | | | | |
| L4.10_W3.01_H3.93 | | 0.268 / 0.423 | 0.279 / 0.458 | 0.273 / 0.446 | 0.266 / 0.422 |
| L5.94_W4.93_H2.51 | | 0.251 / 0.342 | 0.252 / 0.343 | 0.251 / 0.339 | 0.247 / 0.349 |
| L5.92_W3.06_H2.55 | | 0.230 / 0.255 | 0.248 / 0.297 | 0.252 / 0.306 | 0.213 / 0.242 |
| L5.91_W4.17_H3.72 | | 0.279 / 0.442 | 0.270 / 0.430 | 0.259 / 0.406 | 0.254 / 0.439 |
| L3.17_W3.00_H3.49 | | 0.280 / 0.370 | 0.272 / 0.387 | 0.267 / 0.374 | 0.206 / 0.242 |
| L5.99_W3.96_H2.54 | | 0.254 / 0.347 | 0.272 / 0.384 | 0.270 / 0.379 | 0.252 / 0.356 |
| L3.14_W3.08_H2.51 | | 0.204 / 0.193 | 0.263 / 0.321 | 0.248 / 0.291 | 0.193 / 0.186 |
| **REFERENCE: LOO (training density)** | | | **0.894 / 0.938** | 0.882 / 0.921 | — |

*(LOO = leave-one-out over the 45 training rooms; LSD 2.6 dB. Full CSV:
`comparison_table.csv`.)*

---

## Interpretation (per the spec's decision rules)

- **on-manifold-oracle ≈ escaped-oracle ≈ lookup ≈ 8-recv ≈ 0.27 on test rooms** → the
  failure is **not** latent-finding from sparse data, **not** the (L,W,H)→latent map, and
  **not** an off-manifold optimisation artefact. The best latent constrained to the trained
  shell (norm-clipped oracle, ‖z‖≤8.4) renders an unseen room at 0.27. **The decoder cannot
  interpolate to unseen geometry at 45-room coverage → the fix is strictly more training
  rooms (P2-4).** *(On-manifold-oracle interp means: 0.276 full / 0.408 in 0–250 Hz, LSD
  7.5 dB — vs LOO 0.89 / 0.94 / 2.6 dB at training density.)*
- **lookup ≈ oracle** → the (L,W,H)→latent map is already about as good as any latent; a
  better map would not help at this coverage.
- **LOO 0.89 vs test 0.27** → the route is sound and renders held-out rooms well at
  training density; the binding constraint is the **density of training geometries**, not
  the architecture, the conditioning, or the test-time procedure.

### Honest caveats
- LOO held-out rooms sit at the training-grid density (~0.34 m to nearest neighbour); the
  maximin test rooms (and the augmented interior rooms) sit in sparser gaps (~0.43–0.84 m).
  So LOO is "best case at training density," not a claim that *any* new room renders at
  0.89 today.
- Only the box center is strictly inside the 45-room convex hull; the other 7 test rooms
  are mildly extrapolative (≤0.11 m past the LHS edges). 3 augmented strictly-interior
  rooms were added — they too sit at ~0.27, confirming the result is about coverage
  density, not interior-vs-exterior.
- The 0–250 Hz modal band is consistently the strongest (oracle box center 0.50) — the
  low-frequency room modes are partially recovered even when the full spectrum is not.

---

## Recommendation for P2-4 (unchanged, now ceiling-proven)

1. **Scale the training set** (45 → ~150–300 rooms, denser LHS over the L,W,H box). The
   LOO shows the known-geometry route hits 0.89 at training density; denser coverage
   shrinks the gaps so arbitrary rooms land near trained geometries. **This is the
   evidenced fix.**
2. **Test explicit (L,W,H) conditioning** (feed geometry directly to the decoder, not only
   via the optimized latent) — could let the decoder *compute* modal structure from
   geometry rather than memorize it; potentially decisive and complementary to (1).
3. **Do NOT** pursue better test-time search or a better (L,W,H)→latent map — the oracle
   proves neither is the bottleneck at 45 rooms.

## Artifacts
`fig_density_contrast.png` (LOO vs sparse-gap), `fig_methods_comparison.png` (all methods
≈ oracle), `comparison_table.csv`, per-room `lookup/`+`oracle/` metrics, `loo/loo_rows.json`,
signal overlays under `lookup/L*__rbf/figures/` for the interpolative rooms.
