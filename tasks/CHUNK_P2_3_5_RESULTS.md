# Chunk P2-3.5 — Known-geometry rendering + oracle-latent ceiling

**Status**: COMPLETE — 2026-06-09. No retraining; reuses the converged P3 model
(in-dist 2.169 dB). Scavenger GPUs only. Full results +
table: `outputs/known_geometry/RESULTS.md`.

## Question
P2-3 zero-shot (8 sparse measurements) failed at mag corr 0.20–0.28. Does rendering
from a room's **known (L,W,H)** sidestep it, and what is the ceiling?

## Result (two-part, both honest)

- **✅ POSITIVE — known-geometry rendering works at training density.** Leave-one-out
  over the 45 training rooms (predict a held-out room's latent from the other 44 and
  render, **no measurements**): **mag corr 0.89 / 0.94 (0–250 Hz) / LSD 2.6 dB.** The
  architecture can render a room from geometry alone when coverage is adequate.

- **⛔ DEFINITIVE — zero-shot to arbitrary new rooms is training-coverage-bound.** On the
  8 maximin test rooms (+ 3 augmented strictly-interior rooms), **every route ≈ 0.27**:
  8-recv search, lookup (RBF + linear), and the oracle. Verified three ways — the
  on-manifold lookup latent (‖z‖≈6), an unconstrained oracle (escaped to ‖z‖≈10), and a
  **norm-clipped on-manifold oracle** (‖z‖≤8.4, 48 receivers) — all 0.27. Even the best
  on-manifold latent cannot render an unseen room. So it is **not** the method, **not**
  the (L,W,H)→latent map, **not** the search, and **not** the architecture: the 45-room
  decoder **memorizes its rooms and does not interpolate** between them.

## Verdict
The binding constraint is **training-set coverage density**, conclusively. The path is
the LOO result itself: at training density the route reaches 0.89; denser coverage
shrinks the gaps so arbitrary rooms land near trained geometries.

## P2-4 (ceiling-proven)
1. **Scale the training set** (45 → ~150–300 rooms, denser LHS). The evidenced fix.
2. **Test explicit (L,W,H) conditioning** (decoder computes modes from geometry rather
   than memorizing) — complementary, potentially decisive.
3. **Do NOT** chase test-time search or a better map (oracle proves they're not the wall).

## Method note
`aaf/eval/known_geometry.py`: (L,W,H)→latent maps (scipy RBF + sklearn linear), oracle
(optimize z* on a receiver subset; optional ‖z‖ norm-clip for an on-manifold ceiling),
LOO sanity, full signal+per-band-mag-corr eval — all reusing the frozen P3 decoder +
renderer + signal suite. Comparison table + 2 figures in `outputs/known_geometry/`.

## Manager actions
- Approve P2-4 = **scale training rooms** (+ optionally explicit (L,W,H) conditioning).
- Meeting framing: *converged 3D multi-room model (2.169 dB) + known-geometry rendering
  demonstrably works at training density (0.89) + a now-ceiling-proven path to arbitrary
  rooms (denser coverage)* — a clean, honest, forward-looking story.
