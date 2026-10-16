# Chunk P4-7 — Configuration diversity and a physics modal prior: RESULTS (live)

**Status (2026-10-16, 14:40):**
- m1 is done (D170): **physics carries the field shape, so m3 launches.**
- m2 (A-fam, A-500) is training on tron and is due ~02:30.
- m4: the CX500 corpus is built and its arm GX is queued on tron.
- m3 is pre-registered (D170(g)) and being implemented.
- G3 is preempted at 77K and waiting to requeue.

Pre-registration is D169; m1 results and the m3 pre-registration are D170. Both are in [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md).

**The chunk's premise** (manager): in family, 200–300 Hz transfer is data-limited (g1). Out of family, the 25–100 Hz field SHAPE is wrong (t1, t2, p3, g2), and the corpora fix the source at (0.5, 1.6) and the L notch at NW (p1).

---

## m1 — physics-only modal baseline (CPU, no training): m3 LAUNCHES
[m1_modal.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/m1_modal.md) · [m1_modal.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/m1_modal.json) · [script](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_7_m1_modal.py)

**Setup:**
- **Operator:** eigenpairs of the solver's own discrete Laplacian (dx 0.02, the solver's mask, image-node Neumann walls), every mode to 600 Hz including the zero mode, normalised to ∫ψ² dA = 1.
- **Transfer function:** H = C·S(f)·e^{iωdt}·Σ ψₙ(x_r)ψₙ(x_s) / (kₙ² − k² + i k ηₙ), with ηₙ = β∮ψₙ² ds and β = tanh(m/4).
- **Calibration:** one complex C fitted on a C250 training rectangle, then frozen.
- **Scoring:** off-lattice, above 20 Hz, per-band r (dB, per-bin column Pearson) and LSD.

| rooms | modal, primary formula: r | LSD dB | modal + exact zero mode: r | LSD dB | G1 @60K r | PB3 @60K r |
|---|---|---|---|---|---|---|
| 15 test shapes (in family) | 0.922 / 0.923 / 0.936 | 1.6 / 1.6 / 1.4 | 0.988 / 0.976 / 0.960 | 0.6 / 0.8 / 1.0 | 0.931 / 0.721 / 0.581 | 0.895 / 0.647 / 0.496 |
| **T, DN, U (out of family)** | **0.923** / 0.928 / 0.941 | 1.7 / 1.5 / 1.3 | **0.978** / 0.974 / 0.962 | 0.9 / 0.8 / 1.0 | 0.628 / 0.372 / 0.358 | 0.600 / 0.347 / 0.329 |
| U only | 0.914 / 0.927 / 0.936 | 1.8 / 1.5 / 1.4 | 0.973 / 0.972 / 0.960 | 1.0 / 0.9 / 1.1 | 0.718 / 0.434 / 0.402 | 0.631 / 0.320 / 0.336 |

- **Pre-registered gate:** out-of-family 25–100 Hz r **0.923 ≥ 0.85**, so **m3 launches**. In family it reads 0.922, which is not poor.
- **Physics is family-blind:** out-of-family rooms score the same as in-family ones. The network's out-of-family deficit is a deficit of the learned representation.
- **Calibration is physical:** fitted C = 3.2055 + 0.038i against the analytic dA/(c dt)² = 3.2086 (within 0.1%). The sign "+ i k η" is confirmed (residual 0.124; the other sign gives 0.999).
- **Diagnostics:**
  - Mode counts match Weyl's law within ±3.
  - Operator asymmetry is 0 in rectangles and ~1e-3 in notched rooms (D65), at no measurable cost.
  - Modes to 800 Hz change r by ≤ 0.001.
  - Scaling η by ×0.5 or ×2 loses 0.02–0.09 r: the analytic damping is at the optimum.
- **The gap between the primary formula and exact physics is entirely the zero mode.** Its continuous term differs from the zero mode's real content in the 2 s record by ~96% above 20 Hz (the slowly relaxing step; D101, p2). Replacing that one term with its exact record (one scalar recursion per room) closes the whole gap.
- **D101 refinement:** the DC level is 1/(βP) exactly: corr 1.000 across 30 rooms, |H(0)|·P spread 0.1%. The 1/area law (corr 0.905 here) was a proxy through A–P covariance.
- **Eigen-solve time** (to 800 Hz, 4 threads): median 102 s, range 34–309 s per room. The FDTD loop for the same dense rooms takes a median of 58 s.
- **Caveat** (labelled; [OPEN_QUESTIONS Q24](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md)): this is the solver's own operator in a modal basis, so its accuracy on FDTD truth is near-tautological. It needs exact geometry, source and β. What is the INR for if physics reaches 0.96–0.99? That is the manager's call.

## m2 — configuration diversity (D4 flips + reciprocity), two arms at 60K
| arm | recipe | pre-registered | status |
|---|---|---|---|
| A-fam | G2 (family, U held out) + aug_d4_p 0.5 + aug_recip_p 0.25, flips fx/fy/r180 | held-out U at 25–100 Hz r ≥ 0.554 + 0.10 | training (tron, 0.76 s/it), due ~02:30 |
| A-500 | G1 (C500) + the same augmentation | out-of-family mean (T, DN, U) at 25–100 Hz ≥ 0.628 + 0.10 | training (tron, 0.76 s/it), due ~02:30 |

- **Leakage pre-flight** ([m2_leak_preflight.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/m2_leak_preflight.json)): topology is read from each image polygon.
  - **0 U-topology images under all 8 D4 elements in both corpora**, so 0 fall inside the U ranges.
  - The transposing elements leak the held-out d̂ slab (43 / 240 and 88 / 500); the flips leak nothing.
  - Both corpora have a slab, so both arms use the flips (the manager's rule).

## m4 — the configuration-diverse corpus CX500 + arm GX
- **Corpus** (`aaf/data/cx_rooms.py`, [manifest](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweeps_2d_mat/p4_7_cx500_manifest.json)): 500 rooms, built in 63 CPU tasks (median 14 s per room), 1.7 GB.
  - **Notches:** 1–2 at any of 4 corners or 4 side midpoints. Topologies: L 143, T 174, LT (corner + side) 110, TT 54, DN 19. Positions: 68–102 rooms each.
  - **Exclusions:** the U topology (any orientation), the held-out d̂ slab (also the transposed one for corner notches), and passages under 0.8 m.
  - **Source:** uniform in air, ≥ 0.5 m from every edge; 500 distinct positions, x 0.50–6.30, y 0.50–4.92.
  - **Receivers:** 800 per room from the room's own 0.02 m nodes, ≥ 0.12 m from walls, ≥ 0.15 m from reflex corners and ≥ 0.10 m from the source. 11% land on the old lattice (B6-like).
  - **Tests:** 8 CPU tests (`tests/test_cx_rooms.py`), including the exclusions on all 500 rooms and the polygon against an independent rasterisation and against the family code.
- **Trainer per-room source** (06e9010, default off):
  - A corpus whose files disagree on the source is now REFUSED. It used to train silently against the last file's source.
  - Off: `_render` equals the frozen pre-change copy. On, with a single-source corpus: renders are bit-identical. On, with a mixed corpus: each row is rendered with its own source, and the output changes.
  - C500's rows, re-expressed as polygon rows, condition bit-identically (500 / 500).
  - Tests: 119 trainer-related tests pass.
- **Arm GX:** G1's recipe on C500 + CX500 (1000 rooms), 60K, `per_room_source: true`. Queued on tron; reported against G1, in family and out of family.

## m3 — modal hybrid (pre-registered D170(g); implementation in progress)
- **Residual:** H = H_modal + H_net. H_modal is m1's formula (frozen C, modes to 600 Hz) **with the exact zero-mode record**; H_net is PB3; the loss is applied to the sum.
- **Features:** for the first 48 mode clusters, s_c = A·Σψₙ(x_r)ψₙ(x_s) (degenerate modes summed) and k_c / k₆₀₀. They enter through a zero-initialised map added to the signal features, so the network equals PB3 at step 0.
- **Guards:** the renderer refuses to render without the prior.
- **Order:** M3-fam (G2's corpus) first, then M3-500 (G1's corpus).
- **Readings:**
  1. Held-out U at 25–100 Hz ≥ 0.85.
  2. The network adds over physics (≥ 0.01 r at 200–300 Hz or ≥ 0.2 dB).
  3. Harm: any band more than 0.02 r below H_modal alone.

## Continuing
- **G3:** to 120K (pre-registered 200–300 Hz r ≥ 0.52); preempted at 77K and requeued.
- **Parked:** the doorway aperture (revisit with the modal prior), scale augmentation (now possible with the exact zero mode, D170(c)), plane-wave / Helmholtz losses.
