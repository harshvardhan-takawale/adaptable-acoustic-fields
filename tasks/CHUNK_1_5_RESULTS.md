# Chunk 1.5 — Results

**Date**: 2026-05-10. **Scope**: Q9 dedup convention + dataset rebuild at `n_time_samples=8192` + visual sanity pack + re-run noise-floor with dedup. No model code, no renderer, no training.

---

## 1. Q9 dedup convention

### Implementation

`aaf/sim/analytical_modal_2d.py` now distinguishes:
- `_enumerate_pairs(L, W, c, f_max)` — internal, returns every `(n_x, n_y, f)`.
- `eigenfrequencies_2d(...) -> list[EigenFreq]` — public, deduplicates within `dedup_tol_hz=0.01`. Each `EigenFreq` has `f`, `multiplicity`, `pairs`.
- `modal_rir_2d` iterates `_enumerate_pairs` so each `(n_x, n_y)` term contributes its own Lorentzian to the modal sum (physics unchanged).

The matcher (`aaf/eval/modal_verifier.py`) didn't need an API change — it only reads `.f` from its inputs. With dedup'd entries, `n_analytical = len(distinct_freqs)`, so `recall = n_matched / n_distinct_freqs` automatically.

Three new tests cover the contract:
- `test_LW4_first_degenerate_pair`: at L=W=4, first non-zero entry has `multiplicity=2` and pairs `{(1,0),(0,1)}`.
- `test_L4_W2_has_expected_degeneracies`: L=2W creates predictable collisions, e.g., `(2,0)≡(0,1)` at 85.75 Hz.
- `test_L3_W4_collision_at_higher_freq`: even non-square rooms get higher-mode collisions (e.g., `(3,0)≡(0,4)` at 171.5 Hz).

40 tests total now passing on scavenger compute node (`pytest -q` job 6804320).

### Before vs after (L=W=4 m)

| Metric | Before (Chunk 1) | After (Chunk 1.5) |
|--------|-----------------:|------------------:|
| Modal-regime `n_analytical` | 128 (all pairs) | 60 (distinct freqs) |
| Modal-regime matched | 13 | 13 |
| Modal-regime recall | 0.10 | **0.22** |
| Full-band `n_analytical` | 1754 (all pairs) | 728 (distinct freqs) |
| Full-band recall | 0.04 | **0.10** |
| Full-band MAE | 0.55 Hz | 0.38 Hz |

The recall ceiling at L=W=4 is no longer artificially capped by degenerate-pair double-counting.

## 2. Dataset rebuild

### Configuration change

All three sweep YAMLs (`configs/sweeps/{dense, sparse, extrapolation}.yaml`) bumped `n_time_samples: 2048 → 8192`. Same `fs=4096`, `W=4 m`, `α=0.15`, source at `(0.5, 0.5)`, 8×8 receiver grid.

The old dataset was preserved as `data/track_a_2048/` for diff/regression purposes; the new one lives at `data/track_a/`.

### Stats

- **15 HDF5 files** in `data/track_a/`, ~7.85 MB each, **113 MB total** (vs 29 MB at 2048 samples).
- IR length: **2.0 s** at fs=4096.
- T60 EDC measured per room: 515 ms (L=2.5) → 826 ms (L=6.5). Coverage ratio `n_time_samples / (T60·fs) ≈ 1.5–2.5×` everywhere — comfortably above 1× T60 in every room. The -5..-35 dB EDC regression sits well within the captured signal in all 15 rooms.
- ISM `max_order` per room (auto-computed via `ceil(c·4·T60_sabine/min(L,W))`): ranges 80–120, recorded in HDF5 root attrs.

Build wall-time: 60 s for all 15 rooms on `cbcb21.umiacs.umd.edu` (1 CPU job). HDF5 round-trip verified by `tests/test_dataset_io.py`.

## 3. Visual sanity pack

`outputs/visual_sanity/` contains:
- `INDEX.md` — table of contents with links.
- `cross_room.pdf` — three pages: scatter (analytical_f, picked_f), mode-vs-L lines, T60-vs-L.
- `per_room/L_*.pdf` — 15 files, three pages each: 4-panel IR view, 8×8 sparkline grid, spectrum overlay.
- `SANITY_NOTES.md` — agent's eyeballing observations after rendering.

Total 6.8 MB.

### Headline observations (full notes in SANITY_NOTES.md)

- **Cross-room scatter** lies dead on the y=x diagonal across all 15 rooms — visually confirms sub-Hertz MAE.
- **(1,0) and (2,0) family curves** track c/(2L) and 2c/(2L) cleanly. (1,0) tracking disappears at L≥4.5 m where its frequency drops below the (0,1) horizontal at 42.875 Hz — the picker can no longer separate them. Expected, not a bug.
- **T60 dashed line overlaps pra Sabine perfectly** after fixing the constant from `6.91·4·A/(c·α·P)` (= 27.6×) to `0.161·A/(α·P)` (= 55.3×). pra applies the textbook 3D Sabine formula to 2D with V→A and S→P.
- **EDC T60 is consistently ~60% of Sabine prediction** across all L — a stable multiplicative gap, characteristic of moderate absorption (α=0.15) where the diffuse-field assumption breaks. **Train/eval Chunk 2 model decay losses against EDC, not Sabine.**
- **Modal-regime spectra are smoother than expected** at α=0.15: many overlapping Lorentzians of half-width ~4–6 Hz blur the picture even below 200 Hz. Implication: a model that nails the spectral envelope but smears individual peaks may still be physically reasonable.

## 4. Updated noise-floor numbers (post-dedup, post-rebuild)

Re-ran `noise_floor_report.py` with the dedup'd analytical mode list and the 8192-sample dataset. Headline shifts:

| | Chunk 1 | Chunk 1.5 |
|---|--------:|----------:|
| Mean modal-regime recall | 0.123 | **0.139** |
| Mean modal-regime MAE | 0.55 Hz | **0.38 Hz** |
| Full-band recall | 0.043 | **0.052** |
| Full-band MAE | 0.36 Hz | **0.29 Hz** |
| Mean modes ≤ f_Schroeder per room (n_analytical) | 131.3 | 122.1 |
| Mean ISM picks ≤ f_Schroeder per room | 16.0 | 16.4 |
| Spurious picks (full band) | 0.0 (mean) | 0.0 (mean) |

Modal-regime recall improved across the board because `n_analytical` (denominator) shrunk from "raw mode pairs" to "distinct frequencies". MAE improved because the larger n_time_samples buys finer Δf (still 2 Hz, but more bins above the picker's threshold). REPORT.md regenerated; old PNG figures replaced.

## 5. Surprises

- **My dashed T60 line was off by ~2× in the first render** because I used `6.91·4` instead of the textbook Sabine constant `0.161·c ≈ 55.3`. Caught and corrected on the second render. Lesson: the same physical formula has multiple equivalent-looking constants depending on whether you derive it from amplitude vs intensity decay; always anchor against an empirical reference (here, pra) before plotting.
- **Most non-square room shapes still have higher-mode degeneracies** (e.g., L=3, W=4 has (3,0)≡(0,4) at 171.5 Hz). This means dedup is not merely an L=W cosmetic — it materially affects mode counting for any rational L/W ratio. The new test `test_L3_W4_collision_at_higher_freq` documents this with evidence.
- **EDC T60 is far from Sabine T60 in 2D at α=0.15.** ~30–40% gap, stable across L. This is a known feature of low/moderate absorption, but the magnitude was bigger than I'd anticipated. For Chunk 3+ training metrics: do not use Sabine as a reference; use measured EDC.

## 6. Manager actions requested

- **Q9 closed.** No further input needed on dedup.
- **Q1** (ray sampling) and **Q5** (cluster partition) still open; Chunk 2 can default Q1 and Chunk 3 needs Q5.

The data and verification artifacts are now sufficient for Chunk 2 (renderer + model). The visual-sanity pack should give the manager and user a clear picture of what the trained model is being asked to reproduce.
