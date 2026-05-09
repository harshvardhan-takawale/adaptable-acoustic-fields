# Chunk 1 — Results

**Date**: 2026-05-09. **Scope**: 2D shoebox simulation pipeline + analytical modal verifier + 15-room dataset + noise-floor report. No model code, no renderer, no training, no auto-decoder.

This is the input the manager uses to write Chunk 2.

---

## 1. Pipeline summary

| Stage | Module | Driver | SLURM script | Hardware | Wall-time |
|-------|--------|--------|--------------|---------:|---------:|
| Tests | `tests/test_*.py` | `pytest -q` | `scripts/slurm/run_pytest.sh` | scavenger CPU+1 GPU | <30 s |
| Dataset gen | `aaf.sim.{ism_2d, analytical_modal_2d}` + `aaf.data.dataset_builder` | `scripts/build_datasets.py` | `scripts/slurm/build_datasets.sh` | scavenger CPU only | **40 s** total for 15 rooms |
| Noise-floor report | `scripts/noise_floor_report.py` | (same script) | `scripts/slurm/noise_floor.sh` | scavenger CPU only | ~12 s |

End-to-end (tests → datasets → report) is well under 5 minutes of wall-clock.

Vendored INFER reference classes (no production imports) live in `aaf/_inference_ref/`.

## 2. Pyroomacoustics 2D verdict (resolves Q7)

**Pass.** `pra.ShoeBox(p=[L, W], …)` constructs a true 2D room (`room.dim == 2`, dedicated `libroom.Room2D` C++ engine). `room.rt60_theory(formula='sabine')` returns dimension-aware T60 estimates. ISM is dimension-agnostic and produces well-formed IRs. Causality and conjugate symmetry of the resulting RFFT are verified by `tests/test_ism_smoke.py`.

The 15-room dataset built cleanly without exceptions or warnings (apart from the documented "n_time_samples < 4·T60·fs" notice on the tiny smoke-test room, which is intentional).

## 3. Noise-floor findings (the load-bearing science deliverable)

Full text in `outputs/noise_floor/REPORT.md`. Headline:

- **MAE between ISM peaks and analytical eigenfrequencies is sub-Hertz** in every room (mean 0.36 Hz full-band, 0.55 Hz in the modal regime ≤ f_Schroeder). This is below Δf = 2 Hz (one bin) — the picker can't be more accurate than ~half a bin.
- **Zero spurious picks** across all 15 rooms when matched against the dense 2D analytical mode list. Every picked ISM peak finds an analytical mode within tolerance.
- **Recall is genuinely low (~10–15% modal regime, ~4% full band) and that is correct, not a failure.** 2D rooms have very high modal density (1100–2800 modes per room in 0–2 kHz); only the lowest dozen-or-so modes are physically resolvable as discrete peaks. Above the modal-overlap region, the IR's `|H(f)|` becomes a statistical maximum landscape rather than a spectrum of separable resonances. The picker correctly recovers the resolvable subset and not the unresolvable rest.
- **Stratified by mode index** (full band): ordinal 0–5 → 38% recall; 6–15 → 17%; 16+ → 4%. Confirms the picker only recovers the lowest-frequency well-isolated modes, as expected.

**Implication for Chunk 2**: any neural model we train will be evaluated against ISM ground truth (not analytical), so the modal-frequency MAE we'll care about is the model-vs-ISM MAE, not model-vs-analytical. The analytical reference is for orientation, not for primary evaluation.

## 4. Modal-degeneracy observations

At **L = W = 4 m**, the analytical eigenfrequencies (n_x, n_y) and (n_y, n_x) coincide. With our enumeration (which lists both as separate modes):

- 1754 analytical modes occupy 669 distinct frequencies → degeneracy ratio 2.62×.
- The picker finds 67 distinct peaks; the Hungarian matcher attaches each peak to one mode.
- Consequence: the recall ceiling for L = W rooms is `n_distinct_freqs / n_modes` ≈ 0.38, not 1.0. Comparing degenerate vs non-degenerate L rooms in the per-L recall chart will show this dip naturally.

Other near-degeneracies appear at integer L/W ratios (e.g., L = 6 m, W = 4 m: modes (3, 0) and (0, 2) both at 85.75 Hz). Less severe than L = W but worth flagging.

**Decision deferred**: whether to adjust the matcher to allow many-to-one mode→peak assignment for degenerate cases. Adding to `OPEN_QUESTIONS.md` as Q9 since this affects how Chunk 2's eigenfrequency MAE metric is computed.

## 5. Dataset stats

- **Files**: 15 HDF5 files in `data/track_a/`, one per L value `{2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.5}` at fixed W=4.0 m, α=0.15.
- **Per-file size**: ~2.0 MB (gzip-compressed).
- **Per-file contents**:
  - `/ism/H_complex` shape `(64, 1025)`, dtype `complex64`.
  - `/ism/rir_time` shape `(64, 2048)`, dtype `float32`.
  - `/analytical/H_complex` and `/analytical/rir_time` matched shapes/dtypes.
  - 64 receivers per room (8 × 8 grid on `[0.3, L-0.3] × [0.3, W-0.3]`).
- **Total complex coefficients on disk**: 15 × 2 × 64 × 1025 = **1.97 M complex64** (one per (room, model, receiver, freq-bin)) plus the same in time-domain reals.
- **Total disk usage**: 29 MB for the entire `data/track_a/` tree.
- ISM `max_order` per room is auto-chosen via `ceil(c · 4 · T60 / min(L,W))` and recorded in each file's root attrs (typical range 80–120).
- Source position fixed at (0.5, 0.5) m for all rooms.

## 6. Sweep configurations

`configs/sweeps/{dense, sparse, extrapolation}.yaml`. Each enumerates a `train_L` and `test_L` list; `build_datasets.py` takes the union of both lists across all three configs (15 unique L values). The Chunk-2 `ShoeboxDataset` (currently a stub in `aaf/data/loader.py`) will read these configs.

## 7. Surprises and risks for Chunk 2

- **Recall numbers will look low at first glance.** Anyone reading the report without the modal-density context will think the picker is broken. Lead with MAE in any subsequent presentation, and explain the recall caveat once.
- **2D Schroeder frequency**: the 3D-form proxy (V = A · 1m) gives ~500 Hz, but the actual 2D modal-overlap frequency (where peaks become statistically inseparable) is closer to 100–150 Hz for our rooms. The report cuts at the proxy value; if the manager wants a sharper "modal-only" recall, redefine `f_Schroeder` per a 2D Kuttruff formula. This is a presentation-only issue — the underlying data is not affected.
- **Degenerate matcher at L=W**: see §4. Decide before Chunk 2 implements the eigenfrequency MAE metric for the model.
- **`_inference_ref/` is not runnable** (uses `tcnn` + `.cuda()` at module init via `ray_directions`). It only needs to *parse*. This is enforced via `python -c "import ast; …"` checks in CI; currently verified by hand.
- **Pyroomacoustics absorption tail truncation**: at α=0.15, T60 ranges from 0.83 s to 1.33 s across our L sweep. With `n_time_samples=2048` at `fs=4096`, IR length is 0.5 s — well below T60. We truncate the IR tail (energy below ~-20 dB). For the modal-frequency analysis this is fine (peaks are dominant); for eventual energy-based metrics in Chunk 4+, expect the model to learn truncated decays.
- **GPU not used**: ISM is pure CPU; `tinycudann` and torch are loaded only for the test suite. Chunk 2's first model training will be the first real GPU consumer.

## 8. Manager actions requested

- **Q9 (new)**: how should the modal verifier handle degenerate eigenfrequencies (multiple modes at one frequency)? Many-to-one matching, or report them as one combined mode? Affects Chunk 2's eigenfrequency MAE metric.
- **Q1 still deferred**: ray sampling strategy for Chunk 2's renderer (stochastic uniform-azimuth, deterministic, or ISM-aligned). The data is now in place to test all three.
- **Q5 still open**: long-running training partition. Becomes urgent in Chunk 3 once training takes >1 day.

The manager can write Chunk 2 now without waiting on these — Q9 only matters when the eigenfrequency probe runs against model output, and Q5 is a Chunk-3 concern.
