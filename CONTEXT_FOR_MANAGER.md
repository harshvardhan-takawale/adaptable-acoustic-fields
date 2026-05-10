# CONTEXT_FOR_MANAGER.md

Manager re-orientation doc. Optimized for catching up in 5 minutes after time away. Updated at the end of every chunk.

**Last updated**: Chunk 2 — 2026-05-10

## Project state

- **Phase**: 1 (2D shoebox sweep, 0–2 kHz, Track A — science).
- **Chunks completed**: Chunk 0 (scaffolding), Chunk 1 (pipeline + dataset + noise-floor), Chunk 1.5 (Q9 dedup + dataset rebuild + visual sanity), **Chunk 2 (2D model + renderer port + single-room overfit baseline at L ∈ {3.0, 4.5, 6.0} m)**.
- **What exists today**: everything from Chunks 0/1/1.5 plus the 2D model (`aaf/models/inr_2d.py:INR2D_Single`), 2D renderer (`aaf/renderers/freq_2d.py:FreqRenderer2D`), real `ShoeboxDataset` loader, single-room trainer + evaluator, GPU memory check, full pipeline orchestrator. **3 trained checkpoints + 12 figures + 1 cross-room summary** in `outputs/single_room/`. 62 tests pass.
- **What does not exist**: model code (`aaf/models/`), renderer code (`aaf/renderers/`), training loop (`aaf/train/`), `ShoeboxDataset` is interface-stub only, no auto-decoder.
- **Next chunk** (manager will write Chunk 2): expected to port the INFER renderer + model to 2D (drop spherical → circular sampling, `tcnn.Encoding(3, ...)` → `(2, ...)`) and connect to the dataset via the `ShoeboxDataset` stub. See `tasks/CHUNK_0_RESULTS.md` §6 for adaptation needs and `tasks/CHUNK_1_RESULTS.md` for the noise-floor report findings that constrain Chunk 2's eval metrics.

## Codebase map

```
adaptable-acoustic-fields/
├── CLAUDE.md                   standing rules for any agent
├── CONTEXT_FOR_MANAGER.md      this file
├── DECISIONS.md                append-only design log
├── OPEN_QUESTIONS.md           ambiguities (numbered, append-only)
├── CLUSTER_INFO.md             Nexus SLURM how-to, partitions, sbatch template
├── README.md                   public-facing intro
├── environment.yml             frozen conda env (aaf, Py3.8, torch 2.0.1+cu118)
├── pyproject.toml              ruff + black + pytest config
├── .gitignore
├── aaf/                        importable source package
│   ├── _inference_ref/         vendored INFER classes (parse-only; do not import)
│   ├── data/                   dataset_builder.py (HDF5 I/O), loader.py (stub)
│   ├── eval/                   modal_verifier.py (peak picker + Hungarian matcher)
│   ├── models/                 (chunk 2) 2D INR + auto-decoder — STUBBED
│   ├── renderers/              (chunk 2) frequency-domain renderer — STUBBED
│   ├── sim/                    ism_2d.py + analytical_modal_2d.py
│   ├── train/                  (chunk 3) training loop — STUBBED
│   └── utils/                  (logging, criteria) — STUBBED
├── configs/
│   └── sweeps/
│       ├── dense.yaml          7 train + 6 test L values
│       ├── sparse.yaml         3 train + 4 test
│       └── extrapolation.yaml  5 train + 4 test (test outside training support)
├── scripts/
│   ├── build_datasets.py       generate 15 HDF5 files (40 s total)
│   ├── noise_floor_report.py   produce outputs/noise_floor/
│   └── slurm/
│       ├── hello.sh
│       ├── run_pytest.sh
│       ├── build_datasets.sh
│       └── noise_floor.sh
├── tests/                      32 tests, all passing on scavenger
│   ├── test_env.py
│   ├── test_smoke.py
│   ├── test_eigenfrequencies.py
│   ├── test_peak_picker.py
│   ├── test_modal_matcher.py
│   ├── test_ism_smoke.py
│   └── test_dataset_io.py
├── outputs/                    figures + REPORT.md from noise_floor analysis
│   └── noise_floor/
│       ├── REPORT.md           ← the load-bearing science deliverable
│       ├── metrics.json
│       └── figures/*.png       (gitignored)
└── tasks/
    ├── CHUNK_0_RESULTS.md      recon writeup
    └── CHUNK_1_RESULTS.md      ← read this for Chunk 2 input
```

## What we inherited from AVR / INFER

**AVR** (NeurIPS'24, `/fs/nexus-projects/multimodal_recon/AVR`):
- Frequency-domain volume rendering pipeline (`renderer.py:AVRRender`)
- 6-term criterion (`utils/criterion.py`): spec + amplitude + angle + time + energy + multi-STFT
- Acoustic metric battery (`utils/metric.py`): Angle / Amp / Env / T60 / EDT / C50 / multi-STFT
- TensorBoard logging pattern (`avr_runner.py:196-244`)

**INFER** (ICML'26 submission, `project_files/`):
- Main model: `AVRModel_complex_FD_FreqDep_PhaseCorrection` (`unified_models.py:752-883`)
  - Forward signature: `forward(pts, view, tx, tx_view) → (attn_complex, signal_complex)`
  - Per-frequency-bin complex attenuation: `attn = σ + jβ` (real via softplus → absorption, imag → phase velocity change)
- Main renderer (we keep, sans KK): `AVRRenderFD_FreqDep_PhaseCorrection_new` (`unified_renderers.py:716-790`)
  - Renderer's `acoustic_render_fd` decomposes complex `attn` into σ (clamped ≥ 0) and β, builds amplitude transmittance (cumprod of `1-α`) and phase transmittance (`exp(j·cumsum(β·Δu))`), applies geometric phase `exp(-j·2π·f·d/c)`, sums over rays.
- KK ablation variant exists (`AVRRenderFD_FreqDep_PhaseCorrection_KK`, lines 829-1035) — **not** used for Phase 1.

**Note for next agent**: An earlier sub-agent misidentified `INRASFrequencyModel` as the INFER main model. INRAS is a **baseline** (Su et al. 2022 in INFER's Related Work). The actual INFER main model is the `AVRModel_complex_FD_FreqDep_PhaseCorrection` cited above. See `tasks/CHUNK_0_RESULTS.md` for confirming evidence.

## What's working

- **Conda env**: `aaf` exists at `/fs/nexus-scratch/htakawal/miniconda3/envs/aaf`. Frozen in `environment.yml`. The `LD_LIBRARY_PATH` shim is required (CLUSTER_INFO.md, DECISIONS.md).
- **pytest**: 62 tests pass on a scavenger GPU compute node (`scripts/slurm/run_pytest.sh`). Coverage: env imports, 2D eigenfrequencies, peak picker, modal matcher, ISM smoke, HDF5 round-trip, ShoeboxDataset, FreqRenderer2D, INR2D_Single, eval metrics, early-stop logic.
- **Dataset pipeline**: `scripts/build_datasets.py` generates all 15 HDF5 rooms in 60 s on a single CPU. Files in `data/track_a/` (~8 MB each, 113 MB total at 8192-sample IRs).
- **Modal verifier**: `aaf.eval.modal_verifier` provides `pick_peaks`, `match_peaks_to_modes`, `modal_error_metrics`, and `plot_modal_overlay`. Used by both `noise_floor_report.py` and `single_room_eval.py`.
- **Noise-floor report**: `outputs/noise_floor/REPORT.md` and 4 figures. ISM-vs-analytical: **0.36 Hz MAE** (full-band) at the centre receiver; recall ~10-15% modal-regime, ~4% full-band (correct 2D physics).
- **2D model + renderer**: `INR2D_Single` (~44M params) + `FreqRenderer2D` (n_azi=64, stochastic uniform-azimuth sampling). Memory check confirmed `n_pts_per_ray=32, batch=8` fits at 8.09 GB on a 12 GB GTX TITAN X.
- **Single-room overfit baseline**: 3 rooms trained 10K iters each (~1.5h on scavenger). Achieves modal MAE 0.34-0.58 Hz on matched peaks (parity with the noise floor) and full-band LSD 0.36-0.42 dB. 12 figures + cross-room SUMMARY.md committed.
- **GitHub**: public repo at `https://github.com/harshvardhan-takawale/adaptable-acoustic-fields`. Raw URL pattern: `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/<file>.md`.

## What's broken or stubbed

- `aaf/train/single_room.py` is the *single-room* trainer only. Multi-room training + auto-decoder is Chunk 3.
- `aaf/utils/`: empty `__init__.py` only.
- `aaf/_inference_ref/`: vendored INFER classes; reference-only, parse-checked but not runnable on the cluster (uses `.cuda()` at module init).
- No `auraloss` or perceptual loss wired in — deferred to Phase 4.
- No latent table / auto-decoder — Chunk 3 will subclass `INR2D_Single` → `INR2D_AutoDecoder` and use the `z_s` argument that's already in the forward signature.
- **HashGrid is over-parameterised for 2D shoeboxes** (default INFER config is ~120 MB hash params per encoder × 6 encoders). Single-room overfit is trivial; recommended Chunk 3 starts with `log2_hashmap_size=14, n_levels=14` (~16× fewer params) to ensure the auto-decoder's `z_s` actually matters. See `tasks/CHUNK_2_RESULTS.md` §6 + §8.

## Recent changes (this chunk — 2)

- Wrote `aaf/renderers/freq_2d.py:FreqRenderer2D` (2D port of `AVRRenderFD_FreqDep_PhaseCorrection_new`). Stochastic uniform-azimuth ray sampling with per-iteration jitter (n_azi=64, no elevation, no zenith/nadir). 4-wall slab AABB. σ + jβ decomposition + cumulative transmittance + geometric phase verbatim from the INFER reference. `use_geometric_attn=False` per Phase-1 decision.
- Wrote `aaf/models/inr_2d.py:INR2D_Single` (2D port of `AVRModel_complex_FD_FreqDep_PhaseCorrection`). Six `tcnn.Encoding(2, ...)` encoders. Sigma branch (complex per-freq attenuation σ + jβ) + signal branch (complex per-freq emission). RFFT symmetry mask on DC + Nyquist imag. `z_s` argument accepted in `forward(...)` but ignored in `INR2D_Single` — Chunk-3 subclass will use it. Inline injection-point comments mark candidate-A concat points.
- Wrote `aaf/data/loader.py:ShoeboxDataset` (real implementation of the Chunk-1.5 stub). One sample per (room, receiver) pair. `room_filter=[L]` for single-room mode.
- Wrote `aaf/train/single_room.py:SingleRoomTrainer` with 4-term loss (real, imag, log-amp, phase) weighted (1, 1, 1, 0.1), Adam + cosine LR, gradient clip + NaN/Inf masking, checkpoint every 2.5K iters with auto-resume, validation every 500 iters, **relative-improvement early-stop** (1% improvement window over 2K iters past warmup at 2K).
- Wrote `aaf/eval/single_room_eval.py:evaluate_single_room` — dual-metric eval (modal regime + full band) + 4 figures (training_curves, modal_tracking, spectrum_overlay, receiver_grid). Reads renderer config from `train_meta.json` to inherit training params.
- Wrote `scripts/single_room_memory_check.py` with fallback policy: try `(64,64)` → `(64,32)` → `(32,32)`. Confirmed `(64,32)` fits at 8.09 GB on the GTX TITAN X.
- Wrote `scripts/single_room_summary.py` aggregating per-room `eval.json` into `outputs/single_room/SUMMARY.md` + `lsd_vs_L.png`.
- Wrote 4 SLURM scripts and `scripts/run_chunk2_pipeline.sh` orchestrator chaining `memory_check → 3× train → 3× eval` via `--dependency=afterok`.
- Wrote 5 new test files (loader, renderer, model, eval_metrics, early_stop). 62 tests now pass.
- Trained 3 single-room baselines on L ∈ {3.0, 4.5, 6.0}; all reached 10K iters (no early-stop trigger). Headline metrics in `tasks/CHUNK_2_RESULTS.md` §4.
- Updated `.gitignore`: allow `outputs/single_room/**/*.png` exception. Ignore `*.pt` checkpoints and TensorBoard event files under `outputs/single_room/L*/`.
- Appended **9 new entries to DECISIONS.md** covering Q1 (ray sampling), geom-attn re-confirmation, output dim, loss weighting, checkpoint cadence, HashGrid capacity note, png-tracking exception, and the 10K-iter + relative-improvement early-stop change. Closed Q1 in `OPEN_QUESTIONS.md`.

## Recent changes (Chunk 1.5)

- Replaced `Mode` dataclass in `aaf/sim/analytical_modal_2d.py` with `EigenFreq(f, multiplicity, pairs)`. Added internal `_enumerate_pairs()` so `modal_rir_2d` keeps iterating individual `(n_x, n_y)` terms; the public `eigenfrequencies_2d` returns the deduplicated list (Q9 resolution).
- `aaf/eval/modal_verifier.py` unchanged — already consumed any object with `.f`. Naturally interprets `n_analytical = len(distinct_freqs)` after dedup.
- Bumped `n_time_samples: 2048 → 8192` in all three sweep YAMLs. Backed up old dataset to `data/track_a_2048/`; regenerated `data/track_a/` (113 MB total, 2.0 s IRs).
- Wrote `scripts/visual_sanity.py` + `scripts/slurm/visual_sanity.sh`. Produces 15 per-room PDFs + 1 cross-room PDF + INDEX.md + SANITY_NOTES.md in `outputs/visual_sanity/`.
- Re-ran `noise_floor_report.py` on the new dataset with dedup applied. Modal-regime recall improved (0.123 → 0.139), MAE improved (0.55 → 0.38 Hz).
- Updated `tests/test_eigenfrequencies.py` and added `tests/test_modal_dedup.py` (8 new test functions; 40 total now passing).
- Removed `/outputs/**/*.pdf` from `.gitignore` so the visual-sanity PDFs are tracked (they're small, useful for reviewers; PNG/SVG/JPG remain ignored).
- Wrote `tasks/CHUNK_1_5_RESULTS.md`. Appended two `DECISIONS.md` entries (Q9 dedup convention; dataset rebuild at 8192 samples). Closed Q9 in `OPEN_QUESTIONS.md`.

## Recent changes (Chunk 1)

- Vendored `AVRModel_complex_FD_FreqDep_PhaseCorrection` + `AVRRenderFD_FreqDep_PhaseCorrection_new` into `aaf/_inference_ref/` (reference-only).
- Built `aaf/sim/ism_2d.py`, `aaf/sim/analytical_modal_2d.py`, `aaf/eval/modal_verifier.py`, `aaf/data/dataset_builder.py`, `aaf/data/loader.py` (stub).
- Wrote `configs/sweeps/{dense, sparse, extrapolation}.yaml`, `scripts/build_datasets.py`, `scripts/noise_floor_report.py`.
- Wrote 5 tests; generated 15 HDF5 datasets; produced first noise-floor REPORT.md.

## Pointers

- **Read for Chunk 3**: `tasks/CHUNK_2_RESULTS.md` (latest, contains explicit Chunk-3 recommendations in §8), `outputs/single_room/SUMMARY.md` (cross-room table + LSD-vs-L plot). The `outputs/single_room/L*/figures/` plots are the headline visual reference. Then `tasks/CHUNK_1_5_RESULTS.md` and `outputs/visual_sanity/SANITY_NOTES.md` for ground-truth context.
- Open questions: `OPEN_QUESTIONS.md`. Only Q5 (cluster partition for long runs) still open; doesn't block Chunk 3 if we stay on scavenger.
- Design log: `DECISIONS.md`. Now has 25 entries.
- Cluster how-to: `CLUSTER_INFO.md`. Default partition is `scavenger`. Always set `LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}` after `conda activate aaf`. Chunk-2 pipeline driver is `scripts/run_chunk2_pipeline.sh`.

## How the manager can read this repo

Public; raw URLs are stable on `main`. Examples:

```
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/CONTEXT_FOR_MANAGER.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_0_RESULTS.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md
```
