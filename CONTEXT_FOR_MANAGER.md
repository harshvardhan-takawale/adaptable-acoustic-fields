# CONTEXT_FOR_MANAGER.md

Manager re-orientation doc. Optimized for catching up in 5 minutes after time away. Updated at the end of every chunk.

**Last updated**: Chunk 3 — 2026-05-10

## Project state

- **Phase**: 1 (2D shoebox sweep, 0–2 kHz, Track A — science).
- **Chunks completed**: Chunks 0/1/1.5/2 plus **Chunk 3 (auto-decoder + dense-sweep multi-room training + zero-shot adaptation at unseen L + latent probe)**.
- **What exists today**: everything prior plus `INR2D_AutoDecoder`, `MultiRoomTrainer`, `zero_shot_adapt`, `probe_latents`, plus a complete dense-sweep run on disk: trained checkpoint + scalars + train_meta + 6 zero-shot directories (L ∈ {3.25, 3.75, 4.25, 4.75, 5.25, 5.75}) with metrics + figures + a latent-probe directory + cross-room SUMMARY.md. 70 tests pass.
- **Headline Chunk-3 result**: training succeeded (per-room val LSD 0.66-0.98 dB; 6 of 7 rooms ≤ 1 dB target met) but **zero-shot adaptation failed** (held-out LSD 5.7-6.0 dB vs ≤ 2 dB target) because the latents collapsed to a non-physical structure (PC1 vs L R² = −0.63, intrinsic_dim = 10/32). The over-parameterisation risk flagged in `CHUNK_2_RESULTS.md` §6 / DECISIONS.md materialised. Documented per spec ("if worse, document as failure mode rather than blocker"). See `tasks/CHUNK_3_RESULTS.md` §10 for recommended fix (HashGrid resize + retrain).
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
- **pytest**: 70 tests pass on a scavenger GPU compute node (`scripts/slurm/run_pytest.sh`). Coverage: env imports, 2D eigenfrequencies, peak picker, modal matcher, ISM smoke, HDF5 round-trip, ShoeboxDataset, FreqRenderer2D, INR2D_Single, INR2D_AutoDecoder, zero_shot inner loop, latent_probing PCA, eval metrics, early-stop logic.
- **Dataset pipeline**: `scripts/build_datasets.py` generates all 15 HDF5 rooms in 60 s on a single CPU. Files in `data/track_a/` (~8 MB each, 113 MB total at 8192-sample IRs).
- **Modal verifier**: `aaf.eval.modal_verifier` provides `pick_peaks`, `match_peaks_to_modes`, `modal_error_metrics`, and `plot_modal_overlay`. Used by both `noise_floor_report.py` and `single_room_eval.py`.
- **Noise-floor report**: `outputs/noise_floor/REPORT.md` and 4 figures. ISM-vs-analytical: **0.36 Hz MAE** (full-band) at the centre receiver; recall ~10-15% modal-regime, ~4% full-band (correct 2D physics).
- **2D model + renderer**: `INR2D_Single` (~44M params) + `FreqRenderer2D` (n_azi=64, stochastic uniform-azimuth sampling). Memory check confirmed `n_pts_per_ray=32, batch=8` fits at 8.09 GB on a 12 GB GTX TITAN X.
- **Single-room overfit baseline (Chunk 2)**: 3 rooms trained 10K iters each (~1.5h on scavenger). Achieves modal MAE 0.34-0.58 Hz on matched peaks (parity with the noise floor) and full-band LSD 0.36-0.42 dB. 12 figures + cross-room SUMMARY.md committed.
- **Multi-room shared training (Chunk 3)**: 7 rooms (dense-sweep `train_L`) trained 30K iters in 2:38 on tron64 RTX 3070. Per-room val LSD 0.66-0.98 dB (6 of 7 rooms ≤ 1 dB target). Latents stayed healthy in magnitude (mean ‖z_s‖ = 0.81; init 0.18; min 0.70 / max 0.95) but collapsed in *structure* — they encode room ID, not L geometry.
- **Zero-shot eval pipeline**: works mechanically (6 evals, 16-17 min each on scavenger, all completed; 24 PNGs generated). Quality fails per the latent collapse (held-out LSD 5.7-6.0 dB).
- **Latent probe**: `aaf.eval.latent_probing` produces PCA + linear-fit-to-L diagnostic. The Chunk-3 `latent_pca_1d.png` is the load-bearing diagnostic for capacity decisions in any retrain.
- **GitHub**: public repo at `https://github.com/harshvardhan-takawale/adaptable-acoustic-fields`. Raw URL pattern: `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/<file>.md`.

## What's broken or stubbed

- **Zero-shot adaptation does NOT work at the current architecture**: held-out LSD 5.7-6.0 dB on all 6 unseen L values. Latents act as room-ID tags (PC1 vs L R² < 0), not as a smooth geometric embedding. **Recommended fix**: shrink HashGrid to `log2_hashmap_size=14, n_levels=14` and retrain (Chunk-2 recommendation, now empirically motivated by Chunk-3 latent_pca_1d.png). See `tasks/CHUNK_3_RESULTS.md` §10.
- `aaf/utils/`: empty `__init__.py` only.
- `aaf/_inference_ref/`: vendored INFER classes; reference-only, parse-checked but not runnable on the cluster (uses `.cuda()` at module init).
- No `auraloss` or perceptual loss wired in — deferred to Phase 4.
- HashGrid capacity per the Chunk-2 capacity warning: training succeeded *too well* on training rooms; the network memorised via hash params rather than via z_s. The retrain at smaller capacity is the next step.

## Recent changes (this chunk — 3)

- Added `INR2D_AutoDecoder` to `aaf/models/inr_2d.py`: subclass-style new class with `nn.Embedding(n_rooms, latent_dim)` for per-room learnable latents, widened sigma + signal MLP input dims by `latent_dim`, candidate-A z_s injection at *both* concat points. Added `room_id_map` alias on `ShoeboxDataset`.
- Wrote `aaf/train/multi_room.py:MultiRoomTrainer` with two-param-group optimizer (network lr=2e-4, latents lr=1e-3), 5-term loss (real + imag + log-amp + phase + λ‖z_s‖² with λ=1e-4), per-L AABB sub-batching inside each step, val every 1K iters with per-room LSD logging + latent-norm histograms.
- Wrote `aaf/eval/zero_shot.py:zero_shot_adapt`: deterministic 8-of-64 receiver subset, frozen-network + 2K-iter Adam(lr=1e-2) on a fresh `z_star`, then full-grid evaluation. 4 figures per L.
- Wrote `aaf/eval/latent_probing.py:probe_latents`: extracts trained latents, combines with all `zero_shot/L*/z_star.pt`, runs PCA, fits PC1 vs L linear regression. 3 figures + JSON.
- Wrote `scripts/multi_room_memory_check.py` and updated `scripts/multi_room_summary.py` to aggregate train + zero-shot + probe into one cross-room SUMMARY.
- Wrote 4 SLURM scripts (`multi_room_memory_check.sh`, `multi_room_train.sh`, `zero_shot_eval.sh`, `latent_probing.sh`) and `scripts/run_chunk3_pipeline.sh` orchestrator (chains memory_check → train → 6× zero-shot → probe via afterok). Default partition is **tron** for training.
- Wrote 3 new test files (test_autodecoder_2d, test_zero_shot, test_latent_probing). 70 tests now pass.
- Ran the full pipeline: training completed 30K iters on tron64 RTX 3070 in 2:38; 6 zero-shot evals completed in parallel on scavenger; latent probe completed.
- Documented the failure-mode finding in `tasks/CHUNK_3_RESULTS.md`: per-training-room target *met* (6/7 ≤ 1 dB val LSD); zero-shot target *not met* (held-out LSD 5.7-6.0 dB across all 6 unseen L); latent collapse confirmed via PCA (R² = −0.63 with L). Strong recommendation to retrain at smaller HashGrid capacity.

## Recent changes (Chunk 2)

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

- **Read for next chunk**: `tasks/CHUNK_3_RESULTS.md` (latest; §10 contains the recommended retrain config), `outputs/multi_room/dense/SUMMARY.md` (cross-train-and-zero-shot table + LSD-vs-L plot), `outputs/multi_room/dense/latent_probe/figures/latent_pca_1d.png` (the diagnostic that says z_s didn't learn L). Then `tasks/CHUNK_2_RESULTS.md` §6 + §8 for the original capacity warning.
- Open questions: `OPEN_QUESTIONS.md`. Q5 (cluster partition for long runs) closed by Chunk 3 (used 1 tron slot for training). Q10 (new): should the next iteration shrink HashGrid before another training run?
- Design log: `DECISIONS.md`. Now has 29 entries.
- Cluster how-to: `CLUSTER_INFO.md`. Default partition is `scavenger`; non-preemptible work goes on `tron`. Always set `LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}` after `conda activate aaf`. Chunk-3 pipeline driver is `scripts/run_chunk3_pipeline.sh`.

## How the manager can read this repo

Public; raw URLs are stable on `main`. Examples:

```
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/CONTEXT_FOR_MANAGER.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_0_RESULTS.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md
```
