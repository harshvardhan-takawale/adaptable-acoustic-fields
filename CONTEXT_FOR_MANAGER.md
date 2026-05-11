# CONTEXT_FOR_MANAGER.md

Manager re-orientation doc. Optimized for catching up in 5 minutes after time away. Updated at the end of every chunk.

**Last updated**: Chunk 3.5 / 3.5+ partial — 2026-05-10

## Project state

- **Phase**: 1 (2D shoebox sweep, 0–2 kHz, Track A — science).
- **Chunks completed**: Chunks 0/1/1.5/2/3 plus **Chunk 3.5/3.5+ (capacity-reduced retrain + L-head + 9-config sweep R0-R8)**. R0/R6/R7/R8 fully complete; R1-R5 still training on slow scavenger TITAN X nodes (~3 h to go).
- **What exists today**: everything prior plus `INR2D_AutoDecoder.l_head` (linear OR mlp_32 architectures), the 9 sweep YAML configs, `scripts/sweep_smoke_check.py` + `scripts/addendum_smoke.py`, `scripts/sweep_summary.py`, and `scripts/run_chunk3_5_sweep.sh` + `scripts/run_chunk3_5_addendum.sh` orchestrators. **4 fully-complete sweep runs on disk** (R0/R6/R7/R8) with train_meta + scalars + 6 zero-shot dirs each + latent probes; partial cross-sweep `outputs/multi_room/sweep/SWEEP_SUMMARY.md` + 4 headline figures. **95 tests pass.**
- **Headline Chunk-3.5/3.5+ result (4 of 9 runs)**: per-training-room reconstruction met the ≤ 1.5 dB target on most rooms (1.29-1.70 dB val LSD) but **zero-shot at unseen L still fails** (held-out LSD 5.21-5.91 dB across all 4 runs; 0/6 unseen L below the 2 dB target). The L-head + smaller hash + smaller latent did NOT fix zero-shot. The latent probe shows R6's train latents almost-monotonic with L (linear L-head IS shaping z_s) but **zero-shot z_star tensors collapse to one region of latent space regardless of true L** — the inner-loop adaptation is the new bottleneck, not the latent geometry. See `tasks/CHUNK_3_5_RESULTS.md` for the full analysis + concrete next-iteration paths.
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
- **Multi-room shared training (Chunk 3)**: 7 rooms (dense-sweep `train_L`) trained 30K iters in 2:38 on tron64 RTX 3070. Per-room val LSD 0.66-0.98 dB (6 of 7 rooms ≤ 1 dB target). Latents stayed healthy in magnitude but collapsed in *structure*.
- **Zero-shot eval pipeline (Chunk 3)**: works mechanically (held-out LSD 5.7-6.0 dB; latent collapse).
- **Latent probe**: `aaf.eval.latent_probing` produces PCA + linear-fit-to-L diagnostic. Reused across Chunks 3 / 3.5 / 3.5+.
- **Chunk-3.5 sweep infrastructure (R0-R5)**: 6 trainings + 36 zero-shot evals + 6 probes + 1 summary, orchestrated by `scripts/run_chunk3_5_sweep.sh`. R0 complete (tron); R1-R5 still training on TITAN X (slow).
- **Chunk-3.5+ addendum (R6-R8)**: linear-L-head sweep (R6/R7/R8 = small-hash + linear L-head, medium-hash + linear, tiny-2D-latent + linear). All 3 trained on tron RTX 2080 Ti in ~1:55 each, all 18 zero-shot evals + 3 probes complete.
- **Sweep summary**: `outputs/multi_room/sweep/SWEEP_SUMMARY.md` covers the 4 complete runs; the auto re-summary job 6815800 will overwrite when R1-R5 finish.
- **GitHub**: public repo at `https://github.com/harshvardhan-takawale/adaptable-acoustic-fields`. Raw URL pattern: `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/<file>.md`.

## What's broken or stubbed

- **Zero-shot adaptation does NOT work — confirmed across 4 architecturally-diverse runs in Chunk 3.5/3.5+**: held-out LSD 5.21-5.91 dB on all 4 complete runs (R0/R6/R7/R8). The Chunk-3 capacity-reduction recommendation was correctly implemented but the failure persists. **The inner-loop adaptation is the new bottleneck**, not the latent geometry. See `tasks/CHUNK_3_5_RESULTS.md` §6/§11 for the diagnosis + recommended next-iteration paths (more observed receivers, multi-restart inner adaptation, longer inner loop, latent-hull-constrained adaptation).
- `aaf/utils/`: empty `__init__.py` only.
- `aaf/_inference_ref/`: vendored INFER classes; reference-only, parse-checked but not runnable on the cluster (uses `.cuda()` at module init).
- No `auraloss` or perceptual loss wired in — deferred to Phase 4.

## Recent changes (this chunk — 3.5 / 3.5+)

- Added `INR2D_AutoDecoder.l_head` (Chunk-3.5) with two architectures: `mlp_32` (Linear→ReLU→Linear, R0-R5 default) and `linear` (Chunk-3.5+ addendum: single `nn.Linear(d, 1)` for the strongest inductive bias toward 1-D latent manifold, used by R6/R7/R8). Aux loss term `l_head_weight · L1(predict_L(z_s), L_true)` added to `MultiRoomTrainer`. Backward-compat preserved: Chunk-3 train_meta files load via `.get(... default)`.
- Added `--config <yaml>` CLI to `aaf.train.multi_room`; sweep configs in `configs/sweep/R{0..8}_*.yaml` carry per-run hyperparameters (hash size, n_levels, latent_dim, l_head_weight, l_head_arch, λ_latent_l2). Trainer's `train_meta.json` records all sweep params.
- Wrote 9 hyperparameter YAMLs, 2 smoke checks (`scripts/sweep_smoke_check.py` + `scripts/addendum_smoke.py`), `scripts/sweep_summary.py` (aggregates per-train-room + per-zero-shot-L + latent-probe; produces SWEEP_SUMMARY.md + 4 headline figures), `scripts/slurm/{sweep_train, sweep_smoke_check, sweep_summary, addendum_smoke}.sh`, and 2 orchestrators (`scripts/run_chunk3_5_sweep.sh`, `scripts/run_chunk3_5_addendum.sh`).
- Wrote 4 new tests (test_l_head: linear arch + unknown-arch error; test_sweep_configs: 9 YAMLs present + each YAML produces a model with the right hash + latent + L-head). 95 tests pass.
- Ran the full sweep: R0 + addendum R6/R7/R8 complete (4/9 runs done with full ZS + probe). R1-R5 still training (~3 h to go). Re-summary job 6815800 will fire automatically.
- **Chunk 3.5/3.5+ result is conclusive (negative)**: 4 architecturally-diverse runs all fail zero-shot at 5.2-5.9 dB. The Chunk-3 capacity diagnosis was correct but addressing capacity alone doesn't fix zero-shot — the bottleneck shifted to the inner-loop adaptation. Documented in `tasks/CHUNK_3_5_RESULTS.md` with concrete next-iteration paths.

## Recent changes (Chunk 3)

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

- **Read for next chunk**: `tasks/CHUNK_3_5_RESULTS.md` (latest, contains the full ablation analysis + §11 next-iteration recommendations), `outputs/multi_room/sweep/SWEEP_SUMMARY.md` (cross-run table + 4 headline figures), `outputs/multi_room/sweep/R6_tiny_lhead/latent_probe/figures/latent_pca_1d.png` (the diagnostic that shows train latents trending monotonic with L while test latents collapse to one region — this is the load-bearing image for understanding the failure).
- Open questions: `OPEN_QUESTIONS.md`. Q10 (HashGrid resize) was answered by Chunk 3.5 (yes, but doesn't fix zero-shot). Q11 (new): which inner-loop adaptation strategy unblocks zero-shot?
- Design log: `DECISIONS.md`. Now has 32 entries (3 added in 3.5+: linear L-head architecture; sweep design; final-config decision).
- Cluster how-to: `CLUSTER_INFO.md`. Tron RTX 2080 Ti (~5× scavenger TITAN X) is the right place for training; scavenger for evals. Pipeline drivers: `scripts/run_chunk3_5_sweep.sh` + `scripts/run_chunk3_5_addendum.sh`.

## How the manager can read this repo

Public; raw URLs are stable on `main`. Examples:

```
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/CONTEXT_FOR_MANAGER.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_0_RESULTS.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md
```
