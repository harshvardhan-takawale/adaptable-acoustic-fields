# CONTEXT_FOR_MANAGER.md

Manager re-orientation doc. Optimized for catching up in 5 minutes after time away. Updated at the end of every chunk.

**Last updated**: Chunk 3.6 in flight — 2026-05-11

## Project state

- **Phase**: 1 (2D shoebox sweep, 0–2 kHz, Track A — science).
- **Chunks completed**: Chunks 0/1/1.5/2/3 plus **Chunk 3.5/3.5+ (R0-R8 sweep)** plus **Chunk 3.6 in flight (band-limited eval + 6 inner-loop variants + FiLM/latent-jitter retrains)**. R0/R6/R7/R8 fully complete; R1-R5 still finishing on scavenger; Chunk 3.6 Track A (band-limited recompute) **landed**, Tracks B (36 variant ZS jobs) and C (C1 FiLM + C2 latent-jitter retrains on tron, plus their ZS evals) are queued / running.
- **What exists today (Chunk 3.6 additions)**: `aaf/eval/band_limited.py` (band-LSD pure functions); `aaf/eval/zero_shot_variants.py` (SimplexLatent + B1-B6 dispatch); `INR2D_AutoDecoder.conditioning_type ∈ {'concat','film'}` + `latent_jitter_sigma`; generalised `aaf/eval/zero_shot.py` (variable n_obs_receivers, init_strategy, n_restarts, multi-restart, integrated band-limited metrics, saves H_pred_all.pt); `configs/sweep/C{1,2}_*.yaml`; `scripts/{band_limited_recompute,track_a_summary,zero_shot_variant,track_b_summary,zero_shot_with_best_variant,chunk_3_6_final_summary}.py` + matching SLURM wrappers + `scripts/run_chunk3_6.sh` orchestrator (~68 jobs). **121 tests pass.**
- **Headline Chunk-3.6 Track A result**: modal regime (0-250 Hz) zero-shot LSD is significantly better than full-band across R0/R6/R7/R8 (3.54-3.69 dB modal vs 5.27-5.57 dB full), but still **0/24 (run, L) below the 2 dB modal target**. The visual impression that the modal regime tracks correctly is directionally right (~30% LSD improvement), but the gap to the meeting bar is still ~1.5 dB; Tracks B and C must close it. See `outputs/multi_room/sweep/band_limited_summary.md`.
- **Headline Chunk-3.5/3.5+ result (4 of 9 runs)**: per-training-room reconstruction met the ≤ 1.5 dB target on most rooms (1.29-1.70 dB val LSD) but **zero-shot at unseen L fails uniformly** (full-band held-out LSD 5.21-5.91 dB across R0/R6/R7/R8; 0/6 unseen L below 2 dB). The L-head + smaller hash + smaller latent did NOT fix zero-shot; the latent probe shows R6's train latents almost-monotonic with L (linear L-head IS shaping z_s) but zero-shot z_star tensors collapse to one region of latent space regardless of true L — **inner-loop adaptation is the bottleneck, not latent geometry**. See `tasks/CHUNK_3_5_RESULTS.md`.
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

## Recent changes (this chunk — 3.6, in flight)

- Added `aaf/eval/band_limited.py` (`compute_band_limited_metrics`, `band_indices`, `DEFAULT_BANDS`). LSD per band computed as ``mean |20*log10(|H_pred|/|H_target|)|`` over receivers and the bins inside the band.
- Generalised `aaf/eval/zero_shot.py`:
  - Removed the `n_obs_receivers != 8 → NotImplementedError` guard; added `select_obs_indices()` (n=8 keeps the existing 3×3-minus-centre pattern; n=32 uses a checkerboard half of the 8×8 grid; otherwise linspace).
  - Added params `init_strategy ∈ {random, nearest_train, simplex}`, `n_restarts`, `random_seed`. Multi-restart inner loop keeps the lowest-obs-LSD winner.
  - Saves `H_pred_all.pt` (additive, ~2 MB) and a separate `band_limited_metrics.json` next to the existing `metrics.json` for every future zero-shot run.
  - Reads `conditioning_type`, `latent_jitter_sigma` from `train_meta` with `.get(... default)` for backward compat.
- Added `INR2D_AutoDecoder.conditioning_type ∈ {'concat','film'}` and `latent_jitter_sigma`. FiLM is **input-side** (γ·encoded_feat + β before the tcnn block) because tcnn `FullyFusedMLP`/`CutlassMLP` are fused kernels and don't expose intermediate features. γ initialised to 1, β to 0 for identity at construction. Latent jitter is `z + N(0, σ²I)` inside `get_latent`, gated on `self.training` so eval/zero-shot are deterministic.
- Plumbed both new model knobs through `MultiRoomTrainCfg`, the YAML loader, and the CLI. Updated `aaf/eval/latent_probing.py` to read them with backward-compat defaults.
- Added `aaf/eval/zero_shot_variants.py`: `SimplexLatent` module (z_star = softmax(logits) @ Z_train) and `variant_kwargs(B1..B6)` dispatch table.
- Wrote 6 scripts + matching SLURM wrappers + `scripts/run_chunk3_6.sh` (~68-job orchestrator). Track A produces `outputs/multi_room/sweep/band_limited_summary.md` + a 4-panel band-LSD figure; Track B summary picks the inner-loop winner; final summary writes `tasks/CHUNK_3_6_RESULTS.md` and refreshes `SWEEP_SUMMARY.md` to include C1, C2.
- Wrote 4 new tests; updated `test_sweep_configs.py` to discover `C*.yaml` and to compute `expected_sigma_in` per `conditioning_type`. 121 tests pass.
- **Track A landed**: modal regime (0-250 Hz) zero-shot LSD is ~1.7 dB better than full-band across R0/R6/R7/R8 — but still 3.5-3.7 dB modal mean (target ≤ 2 dB), so 0/24 (run, L) below the meeting threshold. Tracks B and C still in flight.

## Recent changes (Chunk 3.5 / 3.5+)

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

- **Read first** (Chunk 3.6): `outputs/multi_room/sweep/band_limited_summary.md` (Track A — modal/transition/diffuse LSD per run), `outputs/inner_loop_experiments/SUMMARY.md` (Track B winner; created when the 36 variant ZS jobs finish), `tasks/CHUNK_3_6_RESULTS.md` (final writeup; created by the chunk's last SLURM job).
- **Background**: `tasks/CHUNK_3_5_RESULTS.md` (full failure-mode analysis from R0/R6/R7/R8), `outputs/multi_room/sweep/SWEEP_SUMMARY.md` (R0-R8 cross-run table + headline figures), `outputs/multi_room/sweep/R6_tiny_lhead/latent_probe/figures/latent_pca_1d.png` (the diagnostic showing train latents almost-monotonic with L while zero-shot z_star collapses).
- Open questions: `OPEN_QUESTIONS.md`. Q10 (HashGrid resize) closed by 3.5. Q11 (inner-loop unblock) is partly answered by Track B once it lands.
- Design log: `DECISIONS.md`. 35 entries after Chunk 3.6 (band-limited eval as standard; FiLM-on-encoded-features design choice; latent jitter at training time only).
- Cluster how-to: `CLUSTER_INFO.md`. Tron RTX 2080 Ti for training; scavenger for evals. Pipeline drivers: `scripts/run_chunk3_5_sweep.sh`, `scripts/run_chunk3_5_addendum.sh`, **`scripts/run_chunk3_6.sh`** (current).

## How the manager can read this repo

Public; raw URLs are stable on `main`. Examples:

```
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/CONTEXT_FOR_MANAGER.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_0_RESULTS.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md
```
