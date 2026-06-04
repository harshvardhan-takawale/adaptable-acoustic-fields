# CONTEXT_FOR_MANAGER.md

Manager re-orientation doc. Optimized for catching up in 5 minutes after time away. Updated at the end of every chunk.

**Last updated**: Chunk P2-2 COMPLETE — 2026-06-04. **Mixed result**.

## Phase 2 — second chunk (P2-2): multi-room 3D conditioning + zero-shot (complete)

**Status**: COMPLETE. Headline target NOT met (0/8 rooms reach mag corr ≥ 0.9; target was ≥ 5/8). But a clear diagnosis lands: the latent manifold correctly encodes (L, W, H) (R² > 0.96 on every axis), while the decoder/renderer never reached good in-distribution LSD because per-iter sampling at `batch=4` over 23K (room, rx) pairs is ~200× sparser than Phase 1's. See `tasks/CHUNK_P2_2_RESULTS.md`.

**Headline numbers**:
- Training: M1 d=16 early-stopped at 24K iters (val LSD 6.16 dB). M2 d=32 early-stopped at 13K iters (val LSD 6.69 dB). Both flat-lined far from the 2.5 dB target.
- **Latent manifold (the bright spot)**: per-axis R²_full = L:0.991, W:0.967, H:0.974 (M1) and L:0.997, W:0.990, H:0.996 (M2). Geometry-head MAE 1.1-3.5 cm per axis. Phase-1's "PC1 vs L R²=0.987" generalizes cleanly to 3D.
- **Zero-shot mag corr**: 0.47-0.59 (M1) / 0.47-0.64 (M2) across the 8 test rooms. Modal MAE 0.64-1.30 Hz (matches 2D's range). Phase-1's full-band LSD ~5 dB → P2-2's 7.5 dB (1.5× worse going from 1D to 3D).

**Q12 closes**: d=16 is sufficient for the latent representation. d=32 didn't translate into better zero-shot mag corr; binding constraint is elsewhere.

**Three pipeline fixes that landed mid-chunk** (all in DECISIONS.md):
- **34aebd4** — move multi-room (room, rx) tensors to CPU; transfer per-iter (was OOM'ing on 10.57 GB tron nodes).
- **8ff7f45** — best-effort TB writes (don't crash training on transient NFS errors).
- **238d126** — chunk zero-shot inner-loop obs receivers (avoid 12 GB OOM at batch=8).

**P2-3 recommendation (manager spec input)**: fix the per-iter sampling problem — pin tron `qos=high` A100/A6000 (40-48 GB), raise `batch_size` 4 → 32, raise `n_pts_per_ray` 16 → 32. Per-iter coverage goes 8×; would close most of the in-distribution gap. Secondary: relax early-stop and/or 3× iters.

---

## Phase 2 — first chunk (P2-1): 3D port (complete)

### (P2-1 section unchanged below)

**Scope**: train an `INR3D_AutoDecoder` (FiLM + latent jitter + linear geometry head) jointly on 45 LHS training rooms; zero-shot adapt to 8 unseen maximin test rooms; evaluate primarily via `aaf/eval/signal_level.py`. Spec-prescribed acceptance: magnitude correlation ≥ 0.9 in 0-500 Hz on ≥ 5/8 test rooms. **No** dimension-sweep eval (P2-3); **no** EDC fidelity work; **no** conditioning-mechanism sweep (FiLM only).

**Key files (new)**:
- `aaf/models/inr_3d.py:INR3D_AutoDecoder` — appended; FiLM at sigma + signal branches, per-room latents, linear `predict_geometry(z) → [B, 3]`.
- `aaf/train/multi_room_3d.py` — joint training; two param groups; 4-spec + λ‖z‖² + 0.1·L1(geom_pred, true) loss.
- `aaf/eval/zero_shot_3d.py` — inner-loop adaptation; 8 corner-receivers `OBS_INDICES_3D=[0,7,56,63,448,455,504,511]`; 2000 iters; full 504-receiver held-out eval.
- `aaf/eval/latent_probe_3d.py` — sklearn PCA + per-axis R² (L, W, H) via `_r2_full_latent` and `_r2_per_pc`.
- `configs/sweep_3d/{M1_45rooms, M2_45rooms_d32}.yaml` — main run + d=32 hedge.
- `scripts/{multi_room_3d_summary, run_p2_2_pipeline}.py` + 6 new SLURM scripts.
- 3 new test files (~20 tests). Memory-check `--mode auto_decoder` flag added to `scripts/memory_check_3d.py`.

**D29 hedge enabled (user-approved)**: M1 (d=16) and M2 (d=32) run in parallel on tron. Definitive d=16 vs d=32 comparison at chunk closeout.

**Decisions logged**: D19-D31 (DECISIONS.md to be appended at closeout).
**Open questions refreshed (P2-1 outcomes folded in)**: Q12 (HashGrid 18/16/1.38 ~right; closes when M1/M2 in-distribution LSD lands), Q13 (max_order=12 covers ~175 ms; EDC T-numbers flagged "not yet calibrated"), Q14 (answer (c) signal mag corr ≥ 0.9 in 0-500 Hz adopted as P2-2 target).

---

## Phase 2 — first chunk (P2-1): 3D port (complete)

**Status**: COMPLETE. All 4 deliverables met. See `tasks/CHUNK_P2_1_RESULTS.md` for full numbers.

**Headline (5 de-risk rooms, single-room 3D overfit)**:
- Modal MAE 0.61-1.18 Hz on f<f_Schroeder (vs spec target ≤3 Hz) ✅
- Full-band LSD 1.31-1.77 dB (vs Phase-1 2D 0.36-0.42 — 3-4× ratio, in spec) ✅
- Signal correlations: mag 0.95-0.98, phase (mw) 0.95-0.98, RIR Pearson 0.97-0.99, env 0.98-0.99 ✅
- 45-room LHS training dataset on disk (2.4 GB, 50 HDF5 files) ✅
- Signal-level eval suite (Dolby foundation) working end-to-end ✅

**Surprise finding**: per-band LSD *flips direction* in 3D — modal band (0-250 Hz) has the **highest** LSD (~2.1 dB), and LSD *decreases* monotonically with frequency. Opposite of 2D. Cause: 3D modal density at f≤250 Hz is ~11× higher than 2D (136 vs ~12 modes for the box-center room) — modal regime is denser → harder.

**Two pipeline fixes landed during the chunk** (both in DECISIONS.md):
- **D6 revised** (commit 8b900a6): the per-mode Python loop in `modal_rir_3d` took 30+ min on the largest room. Vectorized into single complex BLAS matmul + lowered ISM `max_order` cap 17 → 12. Now 23 s/room.
- **SLURM training cascade** (commit 1662aa7): training script now honors the memory-check chosen config (16, 16, 16, 4) instead of defaulting to canonical (32, 8) which OOM'd on smaller tron GPUs.
- **Eval chunk size** (commit 5dc1710): eval was using receiver-chunk=8 → OOM. Now matches the train_batch (4).

**Scope of P2-1**: port the 2D pipeline (sim, renderer, model, train, eval) to 3D shoeboxes; build 5 single-room de-risk overfits; generate the 45-room LHS training dataset for P2-2; add the signal-level eval suite Dolby requested. NO auto-decoder, NO zero-shot, NO multi-room conditioning — those land in P2-2.

**Key files (new)**:
- `aaf/sim/{ism_3d,analytical_modal_3d}.py`, `aaf/renderers/freq_3d.py`, `aaf/models/inr_3d.py`
- `aaf/data/sample_rooms_3d.py` (LHS + de-risk + test samplers), `aaf/data/loader.py:Shoebox3DDataset`
- `aaf/train/single_room_3d.py`, `aaf/eval/single_room_3d_eval.py`
- `aaf/eval/signal_level.py` — **stable Phase-2 API** (3-layer; reused by P2-2 zero-shot eval)
- `configs/sweeps_3d/{derisk,train,test}_rooms.yaml` (generated by `scripts/sample_rooms_3d.py`)
- `scripts/{sample_rooms_3d,budget_check_3d,build_3d_dataset,memory_check_3d,build_3d_manifest,single_room_3d_summary}.py`
- 9 SLURM scripts under `scripts/slurm/*_3d*.sh` + `scripts/run_p2_1_pipeline.sh` (DAG orchestrator)
- 6 new test files. **190 tests pass.**

**Critical design decisions for P2-2 to inherit** (DECISIONS.md D1-D18):
- Room ranges: L∈[3,6], W∈[3,5], H∈[2.5,4]; α=0.15 fixed; receivers 8×8×8=512 @ 0.3 m margin (D1-D4).
- ISM `max_order` hard-capped at 17 (D6) — 3D image-source count would blow up under the 2D auto-rule.
- 3D renderer: n_azi×n_ele + 2 poles = 258 rays; 3-axis AABB slab; σ+jβ + transmittance + geometric phase carry over (D8, D9).
- HashGrid 3D: `log2_hashmap_size=18, n_levels=16, per_level_scale=1.38` (D10, user-approved override of spec's 16/16/1.5 based on collision-rate analysis).
- Training pinned to tron RTX 2080 Ti (D13) — TITAN X too small for 3D activations.
- Dataset generation: 5 de-risk on tron `--array=0-4%4`, 45 training on scavenger wide array; idempotent + atomic via `.h5.done` sentinels (D16).
- Signal-level eval API: 3-layer factoring (D17) — `compute_signal_metrics` + `make_signal_plots` are stable for all of Phase 2.

**Open questions opened by this chunk** (OPEN_QUESTIONS.md Q12, Q13, Q14):
- Q12 — empirical HashGrid capacity diagnosis (will answer after single-room evals).
- Q13 — ISM `max_order=17` tail-truncation tolerance (will answer via signal-level eval's late_corr).
- Q14 — reinterpreting the 2 dB modal target in 3D (variable f_Schroeder; modal density 3× higher in 0-250 Hz than 2D).

**Pipeline runtime expectation**: ~10 hours wall-clock from kickoff to PR-ready, dominated by tron training (5×~4 hr in parallel at qos=default's 4-GPU cap).

---

## Project state (Phase 1 — complete)

- **Phase**: 1 complete (2D shoebox sweep, 0–2 kHz, Track A — science). Phase 2 (3D) underway via Chunk P2-1.
- **Chunks completed**: Chunks 0/1/1.5/2/3 plus Chunk 3.5/3.5+ (R0-R8 sweep), Chunk 3.6 (band-limited eval + 5 inner-loop variants + FiLM/jitter retrains), and **Chunk 3.7 (meeting visual story + denser-sweep + FiLM-LoRA + chunked-receiver)**. All 13 trained configurations (R0-R8 + C1/C2 + D1/D2) complete; spatial-node verification GREEN at all 6 unseen L.
- **What exists today (Chunk 3.7 additions)**: `aaf/eval/spatial_modes.py` (V0 helpers: complex Pearson, node match, mode-shape fit, analytical shape gen); `INR2D_AutoDecoder.conditioning_type='film_lora'` + `lora_rank` (output-side rank-r additive adapter, zero-init proj); `aaf/eval/zero_shot.py` chunked-receiver gradient accumulation via `chunk_size` kwarg + `get_z()`-per-chunk fix; `aaf/eval/zero_shot_variants.py:B7` (n_obs=32 via chunking); `configs/sweeps/dense_15.yaml` + `configs/sweep/{D1_dense15,D2_filmlora}.yaml`; V0/V1/V2/V3/V4 scripts (spatial check, cross-L summary, modal-tracking plot, audio demo, meeting deck assembler); `scripts/run_chunk3_7.sh` orchestrator. **125 tests pass.**
- **Headline Chunk-3.7 result (POSITIVE — first single-chunk improvement of the project)**: Track V's V0+V1 spatial-node alignment is GREEN at all 6 unseen L (mean Pearson 0.86-0.94 between predicted and ISM pressure fields on the 8×8 grid, modal MAE on matched peaks 1.04 Hz). Track I's I1 denser-training sweep (D1_dense15, 15 rooms at 0.2 m vs the original 7 at 0.5 m) **drops modal-regime zero-shot LSD from 3.5 dB to 2.55 dB** — the largest single-chunk modal-LSD improvement of the project. At L=5.25 (2.33) and L=5.75 (2.28) D1 is within **0.3 dB** of the 2 dB target. The three Track I experiments together establish: I3 (n_obs=32) and I2 (rank-8 LoRA on decoder output) don't help, but **I1 (more training rooms) does** — modal LSD is data-density-bound, not capacity-bound or observation-bound. Meeting deck is at `outputs/meeting_assets/` (7 assets + 3 WAVs, recommended order in `00_README.md`). See `tasks/CHUNK_3_7_RESULTS.md`.
- **Headline Chunk-3.6 result (definitive, partially superseded)**: 11 configurations × 5 inner-loop strategies × 6 unseen L all gave **3.5-3.7 dB modal** — none below 2 dB. Was conclusive that inner-loop optimisation isn't the bottleneck; FiLM + latent jitter improve val LSD but not zero-shot. Chunk 3.7 then showed the ACTUAL fix is more training rooms.
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

## Recent changes (this chunk — 3.8, complete)

- **Meeting deck finalised** at `outputs/meeting_assets/` with 10-slide
  narrative + per-slide reviewer Q&A in
  [`outputs/meeting_assets/DECK_NARRATIVE.md`](outputs/meeting_assets/DECK_NARRATIVE.md).
- **Polished three core visuals** to presentation resolution (≥ 1920 px on
  the long edge): 04 modal-tracking (added the required subtitle "Matched
  modes only (31/139)..."), 05 spatial-nodes-grid (added 0.7 threshold
  contour overlay), 06 latent-manifold (regenerated from `latent_probe.json`
  at 1977×1179 with R² = 0.987 annotated prominently).
- **Two new visuals**: 05a (`05a_spatial_modes_L5_25.png`, 2365×1725) — the
  six-mode ISM-vs-predicted headliner at L=5.25 m; 08
  (`08_progress_trajectory.png`, 2018×1171) — pure-modal LSD trajectory
  across Chunks 3/3.5/3.6/3.7 (3.70 → 3.66 → 3.51 → 2.55 dB).
- **Caption honesty audit**: all 9 captions verified. Audio demo caveat
  updated to the spec's verbatim phrasing ("qualitative, full-band LSD
  ~4-5 dB; demo shows smooth latent morphing, not faithful reconstruction").
- **New post-processing scripts**: `scripts/make_05a_spatial_modes.py`,
  `scripts/make_06_latent_manifold.py`, `scripts/make_08_trajectory.py`.
  Modifications to `scripts/modal_tracking_plot.py`,
  `scripts/spatial_nodes_summary.py`, `scripts/assemble_meeting_assets.py`.
- **No retraining, no new SLURM jobs.** Pure post-processing on existing
  Chunk-3.7 artefacts. Full asset checklist in
  [`tasks/CHUNK_3_8_RESULTS.md`](tasks/CHUNK_3_8_RESULTS.md).

## Recent changes (Chunk 3.7, complete)

- **V0 spatial-node alignment check** at L=4.25 produced GREEN: all 6 first eigenfrequencies have complex spatial Pearson correlation ≥ 0.84 between predicted and ISM pressure fields on the 8×8 grid. V1 extended to the other 5 unseen L; ALL 6 L GREEN (6/6 modes ≥ 0.7 per L; mean correlation 0.86-0.94).
- **V2 modal-tracking**: 31 matched analytical-mode pairs across all 6 L on the centre receiver; MAE = 1.04 Hz; recall 22.3% (we get few but the ones we get are correct).
- **V3 audio demo**: 3 length-morphing WAVs at L ∈ {3.25, 4.25, 5.75} passed peak/median ≥ 3 sanity check (33-40).
- **V4 deck assembled** at `outputs/meeting_assets/` (7 assets + 3 WAVs) with `00_README.md` manifest and honest 1-2 sentence captions per asset.
- **I1 denser training sweep (D1_dense15)** — 15 rooms at 0.2 m spacing covering [3.0, 5.8] — trained on tron in 2 h. Val LSD 2.37 dB (worse than C2's 1.43 because the same 8-D latent + capacity-reduced HashGrid has to span more anchors), but **zero-shot modal LSD = 2.55 dB** with B1 baseline inner loop — the LARGEST single-chunk improvement of the project. Per-L: 2.81, 2.58, 2.69, 2.63, 2.33, 2.28 — at the upper-half L's the model is **within 0.3 dB** of the 2 dB target.
- **I2 FiLM + rank-8 LoRA (D2_filmlora)** — additive output-side adapter, zero-init projection so model starts bit-identical to plain FiLM. Train val LSD 1.42 dB, zero-shot modal 3.35 dB (B6 winner). Marginally better than C2's 3.51 but doesn't break the ceiling. The LoRA path's A·B·proj multiplicative chain at zero-init couldn't escape the local optimum FiLM already finds.
- **I3 chunked-receiver n_obs=32 (B7 on C2)** — re-implemented inner-loop forward+backward with per-chunk `loss.backward()`. Two bugs caught + fixed: score-stage OOM on TITAN X (chunked the eval-mode score forward too), and a graph-freed error when init_strategy='simplex' (fixed by calling `get_z()` per chunk so each chunk has its own fresh autograd subgraph; no-op for random init). Modal LSD 3.53 — same as C2 + B6. 4× the observed receivers does not break the ceiling.
- **Mechanism-level finding**: I3 says modal LSD is NOT info-bound; I2 says it's not capacity-bound (in the FiLM-output-LoRA direction); I1 says it IS data-density-bound. The next chunk should push I1 further.

## Recent changes (Chunk 3.6)

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
- **Track A landed**: modal regime (0-250 Hz) zero-shot LSD is ~1.7 dB better than full-band across R0/R6/R7/R8 (3.54-3.69 dB modal vs 5.27-5.57 dB full) — but still 0/24 (run, L) below the 2 dB target.
- **Track B landed**: 5 inner-loop strategies on R6 cluster within 0.14 dB modal (B1 3.66, B3 3.70, B4 3.63, B5 3.62, B6 winner 3.52). Strong evidence that inner-loop optimisation is **NOT** the bottleneck (a more focused diagnosis than Chunk 3.5's hypothesis). B2 n_obs=32 OOM'd on 1080 Ti; in-spec skippable.
- **Track C landed**: C1_film val LSD 1.38 dB (best in-distribution of any config), C2_latent_jitter val LSD 1.43 dB. But zero-shot at unseen L — even with the Track B winner — is essentially unchanged (C1 + B6 = 3.62 dB modal, C2 + B6 = 3.51 dB modal vs R6 + B6 = 3.52 dB). FiLM slightly *hurt* zero-shot (classic expressivity/generalisation tradeoff); latent jitter was neutral.

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

- **Read first (Chunk 3.8 — meeting deck)**: [outputs/meeting_assets/DECK_NARRATIVE.md](outputs/meeting_assets/DECK_NARRATIVE.md) — 10-slide narrative with per-slide talking points, exact numeric claims, anticipated Dolby reviewer Q&A. Then [tasks/CHUNK_3_8_RESULTS.md](tasks/CHUNK_3_8_RESULTS.md) (asset checklist + honesty audit).
- **Background (Chunk 3.7)**: [tasks/CHUNK_3_7_RESULTS.md](tasks/CHUNK_3_7_RESULTS.md) — V0/V1 GREEN, **I1 modal 2.55 dB** (project best), ranked next-iteration recommendations. [outputs/spatial_nodes_check/SUMMARY.md](outputs/spatial_nodes_check/SUMMARY.md) (V1 cross-L correlation matrix). [outputs/meeting_assets/00_README.md](outputs/meeting_assets/00_README.md) (deck manifest).
- **Background**: `tasks/CHUNK_3_6_RESULTS.md` (the negative-result chunk that motivated I1), `tasks/CHUNK_3_5_RESULTS.md`, `outputs/multi_room/sweep/SWEEP_SUMMARY.md` (R0-R8 + C1/C2 cross-run table).
- Open questions: `OPEN_QUESTIONS.md`. Q11 (zero-shot bottleneck): Chunk 3.7's mechanism finding refines it — modal LSD is data-density-bound (I1 confirms), not info-bound (I3) or capacity-bound (I2 LoRA direction).
- Design log: `DECISIONS.md`. ≈37 entries after Chunk 3.7 (additions: FiLM-LoRA output-side rank-r design + zero-init proj; chunked-receiver gradient accumulation requires per-chunk `get_z()` for simplex init).
- Cluster how-to: `CLUSTER_INFO.md`. Tron RTX 2080 Ti for training; scavenger for evals. Pipeline drivers: `scripts/run_chunk3_5_sweep.sh`, `scripts/run_chunk3_5_addendum.sh`, `scripts/run_chunk3_6.sh`, **`scripts/run_chunk3_7.sh`** (current).

## How the manager can read this repo

Public; raw URLs are stable on `main`. Examples:

```
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/CONTEXT_FOR_MANAGER.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_0_RESULTS.md
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md
```
