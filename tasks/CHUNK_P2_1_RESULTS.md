# Chunk P2-1 — Phase 2 begins: 3D port, single-room baseline, dataset, signal-level eval

**Status**: infrastructure complete; cluster pipeline submitted. This document
is updated incrementally as jobs complete.

**Started**: 2026-06-04 (Phase 2 kickoff).

## 1. Goal & scope

First chunk of Phase 2. Goals (verbatim from the manager spec):

1. **3D port** of the renderer, model, simulator. Verify it works.
2. **Single-room 3D baseline** — overfit 5 de-risk 3D shoebox rooms spanning
   the (L, W, H) range; confirm reconstruction at the analytical noise floor.
3. **3D room sampling + dataset** — generate the 45-room Latin hypercube
   training dataset (+ 8 structured test rooms as config-only) that P2-2
   will train on.
4. **Signal-level evaluation suite** — magnitude correlation, phase
   correlation, time-domain RIR analysis, EDC, early/late split, Hilbert
   envelope (Dolby-requested foundation).

This chunk is intentionally single-room only. No auto-decoder, no zero-shot,
no multi-room conditioning — those land in P2-2 once the foundation is solid.

## 2. What was built (infrastructure)

### Simulation pipeline
- [aaf/sim/ism_3d.py](aaf/sim/ism_3d.py): `simulate_room_3d(cfg)` wrapping
  `pyroomacoustics.ShoeBox(p=[L,W,H], dim=3)`. `max_order` hard-capped at
  `MAX_ORDER_CAP=17` (D6). Returns `{rir_time, H_complex, meta}` with 3D-
  specific metadata (`L, W, H, T60_sabine_3d, schroeder_freq_hz`, truncation
  flags). Genuine 3D Schroeder formula `f_s = 2000·√(T60/V)`.
- [aaf/sim/analytical_modal_3d.py](aaf/sim/analytical_modal_3d.py):
  `eigenfrequencies_3d`, `modal_rir_3d`, `sabine_damping_3d`, `EigenFreq3D`
  dataclass. Rigid-wall mode shapes `Φ = cos·cos·cos`; Lorentzian modal sum
  with Sabine 3D damping `γ = c·α·S / (8·V)`. Dedup at 0.01 Hz handles cubic
  degeneracies.

### Rendering pipeline
- [aaf/renderers/freq_3d.py](aaf/renderers/freq_3d.py): `FreqRenderer3D` with
  `n_azi × n_ele + 2 = 258` rays (D8), 3D AABB slab intersection (D9),
  σ+jβ + amplitude/phase transmittance + geometric phase identical to
  Phase 1's 2D version. `use_geometric_attn=False` (D11). Per-iteration
  azimuth jitter in `.train()` mode; deterministic in `.eval()`.

### Model
- [aaf/models/inr_3d.py](aaf/models/inr_3d.py): `INR3D_Single` with six
  `tcnn.Encoding(3, …)` encoders. Default HashGrid `log2_hashmap_size=18,
  n_levels=16, per_level_scale=1.38` (D10, user-approved override of spec's
  16/16/1.5; calibrated for 3D collision rate and λ/2 finest resolution).
  σ+jβ decomposition + softplus on σ + RFFT symmetry mask on DC/Nyquist all
  carry over from 2D.

### Data
- [aaf/data/sample_rooms_3d.py](aaf/data/sample_rooms_3d.py): LHS training
  sampler (`scipy.stats.qmc.LatinHypercube`, seed=42, reject-near-cubic),
  5 spec-prescribed de-risk rooms, 8 greedy-maximin test rooms (D14, D15).
- [aaf/data/dataset_builder.py](aaf/data/dataset_builder.py): added
  `room_filename_3d` and `write_room_3d_to_h5` (same HDF5 layout as 2D with
  3D root attrs).
- [aaf/data/loader.py](aaf/data/loader.py): new `Shoebox3DDataset` (same API
  as `ShoeboxDataset` but rooms-yaml-driven and emits 3D positions).

### Training + eval
- [aaf/train/single_room_3d.py](aaf/train/single_room_3d.py): mirrors
  `single_room.py` — 4-term loss (1, 1, 1, 0.1), Adam lr=2e-4 + cosine
  anneal, grad clip + NaN/Inf masking, relative-improvement early stop
  (1% over 2K window after 2K warmup), checkpoint every 2.5K iters,
  auto-resume. `n_iters=15_000` default (vs 2D's 10K).
- [aaf/eval/single_room_3d_eval.py](aaf/eval/single_room_3d_eval.py):
  mirrors `single_room_eval.py` + integrates the new signal-level suite.
  Modal MAE on `f<f_modal_cap` band only (D18). 5 traditional figures + 5
  signal-level figures per room.
- [aaf/eval/signal_level.py](aaf/eval/signal_level.py): **stable Phase-2 API**.
  3-layer factoring (components / aggregator / plots) per D17. Functions:
  `magnitude_correlation`, `phase_correlation_mag_weighted`, `per_band_lsd`,
  `rir_pearson`, `edc_db`, `edc_error`, `early_late_corr`, `envelope_corr`;
  aggregator `compute_signal_metrics`; plotter `make_signal_plots`.

### Configs
- `configs/sweeps_3d/derisk_rooms.yaml` (5 rooms): box center + 4 corners.
- `configs/sweeps_3d/train_rooms.yaml` (45 rooms): LHS draw at seed=42.
- `configs/sweeps_3d/test_rooms.yaml` (8 rooms): structured maximin interior.
  Test rooms are config-only this chunk; full simulation deferred to P2-2.

### Scripts (new)
- [scripts/sample_rooms_3d.py](scripts/sample_rooms_3d.py): YAML generator.
- [scripts/budget_check_3d.py](scripts/budget_check_3d.py): simulates 2 rooms
  (smallest + largest of de-risk), reports wall-clock + HDF5 size. Hard-fails
  pipeline at >10 min/room or >500 MB/room.
- [scripts/build_3d_dataset.py](scripts/build_3d_dataset.py): SLURM-array
  entry; idempotent + atomic via `.h5.done` sentinel (D16).
- [scripts/memory_check_3d.py](scripts/memory_check_3d.py): GPU cascade per
  D12, writes `outputs/memory_check_3d/REPORT.md`.
- [scripts/build_3d_manifest.py](scripts/build_3d_manifest.py): refreshes
  `data/track_a_3d/manifest.json` from sentinel files.
- [scripts/single_room_3d_summary.py](scripts/single_room_3d_summary.py):
  cross-room aggregator → `outputs/single_room_3d/SUMMARY.md`.

### SLURM scripts (new)
- `scripts/slurm/sample_rooms_3d.sh` (scavenger, 2 CPU, 4 GB, no GPU)
- `scripts/slurm/budget_check_3d.sh` (scavenger, 4 CPU, 16 GB, no GPU)
- `scripts/slurm/build_3d_derisk.sh` (tron, `--array=0-4%4`, 4 CPU, 16 GB, no GPU)
- `scripts/slurm/build_3d_train.sh` (scavenger, `--array=0-44`, 4 CPU, 16 GB, no GPU)
- `scripts/slurm/build_3d_manifest.sh` (scavenger, 2 CPU, 4 GB, no GPU)
- `scripts/slurm/memory_check_3d.sh` (scavenger, 4 CPU, 24 GB, 1 GPU)
- `scripts/slurm/single_room_3d_train.sh` (tron, 4 CPU, 24 GB, 1 GPU) — pinned
  per D13: TITAN X (12 GB) too small for 3D activations.
- `scripts/slurm/single_room_3d_eval.sh` (scavenger, 4 CPU, 16 GB, 1 GPU)
- `scripts/slurm/single_room_3d_summary.sh` (scavenger, 2 CPU, 4 GB, no GPU)
- `scripts/run_p2_1_pipeline.sh` — orchestrator chaining all of the above
  via `--dependency=afterok`. Implements the manager's parallelization
  directive (de-risk on tron `%4`, training on scavenger wide array,
  single-room train depends only on de-risk).

### Tests (new, 6 files)
- `tests/test_ism_3d.py`
- `tests/test_eigenfrequencies_3d.py` (covers cubic degeneracy)
- `tests/test_renderer_3d.py` (ray sampling, AABB, fake-model forward)
- `tests/test_model_3d.py` (GPU-gated; forward shape, σ ≥ 0, RFFT, gradient)
- `tests/test_signal_eval.py` (identity, phase shift, noise, EDC monotonicity)
- `tests/test_lhs_sampling.py` (range, no-duplicates, reproducibility, spread)

Test count: 125 → 190 (+65 new tests; +1 minor on a 2D test cleanup).

## 3. Pipeline status

**(Cluster jobs in progress — this section is updated as jobs complete.)**

### Stage 0: pytest gate
- Job submitted to scavenger; result to be filled in.

### Stage 1: sample rooms 3D
- Writes `configs/sweeps_3d/{derisk,train,test}_rooms.yaml` deterministically
  from seed=42.

### Stage 2: budget check
- Simulates 2 rooms (smallest + largest of de-risk). Reports wall-clock + size.
- **Pass threshold**: ≤10 min/room, ≤500 MB/room.

### Stage 3a: de-risk dataset (5 rooms on tron, 4-concurrent cap)
- Idempotent via `.h5.done` sentinels.

### Stage 3b: training dataset (45 rooms on scavenger wide array)
- Runs in parallel with stage 3a + downstream training.

### Stage 4: dataset manifest refresh
- Re-derives `data/track_a_3d/manifest.json` from sentinels.

### Stage 5: GPU memory check (3D)
- Cascade per D12.

### Stage 6: single-room 3D training (5 jobs on tron)
- Pinned to tron RTX 2080 Ti (D13).

### Stage 7: single-room 3D eval + signal-level metrics
- 5 jobs on scavenger.

### Stage 8: cross-room summary
- `outputs/single_room_3d/SUMMARY.md`.

## 4. Per-de-risk-room results

**(Filled in as eval jobs complete.)**

| Room (L, W, H) | V (m³) | f_S (Hz) | n_modes < f_S | modal MAE (Hz) | full LSD (dB) | mag corr | phase corr (mw) | RIR Pearson | EDC RMS (dB) | early / late | env corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (4.5, 4.0, 3.25) box ctr | — | — | — | — | — | — | — | — | — | — | — |
| (3.0, 3.0, 2.5) | — | — | — | — | — | — | — | — | — | — | — |
| (6.0, 5.0, 4.0) | — | — | — | — | — | — | — | — | — | — | — |
| (3.0, 5.0, 2.5) | — | — | — | — | — | — | — | — | — | — | — |
| (6.0, 3.0, 4.0) | — | — | — | — | — | — | — | — | — | — | — |

Targets:
- Modal MAE ≤ 3 Hz on f<f_Schroeder for ≥4 of 5 rooms.
- Full-band LSD comparable to (within ~2-3×) 2D Phase-1 single-room baseline
  (Phase 1: 0.36-0.42 dB; expect P2-1: ≤ 1.5 dB).
- Mag corr ≥ 0.9 and RIR Pearson ≥ 0.7 (Dolby-grade reconstruction).

## 5. 3D modal density (Q14 input)

**(Filled in after sampling YAMLs land.)**

For room (4.5, 4.0, 3.25) at α=0.15:
- Count of distinct eigenfrequencies ≤ 250 Hz: TBD (expected ~30, vs ~12 for 2D)
- Count of distinct eigenfrequencies ≤ 2 kHz: TBD (expected ~15K, vs ~600 for 2D)

## 6. Budget check results

**(Filled in after stage 2.)**

| Room | L | W | H | wall (s) | size (MB) | max_order (capped?) | T60 (s) | n_modes ≤ 2 kHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smallest | 3.0 | 3.0 | 2.5 | — | — | — | — | — |
| largest | 6.0 | 5.0 | 4.0 | — | — | — | — | — |

## 7. Memory check results

**(Filled in after stage 5.)**

GPU: TBD. Chosen config (n_azi, n_ele, n_pts, batch): TBD.

## 8. 3D port verdict

**(Filled in after all single-room evals complete.)**

## 9. Signal-level eval verdict

**(Filled in after evals complete.)**

## 10. HashGrid capacity diagnosis (Q12 input)

**(Filled in after single-room evals.)**

D10 picks log2_hashmap_size=18 / n_levels=16 / per_level_scale=1.38. Empirical
diagnosis based on the 5 single-room overfits:

- If modal MAE clearly < 3 Hz on majority → capacity is right.
- If poor → try per_level_scale=1.34 (finer).
- If val LSD plateaus very early → over-provisioned, P2-2 can shrink.

## 11. Recommendations for P2-2

**(Filled in after closeout.)**

Provisional from the P2-1 design:
- P2-2 builds `INR3D_AutoDecoder` (subclass of `INR3D_Single` with FiLM
  conditioning + per-room latents + L-head equivalent for (L, W, H) regression).
- Reuse `Shoebox3DDataset` with the 45-room train + 8-room test split.
- Reuse `aaf/eval/signal_level.py` directly for zero-shot eval.

## 12. Surprises and risks

**(Filled in after closeout.)**

## 13. Pointers

- `tasks/CHUNK_3_7_RESULTS.md` — final Phase-1 chunk (D1_dense15 modal 2.55 dB).
- [DECISIONS.md](DECISIONS.md) — D1-D18 for this chunk.
- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) — Q12, Q13, Q14 newly opened.
- `outputs/single_room_3d/SUMMARY.md` — cross-room aggregate (after evals).
