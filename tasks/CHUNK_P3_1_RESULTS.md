# Chunk P3-1 — Geometry-conditioned editing in the modal band — RESULTS (data report)

Factual record. Numbers and methods only; interpretation left to the reader. Status: framework and
evaluation implemented and tested; three arms trained (G+ ongoing); zero-shot metrics recorded on the
frozen test set; matched-convergence comparison and disentanglement eval not yet run.

## 1. Objective

Train three conditioning mechanisms under one identical band-limited (0–300 Hz) protocol and record
their zero-shot behavior on unseen geometry, focusing on modal placement (do predicted spectral
peaks land at the analytic eigenfrequencies of the room). Band 0–300 Hz covers the modal regime for
the room family (Schroeder frequency ≈173–298 Hz across the box). Motivation from Phase 2: P2-4/P2-4b
found generalization is coverage-bound and that coverage did not change modal placement (L@45 and
L@250 both at recall ≈0.09–0.13).

## 2. The three arms

Backbone (6 tcnn hash-grid encoders + sigma/signal tcnn MLPs + `FreqRenderer3D`) and the renderer are
byte-identical across arms (verified by `tests/test_arm_parity.py`: equal backbone param counts,
FiLM-identity at init, w=0 no-op). Only the conditioning path differs.

| Arm | `cond_source` | conditioning | zero-shot route |
|---|---|---|---|
| L | `latent` | `nn.Embedding(45,16)` per-room latent; latent jitter 0.1; linear geometry head | RBF (L,W,H)→latent lookup (P2-3.5) |
| G | `geom_fourier` | 48-d Fourier features of g=((L−3)/3,(W−3)/2,(H−2.5)/1.5) → FiLM | direct feature compute |
| G+ | `eigen` | 64-d sorted analytic eigenfrequencies /300 → FiLM; per-bin resonance map R (Gaussian bumps σ=2 Hz at eigenfreqs ≤310 Hz, max-normalized) modulating the signal output `h·(1+w·R)`, learned scalar w (zero-init) | direct feature compute + resonance map |

## 3. Protocol

- **Band-limited loss/val**: the 4-term frequency loss (spec_real, spec_imag, log_amp, phase; weights
  1,1,1,0.1) and val LSD computed over bins 0–600 (f ≤ 300 Hz) only; out-of-band bins receive exactly
  zero gradient (`tests/test_band_mask.py`).
- **Recipe** (identical across arms; from the P2-3 recipe): hashgrid 18/16/1.38, FiLM conditioning,
  effective batch 64 (4×16, grad_accum 2), n_pts 32, lr_network 2e-4, up to 60K iters, early-stop
  0.3%/10K (warmup 10K). Per-arm differences: L has latent LR 1e-3, jitter 0.1, geometry head,
  lambda_latent 1e-4; G/G+ have none of these (no latent table).
- **Test set**: frozen 15-room interior set `configs/sweeps_3d/test_rooms_interior_frozen.yaml`
  (byte-identical to P2-4/P2-4b). No measurements, no per-room optimization for any arm.

## 4. What was implemented

- `aaf/models/conditioning.py` (new): `fourier_features` [48], `eigen_features` [64], `resonance_map`
  [601], `build_cond_vector` dispatch. tcnn-free.
- `aaf/models/inr_3d.py`: `cond_source`/`cond_dim` params (FiLM input width; backbone unchanged),
  gated latent table, learnable `w` + per-room resonance buffer + `set_resonance`, resonance
  modulation at the signal output.
- `aaf/train/multi_room_3d.py`: band-mask slicing in loss + val; arm dispatch; conditional optimizer
  groups; per-room resonance cache; `cond_source`/`cond_dim`/`band_max_hz` config plumbing. Phase-2
  configs unaffected (defaults reproduce prior behavior).
- `aaf/eval/zero_shot_3d.py::_load_trained_model`: forwards `cond_source`/`cond_dim`.
- `aaf/eval/p3_1_eval.py` (new): band-limited zero-shot suite — modal placement (recall/precision/MAE
  at f_max 250 and 300), 3D mode-shape correlation, band-limited RIR/env, in-band phase/mag/LSD.
- `scripts/p3_1_edit_sweep.py` (new): edit-sweep waterfall + tracked-peak driver.
- Configs `configs/sweep_3d/{arm_L,arm_G,arm_Gplus}.yaml`; six tests, 12/12 passing on GPU before
  training (band mask, eigenfeatures, arm parity + the CPU suite).

## 5. Training

- Arm L: 40,000 iters, final band-limited val LSD 0.72 dB (trajectory 6.16→0.72).
- Arm G: zero-shot evaluated at iter 28,000, val LSD 1.14 dB; training later continued to iter 54,000
  (val LSD 0.59 dB). The §6 zero-shot metrics for G are from the iter-28,000 checkpoint.
- Arm G+: ongoing, iter 10,900, val LSD 2.02 dB. Trajectory per 1K: 5.2, 4.3, 3.8, 3.3, 3.0, 2.7,
  2.5, 2.4, 2.2, 2.1, 2.0.
- Learned resonance weight w over G+ training: 0.03 (1K), 0.14 (2K), 0.35 (4K), 0.41 (6K, peak),
  0.34 (11K).

## 6. Zero-shot metrics (frozen 15 rooms, band 0–300 Hz)

| Arm / checkpoint | in-dist LSD | recall@250 | recall@300 | MAE@300 (Hz) | mode-shape | phase | RIR | env | band-LSD | mag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L (latent) | 0.72 | 0.104 | 0.075 | 0.86 | 0.874 | 0.607 | 0.751 | 0.898 | 5.63 | 0.810 |
| G (geom_fourier) | 1.14 | 0.101 | 0.075 | 0.83 | 0.874 | 0.616 | 0.743 | 0.897 | 5.57 | 0.802 |
| G+ (eigen) @2000 | 4.33 | 0.164 | 0.129 | 0.61 | 0.770 | 0.575 | 0.674 | 0.867 | 5.69 | 0.740 |
| G+ (eigen) @6000 | 2.72 | 0.114 | 0.084 | 0.73 | 0.700 | 0.545 | 0.632 | 0.865 | 5.88 | 0.711 |
| G+ (eigen) @11000 | 2.02 | 0.089 | 0.069 | 0.75 | 0.647 | 0.507 | 0.594 | 0.862 | 6.03 | 0.687 |

- Per-room (G+ @2000 vs @6000): recall@250 lower at 6000 for 14 of 15 rooms.
- The arms are at different in-distribution convergence (L 0.72 dB, G 1.14 dB, G+ 2.02–4.33 dB). A
  matched-convergence comparison (equal in-dist val LSD) has not been run.
- Metric definitions: recall@F = fraction of analytic eigenfrequencies ≤F Hz matched by a picked peak
  (prominence 3 dB, min-distance 2 Hz; tolerance max(4 Hz, 2%)); MAE = mean |Δf| over matched peaks;
  mode-shape = |complex Pearson| of predicted vs ISM field at first-6 mode bins over 512 receivers;
  band-limited RIR/env = >300 Hz zeroed on both sides then IRFFT + Pearson/envelope; mag = in-band
  magnitude correlation.

## 7. Edit-sweep mode tracking

Center-receiver predicted peak vs analytic axial-L mode c·n/2L across an 18-point unseen L-sweep
(W=4.0, H=3.25). G+ = ~6K-iter checkpoint.

| Arm | mode 1 (MAE Hz / recall) | mode 2 | mode 3 |
|---|---|---|---|
| L | 1.46 / 0.39 | 1.61 / 0.72 | 2.69 / 0.78 |
| G | 1.40 / 0.72 | 1.43 / 0.83 | 3.05 / 0.78 |
| G+ | 1.98 / 0.78 | 2.15 / 0.50 | 2.81 / 0.78 |

Waterfall + tracked-peak PNGs: `outputs/p3_1/edits/{arm_L,arm_G,arm_Gplus}/`.

## 8. Not yet run
- G+ training to convergence (ongoing).
- Matched-convergence comparison (arms at equal in-dist val LSD).
- Disentanglement eval (invariant vs moving modes under single-axis L/W/H edits).

## 9. Operational notes
- Shared 500 GB project disk quota reached mid-run (512 MB/checkpoint); checkpoint retention reduced.
  Arm L intermediate checkpoints removed (only 38K/40K remain).
- Scavenger 4-GPU DDP needed `NCCL_P2P_DISABLE`/`NCCL_IB_DISABLE` on some nodes to avoid an init-time
  collective hang (throughput ~500–650 iters/hr on those nodes).
- tron/qos=high capped at 4 GPUs/user; arms could not run concurrently there (L on tron, G/G+ on
  scavenger).

Deliverables: `outputs/p3_1/HEADTOHEAD.md`, `outputs/p3_1/eval/*/summary.json`,
`outputs/p3_1/edits/*/`, `outputs/p3_1/arm_*/scalars.json`, three arm configs, six tests.
