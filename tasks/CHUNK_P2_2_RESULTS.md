# Chunk P2-2 — Multi-room 3D conditioning + zero-shot adaptation

**Status**: COMPLETE — 2026-06-04.

> **⚠ Correction (added during P2-2.5, 2026-06-05):** M1 and M2 silently ran on
> an 11 GB RTX 2080 Ti, because the training SLURM script used a bare
> `--gres=gpu:1` (which the scheduler fills with any free card) rather than
> naming a GPU type. **The `batch=4` ceiling that drove this chunk's "per-iter
> sampling sparsity" diagnosis was therefore partly a GPU-misallocation
> artifact, not a fundamental limit** — the model was never given the 24-48 GB
> headroom it was designed to use. P2-2.5 re-runs the sampling-vs-capacity
> diagnostic on correctly-targeted A5000/A6000 GPUs (`--gres=gpu:rtxa6000:1`).
> See CLUSTER_INFO.md "GPU-type targeting" for the fix. This does not change
> that the *latent manifold* result (R² > 0.96 per axis) was real and correct.

## 1. Goal recap

Prove that cross-room adaptation generalizes from 1D length variation
(Phase 1) to full 3D (L, W, H) variation. Train `INR3D_AutoDecoder` with
FiLM + latent jitter on 45 LHS training rooms; zero-shot adapt to 8 unseen
maximin test rooms; evaluate via the signal-level suite. Acceptance target:
**magnitude correlation ≥ 0.9 in 0-500 Hz on ≥ 5/8 test rooms**.

## 2. Headline verdict — **MIXED**

| Verdict | Result |
|---|---|
| ❌ Zero-shot mag corr ≥ 0.9 on ≥ 5/8 rooms | **0/8** rooms (M1 max 0.59; M2 max 0.64) |
| ❌ In-distribution training converged | Both M1 and M2 early-stopped with final val LSD **6.16 dB (M1) / 6.69 dB (M2)** — never reached the ≤ 2.5 dB target. |
| ✅ Latent manifold encodes 3D geometry | **R²_full = L:0.991, W:0.967, H:0.974 (M1)** and L:0.997, W:0.990, H:0.996 (M2). |
| ✅ Geometry head learned (L, W, H) accurately | Per-axis MAE on training rooms = 1.1-2.6 cm (M1) and 2.6-3.5 cm (M2). |
| ❌ Per-band LSD high in zero-shot | 5-9 dB across bands for both models on every test room. |

**Top-line read**: The latent representation works — the model learned a low-
dimensional geometric manifold that precisely encodes (L, W, H). But the
**INR + renderer never got accurate enough at the per-iteration sampling
rate to make the spectral reconstruction usable**. With 45 rooms × 512
receivers = 23,040 (room, rx) pairs and `batch_size=4`, each iteration
touches only ~0.02% of the data — vs Phase 1's ~3.6% per iter. The early-
stop fired because training plateaued at 6+ dB val LSD, not because the
model converged.

## 3. M1 vs M2 head-to-head (Q12 close-out)

| Metric | M1 d=16 | M2 d=32 |
|---|---|---|
| Training wall-clock | 1:56 (24K iters; early-stop) | 1:03 (13K iters; early-stop) |
| Final val LSD | 6.16 dB | 6.69 dB |
| Geom MAE (L / W / H) | 0.013 / 0.013 / 0.011 m | 0.035 / 0.034 / 0.026 m |
| Intrinsic_dim (95% var) | 12 / 16 | 16 / 32 |
| Per-axis R²_full (L, W, H) | 0.991, 0.967, 0.974 | 0.997, 0.990, 0.996 |
| Zero-shot mag corr (best room) | 0.59 (L=3.17) | 0.64 (L=3.14) |
| Zero-shot mag corr (worst room) | 0.47 (L=5.94) | 0.47 (L=5.94) |

**Q12 verdict**: d=16 is sufficient for the latent geometry encoding — M2's
slightly higher R² doesn't translate into better zero-shot mag corr. d=32's
extra capacity early-stopped FASTER (13K vs 24K iters) because the larger
latent table reaches a plateau sooner. P2-3 inherits **d=16** as the
working choice; the binding constraint is elsewhere.

## 4. Per-test-room zero-shot results (M1, d=16)

| Run | V (m³) | f_S (Hz) | mod MAE (Hz) | LSD (dB) full | mag corr | RIR Pearson | env corr | early/late | geom err L/W/H (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L3.14_W3.08_H2.51 | 24.2 | 292 | 1.30 | 9.31 | **0.580** | 0.360 | 0.733 | 0.38 / 0.33 | 0.15 / 0.32 / 0.23 |
| L3.17_W3.00_H3.49 | 33.2 | 263 | 0.64 | 7.61 | **0.590** | 0.408 | 0.752 | 0.39 / 0.41 | 0.15 / 0.37 / 0.04 |
| L4.10_W3.01_H3.93 | 48.5 | 231 | 1.19 | 7.40 | 0.523 | 0.355 | 0.726 | 0.35 / 0.35 | 0.07 / 0.40 / 0.63 |
| L4.50_W4.00_H3.25 | 58.5 | 217 | 0.89 | 6.86 | 0.518 | 0.359 | 0.725 | 0.39 / 0.28 | 0.04 / 0.37 / 0.10 |
| L5.91_W4.17_H3.72 | 91.7 | 186 | 1.05 | 6.54 | 0.487 | 0.291 | 0.664 | 0.35 / 0.19 | 0.44 / 0.07 / 0.00 |
| L5.92_W3.06_H2.55 | 46.3 | 229 | 0.94 | 7.56 | 0.586 | 0.385 | 0.730 | 0.40 / 0.35 | 0.15 / 0.51 / 0.15 |
| L5.94_W4.93_H2.51 | 73.5 | 195 | 1.09 | 7.13 | 0.467 | 0.265 | 0.652 | 0.32 / 0.16 | 0.60 / 1.31 / 0.14 |
| L5.99_W3.96_H2.54 | 60.1 | 210 | 0.77 | 7.58 | 0.513 | 0.301 | 0.670 | 0.36 / 0.21 | 0.68 / 0.75 / 0.36 |
| **mean** | — | — | 0.98 | 7.50 | 0.533 | 0.341 | 0.706 | 0.37 / 0.29 | — |

**Modal MAE (0.64-1.30 Hz)** stays in the same range as Phase 1's 2D zero-
shot (~1 Hz). The peaks the model commits to are still well-placed in
frequency — the *amplitudes* and *phases* are what fall apart in 3D.

## 5. Comparison to Phase 1 zero-shot

| Metric | Phase 1 D1_dense15 + B1 (2D, 15→6 unseen L) | P2-2 M1 (3D, 45→8 unseen LWH) |
|---|---|---|
| Held-out full-band LSD | ~5 dB | **7.5 dB** (1.5× worse) |
| Modal LSD (0-250 Hz) | 2.55 dB | **5.51 dB** (mean) |
| Modal MAE | 1.04 Hz | 0.98 Hz |
| Mag corr (full spectrum) | not reported (Phase 1 didn't compute this) | 0.533 (mean) |

Going from 1D length-only to 3D full geometry roughly **doubles** the per-
band LSD floor. The modal MAE didn't get worse — the model still finds the
right *frequencies* — but it fails to get their *amplitudes* right.

## 6. Latent manifold probe (the bright spot)

Both M1 and M2 produce a clean low-dimensional latent manifold that linearly
encodes geometry. The 3-panel "PCA colored by axis" figure shows clear
gradients along PC1 for each of L, W, H in both models. (See
`outputs/multi_room_3d/M1_45rooms/latent_probe/figures/`.)

This **answers Phase 1's central methodological question**: yes, the
auto-decoder can encode multiple geometry dimensions jointly. The Phase-1
result "PC1 vs L R² = 0.987" generalizes to 3D as "R²_full per axis ≥ 0.96".

But the latent being right doesn't help if the *decoder* (renderer + INR)
can't turn those latents into accurate spectra.

## 7. Diagnosis — why training plateaued at 6+ dB val LSD

The most likely cause is **per-iter sampling sparsity**:
- Phase 1 multi-room: 7 rooms × 64 receivers = 448 samples; batch=16 → 3.6% per iter.
- P2-2 multi-room: 45 rooms × 512 receivers = 23,040 samples; batch=4 → 0.017% per iter.
- Per-iter coverage drops **210×**. Without compensating with more iters,
  the model just can't see enough of the data to learn it.

At 30K iters target, the model sees 30,000 × 4 = 120,000 samples — 5.2
"epochs" through the 23K-sample pool. Phase 1's 30K iters × 16 batch
= 480,000 samples → 1,071 epochs. **P2-2 trained on ~200× fewer epochs**
in effective terms.

Both runs early-stopped before reaching 30K because val LSD plateaued —
not because the model converged, but because the gradients on rooms outside
the per-iter sample bounced around without enough averaging.

## 8. What would fix it (P2-3 recommendations, ranked)

1. **Larger per-iter batch + bigger GPU**. Pin to A100/A6000 (40-48 GB) on
   tron's high-QoS, run `batch_size=32`, `n_pts_per_ray=32`. Per-iter
   coverage goes to ~0.14% (8× current) at 4× memory. Easy +1.5-2 dB.
2. **More iters at the current batch**. Bump n_iters target 30K → 100K
   and remove the early-stop entirely (or relax to 0.5% over 5K window).
   ~3× wall-clock.
3. **Bigger renderer**: n_pts=32 (was 16) and/or n_rays=512 (was 258).
   Doubles activations; only works with (1).
4. **Decoder capacity**: bump `sigma_encoder_dim` 256 → 512 or
   `log2_hashmap_size` 18 → 20. Single-room overfit was clean at the
   current capacity but multi-room may need more.
5. **Curriculum**: train on a subset first (e.g., 10 rooms) to reach 2 dB,
   then expand to 45. This is the "P2-1 single-room → P2-2 multi-room"
   trajectory writ small.

## 9. Decisions to revisit in P2-3

- **D26 n_iters=30K + early stop**: clearly too tight for 45-room training
  with batch=4. Either raise n_iters substantially or, better, fix the
  per-iter sampling first (which is the binding constraint).
- **D12 memory cascade (n_pts=16, batch=4)**: chosen for the 12 GB TITAN X.
  P2-3 should aim for tron 24 GB at `batch≥16, n_pts≥32` (the original
  P2-1 canonical config that the cascade rejected).
- **D27 zero-shot inner loop**: the inner loop was correctly implemented
  (and chunked after the OOM fix). But adapting a single latent for 2000
  iters against 8 receivers when the underlying decoder produces 6+ dB LSD
  predictions means the adapted z* can't do better than the network's
  prior. Future zero-shot work should always be on a model that already
  fits in-distribution at the target band.

## 10. Pipeline status (for the manager's reproducibility)

- **Code commits**: `85f03d7` (infrastructure), `34aebd4` (move tensors
  to CPU), `8ff7f45` (best-effort TB writes), `238d126` (chunked zero-shot
  inner loop), `<this commit>` (closeout).
- **Test count**: 209 tests pass (P2-1 + 20 P2-2 tests).
- **Cluster artifacts**:
  - `outputs/multi_room_3d/M1_45rooms/train_meta.json` + `scalars.json`
  - `outputs/multi_room_3d/M1_45rooms/zero_shot/L*_W*_H*/{metrics.json, figures/}`
  - `outputs/multi_room_3d/M1_45rooms/latent_probe/{latent_probe.json, figures/}`
  - `outputs/multi_room_3d/M1_45rooms/SUMMARY.md` + `mag_corr_per_room.png`
  - same for `M2_45rooms_d32/`.

## 11. Pointers

- [outputs/multi_room_3d/M1_45rooms/SUMMARY.md](outputs/multi_room_3d/M1_45rooms/SUMMARY.md) — M1 (d=16) per-room metrics
- [outputs/multi_room_3d/M2_45rooms_d32/SUMMARY.md](outputs/multi_room_3d/M2_45rooms_d32/SUMMARY.md) — M2 (d=32) per-room metrics
- [outputs/multi_room_3d/M1_45rooms/latent_probe/](outputs/multi_room_3d/M1_45rooms/latent_probe/) — figures showing the (L, W, H)-encoded latent manifold
- [DECISIONS.md](DECISIONS.md) — D19-D31 (this chunk's design decisions)
- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) — Q12 closes here; Q14 opens new sub-question on the band-LSD scaling

## 12. Manager actions requested

1. **Confirm P2-3 direction**: the right next step is "fix the per-iter
   sampling problem" (rec #1 above: bigger GPU + larger batch). Confirm
   this is the right priority vs other directions (e.g., curriculum
   training, conditioning-mechanism sweep).
2. **GPU budget**: tron `qos=high` would give 4 GPUs at A4000 / A5000 /
   A6000 spec — fast enough to test rec #1. Confirm this is available.
3. **Q14 update**: with the current architecture failing zero-shot mag
   corr ≥ 0.9 globally, should we adopt option (a) — modal LSD ≤ 2 dB on
   f<f_Schroeder — as the P2-3 target? It's narrower and might be
   achievable with the current model. Or option (b) — modal LSD ≤ 2 dB
   on a fixed 0-100 Hz band — which the in-distribution numbers suggest
   is closer to within reach.
