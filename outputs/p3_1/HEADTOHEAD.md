# P3-1 — Geometry-conditioned editing in the modal band: head-to-head (data report)

Factual record of the P3-1 experiment. Numbers only; interpretation is left to the reader.
G+ training is ongoing (latest iter 10,900); G+ rows are checkpoint snapshots.

## Experiment

Three conditioning "arms" trained under one identical band-limited (0–300 Hz) protocol on the
45-room training set. Backbone (6 hash-grid encoders + 3 tcnn MLPs) and the frequency-domain
renderer are byte-identical across arms (verified by `tests/test_arm_parity.py`). Only the
conditioning path differs:
- **Arm L (latent)** — per-room learned latent (`nn.Embedding(45,16)`); zero-shot via RBF
  (L,W,H)→latent lookup.
- **Arm G (raw geometry)** — 48-d Fourier features of normalized (L,W,H) → FiLM.
- **Arm G+ (eigenstructure)** — 64-d analytic eigenfrequency vector → FiLM, plus a per-bin resonance
  map R modulating the signal output as `h·(1+w·R)`, learned scalar `w` (zero-init).

Evaluation: zero-shot (no measurements) on the frozen 15-room interior test set (identical to
P2-4/P2-4b), all metrics computed in-band 0–300 Hz.

## In-distribution convergence at eval time (val LSD, band 0–300 Hz)

| Arm | iters | in-dist val LSD |
|---|---:|---:|
| L | 40,000 | 0.72 dB |
| G | 28,000 | 1.14 dB |
| G+ (still training) | 2,000 / 6,000 / 11,000 | 4.33 / 2.72 / 2.02 dB |

G+ val-LSD trajectory (per 1K iters): 5.2, 4.3, 3.8, 3.3, 3.0, 2.7, 2.5, 2.4, 2.2, 2.1, 2.0.
A matched-convergence comparison (arms evaluated at equal in-dist val LSD) has **not** been run.

Note (2026-08-12, P3-1 paused): training advanced past the evaluated checkpoints — Arm G reached iter
60,000 (in-dist val LSD 0.59 dB) and Arm G+ reached iter 16,000 (1.61 dB) before work was paused. The
zero-shot metrics in the table below were **not** refreshed on these newer checkpoints: G's row is the
evaluated **iter-28,000** (1.14 dB) checkpoint and G+'s rows are iter 2,000/6,000/11,000.

## Zero-shot metrics (frozen 15 rooms, band 0–300 Hz)

| Arm / checkpoint | in-dist LSD | recall@250 | recall@300 | MAE@300 (Hz) | mode-shape | phase(mw) | RIR | env | band-LSD | mag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L (latent) | 0.72 | 0.104 | 0.075 | 0.86 | 0.874 | 0.607 | 0.751 | 0.898 | 5.63 | 0.810 |
| G (geom_fourier) | 1.14 | 0.101 | 0.075 | 0.83 | 0.874 | 0.616 | 0.743 | 0.897 | 5.57 | 0.802 |
| G+ (eigen) @2000 | 4.33 | 0.164 | 0.129 | 0.61 | 0.770 | 0.575 | 0.674 | 0.867 | 5.69 | 0.740 |
| G+ (eigen) @6000 | 2.72 | 0.114 | 0.084 | 0.73 | 0.700 | 0.545 | 0.632 | 0.865 | 5.88 | 0.711 |
| G+ (eigen) @11000 | 2.02 | 0.089 | 0.069 | 0.75 | 0.647 | 0.507 | 0.594 | 0.862 | 6.03 | 0.687 |

Metric definitions: **recall@F** = fraction of analytic eigenfrequencies ≤ F Hz matched by a picked
spectral peak (peak-pick prominence 3 dB, min-distance 2 Hz; match tolerance max(4 Hz, 2%),
Phase-2 settings); **MAE** = mean |Δf| over matched peaks; **mode-shape** = magnitude-of-complex
Pearson of the predicted vs ISM field at the first 6 analytic mode bins across 512 receivers;
**phase** = magnitude-weighted phase correlation; **RIR/env** = band-limited (>300 Hz zeroed on both
sides) impulse-response Pearson / envelope correlation; **band-LSD** = 0–300 Hz log-spectral
distance; **mag** = in-band magnitude correlation.

Per-room check (G+ @2000 vs @6000): recall@250 lower at 6000 for 14 of 15 rooms.

Learned resonance weight w by checkpoint: 0.03 (1K), 0.14 (2K), 0.35 (4K), 0.40 (5K), 0.41 (6K),
0.41 (7K), 0.40 (8K), 0.38 (9K), 0.36 (10K), 0.34 (11K).

## Edit-sweep mode tracking

Predicted center-receiver peak vs analytic axial-L mode c·n/2L across an 18-point unseen L-sweep
(W=4.0, H=3.25 fixed). MAE in Hz over matched sweep points; recall = fraction of sweep points where
the mode is picked and matched. G+ here is the ~6K-iter checkpoint.

| Arm | mode 1 MAE / recall | mode 2 MAE / recall | mode 3 MAE / recall |
|---|---|---|---|
| L | 1.46 Hz / 0.39 | 1.61 Hz / 0.72 | 2.69 Hz / 0.78 |
| G | 1.40 Hz / 0.72 | 1.43 Hz / 0.83 | 3.05 Hz / 0.78 |
| G+ | 1.98 Hz / 0.78 | 2.15 Hz / 0.50 | 2.81 Hz / 0.78 |

Figures: `outputs/p3_1/edits/{arm_L,arm_G,arm_Gplus}/{waterfall,tracked_peaks}.png`.

## Operational notes (for the record)
- Shared 500 GB project disk quota reached during the run (512 MB/checkpoint); checkpoint retention
  reduced. Arm L's intermediate (under-converged) checkpoints were removed; only 38K/40K remain.
- Scavenger 4-GPU DDP required `NCCL_P2P_DISABLE`/`NCCL_IB_DISABLE` on some nodes to avoid an
  init-time collective hang; this reduced G+ throughput to ~500–650 iters/hr.
- tron/qos=high is capped at 4 GPUs/user, so the three arms could not run concurrently there; L ran
  on tron, G and G+ on scavenger.

## Status / not yet run
- G+ training ongoing (iter 10,900).
- Matched-convergence comparison (arms at equal in-dist val LSD): not run (requires dense L
  checkpoints, which were removed; and G+ has not reached L's/G's convergence).
- Disentanglement eval (invariant vs moving modes under single-axis edits): not run.

Sources: `outputs/p3_1/eval/{arm_L_best,arm_G_best,arm_Gplus_iter2000,arm_Gplus_iter6000,arm_Gplus_iter11000}/summary.json`,
`outputs/p3_1/edits/*/edit_sweep_summary.json`, `outputs/p3_1/arm_*/scalars.json`.
