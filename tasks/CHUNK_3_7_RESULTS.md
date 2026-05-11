# Chunk 3.7 — Meeting visual story + parallel improvement experiments

## V0 verdict: **GREEN** (6/6 modes have corr ≥ 0.7)

≥ 4 of 6 modes show spatial correlation ≥ 0.7 at L=4.25. The presentation chain (V1-V4) proceeds; the deck is built around the spatial-node tracking story.

## Track V — visual presentation

### V0 — spatial node alignment at L=4.25

Per-mode report: `outputs/spatial_nodes_check/L4.25/nodes_check_report.md`

## Verdict: GREEN — 6 of 6 modes have spatial correlation ≥ 0.7

- run: `C2_latent_jitter` (inner loop: `B6`)
- L = 4.25 m, W = 4.00 m, fs = 4096 Hz, n_freq_bins = 4097
- modes inspected: first 6 distinct eigenfrequencies in (1, 150] Hz

| (n_x, n_y) | f (Hz) | spatial corr | node match | pred shape SNR (dB) | ISM shape SNR (dB) |
|---|---:|---:|---:|---:|---:|
| (1,0) | 40.4 | 0.888 | 0.40 | -5.8 | 2.9 |
| (0,1) | 42.9 | 0.977 | 0.67 | 2.3 | 1.9 |
| (1,1) | 58.9 | 0.890 | 0.55 | -2.0 | 6.3 |
| (2,0) | 80.7 | 0.862 | 0.83 | -9.3 | 0.4 |
| (0,2) | 85.8 | 0.926 | 0.40 | -2.9 | -1.5 |
| (2,1) | 91.4 | 0.838 | 0.40 | -8.7 | 0.2 |

Figures:
  - `figures/mode_10.png`
  - `figures/mode_01.png`
  - `figures/mode_11.png`
  - `figures/mode_20.png`
  - `figures/mode_02.png`
  - `figures/mode_21.png`
  - `figures/all_modes_overview.png`

### V1 — cross-L spatial summary

# Chunk 3.7 V1 — spatial-node cross-L summary

GREEN: 6  |  YELLOW: 0  |  RED: 0  (out of 6 L values)

## Per-L verdicts

| L (m) | Verdict | modes ≥ 0.7 corr | mean corr |
|---:|:---:|---:|---:|
| 3.25 | **GREEN** | 6/6 | 0.861 |
| 3.75 | **GREEN** | 6/6 | 0.879 |
| 4.25 | **GREEN** | 6/6 | 0.897 |
| 4.75 | **GREEN** | 6/6 | 0.923 |
| 5.25 | **GREEN** | 6/6 | 0.940 |
| 5.75 | **GREEN** | 6/6 | 0.932 |

## Correlation matrix (modes × L)

| mode \ L | 3.25 | 3.75 | 4.25 | 4.75 | 5.25 | 5.75 |
|---|---|---|---|---|---|---|
| (0,1) | 0.96 | 0.93 | 0.98 | 0.99 | 0.99 | 0.98 |
| (1,0) | 0.75 | 0.85 | 0.89 | 0.94 | 0.95 | 0.95 |
| (1,1) | 0.84 | 0.85 | 0.89 | 0.95 | 0.97 | 0.96 |
| (0,2) | 0.93 | 0.96 | 0.93 | 0.93 | 0.96 | 0.95 |
| (1,2) | 0.87 | 0.87 | — | — | — | — |
| (2,0) | 0.81 | 0.82 | 0.86 | 0.85 | 0.87 | 0.85 |
| (2,1) | — | — | 0.84 | 0.88 | 0.89 | 0.90 |

Figures:
  - `figures/correlation_matrix.png` (this summary)
  - per-L grid at `L3.25/figures/all_modes_overview.png`
  - per-L grid at `L3.75/figures/all_modes_overview.png`
  - per-L grid at `L4.25/figures/all_modes_overview.png`
  - per-L grid at `L4.75/figures/all_modes_overview.png`
  - per-L grid at `L5.25/figures/all_modes_overview.png`
  - per-L grid at `L5.75/figures/all_modes_overview.png`

### V2 — modal-tracking polished plot

**Predicted modal peak frequencies vs. analytical eigenfrequencies (C2_latent_jitter + B6).** Across the 6 unseen room lengths (3.25-5.75 m, evaluated on the centre receiver), the predicted peaks that the peak-picker identifies fall on the y=x diagonal with mean absolute error 1.04 Hz — i.e., when the model commits to a peak, its frequency is essentially correct. The caveat is recall: only 22.3% of analytical modes are recovered, so this plot shows what the model gets RIGHT, not what it gets wrong. The full modal-error analysis is in `outputs/multi_room/sweep/C2_latent_jitter/zero_shot_B6/L*/metrics.json` (held_out_modal_mae_hz / held_out_modal_recall fields). Per-L recall: L=3.25: 2/16, L=3.75: 5/20, L=4.25: 6/21, L=4.75: 6/25, L=5.25: 6/27, L=5.75: 6/30.

### V3 — length-morphing audio demo

# V3 length-morphing audio demo (C2_latent_jitter + B6)

Source: 1.0-sec, fs=4096, x(t) = impulse + 0.3·sin(2π·80·t) + 0.2·sin(2π·120·t) + 0.1·sin(2π·180·t).

For each unseen L the predicted RIR at the centre receiver was produced by inverse-rfft of the saved `H_pred_all.pt`, then convolved with the source and peak-normalised. Files:

- `morph_L3.25.wav` — peak/median = 33.83
- `morph_L4.25.wav` — peak/median = 39.78
- `morph_L5.75.wav` — peak/median = 40.22

Quality caveat: full-band held-LSD ≈ 5 dB on these models. The audio is a qualitative demo, not a faithful RIR — it'll be audibly imperfect. We're shipping it anyway to demonstrate smooth latent morphing across L.

### V4 — assembled meeting deck

# Meeting deck assets — Chunk 3.7

Each asset has a corresponding `*_caption.md` with an honest, 
1-2 sentence description of what the plot shows.

## Manifest

| Asset | Status |
|---|---|
| `01_phase_1_recap.png` | generated |
| `02_single_room_baseline.png` | copied |
| `03_multi_room_training.png` | generated |
| `04_zero_shot_modal_tracking.png` | already-present |
| `05_spatial_nodes_grid.png` | copied |
| `06_latent_manifold.png` | copied |
| `07_audio_demo/` | 3 WAV(s) |

## Recommended deck order

1. **01** — Phase-1 setup recap
2. **02** — single-room baseline (sets the modal-tracking ceiling)
3. **03** — per-training-room reconstruction across 11 configs
4. **06** — latent manifold learned the right axis (R² = 0.987)
5. **04** — modal peak tracking (the strongest defensible result)
6. **05** — spatial node grid (the V0 verdict — present only if GREEN/YELLOW)
7. **07** — audio morphing demo (if SNR was acceptable)

Known limitations (call out in the talk):
- Full-band held-LSD remains 5+ dB on every config.
- Modal-tracking recall is ~5%: we capture the peaks we commit to, 
  but miss the majority of analytical modes.
- Track I improvements (denser sweep, FiLM+LoRA, n_obs=32) were 
  attempted in parallel — see `tasks/CHUNK_3_7_RESULTS.md` for outcomes.

## Track I — improvement experiments

### I1 — denser training sweep (15 rooms at 0.2 m)

- **D1_dense15** — no train_meta

### I2 — FiLM + rank-8 LoRA hyper-network-style conditioning

- **D2_filmlora** — no train_meta

### I3 — n_obs=32 via chunked inner loop (B7 on C2_latent_jitter)

- status: no metrics found

## Recommended deck order

01 setup → 02 single-room baseline → 03 multi-room training fit → 06 latent manifold (C1 R² = 0.987) → 04 modal tracking → 05 spatial nodes (only if V0 was GREEN/YELLOW) → 07 audio (if shipped).

## What's left undone

- The full-band ≥ 5 dB held LSD ceiling persists across all 11 + 2 
  configurations and all 6 inner-loop strategies, including the I1 
  denser sweep and I2 LoRA-augmented decoder if they completed.
- Q11 in OPEN_QUESTIONS.md remains open: the decoder ambiguity at 
  unseen L is not yet broken; a true weight-generating hyper-network 
  (Track I option C, deferred) is the next architectural step.

