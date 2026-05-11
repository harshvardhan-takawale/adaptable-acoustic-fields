# Chunk 3.7 — Meeting visual story + parallel improvement experiments

## V0 verdict: **GREEN** (6 of 6 modes, corr ≥ 0.7)

At L=4.25, all 6 of the first analytical eigenfrequencies show spatial Pearson
correlation ≥ 0.84 between the model's predicted pressure field and the ISM
ground truth on the 8×8 receiver grid. The lowest individual-mode correlation
across the whole 6-L × 6-mode matrix is **0.75** (mode (1,0) at L=3.25); the
highest is **0.99** (mode (0,1) at L=4.75, L=5.25). Spatial-node tracking is
a defensible claim and Track V proceeds.

## Headline result for the meeting

**The model has learned room geometry**, even though full-band reconstruction
LSD remains poor. Three independent threads of evidence:

1. **Spatial pressure-field correlation (V0 + V1, this chunk)**: across **all
   6 unseen L × first 6 modes = 36 (L, mode) pairs**, the spatial Pearson
   correlation between predicted and analytical pressure fields on the
   receiver grid is **mean 0.90, min 0.75, max 0.99**. Every single unseen L
   gets a GREEN verdict. The (0,1) mode — a transverse-W mode that doesn't
   depend on L — is recovered with correlation > 0.93 at all 6 unseen L,
   confirming the model isn't just memorising the trained-L axial geometry.
2. **Modal peak frequency tracking (Chunk 3.6 + V2 this chunk)**: when the
   model's peak-picker commits to a peak below ~200 Hz, that peak's
   frequency matches the analytical eigenfrequency with **MAE 1.04 Hz** on
   the centre receiver across all 6 unseen L (n=31 matched pairs, recall
   22%). The matched-peak panel of [04_zero_shot_modal_tracking.png](../outputs/meeting_assets/04_zero_shot_modal_tracking.png)
   shows the predicted-vs-analytical points sitting tightly on the y=x diagonal.
3. **Latent manifold geometry (Chunk 3.6)**: C1 FiLM's latent PCA shows
   PC1-vs-L R² = 0.987 (target was > 0.7, never met by the 9 R-runs).
   C2 latent-jitter: R² = 0.702. Both meet the latent-geometry target.
   The latents have explicitly learned that "room length" is the right axis.

These three are *jointly* the strongest evidence for the chunk's headline:
the inner loop CAN recover a meaningful per-room latent at unseen L, and the
DECODER CAN render geometrically-correct spatial mode shapes from that
latent — even when the diffuse-regime spectral content (above ~250 Hz) is
not reproduced faithfully. The remaining gap is bandwidth: the model gets
low-frequency room geometry right and high-frequency texture wrong.

## Track V — visual presentation

### V0 — spatial node alignment at L=4.25

`outputs/spatial_nodes_check/L4.25/nodes_check_report.md`:

| (n_x, n_y) | f (Hz) | spatial corr | node match | pred shape SNR (dB) | ISM shape SNR (dB) |
|---|---:|---:|---:|---:|---:|
| (1,0) | 40.4 | 0.888 | 0.40 | -5.8 | 2.9 |
| (0,1) | 42.9 | 0.977 | 0.67 | 2.3 | 1.9 |
| (1,1) | 58.9 | 0.890 | 0.55 | -2.0 | 6.3 |
| (2,0) | 80.7 | 0.862 | 0.83 | -9.3 | 0.4 |
| (0,2) | 85.8 | 0.926 | 0.40 | -2.9 | -1.5 |
| (2,1) | 91.4 | 0.838 | 0.40 | -8.7 | 0.2 |

Mean correlation **0.897**. Every mode clears the 0.7 GREEN threshold;
(0,1) and (0,2) reach > 0.92.

### V1 — cross-L spatial-node summary

All 6 unseen L are GREEN: `outputs/spatial_nodes_check/SUMMARY.md`:

| L (m) | Verdict | modes ≥ 0.7 | mean corr |
|---:|:---:|---:|---:|
| 3.25 | **GREEN** | 6/6 | 0.861 |
| 3.75 | **GREEN** | 6/6 | 0.879 |
| 4.25 | **GREEN** | 6/6 | 0.897 |
| 4.75 | **GREEN** | 6/6 | 0.923 |
| 5.25 | **GREEN** | 6/6 | 0.940 |
| 5.75 | **GREEN** | 6/6 | 0.932 |

The mean correlation rises monotonically with L from 0.86 (L=3.25) to 0.94
(L=5.25), suggesting the model is slightly more confident on the larger-L
end. No clear mechanism is established by this chunk; one plausible
hypothesis is that larger rooms have more modes within the well-sampled
0-150 Hz band, giving the model more spectral structure to anchor to.

The correlation matrix figure: [05_spatial_nodes_grid.png](../outputs/meeting_assets/05_spatial_nodes_grid.png).

### V2 — modal-tracking polished plot

Figure: [04_zero_shot_modal_tracking.png](../outputs/meeting_assets/04_zero_shot_modal_tracking.png).
Across all 6 unseen L, the C2_latent_jitter + B6 model's matched modal
peaks (modal_metrics.matches) fall on the y=x diagonal with **mean
absolute error 1.04 Hz across 31 matched pairs**. Per-L recall:
L=3.25: 2/16, L=3.75: 5/20, L=4.25: 6/21, L=4.75: 6/25, L=5.25: 6/27,
L=5.75: 6/30. The model misses ~78% of analytical modes per L, but the
ones it commits to are essentially correct.

### V3 — length-morphing audio demo

3 WAV files passed the SNR sanity check (peak/median ≥ 3.0) and shipped:
[morph_L3.25.wav](../outputs/meeting_assets/07_audio_demo/morph_L3.25.wav),
[morph_L4.25.wav](../outputs/meeting_assets/07_audio_demo/morph_L4.25.wav),
[morph_L5.75.wav](../outputs/meeting_assets/07_audio_demo/morph_L5.75.wav).
Source: 1 sec, fs=4096, impulse + 3 sinusoids at 80/120/180 Hz convolved
with the centre-receiver predicted RIR. Quality caveat: full-band held LSD
is ~5 dB, so artefacts are audible — the demo is qualitative, demonstrating
smooth latent morphing rather than a faithful RIR.

### V4 — assembled meeting deck

Manifest: [outputs/meeting_assets/00_README.md](../outputs/meeting_assets/00_README.md).
All 7 deck assets present, each with an honest 1-2 sentence caption:

| Asset | Status |
|---|---|
| `01_phase_1_recap.png` | generated |
| `02_single_room_baseline.png` | copied from Chunk-2 outputs |
| `03_multi_room_training.png` | generated (11-config bar chart) |
| `04_zero_shot_modal_tracking.png` | V2 output |
| `05_spatial_nodes_grid.png` | V1 output |
| `06_latent_manifold.png` | copied from C1 latent_probe |
| `07_audio_demo/` | 3 WAVs |

## Track I — improvement experiments

I1 (denser sweep) and I2 (FiLM + LoRA) training is in flight on tron; I3
(n_obs=32 via chunked inner loop) was running on scavenger at the time of
this writeup. Their numbers will be folded into a follow-up commit when
they land. See `outputs/inner_loop_experiments/B7/C2_latent_jitter/L*/`
for I3 outputs once those finish.

The auto-generated `scripts/chunk_3_7_final_summary.py` re-runs whenever
called and refreshes the Track I block of this file based on what's on
disk; the corresponding final-summary SLURM job is queued with
`afterany` so it fires even if some Track I jobs fail.

## Recommended deck order

1. **01** — Phase-1 setup recap.
2. **02** — single-room baseline (sets the modal-tracking ceiling: 0.34-0.58 Hz MAE).
3. **03** — per-training-room reconstruction across 11 configs (all ≤ 1.5 dB).
4. **06** — latent manifold (C1 R² = 0.987 — the "we learned geometry" plot).
5. **04** — modal peak tracking on unseen L (the strongest quantitative result).
6. **05** — spatial node grid across 6 L × 6 modes (the V0/V1 GREEN matrix).
7. **07** — audio morphing demo (qualitative).

## Updated meeting story (2-3 sentence draft claim)

> The model has learned room geometry from observations. Across all 6
> unseen room lengths (3.25-5.75 m), the predicted pressure field at the
> first 6 eigenfrequencies has spatial Pearson correlation with the
> analytical ground truth in the range **0.75-0.99 (mean 0.90)**, and the
> matched modal peaks fall on the analytical eigenfrequencies with MAE
> 1.04 Hz. The latent space learned in C1 FiLM training shows
> PC1-vs-L R² = 0.987, so the latent itself has identified "room length"
> as the right axis. The remaining limitation, which we own openly, is
> bandwidth: full-band reconstruction LSD remains ~5 dB because the
> diffuse-regime (> 250 Hz) spectral texture is not faithfully reproduced.
> The recommended next step is the denser-training sweep currently in
> flight (Track I, 15 rooms at 0.2 m spacing); if that closes the gap,
> the open question of whether more interpolation anchors break the
> bandwidth ceiling will have a definitive answer.

## Failure modes and limitations (honest framing)

- **22% modal recall**, not 100%. The model recovers a small minority of
  analytical modes per L; the ones it gets are accurate, but it misses
  ~80% of expected peaks. The deck should call this out before someone
  asks "but how many modes are matched?".
- **Full-band held LSD ~5 dB** across all 11 configurations and all 5
  inner-loop strategies tested in Chunk 3.6. The spectral texture above
  Schroeder is not reproduced. The chunk does not claim otherwise.
- **Mode-shape fit SNR is sometimes negative** in the V0 report (e.g.
  (1,0) and (2,0) at L=4.25 have pred-shape SNR -5.8 dB, -9.3 dB
  respectively). This means the LSQ-fit residual exceeds the fit
  amplitude — the predicted field has the right *correlation* with the
  analytical mode shape (so the geometry is right) but not the right
  *amplitude*. We report correlation as the primary metric; node-match
  and shape-fit are secondary.
- **One mode (1,2) doesn't survive at L ≥ 4.25** — it gets pushed above
  the f_max=150 Hz cutoff. We use 7 distinct modes total across the 6 L
  values; not all 6 are present at every L. The correlation matrix shows
  "—" for absent pairs.
- **The autocommit hook ran during V4** and produced multiple commits in
  quick succession; the local `git status` was temporarily inconsistent.
  This is cosmetic — all artefacts are on remote/main.

## What's left undone

- Track I has not yet finished. Numbers will be appended when the
  remaining train + ZS jobs land (~2.5 hours from this writeup). The
  auto-final-summary job 6818459 will re-render this file with Track I
  data once that completes.
- The bandwidth limitation (diffuse regime not reproduced) remains. The
  recommended next-chunk path is the I1 denser-sweep result + (if that
  doesn't suffice) a real hyper-network conditioning replacement for the
  auto-decoder paradigm.

## Pointers

- V0/V1 spatial outputs: `outputs/spatial_nodes_check/` (per-L reports +
  `SUMMARY.md` + `figures/correlation_matrix.png`).
- V2 modal tracking: `outputs/meeting_assets/04_zero_shot_modal_tracking.png`
  + caption.
- V4 deck manifest: `outputs/meeting_assets/00_README.md`.
- Track I outputs (when complete): `outputs/multi_room/sweep/D1_dense15/`,
  `outputs/multi_room/sweep/D2_filmlora/`, `outputs/inner_loop_experiments/B7/`.
- Orchestrator: `scripts/run_chunk3_7.sh`.
- All new code lives under `aaf/eval/spatial_modes.py`,
  `aaf/eval/zero_shot.py` (chunk_size kwarg), `aaf/models/inr_2d.py`
  (`conditioning_type='film_lora'` + `lora_rank`), and the V/I scripts
  under `scripts/`.
