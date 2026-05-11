# Chunk 3.9 — Replace `02_single_room_baseline.png`

## Done

- **New asset**: [`outputs/meeting_assets/02_single_room_baseline.png`](../outputs/meeting_assets/02_single_room_baseline.png) at 2581×1777 px (was 1198×429). Two-panel
  ISM-vs-predicted spectrum overlay at L=4.5 m, centre receiver:
  - Top: 0-250 Hz, full modal regime, log-magnitude.
  - Bottom: 0-100 Hz zoom on the first 5 modes.
  - Both panels: solid black = ISM ground truth; dashed orange = single-room
    overfit prediction (slight transparency so overlap is visible).
    Analytical eigenfrequencies marked as small ticks along the bottom.
  - Title: "Single-room baseline: predicted spectrum overlays ISM ground
    truth (L=4.5 m, modal MAE 0.34 Hz)".
- **Caption** [`outputs/meeting_assets/02_single_room_baseline_caption.md`](../outputs/meeting_assets/02_single_room_baseline_caption.md): replaced with the spec's
  verbatim text.
- **Manifest** [`outputs/meeting_assets/00_README.md`](../outputs/meeting_assets/00_README.md): re-emitted via `scripts.assemble_meeting_assets`; 02 status now `already-present (≥ 1500 px)`.
- **DECK_NARRATIVE.md** [`outputs/meeting_assets/DECK_NARRATIVE.md`](../outputs/meeting_assets/DECK_NARRATIVE.md): added the "two adaptation routes" bullet to slides 3 and 6 (per the same-commit amendment in the spec).

## Source data

- Model checkpoint: `outputs/single_room/L4.5/ckpt_iter0010000.pt`.
- ISM ground truth: `data/track_a/L_4.50m_W_4.00m_alpha_0.15.h5` (`ism/H_complex`).
- Centre receiver picked by `argmin |rx − (L/2, W/2)|` over the 8×8 grid.
- Analytical eigenfrequencies from `aaf.sim.analytical_modal_2d.eigenfrequencies_2d(L=4.5, W=4.0, c=343.0, f_max=250.0)`.

## How it was made

`scripts/make_02_single_room_baseline.py` (new). Loads the trained `INR2D_Single`
model, forwards all 64 receivers via `FreqRenderer2D` in eval mode, picks the
centre receiver, then plots two panels via matplotlib at `dpi=200`,
`figsize=(13, 9.0)`. Runs in ~30 s on a scavenger GPU (CUDA needed for tcnn).
Submitted via a one-off SLURM script; not orchestrated.

`scripts/assemble_meeting_assets.py` patched so that the 02 copy step
**skips** when a presentation-resolution version (≥ 1500 px on the long edge)
is already at the destination — mirroring the existing logic for 06. Without
this fix the assembler would clobber the new 02 with the legacy 1198×429
chunk-2 modal_tracking copy.

## What did NOT change

- No retraining.
- No other meeting asset changed (only 02 + the assembler skip-logic).
- No technical content in DECK_NARRATIVE.md changed — only two bullets
  added per the spec amendment.
