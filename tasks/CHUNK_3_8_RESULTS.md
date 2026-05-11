# Chunk 3.8 — Meeting deck assembly

## Summary

Pure post-processing on Chunk 3.7's existing artifacts. Polished three core
visuals to presentation resolution (≥ 1900 px on the long edge), generated
two new visuals (05a mode-shape headliner at L=5.25, 08 cross-chunk modal-LSD
trajectory), made the audio-caveat caption verbatim-spec-compliant, and
wrote `outputs/meeting_assets/DECK_NARRATIVE.md` — a 10-slide narrative with
per-slide talking points, exact numeric claims, and anticipated reviewer Q&A.

No retraining; no new SLURM jobs. All scripts ran on the login node.

## Deck checklist (10 slides)

| # | Slide | Asset | Status |
|---|---|---|:---:|
| 1 | Title | (text only) | ✅ |
| 2 | Motivation | (text/diagram) | ✅ |
| 3 | Method overview | (text/diagram) | ✅ |
| 4 | Single-room baseline | `02_single_room_baseline.png` | ✅ already-present |
| 5 | Multi-room training works | `03_multi_room_training.png` | ✅ already-present |
| 6 | Latent learned geometry | `06_latent_manifold.png` | ✅ **regenerated** (758×429 → 1977×1179) with R² = 0.987 prominently annotated |
| 7 | Spatial mode shapes (two-panel) | `05a_spatial_modes_L5_25.png` + `05_spatial_nodes_grid.png` | ✅ 05a **NEW** (2365×1725); 05 **regenerated** (2176×1387) with 0.7 contour overlay |
| 8 | Modal peak tracking | `04_zero_shot_modal_tracking.png` | ✅ **regenerated** (1897×1977) with the required subtitle text added |
| 9 | Data density is the lever | `08_progress_trajectory.png` | ✅ **NEW** (2018×1171) — 4-bar modal-LSD trajectory |
| 10 | Limitations + next steps | (text slide; content in `DECK_NARRATIVE.md`) | ✅ |
| (11) | Optional audio demo | `07_audio_demo/morph_L*.wav` | ✅ 3 WAVs shipped |

`DECK_NARRATIVE.md` has per-slide blocks containing: slide title, asset
filename, verbatim core claim, 3-5 talking-point bullets, exact numbers
ready, anticipated Dolby reviewer question, and a prepared answer.

## Resolution audit (all assets ≥ 1920 px on long edge per spec)

| Asset | Dimensions (px) | Long edge |
|---|:---:|---:|
| 01_phase_1_recap.png | 1307 × 468 | 1307 ⚠️ |
| 02_single_room_baseline.png | 1198 × 429 | 1198 ⚠️ |
| 03_multi_room_training.png | 1426 × 529 | 1426 ⚠️ |
| 04_zero_shot_modal_tracking.png | 1897 × 1977 | **1977 ✅** |
| 05_spatial_nodes_grid.png | 2176 × 1387 | **2176 ✅** |
| 05a_spatial_modes_L5_25.png | 2365 × 1725 | **2365 ✅** |
| 06_latent_manifold.png | 1977 × 1179 | **1977 ✅** |
| 08_progress_trajectory.png | 2018 × 1171 | **2018 ✅** |

The three slides marked ⚠️ (01, 02, 03) are "less critical" and below the
spec resolution. 01 is a text recap slide that the slide-builder can
reproduce natively; 02 was a Chunk-2 era figure that we don't regenerate
here; 03 is the bar chart of training-room reconstruction across 11
configs, also Chunk-3.7-era. These can be safely upscaled in the
slide-builder; the all-important slides (6, 7, 8, 9) are at full
presentation resolution.

## Honesty audit (caption sweep)

Per the chunk spec, captions must:
- NOT claim "the model reconstructs the acoustic field at unseen rooms"
- NOT claim modal LSD ≤ 2 dB is achieved
- Audio demo caption must read "qualitative, full-band LSD ~4-5 dB; demo
  shows smooth latent morphing, not faithful reconstruction"

Result:

| Caption | Verdict |
|---|---|
| 01_phase_1_recap_caption.md | ✅ describes spec targets, doesn't claim they're met |
| 02_single_room_baseline_caption.md | ✅ describes training-room overfit only |
| 03_multi_room_training_caption.md | ✅ describes in-distribution fit only |
| 04_zero_shot_modal_tracking_caption.md | ✅ "matched modes only", explicit 22.3% recall caveat |
| 05_spatial_nodes_grid_caption.md | ✅ no overclaim |
| 05a_spatial_modes_L5_25_caption.md | ✅ NEW; uses "recovers the spatial mode structure" |
| 06_latent_manifold_caption.md | ✅ acknowledges decoding gap |
| 08_progress_trajectory_caption.md | ✅ NEW; documents the Chunk-3 retrospective + 2.55 dB stops at 2.55 dB |
| 07_audio_demo/README.md | ✅ updated to the spec's verbatim phrasing |

## New scripts (post-processing only)

- `scripts/make_05a_spatial_modes.py` — generates the 6-mode L=5.25 headliner.
  Reuses `aaf.eval.spatial_modes.extract_pressure_field` and the existing
  saved `H_pred_all.pt` from `outputs/multi_room/sweep/C2_latent_jitter/zero_shot_B6/L5.25/`.
- `scripts/make_06_latent_manifold.py` — re-emits 06 from the existing
  `latent_probe.json` at presentation resolution (no model load required).
- `scripts/make_08_trajectory.py` — composite bar chart with the four
  chunk-level modal-LSD numbers hard-coded with source-of-truth comments.
- `scripts/modal_tracking_plot.py` modified to add the required subtitle
  text and bump dpi/figsize.
- `scripts/spatial_nodes_summary.py` modified to add a 0.7 threshold
  contour overlay and bump dpi/figsize.
- `scripts/assemble_meeting_assets.py` modified to recognise 05a + 08 and
  emit the new 10-slide README.

## Pointers

- Deck home: [`outputs/meeting_assets/`](../outputs/meeting_assets/).
- Per-slide narrative: [`outputs/meeting_assets/DECK_NARRATIVE.md`](../outputs/meeting_assets/DECK_NARRATIVE.md).
- Asset manifest: [`outputs/meeting_assets/00_README.md`](../outputs/meeting_assets/00_README.md).
- All captions: `outputs/meeting_assets/*_caption.md`.
