# Meeting deck assets — Chunk 3.8

Each asset has a corresponding `*_caption.md` with an honest, 
1-2 sentence description of what the plot shows. See 
`DECK_NARRATIVE.md` for per-slide talking points + anticipated Q&A.

## Manifest

| Asset | Status |
|---|---|
| `01_phase_1_recap.png` | generated |
| `02_single_room_baseline.png` | already-present (≥ 1500 px) |
| `03_multi_room_training.png` | generated |
| `04_zero_shot_modal_tracking.png` | already-present |
| `05_spatial_nodes_grid.png` | copied |
| `06_latent_manifold.png` | already-present (≥ 1500 px) |
| `05a_spatial_modes_L5_25.png` | already-present |
| `07_audio_demo/` | 3 WAV(s) |
| `08_progress_trajectory.png` | already-present |

## Recommended deck order (10 slides)

1. **Title slide** — "Adaptable Acoustic Fields: Zero-Shot Modal Generalization in 2D Shoebox Rooms"
2. **Motivation** — adaptable spatial audio; per-scene retraining is expensive; Phase-1 scope
3. **Method overview** — 2D port of INFER + auto-decoder + linear L-head + latent jitter
4. **Foundation: single-room baseline** — `02_single_room_baseline.png` (modal MAE 0.34-0.58 Hz)
5. **Multi-room training works** — `03_multi_room_training.png` (1.29-1.70 dB val LSD)
6. **Latent learned geometry** — `06_latent_manifold.png` (C1 PC1-vs-L R² = 0.987)
7. **Spatial mode shapes at unseen L** — two-panel: `05a_spatial_modes_L5_25.png` + `05_spatial_nodes_grid.png`
8. **Modal peak tracking** — `04_zero_shot_modal_tracking.png` (1.04 Hz MAE on 31 pairs)
9. **Data density is the lever** — `08_progress_trajectory.png` (3.7 → 2.55 dB across chunks)
10. **Limitations + next steps** — text slide (22% modal recall; diffuse > 250 Hz still missing; 4 ranked next steps)

Optional Slide 11: audio demo (one of the 3 morphing WAVs in `07_audio_demo/`).

## Known limitations (call out honestly in the talk)
- Full-band held LSD remains 4-5 dB on every config (modal-only is the success).
- Modal-tracking recall is ~22%: the peaks we commit to are accurate (MAE 1.04 Hz), but we miss the majority of analytical modes.
- Chunk 3.7's I1 modal LSD 2.55 dB is the project best; L=5.25 and L=5.75 are within 0.3 dB of the 2 dB target, but 0/6 unseen L are below it.
