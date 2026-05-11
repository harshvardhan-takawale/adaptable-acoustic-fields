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
