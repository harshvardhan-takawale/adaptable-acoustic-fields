# Chunk P2-VIZ2 — Phase-2 deck additions

**Status**: COMPLETE — 2026-06-09. CPU/plotting only — **no GPU renders, no retraining**
(the median-LOO room's full spectrum was already cached in
`outputs/known_geometry/loo_median_spectrum.npz`). All numbers traceable to disk.

## Why
The deck's generalization story (the LOO known-geometry route) rested on two aggregate
numbers + one bar chart, while the in-distribution story had a full per-room table — an
asymmetry that could let a mentor conflate the two regimes. These additions give the
generalization claim equal depth and add backup slides.

## Deliverables (in `outputs/phase2_meeting_assets/`)
- **07_median_loo_signal_panels** — median-LOO room: magnitude + phase(mw) + RIR (full
  2 s + 50 ms zoom) overlays, pred vs ISM. 512-rx means: mag 0.90 / phase 0.91 / RIR 0.92.
- **08_signal_metrics_table** — 5 de-risk rooms, full signal suite — "single-room fidelity,
  **in-distribution upper bound**" (the "Dolby asked for signal-level eval" slide).
- **09_spatial_slices** — median-LOO room, pred-vs-ISM |H| on 3 horizontal slices at the
  (1,1,0) mode (54 Hz); the render reproduces the standing-wave node.
- **10_loo_generalization_table** — 6 representative LOO rooms spanning the distribution
  (min/median/max + q1/q3/p90) + all-45 mean (0.894/0.938/2.60) — the deliberate
  **generalization** counterpart to fig 08.
- **11_coverage_anchors** — two measured anchors (sparse-45 = 0.27, LOO/training-density =
  0.89) with **no connecting line** and an explicit "unmeasured — P2-4" gap.
- **train_rooms_list.md** — the 45 training (L,W,H) triples (backup / Q&A).
- IR/dataset spec confirmed in the manifest (fs 4096, 8192 samples = 2.0 s, 4097 bins,
  max_order 12, α 0.15, 512 rx 8×8×8, 1 source — verified on disk).

## Honesty (load-bearing)
The **three regimes are never mixed**: in-distribution upper bound (fig 08) vs leave-one-out
generalization at training density (fig 10) vs zero-shot on unseen rooms (fig 11's sparse
point). Fig 10 uses an honest min/median/max spread (not the best 6). Fig 11 draws **no
curve** between the anchors and marks the gap unmeasured.

## Reproduce
`python scripts/make_p2viz_extra.py` (CPU). Manifest:
`outputs/phase2_meeting_assets/FIGURE_MANIFEST.md`.
