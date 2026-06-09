# Phase-2 Meeting Deck — FIGURE MANIFEST (P2-VIZ pack, 2026-06-09)

The 6-figure meeting pack. Every number is traceable to an on-disk file (cited per row).
All PNGs are 1920×1080. Honest captions — Fig 2 says "at training density (leave-one-out),"
not "any new room"; Fig 6 is explicitly two-regime, not a continuous curve we didn't measure.

**Story arc**: 3D in-distribution is *solved* (Fig 3) → the representation *works* (Fig 1) →
known-geometry rendering *works at training density* (Fig 2) → why 3D is hard (Fig 4) →
the open problem is *ceiling-proven coverage*, not the method (Fig 5) → the path forward,
honestly (Fig 6).

| # | File (1920×1080) | What it shows | Source data | Exact numbers | One-line honest caption |
|---|---|---|---|---|---|
| 1 | `01_latent_manifold_3d.png` | 45 trained latents on top 2 PCs, colored by L/W/H; linear-probe R² per axis. Representation works. | `outputs/multi_room_3d/M1_45rooms/latent_probe/latent_probe.json` + checkpoint | R² L=0.991, W=0.967, H=0.974; PC var 25.4%/14.8% | "The latent space autonomously encodes all three room dimensions." |
| 2 | `02_known_geometry_works.png` | (a) median-LOO room predicted-vs-ISM magnitude overlay (0–2 kHz); (b) per-room LOO score distribution over 45 rooms. | `outputs/known_geometry/loo/loo_rows.json` + `loo_median_spectrum.npz` | median room L4.60/W4.35/H3.53 = 0.896; LOO mean 0.894 (full) / 0.938 (0–250 Hz) / 2.6 dB | "Known-geometry rendering: predict the latent from (L,W,H), render with no measurements — 0.89 mag corr **at training density** (leave-one-out, n=45)." |
| 3 | `03_in_distribution_solved.png` | P3 val LSD vs iteration; the P2-2 plateau and 2.5 dB target. The engineering win. | `outputs/multi_room_3d/P3_45rooms_4gpu/scalars.json` | 6.43 dB @2K → **2.169 dB @60K**; M1 plateau 6.16; target 2.5 | "Converged 3D multi-room model: 2.169 dB in-distribution (45 rooms, 4-GPU DDP)." |
| 4 | `04_modal_density_2d_vs_3d.png` | distinct modes ≤250 Hz: 2D ~12 vs 3D 135 (~11×); cumulative-mode staircase with Schroeder. | recomputed `aaf.sim.analytical_modal_3d.eigenfrequencies_3d` (4.5,4.0,3.25); `tasks/CHUNK_P2_1_RESULTS.md` | 3D = **135** distinct (0<f≤250 Hz; 136 incl. DC); 2D ~12; ratio ~11×; f_Schroeder ≈217 Hz | "3D rooms have ~11× higher modal density below Schroeder — the modal regime is the hardest band in 3D." |
| 5 | `05_coverage_diagnosis.png` | test-room mean for 8-recv / lookup-RBF / lookup-lin / on-manifold-oracle (all ~0.27) vs LOO (0.89). | `outputs/known_geometry/{p2_3_8recv_per_band,lookup,oracle_onmanifold,lookup_summary}` | 8-recv 0.25, lookup-RBF 0.27, lookup-lin 0.26, oracle 0.24 (full); LOO 0.89 | "Ruled out three ways: even the best on-manifold latent fails at 45-room coverage — the bottleneck is training density, not the method." |
| 6 | `06_the_density_lever.png` | scatter: mag corr vs distance to nearest trained room; trained (LOO) vs untrained (test) regimes. | `train_meta.json` (NN dist) + `known_geometry/{loo,lookup}` | LOO 0.82–0.95 across NN 0.14–0.74 m, corr(NN,mag) **−0.74**; test flat 0.25–0.28 at NN 0.33–0.84 m | "Quality vs distance to nearest trained room — **two regimes**: within trained geometries denser→better; untrained rooms fail flat. Denser coverage is the evidenced fix (not a continuous curve)." |

## Honesty notes
- **Fig 2** is leave-one-out over the 45 *training* rooms (held-out room excluded from the
  (L,W,H)→latent map) — "works at training density," NOT a claim about arbitrary new rooms.
  Producing panel (a) required one ~30 s GPU render of the median-LOO room (the LOO spectra
  weren't cached); everything else is CPU-only as specified.
- **Fig 4**: the honest distinct-mode count is **135** (the 136th `eigenfrequencies_3d`
  entry is the DC term at f=0); the ~11× headline is unchanged.
- **Fig 6**: presented as **two regimes**, not a single continuous curve — distance to the
  nearest trained room does *not* by itself separate the clusters (they overlap in x); the
  distinction is trained-vs-untrained geometry. P2-4 (more rooms) is the motivated fix.
- Legacy figures from the earlier pack (`02_modal_density_2d_vs_3d.png`,
  `03_diagnostic_convergence.png`, `04_representation_vs_rendering.png`,
  `05_phase2_progress.png`) remain in this directory but are **superseded** by the 6 above.

---

# P2-VIZ2 deck additions (2026-06-09) — generalization depth + backup slides

All CPU-only (the median-LOO room's full spectrum was cached in
`outputs/known_geometry/loo_median_spectrum.npz`, so figs 07 & 09 needed no GPU render).
Honest captions keep **in-distribution / leave-one-out / zero-shot** distinct.

| # | File (1920×1080) | What it shows | Source data | Exact numbers | One-line honest caption |
|---|---|---|---|---|---|
| 07 | `07_median_loo_signal_panels.png` | median-LOO room L4.60/4.35/3.53, 3 panels: (a) magnitude overlay, (b) phase overlay (mag-weighted), (c) RIR overlay full 2 s + 50 ms zoom — pred vs ISM. | `loo_median_spectrum.npz` (+ signal_level metrics) | 512-rx means: mag corr **0.90**, phase corr (mw) **0.91**, RIR Pearson **0.92** | "known-geometry render, **leave-one-out, at training density**" |
| 08 | `08_signal_metrics_table.png` | 5 de-risk rooms × {mag, phase(mw), RIR Pearson, early/late, env corr, modal MAE, LSD}. | `outputs/single_room_3d/SUMMARY.md` | mag 0.954–0.983; phase 0.954–0.981; RIR 0.965–0.987; env 0.982–0.994; LSD 1.31–1.77; modal MAE 0.61–1.18 | "single-room fidelity — **in-distribution upper bound**" (each room overfit individually) |
| 09 | `09_spatial_slices.png` | median-LOO room: pred-vs-ISM \|H\| on 3 horizontal slices (z=0.72/1.97/2.81 m) at the (1,1,0) tangential mode. | `loo_median_spectrum.npz` | f = **54 Hz** (1,1,0 mode); render reproduces the standing-wave node | "predicted vs ISM at 54 Hz — **a room that works (LOO, training density)**" |
| 10 | `10_loo_generalization_table.png` | 6 representative LOO rooms spanning the score distribution (min/median/max + q1/q3/p90) + all-45 mean. | `outputs/known_geometry/loo/loo_rows.json` | min 0.825, **median 0.896**, max 0.947 (mag full); all-45 mean **0.894 / 0.938 / 2.60 dB** | "Known-geometry rendering — **leave-one-out generalization** (predict latent from (L,W,H), no measurements, at training density)" — counterpart to fig 08 |
| 11 | `11_coverage_anchors.png` | two measured anchors (no connecting line): sparse-45 (0.27) vs LOO/training-density (0.89), with an explicit "unmeasured — P2-4" gap. | `known_geometry/{loo,lookup}` + `train_meta.json` | sparse: NN **0.61 m**, mag **0.27** (whisker 0.25–0.28); LOO: NN **0.34 m**, mag **0.89** (whisker 0.82–0.95) | "Coverage vs rendering quality: **two measured anchors** — the curve between is deliberately not drawn (P2-4 maps it)." |
| — | `train_rooms_list.md` | the exact 45 training (L,W,H) triples + volume. | `configs/sweeps_3d/train_rooms.yaml` | 45 rooms; L 3.0–5.9, W 3.1–5.0, H 2.5–4.0 (approx) | backup / Q&A |

## IR / dataset spec (verified on disk — methods backup slide)
**fs = 4096 Hz · 8192 samples (2.0 s) · 4097 freq bins (Δf ≈ 0.5 Hz) · ISM max_order = 12 ·
α = 0.15 · 512 receivers (8×8×8 grid) · 1 source.** Confirmed against
`data/track_a_3d/*.h5` attrs (`ism_H` [512,4097], `ism_rir` [512,8192]) + the configs.

## Honesty (P2-VIZ2)
- The **three regimes are never mixed**: fig 08 = single-room *in-distribution upper bound*;
  fig 10 = *leave-one-out generalization at training density*; fig 11's sparse point = the
  *zero-shot* result on unseen rooms. (Room L4.50/4.00/3.25 appears in fig 08 as a single-room
  overfit and elsewhere as the zero-shot box center — captions keep them distinct.)
- Fig 10: min/median/max + 3 spaced (**not** the best 6); mean row anchors it; mag + LSD from
  cache (loo_rows.json), phase/RIR not stored per room so omitted (no renders triggered).
- Fig 11: **no interpolating line**; the gap is explicitly marked unmeasured; whiskers are the
  real min–max score spreads.
- Fig 09 uses the median-LOO room (renders at 0.90), never a failed sparse-gap test room.
