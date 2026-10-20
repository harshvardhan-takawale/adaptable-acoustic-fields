# P4-8 meeting deck inputs (meeting_v2) — FIGURE MANIFEST

**Models.**
- **Meeting model:** R0 @120K.
- **Comparison columns:** C250 @60K ("start of phase") and B5 @60K ("last meeting").

**Scope.**
- **S2 is not in any figure** (manager item 6). Its 120K row goes into [primary_table.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/primary_table.md) only (see (6)).
- **No modal-prior or solver results appear anywhere in these outputs.** Every number is a trained network scored against FDTD.
- The primary table's "modal RIR" column is the network's own 20–300 Hz band-limited RIR Pearson (`p4_5_risk_metrics.py`), not a modal-prior model.

**Measurement.**
- Every accuracy number is **off-lattice** (Rule 11): receivers not on the training lattice 0.12 + 0.06k m.
- Metric functions are the frozen `_pearson` / `_lsd` / `_db` (`p4_2_stage2_eval`).
- Bootstrap: 10k, seed 20261013.

**Scripts.**
- [scripts/p4_8_meeting_v2.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_8_meeting_v2.py): `--fields`, `--speckle`, `--hp`.
- [scripts/p4_8_curve_export.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_8_curve_export.py).
- [scripts/p4_8_primary_table.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_8_primary_table.py).

## Files

| file | content | room / rule |
|---|---|---|
| `fields_a_shape1000.png` | S1-layout field maps (see (4)) | shape id 1000 = 15-shape test index 0 (`L_L5.54_W5.28_d1.42_w1.52`), last meeting's S1_a room |
| `fields_b_U.png` | S1-layout field maps | the U room at **R0 @120K's lower median of 25–100 Hz off-lattice r within the frozen 8-room U set**: rank 4 of 8 ascending, r = 0.378; unseen-manifest index 7 (`UL6.36_W5.44_NEd2.18w2.16x0.00_NWd1.72w1.42x0.00`) |
| `fields_c_C3.png` | S1-layout field maps | the C3 room by the same rule within the frozen 8-room C3 set: rank 4 of 8 ascending, r = 0.372; unseen-manifest index 8 (`C3L6.74_W4.58_NWd0.08w2.34_NEd1.78w2.44_SEd1.38w0.50`) |
| `speckle_2cm_280Hz.png` | speckle close-up at 280 Hz (see (5)) | shape 1000, 2 cm FDTD grid; inset origin (0.12, 0.46) m |
| `speckle_2cm_200Hz.png` | speckle close-up at 200 Hz | shape 1000, 2 cm FDTD grid; inset origin (0.12, 0.36) m |
| `FREQ_CURVES.json` | per-bin off-lattice r curves (see (3)) | C250 @60K, B5 @60K, R0 @120K |
| `speckle_2cm.json` | inset origins, per-panel 2 cm d1 and r, high-pass check on R0 projected | — |
| `fine_scale_energy.json` | mask-aware fine-scale energy ratio over the 15 shapes (see (5)) | — |
| `manifest_rows_*.json` | machine-readable rows of this table | — |

All PNGs are 2560×1440 on a white background.

**Rule for (b) and (c), lower median:**
- Sort the 8 rooms of the family by R0 @120K's off-lattice r at 25–100 Hz, ascending.
- Take index (8 − 1) // 2 = 3, the 4th of 8.
- Room (a) is fixed by the manager.

## (2) C250 @60K reference row in `primary_table.md`

Same protocol as every P4-8 row:
- in-family = 15 frozen test shapes, dense receivers;
- unseen = the frozen 16 rooms (manifest sha `ba34b17c…`);
- d1 = the 200–300 Hz speckle neighbour-difference ratio, prediction / FDTD (1.0 = FDTD's roughness), raw and with the render-time band-limit.

| model | HF | test r 25–100 / 100–200 / 200–300 | test LSD dB (3 bands) | U (3 bands) | C3 (3 bands) | d1 raw / proj |
|---|---|---|---|---|---|---|
| **C250 @60K** | **0.118** | 0.512 / 0.149 / 0.118 | 4.70 / 7.03 / 6.98 | 0.282 / 0.153 / 0.119 | 0.195 / 0.117 / 0.112 | 2.113 / 0.490 |
| B5 @60K | 0.467 | 0.887 / 0.631 / 0.467 | 2.00 / 4.24 / 5.00 | 0.473 / 0.360 / 0.355 | 0.337 / 0.261 / 0.213 | 0.968 / -- |
| R0 @120K | 0.582 | 0.918 / 0.701 / 0.582 | 1.72 / 3.83 / 4.25 | 0.481 / 0.404 / 0.410 | 0.454 / 0.356 / 0.332 | 0.945 / 0.912 |

- C250's UF (16 rooms) is 0.238 / 0.135 / 0.116.
- C250 has no fit, gap, RIR, T* or DN* column. Those numbers were not requested and print `--`.
- **C250's 200–300 Hz value is low in every room set.** Its per-bin curve has a narrow peak at 200 Hz (mark 0.309 in family, 0.329 unseen), but it sits at 0.04–0.15 from 225 Hz to 300 Hz.
- Its in-family 25 Hz mark is 0.950 (B5 0.994, R0 0.997).
- C250 is a fixed-source model, as the unseen set is, so it was scored on the same caches with no adaptation.

## (3) `FREQ_CURVES.json` — per-bin off-lattice curves

**Structure** (as `outputs/p4_5/FREQ_CURVE_60K.json`):
- `rows[model][set]` = {"25": [mean, lo95, hi95], …, "300": […], "lsd_200_300": [mean, lo, hi], "n_rooms": n}.
- `rows[model][set + "_per_bin_off"]` = 601 values: bins 0–600 (0–300 Hz, 0.5 Hz), the mean over rooms of per-room off-lattice dB r, unsmoothed, NaN → null.

**Marks and CIs:**
- Marks every 25 Hz, each the mean over ±2.5 Hz (11 bins; the 300 Hz mark is one-sided, bins 590–600).
- 95% CIs come from a 10k bootstrap over rooms.

**Sets:**
- `G15`: the 15 in-family test shapes.
- `UNSEEN16`: the frozen unseen set.
- `U8`: unseen-manifest indices 0–7.
- `C3_8`: unseen-manifest indices 8–15. The split is asserted against the manifest.

**Not exported:** the A2 3-shape protocol of FREQ_CURVE_60K.json (not requested).

**Consistency check:** each `G15` `lsd_200_300` equals the primary table's 200–300 Hz test LSD (6.98 / 5.00 / 4.25). R0's G15 marks at 200–300 Hz average 0.575, against HF 0.582 (the marks are ±2.5 Hz windows, HF is the full band).

Mark values at 50 / 100 / 200 / 250 Hz:

| model | G15 | UNSEEN16 | U8 | C3_8 |
|---|---|---|---|---|
| C250 @60K | 0.633 / 0.220 / 0.309 / 0.093 | 0.183 / 0.122 / 0.329 / 0.082 | 0.293 / 0.136 / 0.335 / 0.104 | 0.073 / 0.108 / 0.323 / 0.060 |
| B5 @60K | 0.930 / 0.755 / 0.522 / 0.450 | 0.296 / 0.262 / 0.369 / 0.253 | 0.368 / 0.324 / 0.465 / 0.327 | 0.224 / 0.199 / 0.274 / 0.179 |
| R0 @120K | 0.962 / 0.824 / 0.649 / 0.557 | 0.374 / 0.349 / 0.492 / 0.363 | 0.367 / 0.366 / 0.498 / 0.403 | 0.380 / 0.332 / 0.486 / 0.322 |

## (4) Field maps (`fields_*.png`)

**Layout (as P4-5's S1):**
- **Rows:** 50 / 100 / 200 / 280 Hz, each a single 0.5 Hz bin.
- **Columns:** "Ground truth (FDTD)" | "Start of phase (C250)" | "Last meeting (B5)" | "Now (R0)".
- **Panels:** viridis dB |H| of the RAW network output (no projection). Each panel has the room outline and a red star at the source.

**Colour limits and corner numbers:**
- Colour limits are shared per row: FDTD's 2nd–98th percentile over the room.
- The corner text is r and LSD over the room's off-lattice receivers at that bin, computed with the frozen metric functions.

**Data:**
- (a) uses the 15-shape dense test caches (`outputs/p4_5/gate2_dense/cache/<model>__track_p4_5_testdense.npz`).
- (b) and (c) use the unseen-set caches (`outputs/p4_8/cache/unseen_<model>.npz`).

## (5) Speckle close-up (`speckle_2cm_*.png`) and fine-scale energy

**Grid and columns:**
- Shape 1000's 2 cm FDTD grid: 61,874 in-room nodes.
- The FDTD panel is exact against the stored receivers (relative error 2e-10).
- Columns: FDTD | B5 raw | R0 raw | R0 + render-time band-limit. The band-limit is a least-squares projection onto |k| ≤ 1.5 k0 on the 2 cm grid, `project_lowpass`, 1000 iterations.
- Top row: full room, with the inset outlined. Bottom row: the 1.5 m × 1.5 m inset, on the same colour scale.

**Inset rule:**
- Among all 1.5 m windows inside the room's bounding box, take the one with the largest Σ |hp(B5 − FDTD)|² over in-room nodes at that bin.
- hp is the field minus its least-squares low-pass at 1.5 k0.
- The search uses 2D cumulative box sums and is run per frequency.
- **Result:** origin (0.12, 0.46) m at 280 Hz and (0.12, 0.36) m at 200 Hz. **At both frequencies the chosen window contains the source (0.5, 1.6).** B5's largest fine-scale error is near the source, so the inset shows near-source behaviour, not a typical patch of the room.

**Per-panel numbers:**
- 2 cm d1 is `p4_5_speckle_2cm.pair_ratio` over all in-room 4-neighbour pairs, prediction vs FDTD (1.0 = FDTD's roughness).
- r is the dB Pearson over the 2 cm nodes; on the inset row both are computed over the window.

| | 280 Hz full | 280 Hz inset | 200 Hz full | 200 Hz inset |
|---|---|---|---|---|
| B5 raw | d1 1.270, r 0.483 | d1 0.819, r 0.352 | d1 1.448, r 0.467 | d1 1.994, r 0.523 |
| R0 raw | d1 1.225, r 0.636 | d1 0.743, r 0.722 | d1 1.414, r 0.525 | d1 1.477, r 0.740 |
| R0 + band-limit | d1 0.927, r 0.638 | d1 0.665, r 0.716 | d1 1.031, r 0.523 | d1 1.129, r 0.719 |

**Checks on the speckle numbers:**
- B5's full-room values reproduce last meeting's d6 exactly (1.270 / 0.483 at 280 Hz, 1.448 / 0.467 at 200 Hz).
- After the band-limit, the high-pass energy left in R0's field is 2.2e-5 (280 Hz) and 1.0e-4 (200 Hz) of FDTD's.

**Mask-aware fine-scale energy ratio**, prediction / FDTD (D152(c)):
- Computed on the 64×64 lattice over off-lattice nodes as |F − LP(F)|², summed over the band's bins.
- Taken per shape over the 15 shapes, then mean [95% CI] from a shape bootstrap.
- 1.0 means FDTD's fine-scale energy.

| | LS (D152 primary) 25–100 | 100–200 | 200–300 | NC (companion) 25–100 | 100–200 | 200–300 |
|---|---|---|---|---|---|---|
| B5 raw | 1.773 [1.552, 2.072] | 2.355 [2.074, 2.752] | 2.546 [2.229, 2.978] | 0.997 [0.984, 1.008] | 0.893 [0.843, 0.954] | 0.793 [0.766, 0.821] |
| R0 raw | 1.334 [1.211, 1.506] | 1.743 [1.471, 2.122] | 1.772 [1.550, 2.027] | 0.987 [0.975, 0.999] | 0.777 [0.739, 0.815] | 0.785 [0.761, 0.807] |
| R0 projected | 0.008 [0.006, 0.011]* | 0.018 [0.015, 0.023]* | 0.017 [0.015, 0.019]* | 0.962 [0.949, 0.973] | 0.704 [0.676, 0.733] | 0.698 [0.675, 0.717] |

\* **R0 projected is ≈ 0 under the LS operator by construction.** The render-time band-limit removes exactly the band that LS high-pass measures, so this column shows the projection works, not that R0's fine-scale content is accurate.

**The NC operator** (Gaussian normalised convolution, half power at 1.5 k0) is the non-trivial companion:
- It does not share the projection's cut-off, so it still sees structure in the projected field.
- Under NC, R0 projected falls to 0.70 of FDTD at 100–300 Hz, which is smoother than FDTD.

**Sources:**
- The B5 and R0 raw rows come from `outputs/p4_5/speckle/{B5_60K,R0_120K}.json` (same code).
- The R0 projected row is computed here against FDTD's cached low-pass.

## (6) S2 @120K

S2 is at 88K of 120K (job 7889758, scavenger, about 0.63 s/it).

When its 120K checkpoint is scored, [scripts/p4_8_s2_row.sh](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_8_s2_row.sh) does the following:
- reruns `p4_8_primary_table.py`;
- adds the S2_120K row with its pre-registered verdict (UF 25–100 ≥ GX@60K + 0.05 = 0.510 AND HF ≥ 0.484);
- commits and pushes `primary_table.*` only.

S2 is not added to any figure.
