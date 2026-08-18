# Arm C demo pack — figure manifest

Every number below is read from `outputs/armC_demo/metrics.json`, which was written by
`scripts/armC_demo_metrics.py` **before any figure existed**. Figures are drawn by
`scripts/armC_demo_figures.py` from the 12 cached `.npz` dumps; no figure recomputes a metric.

## Provenance

| item | value |
|---|---|
| checkpoint | `outputs/p3_2/p3_2b_C_cont_mlinear/ckpt_iter0060000.pt` (iter 60000) |
| conditioning | `m_linear`, 60-d, no latent table |
| arm | P3-2b Arm C — continuous-α ISM corpus, 960 configs |
| geometries | frozen test set `configs/sweeps_2d_mat/p3_2_test_frozen.yaml` |
| GT simulator | `aaf.sim.ism_2d.simulate_room_2d`, fs 4096, n 8192, max_order 60 |
| source | (0.5, 0.5) m, all scenarios |
| receivers | **64 × 64 = 4096**, 0.15 m margin — training used **8 × 8** at 0.30 m margin |
| band | 0 – 300 Hz (601 bins, df = 0.5 Hz) |
| cached fields | `outputs/armC_demo/fields/*.npz`, 12 files, 440.1 MB, complex64 |
| metrics | `outputs/armC_demo/metrics.json` |
| figure numbers | `outputs/armC_demo/figures.json` |

One checkpoint produced all 12 panels. One forward pass per scenario. No measurements of these
rooms were used, and nothing was fitted per room.

## The abort rule

Fixed before the run: **if spatial Pearson fell below 0.70, stop and report instead of
illustrating.** `armC_demo_metrics.py` exits non-zero in that case and `armC_demo_figures.py`
refuses to draw unless `proceed_to_figures` is true.

Result: **worst 0.920, mean 0.951** over 12 scenarios → passed, figures drawn.

## Figures

### `figA_spatial_fields.png` — 4132 × 1908 (headline)

Median geometry 5.93 × 3.18 m, mode **(1,0) at 28.92 Hz** (bin 58). Four scenarios ×
{predicted, ISM GT}, `|H|` in dB on a **shared colour scale across all eight panels**:
vmin −22.36 dB, vmax +17.64 dB (99.5th percentile of the pooled data, 40 dB span).

Per-panel spatial Pearson at this mode: (a) 0.991, (b) 0.984, (c) 0.991, (d) 0.984.

### `figB_signals.png` — 3820 × 1683

Same geometry, centre receiver (index 2079). Top row: `|H|` in dB, 0–300 Hz, predicted vs GT.
Bottom row: band-limited RIR, `irfft` of the same band-limited prediction, 50 ms zoom — no
time-domain fitting anywhere.

Visible in the top row and stated rather than cropped: agreement is close to **~170 Hz**, above
which the prediction tracks the envelope but misplaces individual nulls. The band LSD numbers
below already include that region.

### `figC_spatial_pearson.png` — 3820 × 1688

All 12 scenarios, spatial Pearson beside magnitude correlation, against the 0.70 abort line and
the FDTD-corpus model's 0.24–0.60 band on this same metric.

## Full metric table

Spatial Pearson is pointwise pred-vs-GT over the 4096 receivers in dB, averaged over the 3
lowest non-trivial modes (per-mode values in the "mode-wise" column). LSD-c is at the centre
receiver; LSD-all pools all 4096.

| geometry | L×W (m) | scenario | spatial R | mode-wise | mag R | LSD-c dB | LSD-all dB | phase R | RIR R |
|---|---|---|---|---|---|---|---|---|---|
| small | 3.44×3.14 | a_baseline | 0.923 | 0.974 / 0.921 / 0.875 | 0.625 | 2.11 | 2.69 | 0.860 | 0.959 |
| small | 3.44×3.14 | b_east_curtain | 0.934 | 0.984 / 0.904 / 0.915 | 0.622 | 1.66 | 2.36 | 0.885 | 0.975 |
| small | 3.44×3.14 | **c_north_absorber** ★ | **0.976** | 0.987 / 0.984 / 0.957 | 0.625 | 1.23 | 1.84 | 0.926 | 0.986 |
| small | 3.44×3.14 | d_two_wall | 0.955 | 0.984 / 0.956 / 0.925 | 0.607 | 1.55 | 2.22 | 0.882 | 0.985 |
| median | 5.93×3.18 | a_baseline | 0.957 | 0.991 / 0.960 / 0.921 | 0.714 | 2.71 | 2.81 | 0.778 | 0.921 |
| median | 5.93×3.18 | b_east_curtain | 0.961 | 0.984 / 0.951 / 0.946 | 0.709 | 2.67 | 2.58 | 0.808 | 0.943 |
| median | 5.93×3.18 | **c_north_absorber** ★ | **0.973** | 0.991 / 0.948 / 0.981 | 0.711 | 2.09 | 2.14 | 0.847 | 0.962 |
| median | 5.93×3.18 | d_two_wall | 0.960 | 0.984 / 0.920 / 0.976 | 0.694 | 2.56 | 2.54 | 0.782 | 0.958 |
| large | 5.56×4.90 | a_baseline | 0.920 | 0.964 / 0.891 / 0.903 | 0.758 | 2.80 | 3.43 | 0.758 | 0.898 |
| large | 5.56×4.90 | b_east_curtain | 0.931 | 0.980 / 0.875 / 0.937 | 0.758 | 2.42 | 3.06 | 0.802 | 0.922 |
| large | 5.56×4.90 | **c_north_absorber** ★ | **0.974** | 0.983 / 0.983 / 0.957 | 0.761 | 2.81 | 2.69 | 0.845 | 0.955 |
| large | 5.56×4.90 | d_two_wall | 0.950 | 0.980 / 0.926 / 0.943 | 0.743 | 2.62 | 2.86 | 0.806 | 0.966 |

Ranges over the 12: spatial 0.920–0.976 (mean 0.951) · magnitude 0.607–0.761 (mean 0.694) ·
phase 0.758–0.926 (mean 0.832) · RIR 0.898–0.986 (mean 0.952) · LSD-c 1.23–2.81 dB (mean 2.27).

Modes plotted, per geometry: small (1,0) 49.85 / (0,1) 54.62 / (1,1) 73.95 Hz · median (1,0)
28.92 / (0,1) 53.93 / (2,0) 57.84 Hz · large (1,0) 30.85 / (0,1) 35.00 / (1,1) 46.65 Hz.

## Scenarios

| tag | edit | status |
|---|---|---|
| a_baseline | all walls α = 0.15 | unseen geometry |
| b_east_curtain | east α = 0.50 | unseen geometry, trained combo |
| c_north_absorber ★ | north α = 0.70 | **doubly zero-shot** — see below |
| d_two_wall | east 0.50 + south 0.70 | unseen geometry, trained combos |

★ `north@0.70` maps to m = 1.204, inside Arm C's held-out slab `north (1.13, 1.28)`. That
wall/material combination appears in **no** training config, so (c) is unseen geometry *and*
unseen material placement. It scored highest in all three geometries.

## Four things this manifest deliberately does not blur

1. **Magnitude correlation is 0.607–0.761, not 0.95.** It pools all 601 bins including deep
   nulls, where small dB errors are large in correlation terms; spatial Pearson is evaluated at
   modal bins. Both appear on `figC` for exactly this reason. Quoting only the spatial number
   would overstate the result.

2. **`mode_shape_invariance` 0.9921 is a different quantity.** P3-2b recorded it for this arm as
   agreement with the *analytic cosine shape* on the *8×8* grid. The numbers here are pointwise
   pred-vs-GT on a 64×64 grid. The two are never summed, averaged, or presented as the same
   measurement.

3. **Accuracy degrades above ~170 Hz**, visible in `figB` and folded into the LSD figures. The
   band claim is 0–300 Hz with that caveat, not a flat claim across the band.

4. **Scenario (d) is not one of P3-2b's four frozen two-wall test combos.** Harmless here because
   every GT is re-simulated at 64×64 rather than read from the frozen corpus — noted so nobody
   goes looking for a stored file.

## Reproduce

```bash
python scripts/armC_demo_metrics.py    # GPU (tinycudann has no CPU path); ~4 min, writes metrics.json + fields/
python scripts/armC_demo_figures.py    # CPU only; ~40 s, writes the three PNGs + figures.json
```

`armC_demo_metrics.py` returns exit code 2 if the abort rule trips, and `armC_demo_figures.py`
declines to draw in that case.
