# P3-2b — figure manifest (meeting pack)

Five figures for the P3-2b chunk: one model conditioned on the room's geometry and its four wall absorptions, asked to render edited rooms **zero-shot** — the conditioning vector is computed from the physical parameters alone, no measurement of the target config is read, and nothing is optimised per config.

All numbers below are read at run time from the JSON files named in each row. Nothing on any figure is hand-entered. The only quantities not read from a file are the slope standard errors on figure B (re-derived from the same published sweep points, because the m-response schema carries `a` and `r2` but no SE) and the alpha <-> m conversion, taken from the frozen `aaf.data.mat_configs_cont` helpers.

## Provenance

| field | value |
|---|---|
| arm | `p3_2b_C_cont_mlinear` |
| conditioning | `m_linear`, cond_dim 60 |
| checkpoint | `outputs/p3_2/p3_2b_C_cont_mlinear/ckpt_iter0060000.pt` |
| iteration | 60000 |
| in-distribution val LSD | 1.0132 dB |
| band | [0.0, 300.0] Hz |
| configs evaluated | 280 over 50 geometries |
| training configs | 960 |
| manifest sha256 | `ecf0ee6e620dc56eff508a5c8c0334cdb41f21b88b6359317e5b8a43e690da04` |
| held-out slabs (m) | west m in [0.62, 0.77] (brackets alpha=0.50), north m in [1.13, 1.28] (brackets alpha=0.70) |
| kappa (frozen, P3-2 gate) | 1.6607564051 |
| acceptance spec / sha256 | `p3_2b.accept/1` / `a8479c5e1dcc3ab5b2a505809e0f9d9f7dd4590f009f3549c9070e860a33caa1` |
| sources | `outputs/p3_2b/eval/p3_2b_C_cont_mlinear/summary.json`, `outputs/p3_2b/eval/p3_2b_C_cont_mlinear/m_response.json`, `outputs/p3_2b/ablation.json` |

## Scoping (must accompany any verbal claim)

The wall selectivity that makes this chunk legible is a property of the **ISM simulator**: its reflection coefficient is real and angle-independent, so a pure x-axial mode sees *exactly zero* damping from the north/south walls. Real locally-reacting walls follow Kuttruff and would show only ~2:1, with **no invariant family**. The claim is therefore *"the model learns the simulator's per-wall law"* — not *"the model learns room acoustics"*.

## The kappa correction (every rho on every figure depends on it)

The bandwidth estimator returns a **calibrated** -3 dB width, not the raw Lorentzian width. The P3-2 physics gate's T5 fit gives `BW = 0.302 + 1.6608 * (gamma/pi)`. The intercept cancels in a paired delta; **the slope does not**. So the theoretical slope of a measured delta-bandwidth against delta-m is

```
a_theory = kappa * c / (4*pi*D),   kappa = 1.6607564051417665
D = L for west/east on the x-axial family; D = W for south/north on the y-axial family
a_theory = 0 for the orthogonal family
```

Every rho on every figure is `a_fit / a_theory` against that **kappa-scaled** line, and the raw-Lorentzian comparison is printed alongside (figure B banner, figure E column `rho vs RAW`) so the correction is visible rather than assumed. Scoring against the raw value would hand a perfect model rho ~ 0.60 and fail it.

## S2 verdict

```
P3-2b p3_2b_C_cont_mlinear iter 60000 S2_unseen_geom_slab: PASS | edit_bw_slope=0.959>=0.80; edit_bw_pearson=0.868>=0.80; edit_gain=1.087>1.00; abs_rho_minus_1=0.053<=0.25 | blockers: none | thr a8479c5e1dcc
```

| criterion | value | op | threshold | result |
|---|---|---|---|---|
| `edit_bw_slope` | 0.9588 | >= | 0.800 | PASS |
| `edit_bw_pearson` | 0.8676 | >= | 0.800 | PASS |
| `edit_gain` | 1.0872 | > | 1.000 | PASS |
| `abs_rho_minus_1` | 0.0528 | <= | 0.250 | PASS |

Cross-arm: arm `p3_2b_B_cont_fourier` is the first in ladder order to clear the S2 gate.

Honesty note carried from `EVAL.md`: arm A inherits the P3-2 dataset, in which one holdout was an **EXTRAPOLATION**, so arm A cannot separate *the renderer did not help* from *that holdout was unfair*. Arms B/C/D use the P3-2b manifest, whose held-out slabs are strictly interior to the sampled m range.

## Figures

### `A_pick_your_wall.png`

- **Shows:** Four panels, one per edited wall, all edited to alpha=0.70 (M3): change in -3 dB modal bandwidth per mode family -- ground truth, kappa-scaled ISM-ray theory, and the zero-shot model. Panels whose (wall, alpha) lands in a held-out m-slab are outlined in red as NEVER-SEEN COMBINATIONS.
- **Size:** 1920x1080 px
- **Source files / keys:** `outputs/p3_2b/eval/p3_2b_C_cont_mlinear/summary.json :: selectivity_matrix.M3.<wall>.<family>.{gt_d_bw, pred_d_bw, theory_d_bw, n}`
- **Numbers on the figure:**

  ```
  A west->M3 x-axial GT: +9.676 Hz (n=30)
  A west->M3 y-axial GT: +0.329 Hz (n=20)
  A west->M3 tangential GT: +7.218 Hz (n=33)
  A west->M3 x-axial theory: +9.928 Hz (n=30)
  A west->M3 y-axial theory: +0.000 Hz (n=20)
  A west->M3 tangential theory: +7.622 Hz (n=33)
  A west->M3 x-axial model: +11.064 Hz (n=30)
  A west->M3 y-axial model: +0.408 Hz (n=20)
  A west->M3 tangential model: +6.734 Hz (n=33)
  A east->M3 x-axial GT: +10.673 Hz (n=29)
  A east->M3 y-axial GT: +0.715 Hz (n=17)
  A east->M3 tangential GT: +8.012 Hz (n=35)
  A east->M3 x-axial theory: +9.990 Hz (n=29)
  A east->M3 y-axial theory: +0.000 Hz (n=17)
  A east->M3 tangential theory: +7.562 Hz (n=35)
  A east->M3 x-axial model: +11.442 Hz (n=29)
  A east->M3 y-axial model: +0.943 Hz (n=17)
  A east->M3 tangential model: +8.710 Hz (n=35)
  A south->M3 x-axial GT: +0.340 Hz (n=29)
  A south->M3 y-axial GT: +11.522 Hz (n=21)
  A south->M3 tangential GT: +7.953 Hz (n=33)
  A south->M3 x-axial theory: +0.000 Hz (n=29)
  A south->M3 y-axial theory: +11.902 Hz (n=21)
  A south->M3 tangential theory: +8.334 Hz (n=33)
  A south->M3 x-axial model: +0.318 Hz (n=29)
  A south->M3 y-axial model: +12.488 Hz (n=21)
  A south->M3 tangential model: +7.336 Hz (n=33)
  A north->M3 x-axial GT: +0.642 Hz (n=29)
  A north->M3 y-axial GT: +12.372 Hz (n=20)
  A north->M3 tangential GT: +8.853 Hz (n=32)
  A north->M3 x-axial theory: +0.000 Hz (n=29)
  A north->M3 y-axial theory: +11.919 Hz (n=20)
  A north->M3 tangential theory: +8.414 Hz (n=32)
  A north->M3 x-axial model: +0.860 Hz (n=29)
  A north->M3 y-axial model: +12.533 Hz (n=20)
  A north->M3 tangential model: +8.347 Hz (n=32)
  ```

### `B_m_response.png`

- **Shows:** The m-response, 3 geometries x 4 walls. Model curve, GT points and the kappa-scaled ISM-ray theory line through the origin, with the held-out m-slab shaded on the west and north panels and the orthogonal family drawn as a thin grey ~0 series. Correct behaviour is the predicted curve passing straight through the shaded band along the theory line. The hatched region is alpha > 0.70, an EXTRAPOLATION for arm A only.
- **Size:** 1920x1080 px
- **Source files / keys:** `outputs/p3_2b/eval/p3_2b_C_cont_mlinear/m_response.json :: geometries[].walls.<wall>.{points[], fit, a_theory_hz_per_m, a_theory_raw_hz_per_m, slab_m, D_m}`; `outputs/p3_2b/eval/p3_2b_C_cont_mlinear/summary.json :: slope_fit.aggregate.own_family.slab_local, slope_fit.kappa, slope_fit.rho_vs_raw_theory_median`
- **Derived on the figure:** a_fit SE is not in the schema; it is re-derived here from the same published sweep points as sqrt(SSres/((n-1)*Sxx)) for the through-origin fit.
- **Numbers on the figure:**

  ```
  B geom1 (min_aspect) L=3.27 W=4.98 west wall: a_fit_pred=11.105+/-0.136, a_fit_gt=13.233+/-0.061, a_theory=13.863 (raw 8.347) Hz/m, rho_pred=0.801, rho_gt=0.955, rho_pred_vs_raw=1.330, slab=[0.62, 0.77], n_pts=20
  B geom1 (min_aspect) L=3.27 W=4.98 east wall: a_fit_pred=11.681+/-0.116, a_fit_gt=14.154+/-0.005, a_theory=13.863 (raw 8.347) Hz/m, rho_pred=0.843, rho_gt=1.021, rho_pred_vs_raw=1.399, slab=none, n_pts=20
  B geom1 (min_aspect) L=3.27 W=4.98 south wall: a_fit_pred=7.366+/-0.166, a_fit_gt=8.932+/-0.038, a_theory=9.103 (raw 5.481) Hz/m, rho_pred=0.809, rho_gt=0.981, rho_pred_vs_raw=1.344, slab=none, n_pts=20
  B geom1 (min_aspect) L=3.27 W=4.98 north wall: a_fit_pred=7.926+/-0.269, a_fit_gt=9.725+/-0.018, a_theory=9.103 (raw 5.481) Hz/m, rho_pred=0.871, rho_gt=1.068, rho_pred_vs_raw=1.446, slab=[1.13, 1.28], n_pts=20
  B geom3 (median_aspect) L=5.56 W=4.90 west wall: a_fit_pred=7.243+/-0.110, a_fit_gt=7.843+/-0.035, a_theory=8.153 (raw 4.909) Hz/m, rho_pred=0.888, rho_gt=0.962, rho_pred_vs_raw=1.475, slab=[0.62, 0.77], n_pts=20
  B geom3 (median_aspect) L=5.56 W=4.90 east wall: a_fit_pred=8.082+/-0.067, a_fit_gt=8.525+/-0.013, a_theory=8.153 (raw 4.909) Hz/m, rho_pred=0.991, rho_gt=1.046, rho_pred_vs_raw=1.646, slab=none, n_pts=20
  B geom3 (median_aspect) L=5.56 W=4.90 south wall: a_fit_pred=10.970+/-0.160, a_fit_gt=8.876+/-0.040, a_theory=9.251 (raw 5.570) Hz/m, rho_pred=1.186, rho_gt=0.959, rho_pred_vs_raw=1.969, slab=none, n_pts=20
  B geom3 (median_aspect) L=5.56 W=4.90 north wall: a_fit_pred=10.717+/-0.102, a_fit_gt=9.625+/-0.013, a_theory=9.251 (raw 5.570) Hz/m, rho_pred=1.158, rho_gt=1.040, rho_pred_vs_raw=1.924, slab=[1.13, 1.28], n_pts=20
  B geom2 (max_aspect) L=5.93 W=3.18 west wall: a_fit_pred=7.072+/-0.104, a_fit_gt=7.503+/-0.035, a_theory=7.644 (raw 4.603) Hz/m, rho_pred=0.925, rho_gt=0.981, rho_pred_vs_raw=1.536, slab=[0.62, 0.77], n_pts=20
  B geom2 (max_aspect) L=5.93 W=3.18 east wall: a_fit_pred=7.423+/-0.142, a_fit_gt=8.287+/-0.020, a_theory=7.644 (raw 4.603) Hz/m, rho_pred=0.971, rho_gt=1.084, rho_pred_vs_raw=1.613, slab=none, n_pts=20
  B geom2 (max_aspect) L=5.93 W=3.18 south wall: a_fit_pred=10.630+/-0.132, a_fit_gt=13.498+/-0.063, a_theory=14.255 (raw 8.583) Hz/m, rho_pred=0.746, rho_gt=0.947, rho_pred_vs_raw=1.238, slab=none, n_pts=20
  B geom2 (max_aspect) L=5.93 W=3.18 north wall: a_fit_pred=9.813+/-0.251, a_fit_gt=14.468+/-0.006, a_theory=14.255 (raw 8.583) Hz/m, rho_pred=0.688, rho_gt=1.015, rho_pred_vs_raw=1.143, slab=[1.13, 1.28], n_pts=20
  B aggregate slab_local: rho_median=0.9472, ci95=[0.8184042697801307, 1.188990759151383], a_fit=11.4659, a_theory=10.7181, n_cells=18
  B slab panels on this figure: median rho_pred = 0.8796 over 6 panels (cross-check against the aggregate above)
  ```

### `C_selectivity_matrix.png`

- **Shows:** Wall x family selectivity matrix per material: ground truth, the zero-shot model, and the model-minus-GT residual, in raw Hz of bandwidth change. Rows in a held-out m-slab are outlined gold.
- **Size:** 1920x1080 px
- **Source files / keys:** `outputs/p3_2b/eval/p3_2b_C_cont_mlinear/summary.json :: selectivity_matrix.<material>.<wall>.<family>.{gt_d_bw, pred_d_bw, residual_d_bw}, selectivity_index`
- **Numbers on the figure:**

  ```
  C M1 west x_axial gt_d_bw: -1.068 Hz
  C M1 west y_axial gt_d_bw: -0.035 Hz
  C M1 west tangential gt_d_bw: -0.738 Hz
  C M1 east x_axial gt_d_bw: -1.072 Hz
  C M1 east y_axial gt_d_bw: -0.069 Hz
  C M1 east tangential gt_d_bw: -0.773 Hz
  C M1 south x_axial gt_d_bw: -0.039 Hz
  C M1 south y_axial gt_d_bw: -1.327 Hz
  C M1 south tangential gt_d_bw: -0.899 Hz
  C M1 north x_axial gt_d_bw: -0.063 Hz
  C M1 north y_axial gt_d_bw: -1.332 Hz
  C M1 north tangential gt_d_bw: -0.897 Hz
  C M1 west x_axial pred_d_bw: -0.467 Hz
  C M1 west y_axial pred_d_bw: -0.131 Hz
  C M1 west tangential pred_d_bw: -0.474 Hz
  C M1 east x_axial pred_d_bw: -0.419 Hz
  C M1 east y_axial pred_d_bw: -0.204 Hz
  C M1 east tangential pred_d_bw: -0.536 Hz
  C M1 south x_axial pred_d_bw: -0.071 Hz
  C M1 south y_axial pred_d_bw: -0.583 Hz
  C M1 south tangential pred_d_bw: -0.480 Hz
  C M1 north x_axial pred_d_bw: -0.148 Hz
  C M1 north y_axial pred_d_bw: -0.567 Hz
  C M1 north tangential pred_d_bw: -0.569 Hz
  C M1 west x_axial residual_d_bw: +0.601 Hz
  C M1 west y_axial residual_d_bw: -0.095 Hz
  C M1 west tangential residual_d_bw: +0.265 Hz
  C M1 east x_axial residual_d_bw: +0.654 Hz
  C M1 east y_axial residual_d_bw: -0.135 Hz
  C M1 east tangential residual_d_bw: +0.237 Hz
  C M1 south x_axial residual_d_bw: -0.032 Hz
  C M1 south y_axial residual_d_bw: +0.744 Hz
  C M1 south tangential residual_d_bw: +0.419 Hz
  C M1 north x_axial residual_d_bw: -0.085 Hz
  C M1 north y_axial residual_d_bw: +0.765 Hz
  C M1 north tangential residual_d_bw: +0.328 Hz
  C A030 west x_axial gt_d_bw: +1.889 Hz
  C A030 west y_axial gt_d_bw: +0.062 Hz
  C A030 west tangential gt_d_bw: +1.395 Hz
  C A030 east x_axial gt_d_bw: +1.919 Hz
  C A030 east y_axial gt_d_bw: +0.122 Hz
  C A030 east tangential gt_d_bw: +1.457 Hz
  C A030 south x_axial gt_d_bw: +0.068 Hz
  C A030 south y_axial gt_d_bw: +2.249 Hz
  C A030 south tangential gt_d_bw: +1.531 Hz
  C A030 north x_axial gt_d_bw: +0.114 Hz
  C A030 north y_axial gt_d_bw: +2.281 Hz
  C A030 north tangential gt_d_bw: +1.570 Hz
  C A030 west x_axial pred_d_bw: +1.448 Hz
  C A030 west y_axial pred_d_bw: +0.095 Hz
  C A030 west tangential pred_d_bw: +1.194 Hz
  C A030 east x_axial pred_d_bw: +1.325 Hz
  C A030 east y_axial pred_d_bw: +0.171 Hz
  C A030 east tangential pred_d_bw: +1.260 Hz
  C A030 south x_axial pred_d_bw: +0.049 Hz
  C A030 south y_axial pred_d_bw: +2.190 Hz
  C A030 south tangential pred_d_bw: +1.104 Hz
  C A030 north x_axial pred_d_bw: +0.161 Hz
  C A030 north y_axial pred_d_bw: +1.972 Hz
  C A030 north tangential pred_d_bw: +1.295 Hz
  C A030 west x_axial residual_d_bw: -0.440 Hz
  C A030 west y_axial residual_d_bw: +0.033 Hz
  C A030 west tangential residual_d_bw: -0.201 Hz
  C A030 east x_axial residual_d_bw: -0.593 Hz
  C A030 east y_axial residual_d_bw: +0.049 Hz
  C A030 east tangential residual_d_bw: -0.198 Hz
  C A030 south x_axial residual_d_bw: -0.019 Hz
  C A030 south y_axial residual_d_bw: -0.060 Hz
  C A030 south tangential residual_d_bw: -0.427 Hz
  C A030 north x_axial residual_d_bw: +0.047 Hz
  C A030 north y_axial residual_d_bw: -0.309 Hz
  C A030 north tangential residual_d_bw: -0.276 Hz
  C M2 west x_axial gt_d_bw: +5.102 Hz
  C M2 west y_axial gt_d_bw: +0.170 Hz
  C M2 west tangential gt_d_bw: +3.792 Hz
  C M2 east x_axial gt_d_bw: +5.297 Hz
  C M2 east y_axial gt_d_bw: +0.346 Hz
  C M2 east tangential gt_d_bw: +4.021 Hz
  C M2 south x_axial gt_d_bw: +0.185 Hz
  C M2 south y_axial gt_d_bw: +6.074 Hz
  C M2 south tangential gt_d_bw: +4.201 Hz
  C M2 north x_axial gt_d_bw: +0.319 Hz
  C M2 north y_axial gt_d_bw: +6.272 Hz
  C M2 north tangential gt_d_bw: +4.449 Hz
  C M2 west x_axial pred_d_bw: +5.159 Hz
  C M2 west y_axial pred_d_bw: +0.313 Hz
  C M2 west tangential pred_d_bw: +3.608 Hz
  C M2 east x_axial pred_d_bw: +4.597 Hz
  C M2 east y_axial pred_d_bw: +0.567 Hz
  C M2 east tangential pred_d_bw: +3.769 Hz
  C M2 south x_axial pred_d_bw: +0.220 Hz
  C M2 south y_axial pred_d_bw: +7.090 Hz
  C M2 south tangential pred_d_bw: +3.986 Hz
  C M2 north x_axial pred_d_bw: +0.450 Hz
  C M2 north y_axial pred_d_bw: +6.029 Hz
  C M2 north tangential pred_d_bw: +3.735 Hz
  C M2 west x_axial residual_d_bw: +0.057 Hz
  C M2 west y_axial residual_d_bw: +0.143 Hz
  C M2 west tangential residual_d_bw: -0.184 Hz
  C M2 east x_axial residual_d_bw: -0.700 Hz
  C M2 east y_axial residual_d_bw: +0.221 Hz
  C M2 east tangential residual_d_bw: -0.253 Hz
  C M2 south x_axial residual_d_bw: +0.036 Hz
  C M2 south y_axial residual_d_bw: +1.017 Hz
  C M2 south tangential residual_d_bw: -0.215 Hz
  C M2 north x_axial residual_d_bw: +0.131 Hz
  C M2 north y_axial residual_d_bw: -0.244 Hz
  C M2 north tangential residual_d_bw: -0.714 Hz
  C M3 west x_axial gt_d_bw: +9.676 Hz
  C M3 west y_axial gt_d_bw: +0.329 Hz
  C M3 west tangential gt_d_bw: +7.218 Hz
  C M3 east x_axial gt_d_bw: +10.673 Hz
  C M3 east y_axial gt_d_bw: +0.715 Hz
  C M3 east tangential gt_d_bw: +8.012 Hz
  C M3 south x_axial gt_d_bw: +0.340 Hz
  C M3 south y_axial gt_d_bw: +11.522 Hz
  C M3 south tangential gt_d_bw: +7.953 Hz
  C M3 north x_axial gt_d_bw: +0.642 Hz
  C M3 north y_axial gt_d_bw: +12.372 Hz
  C M3 north tangential gt_d_bw: +8.853 Hz
  C M3 west x_axial pred_d_bw: +11.064 Hz
  C M3 west y_axial pred_d_bw: +0.408 Hz
  C M3 west tangential pred_d_bw: +6.734 Hz
  C M3 east x_axial pred_d_bw: +11.442 Hz
  C M3 east y_axial pred_d_bw: +0.943 Hz
  C M3 east tangential pred_d_bw: +8.710 Hz
  C M3 south x_axial pred_d_bw: +0.318 Hz
  C M3 south y_axial pred_d_bw: +12.488 Hz
  C M3 south tangential pred_d_bw: +7.336 Hz
  C M3 north x_axial pred_d_bw: +0.860 Hz
  C M3 north y_axial pred_d_bw: +12.533 Hz
  C M3 north tangential pred_d_bw: +8.347 Hz
  C M3 west x_axial residual_d_bw: +1.388 Hz
  C M3 west y_axial residual_d_bw: +0.080 Hz
  C M3 west tangential residual_d_bw: -0.483 Hz
  C M3 east x_axial residual_d_bw: +0.769 Hz
  C M3 east y_axial residual_d_bw: +0.228 Hz
  C M3 east tangential residual_d_bw: +0.698 Hz
  C M3 south x_axial residual_d_bw: -0.022 Hz
  C M3 south y_axial residual_d_bw: +0.965 Hz
  C M3 south tangential residual_d_bw: -0.616 Hz
  C M3 north x_axial residual_d_bw: +0.217 Hz
  C M3 north y_axial residual_d_bw: +0.161 Hz
  C M3 north tangential residual_d_bw: -0.506 Hz
  C selectivity_index: gt=17.688, theory=32.829, pred=13.770
  ```

### `D_s2_headline.png`

- **Shows:** The S2 headline table: the four frozen acceptance criteria with their pass/fail state and the thresholds' sha256, the S2 fidelity and edit measurements, the edit_gain against the null (baseline) render, and the two held-out slab combos broken out individually.
- **Size:** 1920x1080 px
- **Source files / keys:** `outputs/p3_2b/eval/p3_2b_C_cont_mlinear/summary.json :: splits.S2_unseen_geom_slab.{n_configs, n_cells, frac_modes_dropped, fidelity, null_fidelity, edit, edit_detail, per_combo}, verdict, slope_fit.aggregate.own_family.slab_local`
- **Numbers on the figure:**

  ```
  D criterion edit_bw_slope: value=0.9588 >= 0.800 -> PASS
  D criterion edit_bw_pearson: value=0.8676 >= 0.800 -> PASS
  D criterion edit_gain: value=1.0872 > 1.000 -> PASS
  D criterion abs_rho_minus_1: value=0.0528 <= 0.250 -> PASS
  D spec_sha: a8479c5e1dcc3ab5b2a505809e0f9d9f7dd4590f009f3549c9070e860a33caa1
  D S2 configs / mode cells: 20 / 168
  D S2 frac modes dropped: 0.455
  D S2 band LSD (dB): 2.669
  D S2 mag corr: 0.912
  D S2 phase corr (mag-weighted): 0.911
  D S2 RIR pearson: 0.925
  D S2 E_BW (Hz, lower better): 1.560
  D S2 E_LVL (dB, lower better): 1.690
  D S2 GT effect size (Hz): 5.037
  D S2 model effect size (Hz): 4.992
  D S2 rho slab-local (kappa): 0.947
  D S2 rho 95% CI: [0.82, 1.19]
  D S2 rho vs RAW theory: 1.612
  D S2 in-dist val LSD (dB): 1.013
  D S2 edit_gain: 1.0872 (model band LSD 2.6692 dB, null band LSD 2.9018 dB)
  D S2 combo north0.70: GT effect 6.782 Hz, model effect 6.700 Hz, slope 0.950, gain 1.117 (n_configs=10)
  D S2 combo west0.50: GT effect 3.413 Hz, model effect 3.403 Hz, slope 0.981, gain 1.062 (n_configs=10)
  D verdict: passed=True | P3-2b p3_2b_C_cont_mlinear iter 60000 S2_unseen_geom_slab: PASS | edit_bw_slope=0.959>=0.80; edit_bw_pearson=0.868>=0.80; edit_gain=1.087>1.00; abs_rho_minus_1=0.053<=0.25 | blockers: none | thr a8479c5e1dcc
  ```

### `E_ablation.png`

- **Shows:** The cross-arm ablation table -- per-split edit slope and edit_gain, in-distribution val LSD and kappa-scaled rho for each arm plus the P3-2 baseline -- with the attribution deltas for each rung of the ladder and a banner naming the first arm (if any) to clear the S2 gate.
- **Size:** 1920x1080 px
- **Source files / keys:** `outputs/p3_2b/ablation.json :: rows[], attribution[], s2.first_clearing_arm`
- **Numbers on the figure:**

  ```
  E row P3-2: cond=geom_alpha, val_lsd=2.6869, S2 slope=0.1331, S2 gain=0.8736, rho=n/a, verdict=not evaluated
  E row A: cond=geom_alpha_fourier, val_lsd=0.9308, S2 slope=0.1530, S2 gain=0.8680, rho=0.5093, verdict=FAIL
  E row B: cond=geom_alpha_fourier, val_lsd=0.9984, S2 slope=1.1471, S2 gain=1.0842, rho=1.0394, verdict=PASS
  E row C: cond=m_linear, val_lsd=1.0132, S2 slope=0.9588, S2 gain=1.0872, rho=0.9472, verdict=PASS
  E row D: cond=m_linear, val_lsd=1.4637, S2 slope=1.0378, S2 gain=1.0824, rho=1.0310, verdict=PASS
  E delta renderer + eval protocol [A vs P3-2 baseline] s2_edit_bw_slope: +0.0198
  E delta continuous alpha sampling [B - A] s2_edit_bw_slope: +0.9941
  E delta m = -ln(1-alpha) coordinate [C - B] s2_edit_bw_slope: -0.1883
  E delta multi-wall training [C - D] s2_edit_bw_slope: -0.0789
  E delta renderer + eval protocol [A vs P3-2 baseline] s2_edit_bw_pearson: -0.0337
  E delta continuous alpha sampling [B - A] s2_edit_bw_pearson: +0.3723
  E delta m = -ln(1-alpha) coordinate [C - B] s2_edit_bw_pearson: -0.0035
  E delta multi-wall training [C - D] s2_edit_bw_pearson: -0.0031
  E delta renderer + eval protocol [A vs P3-2 baseline] rho_slab_local: n/a
  E delta continuous alpha sampling [B - A] rho_slab_local: +0.5301
  E delta m = -ln(1-alpha) coordinate [C - B] rho_slab_local: -0.0922
  E delta multi-wall training [C - D] rho_slab_local: -0.0838
  E s2.first_clearing_arm: p3_2b_B_cont_fourier
  ```

## Headline numbers (all from `summary.json`)

| split | n cfg | n cells | band LSD (dB) | E_BW (Hz) | edit slope | edit pearson | edit gain |
|---|---|---|---|---|---|---|---|
| S1_unseen_geom_nonslab_1wall | 100 | 802 | 2.979 | 1.244 | 0.997 | 0.877 | 1.053 |
| S2_unseen_geom_slab | 20 | 168 | 2.669 | 1.560 | 0.959 | 0.868 | 1.087 |
| S3_seen_geom_slab | 80 | 1176 | 0.617 | 2.219 | 0.720 | 0.944 | 2.734 |
| S4_unseen_geom_alpha030 | 40 | 347 | 3.000 | 0.606 | 0.789 | 0.612 | 1.001 |
| S5_unseen_geom_2wall | 40 | 326 | 2.632 | 2.286 | 1.010 | 0.828 | 1.176 |

**S2 is the chunk's question.** A strong S1 or S3 with a dead S2 is exactly the P3-2 result being re-tested and must not be reported as progress.

## Reproduction

```bash
export PYTHONPATH="$PWD"
python scripts/p3_2b_ablation.py                       # CPU: ablation.json + EVAL.md
python scripts/make_p3_2b_figures.py --arm p3_2b_C_cont_mlinear   # CPU: figures + this manifest
```

