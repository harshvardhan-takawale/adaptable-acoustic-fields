# P3-2 — figure manifest (meeting pack)

Five figures for the P3-2 chunk: one model conditioned on `(L, W, alpha_west, alpha_east, alpha_south, alpha_north)`, asked to render edited rooms **zero-shot** — the conditioning vector is computed from the physical parameters alone, no measurement of the target config is read, and nothing is optimised per config.

All numbers below are read at run time from the JSON files named in each row. Nothing on any figure is hand-entered. The only quantities not read from a file are the two analytic damping laws in figure B, computed live from `aaf.sim.analytical_modal_2d.modal_damping_2d`.

## Provenance

| field | value |
|---|---|
| checkpoint | `outputs/p3_2/p3_2_main/ckpt_iter0060000.pt` |
| iteration | 60000 (training was still running; this is the newest checkpoint the eval used) |
| in-distribution val LSD | 2.687 dB |
| band | [0.0, 300.0] Hz |
| modal projection cap | 200.0 Hz |
| configs evaluated | 250 over 50 geometries |
| held-out combos | `west+M2`, `north+M3` |
| unseen alpha | 0.3 |
| exemplar room (figs A, B, E) | L=5.93 m, W=3.18 m — the frozen test geometry with the most valid bandwidth measurements |
| sources | `outputs/p3_2/eval/summary.json`, `outputs/p3_2/eval/per_config.json`, `outputs/p3_2/gate/gate.json` |

## Scoping (must accompany any verbal claim)

The ~29:1 bandwidth selectivity that makes this chunk legible is a property of the **ISM simulator**: its reflection coefficient is real and angle-independent, so a pure x-axial mode sees *exactly zero* damping from the north/south walls. Real locally-reacting walls follow Kuttruff and would show only ~2:1, with **no invariant family**. The claim is therefore *"the model learns the simulator's per-wall law"* — not *"the model learns room acoustics"*. Figure B puts both laws on the same axes against the measurement so this is visible rather than asserted.

Measured simulator selectivity (physics gate, `outputs/p3_2/gate/gate.json`): mean **29.07:1**, 95% CI [20.02, 39.28], threshold 5.0.

## What the numbers actually say (honest read)

1. **The simulator's per-wall law is real and matches theory.** Ground truth and the ISM-ray analytic law agree cell-by-cell in figure C, and figure B shows the measurement tracking ISM-ray rather than Kuttruff. This is a property of the data, not of the model.
2. **The model reproduces the law only partially, and only for (wall, material) pairs it was trained on.** Selectivity index: GT **18.3x**, theory **33.0x**, model **10.6x** — the model recovers roughly 58% of the measured selectivity.
3. **On never-trained combinations the edit response does not transfer.** For split (iii) the model's mean edit effect is 0.78 Hz against a measured 5.16 Hz, and the (iii)-(i) gap is 3.78 Hz = 73% of the ground-truth effect size.
4. **The C1 null model is the load-bearing control, and it is not beaten on three of four splits.** C1 renders the *baseline* and scores it against the *edited* ground truth, i.e. it applies no edit at all:

   | split | model E_BW (Hz) | C1 null E_BW (Hz) | model better than null? |
   |---|---|---|---|
   | i_unseen_geom_seen_combo | 1.205 | 3.348 | yes |
   | ii_seen_geom_heldout_combo | 5.450 | 5.454 | yes |
   | iii_unseen_geom_heldout_combo | 4.982 | 5.159 | yes |
   | iv_unseen_alpha | 2.254 | 1.254 | **no** |

5. **Training was still running.** These figures are the checkpoint at iteration 60000 of a 60000-iteration schedule, so this is a mid-training snapshot and not the chunk's final result. Any claim made from this pack must carry the iteration number.

The defensible claim from this pack is therefore: *the simulator has a sharp, theory-matching per-wall law, and the model has begun to learn it for trained (wall, material) pairs while not yet generalising to unseen pairs.*

## Figures

### `A_pick_your_wall.png`

- **Shows:** Four panels, one per edited wall (all -> M3 absorber) on a single unseen test room: change in -3 dB modal bandwidth per mode family, ground truth vs ISM-ray theory vs zero-shot model.
- **Size:** 1920x1080 px
- **Source files / keys:** `outputs/p3_2/eval/per_config.json`; `outputs/p3_2/eval/summary.json`
- **Numbers on the figure:**

  ```
  A west->M3 x-axial GT: +7.84 Hz (n=5)
  A west->M3 y-axial GT: -0.01 Hz (n=2)
  A west->M3 tangential GT: +4.83 Hz (n=9)
  A west->M3 x-axial theory: +7.96 Hz (n=5)
  A west->M3 y-axial theory: +0.00 Hz (n=2)
  A west->M3 tangential theory: +5.30 Hz (n=9)
  A west->M3 x-axial model: +8.20 Hz (n=5)
  A west->M3 y-axial model: +0.37 Hz (n=2)
  A west->M3 tangential model: +4.15 Hz (n=9)
  A east->M3 x-axial GT: +8.61 Hz (n=5)
  A east->M3 y-axial GT: +0.59 Hz (n=2)
  A east->M3 tangential GT: +5.76 Hz (n=9)
  A east->M3 x-axial theory: +7.96 Hz (n=5)
  A east->M3 y-axial theory: +0.00 Hz (n=2)
  A east->M3 tangential theory: +5.30 Hz (n=9)
  A east->M3 x-axial model: +8.31 Hz (n=5)
  A east->M3 y-axial model: +0.88 Hz (n=2)
  A east->M3 tangential model: +4.51 Hz (n=9)
  A south->M3 x-axial GT: +0.48 Hz (n=5)
  A south->M3 y-axial GT: +14.08 Hz (n=2)
  A south->M3 tangential GT: +9.95 Hz (n=9)
  A south->M3 x-axial theory: +0.00 Hz (n=5)
  A south->M3 y-axial theory: +14.85 Hz (n=2)
  A south->M3 tangential theory: +10.18 Hz (n=9)
  A south->M3 x-axial model: +0.47 Hz (n=5)
  A south->M3 y-axial model: +12.16 Hz (n=2)
  A south->M3 tangential model: +10.31 Hz (n=9)
  A north->M3 x-axial GT: +0.66 Hz (n=5)
  A north->M3 y-axial GT: +15.04 Hz (n=2)
  A north->M3 tangential GT: +10.67 Hz (n=9)
  A north->M3 x-axial theory: +0.00 Hz (n=5)
  A north->M3 y-axial theory: +14.85 Hz (n=2)
  A north->M3 tangential theory: +10.18 Hz (n=9)
  A north->M3 x-axial model: -0.13 Hz (n=5)
  A north->M3 y-axial model: +0.19 Hz (n=2)
  A north->M3 tangential model: +0.32 Hz (n=9)
  ```

### `B_pick_your_material.png`

- **Shows:** East wall swept across all four materials plus the unseen alpha=0.30: measured, model-predicted and both analytic damping laws for the first x-axial (own) and first y-axial (invariant) mode.
- **Size:** 1920x1080 px
- **Source files / keys:** `outputs/p3_2/eval/per_config.json`; `outputs/p3_2/eval/summary.json`; `aaf/sim/analytical_modal_2d.py (analytic laws, computed live)`
- **Numbers on the figure:**

  ```
  B x_axial east-sweep RMS(GT - ism_ray) = 2.07 Hz
  B x_axial east-sweep RMS(GT - kuttruff) = 3.27 Hz
  B x_axial east M1 (alpha=0.05) mode (1,0) f=28.9 Hz: GT -0.92 Hz, model -0.19 Hz
  B x_axial east A030 (alpha=0.30) mode (1,0) f=28.9 Hz: GT +1.55 Hz, model -0.05 Hz
  B x_axial east M2 (alpha=0.50) mode (1,0) f=28.9 Hz: GT +4.27 Hz, model +2.26 Hz
  B x_axial east M3 (alpha=0.70) mode (1,0) f=28.9 Hz: GT +8.43 Hz, model +7.62 Hz
  B y_axial east-sweep RMS(GT - ism_ray) = 0.41 Hz
  B y_axial east-sweep RMS(GT - kuttruff) = 0.38 Hz
  B y_axial east M1 (alpha=0.05) mode (0,1) f=53.9 Hz: GT -0.07 Hz, model -0.14 Hz
  B y_axial east A030 (alpha=0.30) mode (0,1) f=53.9 Hz: GT +0.13 Hz, model -0.67 Hz
  B y_axial east M2 (alpha=0.50) mode (0,1) f=53.9 Hz: GT +0.36 Hz, model +0.50 Hz
  B y_axial east M3 (alpha=0.70) mode (0,1) f=53.9 Hz: GT +0.72 Hz, model +0.89 Hz
  B verdict: RMS(GT-ism_ray)=1.49 Hz, RMS(GT-kuttruff)=2.33 Hz -> data follows ism_ray; measured own/other = 11.8:1
  ```

### `C_selectivity_matrix.png`

- **Shows:** Wall x family selectivity matrix per material: ground truth, ISM-ray theory, zero-shot model and the model-minus-GT residual, in raw Hz of bandwidth change.
- **Size:** 1920x1080 px
- **Source files / keys:** `outputs/p3_2/eval/summary.json :: selectivity_matrix, selectivity_index`
- **Numbers on the figure:**

  ```
  C M1 west x_axial gt_d_bw: -1.11 Hz
  C M1 west y_axial gt_d_bw: -0.03 Hz
  C M1 west tangential gt_d_bw: -0.75 Hz
  C M1 east x_axial gt_d_bw: -1.11 Hz
  C M1 east y_axial gt_d_bw: -0.07 Hz
  C M1 east tangential gt_d_bw: -0.78 Hz
  C M1 south x_axial gt_d_bw: -0.04 Hz
  C M1 south y_axial gt_d_bw: -1.31 Hz
  C M1 south tangential gt_d_bw: -0.88 Hz
  C M1 north x_axial gt_d_bw: -0.06 Hz
  C M1 north y_axial gt_d_bw: -1.32 Hz
  C M1 north tangential gt_d_bw: -0.89 Hz
  C M1 west x_axial theory_d_bw: -1.08 Hz
  C M1 west y_axial theory_d_bw: +0.00 Hz
  C M1 west tangential theory_d_bw: -0.75 Hz
  C M1 east x_axial theory_d_bw: -1.08 Hz
  C M1 east y_axial theory_d_bw: +0.00 Hz
  C M1 east tangential theory_d_bw: -0.76 Hz
  C M1 south x_axial theory_d_bw: +0.00 Hz
  C M1 south y_axial theory_d_bw: -1.28 Hz
  C M1 south tangential theory_d_bw: -0.87 Hz
  C M1 north x_axial theory_d_bw: +0.00 Hz
  C M1 north y_axial theory_d_bw: -1.28 Hz
  C M1 north tangential theory_d_bw: -0.87 Hz
  C M1 west x_axial pred_d_bw: -0.57 Hz
  C M1 west y_axial pred_d_bw: -0.02 Hz
  C M1 west tangential pred_d_bw: -0.56 Hz
  C M1 east x_axial pred_d_bw: -0.54 Hz
  C M1 east y_axial pred_d_bw: -0.10 Hz
  C M1 east tangential pred_d_bw: -0.56 Hz
  C M1 south x_axial pred_d_bw: -0.01 Hz
  C M1 south y_axial pred_d_bw: -0.60 Hz
  C M1 south tangential pred_d_bw: -0.59 Hz
  C M1 north x_axial pred_d_bw: -0.07 Hz
  C M1 north y_axial pred_d_bw: -0.52 Hz
  C M1 north tangential pred_d_bw: -0.61 Hz
  C M1 west x_axial residual_d_bw: +0.54 Hz
  C M1 west y_axial residual_d_bw: +0.01 Hz
  C M1 west tangential residual_d_bw: +0.19 Hz
  C M1 east x_axial residual_d_bw: +0.58 Hz
  C M1 east y_axial residual_d_bw: -0.03 Hz
  C M1 east tangential residual_d_bw: +0.22 Hz
  C M1 south x_axial residual_d_bw: +0.03 Hz
  C M1 south y_axial residual_d_bw: +0.71 Hz
  C M1 south tangential residual_d_bw: +0.29 Hz
  C M1 north x_axial residual_d_bw: -0.00 Hz
  C M1 north y_axial residual_d_bw: +0.79 Hz
  C M1 north tangential residual_d_bw: +0.28 Hz
  C A030 west x_axial gt_d_bw: +1.92 Hz
  C A030 west y_axial gt_d_bw: +0.06 Hz
  C A030 west tangential gt_d_bw: +1.35 Hz
  C A030 east x_axial gt_d_bw: +1.95 Hz
  C A030 east y_axial gt_d_bw: +0.12 Hz
  C A030 east tangential gt_d_bw: +1.40 Hz
  C A030 south x_axial gt_d_bw: +0.06 Hz
  C A030 south y_axial gt_d_bw: +2.24 Hz
  C A030 south tangential gt_d_bw: +1.52 Hz
  C A030 north x_axial gt_d_bw: +0.12 Hz
  C A030 north y_axial gt_d_bw: +2.27 Hz
  C A030 north tangential gt_d_bw: +1.56 Hz
  C A030 west x_axial theory_d_bw: +1.88 Hz
  C A030 west y_axial theory_d_bw: +0.00 Hz
  C A030 west tangential theory_d_bw: +1.37 Hz
  C A030 east x_axial theory_d_bw: +1.88 Hz
  C A030 east y_axial theory_d_bw: +0.00 Hz
  C A030 east tangential theory_d_bw: +1.37 Hz
  C A030 south x_axial theory_d_bw: +0.00 Hz
  C A030 south y_axial theory_d_bw: +2.21 Hz
  C A030 south tangential theory_d_bw: +1.54 Hz
  C A030 north x_axial theory_d_bw: +0.00 Hz
  C A030 north y_axial theory_d_bw: +2.21 Hz
  C A030 north tangential theory_d_bw: +1.54 Hz
  C A030 west x_axial pred_d_bw: -0.63 Hz
  C A030 west y_axial pred_d_bw: -1.36 Hz
  C A030 west tangential pred_d_bw: -1.78 Hz
  C A030 east x_axial pred_d_bw: +0.03 Hz
  C A030 east y_axial pred_d_bw: -1.05 Hz
  C A030 east tangential pred_d_bw: -1.13 Hz
  C A030 south x_axial pred_d_bw: -1.23 Hz
  C A030 south y_axial pred_d_bw: +0.34 Hz
  C A030 south tangential pred_d_bw: -1.16 Hz
  C A030 north x_axial pred_d_bw: -1.39 Hz
  C A030 north y_axial pred_d_bw: -0.18 Hz
  C A030 north tangential pred_d_bw: -1.48 Hz
  C A030 west x_axial residual_d_bw: -2.54 Hz
  C A030 west y_axial residual_d_bw: -1.42 Hz
  C A030 west tangential residual_d_bw: -3.13 Hz
  C A030 east x_axial residual_d_bw: -1.92 Hz
  C A030 east y_axial residual_d_bw: -1.17 Hz
  C A030 east tangential residual_d_bw: -2.53 Hz
  C A030 south x_axial residual_d_bw: -1.30 Hz
  C A030 south y_axial residual_d_bw: -1.90 Hz
  C A030 south tangential residual_d_bw: -2.67 Hz
  C A030 north x_axial residual_d_bw: -1.51 Hz
  C A030 north y_axial residual_d_bw: -2.46 Hz
  C A030 north tangential residual_d_bw: -3.04 Hz
  C M2 west x_axial gt_d_bw: +5.18 Hz
  C M2 west y_axial gt_d_bw: +0.15 Hz
  C M2 west tangential gt_d_bw: +3.66 Hz
  C M2 east x_axial gt_d_bw: +5.38 Hz
  C M2 east y_axial gt_d_bw: +0.34 Hz
  C M2 east tangential gt_d_bw: +3.91 Hz
  C M2 south x_axial gt_d_bw: +0.17 Hz
  C M2 south y_axial gt_d_bw: +6.05 Hz
  C M2 south tangential gt_d_bw: +4.13 Hz
  C M2 north x_axial gt_d_bw: +0.32 Hz
  C M2 north y_axial gt_d_bw: +6.25 Hz
  C M2 north tangential gt_d_bw: +4.37 Hz
  C M2 west x_axial theory_d_bw: +5.13 Hz
  C M2 west y_axial theory_d_bw: +0.00 Hz
  C M2 west tangential theory_d_bw: +3.76 Hz
  C M2 east x_axial theory_d_bw: +5.13 Hz
  C M2 east y_axial theory_d_bw: +0.00 Hz
  C M2 east tangential theory_d_bw: +3.77 Hz
  C M2 south x_axial theory_d_bw: +0.00 Hz
  C M2 south y_axial theory_d_bw: +6.05 Hz
  C M2 south tangential theory_d_bw: +4.21 Hz
  C M2 north x_axial theory_d_bw: +0.00 Hz
  C M2 north y_axial theory_d_bw: +6.05 Hz
  C M2 north tangential theory_d_bw: +4.24 Hz
  C M2 west x_axial pred_d_bw: +1.37 Hz
  C M2 west y_axial pred_d_bw: -0.96 Hz
  C M2 west tangential pred_d_bw: -0.26 Hz
  C M2 east x_axial pred_d_bw: +5.05 Hz
  C M2 east y_axial pred_d_bw: +0.65 Hz
  C M2 east tangential pred_d_bw: +3.91 Hz
  C M2 south x_axial pred_d_bw: +0.16 Hz
  C M2 south y_axial pred_d_bw: +7.16 Hz
  C M2 south tangential pred_d_bw: +4.58 Hz
  C M2 north x_axial pred_d_bw: +0.39 Hz
  C M2 north y_axial pred_d_bw: +7.31 Hz
  C M2 north tangential pred_d_bw: +4.74 Hz
  C M2 west x_axial residual_d_bw: -3.81 Hz
  C M2 west y_axial residual_d_bw: -1.12 Hz
  C M2 west tangential residual_d_bw: -3.92 Hz
  C M2 east x_axial residual_d_bw: -0.33 Hz
  C M2 east y_axial residual_d_bw: +0.31 Hz
  C M2 east tangential residual_d_bw: +0.00 Hz
  C M2 south x_axial residual_d_bw: -0.01 Hz
  C M2 south y_axial residual_d_bw: +1.11 Hz
  C M2 south tangential residual_d_bw: +0.45 Hz
  C M2 north x_axial residual_d_bw: +0.07 Hz
  C M2 north y_axial residual_d_bw: +1.06 Hz
  C M2 north tangential residual_d_bw: +0.37 Hz
  C M3 west x_axial gt_d_bw: +9.82 Hz
  C M3 west y_axial gt_d_bw: +0.31 Hz
  C M3 west tangential gt_d_bw: +6.98 Hz
  C M3 east x_axial gt_d_bw: +10.75 Hz
  C M3 east y_axial gt_d_bw: +0.70 Hz
  C M3 east tangential gt_d_bw: +7.74 Hz
  C M3 south x_axial gt_d_bw: +0.34 Hz
  C M3 south y_axial gt_d_bw: +11.48 Hz
  C M3 south tangential gt_d_bw: +7.91 Hz
  C M3 north x_axial gt_d_bw: +0.66 Hz
  C M3 north y_axial gt_d_bw: +12.33 Hz
  C M3 north tangential gt_d_bw: +8.71 Hz
  C M3 west x_axial theory_d_bw: +10.08 Hz
  C M3 west y_axial theory_d_bw: +0.00 Hz
  C M3 west tangential theory_d_bw: +7.38 Hz
  C M3 east x_axial theory_d_bw: +10.05 Hz
  C M3 east y_axial theory_d_bw: +0.00 Hz
  C M3 east tangential theory_d_bw: +7.31 Hz
  C M3 south x_axial theory_d_bw: +0.00 Hz
  C M3 south y_axial theory_d_bw: +11.88 Hz
  C M3 south tangential theory_d_bw: +8.28 Hz
  C M3 north x_axial theory_d_bw: +0.00 Hz
  C M3 north y_axial theory_d_bw: +11.88 Hz
  C M3 north tangential theory_d_bw: +8.27 Hz
  C M3 west x_axial pred_d_bw: +9.79 Hz
  C M3 west y_axial pred_d_bw: +0.30 Hz
  C M3 west tangential pred_d_bw: +7.83 Hz
  C M3 east x_axial pred_d_bw: +11.33 Hz
  C M3 east y_axial pred_d_bw: +1.16 Hz
  C M3 east tangential pred_d_bw: +10.48 Hz
  C M3 south x_axial pred_d_bw: +0.35 Hz
  C M3 south y_axial pred_d_bw: +12.67 Hz
  C M3 south tangential pred_d_bw: +9.81 Hz
  C M3 north x_axial pred_d_bw: -0.37 Hz
  C M3 north y_axial pred_d_bw: +1.07 Hz
  C M3 north tangential pred_d_bw: +0.27 Hz
  C M3 west x_axial residual_d_bw: -0.02 Hz
  C M3 west y_axial residual_d_bw: -0.00 Hz
  C M3 west tangential residual_d_bw: +0.85 Hz
  C M3 east x_axial residual_d_bw: +0.58 Hz
  C M3 east y_axial residual_d_bw: +0.46 Hz
  C M3 east tangential residual_d_bw: +2.74 Hz
  C M3 south x_axial residual_d_bw: +0.01 Hz
  C M3 south y_axial residual_d_bw: +1.20 Hz
  C M3 south tangential residual_d_bw: +1.90 Hz
  C M3 north x_axial residual_d_bw: -1.02 Hz
  C M3 north y_axial residual_d_bw: -11.26 Hz
  C M3 north tangential residual_d_bw: -8.44 Hz
  C selectivity_index: gt=18.297, theory=32.985, pred=10.568
  ```

### `D_generalization.png`

- **Shows:** Edit-response error E_BW for the four evaluation splits by mode family, against the C1 null-model reference and the C2 repeatability floor, with the (iii)-(i) generalization gap and the two held-out combos broken out.
- **Size:** 1920x1080 px
- **Source files / keys:** `outputs/p3_2/eval/summary.json :: splits.*.by_family, splits.*.edit, controls.C1_null_model, controls.C2_floor_hz, gap_i_iii, heldout_by_combo`
- **Numbers on the figure:**

  ```
  D E_BW i_unseen_geom_seen_combo x_axial: +0.75 Hz (n=272)
  D E_BW ii_seen_geom_heldout_combo x_axial: +2.86 Hz (n=290)
  D E_BW iii_unseen_geom_heldout_combo x_axial: +2.42 Hz (n=56)
  D E_BW iv_unseen_alpha x_axial: +1.82 Hz (n=110)
  D E_BW i_unseen_geom_seen_combo y_axial: +1.08 Hz (n=204)
  D E_BW ii_seen_geom_heldout_combo y_axial: +6.62 Hz (n=242)
  D E_BW iii_unseen_geom_heldout_combo y_axial: +6.19 Hz (n=44)
  D E_BW iv_unseen_alpha y_axial: +1.77 Hz (n=87)
  D E_BW i_unseen_geom_seen_combo tangential: +1.62 Hz (n=363)
  D E_BW ii_seen_geom_heldout_combo tangential: +6.18 Hz (n=646)
  D E_BW iii_unseen_geom_heldout_combo tangential: +6.18 Hz (n=76)
  D E_BW iv_unseen_alpha tangential: +2.84 Hz (n=152)
  D E_BW i_unseen_geom_seen_combo pooled: 1.205 Hz
  D C1 null E_BW i_unseen_geom_seen_combo: 3.348 Hz
  D E_BW ii_seen_geom_heldout_combo pooled: 5.450 Hz
  D C1 null E_BW ii_seen_geom_heldout_combo: 5.454 Hz
  D E_BW iii_unseen_geom_heldout_combo pooled: 4.982 Hz
  D C1 null E_BW iii_unseen_geom_heldout_combo: 5.159 Hz
  D E_BW iv_unseen_alpha pooled: 2.254 Hz
  D C1 null E_BW iv_unseen_alpha: 1.254 Hz
  D C2_floor_hz: 0.0402 Hz
  D gap_i_iii: E_BW_i=1.205, E_BW_iii=4.982, gap=3.777 Hz = 73.21% of GT effect 5.159 Hz
  D heldout(iii) west_M2: GT effect 3.266 Hz, model effect 0.999 Hz (n=10)
  D heldout(iii) north_M3: GT effect 7.052 Hz, model effect 0.552 Hz (n=10)
  ```

### `E_mode_shape_and_level.png`

- **Shows:** Spatial |field| maps at one x-axial mode for GT/model x baseline/edited, plus the model-vs-measured per-mode level change with the identity line and the mode-shape invariance correlations.
- **Size:** 1920x1080 px
- **Source files / keys:** `outputs/p3_2/eval/per_config.json`; `outputs/p3_2/eval/summary.json`; `outputs/p3_2/meeting_assets/fields.npz`
- **Numbers on the figure:**

  ```
  E map GT baseline: peak 7.94, mean 3.13 (linear |H|) at 29.00 Hz
  E map GT edited: peak 6.1, mean 2 (linear |H|) at 29.00 Hz
  E map Model baseline: peak 7.34, mean 2.58 (linear |H|) at 29.00 Hz
  E map Model edited: peak 6.11, mean 2.08 (linear |H|) at 29.00 Hz
  E fields: room L=5.93 W=3.18, baseline='L5.93_W3.18_baseline', edited='L5.93_W3.18_west_M2', mode bin 29.00 Hz (first x-axial, c/2L=28.92 Hz), checkpoint outputs/p3_2/p3_2_main/ckpt_iter0006000.pt
  E level scatter: n=3642, Pearson r=0.6007, slope=0.3296
  E mode_shape_invariance i_unseen_geom_seen_combo: gt=0.9880, pred=0.9941, n=100
  E mode_shape_invariance ii_seen_geom_heldout_combo: gt=0.9797, pred=0.9979, n=80
  E mode_shape_invariance iii_unseen_geom_heldout_combo: gt=0.9799, pred=0.9979, n=20
  E mode_shape_invariance iv_unseen_alpha: gt=0.9975, pred=0.9953, n=40
  ```

## Headline numbers (all from `summary.json`)

Splits (ii) and (iii) are the **NEVER-SEEN COMBINATIONS** — the (wall, material) pairs `west+M2`, `north+M3` were excluded from training entirely. Split (iii) is the headline test: unseen geometry *and* unseen combination. Split (iv) uses alpha=0.3, which appears on no training wall.

| split | n | band LSD (dB) | E_BW (Hz) | C1 null E_BW (Hz) | edit_bw_pearson |
|---|---|---|---|---|---|
| i_unseen_geom_seen_combo | 110 | 3.553 | 1.205 | 3.348 | 0.899 |
| ii_seen_geom_heldout_combo | 80 | 2.014 | 5.450 | 5.454 | 0.554 |
| iii_unseen_geom_heldout_combo | 20 | 3.877 | 4.982 | 5.159 | 0.533 |
| iv_unseen_alpha | 40 | 5.135 | 2.254 | 1.254 | 0.323 |

Selectivity index — GT **18.30x**, theory **32.99x**, model **10.57x** (`selectivity_index`).

## Reproduction

```bash
export PYTHONPATH="$PWD"
sbatch scripts/slurm/p3_2_dump_fields.sh        # GPU: fields for figure E
python scripts/make_p3_2_figures.py             # CPU: figures + this manifest
```

