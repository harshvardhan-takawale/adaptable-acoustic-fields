# P4-4 demo v2 -- morph-figure expansion pack

Every figure below is **zero-shot**: the rooms are unseen shapes, each frame is **one forward
pass**, nothing is optimised at demo time, and **no measurement of any of these rooms** is used
-- the model is given the boundary polygon and nothing else.

## How to read the numbers

`spatial Pearson` is the correlation between predicted and FDTD |H| in dB across all receivers,
**averaged over the first 3 resolvable modes of each frame's own bounding box**. That is
`figN_morph`'s protocol, kept so a v2 strip can be read beside it. **Gate 2, the P4-2/P4-4
sweep curves and `p4_4_family_eval` use 6 modes**, so no number here is comparable to one of
theirs and **none of it may be quoted against a gate threshold**. The difference is not a fixed
offset in a known direction: on the P4-3 demo rooms the 3-mode number read above the gate, while
`s3c`'s terminal frame -- the same room the family evaluator scored +0.7968 -- reads +0.7052
here. Two things differ there, not one: the mode count, and the receiver set (800 scattered
points in the family corpus against ~2.6k on a 0.08 m grid).

`R(shown)` is the correlation at the single mode the figure actually draws. `band LSD` is the
mean absolute dB error over **every** bin in 0-300 Hz and every receiver -- a whole-band number,
not a modal one, which is why it is several dB even where the spatial correlation is high.

`enumerate_modes` is the analytic mode list of the **bounding box** (D79). Prediction and truth
are always read at the same bin, so a bin that misses a true resonance of a non-convex room
costs sensitivity; it cannot flatter the model.

Ground truth is 2D FDTD at `dx = 0.02 m`, `fs = 30720 Hz`, band 0-300 Hz at `df = 0.5 Hz`.
Receivers are on a 0.08 m grid with 0.12 m of wall clearance and 0.15 m cleared around **every**
reflex corner.


## fig 0 -- `figN_morph` (the original, title corrected)

https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/figN_morph.png

* **Swept**: notch DEPTH, d_hat 0.00 -> 1.00 (a rectangle to the deepest notch)
* **Fixed**: L = 6.00 m, W = 5.00 m, notch width 2.00 m
* **Mode**: (0,1) at 34.3 Hz
* **Checkpoint**: `outputs/p4_3/p4_3_D_data/ckpt_iter0060000.pt`

The published title read `notch width 0.00 m`. `fig_morph` took the width from frame 0, and frame 0 of a full-range morph is `d = 0` -- a pure rectangle, stored with `w = 0.0`. Every notched frame carries `w = 2.00`. Only the title changed; the render cache was intact, so the numbers are the published ones.

| frame (d_hat) | notch depth d (m) | n_rx | spatial r (3-mode) | band LSD dB |
|---|---|---|---|---|
| 0.000 | 0.00 | 4322 | **+0.9802** | 3.49 |
| 0.160 | 0.36 | 4195 | **+0.9565** | 4.32 |
| 0.267 | 0.60 | 4120 | **+0.9542** | 4.31 |
| 0.418 | 0.94 | 4020 | **+0.9155** | 4.63 |
| 0.578 | 1.30 | 3918 | **+0.8939** | 4.70 |
| 0.738 | 1.66 | 3795 | **+0.9606** | 4.60 |
| 0.844 | 1.90 | 3720 | **+0.9682** | 4.65 |
| 0.996 | 2.24 | 3619 | **+0.9605** | 3.96 |

## fig 1 -- `s1_notch_width`

https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s1_notch_width.png

* **Swept**: notch width w, 0.30 -> 2.70 m
* **Fixed**: L = 6.00 m, W = 5.00 m, notch depth d = 1.50 m (d_hat = 0.667)
* **Checkpoint**: `outputs/p4_3/p4_3_D_data/ckpt_iter0060000.pt` (iter 60000)
* **Mode drawn**: index 1 of the bounding box's analytic list

| frame | tokens | n_rx | NLOS % | spatial r (3-mode) | r (shown mode) | band LSD dB |
|---|---|---|---|---|---|---|
| w=0.30 | 6 | 4243 | 0.0 | **+0.9503** | +0.9233 | 4.32 |
| w=0.64 | 6 | 4166 | 0.0 | **+0.9645** | +0.9584 | 4.19 |
| w=0.98 | 6 | 4073 | 0.3 | **+0.9713** | +0.9621 | 4.27 |
| w=1.32 | 6 | 3997 | 1.0 | **+0.9591** | +0.9539 | 4.31 |
| w=1.68 | 6 | 3919 | 1.9 | **+0.9332** | +0.9102 | 4.50 |
| w=2.02 | 6 | 3826 | 2.4 | **+0.9395** | +0.9274 | 4.36 |
| w=2.36 | 6 | 3750 | 3.2 | **+0.9561** | +0.9774 | 4.52 |
| w=2.70 | 6 | 3673 | 4.1 | **+0.9348** | +0.9322 | 5.07 |

Mean spatial r **+0.9511**, mean band LSD **4.44 dB** over the 8 frames.

## fig 2 -- `s2_room_width`

https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s2_room_width.png

* **Swept**: room width W, 4.00 -> 5.50 m
* **Fixed**: L = 6.00 m, notch d = 1.20 m and w = 2.00 m held in METRES, so d_hat falls 0.667 -> 0.485 as a consequence of the size, not as a second knob
* **Checkpoint**: `outputs/p4_3/p4_3_D_data/ckpt_iter0060000.pt` (iter 60000)
* **Mode drawn**: index 1 of the bounding box's analytic list

| frame | tokens | n_rx | NLOS % | mode | spatial r (3-mode) | r (shown mode) | band LSD dB |
|---|---|---|---|---|---|---|---|
| W=4.00 | 6 | 3007 | 3.1 | (0,1) 42.9 Hz | **+0.6988** | +0.6794 | 5.58 |
| W=4.22 | 6 | 3223 | 2.4 | (0,1) 40.6 Hz | **+0.9199** | +0.9330 | 4.92 |
| W=4.42 | 6 | 3437 | 2.2 | (0,1) 38.8 Hz | **+0.9209** | +0.8644 | 4.40 |
| W=4.64 | 6 | 3583 | 1.6 | (0,1) 37.0 Hz | **+0.9173** | +0.9125 | 4.11 |
| W=4.86 | 6 | 3799 | 1.3 | (0,1) 35.3 Hz | **+0.8940** | +0.8701 | 4.72 |
| W=5.08 | 6 | 4014 | 1.2 | (0,1) 33.8 Hz | **+0.8829** | +0.9348 | 5.02 |
| W=5.28 | 6 | 4159 | 0.9 | (0,1) 32.5 Hz | **+0.9290** | +0.9614 | 4.91 |
| W=5.50 | 6 | 4375 | 0.8 | (0,1) 31.2 Hz | **+0.9339** | +0.8982 | 4.62 |

Mean spatial r **+0.8871**, mean band LSD **4.79 dB** over the 8 frames.

## fig 3 -- `s3a_rect_L_U`

https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s3a_rect_L_U.png

* **Swept**: notch DEPTH, in TWO phases: frames 1-4 grow the NW notch (rect -> L), frames 5-8 grow the NE notch (L -> U). Two knobs move, one at a time -- this is not a single-parameter sweep
* **Fixed**: L = 6.00 m, W = 5.00 m, NW width 2.00 m, NE width 2.00 m; boundary tokens 4 -> 6 -> 8
* **Checkpoint**: `outputs/p4_4/p4_4_FAM/ckpt_iter0060000.pt` (iter 60000)
* **Mode drawn**: index 1 of the bounding box's analytic list

| frame | tokens | n_rx | NLOS % | spatial r (3-mode) | r (shown mode) | band LSD dB |
|---|---|---|---|---|---|---|
| NW d=0.00 | 4 | 4320 | 0.0 | **+0.8400** | +0.7674 | 6.13 |
| NW d=0.50 | 6 | 4166 | 0.0 | **+0.8194** | +0.7089 | 6.15 |
| NW d=1.00 | 6 | 3993 | 0.7 | **+0.8675** | +0.8972 | 6.17 |
| NW d=1.50 | 6 | 3843 | 2.6 | **+0.7488** | +0.8084 | 6.37 |
| NE d=0.38 | 8 | 3718 | 2.7 | **+0.7866** | +0.8953 | 6.31 |
| NE d=0.76 | 8 | 3593 | 2.8 | **+0.7898** | +0.8999 | 6.33 |
| NE d=1.12 | 8 | 3493 | 2.8 | **+0.7564** | +0.7458 | 6.53 |
| NE d=1.50 | 8 | 3368 | 2.9 | **+0.6713** | +0.5312 | 6.59 |

Mean spatial r **+0.7850**, mean band LSD **6.32 dB** over the 8 frames.

## fig 4 -- `s3b_rect_L_DN`

https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s3b_rect_L_DN.png

* **Swept**: notch DEPTH, in TWO phases: frames 1-4 grow the NW notch (rect -> L), frames 5-8 grow the SE notch (L -> DN). Two knobs move, one at a time -- this is not a single-parameter sweep
* **Fixed**: L = 6.00 m, W = 5.00 m, NW width 2.00 m, SE width 2.00 m; boundary tokens 4 -> 6 -> 8
* **Checkpoint**: `outputs/p4_4/p4_4_FAM/ckpt_iter0060000.pt` (iter 60000)
* **Mode drawn**: index 1 of the bounding box's analytic list

| frame | tokens | n_rx | NLOS % | spatial r (3-mode) | r (shown mode) | band LSD dB |
|---|---|---|---|---|---|---|
| NW d=0.00 | 4 | 4320 | 0.0 | **+0.8400** | +0.7674 | 6.13 |
| NW d=0.50 | 6 | 4166 | 0.0 | **+0.8194** | +0.7089 | 6.15 |
| NW d=1.00 | 6 | 3993 | 0.7 | **+0.8675** | +0.8972 | 6.17 |
| NW d=1.50 | 6 | 3843 | 2.6 | **+0.7488** | +0.8084 | 6.37 |
| SE d=0.38 | 8 | 3716 | 2.7 | **+0.7482** | +0.9109 | 6.47 |
| SE d=0.76 | 8 | 3593 | 2.8 | **+0.7469** | +0.8420 | 6.43 |
| SE d=1.12 | 8 | 3490 | 2.8 | **+0.7526** | +0.7879 | 6.45 |
| SE d=1.50 | 8 | 3366 | 2.9 | **+0.7652** | +0.7666 | 6.46 |

Mean spatial r **+0.7861**, mean band LSD **6.33 dB** over the 8 frames.

## fig 5 -- `s3c_rect_L_U_fav`

https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s3c_rect_L_U_fav.png

* **Swept**: notch DEPTH, in TWO phases: frames 1-4 grow the NW notch (rect -> L), frames 5-8 grow the NE notch (L -> U). Two knobs move, one at a time -- this is not a single-parameter sweep
* **Fixed**: L = 5.70 m, W = 4.02 m, NW width 0.62 m, NE width 2.22 m; boundary tokens 4 -> 6 -> 8
* **Checkpoint**: `outputs/p4_4/p4_4_FAM/ckpt_iter0060000.pt` (iter 60000)
* **Mode drawn**: index 1 of the bounding box's analytic list

| frame | tokens | n_rx | NLOS % | spatial r (3-mode) | r (shown mode) | band LSD dB |
|---|---|---|---|---|---|---|
| NW d=0.00 | 4 | 3312 | 0.0 | **+0.8132** | +0.7767 | 6.21 |
| NW d=0.22 | 6 | 3287 | 0.0 | **+0.8064** | +0.7450 | 6.38 |
| NW d=0.46 | 6 | 3263 | 0.0 | **+0.7865** | +0.7021 | 6.16 |
| NW d=0.68 | 6 | 3239 | 0.0 | **+0.7591** | +0.6700 | 5.94 |
| NE d=0.44 | 8 | 3071 | 0.0 | **+0.6654** | +0.4955 | 6.10 |
| NE d=0.88 | 8 | 2930 | 0.0 | **+0.6623** | +0.5673 | 6.16 |
| NE d=1.30 | 8 | 2763 | 0.0 | **+0.6826** | +0.7196 | 6.20 |
| NE d=1.74 | 8 | 2623 | 0.0 | **+0.7052** | +0.8240 | 6.33 |

Mean spatial r **+0.7351**, mean band LSD **6.18 dB** over the 8 frames.

## fig 6 -- `gallery`

https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_gallery.png

* **Swept**: nothing -- this is a gallery, not a sweep: five different room FAMILIES (rectangle, L, T, double-notch, U) side by side, each drawn at its own room's mode (0,1)
* **Fixed**: five HELD-OUT rooms, one per family, chosen for comparable bounding boxes (L 5.30-5.72 m, W 4.02-4.64 m) and a visible notch. The geometry is the family corpus's; the FDTD was re-run at figN's RECEIVER density (0.08 m grid, ~3-4k points) instead of the corpus's 800 scattered points, so the panels resolve the field. Same solver, same dx, same source -- only the sampling changed
* **Checkpoint**: `outputs/p4_4/p4_4_FAM/ckpt_iter0060000.pt` (iter 60000)
* **Mode drawn**: index 1 of the bounding box's analytic list

* **Note**: This is a SELECTED set: 5 of the 50 held-out family rooms, picked for comparable bounding boxes and a visible notch, NOT the average room. For context, the family evaluator's means over all 50 are rect +0.847, L +0.766, T +0.758, DN +0.704, U +0.673 -- but those use its SIX-mode protocol and the per-panel r above uses figN's THREE-mode one. The two are NOT comparable in either direction, and neither is a gate result.

| frame | tokens | n_rx | NLOS % | mode | spatial r (3-mode) | r (shown mode) | band LSD dB |
|---|---|---|---|---|---|---|---|
| rectangle | 4 | 3400 | 0.0 | (0,1) 41.0 Hz | **+0.7789** | +0.8417 | 6.16 |
| L | 6 | 3225 | 1.7 | (0,1) 37.9 Hz | **+0.8232** | +0.9000 | 6.32 |
| T | 8 | 3695 | 0.1 | (0,1) 37.0 Hz | **+0.7083** | +0.5207 | 6.52 |
| double-notch | 8 | 3162 | 0.3 | (0,1) 38.1 Hz | **+0.8381** | +0.9047 | 6.28 |
| U | 8 | 2623 | 0.0 | (0,1) 42.7 Hz | **+0.7052** | +0.8240 | 6.33 |

Mean spatial r **+0.7707**, mean band LSD **6.32 dB** over the 5 frames.
