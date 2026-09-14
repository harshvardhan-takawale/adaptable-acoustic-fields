# P4-4 demo v2 — the morph-figure expansion pack

**Rendering only. No training. No new simulation for any room that already had the receivers the
figure needed.** Seven figures, all >= 2560x1440, all zero-shot: unseen shapes, one forward pass
per frame, nothing optimised at demo time, no measurement of any room used.

Decisions **D85-D90**. Full per-frame numbers in
[https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/FIGURE_MANIFEST.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/FIGURE_MANIFEST.md).

## The figures

| # | figure | swept | mean r | ckpt |
|---|---|---|---|---|
| 0 | [figN_morph](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/figN_morph.png) | notch DEPTH (title corrected) | +0.9487 | p4_3_D_data |
| 1 | [s1_notch_width](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s1_notch_width.png) | notch WIDTH 0.30-2.70 m | **+0.9511** | p4_3_D_data |
| 2 | [s2_room_width](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s2_room_width.png) | ROOM width 4.00-5.50 m | +0.8871 | p4_3_D_data |
| 3a | [s3a_rect_L_U](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s3a_rect_L_U.png) | rect -> L -> U, two-phase | +0.7850 | p4_4_FAM |
| 3b | [s3b_rect_L_DN](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s3b_rect_L_DN.png) | rect -> L -> DN, two-phase | +0.7861 | p4_4_FAM |
| 3c | [s3c_rect_L_U_fav](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s3c_rect_L_U_fav.png) | rect -> L -> U, strong region | +0.7351 | p4_4_FAM |
| 4 | [gallery](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_gallery.png) | 5 families side by side | +0.7707 | p4_4_FAM |
| 5 | [s1 mode2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s1_notch_width_mode2.png) | s1 at mode (1,1) 44.6 Hz | +0.9511 | p4_3_D_data |

## What is new science rather than new pictures

**1. Accuracy is FLAT across a notch-width sweep while the field is transformed.** s1 holds the
bounding box and the depth and sweeps the width 0.30 -> 2.70 m: mean **+0.9511**, spread **0.038**,
while the mode-(0,1) null goes from a compact point to a full diagonal nodal line and rotates.
NLOS climbs 0.0% -> 4.1%, so the flatness is not a shadow-free artefact. It holds at a second
mode (fig 5), so it is not a property of one bin.

**2. The U weakness is about PLACEMENT and DEPTH, not corner count.** s3a and s3b share a
bounding box, both notch widths, both depths, the checkpoint and the receiver sampling, and
differ only in WHERE the second notch sits. Their first four frames are the same rooms and
reproduce to four decimals. Then:

| second-notch depth | 0.38 | 0.76 | 1.12 | 1.50 |
|---|---|---|---|---|
| U (NW+NE, same wall) | +0.7866 | +0.7898 | +0.7564 | **+0.6713** |
| DN (NW+SE, opposite) | +0.7482 | +0.7469 | +0.7526 | **+0.7652** |

The U degrades as the second notch deepens; the DN does not. At full depth they are **0.094**
apart with the draw removed by construction. The per-family held-out means (U +0.673, DN +0.704)
confound placement with the draw; this does not. Two notches narrowing the SAME wall -- the U's
north stem is 2.00 m -- is the hard case. In s3a the second corner costs nothing when it
APPEARS (+0.7488 -> +0.7866); the cost arrives as it DEEPENS.

**3. Sparse receiver sampling FLATTERS the model, by more than the mode count does.** The five
gallery rooms, scored on the corpus's 800 scattered receivers against ~3.2k on a 0.08 m grid,
same geometry and same checkpoint:

| room | 800 rx | dense | delta |
|---|---|---|---|
| rectangle | +0.9605 | +0.7789 | **-0.18** |
| L | +0.8596 | +0.8232 | -0.04 |
| T | +0.9281 | +0.7083 | **-0.22** |
| double-notch | +0.9165 | +0.8381 | -0.08 |
| U | +0.8967 | +0.7052 | **-0.19** |

A sparse scatter samples the smooth interior and under-samples the near-wall and near-corner
field where the model is weakest. **This is a live threat to any accuracy number this project
reports**, and it is larger than the 3-mode/6-mode protocol gap.

**4. The training-range edge is visible in a single frame.** s2's W=4.00 column scores +0.6988
against +0.88-0.93 for every other column, and `W_RANGE = (4.0, 5.5)` -- that frame sits exactly
ON the lower bound. The prediction loses the crisp nodal line the FDTD has and recovers it one
step inside the range.

## Corrections made this chunk

* **D85** `figN_morph` printed `notch width 0.00 m` because the title read `w` off frame 0, which
  is the RECTANGLE and has no notch width. Fixed; no published number changed.
* **D87 retracted its own first version.** It claimed the 3-mode protocol "reads higher" than the
  6-mode one. s3c's terminal frame refutes it: the same room reads +0.7968 (family eval) and
  +0.7052 here. Two things differ, not one -- mode count AND receiver set. Incomparable in either
  direction.
* **D89** the render cache was keyed on checkpoint + filenames but NOT the data directory, and
  the gallery rooms exist in two corpora at different densities under identical filenames. It
  missed publication only because the stale cache was deleted by hand.
* **D90** marker size scaled per frame by 1/n_rx, drawing the worst-scoring columns with the
  fattest dots. Now one size per row from the receiver pitch.

D89 and D90 were found by an adversarial review of the pipeline, not by anything failing.

## Cost

36 unique FDTD rooms for the sweeps (15-73 s each) plus 4 for the dense gallery; 5 shared between
s3a/s3b and 1 between s3c and the gallery, deduped by filename. Renders ~2-3 min/frame. No
training. `data/track_p4_4_family` was NOT overwritten -- the published family eval still stands
on its own 800-receiver corpus.

## Validation

* `rectL6.00_W5.00` built by the new multi-notch builder has **4320** receivers; figN's d_hat=0
  room has **4322** -- exactly the two fixed probes figN adds and this builder drops.
* Every room passed the node-for-node `polygon_contains` cross-check and the D65 extent assert.
* s3a and s3b's four shared frames reproduce to four decimals across separate jobs; the U room
  reads +0.7052 in both s3c and the gallery from separate jobs (D49 C3 eval mode holds).
* Every figure asserts >= 2560x1440 and non-trivial ink at save time; all seven URLs return 200.
