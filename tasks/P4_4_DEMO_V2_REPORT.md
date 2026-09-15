# P4-4 demo v2 — morph-figure expansion pack: FULL REPORT

**Date**: 2026-09-14 · **Status**: COMPLETE · **Scope**: rendering only — no training, no new
architecture, no gate re-run. **Decisions added**: D85–D91.

All links below are full raw-GitHub URLs and were each verified to return HTTP 200.

---

## 1. What was asked and what was delivered

The task was to produce more sweeps in the exact visual format of `figN_morph.png` — predicted
row above, FDTD row below, one parameter varying left to right, everything else fixed, per-frame
spatial Pearson printed above each column, one shared colour scale per row — plus a 2-minute fix
to a variable-printing bug in `figN_morph.png`'s title. Time-boxed; two of five figures would
have counted as success.

**All five sweeps were delivered, plus the optional sixth and seventh items.** 8 of 8 deliverables.

| # | deliverable | asked | delivered |
|---|---|---|---|
| 0 | figN title fix | required | yes |
| 1 | notch WIDTH sweep | priority 1 | yes |
| 2 | room WIDTH sweep | priority 2 | yes |
| 3a | rect → L → U | priority 3 | yes |
| 3b | U-weakness variant (family swap) | "do all three in parallel" | yes (rect → L → **DN**) |
| 3c | U-weakness variant (favourable params) | "do all three in parallel" | yes |
| 4 | family gallery | priority 4 | yes |
| 5 | second mode for sweep 1 | "only if time remains" | yes |

Every figure is ≥ 2560×1440 and carries the required caption: zero-shot, unseen shapes, one
forward pass per frame, no measurements, and the name of what is swept.

---

## 2. The figures

Manifest with every per-frame number:
https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/FIGURE_MANIFEST.md

| # | figure | swept | mean r | mean LSD | px |
|---|---|---|---|---|---|
| 0 | figN_morph (corrected) | notch depth | +0.9487 | 4.33 dB | 3790×1135 |
| 1 | s1_notch_width | notch width 0.30→2.70 m | **+0.9511** | 4.44 dB | 4555×1559 |
| 2 | s2_room_width | room width 4.00→5.50 m | +0.8871 | 4.79 dB | 4591×1645 |
| 3a | s3a_rect_L_U | depth, two-phase → U | +0.7850 | 6.32 dB | 5524×1558 |
| 3b | s3b_rect_L_DN | depth, two-phase → DN | +0.7861 | 6.33 dB | 5542×1558 |
| 3c | s3c_rect_L_U_fav | depth, two-phase → U (strong region) | +0.7351 | 6.18 dB | 5881×1559 |
| 4 | fig_gallery | five families side by side | +0.7707 | 6.32 dB | 4531×1898 |
| 5 | s1_notch_width_mode2 | as #1, at mode (1,1) | +0.9511 | 4.44 dB | 4555×1559 |

**Direct links:**

* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/figN_morph.png
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s1_notch_width.png
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s2_room_width.png
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s3a_rect_L_U.png
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s3b_rect_L_DN.png
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s3c_rect_L_U_fav.png
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_gallery.png
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/fig_s1_notch_width_mode2.png

---

## 3. HOW TO READ THESE NUMBERS — read this before quoting any of them

**`spatial Pearson` here is averaged over the first 3 resolvable modes of each frame's own
bounding box.** That is `figN_morph`'s protocol, kept so a v2 strip can be laid beside it.
**Gate 2, the P4-2/P4-4 sweep curves and `p4_4_family_eval` all use 6 modes.**

**No number in this pack may be quoted against a gate threshold.** Three things differ between a
v2 number and a family-eval number, and they do not share a direction:

1. **Mode count** — 3 here, 6 there.
2. **Receiver set** — see §5.3; this is the larger effect.
3. **Which rooms** — several v2 rooms are newly simulated and appear in no corpus.

D87 originally claimed the 3-mode protocol "reads higher". **That claim is retracted.** `s3c`'s
terminal frame is the same room `p4_4_family_eval` scored **+0.7968**, and it reads **+0.7052**
here. The offset has no fixed direction and cannot be corrected for.

`band LSD` is the mean absolute dB error over **every** bin in 0–300 Hz and every receiver — a
whole-band number, not a modal one, which is why it is several dB even where spatial r is high.

`enumerate_modes` is the analytic mode list of the **bounding box** (D79). Prediction and truth
are always read at the same bin, so a bin that misses a true resonance of a non-convex room costs
sensitivity; it cannot flatter the model.

---

## 4. Per-figure detail

### Figure 0 — figN_morph, title corrected
`p4_3_D_data` iter 60000 · mode (0,1) 34.3 Hz · L = 6.00 m, W = 5.00 m, notch width 2.00 m

| d̂ | 0.000 | 0.160 | 0.267 | 0.418 | 0.578 | 0.738 | 0.844 | 0.996 |
|---|---|---|---|---|---|---|---|---|
| d (m) | 0.00 | 0.36 | 0.60 | 0.94 | 1.30 | 1.66 | 1.90 | 2.24 |
| r | +0.9802 | +0.9565 | +0.9542 | +0.9155 | +0.8939 | +0.9606 | +0.9682 | +0.9605 |
| LSD dB | 3.49 | 4.32 | 4.31 | 4.63 | 4.70 | 4.60 | 4.65 | 3.96 |

The published title read `notch width 0.00 m`. **Only the title changed** — the render cache was
intact, so every number above is the previously published one.

### Figure 1 — notch WIDTH sweep (the direct figN complement)
`p4_3_D_data` · mode (0,1) 34.3 Hz · **fixed** L = 6.00 m, W = 5.00 m, notch **depth** 1.50 m (d̂ = 0.667)

| w (m) | 0.30 | 0.64 | 0.98 | 1.32 | 1.68 | 2.02 | 2.36 | 2.70 |
|---|---|---|---|---|---|---|---|---|
| r | +0.9503 | +0.9645 | +0.9713 | +0.9591 | +0.9332 | +0.9395 | +0.9561 | +0.9348 |
| LSD dB | 4.32 | 4.19 | 4.27 | 4.31 | 4.50 | 4.36 | 4.52 | 5.07 |
| n_rx | 4243 | 4166 | 4073 | 3997 | 3919 | 3826 | 3750 | 3673 |
| NLOS % | 0.0 | 0.0 | 0.3 | 1.0 | 1.9 | 2.4 | 3.2 | 4.1 |

Built on figN's **exact** bounding box, checkpoint, mode and receiver sampling, so the two strips
are directly comparable: figN grows the corner downward, this grows it sideways.

### Figure 2 — room WIDTH sweep
`p4_3_D_data` · mode (0,1), **31.2–42.9 Hz across the row** · **fixed** L = 6.00 m, notch d = 1.20 m
and w = 2.00 m held in **metres**

| W (m) | 4.00 | 4.22 | 4.42 | 4.64 | 4.86 | 5.08 | 5.28 | 5.50 |
|---|---|---|---|---|---|---|---|---|
| r | +0.6988 | +0.9199 | +0.9209 | +0.9173 | +0.8940 | +0.8829 | +0.9290 | +0.9339 |
| LSD dB | 5.58 | 4.92 | 4.40 | 4.11 | 4.72 | 5.02 | 4.91 | 4.62 |
| mode f | 42.9 | 40.6 | 38.8 | 37.0 | 35.3 | 33.8 | 32.5 | 31.2 |
| n_rx | 3007 | 3223 | 3437 | 3583 | 3799 | 4014 | 4159 | 4375 |

Two caveats that the caption carries and any slide must repeat:
* **d̂ falls 0.667 → 0.485** across the row although the notch never changes, because it is
  normalised by a growing W. A consequence of the size edit, not a second knob.
* **Mode (0,1) sits at c/2W**, so it moves 43 → 31 Hz. Each column is read at **its own** bin.

### Figures 3a / 3b — the placement control (see §5.2)
### Figure 3c — rect → L → U in the family model's strongest region
`p4_4_FAM` · mode (0,1) 42.7 Hz · L = 5.70 m, W = 4.02 m, NW width 0.62 m, NE width 2.22 m

| frame | rect | NW 0.22 | NW 0.46 | NW 0.68 | NE 0.44 | NE 0.88 | NE 1.30 | NE 1.74 |
|---|---|---|---|---|---|---|---|---|
| r | +0.8132 | +0.8064 | +0.7865 | +0.7591 | +0.6654 | +0.6623 | +0.6826 | +0.7052 |
| tokens | 4 | 6 | 6 | 6 | 8 | 8 | 8 | 8 |

Terminal frame is the best-scoring held-out U room in the family test set.

### Figure 4 — family gallery
`p4_4_FAM` · mode (0,1), 37.0–42.7 Hz · five HELD-OUT rooms, L 5.30–5.72 m, W 4.02–4.64 m

| | rectangle | L | T | double-notch | U |
|---|---|---|---|---|---|
| r | +0.7789 | +0.8232 | +0.7083 | +0.8381 | +0.7052 |
| LSD dB | 6.16 | 6.32 | 6.52 | 6.28 | 6.33 |
| tokens | 4 | 6 | 8 | 8 | 8 |
| n_rx | 3400 | 3225 | 3695 | 3162 | 2623 |

**This is a SELECTED set** — 5 of the 50 held-out family rooms, chosen for comparable bounding
boxes and a visible notch. It is not the average room, and the figure says so.

### Figure 5 — sweep 1 at a second mode
Identical rooms and cache, drawn at mode **(1,1) 44.6 Hz** instead of (0,1) 34.3 Hz. Per-frame r
at the drawn mode: +0.9393, +0.9469, +0.9629, +0.9403, +0.9122, +0.9018, +0.9046, +0.9133. The
3-mode averages are by construction identical to figure 1's.

---

## 5. Findings the manager should plan against

### 5.1 Accuracy is FLAT across a notch-width sweep while the field is transformed
Sweep 1: **mean +0.9511, spread 0.038** over w = 0.30 → 2.70 m, while the mode-(0,1) null goes
from a compact point to a full diagonal nodal line and rotates. The NLOS fraction climbs
**0.0% → 4.1%**, so the flatness is not a shadow-free artefact, and it **holds at a second mode**
(figure 5), so it is not a property of one bin.

### 5.2 The U weakness is PLACEMENT and DEPTH, not corner count
`s3a` and `s3b` share the bounding box, both notch widths, both depths, the checkpoint and the
receiver sampling. They differ **only in which corner the second notch occupies**. Their first
four frames are literally the same rooms and reproduce to four decimals.

| second-notch depth (m) | 0.38 | 0.76 | 1.12 | 1.50 |
|---|---|---|---|---|
| **U** (NW+NE, same wall) | +0.7866 | +0.7898 | +0.7564 | **+0.6713** |
| **DN** (NW+SE, opposite corners) | +0.7482 | +0.7469 | +0.7526 | **+0.7652** |

The U **degrades** as the second notch deepens; the DN does not. At full depth they are **0.094
apart with the draw removed by construction**. The per-family held-out means (U +0.673, DN
+0.704) confound placement with the draw; this comparison does not.

Two notches narrowing the **same wall** — the U's north stem is L − 2.00 − 2.00 = 2.00 m — is the
hard case. In `s3a` the second corner costs nothing when it **appears** (+0.7488 → +0.7866); the
cost arrives as it **deepens**.

**Implication for the Z-corridor**: a Z has three reflex corners and narrow stems. This says the
stem geometry, not the corner count, is what to stress-test next.

### 5.3 Sparse receiver sampling FLATTERS the model — by more than the mode count does
The five gallery rooms, identical geometry and checkpoint, scored on the corpus's 800 scattered
receivers versus ~3.2k on a 0.08 m grid:

| room | 800 rx | dense | Δ |
|---|---|---|---|
| rectangle | +0.9605 | +0.7789 | **−0.18** |
| L | +0.8596 | +0.8232 | −0.04 |
| T | +0.9281 | +0.7083 | **−0.22** |
| double-notch | +0.9165 | +0.8381 | −0.08 |
| U | +0.8967 | +0.7052 | **−0.19** |

A sparse scatter samples the smooth interior and under-samples the near-wall and near-corner
field where the model is weakest. **This threatens every accuracy number this project reports on
an 800-receiver corpus, including Gate 2 and the family eval, and it is larger than the
3-mode/6-mode protocol gap.** Recommend settling it before the next gate: re-score the frozen
test shapes at dense receivers and report both.

### 5.4 The training-range edge is visible in a single frame
Sweep 2's W = 4.00 column reads **+0.6988** against +0.88–0.93 for every other column, and
`W_RANGE = (4.0, 5.5)` — that frame sits exactly **on** the lower bound. The prediction loses the
crisp nodal line the FDTD has and recovers it one step inside the range. A clean, cheap
extrapolation-edge demonstration.

---

## 6. Corrections and defects found this chunk (D85–D91)

Full text: https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md

| id | defect | reached a published figure? |
|---|---|---|
| **D85** | `figN_morph` printed `notch width 0.00 m` — the title read `w` off frame 0, which is the RECTANGLE and has no notch width | **yes**, for two chunks; fixed |
| **D86** | (justification, not a defect) sweeps are new FDTD because no corpus contains a sweep — largest group sharing `(L,W,d)` holds 2 distinct widths, a sweep needs 8 | n/a |
| **D87** | claimed the 3-mode protocol "reads higher" than the 6-mode one — **retracted**, refuted by `s3c`'s terminal frame | no; caught before the manifest shipped |
| **D88** | node `legacy43` killed 24/24 array tasks at launch with empty logs while every other node passed | no; cost time only |
| **D89** | render cache keyed on checkpoint + filenames but **not the data directory**; the same room filename exists in two corpora at different densities | **no — by luck**, the stale cache was deleted by hand |
| **D90** | marker size scaled per frame by 1/n_rx, drawing the worst-scoring columns with the fattest dots | yes, in the first push; re-plotted |
| **D91** | gallery caption **asserted** a receiver density chosen by a flag defaulting to the wrong corpus; and the blank-figure guard was set at 2% ink when an all-NaN figure measures **6.8%** | no; the correct path was passed by hand |

**D89, D90 and D91 were found by an adversarial multi-agent review of the figure pipeline, not by
anything failing.** Three of the four are the silent kind: the figure renders, the pipeline exits
0, and the caption is false.

Two structural fixes, not one-off corrections:
* **Captions now MEASURE.** Every figure appends "Receivers actually drawn: N–M per panel",
  computed from the rooms it is drawing. A caption that counts cannot disagree with its arrays.
* **The blank guard now checks the ARRAYS** — at least half of each frame's values finite, per
  frame and per row, naming what failed — with the ink threshold kept as a backstop and raised to
  15%. Both were verified to **fire** on a deliberately NaN-ed frame before being trusted.

---

## 7. Validation performed

* **The new multi-notch builder reproduces the old one.** `rectL6.00_W5.00` built by
  `build_p4_4_sweep_rooms.py` has **4320** receivers; figN's d̂ = 0 room has **4322** — exactly the
  two fixed probes figN adds and this builder deliberately drops.
* Every room passed the per-room **node-for-node** `polygon_contains` cross-check against the
  solver's own air mask, and the D65 extent assert.
* Receiver counts move the right way in every sweep: s1 falls 4243 → 3673 as the notch widens;
  s2 rises 3007 → 4375 as the room grows; the U and DN rows match to ~2 receivers, as they must
  since the same area is removed from a different corner.
* **Determinism across jobs**: s3a and s3b's four shared frames reproduce to four decimals from
  separate SLURM jobs; the U room reads +0.7052 in both `s3c` and the gallery from separate jobs
  (D49 C3 eval mode holds).
* Every figure asserts ≥ 2560×1440 and non-trivial ink at save time; the gallery default-corpus
  fix was proved by re-running with **no** `--family-dir` flag and confirming 2623–3695 receivers.
* All figure and document URLs verified HTTP 200.

---

## 8. Cost and artefacts

* **40 FDTD rooms**: 36 for the sweeps + 4 new for the dense gallery (1 of the 5 was already
  built as `s3c`'s terminal frame). 15–73 s each, run as parallel arrays. 5 rooms shared between
  s3a/s3b, deduped by filename so they are simulated once and the two figures remain a controlled
  comparison.
* **No training.** No checkpoint was produced or modified.
* `data/track_p4_4_family` was **NOT** overwritten — the published family eval still stands on its
  own 800-receiver corpus. The dense re-simulations live in `data/track_p4_4_sweeps/`.
* Checkpoints used, both pre-existing at iter 60000: `outputs/p4_3/p4_3_D_data` (single-notch
  sweeps 1, 2 and figN) and `outputs/p4_4/p4_4_FAM` (multi-notch sweeps 3a/3b/3c and the gallery).

**Code:**
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_4_sweeps.py
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/build_p4_4_sweep_rooms.py
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_4_sweep_figures_v2.py
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_4_figure_manifest.py
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/build_p4_4_sweeps_common.py

**Docs:**
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_4/demo/v2/FIGURE_MANIFEST.md
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_P4_4_DEMO_V2_RESULTS.md
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/CONTEXT_FOR_MANAGER.md
* https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md

---

## 9. Recommended next steps

1. **Settle the receiver-sampling question (§5.3) before the next gate.** Re-score the 15 frozen
   test shapes at dense receivers and report both numbers. If the −0.18 to −0.22 gap holds on the
   gate set, several published accuracy claims read high.
2. **Stress-test narrow stems, not corner counts (§5.2).** The U/DN split says the model struggles
   when two notches narrow one wall. Before committing to the Z-corridor, run a stem-width sweep
   at fixed corner count.
3. **Sweep 1's flatness (§5.1) is the strongest claim in the pack** and is cheap to extend: the
   same format on a third parameter (source position, or wall absorption) would test whether
   flatness is specific to shape edits.
