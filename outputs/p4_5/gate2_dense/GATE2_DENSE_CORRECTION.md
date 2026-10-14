# Gate 2 at dense receivers -- a correction to the P4 record (D99, D102)

**Date** 2026-10-13. **Requested by** the manager, as a correction to the record.

## What was wrong

Every P4 room -- training and test -- places its 800 receivers on one absolute lattice, x, y = 0.12 + 0.06 k m. The attention-pooling arms memorised that lattice: accuracy at those exact positions is far higher than 2 cm away (1/57 of a wavelength at 300 Hz). Gate 2, the sweep curves and the family eval were all measured AT those positions, and every validation set was drawn from the same lattice, so nothing could have detected it.

## Method

- The 15 frozen test shapes re-simulated on a 64x64 lattice masked to the polygon (same geometry, same solver, dx 0.02): ~3.2-4.1k receivers per room, ~1 in 9 on the old lattice after snapping to FDTD nodes.
- Scored with `scripts/p4_5_gate2_dense.py`: the frozen evaluator's render loop and metrics **verbatim**, split into on-lattice / all / off-lattice receivers from one render.
- **Exactness check 1:** on the original 800-receiver data it reproduces C250's published GATE2.json on all 14 fields, to the last digit (including pooled NLOS counts 354 and 792).
- **Exactness check 2:** on the dense data its `all` subset reproduces the frozen evaluator itself, run unchanged, on all 14 fields. (The frozen evaluator exits 2 for C250 -- its own FAIL verdict.)

## In-slab spatial Pearson (Gate 2 threshold 0.80)

| checkpoint | pooling | corpus | published | dense on-lattice | dense all | **dense off-lattice** | published | **dense (off)** |
|---|---|---|---|---|---|---|---|---|
| P4-3 D | masked_mean | 92 | 0.8021 | 0.8146 | 0.8066 | **0.8063** | PASS | **PASS** |
| P4-3 X1 | attn_residual | 60 | 0.8438 | 0.8430 | 0.7686 | **0.7581** | PASS | **FAIL** |
| P4-4 C92 | attn_residual_extent | 92 | 0.8337 | 0.8306 | 0.7638 | **0.7545** | PASS | **FAIL** |
| P4-4 C250 | attn_residual_extent | 250 | 0.8420 | 0.8412 | 0.6368 | **0.6109** | PASS | **FAIL** |
| P4-4 C500 | attn_residual_extent | 500 | 0.8336 | 0.8341 | 0.5951 | **0.5641** | PASS | **FAIL** |
| P4-3 X2 | attn | 60 | 0.7789 | 0.7808 | 0.6765 | **0.6627** | FAIL | **FAIL** |
| P4-3 S | masked_mean (+sigma aux) | 60 | 0.7329 | 0.7324 | 0.7336 | **0.7338** | FAIL | **FAIL** |

## Out-of-slab spatial Pearson

| checkpoint | published | dense on-lattice | dense all | dense off-lattice |
|---|---|---|---|---|
| P4-3 D | 0.7641 | 0.7651 | 0.7651 | 0.7651 |
| P4-3 X1 | 0.7612 | 0.7525 | 0.7031 | 0.6971 |
| P4-4 C92 | 0.8060 | 0.7976 | 0.7357 | 0.7282 |
| P4-4 C250 | 0.8359 | 0.8310 | 0.6299 | 0.6087 |
| P4-4 C500 | 0.8292 | 0.8250 | 0.6062 | 0.5828 |
| P4-3 X2 | 0.7427 | 0.7406 | 0.6550 | 0.6451 |
| P4-3 S | 0.6650 | 0.6714 | 0.6660 | 0.6653 |

## NLOS deficit (pass <= 0.15) -- still passes everywhere

| checkpoint | published | dense all | dense off-lattice |
|---|---|---|---|
| P4-3 D | -0.087 | -0.084 | -0.083 |
| P4-3 X1 | -0.002 | -0.060 | -0.064 |
| P4-4 C92 | -0.056 | -0.094 | -0.100 |
| P4-4 C250 | -0.038 | -0.024 | -0.013 |
| P4-4 C500 | -0.054 | -0.044 | -0.031 |
| P4-3 X2 | +0.070 | -0.046 | -0.060 |
| P4-3 S | +0.020 | +0.008 | +0.002 |


## AMENDMENT (D105): bootstrap CIs -- the verdicts are noise-limited, the shifts are not

Across-test-shape bootstrap, stratified by slab, percentile 95% CI, 10k resamples. A CI that includes 0.80 reads **at threshold**, whichever side the estimate sits. This amends point 1 below: the published verdicts were mostly unresolvable too, so "four of five passes do not survive" overstated what a 6-shape gate can resolve.

| checkpoint | published in-slab [95% CI] | verdict | dense off-lattice [95% CI] | verdict |
|---|---|---|---|---|
| P4-3 D | 0.8021 [0.764, 0.839] | AT THRESHOLD | 0.8063 [0.766, 0.845] | AT THRESHOLD |
| P4-3 X1 | 0.8438 [0.794, 0.886] | AT THRESHOLD | 0.7581 [0.715, 0.811] | AT THRESHOLD |
| P4-4 C92 | 0.8337 [0.794, 0.869] | AT THRESHOLD | 0.7545 [0.710, 0.795] | FAIL |
| P4-4 C250 | 0.8420 [0.804, 0.878] | PASS | 0.6109 [0.580, 0.656] | FAIL |
| P4-4 C500 | 0.8336 [0.782, 0.875] | AT THRESHOLD | 0.5641 [0.544, 0.587] | FAIL |
| P4-3 X2 | 0.7789 [0.714, 0.836] | AT THRESHOLD | 0.6627 [0.587, 0.718] | FAIL |
| P4-3 S | 0.7329 [0.691, 0.773] | FAIL | 0.7338 [0.690, 0.778] | FAIL |

**Robust:** the shift (C250 0.842 [0.804, 0.878] -> 0.611 [0.580, 0.656], non-overlapping) and the pooling contrast, paired over all 15 frozen shapes, off-lattice dense:

| band | D (mean pooling) | C92 (attention, same corpus) | D - C92 [95% CI] | D better in |
|---|---|---|---|---|
| 25-100 Hz | 0.752 | 0.630 | +0.122 [+0.092, +0.148] | 14/15 |
| 100-200 Hz | 0.453 | 0.130 | +0.324 [+0.279, +0.365] | 15/15 |
| 200-300 Hz | 0.319 | 0.083 | +0.237 [+0.210, +0.265] | 15/15 |

## AMENDMENT (D106): the material comparison inverts

Material receivers were drawn PER ROOM (8x8 linspace scaled to each room's L, W: `scripts/build_2d_mat_dataset.py:39-42`, `scripts/build_p3_3fast_trackA.py:32-35`), test geometries disjoint from training, 3.7-6.1% positional coincidence. The material work's 1.7-2.3 dB was a field measurement; shape's 2.32 dB below 100 Hz was at memorised positions. As a field, shape below 100 Hz is 3.4-3.9 dB -- **not at parity**.

## What it means

1. **Four of the five published Gate-2 passes do not survive** (X1, C92, C250, C500). **Arm D's does**, and is the only one.
2. **The memorisation comes from attention pooling.** Both masked_mean arms (D, S) show no on/off-lattice gap; all five attention arms lose 0.08-0.27. **D and C92 share the identical 92-shape corpus and differ only in pooling:** on-lattice C92 leads (0.8306 vs 0.8146); off-lattice D leads (**0.8063 vs 0.7545**).
3. **D81 reverses**: attention pooling's +0.0316 on an identical corpus becomes **-0.052** as a field.
4. **D80 reverses**: the saturating scaling curve (0.8337 / 0.8420 / 0.8336) **declines** as a field (0.7545 / 0.6109 / 0.5641).
5. **Family eval -- BOUNDED since (D111)**: one median-d_hat room per family re-simulated dense; published values reproduced to 7e-10; off-lattice shift mean **-0.222**, range [-0.282, -0.168], 5 of 5 rooms. D83's corner ordering unsupported as a field; D84's token cost shrinks off-lattice (rect -0.11 -> -0.04, L -0.30 -> -0.21). `FAMILY_DENSE_BOUND.json`, `FAMILY_TOKENS_DENSE.json`. **Not re-scored, flagged**: P4-2's four arms (checkpoints no longer on disk; mean-type pooling, so probably not inflated -- inference, not measurement).

## Files

- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/SUMMARY.json
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/P4-3_D__track_p4_5_testdense.json
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/P4-3_X1__track_p4_5_testdense.json
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/P4-4_C92__track_p4_5_testdense.json
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/P4-4_C250__track_p4_5_testdense.json
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/P4-4_C500__track_p4_5_testdense.json
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/P4-3_X2__track_p4_5_testdense.json
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/P4-3_S__track_p4_5_testdense.json
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/frozen_evaluator_C250_dense/GATE2.json
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_5_gate2_dense.py
- D99 and D102 in https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md
