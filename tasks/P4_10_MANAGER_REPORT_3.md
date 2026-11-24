# P4-10 final report for the manager, report 3 (2026-11-24 07:58 ET)

This replaces the state and questions in [report 2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_MANAGER_REPORT_2.md) (answered in your notes, D206). Every run, read, diagnostic and figure of P4-10 is finished and nothing is running. Numbers were read from files for this report; times come from `date`. No pre-registered bar changed. Triples are 25-100 / 100-200 / 200-300 Hz; r is the off-lattice accuracy from the frozen scorer; "in-family" is the 15 known-family test rooms and "unseen" the frozen 16 new rooms. HF is in-family r at 200-300 Hz; UF is unseen r per band.

## 1. Where things stand

1. **No arm passes its primary bar; every no-harm bar holds** (R1 costs 0.013 on HF, inside its 0.02 margin). R1 stays the base (D201 Q28). No arm was extended.
2. **R1-M (materials) is the first model that responds to a material change.** Its decay rate (1 / EDT) moves in the right direction and nearly in proportion (Pearson 0.95) but by 0.54 of the true change; the bar is 0.8 to 1.2.
3. **R1-S (stretched rooms) improves stretch edits over R1 in every band, in known and new rooms, by less than the bar** (+0.032 / +0.017 / +0.012 in-family; bar +0.05 in two bands).
4. **Decay (E1): two single-room models reproduce their rooms' resonance widths (ratio 1.00) and EDT (0-2% error). R1 does not, even in rooms it trained on (1.73).** By D207's reading, sharing one model causes the broadening. Sharing and per-room exposure are still not separated (Q25).
5. **E2: reciprocal averaging is not a P4-11 candidate. E3: the late floor has an excess common-to-the-room part only at 63 Hz.**
6. **Correction to what I published at 03:24 (section 6):** wall clearance is not shown to be the reason the second source lags. Please do not build on the earlier wording.

**You have five decisions to make (section 2).** Only the meeting date (Q33) is the user's.

## 2. Decisions needed

Full texts: [OPEN_QUESTIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md). The recommendations are mine; none has been acted on.

| # | decision | evidence | options | my recommendation |
|---|---|---|---|---|
| **Q34** | What P4-11 does first about decay and fit | E1 (section 5). Width ratio in the in-family test rooms, median (mean): R0 @120K 1.50 (1.51), CTL @150K 1.48 (1.50), S2X @240K on twice the rooms 1.50 (1.44). So 25% more steps, or twice the rooms at the same per-room exposure, leave it about the same | (a) eval-only: E1's width and EDT read on R0 @30K / 60K / 90K / 120K and on S-1 @4K, 17 rooms, truth already on disk; (b) wider decoder at R0's exposure (Q25 a); (c) per-room adapters (Q25 b); (d) fewer rooms (Q25 c, confounds the two); (e) a decay loss on R1 | **(a) first**, because it shows whether width falls with exposure before any GPU-day is spent; then (b) or (c) on its result. Not (e) first: the same loss reaches the true width in one room |
| **Q35** | Which model and training data P4-11 starts from | R1-S and R1-M were trained separately; SC1 and MC1 hold; against R1 their static accuracy is +0.001 and -0.003 on HF, and their unseen accuracy is slightly higher, with every paired CI above zero on the frozen set (section 3; R1-S's lowest band only just). Under Q28 R1 is the base | (a) R1 as is; (b) one combined arm, MS + STR + MAT rows, 30K steps from R0, pre-registered against R1; (c) R1-S or R1-M as base | **(b)**, with RC1-style no-harm bars against R1. It is untested that the gains add, and each variant gets a smaller share of the batch. A 30K arm took 6.6 to 6.9 h on one RTX A5000 |
| **Q36** | Whether P4-11 spends anything on the moved source | RS1-RS3 fail; gaps above 100 Hz moved little after +10K (by at most 0.03 on the lite rooms); E2 is out; at 1.5 m from the wall the gap is still 0.112 / 0.223 / 0.272 (section 6) | (a) stop, use your approved wording; (b) more sources per room (R1 used 4); (c) eval-only first: moved-source r against distance to training-source positions; (d) a reciprocity loss | **(a) for the meeting; (c) before any of (b) or (d)** |
| **Q37** | Materials: how to treat MM1's half-size response. Please also confirm or override D208, my reading of R1-M's extension rule | Slope 0.54 in-family, 0.66 unseen. Lite slope by read: 0.36, 0.48, 0.61, 0.53, 0.57, 0.62. D208 needed at least 0.75 at +30K on the lite set; it is 0.62, so the extension is not close under that reading | (a) accept as a result; (b) more steps; (c) more or stronger material rows; (d) fold into the decay work | **(d)**. R1-M's original rooms already decay too fast by 18 / 25 / 34%. Broadening alone does not explain the half slope: MM1 is on 1 / EDT, where an added constant width leaves the slope at 1 (reasoning, not a measurement) |
| **Q38** | Meeting claims for R1-S, R1-M and E1; and 2f (capture-then-edit): run or drop | Section 8 has proposed sentences | approve, edit, or drop each | approve with the caveats as written; **drop 2f** unless the meeting needs a capture story |

## 3. Verdicts (all at 150K = +30K from R0, full set, paired bootstrap 10k)

| arm | rule | result |
|---|---|---|
| **R1** (4 sources per room + swaps) | RS1: six in-family gaps <= 0.10 | **FAILS**: second 0.203 / 0.373 / 0.372, third 0.128 / 0.238 / 0.250 |
| | RS2: unseen moved r >= ignore + 0.05, six cells | **FAILS** in one of six: second source, 200-300 Hz, -0.010 [-0.038, +0.017] |
| | RS3: swapped >= normal - 0.05 | **FAILS**: in-family normal / swapped 0.909 / 0.759, 0.683 / 0.344, 0.571 / 0.222 |
| | RC1: no harm against CTL @150K | HOLDS: HF 0.571 vs 0.583, -0.013 [-0.019, -0.007]; UF above in each band |
| | extension | not triggered |
| **R1-S** (+ stretched rooms) | SS1: stretch r_delta +0.05 over R1 in 2 of 3 bands | **FAILS**, 0 of 3: +0.032 [+0.018, +0.047] / +0.017 [+0.010, +0.025] / +0.012 [+0.007, +0.018]; levels 0.808 / 0.571 / 0.410 vs 0.776 / 0.554 / 0.398 |
| | unseen (descriptive) | +0.036 [+0.024, +0.050] / +0.026 [+0.020, +0.032] / +0.022 [+0.016, +0.029]; levels 0.313 / 0.231 / 0.236 vs 0.277 / 0.205 / 0.213 |
| | SC1: no harm | HOLDS: HF 0.572 vs 0.571, +0.001 [-0.003, +0.004]; UF 0.515 / 0.419 / 0.411 vs 0.506 / 0.409 / 0.399, +0.009 [+0.000, +0.018] / +0.011 [+0.006, +0.016] / +0.013 [+0.005, +0.020] |
| | extension | not triggered: band mean rose 0.0089, needs 0.02 |
| **R1-M** (+ per-wall materials) | MM1: slope of the change in 1 / EDT in [0.8, 1.2], Pearson >= 0.8 | **FAILS** on the slope: 0.54, Pearson 0.95 (per octave 63 / 125 / 250 Hz: 0.48 / 0.58 / 0.54). Unseen 0.66, Pearson 0.94 (0.48 / 0.77 / 0.59) |
| | MM2: edited EDT error <= own unedited + 5 points | HOLDS: 22.5% vs 24.6%; unseen 29.0% vs 29.4%. **Absolute levels are far above the JNDs (5% ISO 3382-1, about 18% measured)** |
| | MC1: no harm | HOLDS: HF 0.568 vs 0.571, -0.003 [-0.007, +0.001]; UF 0.522 / 0.429 / 0.407 vs 0.506 / 0.409 / 0.399, +0.016 [+0.005, +0.029] / +0.020 [+0.013, +0.028] / +0.008 [+0.001, +0.016] |
| | extension (D208) | not met: lite slope 0.62, needs 0.75 |
| **R1-P** (physics penalty) | gate 0d | **FAILS** (D205): raw / band-limited 1.49 / 1.27 / 1.36, needs 3. Not launched |
| **R1-C** (ray clipping) | closed | rule 10 stands (D201 Q29) |

MM1 on the same cells: R1 0.01, R1-S 0.01, R0 -0.04. MM1's slope is a pooled regression with no confidence interval. T20 is not usable: the floor rule drops 429477 of 543192 in-family receiver-bands for R1-M.

Files: [verdict_R1_150K.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1_150K.md) · [verdict_R1S_150K.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1S_150K.md) · [verdict_R1M_150K.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1M_150K.md)

## 4. The scorecard after all arms (edited-field r, full set)

| family | rooms | R0 | R1 | R1-S | R1-M | ignore the edit |
|---|---|---|---|---|---|---|
| Stretch-U | in-family | 0.833 / 0.606 / 0.493 | 0.825 / 0.599 / 0.492 | 0.863 / 0.629 / 0.522 | 0.827 / 0.599 / 0.494 | 0.410 / 0.171 / 0.119 |
| Stretch-A | in-family | 0.835 / 0.613 / 0.495 | 0.832 / 0.609 / 0.494 | 0.872 / 0.633 / 0.527 | 0.834 / 0.608 / 0.499 | 0.514 / 0.250 / 0.281 |
| Wall | in-family | 0.900 / 0.673 / 0.562 | 0.891 / 0.658 / 0.557 | 0.899 / 0.667 / 0.565 | 0.888 / 0.651 / 0.554 | 0.735 / 0.421 / 0.233 |
| Notch | in-family | 0.917 / 0.697 / 0.568 | 0.906 / 0.681 / 0.560 | 0.905 / 0.682 / 0.561 | 0.901 / 0.672 / 0.557 | 0.885 / 0.715 / 0.534 |
| Source | in-family | 0.231 / 0.083 / 0.044 | 0.743 / 0.377 / 0.260 | 0.755 / 0.383 / 0.266 | 0.733 / 0.380 / 0.271 | 0.305 / 0.217 / 0.215 |
| Material | in-family | 0.772 / 0.583 / 0.469 | 0.749 / 0.567 / 0.476 | 0.749 / 0.558 / 0.484 | 0.894 / 0.713 / 0.618 | 0.911 / 0.891 / 0.880 |
| Stretch-U | unseen | 0.402 / 0.325 / 0.304 | 0.442 / 0.356 / 0.340 | 0.492 / 0.409 / 0.389 | 0.461 / 0.382 / 0.347 | 0.440 / 0.153 / 0.103 |
| Stretch-A | unseen | 0.416 / 0.347 / 0.327 | 0.461 / 0.373 / 0.359 | 0.499 / 0.410 / 0.396 | 0.477 / 0.394 / 0.368 | 0.544 / 0.255 / 0.255 |
| Wall | unseen | 0.438 / 0.363 / 0.347 | 0.483 / 0.392 / 0.372 | 0.505 / 0.417 / 0.401 | 0.499 / 0.410 / 0.380 | 0.752 / 0.413 / 0.213 |
| Notch | unseen | 0.464 / 0.380 / 0.372 | 0.503 / 0.409 / 0.399 | 0.512 / 0.419 / 0.411 | 0.520 / 0.424 / 0.407 | 0.916 / 0.760 / 0.624 |
| Source | unseen | 0.212 / 0.110 / 0.062 | 0.546 / 0.321 / 0.246 | 0.547 / 0.328 / 0.250 | 0.550 / 0.330 / 0.262 | 0.344 / 0.246 / 0.209 |
| Material | unseen | 0.428 / 0.357 / 0.347 | 0.447 / 0.376 / 0.373 | 0.455 / 0.385 / 0.397 | 0.558 / 0.483 / 0.460 | 0.921 / 0.897 / 0.884 |

**How to read it.**
- "Ignore the edit" is the original room's **truth** scored against the edited truth. No model has that truth, so this reference is hardest to beat where the edit changes little (Notch, Material) and where the model's own accuracy is low (unseen).
- By the means (no CI), the number of family-band cells above "ignore the edit" is: in-family R0 11, R1 14, R1-S 14, R1-M 14 of 18; unseen R0 5, R1 9, R1-S 10, R1-M 9 of 18.
- In-family, R1-S and R1-M miss it in the same four cells: Notch 100-200 Hz and all three Material bands.
- **R1-M's material row is below "ignore the edit" in every band** (0.894 / 0.713 / 0.618 vs 0.911 / 0.891 / 0.880). The 8-room lite reads in report 2 had it above at 25-100 Hz; the full set does not. But its edited render is closer to the edited truth than its own unedited render: g_delta 1.509 / 1.119 / 1.059 in-family and 1.083 / 1.026 / 1.027 unseen (1 = no gain; R1 0.842 / 0.857 / 0.867 in-family).
- **Known-room geometry edits:** R1 is slightly below R0 in r_delta; R1-S recovers this for stretch, not for Wall and Notch.
- **New rooms are the weak point in every family.** Notch r_delta is at most 0.077 / 0.023 / 0.019, so unseen notch edits are not predicted.

Pattern of change (r_delta), same cells:

| family | rooms | R0 | R1 | R1-S | R1-M |
|---|---|---|---|---|---|
| Stretch-U | in-family | 0.818 / 0.589 / 0.486 | 0.801 / 0.573 / 0.478 | 0.828 / 0.590 / 0.492 | 0.796 / 0.562 / 0.473 |
| Stretch-A | in-family | 0.766 / 0.552 / 0.327 | 0.751 / 0.536 / 0.317 | 0.788 / 0.552 / 0.328 | 0.747 / 0.522 / 0.309 |
| Wall | in-family | 0.743 / 0.506 / 0.458 | 0.721 / 0.486 / 0.448 | 0.732 / 0.491 / 0.454 | 0.713 / 0.471 / 0.440 |
| Notch | in-family | 0.637 / 0.328 / 0.216 | 0.614 / 0.310 / 0.209 | 0.617 / 0.310 / 0.213 | 0.602 / 0.301 / 0.207 |
| Source | in-family | 0.504 / 0.391 / 0.304 | 0.776 / 0.518 / 0.405 | 0.783 / 0.520 / 0.409 | 0.768 / 0.516 / 0.411 |
| Material | in-family | 0.067 / 0.026 / 0.017 | 0.094 / 0.023 / 0.017 | 0.091 / 0.026 / 0.020 | 0.442 / 0.180 / 0.122 |
| Stretch-U | unseen | 0.253 / 0.203 / 0.249 | 0.287 / 0.230 / 0.278 | 0.325 / 0.264 / 0.308 | 0.296 / 0.249 / 0.281 |
| Stretch-A | unseen | 0.224 / 0.164 / 0.135 | 0.267 / 0.180 / 0.149 | 0.301 / 0.199 / 0.163 | 0.270 / 0.193 / 0.150 |
| Wall | unseen | 0.179 / 0.144 / 0.210 | 0.218 / 0.158 / 0.223 | 0.244 / 0.175 / 0.245 | 0.225 / 0.166 / 0.229 |
| Notch | unseen | 0.067 / 0.019 / 0.019 | 0.077 / 0.022 / 0.018 | 0.077 / 0.023 / 0.018 | 0.074 / 0.023 / 0.019 |
| Source | unseen | 0.283 / 0.253 / 0.226 | 0.469 / 0.367 / 0.317 | 0.473 / 0.378 / 0.326 | 0.478 / 0.381 / 0.329 |
| Material | unseen | 0.027 / -0.006 / -0.000 | 0.044 / -0.001 / -0.006 | 0.045 / -0.002 / 0.002 | 0.199 / 0.112 / 0.071 |

Files: [scorecard_R0_full.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R0_full.md) · [scorecard_R1_full.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1_full.md) · [scorecard_R1S_full.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1S_full.md) · [scorecard_R1M_full.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1M_full.md) · [board](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md) · [F2 trajectories](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/F2_trajectories.png)

## 5. Decay: E1, E3, D1, D2, A8

**E1 (fit or transfer; dense off-lattice truth simulated anew).**

| group | rooms | width ratio model / truth | peaks matched | EDT error 63 / 125 / 250 Hz | floor model / truth, dB |
|---|---|---|---|---|---|
| **S-1 single-room models, own rooms** | 2 | **1.00** (0.9996, 0.9965) | 11 of 11 | 0 / 1 / 2% | -36 / -72, -34 / -75, -31 / -36 |
| R1 in the same 2 rooms | 2 | 1.40 (1.34, 1.46) | 7 of 11 | 13 / 26 / 30% | -36 / -72, -30 / -75, -27 / -36 |
| **R1, 15 random training rooms** | 15 | **1.73 [1.53, 1.94]** | 36 of 63 | 12 / 25 / 31% | -35 / -72, -29 / -77, -26 / -41 |
| R1, 15 in-family test rooms (13 usable) | 13 | 1.40 (median; mean 1.44) | 33 of 65 | 14 / 24 / 30% | -35 / -71, -29 / -78, -27 / -40 |

- **D207's reading:** S-1 at the truth's width (<= 1.15) and R1's training rooms broad (>= 1.30), so the broadening comes from sharing one model, not from the loss or the representation.
- **What it does not show:**
  - S-1 is 2 rooms with 11 peaks, at about 83 times R0's per-room exposure. Sharing and exposure are confounded (Q25, Q34).
  - The width error may not be a separate problem: S-1 also removes the 200-300 Hz fit cap (D185), so it may be the same fit deficit seen in another measure. E1 cannot tell.
  - R1 matches 36 of 63 truth peaks in its training rooms, so each room's ratio rests on 1 to 4 peaks (per-room 1.06 to 2.46).
- **There are two decay problems.** The S-1 models still have a floor of -31 to -36 dB (truth -72 / -75 at 63 / 125 Hz). Single-room training fixes the early decay, not the late floor. T20 and T30 would stay unusable at 63 and 125 Hz even if sharing were solved.

**E3 (is the late floor common to the room? ratio model / truth).** 63 Hz: R0 2.2, CTL 2.5, RCP 1.9, S2X 2.6, R1 2.4, R1-S 3.2, R1-M 3.1. 125 Hz: 0.6 to 1.0. 250 Hz: 0.6 to 1.2. So the excess room-uniform part is at 63 Hz only.

**D1 (EDT error, original rooms, in-family, 63 / 125 / 250 Hz; every model is too fast).** R0 12.6 / 23.1 / 28.3%, R1 14.1 / 24.2 / 29.6%, R1-S 15.2 / 24.1 / 29.2%, R1-M 18.2 / 25.3 / 33.7%. Unseen: R0 22.6 / 26.4 / 29.1%, R1 23.4 / 27.5 / 30.3%, R1-S 23.4 / 28.4 / 31.6%, R1-M 24.8 / 29.5 / 35.7%. **R1-M's original rooms are a little worse than R1's in every octave.**

**D2 (in-family width ratio, median).** R0 1.50, CTL 1.48, RCP 1.49, S2X 1.50, R1 1.40, R1-S 1.43, R1-M 1.47; 13 or 14 rooms, 32 to 43 of 65 peaks matched.

**A8 (mode-shift slope, 25-150 Hz, in-family; Stretch-U / Stretch-A / Wall / Notch).** R0 0.65 / 0.54 / 0.71 / 0.25, R1 0.60 / 0.58 / 0.60 / 0.30, R1-S 0.69 / 0.52 / 0.68 / 0.40, R1-M 0.67 / 0.68 / 0.67 / 0.46. Only 121 to 185 of 431 to 440 truth peaks matched, so treat differences between arms as descriptive.

Files: [E1](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/e1/report.md) · [E3](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/e3/report.md) · [D1](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/d1/report.md) · [D2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/d2.json) · A8 [R0](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/a8_R0_120K.json), [R1](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/a8_R1_150K.json), [R1-S](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/a8_R1S_150K.json), [R1-M](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/a8_R1M_150K.json)

## 6. Moving the source: E2 and the clearance diagnostic

**E2 (reciprocal averaging, all 31 rooms). Pre-registered reading: NOT a candidate.**

| position | normal | swapped | average | average - normal |
|---|---|---|---|---|
| second | 0.619 / 0.291 / 0.199 | 0.634 / 0.289 / 0.185 | 0.654 / 0.326 / 0.176 | +0.035 [+0.024, +0.047] / +0.036 [+0.019, +0.052] / -0.023 [-0.035, -0.013] |
| third | 0.663 / 0.406 / 0.306 | 0.615 / 0.340 / 0.247 | 0.669 / 0.386 / 0.242 | +0.005 [-0.000, +0.011] / -0.019 [-0.024, -0.015] / -0.064 [-0.069, -0.059] |
| original | 0.701 / 0.541 / 0.482 | 0.607 / 0.316 / 0.230 | 0.691 / 0.480 / 0.389 | -0.010 [-0.020, +0.001] / -0.062 [-0.078, -0.045] / -0.093 [-0.110, -0.077] |

It helps the second source below 200 Hz, hurts the third and the original from 100 Hz up, and hurts all three above 200 Hz.

**Source clearance (D209; R1, 8 in-family lite rooms, source on y = 1.6 at a set distance from the east wall).**

| clearance | r | gap to the original source (0.909 / 0.670 / 0.566) |
|---|---|---|
| 0.50 m | 0.712 / 0.305 / 0.200 | 0.197 [0.144, 0.255] / 0.365 [0.290, 0.439] / 0.366 [0.330, 0.412] |
| 0.75 m | 0.762 / 0.427 / 0.220 | 0.147 [0.123, 0.172] / 0.242 [0.183, 0.310] / 0.346 [0.301, 0.403] |
| 1.00 m | 0.787 / 0.484 / 0.272 | 0.122 [0.106, 0.137] / 0.186 [0.161, 0.210] / 0.294 [0.268, 0.324] |
| 1.50 m | 0.797 / 0.446 / 0.293 | 0.112 [0.098, 0.127] / 0.223 [0.187, 0.257] / 0.272 [0.251, 0.294] |

**You asked whether clearance is the cause. This diagnostic does not show that it is, and my 03:24 entry was wrong to say so:**
1. The 0.5 m point is the second source itself (identical coordinates in all 8 rooms), so its agreement with the second source's gap is not independent evidence.
2. The original source is also 0.5 m from a wall (west) and is the most accurate position tested, 0.909 / 0.670 / 0.566. Clearance alone cannot be the cause.
3. The 100-200 Hz gap is not steady: 0.365, 0.242, 0.186, 0.223.

What does hold: going from 0.5 m to 1.5 m removes 0.085 / 0.142 / 0.094 of the gap, about what separates the second source from the third in the same rooms (0.080 / 0.131 / 0.115). More than half of the gap remains at 1.5 m, and it is above RS1's 0.10 in every band. Training coverage is an untested alternative: (0.5, 1.6) is the only original source of the 500 C500 rooms, while the multi-source corpus has 1412 of 4000 sources within 0.75 m of a wall. That would also mean RS1's gap partly measures how well-trained its reference position is.

Files: [E2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/e2/report.md) · [clearance](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/srcclear/report.md)

## 7. Unseen accuracy on both sets (as you asked)

| model | frozen unseen 16 | fresh confirmation 16 (each model scored once) |
|---|---|---|
| R0 @120K | 0.468 / 0.380 / 0.371 | 0.607 / 0.447 / 0.400 |
| S2X @240K | 0.495 / 0.403 / 0.398 | 0.605 / 0.466 / 0.430 |
| **R1 @150K** | 0.506 / 0.409 / 0.399 | 0.644 / 0.469 / 0.416 |
| R1-S @150K | 0.515 / 0.419 / 0.411 | not scored (Q32 covered R1 only) |
| R1-M @150K | 0.522 / 0.429 / 0.407 | not scored |

Confirmation, paired: R1 - R0 +0.037 [+0.020, +0.055] / +0.022 [+0.008, +0.039] / +0.015 [+0.006, +0.024]; R1 - S2X +0.039 [+0.022, +0.057] / +0.003 [-0.012, +0.016] / -0.014 [-0.041, +0.009]. The P4-11 confirmation set (D203, sha `1a49a13b9b8b3912...`) is frozen and not scored. R1-S's and R1-M's frozen-set gains over R1 (0.008 to 0.020) are below your 0.03 caution; they have paired CIs above zero, on one set only.

## 8. Proposed meeting wording (Q38)

- **R1** (already approved): "Moved sources are far better than before, not solved," with the numbers in D206.
- **R1-S:** "Training on stretched rooms improves stretch edits a little in every band, in known and new rooms, and costs nothing at the original rooms. It did not reach the bar we set."
- **R1-M:** "The first model that responds to a change of wall material: right direction, nearly proportional, about half the true size. After the edit it is closer to the truth than before the edit, but decay errors of 22-29% remain, above what a listener notices."
- **E1:** "Two models trained on one room each get the decay right. One model shared by 1000 rooms does not, even in rooms it trained on. We have not separated sharing from training time per room."
- **Do not claim:** that clearance explains the second source; that material edits beat the original room's true field; that geometry edits transfer to new rooms (unseen geometry r_delta is at most 0.325, and Notch is not predicted).

Figures: [F1 R0 scorecard](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/F1_r0_scorecard.png) · [F2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/F2_trajectories.png) · [F3 moved-source maps](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/F3_moved_source_maps.png) · [F4 stretch](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/F4_stretch_delta.png) · [F5 material decay](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/F5_material_decay.png) · [F7 scorecard after](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/F7_scorecard_after.png) · [manifest](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/FIGURE_MANIFEST.md). F6 is void (R1-P not launched).

## 9. Process record since report 2

1. **The clearance conclusion was wrong and stood from 03:24 until the commit that adds this report** in the status report and the results document; both are corrected and say so. The numbers were right. I found it while writing this report.
2. **"Decay time" in the MM1 entry should read decay rate (1 / EDT);** corrected. The slope is unchanged.
3. **Two static reads of R1-M were late:** 140K by six minutes, 145K by one minute (noted in the log at the time).
4. **Wording fixes to R1-M, R1-S and E1 entries** were made within minutes of posting (commits `5b162ba`, `768f9c7`, `ca5da89`, `7ca57e2`); their tables were right.
5. **The board listed a finished job as running** until it was rebuilt at 07:54.
6. The 30-minute monitoring tick was cancelled at 04:17 because no job was left. No job has failed since midnight (checked 05:55 with the corrected query).

## 10. Where everything is

Results document: [CHUNK_P4_10_RESULTS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_P4_10_RESULTS.md) · Live log: [P4_10_STATUS_REPORT.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_STATUS_REPORT.md) · Decisions D195-D210: [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md) · Open questions: [OPEN_QUESTIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md) · Your notes: [P4_10_MANAGER_NOTES_2.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_MANAGER_NOTES_2.md) · Your rulings: [P4_10_RULINGS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_RULINGS.md) · Gate 0d: [0d](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/r1p_0d/report.md) · Scripts: [p4_10_scorecard.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_scorecard.py), [p4_10_e1_decay.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_e1_decay.py), [p4_10_e2_recip_avg.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_e2_recip_avg.py), [p4_10_e3_floor.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_e3_floor.py), [p4_10_src_clearance.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_src_clearance.py), [p4_10_verdict_r1m.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_verdict_r1m.py), [p4_10_verdict_arms.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_verdict_arms.py) · Confirmation reads: [R0](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/confirm/R0_120K.json), [S2X](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/confirm/S2X_240K.json), [R1](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/confirm/R1_150K.json)
