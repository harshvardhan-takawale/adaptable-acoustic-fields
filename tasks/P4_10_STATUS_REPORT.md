# P4-10 status report (live log) — the editability scorecard, the source fix, four single-change arms

**Pre-registration:** D195 (spec, verbatim; also `tasks/P4_10_SPEC.md`), D196 (progress contract), D197 (amendments A1-A14), D198 (edit manifest frozen), D199 (per-edge absorption validated). The P4-9 erratum (D194) corrects a premise the spec inherited.
**Board:** [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md). One dated entry per read, newest first, entry times from `date`. Raw links checked for 200.

## 2026-11-24 (02:59) — R1-M's D1, D2 and A8 at its final checkpoint: its unedited decay is a little worse than R1's; mode-shift slopes are similar

**Headline:** R1-M's sound decays too fast by 18% / 25% / 34% in known rooms (63 / 125 / 250 Hz), a little worse than R1's 14% / 24% / 30%, and its peaks are 1.47 times as broad as the truth (R1 1.40). So learning materials did not fix the decay problem; the original rooms got slightly worse. Mode-shift slopes after edits are similar to the other arms'.

**D1 (R1-M @150K, off-lattice receivers at least 1 m from the source; EDT error (bias) at 63 / 125 / 250 Hz):** in-family 18.2% (-18.0) / 25.3% (-24.0) / 33.7% (-33.3); unseen 24.8% / 29.5% / 35.7%. R1 in-family 14.1% / 24.2% / 29.6%, R0 12.6% / 23.1% / 28.3%. Floor (model): -35 / -30 / -28 dB in-family, similar to R1 (-35 / -29 / -27). **D2:** peak width 1.47 times the truth's (R1-M 13 rooms, 32 of 65 peaks matched; R1 1.40 on 13 rooms, R1-S 1.43 on 14). **A8 (mode-shift slope, full geometry cells, 25-150 Hz):** R1-M / R1 / R1-S in-family Stretch-U 0.67 / 0.60 / 0.69, Stretch-A 0.68 / 0.58 / 0.52, Wall 0.67 / 0.60 / 0.68, Notch 0.46 / 0.30 / 0.40; unseen Stretch-U 0.58 / 0.51 / 0.70, Stretch-A 0.49 / 0.49 / 0.38, Wall 0.59 / 0.65 / 0.51, Notch -0.09 / -0.01 / 0.13. Descriptive.

Links: [d1/report.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/d1/report.md) · [modes/d2.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/d2.json) · [a8_R1M_150K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/a8_R1M_150K.json)

## 2026-11-24 (02:54) — I22 (R1-M +30K, the full scorecard): better material edits than any other model, but on the full set still below "ignore the edit"; other edits level with R1

**Headline:** The full scorecard of the materials arm is in (675 cells). On wall-absorption edits R1-M scores 0.894 / 0.713 / 0.618 (25-100 / 100-200 / 200-300 Hz) in known rooms, against 0.749 / 0.567 / 0.476 for R1, and its pattern-of-change score (r_delta) is 4.7 to 7.8 times R1's in known rooms. **But "ignore the edit" still scores 0.911 / 0.891 / 0.880 on the same full set, so R1-M is below it in every band.** (The earlier reads on 8 lite rooms had it above at the lowest band; the full set does not.) On all other edit families it is level with R1.

R1-M @150K, off-lattice, full set (15 in-family, 16 unseen rooms), edited-field r (25-100 / 100-200 / 200-300 Hz), "ignore the edit" beside:

| family | set | R1-M | R1 | ignore the edit | R1-M r_delta (R1) |
|---|---|---|---|---|---|
| **Material** | in-family | **0.894 / 0.713 / 0.618** | 0.749 / 0.567 / 0.476 | 0.911 / 0.891 / 0.880 | 0.442 / 0.180 / 0.122 (0.094 / 0.023 / 0.017) |
| **Material** | unseen | **0.558 / 0.483 / 0.460** | 0.447 / 0.376 / 0.373 | 0.921 / 0.897 / 0.884 | 0.199 / 0.112 / 0.071 (0.044 / -0.001 / -0.006) |
| Stretch-U | in-family | 0.827 / 0.599 / 0.494 | 0.825 / 0.599 / 0.492 | 0.410 / 0.171 / 0.119 | 0.796 / 0.562 / 0.473 (0.801 / 0.573 / 0.478) |
| Stretch-U | unseen | 0.461 / 0.382 / 0.347 | 0.442 / 0.356 / 0.340 | 0.440 / 0.153 / 0.103 | 0.296 / 0.249 / 0.281 (0.287 / 0.230 / 0.278) |
| Stretch-A | in-family | 0.834 / 0.608 / 0.499 | 0.832 / 0.609 / 0.494 | 0.514 / 0.250 / 0.281 | 0.747 / 0.522 / 0.309 (0.751 / 0.536 / 0.317) |
| Stretch-A | unseen | 0.477 / 0.394 / 0.368 | 0.461 / 0.373 / 0.359 | 0.544 / 0.255 / 0.255 | 0.270 / 0.193 / 0.150 (0.267 / 0.180 / 0.149) |
| Wall | in-family | 0.888 / 0.651 / 0.554 | 0.891 / 0.658 / 0.557 | 0.735 / 0.421 / 0.233 | 0.713 / 0.471 / 0.440 (0.721 / 0.486 / 0.448) |
| Wall | unseen | 0.499 / 0.410 / 0.380 | 0.483 / 0.392 / 0.372 | 0.752 / 0.413 / 0.213 | 0.225 / 0.166 / 0.229 (0.218 / 0.158 / 0.223) |
| Notch | in-family | 0.901 / 0.672 / 0.557 | 0.906 / 0.681 / 0.560 | 0.885 / 0.715 / 0.534 | 0.602 / 0.301 / 0.207 (0.614 / 0.310 / 0.209) |
| Notch | unseen | 0.520 / 0.424 / 0.407 | 0.503 / 0.409 / 0.399 | 0.916 / 0.760 / 0.624 | 0.074 / 0.023 / 0.019 (0.077 / 0.022 / 0.018) |
| Source (both) | in-family | 0.733 / 0.380 / 0.271 | 0.743 / 0.377 / 0.260 | 0.305 / 0.217 / 0.215 | 0.768 / 0.516 / 0.411 (0.776 / 0.518 / 0.405) |
| Source (both) | unseen | 0.550 / 0.330 / 0.262 | 0.546 / 0.321 / 0.246 | 0.344 / 0.246 / 0.209 | 0.478 / 0.381 / 0.329 (0.469 / 0.367 / 0.317) |

- **Reading of the material row:** a correct edit and no edit score about the same on the edited-field measure here because the absorption change moves the field only a little (the r_delta column is the measure of whether the change itself is predicted: R1-M 0.442 / 0.180 / 0.122 in family against R1's 0.094 / 0.023 / 0.017, and its g_delta is 1.51 / 1.12 / 1.06, above 1 in every band, so R1-M's edited render is closer to the truth than its unedited one). The decay-time slope (I21, MM1) is 0.54 of the true amount with Pearson 0.95.
- **Other families:** level with R1 in known rooms (within 0.011 in edited-field r), and better by 0.008 to 0.026 on unseen stretch and wall edits.
- R1-M's MM1, MM2 and MC1 verdicts are in I21. D1, D2 and A8 for R1-M follow.

Links: [scorecard_R1M_full.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1M_full.md) · [scorecard_R1M_full.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1M_full.json) · [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-24 (02:52) — E1, E2, E3 (D206 / D207): decay — the single-room models get the decay right, the shared model does not; reciprocal averaging is not a candidate; the late floor is mostly not room-common

**Headline:** (E1) Models trained on one room each reproduce that room's resonance widths exactly (ratio 1.00 in both rooms) and its decay times to within 0-2%, while R1, one model for 1000 rooms, is 1.7 times too broad in its own training rooms and 25-31% too fast in decay (125-250 Hz). By the reading written before the numbers, the broadening comes from sharing one model across rooms. (E2) Averaging the normal and swapped renders does not improve moved sources above 100 Hz and costs the original source, so it is not a P4-11 candidate. (E3) The models' late floor is a common-to-the-room component only at 63 Hz (2-3 times the truth's), not at 125-250 Hz.

### E1. Decay: fit or transfer? (17 rooms, dense off-lattice truth simulated anew; the room's own source)

| group | rooms | width ratio model / truth | peaks matched | EDT error 63 / 125 / 250 Hz | floor model / truth, dB |
|---|---|---|---|---|---|
| **S-1 models, in their own rooms** (single-room models, 10K steps) | 2 | **1.00** (0.9996, 0.9965; 11 of 11 peaks) | 11 of 11 | **0% / 1% / 2%** | -36 / -72, -34 / -75, -31 / -36 |
| R1, the 2 S-1 rooms | 2 | 1.40 (1.34, 1.46) | 7 of 11 | 13% / 26% / 30% | -36 / -72, -30 / -75, -27 / -36 |
| **R1, 15 random training rooms** | 15 | **1.73 [1.53, 1.94]** | 36 of 63 | 12% / 25% / 31% | -35 / -72, -29 / -77, -26 / -41 |
| R1, the 15 in-family TEST rooms | 13 | 1.40 | 33 of 65 | 14% / 24% / 30% | -35 / -71, -29 / -78, -27 / -40 |

- **Pre-set reading (D207):** S-1 at the truth's width (<= 1.15): yes (1.00). R1's training rooms still broad (>= 1.30): yes (1.73). So the broadening comes from sharing one model across rooms (Q25), not from the loss or the representation, which the single-room models have too.
- **Caveats.** (1) S-1 is 2 rooms, each with 5-6 matched peaks, and each trained for 10K steps with about 83 times R0's per-room exposure (D185): sharing and exposure per room are not separated here, which is Q25's open part. (2) R1's training rooms match only 36 of 63 truth peaks within 3 Hz (57%; per-room 1-4 peaks), so each room's ratio rests on few peaks (per-room values 1.06 to 2.46). (3) The S-1 floor is still -31 to -36 dB, similar to R1's; R1's is -26 to -35 (the truth's is -41 to -77): single-room training does not remove the late floor.
- R1 is broad in its own training rooms, so this is not a transfer problem: the training rooms' ratio (mean 1.73, median 1.67) is above the in-family test rooms' (mean 1.44, median 1.40; a different room family).

### E2. Reciprocal averaging on R1 (zero training; Source cells, all 31 rooms; frozen scorer)

| position (all 31 rooms) | (a) normal | (b) swapped | (c) average of the two | (c) - (a), 25-100 / 100-200 / 200-300 Hz |
|---|---|---|---|---|
| second source | 0.619 / 0.291 / 0.199 | 0.634 / 0.289 / 0.185 | 0.654 / 0.326 / 0.176 | +0.035 [+0.024, +0.047] / +0.036 [+0.019, +0.052] / -0.023 [-0.035, -0.013] |
| third source | 0.663 / 0.406 / 0.306 | 0.615 / 0.340 / 0.247 | 0.669 / 0.386 / 0.242 | +0.005 [-0.000, +0.011] / -0.019 [-0.024, -0.015] / -0.064 [-0.069, -0.059] |
| original source (no-harm check) | 0.701 / 0.541 / 0.482 | 0.607 / 0.316 / 0.230 | 0.691 / 0.480 / 0.389 | -0.010 [-0.020, +0.001] / -0.062 [-0.078, -0.045] / -0.093 [-0.110, -0.077] |

- **Pre-registered reading (D206 / D207): averaging is a P4-11 candidate: NO.** No gain at the third source above 100 Hz (it loses 0.019 and 0.064), and the original source loses more than 0.01 in two of three bands. Averaging helps the second source below 200 Hz (+0.035) but not above; the two renders differ too much at high frequency.

### E3. Is the late floor common to the room? (in-family 15, octaves 63 / 125 / 250 Hz, last 0.3 s of a 1.5 s window)

| | 63 Hz: model / truth ratio | 125 Hz | 250 Hz |
|---|---|---|---|
| R0 @120K | 0.084 / 0.039 (2.2) | 0.025 / 0.038 (0.6) | 0.012 / 0.018 (0.7) |
| R1 @150K | 0.093 / 0.039 (2.4) | 0.030 / 0.038 (0.8) | 0.021 / 0.018 (1.2) |
| R1-S @150K | 0.125 / 0.039 (3.2) | 0.030 / 0.038 (0.8) | 0.021 / 0.018 (1.2) |
| R1-M @150K | 0.121 / 0.039 (3.1) | 0.040 / 0.038 (1.0) | 0.018 / 0.018 (1.0) |

- The ratio is the energy of the receiver-mean signal in the late window over the mean per-receiver energy (1 = the same signal at every receiver; 1 / N, about 0.0003 here, for independent noise). **The models' ratio is clearly above the truth's only at 63 Hz (2-3 times);** at 125 and 250 Hz it is 0.6 to 1.2 times the truth's. So at 63 Hz part of the floor is common to the room (a structural component); at 125-250 Hz the late floor is mostly per-receiver noise with no excess common part over the truth.
- The 0d hint (the single-radius mean-value residual, removed by the constant-cancelling form) is therefore most relevant near 63 Hz.

Links: [E1 report](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/e1/report.md) · [E2 report](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/e2/report.md) · [E3 report](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/e3/report.md) · scripts [E1](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_e1_decay.py), [E2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_e2_recip_avg.py), [E3](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_e3_floor.py)

## 2026-11-24 (02:46) — I21 (R1-M +30K, the final Material cells; the full card follows): MM1 FAILS (slope 0.54, not 0.8), MM2 and MC1 HOLD, no extension

**Headline:** The arm trained on per-wall absorption (R1-M) finished its 30K steps. When a wall's absorption is changed, its decay time changes in the right direction and in strict proportion (Pearson 0.95), but by only 0.54 of the true amount in known rooms and 0.66 in new ones; the bar asks for 0.8 to 1.2. It adds no decay error beyond its own baseline and does no harm at the original material, so those two rules hold. R1-M is not extended. R0 and R1 read about 0.0 (no response) on the same measure.

**R1-M @150K, full Material set (60 in-family cells, 64 unseen; 180 and 192 cell-bands; D201 / D202; off-lattice; no cell-bands dropped):**

| | in-family | unseen |
|---|---|---|
| **MM1** EDT edit slope (Pearson) | **0.54 (0.95)** (bar [0.8, 1.2] with Pearson >= 0.8: **FAILS** on the slope) | 0.66 (0.94) |
| per octave 63 / 125 / 250 Hz | 0.48 / 0.58 / 0.54 | 0.48 / 0.77 / 0.59 |
| **MM2** edited vs own unedited |EDT error| (limit: at most +5 points) | 22.5% vs 24.6% (-2.1 points): **HOLDS** | 29.0% vs 29.4% (-0.4 points) |
| level-change slope (Pearson), descriptive | 0.45 (0.96) | 0.44 (0.96) |
| T20 slope (kept receivers only; 129 of 180 cell-bands had a usable T20) | 0.17 (0.46); 429477 of 543192 receiver-bands dropped by the floor rule | 0.43 (0.81); 370742 of 505819 dropped |

- **Reading of MM1:** the response is linear (Pearson 0.95) and about half the true size in every octave (0.48 to 0.58 in known rooms); in new rooms it is a little larger (0.66 pooled, 0.48 / 0.77 / 0.59 per octave).
- **MM2 absolute levels and JNDs (manager's note, D206):** 22.5% in known rooms and 29.0% in new ones after the edit, against 5% (ISO 3382-1) and about 18% measured for a listener; the unedited error is of the same size, so this bar cannot show that the field is close to the truth.
- **MC1 (no harm at alpha = 0.15, HF and each UF band >= R1 @150K - 0.02): HOLDS.** HF 0.568 against 0.571: -0.003 [-0.007, +0.001]; UF 0.522 / 0.429 / 0.407 against 0.506 / 0.409 / 0.399 (above in every band).
- **Learning check (+10K) passed, harm kill not triggered (earlier reads). Extension reading (D208): not met** (lite slope 0.62, Pearson 0.85; the reading needs a slope of at least 0.75). R1 has not been extended either, so nothing is.
- **Q28:** R1 stays the base; R1-M fails its primary (MM1).
- R1 @150K and R0 on the same full Material cells: EDT slope 0.01 / -0.04 and MM2 margins +12.6 / +14.4 points (both about 24% unedited), so R1-M is the first arm that responds.
- Not yet posted: the other families of R1-M's full card (scoring now), D1, D2 and A8 for R1-M.

Links: [verdict_R1M_150K.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1M_150K.md) · [verdict_R1M_150K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1M_150K.json) · [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-24 (01:12) — I20 (R1-M +25K, Material lite): slope 0.57 in known rooms, 0.79 in new ones; MM1 is judged at the next read

**Headline:** After 25K of 30K steps R1-M changes its decay time with the wall absorption by 0.57 of the true amount in known rooms (Pearson 0.80) and 0.79 in new ones (0.83), against 0.53 and 0.75 at +20K. The bar, judged at +30K, is 0.8 to 1.2 with Pearson at least 0.8. Its accuracy after a material edit is 0.921 / 0.756 / 0.676 (25-100 / 100-200 / 200-300 Hz), which beats "ignore the edit" at 25-100 Hz only (0.882).

**R1-M @145K, Material lite (16 in-family cells, 16 unseen; 48 cell-bands; JNDs 5% ISO 3382-1, about 18% measured):**

| set | EDT slope (Pearson) | per octave 63 / 125 / 250 Hz | EDT error edited vs own unedited (MM2 margin) | edited r (ignore the edit) | r_delta | g_delta |
|---|---|---|---|---|---|---|
| in-family, +20K -> +25K | 0.53 (0.77) -> 0.57 (0.80) | 0.73 / 0.88 / 0.54 | 19.1% vs 24.9% (-5.8 points) | 0.921 / 0.756 / 0.676 (0.882 / 0.849 / 0.826) | 0.587 / 0.237 / 0.156 | 1.76 / 1.17 / 1.10 |
| unseen, +20K -> +25K | 0.75 (0.84) -> 0.79 (0.83) | 0.49 / 0.60 / 0.59 | 26.8% vs 30.9% (-4.1 points) | 0.643 / 0.579 / 0.527 (0.898 / 0.854 / 0.828) | 0.294 / 0.155 / 0.095 | 1.15 / 1.03 / 1.05 |

- T20 is dropped in 84% of receiver-bands (123724 of 146689 in-family) and is not used.
- **Static (added 01:47, a minute past the one-hour mark), against R1 @145K:** HF 0.563 against 0.569: -0.006 [-0.012, +0.000]; UF 0.528 / 0.427 / 0.403 against 0.505 / 0.409 / 0.397: +0.022 [+0.011, +0.034] / +0.018 [+0.011, +0.025] / +0.006 [-0.005, +0.016]. MC1 is judged at +30K.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-24 (00:11) — I19 (R1-M +20K, all lite families): the material response holds at 0.53 in known rooms, 0.75 in new ones; its other edits are level with R1's

**Headline:** After 20K of 30K steps R1-M changes its decay time with the wall absorption by 0.53 of the true amount in known rooms (Pearson 0.77) and 0.75 in new ones (Pearson 0.84), against 0.61 and 0.77 at +15K. The in-family slope dropped 0.08 from the previous read (the lite set gives no error bar on the slope, so one dip is not a trend). Its accuracy after a material edit is 0.915 / 0.749 / 0.671 (25-100 / 100-200 / 200-300 Hz), above R1's 0.686 / 0.509 / 0.435 and above "ignore the edit" at 25-100 Hz only. Outside material edits it is level with R1 at the same step.

**R1-M @140K, Material lite (16 in-family cells, 16 unseen; 48 cell-bands; D201 / D202; JNDs 5% ISO 3382-1, about 18% measured):**

| set | EDT slope (Pearson) | per octave 63 / 125 / 250 Hz | EDT error edited vs own unedited (MM2 margin) | edited r (ignore the edit) | r_delta | g_delta |
|---|---|---|---|---|---|---|
| in-family | 0.53 (0.77) | 0.72 / 0.82 / 0.45 | 19.2% vs 24.8% (-5.6 points) | 0.915 / 0.749 / 0.671 (0.882 / 0.849 / 0.826) | 0.577 / 0.230 / 0.153 | 1.76 / 1.19 / 1.11 |
| unseen | 0.75 (0.84) | 0.50 / 0.58 / 0.61 | 27.6% vs 30.9% (-3.3 points) | 0.638 / 0.573 / 0.531 (0.898 / 0.854 / 0.828) | 0.290 / 0.150 / 0.092 | 1.17 / 1.04 / 1.05 |
| R1 @140K, in-family | -0.10 (-0.31) | -- | 38.4% vs 23.5% (+14.9) | 0.686 / 0.509 / 0.435 | 0.131 / 0.027 / -0.000 | 0.79 / 0.78 / 0.78 |

- **Extension baseline (D208, registered before this read's numbers were used for any decision):** slope 0.53, Pearson 0.77. Nothing is extended before +30K.
- **Other edits, in-family, R1-M / R1 at the same step:** Stretch-U 0.759 / 0.530 / 0.421 against 0.758 / 0.524 / 0.407, Stretch-A 0.816 / 0.579 / 0.484 (0.817 / 0.584 / 0.481), Wall 0.889 / 0.630 / 0.541 (0.895 / 0.639 / 0.544), Notch 0.894 / 0.648 / 0.540 (0.899 / 0.656 / 0.543), Source 0.707 / 0.354 / 0.256 (0.720 / 0.352 / 0.250).
- **Static (added 00:46; six minutes past the one-hour mark, its analysis jobs finished late), against R1 @140K:** HF 0.557 against 0.561: -0.004 [-0.008, +0.000]; UF 0.524 / 0.422 / 0.399 against 0.498 / 0.407 / 0.393: +0.026 [+0.014, +0.038] / +0.015 [+0.007, +0.022] / +0.006 [-0.005, +0.017]. MC1 is judged at +30K.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md) · [D208 in DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md)

## 2026-11-23 (23:40) — R1-S's D1, D2 and A8 at its final checkpoint: the decay is the same as R1's; the mode shifts are a little better on uniform stretches

**Headline:** R1-S's sound decays too fast by the same amount as R1's and R0's (15% / 24% / 29% EDT error in known rooms at 63 / 125 / 250 Hz), so training on stretched rooms does not touch the decay problem. Its mode-shift slopes after an edit (0.69 for uniform stretches and 0.68 for walls in known rooms) are close to R1's (0.60 and 0.60) and to R0's (0.65 and 0.71).

**D1 (R1-S @150K):** in-family 15.2% / 24.1% / 29.2% (bias -15.0 / -22.6 / -28.5), unseen 23.4% / 28.4% / 31.6%; R1 in-family 14.1% / 24.2% / 29.6%, R0 12.6% / 23.1% / 28.3%. Floor -35 / -30 / -26 dB (R1 -35 / -29 / -27, R0 -37 / -31 / -27). **D2:** peak width 1.43 times the truth's (2.89 against 4.62 Hz; 14 rooms; 37 of 65 truth peaks matched; R1 1.40 on 13 rooms with 33 matched, R0 1.50 on 14 with 43). **A8 (mode-shift slope, full geometry cells, 25-150 Hz), R1-S / R1 / R0:** in-family Stretch-U 0.69 / 0.60 / 0.65, Stretch-A 0.52 / 0.58 / 0.54, Wall 0.68 / 0.60 / 0.71, Notch 0.40 / 0.30 / 0.25; unseen Stretch-U 0.70 / 0.51 / 0.36, Stretch-A 0.38 / 0.49 / 0.50, Wall 0.51 / 0.65 / 0.29, Notch 0.13 / -0.01 / 0.04. The pairs per cell are 81-165 of 431-473 truth peaks. Descriptive (SS2 reads the slope against R1's; R1-S's is higher for uniform stretches and notch, not for axis stretches).

Links: [d1/report.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/d1/report.md) · [modes/d2.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/d2.json) · [a8_R1S_150K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/a8_R1S_150K.json)

## 2026-11-23 (23:40) — I18 (R1-S +30K, the full scorecard): R1-S is ahead of R1 in every edit family but material, and of R0 on every stretch and wall cell

**Headline:** The full scorecard of the stretched-room arm is scored. At the same training step R1-S is ahead of R1 on stretch, wall and source edits and level with it on notch and material (within 0.01 in edited-field r): on stretch edits 0.863 / 0.629 / 0.522 against 0.825 / 0.599 / 0.492 in known rooms (25-100 / 100-200 / 200-300 Hz), and the wall and source cells gain a little too. It is ahead of the original model R0 on every stretch cell (both room sets) and on unseen walls; on walls in known rooms it is level with R0 (0.899 / 0.667 / 0.565 against 0.900 / 0.673 / 0.562). The pre-registered primary (SS1) still fails, because its bar is on the pattern-of-change score, where the gain over R1 is +0.032 / +0.017 / +0.012 and not 0.05.

R1-S @150K against R1 @150K and R0 @120K, off-lattice, full set (15 in-family, 16 unseen rooms); edited-field r and r_delta (25-100 / 100-200 / 200-300 Hz):

| family | set | R1-S | R1 | R0 | R1-S r_delta (R1) |
|---|---|---|---|---|---|
| Stretch-U | in-family | 0.863 / 0.629 / 0.522 | 0.825 / 0.599 / 0.492 | 0.833 / 0.606 / 0.493 | 0.828 / 0.590 / 0.492 (0.801 / 0.573 / 0.478) |
| Stretch-U | unseen | 0.492 / 0.409 / 0.389 | 0.442 / 0.356 / 0.340 | 0.402 / 0.325 / 0.304 | 0.325 / 0.264 / 0.308 (0.287 / 0.230 / 0.278) |
| Stretch-A | in-family | 0.872 / 0.633 / 0.527 | 0.832 / 0.609 / 0.494 | 0.835 / 0.613 / 0.495 | 0.788 / 0.552 / 0.328 (0.751 / 0.536 / 0.317) |
| Stretch-A | unseen | 0.499 / 0.410 / 0.396 | 0.461 / 0.373 / 0.359 | 0.416 / 0.347 / 0.327 | 0.301 / 0.199 / 0.163 (0.267 / 0.180 / 0.149) |
| Wall | in-family | 0.899 / 0.667 / 0.565 | 0.891 / 0.658 / 0.557 | 0.900 / 0.673 / 0.562 | 0.732 / 0.491 / 0.454 (0.721 / 0.486 / 0.448) |
| Wall | unseen | 0.505 / 0.417 / 0.401 | 0.483 / 0.392 / 0.372 | 0.438 / 0.363 / 0.347 | 0.244 / 0.175 / 0.245 (0.218 / 0.158 / 0.223) |
| Notch | in-family | 0.905 / 0.682 / 0.561 | 0.906 / 0.681 / 0.560 | 0.917 / 0.697 / 0.568 | 0.617 / 0.310 / 0.213 (0.614 / 0.310 / 0.209) |
| Notch | unseen | 0.512 / 0.419 / 0.411 | 0.503 / 0.409 / 0.399 | 0.464 / 0.380 / 0.372 | 0.077 / 0.023 / 0.018 (0.077 / 0.022 / 0.018) |
| Source (both) | in-family | 0.755 / 0.383 / 0.266 | 0.743 / 0.377 / 0.260 | 0.231 / 0.083 / 0.044 | 0.783 / 0.520 / 0.409 (0.776 / 0.518 / 0.405) |
| Source (both) | unseen | 0.547 / 0.328 / 0.250 | 0.546 / 0.321 / 0.246 | 0.212 / 0.110 / 0.062 | 0.473 / 0.378 / 0.326 (0.469 / 0.367 / 0.317) |
| Material | in-family | 0.749 / 0.558 / 0.484 | 0.749 / 0.567 / 0.476 | 0.772 / 0.583 / 0.469 | 0.091 / 0.026 / 0.020 (0.094 / 0.023 / 0.017) |
| Material | unseen | 0.455 / 0.385 / 0.397 | 0.447 / 0.376 / 0.373 | 0.428 / 0.357 / 0.347 | 0.045 / -0.002 / 0.002 (0.044 / -0.001 / -0.006) |

- **Reading the table:** R1-S beats R1 in stretch edits in both room sets in every band (edited r by 0.02 to 0.05), in wall edits (by about 0.01 in known rooms and 0.02-0.03 in new ones) and slightly in source cells; it is level on notch and material (within 0.01). The SS1 bar asks for +0.05 in r_delta: the in-family gain is 0.03 in the lowest band (pooled Stretch-U and Stretch-A r_delta +0.032 / +0.017 / +0.012).
- **R1-S's MM1 / MM2 on its material cells (it has no material training):** EDT slope 0.01 in-family and 0.00 unseen; MM2 margin +11.8 points in-family: as for R1.
- Ignoring the edit still beats both models in the notch, material and unseen wall / stretch-A cells listed in I15.

Links: [scorecard_R1S_full.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1S_full.md) · [scorecard_R1S_full.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1S_full.json)

## 2026-11-23 (23:33) — I17 (R1-S +30K, the final read of the stretched-room arm): SS1 FAILS, SC1 HOLDS, no extension; no arm is extended

**Headline:** The arm that trains on stretched rooms (R1-S) finished its 30K steps. It predicts stretch edits a little better than R1 at the same step, in every band, but not by the +0.05 in two bands the pre-registration asks for: +0.032 / +0.017 / +0.012 (25-100 / 100-200 / 200-300 Hz) in known rooms, every interval above zero. Its accuracy at the original source is unchanged, so it does no harm. Its gain did not grow enough in the last 10K steps to extend the run, and no other arm extends either, so R1 stays at 150K.

Off-lattice, full set (15 in-family and 16 unseen rooms), Stretch-U and Stretch-A pooled per room, paired bootstrap 10k.

| stretch r_delta | R1-S @150K | R1 @150K | R1-S - R1 [95% CI] |
|---|---|---|---|
| in-family (SS1) | 0.808 / 0.571 / 0.410 | 0.776 / 0.554 / 0.398 | +0.032 [+0.018, +0.047] / +0.017 [+0.010, +0.025] / +0.012 [+0.007, +0.018] |
| unseen (descriptive) | 0.313 / 0.231 / 0.236 | 0.277 / 0.205 / 0.213 | +0.036 [+0.024, +0.050] / +0.026 [+0.020, +0.032] / +0.022 [+0.016, +0.029] |

- **SS1 (+0.05 with CI above zero in at least 2 of 3 bands): FAILS** (0 of 3 bands at +0.05; all three intervals are above zero).
- **SC1 (HF and each UF band >= R1 - 0.02): HOLDS.** HF 0.572 against 0.571: +0.001 [-0.003, +0.004].
- **Extension rule (D201 Q27; 8 lite rooms):** second-best band +0.0225 (>= 0.00, met), band mean +0.0304 against +0.0215 at +20K: rise +0.0089, needs +0.02: **not triggered.** R1 is therefore not continued either.
- **Base model (Q28): stays R1.** R1-S fails its primary, so it does not become the base (Q28: an arm must pass its primary and its no-harm bar).
- **Against R0 in known rooms: see I18 for the final read** (at +15K the r_delta of R1-S was below R0's above 100 Hz; at the final read it is not, and its edited accuracy on stretch cells is above R0's in every band).

Links: [verdict_R1S_150K.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1S_150K.md) · [verdict_R1S_150K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1S_150K.json)

## 2026-11-23 (22:57) — I16 (R1-M +15K, Material lite): the material response grows to 0.61 of the true amount in known rooms

**Headline:** After 15K of 30K steps R1-M changes its decay time with the wall absorption by 0.61 of the true amount in known rooms (Pearson 0.84; 0.48 at +10K, 0.36 at +5K) and 0.77 in new ones (0.83). The bar, judged at +30K, is 0.8 to 1.2 with Pearson at least 0.8. Its accuracy after a material edit is 0.913 / 0.742 / 0.649, above "ignore the edit" at 25-100 Hz (0.882) only.

**R1-M @135K, Material lite (16 in-family cells, 16 unseen; 48 cell-bands; D201 / D202):**

| set | EDT slope (Pearson) | per octave 63 / 125 / 250 Hz | EDT error edited vs own unedited (MM2 margin) | edited r (ignore the edit) | r_delta | g_delta |
|---|---|---|---|---|---|---|
| in-family | 0.61 (0.84) | 0.61 / 0.84 / 0.54 | 21.1% vs 26.2% (-5.1 points) | 0.913 / 0.742 / 0.649 (0.882 / 0.849 / 0.826) | 0.569 / 0.221 / 0.127 | 1.70 / 1.13 / 1.05 |
| unseen | 0.77 (0.83) | 0.32 / 0.67 / 0.52 | 28.5% vs 31.7% (-3.2 points) | 0.635 / 0.581 / 0.525 (0.898 / 0.854 / 0.828) | 0.281 / 0.149 / 0.082 | 1.16 / 1.05 / 1.05 |

- JNDs beside the absolute EDT errors (Rule 12, manager's note): 5% (ISO 3382-1), about 18% measured; R1-M's edited error is 21% in known rooms and 28% in new ones.
- The per-octave slopes are uneven (0.54 at 250 Hz in-family, 0.32 at 63 Hz unseen); the pooled slope mixes them. T20 is dropped in 93% of receiver-bands (135742 of 146689) and is not used.
- **Static (added 23:18), against R1 @135K:** HF 0.541 against 0.555: -0.014 [-0.020, -0.008]; UF 0.518 / 0.429 / 0.402 against 0.497 / 0.412 / 0.398: +0.021 [+0.011, +0.032] / +0.017 [+0.010, +0.025] / +0.004 [-0.007, +0.015]. MC1 (HF and UF >= R1 - 0.02) is judged at +30K.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-23 (22:49) — R1's one-time confirmation read (P4-9 set, D201 Q32): R1 beats R0 in every band, beats S2X only at 25-100 Hz

**Headline:** On the 16 new rooms that no model was tuned on, R1 scores 0.644 / 0.469 / 0.416 (25-100 / 100-200 / 200-300 Hz) against R0's 0.607 / 0.447 / 0.400 and S2X's 0.605 / 0.466 / 0.430. R1 is ahead of R0 by 0.037 / 0.022 / 0.015 (all intervals above zero), ahead of S2X by 0.039 at the lowest band and level with it above. This read was run once, at R1's final checkpoint, as pre-registered; no P4-10 choice is made from it.

Off-lattice, raw, unseen rooms (8 U + 8 C3), manifest sha 688e319f18f6dd32 (frozen before any score), same protocol as the S2X / R0 reads of 18:54; paired bootstrap over rooms (10k, seed 20261013).

| | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| R1 @150K | 0.644 | 0.469 | 0.416 |
| S2X @240K (reference level) | 0.605 | 0.466 | 0.430 |
| R0 @120K (reference level) | 0.607 | 0.447 | 0.400 |
| **R1 - R0** | +0.037 [+0.020, +0.055] | +0.022 [+0.008, +0.039] | +0.015 [+0.006, +0.024] |
| **R1 - S2X** | +0.039 [+0.022, +0.057] | +0.003 [-0.012, +0.016] | -0.014 [-0.041, +0.009] |
| S2X - R0 | -0.002 [-0.012, +0.008] | +0.020 [+0.003, +0.037] | +0.029 [+0.008, +0.052] |

- Rooms where R1 is better than R0: 13 / 12 / 11 of 16; than S2X: 14 / 8 / 7 of 16.
- By family (R1 / R0 / S2X): U 0.662 / 0.459 / 0.452, 0.643 / 0.447 / 0.445, 0.636 / 0.464 / 0.461; C3 0.626 / 0.479 / 0.379, 0.570 / 0.447 / 0.356, 0.573 / 0.469 / 0.399.
- On the frozen unseen set R1 was ahead of S2X at 25-100 Hz (0.506 against 0.495) and level at 100-200 and 200-300 Hz (0.409 and 0.399 against 0.403 and 0.398); the fresh rooms agree at the lowest band and 100-200 Hz, and show S2X ahead at 200-300 Hz by a margin whose interval includes zero.

Links: [R1_150K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_9/confirm/R1_150K.json)

## 2026-11-23 (22:46) — R1's RS3 (reciprocity) FAILS; the four R1 verdicts are complete: RS1, RS2, RS3 fail, RC1 holds

**Headline:** When the model renders the field with the receiver and the source swapped, R1's accuracy drops from 0.909 / 0.683 / 0.571 to 0.759 / 0.344 / 0.222 (25-100 / 100-200 / 200-300 Hz, known rooms). The bar was "no more than 0.05 below the normal render". R1's swapped-render accuracy is the same as the swap-only model's (RCP, 0.758 / 0.343 / 0.227), so training on four source positions per room improved moved sources but not reciprocity beyond what the swaps gave. The plain models (R0, CTL) are far worse (0.17 / 0.05 / 0.03).

| model | in-family 15: normal / swapped (25-100, 100-200, 200-300 Hz) | unseen 16: normal / swapped |
|---|---|---|
| R1 @150K | 0.909 / 0.759, 0.683 / 0.344, 0.571 / 0.222 | 0.506 / 0.465, 0.409 / 0.289, 0.399 / 0.238 |
| RCP @150K | 0.920 / 0.758, 0.705 / 0.343, 0.582 / 0.227 | 0.478 / 0.434, 0.385 / 0.290, 0.379 / 0.229 |
| CTL @150K | 0.920 / 0.170, 0.708 / 0.050, 0.583 / 0.034 | 0.467 / 0.121, 0.376 / 0.052, 0.373 / 0.028 |
| R0 @120K | 0.918 / 0.167, 0.701 / 0.050, 0.582 / 0.027 | 0.468 / 0.122, 0.380 / 0.054, 0.371 / 0.027 |

- **RS3 (swapped >= normal - 0.05, in family, every band): FAILS.** Off-lattice interior receivers, r against the same truth (D193's measure).
- R1's verdict file now carries all four rules: RS1 FAILS, RS2 FAILS, RS3 FAILS, RC1 HOLDS, extension not triggered, base = R1 (Q28).

Links: [verdict_R1_150K.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1_150K.md)

## 2026-11-23 (22:24) — R1's D1, D2 and A8 at its final checkpoint: the decay problem is unchanged (slightly worse), mode shifts are similar to R0's

**Headline:** R1's sound dies away about as fast as R0's: in known rooms the early decay time is 14% / 24% / 30% off (63 / 125 / 250 Hz) against R0's 13% / 23% / 28%, with a noise floor near -27 to -35 dB, so the source fix did not touch this. How well R1 predicts the shift of resonance frequencies after an edit is much the same as R0's, except in new rooms, where wall and uniform-stretch edits are now better (slopes 0.65 and 0.51 against 0.29 and 0.36); the other rows (all in-family, axis-stretch and notch) differ from R0 by at most 0.11 (the largest: Wall in-family, 0.60 against 0.71).

**D1 (decay on the original rooms, off-lattice receivers at least 1 m from the source; EDT error = median relative error, bias signed):**

| R1 @150K | 63 Hz: EDT error (bias) | 125 Hz | 250 Hz | floor, dB (63 / 125 / 250) |
|---|---|---|---|---|
| in-family | 14.1% (-13.8%) | 24.2% (-22.6%) | 29.6% (-28.8%) | -35 / -29 / -27 |
| unseen | 23.4% (-22.7%) | 27.5% (-26.5%) | 30.3% (-29.2%) | -35 / -30 / -27 |
| R0 @120K (in-family, for reference) | 12.6% (-12.1%) | 23.1% (-22.0%) | 28.3% (-27.7%) | -37 / -31 / -27 |

T20 (kept only where the model's decay clears its floor): 192 of 11421 in-family receivers at 125 Hz and none at 250 Hz, so it is not reported.

**D2 (half-power widths of the first 25-100 Hz peaks; 13 usable rooms for R1):** R1's peaks are 1.40 times as wide as the truth (2.91 Hz against 4.87 Hz medians; R0 1.50, CTL 1.48, RCP 1.49, S2X 1.50 on 14 rooms). **R1 matches fewer of the truth's peaks within 3 Hz (33 of 65, against R0's 43),** so the width comparison uses fewer peaks. The per-bin field error at 25-100 Hz is -14.6 dB for R1 (R0 -15.4 dB).

**A8 (mode-shift slope, full geometry cells, 25-150 Hz; the 25-100 Hz reading is in the JSON; R0's on the same cells, pooled in the same way):**

| | Stretch-U | Stretch-A | Wall | Notch |
|---|---|---|---|---|
| R1 in-family: slope (Pearson) | 0.60 (0.64) | 0.58 (0.63) | 0.60 (0.57) | 0.30 (0.30) |
| R0 in-family | 0.65 (0.71) | 0.54 (0.56) | 0.71 (0.69) | 0.25 (0.22) |
| R1 unseen | 0.51 (0.52) | 0.49 (0.45) | 0.65 (0.71) | -0.01 (-0.01) |
| R0 unseen | 0.36 (0.39) | 0.50 (0.46) | 0.29 (0.36) | 0.04 (0.06) |

- The mode shifts move the right way by about 0.5 to 0.7 of the true amount for stretch and wall edits, and hardly at all for notches. The matched pairs are 124 to 153 of 431-440 in-family truth peaks (R1), so the slopes describe the peaks that could be paired (see the caution in I10).
- R1-M and R1-S join at their final checkpoints.

Links: [d1/report.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/d1/report.md) · [modes/d2.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/d2.json) · [a8_R1_150K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/a8_R1_150K.json)

## 2026-11-23 (22:17) — I15 (R1 +30K, the rest of the full scorecard; R1-M harm kill and R1-S static): R1 improves every new-room edit over R0 and keeps its known-room edits within 0.01-0.02 of R0's

**Headline:** R1's full scorecard (675 cells, all six families) is scored. Compared with the original model R0, R1 does better on every edit family in new rooms (edited-field accuracy at 25-100 Hz up by 0.04 to 0.05 on stretch, wall and notch, 0.02 on material), is within 0.016 of R0 on geometry edits in known rooms, and is far better on source moves. Its decay behaviour under material edits is unchanged from R0, as expected: R1 saw no material labels. The materials arm R1-M does no harm at its first 10K steps.

R1 @150K against R0 @120K, off-lattice, 15 in-family and 16 unseen rooms, edited-field r (25-100 / 100-200 / 200-300 Hz), "ignore the edit" beside:

| family | set | R1 | R0 | ignore the edit | R1 r_delta (R0) |
|---|---|---|---|---|---|
| Stretch-U | in-family | 0.825 / 0.599 / 0.492 | 0.833 / 0.606 / 0.493 | 0.410 / 0.171 / 0.119 | 0.801 / 0.573 / 0.478 (0.818 / 0.589 / 0.486) |
| Stretch-U | unseen | 0.442 / 0.356 / 0.340 | 0.402 / 0.325 / 0.304 | 0.440 / 0.153 / 0.103 | 0.287 / 0.230 / 0.278 (0.253 / 0.203 / 0.249) |
| Stretch-A | in-family | 0.832 / 0.609 / 0.494 | 0.835 / 0.613 / 0.495 | 0.514 / 0.250 / 0.281 | 0.751 / 0.536 / 0.317 (0.766 / 0.552 / 0.327) |
| Stretch-A | unseen | 0.461 / 0.373 / 0.359 | 0.416 / 0.347 / 0.327 | 0.544 / 0.255 / 0.255 | 0.267 / 0.180 / 0.149 (0.224 / 0.164 / 0.135) |
| Wall | in-family | 0.891 / 0.658 / 0.557 | 0.900 / 0.673 / 0.562 | 0.735 / 0.421 / 0.233 | 0.721 / 0.486 / 0.448 (0.743 / 0.506 / 0.458) |
| Wall | unseen | 0.483 / 0.392 / 0.372 | 0.438 / 0.363 / 0.347 | 0.752 / 0.413 / 0.213 | 0.218 / 0.158 / 0.223 (0.179 / 0.144 / 0.210) |
| Notch | in-family | 0.906 / 0.681 / 0.560 | 0.917 / 0.697 / 0.568 | 0.885 / 0.715 / 0.534 | 0.614 / 0.310 / 0.209 (0.637 / 0.328 / 0.216) |
| Notch | unseen | 0.503 / 0.409 / 0.399 | 0.464 / 0.380 / 0.372 | 0.916 / 0.760 / 0.624 | 0.077 / 0.022 / 0.018 (0.067 / 0.019 / 0.019) |
| Source (both positions) | in-family | 0.743 / 0.377 / 0.260 | 0.231 / 0.083 / 0.044 | 0.305 / 0.217 / 0.215 | 0.776 / 0.518 / 0.405 (0.504 / 0.391 / 0.304) |
| Source (both positions) | unseen | 0.546 / 0.321 / 0.246 | 0.212 / 0.110 / 0.062 | 0.344 / 0.246 / 0.209 | 0.469 / 0.367 / 0.317 (0.283 / 0.253 / 0.226) |
| Material | in-family | 0.749 / 0.567 / 0.476 | 0.772 / 0.583 / 0.469 | 0.911 / 0.891 / 0.880 | 0.094 / 0.023 / 0.017 (0.067 / 0.026 / 0.017) |
| Material | unseen | 0.447 / 0.376 / 0.373 | 0.428 / 0.357 / 0.347 | 0.921 / 0.897 / 0.884 | 0.044 / -0.001 / -0.006 (0.027 / -0.006 / -0.000) |

- **Where "ignore the edit" still beats R1, the model adds error:** Notch, in-family at 100-200 Hz (0.715 against 0.681) and in every unseen band; Wall unseen at 25-100 and 100-200 Hz; Stretch-A unseen at 25-100 Hz (0.544 against 0.461); Material in every band of both sets; Stretch-U unseen at 25-100 Hz is level (0.440 against 0.442). On the source cells R1 is above ignoring the move in every band of both sets (in-family 0.743 / 0.377 / 0.260 against 0.305 / 0.217 / 0.215).
- **R1's decay under material edits (MM1 / MM2 as re-registered), full set:** EDT slope 0.01 (Pearson 0.06) in-family, -0.05 unseen; MM2 margin +12.6 points in-family (edited 36.5% against 23.9% unedited), +7.3 unseen: as R0's (slope -0.04; margin +14.4). R1 was not trained on materials.
- **R1-M @130K (+10K), harm kill: NOT triggered** (against R1 @130K): HF -0.010 [-0.014, -0.005], UF +0.017 [+0.006, +0.028] / +0.014 [+0.004, +0.025] / +0.003; levels HF 0.541, UF 0.509 / 0.413 / 0.396 (R1 @130K 0.550, 0.492 / 0.398 / 0.393). MC1's -0.02 limit is judged at +30K.
- **R1-S @145K (+25K), static, against R1 @145K:** HF +0.000 [-0.004, +0.004], UF +0.003 / +0.009 / +0.006 (levels HF 0.569, UF 0.509 / 0.418 / 0.403). Descriptive.

Links: [scorecard_R1_full.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1_full.md) · [scorecard_R1_full.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1_full.json) · [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-23 (22:10) — I14 (R1 +30K, the final read, Source family): RS1 FAILS, RS2 FAILS, RC1 HOLDS; no extension; R1 is the base model (D201 Q28); RS3 to follow

**Headline:** R1, the source-fix model, finished its 30K steps and its Source cells are scored on all 31 rooms. It can move a source far better than any earlier model, but not to the pre-registered standard: the gap between its accuracy at a moved source and at the original one is above 0.10 in all six band-and-source cells (the smallest is 0.128, the worst 0.373). It did not hurt the original room's accuracy beyond the allowed margin. R1 therefore stays at 150K, and under the manager's rule it is the base model for P4-11 and the meeting, with R0 kept as the fallback that was not needed.

Off-lattice, dense receivers, 15 in-family and 16 unseen rooms, paired bootstrap over rooms (10k, seed 20261013). Static reference levels: R1 at the original source 0.909 / 0.683 / 0.571 in-family, 0.506 / 0.409 / 0.399 unseen. (Full set: the earlier reads used 8 lite rooms, so the numbers are not comparable one to one.)

| moved source | in-family r | in-family gap (RS1: each <= 0.10) | unseen r | "ignore the move" (in-family / unseen) |
|---|---|---|---|---|
| second | 0.705 / 0.309 / 0.199 | 0.203 [0.163, 0.247] / 0.373 [0.321, 0.427] / 0.372 [0.339, 0.410] | 0.538 / 0.273 / 0.200 | 0.311 / 0.211 / 0.247 ; 0.357 / 0.204 / 0.210 |
| third | 0.780 / 0.445 / 0.321 | 0.128 [0.110, 0.146] / 0.238 [0.216, 0.256] / 0.250 [0.234, 0.266] | 0.554 / 0.369 / 0.292 | 0.298 / 0.223 / 0.183 ; 0.332 / 0.287 / 0.208 |

- **RS1 (strict, every one of the six gaps at most 0.10): FAILS** (smallest gap 0.128, worst 0.373). Unseen gaps beside (static r minus moved r; a negative gap means the moved source scores higher): second -0.031 / 0.136 / 0.199, third -0.048 / 0.040 / 0.106.
- **RS2 (unseen moved-source r at least ignore-the-move + 0.05, every band, both sources): FAILS** at one of the six cells: second source, 200-300 Hz (-0.010 [-0.038, +0.017]); the other five are above +0.05 in the means: second +0.181 [+0.109, +0.249] / +0.069 [+0.025, +0.113], third +0.222 [+0.136, +0.313] / +0.082 [+0.052, +0.116] / +0.084 [+0.058, +0.111].
- **RC1 (no harm at the original source): HOLDS.** HF 0.571 against CTL @150K 0.583: -0.013 [-0.019, -0.007] (limit -0.02); UF 0.506 / 0.409 / 0.399 against 0.467 / 0.376 / 0.373 (above in every band).
- **Extension rule (D201 Q27, the same 8 lite rooms as before): NOT triggered.** Worst gap 0.366 (limit 0.15) and the mean of the six fell by 0.0127 from +20K (0.2677 to 0.2550; the rule needs 0.02 and a worst of 0.15 or less). R1 stops at 150K, its final checkpoint.
- **Q28 (base model, D201): R1 is the base.** The fallback to R0 is not triggered: on the lite third source, R1 now reads 0.792 / 0.435 / 0.315, above its +15K values in every band (0.748 / 0.407 / 0.283), and its HF cost against CTL @150K is 0.013 (limit 0.04).
- **Pending:** RS3 (reciprocity); the rest of the full scorecard; D1, D2 and the decay read for R1; R1's one-time read on the P4-9 confirmation set (running now, beside S2X @240K and R0 @120K; D201 Q32).

Links: [verdict_R1_150K.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1_150K.md) · [verdict_R1_150K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1_150K.json)

## 2026-11-23 (21:59) — I13 (R1-M +10K, all lite families; R1-S +25K stretch): R1-M passes its learning check and its Material accuracy now exceeds "ignore the edit" at the lowest band

**Headline:** After 10K of 30K steps R1-M changes its decay time with the wall absorption at 0.48 of the true amount in known rooms (Pearson 0.78) and 0.74 in new ones (a correct model reads near 1; the +30K bar is 0.8 to 1.2). **The pre-registered learning check (decay-rate slope at least 0.3 at +10K) passes.** Its field accuracy after a material edit is 0.901 / 0.716 / 0.649 (25-100 / 100-200 / 200-300 Hz), against 0.687 / 0.493 / 0.401 for R1 at the same step and 0.882 / 0.849 / 0.826 for "ignore the edit": it beats ignoring the edit at the lowest band only.

**R1-M @130K (+10K), Material lite (16 in-family cells, 16 unseen; 48 cell-bands each; 0 cell-bands dropped):**

| set | EDT slope (Pearson) | per octave 63 / 125 / 250 Hz | level-change slope | EDT error edited vs own unedited (MM2 margin) | edited r (ignore the edit) | r_delta | g_delta |
|---|---|---|---|---|---|---|---|
| in-family, +5K -> +10K | 0.36 (0.59) -> **0.48 (0.78)** | 0.66 / 0.76 / 0.40 | 0.32 (0.46) | 21.1% vs 26.3% (-5.2) | 0.901 / 0.716 / 0.649 (0.882 / 0.849 / 0.826) | 0.535 / 0.195 / 0.114 | 1.64 / 1.12 / 1.06 |
| unseen, +5K -> +10K | 0.60 (0.68) -> **0.74 (0.79)** | 0.42 / 0.56 / 0.57 | 0.37 (0.63) | 26.9% vs 32.7% (-5.8) | 0.616 / 0.560 / 0.519 (0.898 / 0.854 / 0.828) | 0.257 / 0.135 / 0.073 | 1.16 / 1.05 / 1.04 |
| R1 @130K in-family (reference) | -0.17 (-0.51) | -0.03 / -0.07 / -0.21 | 0.27 (0.25) | 37.2% vs 25.0% (+12.2) | 0.687 / 0.493 / 0.401 | 0.122 / 0.027 / -0.008 | 0.82 / 0.78 / 0.77 |

- **MM2 margin** (edited minus own unedited, bar at most +5 points): -5.2 and -5.8, so the edit adds no decay error; the unedited level itself is 26-33%. T20 is dropped in about 95% of receiver-bands (139866 of 146689 in-family) and is not used.
- **Harm kill (+10K, against R1 @130K): not judged yet**, the static HF / UF report is still running. On the other edit families R1-M is close to R1 at the same step, in-family (differences up to 0.021, the largest at Notch 100-200 Hz): Stretch-U 0.757 / 0.520 / 0.408 (R1 0.749 / 0.518 / 0.402), Stretch-A 0.813 / 0.565 / 0.473 (0.807 / 0.575 / 0.476), Wall 0.881 / 0.606 / 0.524 (0.888 / 0.624 / 0.536), Notch 0.885 / 0.626 / 0.528 (0.895 / 0.647 / 0.533), Source 0.646 / 0.329 / 0.237 (0.650 / 0.328 / 0.240).
- **R1-S @145K (+25K), stretch cells (descriptive; the reference R1 is at 140K, a step behind):** r_delta in-family 0.823 / 0.557 / 0.364, +0.051 [+0.031, +0.072] / +0.019 [+0.004, +0.033] / +0.021 [+0.011, +0.030]; unseen 0.385 / 0.287 / 0.228, +0.032 / +0.018 / +0.016. The +30K read is the verdict.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-23 (21:06) — I12 (R1-M +5K, first read; static reads of R1 +25K and R1-S +20K): the materials arm is the first model that answers a material change; R1's cost at the original source is now inside the no-harm limit

**Headline:** R1-M, the arm trained with per-wall absorption labels, at its first checkpoint (+5K of 30K) already changes its decay time when a wall's absorption is changed: the EDT edit slope is 0.36 in known rooms and 0.60 in new ones, against about -0.1 to -0.2 for R0 and R1 (a correct model reads near 1; the bar, read at +30K, is 0.8 to 1.2). Separately, R1's accuracy at the original source in known rooms is 0.015 below the plain continuation at 200-300 Hz, inside RC1's limit of 0.020 for the first time, and it stays ahead in unseen rooms.

**R1-M @125K, Material lite cells (D201 / D202; 16 in-family cells, 48 cell-bands; 16 unseen):**

| set | EDT slope (Pearson) | per octave 63 / 125 / 250 Hz | T20 slope (kept receivers) | level-change slope (Pearson) | EDT error edited vs own unedited (MM2 margin) |
|---|---|---|---|---|---|
| in-family | 0.36 (0.59) | 0.57 / 0.74 / 0.37 | 0.81, n 17 cell-bands (142582 of 146689 receiver-bands dropped) | 0.40 (0.50) | 22.7% vs 27.9% (-5.2) |
| unseen | 0.60 (0.68) | 0.43 / 0.53 / 0.52 | -0.19, n 18 (124793 of 129438) | 0.42 (0.54) | 27.4% vs 33.2% (-5.8) |
| R1 @140K (reference, no material training) | -0.10 (-0.31) | -- | -- | 0.25 (0.25) | 38.4% vs 23.5% (+14.9) |

- **The T20 slope is not usable here:** the floor rule drops about 97% of receiver-bands, as for the other arms; EDT is MM1.
- **MM2:** the edited-room error is BELOW the model's own unedited error (the bar is at most +5 points above it), so the edit adds no decay error so far. The unedited level itself, 27.9%, is above R1's (25.0% at +10K) and far above the 5% to 18% JNDs.
- **MC1 (no harm at alpha = 0.15; read at +30K):** R1-M @125K - R1 @125K: HF -0.011 [-0.020, -0.002], UF +0.010 [+0.002, +0.017] / -0.004 / -0.007. Levels: R1-M HF 0.525, UF 0.493 / 0.388 / 0.369 (R1 @125K 0.536, 0.483 / 0.393 / 0.377).
- **Edited-field accuracy on the Material lite cells (in-family; reference levels: "ignore the edit" 0.882 / 0.849 / 0.826; model's own accuracy on the unedited room 0.880 / 0.620 / 0.523):** R1-M 0.874 / 0.688 / 0.614, R1 @140K 0.686 / 0.509 / 0.435, R0 0.719 / 0.518 / 0.407 (25-100 / 100-200 / 200-300 Hz). R1-M's accuracy after the edit is about equal to its accuracy before it, and above R1's and R0's in every band; but "ignore the edit" still scores higher above 100 Hz, because absorption edits change the field little. r_delta 0.438 / 0.160 / 0.081 (R1 0.131 / 0.027 / -0.000), g_delta 1.41 / 1.06 / 1.01 (above 1: the edited render is closer to the truth than the unedited one; R1 0.79 / 0.78 / 0.78). Unseen: edited r 0.594 / 0.521 / 0.486, ignore the edit 0.898 / 0.854 / 0.828.
- The learning check (slope >= 0.3) is read at +10K (130K); this +5K read is descriptive.

**R1 @145K (+25K), static:** HF 0.569, UF 0.505 / 0.409 / 0.397. Against CTL @150K: HF -0.015 [-0.022, -0.007] (RC1's limit -0.02, judged at +30K), UF +0.038 / +0.034 / +0.024. Against CTL @140K (the same step count): HF -0.013 [-0.019, -0.007]. HF cost over the reads: -0.047, -0.033, -0.028, -0.022, -0.015.
**R1-S @140K (+20K), static, against R1 @140K:** HF -0.001 [-0.005, +0.002], UF +0.008 [+0.003, +0.014] / +0.009 / +0.006; levels HF 0.560, UF 0.506 / 0.417 / 0.398.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md) · [F2_trajectories.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/F2_trajectories.png) · [P4_10_R1M.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_10_R1M.yaml)

## 2026-11-23 (20:48) — I11 (R1-S +20K, all lite families, the first read matched to R1's step): stretch edits better than R1 by 0.043 / 0.013 / 0.009; the extension baseline is recorded

**Headline:** At the same training step (+20K), R1-S, which also trains on stretched rooms, predicts stretch edits better than R1 in every band (pattern-of-change r_delta +0.043 / +0.013 / +0.009 at 25-100 / 100-200 / 200-300 Hz, all three intervals above zero). The bar, read at +30K, is +0.05 in at least two bands: no band reaches it yet, and the lowest band is the closest. On everything else the two models are level, and on the materials cells R1-S is slightly below R1 at 100-200 Hz.

Off-lattice, lite rooms (8 in-family, 8 unseen), 190/190 cells, paired per room, bootstrap 10k. Stretch-U and Stretch-A pooled per room (SS1's reading, D201 Q27).

| stretch r_delta | R1-S @140K | R1 @140K | R1-S - R1 [95% CI] |
|---|---|---|---|
| in-family (SS1) | 0.814 / 0.551 / 0.353 | 0.771 / 0.538 / 0.344 | +0.043 [+0.020, +0.066] / +0.013 [+0.004, +0.022] / +0.009 [+0.004, +0.014] |
| unseen | 0.382 / 0.288 / 0.229 | 0.353 / 0.269 / 0.213 | +0.029 [+0.017, +0.043] / +0.019 [+0.016, +0.022] / +0.016 [+0.005, +0.027] |

- **Extension-rule baseline (D201 Q27), recorded now:** band-mean in-family difference at +20K is +0.0215 (by band, best first: +0.0425, +0.0130, +0.0092; at +10K it was +0.018). R1-S is extended past +30K only if, at +30K, the second-best band's difference is >= 0.00 AND the band-mean difference has risen by >= 0.02 (to >= 0.0415). Nothing is decided before that read.
- **By family (in-family, edited r, R1-S - R1):** Stretch-U +0.060 / +0.035 / +0.055, Stretch-A +0.049 / +0.014 / +0.030 (r_delta: Stretch-U +0.042 / +0.018 / +0.019, Stretch-A +0.043 / +0.008 / -0.001); Wall -0.000 / -0.003 / +0.002, Notch +0.001 / -0.002 / +0.000, Source (second and third pooled) +0.010 / +0.006 / +0.007, Material -0.004 / -0.014 [-0.028, -0.002] / -0.000.
- **Pending:** R1-S's static HF / UF at 140K and R1's at 145K. R1-S's no-harm rule (SC1) is judged at +30K against R1 at the same step.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md) · [P4_10_R1S.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_10_R1S.yaml)

## 2026-11-23 (20:34) — I10 (R1 +25K, source cells) and A8 (mode-shift slopes, R0, and lite sets of R1 and R1-S)

**Headline:** After 25K of 30K steps R1's moved-source accuracy at the clean third source in known rooms is 0.784 / 0.425 / 0.306 (25-100 / 100-200 / 200-300 Hz), against 0.771 / 0.432 / 0.308 at +20K, and the mean of its six gaps is 0.264 (0.268 at +20K, 0.288 at +15K). The worst gap is 0.377 (second source, 100-200 Hz); the bar is 0.10 in every band at +30K. Separately, R0 predicts how room edits shift resonance frequencies about right for wall moves (slope 0.71) and stretches (0.54-0.65), and not for notches (0.25): the model moves the modes, but by too little.

Off-lattice, lite rooms, 32/32 source cells. Gap = static r minus moved r; static r 0.908 / 0.665 / 0.567 in-family, 0.542 / 0.442 / 0.405 unseen.

| set, source | edited r | gap |
|---|---|---|
| in-family, second | 0.692 / 0.288 / 0.202 | 0.216 / 0.377 / 0.364 |
| in-family, third | 0.784 / 0.425 / 0.306 | 0.124 / 0.240 / 0.261 |
| unseen, second | 0.562 / 0.276 / 0.208 | -0.020 / 0.166 / 0.197 |
| unseen, third | 0.558 / 0.383 / 0.289 | -0.016 / 0.059 / 0.116 |

- Extension rule (D201): the +20K baseline is worst 0.391, mean 0.2677. At +25K, worst 0.377 and mean 0.2636. Both are read at +30K (the worst must be <= 0.15 and the mean <= 0.2477). Nothing is decided before that read.
- **Static accuracy at 145K is not in yet** (its own report is running).

**A8, mode-shift slope (descriptive; the spec's first 10 peaks in 25-150 Hz of the receiver-averaged spectrum, Hungarian matched; a second reading on 25-100 Hz beside).** Predicted on true shift, pooled per family, R0 full set, lite sets for R1 and R1-S (not comparable row by row with R0's full set):

| model | Stretch-U | Stretch-A | Wall | Notch |
|---|---|---|---|---|
| R0 in-family, 25-150 Hz: slope (Pearson; mean |error| Hz; pairs of truth peaks) | 0.65 (0.71; 1.7; 159 of 440) | 0.54 (0.56; 1.6; 160 of 440) | 0.71 (0.69; 1.0; 185 of 440) | 0.25 (0.22; 0.8; 175 of 431) |
| R0 in-family, 25-100 Hz | 0.62 (0.71; 1.4; 130 of 260) | 0.67 (0.77; 1.1; 129 of 260) | 0.77 (0.73; 0.7; 149 of 260) | 0.45 (0.29; 0.6; 145 of 255) |
| R0 unseen, 25-150 Hz | 0.36 (0.39; 2.3; 83 of 473) | 0.50 (0.46; 1.7; 93 of 473) | 0.29 (0.36; 1.2; 106 of 472) | 0.04 (0.06; 0.7; 112 of 454) |
| R1 @140K, in-family lite, 25-150 Hz | 0.59 (0.58; 2.5; 26 of 122) | 0.53 (0.59; 1.6; 36 of 122) | 1.09 (0.95; 0.6; 41 of 122) | 0.31 (0.34; 0.8; 41 of 122) |
| R1-S @135K, in-family lite, 25-150 Hz | 0.58 (0.56; 2.4; 28 of 122) | 0.49 (0.49; 1.8; 44 of 122) | -- (stretch cells only) | -- |

- **Read with care.** Only 36-42% of R0's in-family truth peaks form a matched pair (the peak's place differs by > 3 Hz or the shift is > 8 Hz and is dropped), so the slope is computed on the peaks the model places roughly right, and dropping large shifts pulls the slope down. Notch edits move the peaks little (truth shift std 1.4 Hz on 25-150 Hz), so its slope is the least reliable. SS2 (the mode-shift slope against R1's) is descriptive.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md) · [a8_R0_120K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/a8_R0_120K.json) · [a8_R1_140K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/a8_R1_140K.json) · [a8_R1S_135K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/a8_R1S_135K.json)

## 2026-11-23 (20:24) — D2 (D201), mechanism check on four models: the models' low-frequency peaks are about 1.5 times as broad as the truth's, in every room

**Headline:** In known rooms the lowest resonance peaks (25-100 Hz) of every model are about 1.5 times as wide as the true ones: truth 2.9 Hz (half-power width), model 4.8 Hz, in all 14 rooms that have usable peaks. Broader peaks mean the model blurs the mode frequencies, which matches the faster decay seen in D1, but a broad peak can also come from extra damping, so this check alone does not separate the two.

| model | truth width (median) | model width (median) | width ratio, median over rooms [95% CI over rooms] | model broader | matched peaks |
|---|---|---|---|---|---|
| R0_120K | 2.98 Hz | 4.80 Hz | **1.50** [1.39, 1.65] | 100% of rooms | 43 of 65 |
| CTL_150K | 2.94 Hz | 4.75 Hz | **1.48** [1.36, 1.64] | 100% of rooms | 42 of 65 |
| RCP_150K | 2.94 Hz | 4.77 Hz | **1.49** [1.37, 1.67] | 100% of rooms | 40 of 65 |
| S2X_240K | 2.88 Hz | 4.82 Hz | **1.50** [1.31, 1.57] | 100% of rooms | 39 of 65 |

- **How:** per room, the receiver-averaged power spectrum (off-lattice receivers) of the cached original-room render; first 10 peaks (3 dB prominence) by frequency in 25-100 Hz; each truth peak paired with the nearest model peak within 3 Hz; half-power full width by linear interpolation; a peak whose half-power points are not reached before a valley is dropped as overlapped (8-11 of 65 in these runs) and counted.
- **A third of truth peaks have no model peak within 3 Hz** (40-43 of 65 matched): the model also places some modes at the wrong frequency.
- **A caution about the link to D1:** if the extra width came only from higher damping, peak width and decay would change together; here the width is up about 50% while EDT at 63 Hz is down 12%. That suggests (it does not prove) that averaging over mode frequencies is part of it. The per-bin field error in 25-100 Hz is about -15 dB relative to the truth's energy (R0); the time-domain floor at 63 Hz is near -37 dB (D1; different normalisations, not comparable).
- R1 and R1-S join at their final checkpoints.

Links: [modes/d2.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/d2.json) · [p4_10_modes.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_modes.py)

## 2026-11-23 (20:21) — D1 (D201), first four models: every trained model's sound dies away too fast in the original, unedited rooms, by the same amount; the cause is not yet known

**Headline:** In the rooms as they are (no edit), R0, its plain continuation (CTL), the swap-only model (RCP) and the larger-corpus model (S2X) all decay faster than the truth: the early decay time (EDT, the first 10 dB) has a median error of 12-13% at 63 Hz, 22-23% at 125 Hz and 28-30% at 250 Hz in known rooms, and 21-23%, 26-27% and 29-31% in new rooms; the error is almost entirely "too fast" (bias -12% to -30%). (Listener threshold: 5% in ISO 3382-1, about 18% measured for EDT.) All four models sit on a noise floor near -27 to -38 dB, while the truth keeps falling (-71 dB at 63 Hz and -78 dB at 125 Hz); at 250 Hz the truth itself levels off at -40 dB, so T20 and T30 there are not usable for any model. Source moves, geometry edits and materials all inherit this. R1 and R1-S join at their final checkpoints.

Off-lattice receivers at least 1 m from the source, every 4th; room median per octave, mean over rooms (bootstrap CI, 10k); 15 in-family, 16 unseen rooms. EDT error = median relative error, EDT bias = signed (negative = model too fast).

| model | rooms | 63 Hz: EDT error (bias) | 125 Hz | 250 Hz | floor model, dB (63 / 125 / 250) |
|---|---|---|---|---|---|
| R0 @120K | in-family | 12.6% (-12.1%) | 23.1% (-22.0%) | 28.3% (-27.7%) | -37 / -31 / -27 |
| R0 @120K | unseen | 22.6% (-21.6%) | 26.4% (-25.2%) | 29.1% (-27.8%) | -37 / -32 / -27 |
| CTL @150K | in-family | 12.3% (-11.7%) | 21.9% (-21.0%) | 28.0% (-27.5%) | -38 / -32 / -29 |
| CTL @150K | unseen | 21.9% (-20.9%) | 25.6% (-24.3%) | 29.3% (-28.1%) | -38 / -33 / -29 |
| RCP @150K | in-family | 12.1% (-11.8%) | 22.5% (-21.5%) | 28.2% (-27.5%) | -37 / -32 / -28 |
| RCP @150K | unseen | 22.4% (-21.6%) | 26.4% (-25.0%) | 29.9% (-28.6%) | -37 / -32 / -28 |
| S2X @240K | in-family | 12.2% (-11.7%) | 22.7% (-21.3%) | 30.3% (-29.6%) | -37 / -32 / -28 |
| S2X @240K | unseen | 21.1% (-19.8%) | 27.2% (-25.5%) | 30.5% (-29.4%) | -37 / -33 / -28 |

- **T20 with the floor rule keeps few receivers** (125 Hz in-family: R0 1173 of 11421, CTL 2264; at 250 Hz 0-19 of 11157 per model, so it is not reported as a number).
- **The four models differ by about 2 points or less at every octave in-family** and the same holds unseen, so training on more rooms (S2X) or swaps (RCP) does not change the decay.
- **D2 (the mechanism, half-power linewidths of the first peaks in 25-100 Hz) is next.** The D160 standard report will carry D1 for every new arm's final checkpoint; R1, R1-S and R1-M run it at their final checkpoints.

Links: [d1/report.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/d1/report.md) · [d1/report.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/d1/report.json)

## 2026-11-23 (20:16) — I9 (R1-S +15K, stretch cells and static): stretch edits still improving; no matched R1 read at this step, so descriptive only

**Headline:** After 15K of 30K steps, the arm that also trains on stretched rooms (R1-S) scores 0.808 / 0.546 / 0.347 (25-100 / 100-200 / 200-300 Hz) on stretch r_delta in known rooms, against 0.801 / 0.539 / 0.341 at +10K. R1 has no stretch cells at this step, so there is no paired comparison at the same step until +20K. Beside R1's neighbouring reads it is ahead by 0.037 to 0.046 in the lowest band and by 0.003 to 0.014 above. **Against R0, R1-S is not ahead in known rooms above 100 Hz (-0.013 and -0.017); part of its lead over R1 is R1's own shortfall against R0.**

Off-lattice, lite rooms (8 in-family, 8 unseen), 62/62 cells, paired bootstrap 10k. Stretch-U and Stretch-A pooled per room.

| reference (in-family r_delta) | R1-S @135K | reference | difference [95% CI] |
|---|---|---|---|
| R1 @130K (a step behind; not matched) | 0.808 / 0.546 / 0.347 | 0.762 / 0.532 / 0.333 | +0.046 [+0.019, +0.076] / +0.014 [+0.003, +0.024] / +0.014 [+0.006, +0.020] |
| R1 @140K (a step ahead; not matched) | 0.808 / 0.546 / 0.347 | 0.771 / 0.538 / 0.344 | +0.037 [+0.014, +0.060] / +0.008 [-0.009, +0.023] / +0.003 [-0.004, +0.011] |
| **R0 @120K (the original model)** | 0.808 / 0.546 / 0.347 | 0.786 / 0.559 / 0.364 | +0.023 [-0.009, +0.058] / -0.013 [-0.031, +0.005] / -0.017 [-0.027, -0.007] |

- **Unseen rooms:** 0.373 / 0.288 / 0.225 (R1 @140K 0.353 / 0.269 / 0.213; R0 0.319 / 0.243 / 0.195). Against R0: +0.053 [+0.035, +0.074] / +0.045 [+0.035, +0.053] / +0.030 [+0.022, +0.038], so in unseen rooms the gain over R0 is real.
- **SS1's bar is on R1-S - R1 at +30K; nothing is decided here.**
- **Static (R1-S @135K):** HF 0.554, UF 0.504 / 0.419 / 0.405, level with R1 @135K: HF -0.001 [-0.008, +0.005], UF +0.007 [-0.001, +0.016] / +0.007 [+0.001, +0.014] / +0.006 [+0.000, +0.013] (R1 0.555 and 0.497 / 0.412 / 0.398). Against CTL @150K, HF -0.030 [-0.038, -0.021].
- **The read was late by the 1 h rule:** its render had a slow node; posted 20:16 against a 20:13 deadline (announced at 19:56).
- **R1's +25K read (145K) is rendering; R1-M's first read (125K) comes next.**

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-23 (19:56) — I8, static part (R1 @+20K): R1's cost at the original source is now 0.022 at 200-300 Hz; R1-S @+15K's read slips, new ETA about 20:25

**Headline:** R1's accuracy at the original source in known rooms is 0.022 below the plain continuation (CTL @150K) at 200-300 Hz, down from 0.047, 0.033 and 0.028 at +5K, +10K and +15K; the no-harm rule (RC1, judged at +30K) allows 0.020. Unseen rooms are still ahead in every band.

- **R1 @140K:** HF 0.561 (0.555 at +15K), UF 0.498 / 0.407 / 0.393; test r 0.905 / 0.675 / 0.561. Against CTL @150K: HF -0.022 [-0.029, -0.016]; UF +0.031 [+0.018, +0.044] / +0.032 [+0.021, +0.044] / +0.020 [+0.009, +0.032]. Against CTL @140K (matched step): HF -0.020 [-0.026, -0.015]. (CTL @150K HF 0.583, UF 0.467 / 0.376 / 0.373; R0 @120K HF 0.582, UF 0.468 / 0.380 / 0.371.)
- **Change R1 @140K - @130K:** HF +0.011 [+0.006, +0.016], UF 25-100 +0.006 [+0.001, +0.011]. Nothing is concluded for RC1 before +30K.
- **Slip, R1-S @135K (+15K):** one render task ran slowly (tron36), was requeued to another node with no file loss (its 55 finished predictions and 15 original-room renders were checked and are intact). The read is expected at about 20:25 instead of 20:13.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-23 (19:42) — R1-P's gate (0d) FAILS: R1-P will not be launched (D205)

**Headline:** We tested, without training, whether a physics-based penalty could improve the model's fields: the mean-value identity for waves says a field's average over a small circle equals a known multiple of its centre value. The model's raw field breaks this identity only 1.3 to 1.5 times as much as its smoothed (band-limited) version, not the 3 times the pre-registered gate requires, so the planned penalty would have almost nothing to remove. R1-P is not launched.

- **Truth residual (FDTD, code check):** 0.0074 / 0.0369 / 0.0301 (25-100 / 100-200 / 200-300 Hz), all at most 0.05 on the means: the check holds and the single-radius form is the one the rule selects. (Worst room at 100-200 Hz is 0.058.)
- **R0 raw vs truth:** 0.0307 / 0.0803 / 0.0632, about 2 to 4 times truth. **R0 band-limited:** 0.0206 / 0.0633 / 0.0464.
- **Gate 1, R0 band-limited at most 2 times truth:** 2.78 / 1.71 / 1.54; fails at 25-100 Hz. **Gate 2, R0 raw at least 3 times band-limited:** 1.49 / 1.27 / 1.36; fails in every band (raw minus band-limited is above zero, +0.010 [+0.007, +0.013], +0.017 [+0.011, +0.023], +0.017 [+0.014, +0.020], but far from 3x).
- **The two-radius form is reported for information only** (the gate would use it only if the truth's single-radius residual had exceeded 0.05); its truth residual is numerically zero, so ratios to truth are not meaningful there.

Links: [report.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/r1p_0d/report.md) · [report.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/r1p_0d/report.json) · [D204 / D205](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md)

## 2026-11-23 (19:36) — I8 (R1 +20K, all lite families): the moved source keeps improving in every band, but more slowly above 100 Hz; the +20K baseline for R1's extension rule is now recorded

**Headline:** After 20K of 30K steps, R1 reaches 0.771 / 0.432 / 0.308 (25-100 / 100-200 / 200-300 Hz) at the clean third source in known rooms, up from 0.748 / 0.407 / 0.283 at +15K; the second source at 0.668 / 0.272 / 0.192. The worst of its six gaps to its own static accuracy is 0.391 (second source, 100-200 Hz), and the mean of the six has fallen from 0.295 at +10K to 0.268. RS1 asks for every gap at most 0.10, at +30K.

Off-lattice; lite set (8 in-family, 8 unseen rooms); frozen scorer; 190/190 cells. Reference levels: static r 0.904 / 0.663 / 0.558 in-family, 0.537 / 0.443 / 0.402 unseen; "ignore the move" beside each row.

| set, source | edited r | gap (+10K gap) | ignore the move |
|---|---|---|---|
| in-family, second | 0.668 / 0.272 / 0.192 | 0.236 / 0.391 / 0.366 (0.312 / 0.390 / 0.347) | 0.336 / 0.197 / 0.249 |
| in-family, third | 0.771 / 0.432 / 0.308 | 0.133 / 0.231 / 0.249 (0.187 / 0.264 / 0.269) | 0.285 / 0.231 / 0.177 |
| unseen, second | 0.559 / 0.269 / 0.193 | -0.023 / 0.174 / 0.208 (0.008 / 0.202 / 0.204) | 0.381 / 0.229 / 0.229 |
| unseen, third | 0.554 / 0.376 / 0.288 | -0.017 / 0.066 / 0.113 (0.000 / 0.080 / 0.148) | 0.314 / 0.276 / 0.200 |

- **Extension-rule baseline, recorded now (D201 Q27):** at +20K the six in-family gaps are 0.236 / 0.391 / 0.366 (second) and 0.133 / 0.231 / 0.249 (third); worst 0.391, mean 0.2677. R1 is extended past +30K only if, at +30K, the worst is <= 0.15 AND the mean has fallen by >= 0.02 (to <= 0.2477). At today's worst gap the first condition is far off; nothing is decided before the +30K read.
- **R1's second source at 200-300 Hz (0.192) is still below "ignore the move" (0.249), in-family; unseen the same source is 0.193 against 0.229.**
- **Other families, in-family, edited r, R1 @140K (R0 @120K beside):** Stretch-U 0.758 / 0.524 / 0.407 (0.761 / 0.532 / 0.417), Stretch-A 0.817 / 0.584 / 0.481 (0.819 / 0.595 / 0.488), Wall 0.895 / 0.639 / 0.544 (0.909 / 0.655 / 0.555), Notch 0.899 / 0.656 / 0.543 (0.914 / 0.678 / 0.559). The gap to R0 on these shrank from +10K.
- **Pending:** static HF / UF at 140K (speckle analysis still running); posted as its own entry.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md) · [F2_trajectories.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/F2_trajectories.png)

## 2026-11-23 (19:14) — R0's Material decay re-scored with the D201 / D202 definitions (replaces the P-e line); R1's and R1-S's lite cells too

**Headline:** We re-measured how much the models change their decay time when a wall's absorption changes, now using EDT (decay over the first 10 dB), because T30 on model fields reads a noise floor. R0 does not respond at all: EDT edit slope -0.04 (a correct model reads near 1), Pearson -0.26. The edit makes R0's decay error larger, from 23% in unedited rooms to 38% in edited ones (the bar for R1-M is at most +5 points; R0 and the source-fix model are the reference).

| model | set | EDT slope (Pearson) | n cell-bands | abs. decay error, edited vs unedited (margin, points) | T20 receivers dropped by the floor rule |
|---|---|---|---|---|---|
| R0 @120K | in-family (full, 60 cells) | -0.04 (-0.26) | 180 | 37.8% vs 23.4% (+14.4) | 386162 of 543192 |
| R0 @120K | unseen (full, 64 cells) | -0.09 (-0.36) | 192 | 34.8% vs 27.0% (+7.8) | 358031 of 505819 |
| R1 @130K | in-family (lite, 16 cells) | -0.17 (-0.51) | 48 | 37.2% vs 25.0% (+12.2) | 143149 of 146689 |
| R1-S @130K | in-family (lite, 16 cells) | -0.15 (-0.41) | 48 | 35.8% vs 25.3% (+10.5) | 145724 of 146689 |

- **T20 cannot be used for R1 or R1-S:** the floor rule (a receiver-band is kept only if the model's decay clears its own floor by 10 dB at the -25 dB point) drops 98-99% of receiver-bands there, against 71% for R0. The EDT slope is the primary (MM1); the T20 slope is reported beside it and is descriptive.
- 0 cell-bands were dropped for EDT.
- R0's other metrics in the scorecard are unchanged (checked against the previous table: 0 differences).
- Reading: none of these models was trained on materials. The MM2 reference for R1-M is its own unedited level, from its first Material read.

## 2026-11-23 (19:11) — Rulings received and recorded (D201); R1-M released; the decay metric is re-registered (D202); R1's other edits are slightly below R0 at +10K

**Headline:** The manager answered Q26-Q32 (D201, recorded 19:05 before R1's +30K read). The materials arm (R1-M) was released and started training at 19:08:31. Decay is now measured from EDT, not T30, on model fields, and the rule for leaving out receivers whose own floor swamps the decay is written down (D202) before any R1-M number exists.

- **Read of the rulings:** RS1 strict (all six gaps <= 0.10), extension for R1 only if the worst gap <= 0.15 and the mean fell by >= 0.02 from +20K to +30K; SS1 pooled over the two stretch families; R1 at its final read is the base for P4-11 and the meeting, with R0 as the fixed fallback; R1-C closed (rule 10 stands); R1-P kept at lowest priority with a collapse monitor; train AND score in `aaf` (A3's text, which said scoring in `aaf_sm89`, was wrong and is corrected by D201). Confirmation set: R1 scored once at its final checkpoint. Meeting date (Q33): Amisha will answer.
- **R0's Material decay (re-scored with D202):** the job array is queued; its numbers replace the P-e decay line when they land. The old line (slope -0.11, Pearson -0.64) is withdrawn as a material response.
- **Board:** Table C (scorecard headline per model, in-family lite rooms), the next-read ETA and the F2 trajectories figure now exist (they were missing from the contract's list).
- **A fact for Q28, not a verdict:** at +10K R1's other edits are slightly below R0's on the same 8 in-family lite rooms: Wall 0.888 / 0.624 / 0.536 (R0 0.909 / 0.655 / 0.555), Notch 0.895 / 0.647 / 0.533 (0.914 / 0.678 / 0.559), Stretch-U 0.749 / 0.518 / 0.402 (0.761 / 0.532 / 0.417), Stretch-A 0.807 / 0.575 / 0.476 (0.819 / 0.595 / 0.488). D201's fallback looks at the source cells and HF, not at these; the final read shows whether the cost persists.
- **Q15 check:** no P4-10 metric calls the three old functions (grep clean).

Links: [D201 / D202 in DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md) · [P4_10_RULINGS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_RULINGS.md) · [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md) · [F2_trajectories.png](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/F2_trajectories.png)

## 2026-11-23 (18:48) — I7 (static accuracy, R1 +15K and R1-S +10K): R1's cost at the original source is shrinking but still over RC1's limit; R1-S does no harm

**Headline:** R1's accuracy at the original source in known rooms is 0.028 below the plain continuation (CTL @150K) at 200-300 Hz, down from 0.033 at +10K. The no-harm rule (RC1, judged at +30K) allows 0.02. R1 is ahead in unseen rooms in every band. R1-S, the stretched-room arm, is level with R1 at the same step at the original source and passes its +10K harm check.

- **R1 @135K (+15K):** HF 0.555 (0.550 at +10K), UF 0.497 / 0.412 / 0.398. Against CTL @150K: HF -0.028 [-0.035, -0.022]; UF +0.030 [+0.018, +0.043] / +0.036 [+0.026, +0.048] / +0.025 [+0.015, +0.035]. Reference levels: CTL @150K HF 0.583, UF 0.467 / 0.376 / 0.373. R1 @130K -> 135K: HF +0.005 [-0.001, +0.010].
- **R1-S harm kill (+10K, against R1 at the matched step): NOT triggered.** R1-S @130K - R1 @130K: HF -0.002 [-0.006, +0.002]; UF +0.006 [+0.001, +0.012] / +0.015 [+0.008, +0.022] / +0.002 [-0.005, +0.009]. Levels: R1-S HF 0.549, UF 0.499 / 0.413 / 0.395; R1 0.550, 0.492 / 0.398 / 0.393. (R1-S against CTL @150K is -0.035 at HF; R1-S is judged against R1.)
- **Reading for RC1:** on R1's +10K and +15K reads HF is still outside the limit. Whether it recovers by +30K is the pre-registered read; nothing is concluded from +15K.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-23 (18:36) — Correction: T30 on the model's fields measures a noise floor, not decay; A7's recommendation withdrawn

**Headline:** We found that the decay-time measurement we planned for the materials test is unreliable on the model's own fields. R0's sound decays fast, then sits on a flat late floor, so T30 reads 2.2 s at 125 Hz and 5.3 s at 250 Hz in unedited rooms where the truth is 0.70 s and 0.78 s. The materials test (MM2) cannot be judged with it as written. R1-M remains on hold (Q26), now with a stronger reason, and the decay numbers in I2 (P-e) should not be read as a material response. Full report for the manager: [P4_10_MANAGER_REPORT.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_MANAGER_REPORT.md).

- Measured on 8 in-family lite rooms, 1 in 40 off-lattice receivers (>= 1 m from the source), the original room's truth and R0's render of it: T20 is 35 / 59 / 43% too short (63 / 125 / 250 Hz), T30 off by 34 / 215 / 586%, EDT by 15 / 21 / 28%.
- The late floor (3 in-family lite rooms, 1 in 25 receivers), in energy relative to the first 0.1 s: R0 about -43 dB at 125 Hz, -38 to -40 dB at 250 Hz, flat from 0.5 to 1.4 s; the truth is near -86 dB at 1.4 s at 125 Hz.
- The earlier check of A7 (T30 within about 4% of physics) was on truth fields only.

## 2026-11-23 (18:30) — I6 (R1-S +10K, all lite families): training on stretched rooms helps stretch edits a little, less than the bar asks

**Headline:** After 10K of 30K steps, the arm that also trains on stretched rooms (R1-S) predicts stretch edits slightly better than R1 at the same step: in known rooms the pattern-of-change score r_delta is up by 0.039 / 0.007 / 0.007 (25-100 / 100-200 / 200-300 Hz). The bar, judged at +30K, is +0.05 in at least 2 of 3 bands; at +10K no band reaches it. The no-harm rule (HF / UF) is not judged yet: its static reports are still rendering.

Off-lattice; lite set (8 in-family, 8 unseen rooms); frozen scorer; 190/190 cells; paired with R1 @130K per room (mean over the room's cells), bootstrap 10k.

| stretch r_delta (Stretch-U + Stretch-A) | R1-S | R1 (reference level) | R1-S - R1 [95% CI] |
|---|---|---|---|
| in-family (SS1's metric) | 0.801 / 0.539 / 0.341 | 0.762 / 0.532 / 0.333 | +0.039 [+0.015, +0.066] / +0.007 [+0.001, +0.012] / +0.007 [+0.002, +0.013] |
| unseen | 0.362 / 0.290 / 0.225 | 0.338 / 0.266 / 0.211 | +0.025 [+0.014, +0.036] / +0.024 [+0.018, +0.031] / +0.014 [+0.005, +0.022] |

- **Edited-field r on stretches (descriptive):** in-family Stretch-U 0.798 / 0.549 / 0.447 against R1's 0.749 / 0.518 / 0.402 (+0.050 [+0.014, +0.086] / +0.031 [+0.013, +0.047] / +0.045 [+0.025, +0.063]); unseen Stretch-U 0.474 / 0.401 / 0.367 against 0.427 / 0.351 / 0.322 (+0.047 / +0.050 / +0.045, every CI above zero).
- **Other families (in-family, edited r, R1-S - R1):** Wall -0.004 / -0.006 / -0.003, Notch -0.003 / -0.008 / -0.003, Source (second and third pooled) +0.014 / +0.000 / -0.009, Material +0.006 / -0.002 / +0.016.
- **Harm kill: not judged yet** (HF / UF of R1-S @130K pending). R1-S has no learning check in the spec.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md) · [P4_10_R1S.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_10_R1S.yaml)

## 2026-11-23 (18:28) — I5 (R1 +15K, source cells): still improving at 25-100 Hz, but the moved-source gap has stopped closing above 100 Hz at the second source

**Headline:** After 15K of 30K steps, the source-fix model (R1) still gains on moved sources in the lowest band: at the clean third source in known rooms it reaches 0.748 / 0.407 / 0.283 (25-100 / 100-200 / 200-300 Hz), from 0.713 / 0.391 / 0.279 at +10K. Above 100 Hz progress has slowed: the gap to its own static accuracy is unchanged at the third source at 200-300 Hz (0.269) and slightly wider at the second source (0.394 and 0.372, from 0.390 and 0.347). The bar (at most 0.10 in every band, at both sources, at +30K) is not met in any in-family band, and no pre-registered rule is applied at this read.

Off-lattice; lite set (8 in-family, 8 unseen rooms); frozen scorer; 32/32 cells. Edited r / gap to the original room's static r. Reference levels (Rule 12): R1's static r 0.904 / 0.660 / 0.552 in-family, 0.537 / 0.446 / 0.407 unseen.

| set, source | edited r | gap (+10K gap) | ignore the move |
|---|---|---|---|
| in-family, second | 0.620 / 0.266 / 0.180 | 0.284 / 0.394 / 0.372 (0.312 / 0.390 / 0.347) | 0.336 / 0.197 / 0.249 |
| in-family, third | 0.748 / 0.407 / 0.283 | 0.156 / 0.253 / 0.269 (0.187 / 0.264 / 0.269) | 0.285 / 0.231 / 0.177 |
| unseen, second | 0.541 / 0.248 / 0.181 | -0.005 / 0.198 / 0.226 (0.008 / 0.202 / 0.204) | 0.381 / 0.229 / 0.229 |
| unseen, third | 0.554 / 0.374 / 0.275 | -0.018 / 0.072 / 0.132 (0.000 / 0.080 / 0.148) | 0.314 / 0.276 / 0.200 |

- **In two cells R1 is below "ignore the move":** the second source at 200-300 Hz, in-family (0.180 against 0.249) and unseen (0.181 against 0.229). The unseen gaps near zero at 25-100 Hz sit beside a low static level (0.537), so they are not evidence of a good field.
- **Static accuracy (HF / UF) at +15K is not in yet.** Its report job was queued for an A5000 with a start estimate of 23:30; it now asks for a P6000.
- R1-M stays held (Q26).

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-23 (18:10) — Correction: R1-M was launched before a required sign-off; it is now held

**Headline:** The materials arm (R1-M) started training at 18:07 without the manager's OK on amendment A7 (how decay time is measured), which our own record (D197) requires first. We put it back on hold at 18:10:23, before any checkpoint or read, so no result depends on it. It restarts from step 0 once the manager answers Q26. R1 and R1-S are unaffected.

- The commits "R1-M launched" (7fa8614) and "R1-M decay-slope table" (6742919) are superseded on R1-M's state by this entry; the table stays empty.
- R0's Material decay numbers posted in I2 (P-e) use A7's definition, so they too are provisional until Q26 is answered.
- Job 8542917: requeued, held (`scontrol release 8542917` starts it). Its watcher is idle, waiting for a checkpoint.

Link: [OPEN_QUESTIONS.md (Q26)](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md)

## 2026-11-23 (17:52) — I4 (R1 +10K, R1-S +5K): R1 passes its learning check, keeps gaining on the moved source, and is paying a little static accuracy

**Headline:** After 10K steps the source-fix model (R1) is still gaining on source moves: at the clean third source in known rooms it reaches 0.713 / 0.391 / 0.279 (25-100 / 100-200 / 200-300 Hz), up from 0.648 / 0.342 / 0.202 at +5K, and the gap to its own static accuracy has narrowed to 0.19 / 0.26 / 0.27 (bar: at most 0.10 in every band, read at +30K). Its accuracy at the original source, the price of training on several positions, is 0.024 below the plain continuation at the same step count (HF 0.550 against 0.575), while unseen rooms are up. The second arm, R1-S (stretched rooms mixed in), reached its first checkpoint; at +5K it is level with R1 on stretch edits, as expected that early.

Off-lattice; lite set (8 in-family, 8 unseen rooms) for the source cells; frozen scorer.

- **Learning check (pre-registered, +10K): PASSES.** In-family second-source r at 25-100 Hz 0.587 (bar 0.38; R0 0.227; RCP at +30K 0.503).
- **Harm kill (+10K; HF or any UF band below the reference by more than 0.05 with the whole CI below -0.02): NOT triggered.** R1 @130K - CTL @130K (matched step): HF -0.024 [-0.031, -0.018]; UF +0.028 [+0.015, +0.041] / +0.013 [+0.001, +0.025] / +0.013 [+0.002, +0.025]. Against CTL @150K (the final reference): HF -0.033 [-0.039, -0.027], UF +0.025 / +0.023 / +0.020. Static levels (Rule 12): R1 @130K HF 0.550, UF 0.492 / 0.398 / 0.393 (R1 @125K 0.536, 0.483 / 0.393 / 0.377; CTL @130K 0.575, 0.464 / 0.385 / 0.380). The RC1 rule at +30K allows R1's HF to be at most 0.02 below CTL @150K.
- **Source cells at +10K** (edited r / gap to the original room's static r): second source in-family 0.587 / 0.265 / 0.201 (gap 0.312 / 0.390 / 0.347), unseen 0.518 / 0.235 / 0.198 (gap 0.008 / 0.202 / 0.204); **third source in-family 0.713 / 0.391 / 0.279 (gap 0.187 / 0.264 / 0.269), unseen 0.525 / 0.356 / 0.255 (gap 0.000 / 0.080 / 0.148).** RS1's bar (<= 0.10 in every band, both sources, at +30K) is not met yet but every gap shrank from +5K (third source in-family 0.243 / 0.305 / 0.328 at +5K).
- **R1-S @+5K (descriptive; its reference R1 is at a different step and is labelled).** Stretch-U r_delta (8 in-family rooms) 0.809 / 0.558 / 0.370; Stretch-A 0.732 / 0.482 / 0.277; edited r Stretch-U 0.787 / 0.539 / 0.437 (R0 0.761 / 0.532 / 0.417). Against R1's +10K (a step ahead): r_delta +0.009 [-0.014, +0.035] / -0.012 [-0.022, -0.003] / -0.010 [-0.023, +0.001]. The paired read against R1 at the same step starts at +10K.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md) · [P4_10_R1.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_10_R1.yaml) · [P4_10_R1S.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_10_R1S.yaml)

## 2026-11-23 (16:27) — I2 complete: R0's full editability scorecard (675 cells, 6 families) -> it edits geometry well in rooms it knows, not in new ones; it cannot edit the source or materials

**Headline:** We scored every edit the pre-registration lists against re-simulated truth. In rooms the model knows, changing the room's geometry works well: a stretched room is reproduced with accuracy 0.83 / 0.61 / 0.49 (25-100 / 100-200 / 200-300 Hz), and the moved wall and the notch edit with 0.90 / 0.67 / 0.56 and 0.92 / 0.70 / 0.57. Its predicted changes follow the true ones closely (pattern correlation 0.82 for stretches, 0.74 for moved walls). In unseen rooms the same edits are far worse (0.40 / 0.33 / 0.30). Moving the source and changing absorption do not work: the moved source scores 0.23 / 0.08 / 0.04, and changing the wall absorption moves the model the wrong way (the new render is farther from the truth than the old one). Cheap physics, re-solved for the edit, scores 0.97-0.98 in every family, so this is a score sheet, not a claim of superiority.

In-family 15 rooms (off-lattice; mean over rooms of each room's average over magnitudes; m1 = physics at the edited geometry, 10 cm grid, a reference):

| family | R0 edited r | ignore the edit | R0 r_delta | R0 g_delta | m1 physics | unseen: R0 edited r / r_delta |
|---|---|---|---|---|---|---|
| Stretch-U | 0.833 / 0.606 / 0.493 | 0.410 / 0.171 / 0.119 | 0.818 / 0.589 / 0.486 | 2.99 / 1.65 / 1.49 | 0.982 / 0.924 / 0.762 | 0.402 / 0.325 / 0.304 ; 0.253 / 0.203 / 0.249 |
| Stretch-A | 0.835 / 0.613 / 0.495 | 0.514 / 0.250 / 0.281 | 0.766 / 0.552 / 0.327 | 2.54 / 1.55 / 1.25 | 0.984 / 0.932 / 0.769 | 0.416 / 0.347 / 0.327 ; 0.224 / 0.164 / 0.135 |
| Wall | 0.900 / 0.673 / 0.562 | 0.735 / 0.421 / 0.233 | 0.743 / 0.506 / 0.458 | 2.14 / 1.45 / 1.45 | 0.984 / 0.928 / 0.764 | 0.438 / 0.363 / 0.347 ; 0.179 / 0.144 / 0.210 |
| Notch | 0.917 / 0.697 / 0.568 | 0.885 / 0.715 / 0.534 | 0.637 / 0.328 / 0.216 | 1.53 / 1.26 / 1.19 | 0.984 / 0.928 / 0.771 | 0.464 / 0.380 / 0.372 ; 0.067 / 0.019 / 0.019 |
| Source (second + third) | 0.231 / 0.083 / 0.044 | 0.305 / 0.217 / 0.215 | 0.504 / 0.391 / 0.304 | 0.97 / 0.99 / 0.94 | 0.985 / 0.947 / 0.785 | 0.212 / 0.110 / 0.062 ; 0.283 / 0.253 / 0.226 |
| Material | 0.772 / 0.583 / 0.469 | 0.911 / 0.891 / 0.880 | 0.067 / 0.026 / 0.017 | 0.84 / 0.84 / 0.85 | omitted (no per-wall beta) | 0.428 / 0.357 / 0.347 ; 0.027 / -0.006 / -0.000 |

(g_delta > 1 means the edited render is closer to the edited truth than the original render was. r_delta = per-bin spatial correlation of the dB change over the common receivers, clipped at +/-20 dB.)

- **P-a (edited r within 0.05 of static r, in-family): FAILS for Stretch, HOLDS for Wall.** Stretch-U -0.085 [-0.129, -0.047] / -0.094 [-0.125, -0.066] / -0.089 [-0.121, -0.059]; Stretch-A -0.084 [-0.124, -0.048] / -0.088 [-0.112, -0.064] / -0.086 [-0.118, -0.057]; Wall -0.019 [-0.035, -0.005] / -0.028 [-0.043, -0.012] / -0.020 [-0.037, -0.004].
- **P-b (g_delta > 1 for Stretch-U and Wall at 100-300 Hz, in-family): HOLDS** (Stretch-U 1.65 / 1.49, Wall 1.45 / 1.45).
- **P-c (Notch r_delta below Stretch and Wall r_delta, in-family): HOLDS.** Paired: Notch - Stretch-U -0.181 [-0.217, -0.146] / -0.261 / -0.271; - Stretch-A -0.129 [-0.166, -0.092] / -0.224 / -0.111; - Wall -0.106 [-0.140, -0.074] / -0.177 / -0.242.
- **P-d (Source below "ignore the move" in every band): HOLDS** (0.231 / 0.083 / 0.044 against 0.305 / 0.217 / 0.215, and the clean third source).
- **P-e (Material: g_delta near 1, decay-rate slope near 0): [decay line replaced 2026-11-23 (19:14), D201 / D202; the T30 slope -0.11, Pearson -0.64 is withdrawn].** g_delta 0.84 in every band (the edited render is farther from the truth than the original render, not near 1). EDT-based edit slope (MM1's measure): -0.04 with Pearson -0.26 in-family (n 180 cell-bands; 63 / 125 / 250 Hz: -0.03 / -0.03 / -0.05), -0.09 with Pearson -0.36 unseen (n 192): the model does not change its decay when the absorption changes. MM2's quantity, for R0: |EDT error| edited 37.8% against 23.4% unedited, +14.4 points (bar for R1-M: at most +5); unseen 34.8% against 27.0%, +7.8 points. JNDs: 5% (ISO 3382-1), about 18% measured (EDT). T20 slope -0.12 in-family, but the floor rule (D202) drops 386162 of 543192 receiver-bands, so it is not a usable number for R0. Level-change slope 0.09 (Pearson 0.33).
- **P-f (unseen: every family at or below R0's static unseen r): HOLDS.** Edited minus static unseen: Stretch-U -0.065 [-0.099, -0.036] / -0.055 / -0.067, Stretch-A -0.052 / -0.032 / -0.044, Wall -0.030 [-0.045, -0.015] / -0.017 / -0.024, Notch -0.004 / +0.001 / +0.001 (equal within CI), Source -0.255 / -0.270 / -0.309, Material -0.039 / -0.023 / -0.024 (all CIs below or at 0).
- **Reading:** R0 is editable for geometry inside the family it was trained on, with a real pattern of change (r_delta up to 0.82), and degrades quickly outside it. The source and material edits are not there, which is what R1 and R1-M target. **Not yet scored:** mode-shift slope (A8), F1 figure, and the second-source/third-source split for R1 at +10K.

Links: [scorecard_R0_full.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R0_full.md) · [json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R0_full.json)

## 2026-11-23 (16:08) — I3 (smoke read): R1 (the source fix) at +5K steps already beats the swap-only model at the moved source

**Headline:** R1, the model trained with several source positions per room, reached its first scored checkpoint after only 5K steps. At the moved source it is already as good as the swap-only model (RCP) after six times as much training, and better above 100 Hz: at the clean third source, in-family, 0.648 / 0.342 / 0.202 against RCP's 0.604 / 0.266 / 0.147 (25-100 / 100-200 / 200-300 Hz), and against "leave the source alone" 0.285 / 0.231 / 0.177 it is ahead in every band. Unseen, 0.473 / 0.329 / 0.199 against RCP 0.439 / 0.273 / 0.154. This is a small-sample reading: the 16-room "lite" set and one early checkpoint.

Off-lattice, frozen scorer; scorecard-lite rooms (8 in-family, 8 unseen) at the second and third source; paired bootstrap over rooms (10k, seed 20261013). "gap" = the model's static r on the original room minus its r at the moved source.

| source | set | R1 @+5K (125K) | RCP @+30K | CTL @+30K | ignore the edit | R1 - CTL (25-100 / 100-200 / 200-300) | R1 - RCP |
|---|---|---|---|---|---|---|---|
| second | in-family | 0.539 / 0.241 / 0.152 | 0.503 / 0.189 / 0.102 | 0.234 / 0.067 / 0.041 | 0.336 / 0.197 / 0.249 | +0.305 / +0.175 / +0.111 | +0.036 [-0.033, +0.106] / +0.053 [+0.033, +0.076] / +0.050 [+0.033, +0.066] |
| **third** | **in-family** | **0.648 / 0.342 / 0.202** | 0.604 / 0.266 / 0.147 | 0.220 / 0.071 / 0.028 | 0.285 / 0.231 / 0.177 | +0.427 [+0.358, +0.490] / +0.271 / +0.174 | +0.044 [+0.017, +0.071] / +0.076 / +0.055 |
| second | unseen | 0.454 / 0.204 / 0.168 | 0.448 / 0.173 / 0.116 | 0.158 / 0.074 / 0.042 | 0.381 / 0.229 / 0.229 | +0.296 / +0.130 / +0.126 | +0.007 [-0.057, +0.071] / +0.032 / +0.052 |
| **third** | **unseen** | **0.473 / 0.329 / 0.199** | 0.439 / 0.273 / 0.154 | 0.241 / 0.135 / 0.041 | 0.314 / 0.276 / 0.200 | +0.232 / +0.193 / +0.157 | +0.034 [-0.001, +0.072] / +0.056 / +0.044 |

- **[Added 16:58] Static accuracy at +5K (standard report): HF 0.536, UF 0.483 / 0.393 / 0.377**, against CTL @150K 0.583 and 0.467 / 0.376 / 0.373 (R1 - CTL: HF -0.047 [-0.054, -0.041], UF 25-100 Hz +0.016 [+0.003, +0.028]). In-family static accuracy is paying for the original-source exposure (each row sees the original source about a quarter as often as before): HF -0.047 at +5K. The pre-registered harm kill at +10K is a drop of more than 0.05 (R1's HF is -0.039 against CTL @130K, the same step count from R0). RC1 (the no-harm rule at +30K) allows up to 0.02; the +10K read is running.
- **Still at 125K steps (+5K).** Moved-source gap (R1's static r on the original room minus its r at the moved source; RS1's bar is <= 0.10 in every band at BOTH sources, read at +30K): in-family third source 0.243 / 0.305 / 0.328, second 0.351 / 0.405 / 0.378. R0's was 0.68 / 0.60 / 0.53 (third). R1 is not at the bar yet; whether it reaches it by +30K is the pre-registered read.
- **No static accuracy yet:** the HF / UF read for R1 @125K (standard report) is still rendering on the cluster; it follows this entry. The +10K learning check (in-family second-source r at 25-100 Hz >= 0.38) is read at 130K.
- **RCP's lite numbers differ slightly from its full-set numbers** because the lite set has 8 rooms per group (RCP full in-family third source 0.575 / 0.266 / 0.153).

Links: [scorecard harness](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_scorecard.py) · [P4_10_R1.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_10_R1.yaml) · [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-23 (15:36) — I2 cont.: Source, three models; the swap-only model (RCP) does move the source, at low frequencies, in both sources

**Headline:** Scoring the source-swap model (RCP) beside the plain continuation (CTL) and the meeting model (R0) at the same two moved positions. RCP's gain over CTL is clear at both sources: at the clean third source, +0.34 / +0.18 / +0.11 (25-100 / 100-200 / 200-300 Hz, in-family) and +0.20 / +0.13 / +0.11 unseen. It beats "leave the source alone" at 25-100 Hz in both room sets (third source 0.575 vs 0.298 in-family, 0.443 vs 0.332 unseen), and at 100-200 Hz in-family (0.266 vs 0.223), but not at 200-300 Hz. Physics still sits at 0.99 / 0.95 / 0.79. The R1 arm adds multi-source training on top of RCP's swaps.

Off-lattice, paired over rooms, bootstrap 10k (seed 20261013); CTL's and RCP's own static accuracy is unchanged (0.92 / 0.71 / 0.58; D194, earlier reads).

| source | set | R0 edited r | CTL edited r | **RCP edited r** | ignore the edit | RCP - CTL |
|---|---|---|---|---|---|---|
| second | in-family 15 | 0.227 / 0.067 / 0.040 | 0.221 / 0.068 / 0.048 | 0.487 / 0.191 / 0.098 | 0.311 / 0.211 / 0.247 | +0.266 [+0.228, +0.306] / +0.123 / +0.051 |
| **third** | **in-family 15** | 0.235 / 0.099 / 0.048 | 0.232 / 0.090 / 0.041 | **0.575 / 0.266 / 0.153** | 0.298 / 0.223 / 0.183 | **+0.343 [+0.291, +0.397] / +0.176 / +0.112** |
| second | unseen 16 | 0.172 / 0.077 / 0.055 | 0.176 / 0.068 / 0.052 | 0.444 / 0.163 / 0.114 | 0.357 / 0.204 / 0.210 | +0.267 [+0.217, +0.312] / +0.094 / +0.062 |
| **third** | **unseen 16** | 0.252 / 0.144 / 0.069 | 0.244 / 0.135 / 0.056 | **0.443 / 0.266 / 0.164** | 0.332 / 0.287 / 0.208 | **+0.199 [+0.164, +0.236] / +0.131 / +0.107** |

- **Moved-source gap** (static r on the original room minus moved r; RS1's metric for R1, bar <= 0.10 in every band; read here as a reference for R1): R0 0.683 / 0.602 / 0.534 in-family third source (CTL 0.689 / 0.618 / 0.542); RCP 0.346 / 0.439 / 0.429. RCP is a large step towards, but not at, the bar of 0.10 in every band.
- **Reading, replacing the D194 wording for RCP:** RCP's gain at the clean third source is larger than at the (mirror) second one at every band, so it does not come from the flips. Swaps alone carry RCP's gain at 25-200 Hz; at 200-300 Hz RCP stays below ignore-the-move (0.153 vs 0.183 in-family; 0.164 vs 0.208 unseen). Whether multi-source training closes the rest is R1's question.
- **A corruption fixed:** nine score files (CTL and RCP, Source) were corrupted by two jobs writing the same cell at once. They were regenerated; the scorer now writes atomically, and 0 corrupt files remain. R0's published table was not affected.

Links: [scorecard_source_R0_CTL_RCP.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_source_R0_CTL_RCP.md) · [json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_source_R0_CTL_RCP.json)

## 2026-11-23 (15:28) — I2 (first read): R0's Source family; R0 cannot move the source to a NEW position either

**Headline:** This is the first scored sound-field edit. We moved the source to two new positions: the "second" one (the mirror position from P4-9) and a "third" one that neither the training nor the mirror flips ever used. For known room shapes R0's accuracy at the moved source is 0.23 / 0.08 / 0.04 (25-100 / 100-200 / 200-300 Hz), which is worse than leaving the source alone (0.30 / 0.22 / 0.22) and far below cheap physics re-solved for the new position (0.99 / 0.95 / 0.79). The third, cleanest position gives the same answer as the second, so the failure is not special to the mirror position.

Off-lattice, 601-bin renders; off-lattice receivers and the frozen scorer; paired bootstrap over rooms (10k, seed 20261013). "edited r" = the model at the new source against the re-simulated field; "ignore" = the ORIGINAL field scored against the moved-source truth; m1 = physics re-solved at the new source (10 cm grid, the reference). 'second' = the 4 source position of D182/D191; 'third' = the new one (D198).

| set | source | R0 edited r | ignore the edit | R0 r_delta | R0 g_delta | m1 physics |
|---|---|---|---|---|---|---|
| in-family 15 | second | 0.227 [0.183, 0.270] / 0.067 / 0.040 | 0.311 / 0.211 / 0.247 | 0.476 / 0.383 / 0.296 | 0.76 / 0.95 / 0.94 | 0.984 / 0.949 / 0.778 |
| in-family 15 | **third** | **0.235 [0.169, 0.304] / 0.099 / 0.048** | 0.298 / 0.223 / 0.183 | 0.532 / 0.400 / 0.313 | 1.18 / 1.04 / 0.93 | 0.985 / 0.945 / 0.792 |
| unseen 16 | second | 0.172 / 0.077 / 0.055 | 0.357 / 0.204 / 0.210 | 0.265 / 0.246 / 0.224 | 0.86 / 1.02 / 1.02 | 0.972 / 0.899 / 0.761 |
| unseen 16 | **third** | 0.252 [0.216, 0.291] / 0.144 / 0.069 | 0.332 / 0.287 / 0.208 | 0.300 / 0.260 / 0.228 | 1.19 / 1.06 / 0.99 | 0.972 / 0.916 / 0.770 |

- **Parity gate PASSES:** R0's scorecard render of the ORIGINAL rooms reproduces its stored static accuracy (in-family 0.9185 / 0.7009 / 0.5818; unseen 0.4676 / 0.3798 / 0.3711).
- **Manager's prediction P-d (R0 below ignore-the-move in every band): HOLDS,** both sources, both sets (the 'second' row repeats S-3).
- **r_delta** (does the predicted change follow the true change in pattern?): 0.3-0.5 in-family, 0.22-0.30 unseen, so the change map has some truth in it even though the moved field itself is wrong. **g_delta** (is the new render closer to the moved truth than the old render was): above 1 only for the third source at 25-100 Hz (1.18 in-family, 1.19 unseen).
- **Cells:** Source is complete for R0 (62 of 62). CTL @150K and RCP @150K Source are rendering and follow; R0's other four geometry families and materials follow as they finish.
- **R1** (the source fix) is at 123.2K of 150K steps at the time of writing and 0.78 s per step; its first scored checkpoint (+5K) is due around 15:50.

Links: [scorecard_R0_source.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R0_source.md) · [scorecard_R0_source.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R0_source.json) · [edit_manifest.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/p4_10/edit_manifest.json)

## 2026-11-23 (14:54) — I0 / I1: pre-registration, gates, corpora; R1 is training

**Headline:** Everything needed to score sound-field edits against re-simulated truth is in place and verified. The source fix (R1, multi-source training) started training this afternoon; its first scored checkpoint is due around 16:00, and it reads every 5K steps (~1.1 hours) after that.

- **Edit manifest frozen (D198):** 675 cells over 31 rooms (in-family 329, unseen 346), 7 logged skips, sha 7ce02a5fb5ec. The third source is feasible in all 31 rooms.
- **Truth-parity gate PASSES (no number scored before it):** the dense-receiver rule reproduces all 31 stored rooms' receivers exactly; re-simulating original rooms matches the stored field to 2e-10 (relative L2 over 25-300 Hz, normalised by the band). **All 551 geometry + source truth cells and all 124 material cells are simulated.**
- **Per-edge absorption gate PASSES (D199):** against an exact rectangle solution (complex Newton, no small-alpha expansion), FDTD decay rates match within 0.4% for every mode up to alpha 0.70, including the case where the old gate failed. The old reference was approximate, not the solver.
- **Corpora:** MS (1000 rooms x 4 sources; slot 0 = the room's own source (A1); 142 sources redrawn for the 1.0 m rule, 1 room with no 1.0 m node; every new slot re-simulates exactly, slot 0 matches the corpus to 6e-10). STR (2,000 stretched variants x 4 sources) assembled. MAT (per-edge alpha, continuous on m = -ln(1 - alpha) over [0.08, 0.60]) is simulating.
- **Trainer (default-off, Rule 8 tested):** multi-source memmap rows with one per-row resolver (slot, source, target, swap mask, jitter), per-row jitter from dedicated checkpointed generators; streams are identical across arms with and without replaced rows; slots and targets verified to pair correctly. **Amendment A3 stands: training in `aaf` (fp32), scoring in `aaf`.**
- **R1 launched** (tron, high-memory queue): R0 @120K warm start, MS corpus, swaps p 0.25, reads at +5K ... +30K every 5K. 0.76 s per step.
- **Not yet scored:** R0's scorecard Source family (GPU queue), CTL and RCP Source.

