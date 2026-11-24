# Motion and physics report for the manager (built 2026-11-24 11:59 ET; meeting 2026-11-24 about 16:00 ET)

For the manager finishing the deck. It gathers everything on **moving the source, moving the listener, and the physics experiments**: figures, numbers, wording, what is final, and what not to say. The general presentation report (stretch, materials, decay, slide order) is [P4_10_PRESENTATION_REPORT.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_PRESENTATION_REPORT.md). Every result number below is read from JSON by `scripts/p4_10_motion_physics_report.py`; times, ranks and status are text.

Triples are 25-100 / 100-200 / 200-300 Hz. r = off-lattice accuracy from the frozen scorer (spatial correlation of the sound-level map with the truth, dense receivers). "In-family (held-out)" = 15 test rooms of the training family; "unseen" = 16 rooms of families never trained on. Brackets are 95% intervals (paired bootstrap over rooms).

**Status:** everything in this report is FINAL except R1-SM (the combined model, still training; its verdict comes after the meeting and is not quoted here).

## 1. What to say, in order

1. **A model trained with one source per room cannot move its source.** R0 reads 0.231 / 0.083 / 0.044 on moved sources in-family, below leaving the source where it was (0.305 / 0.217 / 0.215).
2. **Source-listener swaps alone help at 25-100 Hz only** (RCP 0.531 / 0.229 / 0.126; leaving the source still 0.305 / 0.217 / 0.215: about level at 100-200 Hz, below it at 200-300 Hz).
3. **Four source positions per room plus swaps (R1) makes the moved source usable at low frequency:** 0.743 / 0.377 / 0.260 in-family, 0.546 / 0.321 / 0.246 in unseen rooms (leaving the source still: 0.344 / 0.246 / 0.209). Above 100 Hz it is far from solved.
4. **Moving the listener is cheap and works at the model's own accuracy:** 12 ms per listener after a geometry change on one A5000.
5. **The collaborator's physics idea was tried and it costs accuracy at this setting.** A mean-value (wave-equation) penalty lowered the wave-equation error but cost high-frequency accuracy and made decay worse, so its pre-registered stop rule ended the run.
6. **Physics audit of the trained models:** energy before the direct sound is 1.04 to 1.11 times the truth's (the causality condition of 2 times is not met), the swapped render is far less accurate than the normal one, and the wave-equation residual is 2.1 to 4.9 times the truth's (section 6.2).

## 1b. Proposed slide order (motion first, physics after; the general report has the full deck)

| # | slide | figure | status |
|---|---|---|---|
| 1 | Moving the source: three recipes, 50 Hz | [M1_source_three_recipes_50Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M1_source_three_recipes_50Hz.png) | FINAL |
| 2 | The same at 125 Hz, with the median room | [M1_source_three_recipes_125Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M1_source_three_recipes_125Hz.png), [M1m_source_three_recipes_125Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M1m_source_three_recipes_125Hz.png) | FINAL |
| 3 | Why: swap source and listener | [M2_reciprocity_maps](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M2_reciprocity_maps.png) | FINAL |
| 4 | Motion animation (optional) | [V2_source_animation](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2_source_animation.mp4) | FINAL |
| 5 | The listener's walk, source moved | [M3_walk_moved_source](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M3_walk_moved_source.png) (and [V4_walk_original](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V4_walk_original.png) for the still source) | FINAL |
| 6 | Where the energy flows, source moved | [M5_energy_flow_moved_source](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M5_energy_flow_moved_source.png) (and [V5_energy_flow](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V5_energy_flow.png)) | FINAL |
| 7 | Capture, then move the source | [M4_capture_then_move](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M4_capture_then_move.png) (and [EB_capture_then_edit](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/EB_capture_then_edit.png)) | FINAL, negative |
| 8 | How physical are our fields? | [P2_how_physical](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/P2_how_physical.png) | FINAL |
| 9 | Adding physics to training | [P1F_physics_penalty](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/P1F_physics_penalty.png) | FINAL, stopped |
| backup | Unseen room types | [V2u_source_unseen_50Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2u_source_unseen_50Hz.png), [V2u_source_unseen_125Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2u_source_unseen_125Hz.png) | FINAL |

## 2. Moving the source

### 2.1 Three training recipes on one line (slide 1)

Figures: [M1_source_three_recipes_50Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M1_source_three_recipes_50Hz.png), [M1_source_three_recipes_125Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M1_source_three_recipes_125Hz.png); second example (median room) [M1m_source_three_recipes_125Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M1m_source_three_recipes_125Hz.png); animation of V2 [V2_source_animation](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2_source_animation.mp4); V2 strips [V2_source_50Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2_source_50Hz.png), [V2_source_125Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2_source_125Hz.png); unseen backup [V2u_source_unseen_50Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2u_source_unseen_50Hz.png), [V2u_source_unseen_125Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2u_source_unseen_125Hz.png).

Accuracy r at each of six positions from the original source (start) to the third source, in the in-family (held-out) showcase room:
- **50 Hz** (room ranks 12 of 15 at 25-100 Hz): R0: 0.93 / 0.07 / 0.62 / 0.58 / 0.34 / -0.05; RCP: 0.92 / 0.84 / 0.79 / 0.68 / 0.48 / 0.38; R1: 0.96 / 0.89 / 0.87 / 0.70 / 0.81 / 0.83
- **125 Hz** (room ranks 3 of 15 at 100-200 Hz, so the median room is shown too): R0: 0.39 / -0.14 / 0.27 / 0.12 / -0.04 / -0.15; RCP: 0.29 / 0.27 / 0.41 / 0.40 / 0.33 / 0.46; R1: 0.40 / 0.54 / 0.55 / 0.43 / 0.47 / 0.58
- **125 Hz, median room** L6.06_W4.52_d1.08_w2.44: R0: 0.80 / 0.10 / 0.21 / 0.31 / 0.09 / 0.02; RCP: 0.78 / 0.52 / 0.20 / 0.19 / 0.18 / 0.31; R1: 0.80 / 0.61 / 0.53 / 0.53 / 0.49 / 0.43

**One line:** "At 50 Hz the one-source model loses the moved source (its accuracy falls to about zero at the far end), swaps keep part of it, and training on four source positions keeps most of it. At 125 Hz all three are much weaker."

### 2.2 The numbers behind it (all rooms, full set)

| | R0 (one source) | RCP (one source + swaps) | R1 (four sources + swaps) | leaving the source still | physics at the same edit (m1, reference) |
|---|---|---|---|---|---|
| source-family accuracy, in-family 15 | 0.231 / 0.083 / 0.044 | 0.531 / 0.229 / 0.126 | 0.743 / 0.377 / 0.260 | 0.305 / 0.217 / 0.215 | 0.985 / 0.947 / 0.785 |
| source-family accuracy, unseen 16 | 0.212 / 0.110 / 0.062 | 0.443 / 0.214 / 0.139 | 0.546 / 0.321 / 0.246 | 0.344 / 0.246 / 0.209 | 0.972 / 0.907 / 0.765 |

("Source family" pools the second and third source positions. "Leaving the source still" is the original-source truth scored against the moved truth: a reference, not a model.) **R1 by position, in-family:** second 0.705 / 0.309 / 0.199 (ignoring the move 0.311 / 0.211 / 0.247); third 0.780 / 0.445 / 0.321 (ignoring 0.298 / 0.223 / 0.183). **Unseen:** second 0.538 / 0.273 / 0.200 (ignoring 0.357 / 0.204 / 0.210); third 0.554 / 0.369 / 0.292 (ignoring 0.332 / 0.287 / 0.208).

**Pre-registered verdicts (final):** R1 fails RS1 (gap to its own original-position accuracy: second 0.203 / 0.373 / 0.372, third 0.128 / 0.238 / 0.250; bar 0.10 in every band), RS2 (one of six cells: second source, 200-300 Hz) and RS3; RC1 (no harm at the original source) holds. Approved wording: **"Moved sources are far better than before, not solved."**

### 2.3 Why, and what was tried

- **Reciprocity maps:** [M2_reciprocity_maps](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M2_reciprocity_maps.png). Swap source and listener: a physical field does not change. In the showcase room at 50 Hz: R0 normal 0.93, R0 swapped -0.04, R1 normal 0.96, R1 swapped 0.81; R0's swapped map is noise. It ranks 7 of 15 on R1's swapped-minus-normal accuracy at 25-100 Hz (-0.13; median room -0.14).
- **Reciprocity over all rooms:** table in section 6.2. RCP's swapped accuracy (in-family) is 0.758 / 0.343 / 0.227, the same as R1's: the swaps set it.
- **Averaging the normal and swapped renders is not a candidate** (E2, all 31 rooms; it fails its pre-registered reading, which needs a gain above 100 Hz at the third source and almost no loss at the original one). It helps the second source below 200 Hz and loses at the third above 100 Hz: third source, average minus normal +0.005 [-0.000, +0.011] / -0.019 [-0.024, -0.015] / -0.064 [-0.069, -0.059]; second source +0.035 [+0.024, +0.047] / +0.036 [+0.019, +0.052] / -0.023 [-0.035, -0.013].
- **Does accuracy follow where training sources were?** (E-E, 160 positions, 8 in-family rooms.) Yes: more training sources nearby goes with higher accuracy (Spearman within 0.5 m +0.35 / +0.45 / +0.35; within-room mean +0.56 / +0.45 / +0.37). **Read it as an association:** density and wall clearance go together (Spearman +0.56 to +0.66 depending on the radius). In two post-hoc checks that control for clearance (not pre-registered) the link stays positive in every band: pooled over rooms +0.17 to +0.40, within room +0.10 to +0.37. **The two checks disagree about which band carries it** (pooled: stronger above 100 Hz; within room: stronger below), so do not name a band. The p-values in the E-E report ignore that nearby positions resemble each other. An experiment that varies the training sources would be needed to call it a cause.
- **Withdrawn:** that wall clearance explains the second source (D209 corrected). The original source is as close to a wall and is the most accurate position.

## 3. Moving the listener

Figures: [V4_walk_original](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V4_walk_original.png), [V4_walk_stretched](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V4_walk_stretched.png) (source at the original place); [M3_walk_moved_source](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M3_walk_moved_source.png) (source moved to the third position).

| walk | source | R1 | R0 |
|---|---|---|---|
| original room | original place | 0.87 / 0.63 / 0.56 | not drawn |
| room stretched 1.10 | original place | 0.91 / 0.69 / 0.56 | not drawn |
| original room | third position | 0.76 / 0.50 / 0.30 | 0.11 / 0.02 / 0.03 |

(Accuracy over the room's off-lattice receivers at that source, 25-100 / 100-200 / 200-300 Hz; the walk is y = 1.0 m.) **One line:** "Moving the listener costs 10 ms per position; with the source moved, R1's field stays useful at low frequency and R0's does not." The only real-time claim is one listener: 12 ms after a geometry change, 10 ms for a listener move (one RTX A5000; a full grid takes about 32 s).

## 4. Where the energy flows

Figures: [V5_energy_flow](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V5_energy_flow.png) (original source, 50 and 100 Hz), [M5_energy_flow_moved_source](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M5_energy_flow_moved_source.png) (source at the third position, 50 Hz).

| source | frequency | model | direction agreement (mean cosine) | median angle |
|---|---|---|---|---|
| original | 50 Hz | R1 | 0.78 | 9 deg |
| original | 100 Hz | R1 | 0.42 | 34 deg |
| third | 50 Hz | R1 | 0.50 | 32 deg |
| third | 50 Hz | R0 | -0.30 | 125 deg |

At the third position the source radiates little at 50 Hz (net outward flow 0.15 m around it 0.068, against 0.990 at the original place), so those arrows show mostly circulating energy. The showcase ranks 10 of 15 on direction agreement at 50 Hz and 14 at 100 Hz (median room 0.83 and 0.71); M5 has no rank of its own.

## 5. Capture, then move the source (E-B, M4)

Figures: [EB_capture_then_edit](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/EB_capture_then_edit.png), [M4_capture_then_move](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M4_capture_then_move.png). Model R1-M, 32 microphones, 16 unseen rooms, off-lattice, the 32 microphones left out.

- **Moving the source after capture (16 rooms, source at the third position):** zero-shot move 0.55 / 0.37 / 0.30; captured then moved 0.42 / 0.25 / 0.20; captured and not moved 0.27; KRR on the same 32 microphones 0.32 (25-100 Hz); ignoring the move 0.33. **Capturing first makes the source move worse than doing it zero-shot.**
- **M4 shows one such room** (unseen, 50 Hz): original source zero-shot 0.12, captured 0.65; source moved zero-shot 0.61, captured then moved 0.12, KRR 0.11. The re-fit and E-B's stored fit give identical numbers.
- **CE1 (geometry and material edits, survival >= 0.5 at 25-100 Hz):** StretchU 1.10 0.11 [-0.05, 0.22], fails; Wall L+0.3 0.75 [0.68, 0.80], holds; Material all0.30 0.64 [0.57, 0.69], holds. The passes mostly reflect the capture itself: a captured model with no edit already scores 0.55 (Wall) and 0.53 (all walls 0.30) on the same scale, and interpolating the microphones scores as high on Wall and Material.

## 6. The physics experiments

### 6.1 P-1, adding physics to training (slide "We tried a physics penalty")

Figure: [P1F_physics_penalty](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/P1F_physics_penalty.png). R1 fine-tuned with a mean-value (wave-equation) penalty for up to 10,000 steps, against the same run without it. This is an exploratory re-registration (D217); **the pre-registered arm R1-P was never launched: its gate failed (D205).** One read (+2,500 steps), 15 in-family and 16 unseen rooms, off-lattice, paired; **the pre-registered stop rule fired (HF more than 0.02 below the control with the interval below zero) and the arm was cancelled at 11:30.** (All 15 in-family rooms were lower; recomputed from the stored renders by the session that ran P-1.) The control ran to the end of its 10,000 steps.

| measure (+2,500 steps) | penalty | control | penalty minus control [95%] |
|---|---|---|---|
| HF: in-family accuracy at 200-300 Hz | 0.534 | 0.572 | -0.038 [-0.048, -0.029] |
| UF: unseen accuracy 25-100 Hz | 0.508 | 0.507 | +0.001 [-0.008, +0.009] |
| UF: 100-200 Hz | 0.408 | 0.409 | -0.001 [-0.009, +0.007] |
| UF: 200-300 Hz | 0.391 | 0.400 | -0.008 [-0.014, -0.001] |
| wave-equation residual over the truth's, 25-100 Hz | 3.90 | 4.80 | residual -0.0067 [-0.0087, -0.0047] |
| wave-equation residual over the truth's, 100-200 Hz | 1.60 | 2.31 | residual -0.0260 [-0.0313, -0.0206] |
| wave-equation residual over the truth's, 200-300 Hz | 1.79 | 2.35 | residual -0.0169 [-0.0212, -0.0126] |
| band-limited residual over the truth's, 25-100 Hz | 2.77 | 2.89 | (0d for R0 was 2.78) |
| 63 Hz energy common to the room, over the truth's | 2.89 | 2.20 | level +0.027 [+0.013, +0.041] |
| early-decay (EDT) error 63 / 125 / 250 Hz | 17% / 33% / 54% | 15% / 24% / 29% | +2.6% / +9.1% / +24.3% |
| late floor 63 / 125 / 250 Hz | -30.4 / -24.5 / -22.9 dB | -35.0 / -29.4 / -27.1 dB | +4.6 / +4.9 / +4.2 dB |
| raw direction (finite differences) 25-100 Hz | 0.845 | 0.829 | +0.016 [+0.013, +0.019] |

**Say:** "We tried a physics penalty on the trained model. It pulls the field closer to the wave equation (residual 4.80 to 3.90 times the truth's at 25-100 Hz, 2.31 to 1.60 at 100-200 Hz, 2.35 to 1.79 at 200-300 Hz) but costs accuracy at 200-300 Hz and makes the decay worse. Our stop rule ended the run after 2,500 steps. This is one arm, one read, one setting." **The manager's expectation** (written before the numbers): the residual falls (it did); static r stays within 0.01 of the control (it did not at 200-300 Hz); direction improves at 25-100 Hz (by +0.016 [+0.013, +0.019]); EDT does not change (it got worse); whether the floor falls was open (it rose by 4.2 to 4.9 dB).

### 6.2 P-2, how physical are our fields? (zero training; final)

Figure: [P2_how_physical](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/P2_how_physical.png). 15 in-family rooms, R0 / R1 / R1-S / R1-M against the truth.

**Wave equation (mean-value residual over the truth's; 1 = as physical as the truth):**

| band | truth | raw R0 | raw R1 | raw R1-S | raw R1-M | band-limited R0 | band-limited R1 | band-limited R1-S | band-limited R1-M |
|---|---|---|---|---|---|---|---|---|---|
| 25-100 | 0.0074 | 4.1 | 4.8 | 4.7 | 4.9 | 2.8 | 3.0 | 3.1 | 3.0 |
| 100-200 | 0.0369 | 2.2 | 2.3 | 2.3 | 2.2 | 1.7 | 1.9 | 1.8 | 1.6 |
| 200-300 | 0.0301 | 2.1 | 2.4 | 2.2 | 2.3 | 1.5 | 1.7 | 1.6 | 1.4 |

**Reciprocity (accuracy of the normal and the swapped render; in-family 15; the last column is the unseen swapped accuracy):**

| model | normal | swapped | swapped minus normal | unseen, swapped |
|---|---|---|---|---|
| R0 | 0.918 / 0.701 / 0.582 | 0.167 / 0.050 / 0.027 | -0.751 [-0.776, -0.727] / -0.651 [-0.675, -0.626] / -0.554 [-0.576, -0.533] | 0.122 / 0.054 / 0.027 |
| R1 | 0.909 / 0.683 / 0.571 | 0.759 / 0.344 / 0.222 | -0.150 [-0.174, -0.126] / -0.339 [-0.354, -0.320] / -0.349 [-0.366, -0.333] | 0.465 / 0.289 / 0.238 |
| R1-S | 0.908 / 0.682 / 0.572 | 0.766 / 0.351 / 0.237 | -0.142 [-0.164, -0.120] / -0.331 [-0.346, -0.312] / -0.335 [-0.351, -0.319] | 0.468 / 0.300 / 0.253 |
| R1-M | 0.904 / 0.672 / 0.568 | 0.744 / 0.343 / 0.254 | -0.160 [-0.182, -0.138] / -0.330 [-0.344, -0.312] / -0.314 [-0.329, -0.300] | 0.468 / 0.304 / 0.262 |

**Causality (energy before the direct sound, over the direct-sound energy; condition = at least 2 times the truth's with its interval above zero):** truth 0.109; R0 0.115 (1.06 times), R1 0.120 (1.11), R1-S 0.117 (1.08), R1-M 0.113 (1.04). **Not met by any model, so the Kramers-Kronig term is not revisited.** Limit: the truth's own level is the band limit's pre-ringing, so non-causality smaller than that cannot be seen.

### 6.3 R1-P's gate and the late floor (for questions)

- **R1-P (the pre-registered penalty) was not launched** because its gate 0d failed (D205): raw residual over band-limited 1.49 / 1.27 / 1.36 (needs 3).
- **E3:** the excess room-uniform late energy is at 63 Hz only (times the truth's: R0 2.2, CTL 2.5, RCP 1.9, S2X 2.6, R1 2.4, R1-S 3.2, R1-M 3.1). P-1 raised it, not lowered it (2.89 against 2.20).

## 7. Do not claim

- That the source-motion problem is solved, or that the moved source is accurate above 100 Hz.
- That capture then moving the source works; that material or geometry edits transfer to unseen rooms.
- That wall clearance explains the second source, or that training-source density is the cause (E-E is an association).
- That the physics penalty helps the field or the decay; or that R1-P was launched. P-1 is one arm, one seed, one read.
- That our fields are physical: the wave-equation residual is 2.1 to 4.9 times the truth's raw, and reciprocity is poor above 100 Hz.
- Anything about R1-SM's gains; anything real-time except one listener (12 ms).

## 8a. Needed but not available today

- Source-motion numbers for the combined model (R1-SM) beyond its lite moved-source gap: the full scorecard is after the meeting.
- RCP and R0 moved-source maps in an unseen room: not drawn (V2u shows R1 and R0 only). RCP's unseen accuracy is in the table in 2.2.
- A walk (M3) in an unseen room, and a walk with RCP: not drawn.
- A 15-room rank for M5 (no per-room truth gradients at a moved source).
- A second penalty weight or a longer P-1 run: the stop rule ended the only run.

## 8. Still coming, and when

| item | state | expected |
|---|---|---|
| R1-SM (combined model) | training; newest read +10K, harm check not triggered; no verdict | +15K about 12:15, +20K about 13:15; +25K after the cut-off; verdict after the meeting |
| P-1 +5K read | both arms' checkpoints exist, unread; the manager decides if it is wanted | on request |
| E-E | ran | done (section 2.3) |

Nothing else on motion or physics is running. New files are announced in [the status report](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_STATUS_REPORT.md) and [the meeting pack](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/MEETING_PACK.md).

## 9. Where the numbers live

[R1 verdict](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1_150K.md) · [source scorecard R0 / CTL / RCP](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_source_R0_CTL_RCP.md) · [R1 scorecard](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1_full.md) · [P-2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/p2/report.md) · [P-1 read](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/pt/read_152K.md) · [E-B](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/eb/report.md) · [E2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/e2/report.md) · [E-E](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/ee/report.md) · [gate 0d](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/r1p_0d/report.md) · [physics add-on](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_PHYSICS_ADDON.md) · [motion add-on](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_MOTION_ADDON.md) · [E-E post-hoc](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/ee/posthoc_clearance.json) · figure sidecars are beside each PNG as `.json`.
