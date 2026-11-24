# Presentation report for the manager (built 2026-11-24 11:33 ET; meeting 2026-11-24 about 16:00 ET)

For the manager building the deck. It lists a proposed slide order, the figures with raw links, slide-ready numbers, the approved wording, what is final or provisional, and what must not be said. **The result numbers are read from JSON at build time** (times, ranks and schedules are text) by `scripts/p4_10_presentation_report.py`; rebuild it after new reads land. Triples are 25-100 / 100-200 / 200-300 Hz; accuracy r is the off-lattice score from the frozen scorer (spatial correlation of the sound-level map). "Known rooms" = the 15 in-family test rooms; "new rooms" = the frozen 16 unseen rooms.

Status labels: **FINAL** (scored, verdict recorded), **PROVISIONAL** (a later part is still being computed), **PENDING** (not yet landed).

## 1. The story in six sentences

1. One network, trained on 1,000 rooms, renders a room's sound field from its floor plan, and can be **edited** in its input: stretched, reshaped, given new wall materials, or with its source moved.
2. In known rooms the geometry edits work (Stretch 0.863 / 0.629 / 0.522 with R1-S, against 0.410 / 0.171 / 0.119 for ignoring the edit); in new rooms they are much weaker (0.492 / 0.409 / 0.389).
3. The moved source, which R0 could not do, now works partly: a new interior source in known rooms reads 0.780 / 0.445 / 0.321, against 0.298 / 0.223 / 0.183 for ignoring the move.
4. R1-M is the first model that answers a change of wall absorption: right direction, nearly proportional (Pearson 0.95), about half the true size (slope 0.54; the bar was 0.8 to 1.2).
5. Every shared model decays too fast; two one-room models get the resonance widths (ratio 1.00) and early decay right, and R1 does not, even in rooms it trained on (1.73 times the true width).
6. Concentrating about 1,000 to 2,000 training rows on one room makes the shared model's widths right (two rooms), at a small cost to editing that room.

## 2. Proposed slide order

| # | slide | figure(s) | one line to say | status |
|---|---|---|---|---|
| 1 | What can the model edit? | [V8_scorecards](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V8_scorecards.png) (backup [F7_scorecard_after](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/F7_scorecard_after.png)) | Per family: R0 in grey, this round's model beside it, red tick = ignoring the edit. Largest gains: moving the source and changing materials. | FINAL |
| 2 | Stretching a room | [V1_stretch_50Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V1_stretch_50Hz.png) or [V1_stretch_125Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V1_stretch_125Hz.png); animation [V1_stretch_animation](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V1_stretch_animation.mp4) | Known room: at 50 Hz R1-S reads 0.85 / 0.80 / 0.89 / 0.96 / 0.95 for s = 0.90 to 1.10; at 125 Hz 0.50 / 0.56 / 0.38 / 0.56 / 0.84. | FINAL |
| 3 | Moving the source | [V2_source_50Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2_source_50Hz.png); animation [V2_source_animation](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2_source_animation.mp4) | R1 follows the source along the line, R0 does not: 50 Hz R1 0.96 / 0.89 / 0.87 / 0.70 / 0.81 / 0.83; R0 0.93 / 0.07 / 0.62 / 0.58 / 0.34 / -0.05. | FINAL |
| 4 | Changing wall materials | [V3d_mm1_scatter](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V3d_mm1_scatter.png); [V3ab_decay_rate_vs_absorption](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V3ab_decay_rate_vs_absorption.png); [V3c_early_decay](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V3c_early_decay.png) | Over 180 cell-bands R1-M's response is 0.54 of the true one (R1 0.01). In the showcase room at 125 Hz it tracks the truth at high absorption (0.50: 3.33 against 3.25 1/s) and decays too fast at low absorption (0.10: 1.65 against 0.98). | FINAL |
| 5 | The listener's walk | [V4_walk_original](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V4_walk_original.png), [V4_walk_stretched](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V4_walk_stretched.png) | R1's accuracy in the room: 0.87 / 0.63 / 0.56 (original) and 0.91 / 0.69 / 0.56 (stretched 1.10). | FINAL |
| 6 | Where the energy flows | [V5_energy_flow](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V5_energy_flow.png) | Direction agreement 0.78 at 50 Hz and 0.42 at 100 Hz (mean cosine; the showcase ranks 10 and 14 of 15; median room 0.83 and 0.71). | FINAL |
| 7 | Why decay is off | [V6_why_decay_is_off](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V6_why_decay_is_off.png); [V7_late_floor](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V7_late_floor.png) | One room: the one-room model matches the truth's resonances and early decay; the shared one blurs them. Removing the late floor does not fix R1's decay at 125 Hz (EDT 24% to 24%, T20 57% to 61%). | FINAL |
| 8 | How much training per room fixes it | [EC_exposure](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/EC_exposure.png) | Fine-tuning R1 on one room: width 1.40 at the start, 1.04 after 120 steps (4,320 rows in all); the shared models read 1.21 to 1.50. | FINAL (two rooms, one seed) |
| 9 | Capture, then edit | [EB_capture_then_edit](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/EB_capture_then_edit.png) | See section 4 before using it: this one needs care. | FINAL, 16 of 16 rooms |
| 10 | How physical are the fields? | [P2_how_physical](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/P2_how_physical.png) | Causality: the models' energy before the direct sound is 1.06, 1.11, 1.08, 1.04 times the truth's (R0, R1, R1-S, R1-M); the revisit condition (2 times) is not met. Swapping source and listener costs R1 0.15 to 0.35 of accuracy (R0 0.55 to 0.75). The wave-equation error is 4 to 5 times the simulation's at 25-100 Hz (raw). | FINAL |
| 11 | What is running | no figure | The combined model (R1-SM) trains to about 15:30; its verdict comes after the meeting. At +5K it reads the same as its parents: material response slope 0.37 (R1-M 0.36), third source 0.655 / 0.357 / 0.217 (R1 0.648 / 0.342 / 0.202), stretch pattern of the change 0.763 / 0.498 / 0.317 (R1-S 0.771 / 0.520 / 0.323). Not a verdict. A mean-value penalty (P-1) was an exploratory re-registration: its stop rule triggered at the first read (see section 5b). | P-1 FINAL, R1-SM running |
| 12 | Source-moved energy flow | [M5_energy_flow_moved_source](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/M5_energy_flow_moved_source.png) | At 50 Hz with the source at the third position R1's energy-flow direction agrees with the truth's 0.50 (median angle 32 degrees), R0's -0.30 (125 degrees). At this position the source radiates little at 50 Hz, so most arrows are circulating energy. | FINAL |
| 13 | Adding physics to training | [P1F_physics_penalty](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/P1F_physics_penalty.png) | The penalty lowers the wave-equation error but costs accuracy and worsens decay; stopped by its own rule (section 5b). | FINAL |
| backup | A new room type | [V1u_stretch_unseen_50Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V1u_stretch_unseen_50Hz.png), [V1u_stretch_unseen_125Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V1u_stretch_unseen_125Hz.png); [V2u_source_unseen_50Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2u_source_unseen_50Hz.png), [V2u_source_unseen_125Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2u_source_unseen_125Hz.png); median room [V2m_source_median_125Hz](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/V2m_source_median_125Hz.png) | The stretch is not predicted at 50 Hz in the unseen room (r 0.03 / 0.13 / 0.10 / 0.15 / 0.23) and weakly at 125 Hz (0.46 / 0.39 / 0.57 / 0.56 / 0.39). | FINAL |

The showcase room is L5.36_W4.34_d1.88_w1.40; on this round's metrics it ranks 11th to 13th of 15 on most, 3rd on one (moved source, 100-200 Hz; the median room is shown beside it) and 10th / 14th on energy flow. It is not a best case. Every figure is 16:9, 14 pt or more, with a PDF and a JSON beside the PNG.

## 3. Slide-ready numbers

| number | value | source |
|---|---|---|
| R1, moved source in known rooms, second position (ignoring the move) | 0.705 / 0.309 / 0.199 (0.311 / 0.211 / 0.247) | verdict_R1_150K.json |
| R1, moved source in known rooms, third position (ignoring the move) | 0.780 / 0.445 / 0.321 (0.298 / 0.223 / 0.183) | verdict_R1_150K.json |
| R1, gap to its own original-position accuracy, third source (bar 0.10) | 0.128 / 0.238 / 0.250 | verdict_R1_150K.json |
| R1, unseen rooms, third source (ignoring the move) | 0.554 / 0.369 / 0.292 (0.332 / 0.287 / 0.208) | verdict_R1_150K.json |
| R1, swapped render against normal (known rooms; 25-100 Hz) | 0.76 against 0.91 | verdict_R1_150K.json |
| R1-S minus R1, stretch pattern of the change, known rooms (bar +0.05 in 2 bands) | +0.032 [+0.018, +0.047] / +0.017 [+0.010, +0.025] / +0.012 [+0.007, +0.018] | verdict_R1S_150K.json |
| R1-M, change in decay rate (1/EDT): slope, Pearson (known / new rooms; bar 0.8 to 1.2) | 0.54, 0.95 / 0.66, 0.94 | verdict_R1M_150K.json |
| R1-M, EDT error edited against its own unedited (known / new) | 22.5% vs 24.6% / 29.0% vs 29.4% | verdict_R1M_150K.json |
| Stretch-U accuracy, known rooms: R1-S (R0) | 0.863 / 0.629 / 0.522 (0.833 / 0.606 / 0.493) | scorecard_R1S_full.json, scorecard_R0_full.json |
| Stretch-U accuracy, new rooms: R1-S (R0) | 0.492 / 0.409 / 0.389 (0.402 / 0.325 / 0.304) | the same files |
| Material accuracy, known rooms: R1-M (R1); ignore the edit | 0.894 / 0.713 / 0.618 (0.749 / 0.567 / 0.476); 0.911 / 0.891 / 0.880 | scorecard_R1M_full.json, scorecard_R1_full.json |
| Resonance width / truth: one-room models (2 rooms), R1 in 15 training rooms | 1.00; 1.73 [1.53, 1.94] | e1/report.json |
| EDT error 63 / 125 / 250 Hz: one-room models; R1 in 15 training rooms | 0.4% / 1.1% / 2.2%; 12% / 25% / 31% | e1/report.json |
| R1 decay error at 125 Hz, in-family, after removing the late floor: EDT / T20 (before) | 24% / 61% (24% / 57%) | v7/report.json |
| Confirmation set (16 fresh unseen rooms): R0 | 0.607 / 0.447 / 0.400 | p4_9/confirm/R0_120K.json |
| Confirmation set (16 fresh unseen rooms): S2X | 0.605 / 0.466 / 0.430 | p4_9/confirm/S2X_240K.json |
| Confirmation set (16 fresh unseen rooms): R1 | 0.644 / 0.469 / 0.416 | p4_9/confirm/R1_150K.json |
| R1 on the frozen unseen set (16) | 0.506 / 0.409 / 0.399 | verdict_R1M_150K.json (reference column) |
| R1 no-harm at the original source, HF (CTL @150K) | 0.571 (0.583), -0.013 [-0.019, -0.007] | verdict_R1_150K.json |
| Single-listener render time after a geometry edit (network, one A5000) -- the only real-time claim | 12 ms | outputs/p4_9/latency.json |

More for the slides on decay and training per room (E1, E-C):
- One-room models, 2 rooms: width 1.00 times the truth, 11 of 11 peaks; EDT error 0.4% / 1.1% / 2.2%.
- R1 in 15 of its training rooms: width 1.73 [1.53, 1.94], EDT error 12% / 25% / 31%.
- E-C (2 rooms): shared width R0 @120K 1.44, CTL @150K 1.42, R1 @150K 1.40, S2X @240K (2,000 rooms) 1.39; fine-tuned on one room: 1.24 / 1.14 / 1.04 at 30 / 60 / 120 steps. From scratch on one room, the first point that fits the room has width 1.02, at 16,000 rows; earlier ones are not fits.

Wording of E-C, draft for your edit: "In two rooms, about 1,000 to 2,000 extra training rows concentrated on one room bring the shared model's resonance widths to the right value; early decay takes several times longer. The same kind of rows spread over all rooms barely helps. Specialising costs a little on edits of that room."

## 4. Read this before using the capture-then-edit slide (E-B, FINAL, 16 of 16 rooms)

**Accuracy after the edit at 25-100 Hz, new rooms, same 32 microphones and receivers for every method:**

| edit | zero-shot edit | captured, then edited | captured, not edited | interpolate the 32 mics | ignore the edit |
|---|---|---|---|---|---|
| Stretch-U 1.10 | 0.42 | 0.45 | 0.28 | 0.26 | 0.26 |
| Wall +0.3 m | 0.51 | 0.73 | 0.67 | 0.74 | 0.76 |
| All walls 0.30 | 0.64 | 0.84 | 0.80 | 0.88 | 0.89 |
| Longest wall 0.50 | 0.58 | 0.81 | 0.80 | 0.89 | 0.90 |
| Source moved | 0.55 | 0.42 | 0.27 | 0.32 | 0.33 |

Pre-registered test CE1 (survival >= 0.5 at 25-100 Hz): Stretch-U 1.10: survival 0.11 [-0.05, 0.22] -> fails; Wall +0.3 m: survival 0.75 [0.68, 0.80] -> holds; All walls 0.30: survival 0.64 [0.57, 0.69] -> holds.

**Why this is not a headline yet:**
1. **The same survival bar is passed without editing.** The captured model with no edit scores 0.55 (Wall) and 0.53 (all walls 0.30) on the same scale (a reading aid added after seeing the numbers, not pre-registered). So CE1 "holds" for Wall and Material mostly because capturing helps, not because the edit does. The edit's own contribution (captured-and-edited minus captured-not-edited) is +0.06 (Wall) and +0.03 (Material) at 25-100 Hz.
2. **Interpolating the 32 microphones (KRR) scores as high as, or higher than, capture-then-edit on Wall and Material** (the true edit changes the field little there; ignoring the edit also scores 0.76 and 0.89). Capture-then-edit is clearly better only where the edit moves the field a lot, i.e. Stretch (0.45 against 0.28 unedited and 0.26 interpolated at 25-100 Hz), and for Source it loses to the zero-shot edit (0.42 against 0.55).
3. Zero-shot edit on Stretch (0.42) is almost as good as capture-then-edit (0.45), so the capture adds little there, and Stretch fails CE1.
4. Decay after capture (both Material cells): the captured model's response to a material change is 0.35 of the true one (Pearson 0.70), against 0.82 (0.85) zero-shot. The decay error falls (20.5% edited against 25.2% unedited) but the response shrinks. These cover two magnitudes only: do not compare the zero-shot slope with MM1's.
5. Capturing hurts the edit for Stretch above 100 Hz and for Source in every band (the uncaptured model is better).

A safe sentence if the slide is used: "Capturing a new room with 32 microphones raises its accuracy a lot before any edit (0.52 to 0.82 on the original room). Editing afterwards adds +0.06 (wall) and +0.03 (materials) to the captured model and +0.17 for a stretch, which the zero-shot edit nearly matches; a moved source gets worse than zero-shot." (25-100 Hz)

## 5. Physics audit (P-2, FINAL)

- **Causality (complete, 15 rooms, 4 models; FINAL):** energy before the direct sound over the direct-sound energy: truth 0.109; R0 0.115, R1 0.120, R1-S 0.117, R1-M 0.113. The revisit condition for the Kramers-Kronig term (model at least 2 times the truth, with its interval above zero) is **not met by any model**. Limit: the truth's own level (0.11) is the band limit's own pre-ringing, so the test cannot see non-causality smaller than that.
- **Reciprocity (complete, 4 models):** swapped against normal accuracy in known rooms at 25-100 Hz: R0 0.17 against 0.92; R1 0.76 against 0.91. At 200-300 Hz R1 is 0.22 against 0.57.
- **Wave equation (complete, 4 models):** the mean-value residual, times the truth's, raw: R0 4.1, R1 4.8, R1-S 4.7, R1-M 4.9 at 25-100 Hz. R0 is 4.1, 2.2 and 2.1 times the truth's (raw, three bands); band-limited 2.8, 1.7, 1.5.

## 5b. P-1, the mean-value penalty (FINAL: stopped at +2.5K, D217 / D218)

R1-P(t) (R1 plus the penalty) against its matched control R1-ctl after 2.5K of 10K steps: accuracy HF 0.534 against 0.572 (-0.038; below the control in 15 of 15 known rooms); the pre-registered stop rule triggered and the arm was cancelled. The wave-equation residual fell (raw, over the truth: 4.80 to 3.90 at 25-100 Hz), but the early-decay error rose at every octave (63 / 125 / 250 Hz: 15 / 24 / 29% to 17 / 33 / 54%) and the late floor rose. **For the collaborator who asked: the penalty was tried, it reduces the physics error, and it hurts accuracy and decay at this strength.** The expected lever for the 63 Hz floor did not show. One read, one seed.

## 6. Wording that is approved

- **R1:** "Moved sources are far better than before, not solved." (D206)
- **R1-M (D211):** "The first model that responds to a change of wall material through its inputs: right direction, nearly proportional (Pearson 0.95), about half the true size. After the edit its render is closer to the truth than before. Its remaining decay error is the same as every shared model's; the edit does not add to it (MM2)."
- **E1 (D211):** "Models trained on one room reproduce their rooms' resonance widths and early decay (EDT within 2%). The model shared across 1,000 rooms blurs resonances by 1.4-1.7 times, even in rooms it trained on. The architecture can express the right decay; sharing one model across rooms blurs it." E-C softens the last sentence: concentrated training on a room sharpens it, so it is a statement about today's shared training, not about the architecture.
- **R1-S:** "Training on stretched rooms improves stretch edits a little in every band, in known and new rooms, and costs nothing at the original rooms. It did not reach the bar we set." (proposed in report 3)

## 7. Do not claim

- that clearance explains the second source;
- that material edits beat the original room's true field;
- that geometry edits transfer to new rooms (unseen pattern of change <= 0.33; notches not predicted);
- that R1-SM's gains add, before its +30K verdict (about 15:30 ET);
- that anything renders in real time except single listeners (12 ms);
- that decay is 12-34% too fast: that is the early decay only; R1's T20 is 33-61% off after the late floor is removed;
- that R1-P was launched: the pre-registered arm is not (D205); P-1 today was an exploratory re-registration (D217) and was stopped at +2.5K by its stop rule (HF 0.534 against 0.572);
- that the physics penalty helps: it lowers the residual but costs accuracy and worsens the decay.
- that capture-then-edit works in general (section 4);
- that the physics penalty helps (it was stopped; section 5b);
- that decay is fixed, or that removing the floor fixes it (V7);
- that the showcase room is typical of new room types (the unseen backup pictures are much worse).

## 8. Status and expected times

| item | state | expected |
|---|---|---|
| V1-V8, V3(d), V5 caption redraw | FINAL (V5 re-draw queued) | now |
| E-C | FINAL | posted in the live log at 09:31 |
| E-B | FINAL, 16 of 16 rooms | posted in the live log |
| P-2 | FINAL | posted in the live log |
| R1-SM +5K | FINAL (descriptive) | posted in the live log |
| R1-SM +10K, +15K, +20K, +25K | training | +10K about 10:52; +25K about 14:15, posted as after the cut-off |
| P-1 (R1-P(t) and control) | FINAL: stopped at +2.5K by its stop rule | posted in the live log |
| E-E (source coverage) | truth done, renders only on spare GPUs | if it runs |
| R1-SM's +30K verdict | after the meeting | about 15:30 |

Cluster note: about 134 nodes were down this morning, so some reads land later than the schedule; nothing was lost. This report can be rebuilt at any time: `python scripts/p4_10_presentation_report.py`.

## 9. All links

- Board: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md) · Live log: [P4_10_STATUS_REPORT.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_STATUS_REPORT.md) · Meeting pack: [MEETING_PACK.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/meeting/MEETING_PACK.md) · Plan: [P4_10_MEETING_DAY.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_MEETING_DAY.md) · Physics add-on: [P4_10_PHYSICS_ADDON.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_PHYSICS_ADDON.md) · Decisions: [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md)
- Reports: [E1](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/e1/report.md) · [V7](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/v7/report.md) · [E-C](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/ec/report.md) · [E-B](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/eb/report.md) · [report 3](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/P4_10_MANAGER_REPORT_3.md)
