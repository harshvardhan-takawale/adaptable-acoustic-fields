# Chunk P4-10 — the editability scorecard, the source fix, and four single-change arms: RESULTS

**Status (2026-11-24 03:25 ET): every planned read and diagnostic is done except the meeting figures (F1, F3-F7) and the daily digest. No run is in progress. The live log is `tasks/P4_10_STATUS_REPORT.md`; the rules are D195 (spec), D197 (amendments), D201 (rulings), D202 (decay), D206 (manager notes), D207 / D209 (diagnostic designs), D208 (R1-M extension reading).**

**Question.** Is the sound field editable in the model's latent space? Can the moved source be fixed (R1), and do stretched-room training (R1-S), per-wall materials (R1-M), a physics penalty (R1-P) or clipped rays (R1-C) help?

**Answers so far (one line each).**
* **The model is editable for geometry inside its training family, and much less outside it.** R0's stretch, wall and notch edits score 0.83 / 0.90 / 0.92 at 25-100 Hz in known rooms and 0.40 / 0.44 / 0.46 in new ones; it cannot move the source or change materials (I2).
* **The source fix (R1) works partly.** A moved source in known rooms scores 0.71 / 0.31 / 0.20 at the second position and 0.78 / 0.45 / 0.32 at the third (ignoring the move: 0.31 / 0.21 / 0.25 and 0.30 / 0.22 / 0.18), but RS1, RS2 and RS3 FAIL and RC1 holds. It is the base model (D201 Q28). Second-source lag is mostly wall clearance (0.5 m; I24).
* **R1-S (stretched rooms): SS1 FAILS, SC1 HOLDS, no extension.** Stretch edits better than R1 by +0.032 / +0.017 / +0.012 (bar 0.05 in two bands); ahead of R1 on stretch and wall, level on notch and material.
* **R1-M (materials): MM1 FAILS on the slope (0.54 of the true decay change, Pearson 0.95; bar 0.8), MM2 and MC1 hold.** The first arm that responds to a material edit; on the full set its material-edit accuracy is below "ignore the edit" in every band.
* **R1-P: its gate (0d) fails (D205), not launched. R1-C: closed (rule 10 stands, D201 Q29).**
* **Decay is a model problem in every arm (12-34% too fast), and E1 points to sharing one model, not the loss or representation.** Single-room models reproduce decay widths (ratio 1.00) and EDT to 0-2%; R1 is 1.73 in its own training rooms (D207's reading).
* **E2 (averaging normal and swapped renders): not a P4-11 candidate. E3: the late floor is common to the room only at 63 Hz.**

## R1 (the source fix), final read @150K

RS1: FAILS -- gaps second 0.203 / 0.373 / 0.372; third 0.128 / 0.238 / 0.250 (bar 0.10 each). RS2: FAILS in one of six cells. RS3: FAILS (swapped 0.759 / 0.344 / 0.222 in family against normal 0.909 / 0.683 / 0.571; the same as RCP's 0.758 / 0.343 / 0.227). RC1: HOLDS (HF 0.571 against CTL @150K 0.583, UF above). Extension (D201 Q27): not triggered. Verdict: [verdict_R1_150K.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1_150K.md); confirmation read: 0.644 / 0.469 / 0.416 against R0's 0.607 / 0.447 / 0.400 and S2X's 0.605 / 0.466 / 0.430.

## R1-S, final read @150K

SS1 (full in-family set): R1-S - R1 stretch r_delta +0.032 / +0.017 / +0.012 (all CIs above zero): FAILS, 0 of 3 bands at +0.05; SC1 holds (HF 0.572 vs R1 0.571); extension not triggered. Unseen: +0.036 / +0.026 / +0.022. Verdict: [verdict_R1S_150K.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1S_150K.md).

## R1-M, final read @150K (per-wall materials)

MM1: FAILS -- EDT edit slope 0.54 in family (Pearson 0.95, bar [0.8, 1.2]), 0.66 unseen (Pearson 0.94). MM2: HOLDS (edited |EDT error| 22.5% against own unedited 24.6%; the absolute level is high, JNDs 5% ISO / 18% measured). MC1: HOLDS (HF 0.568 vs 0.571). Learning check passed at +10K. Extension reading (D208): not met. Verdict: [verdict_R1M_150K.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/verdict_R1M_150K.md). On the full material set "ignore the edit" scores 0.911 / 0.891 / 0.880 against R1-M's 0.894 / 0.713 / 0.618.

## What changed in the measures

* Decay: T30 on model fields measures a late floor (A7); MM1 uses EDT, MM2 compares edited and unedited EDT error, T20 is reported only where the decay clears its floor (D201 Q26, D202).
* Moved-source cells use off-lattice dense receivers in the moved-source condition; the R0 scorecard (I2) was re-scored for the decay definition.
* The multi-source (MS), stretched (STR) and material (MAT) corpora are fixed draws (D210).

## Artefacts

* Scorecards: [R0](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R0_full.md), [R1](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1_full.md), [R1-S](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1S_full.md), [R1-M](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R1M_full.md).
* Decay audits: [D1](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/d1/report.md), [D2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/d2.json), [A8](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/modes/a8_R1_150K.json); diagnostics [E1](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/e1/report.md), [E2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/e2/report.md), [E3](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/e3/report.md), [source clearance](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/srcclear/report.md); gate [0d](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/r1p_0d/report.md).
* Board: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md), F2 [trajectories](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/F2_trajectories.png).

## Not done (carried to the next session, only if the manager asks)
The capture-then-edit step (2f, lowest priority). The figures F1, F2, F3, F4, F5 and F7 are done (see `outputs/p4_10/meeting/FIGURE_MANIFEST.md`); F6 is void because R1-P was not launched. The 09:00 digest for 2026-11-24 is the next update to the status report.
