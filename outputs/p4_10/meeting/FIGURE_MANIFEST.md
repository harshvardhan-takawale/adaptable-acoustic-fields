# P4-10 meeting figures (D195 reporting list)

Colours fixed by entity: R0 grey (reference), R1 blue, R1-S green, R1-M magenta, RCP yellow (validated for the light surface, adjacent-pair CVD separation 8.5; yellow and green are labelled where used).
Every PNG has a JSON sidecar with the plotted values; numbers come only from the posted scorecards / decay reports.

| figure | file | shows | status |
|---|---|---|---|
| F1 | F1_r0_scorecard.png / .json | R0's scorecard per family and band: edited-field r (with ignore-the-edit and m1 marks) and pattern-of-change r_delta, known vs new rooms | done |
| F2 | ../progress/F2_trajectories.png | Trajectories: HF, UF and each arm's primary metric against training step, with R0 / CTL / RCP reference lines | done (redrawn after every read) |
| F3 | F3_moved_source_maps.png / .json | Moved-source level maps in a known room (L5.36_W4.34_d1.88_w1.40): truth, R0, RCP, R1, second and third source, 50 and 200 Hz; colour = level relative to the truth's loud receivers | done |
| F4 | F4_stretch_delta.png / .json | Stretch-U s = 1.10 in the same room: change in level (dB, clipped at +/-20) for truth, R1 and R1-S at 125 and 200 Hz; right panel: the A8 mode-shift slope in known rooms (R0, R1, R1-S) | done |
| F5 | F5_material_decay.png / .json | Energy decay after a wall-absorption edit (all walls to 0.30; the in-family Material cell with the largest truth level drop at 0.25 s): truth before and after, R1 and R1-M after, 63 / 125 / 250 Hz octaves | done |
| F6 | -- | R1-P figures | void: R1-P was not launched (gate 0d failed, D205) |
| F7 | F7_scorecard_after.png / .json | The scorecard after: R0, R1, R1-S, R1-M per family and band, in F1's layout | done |
