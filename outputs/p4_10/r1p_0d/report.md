# P4-10 0d (R1-P calibration): mean-value residual on truth, R0 raw and R0 band-limited

15 in-family rooms; 8 centres per room, 8 points per circle; normalised RMS residual per bin (3 bins per band), mean over bins; form used by the gate: single

| form | band | truth mean (max room) | R0 raw | R0 band-limited | raw / bl | bl / truth | raw - 3 bl [95% CI] |
|---|---|---|---|---|---|---|---|
| single | 25-100 | 0.0074 (0.0102) | 0.0307 | 0.0206 | 1.49 | 2.78 | -0.0312 [-0.0383, -0.0255] |
| single | 100-200 | 0.0369 (0.0583) | 0.0803 | 0.0633 | 1.27 | 1.71 | -0.1095 [-0.1233, -0.0970] |
| single | 200-300 | 0.0301 (0.0432) | 0.0632 | 0.0464 | 1.36 | 1.54 | -0.0759 [-0.0892, -0.0642] |
| two | 25-100 | 0.0000 (0.0000) | 0.0009 | 0.0000 | 33.38 | 36.61 | +0.0008 [+0.0007, +0.0010] |
| two | 100-200 | 0.0000 (0.0001) | 0.0100 | 0.0008 | 11.84 | 134.08 | +0.0075 [+0.0061, +0.0089] |
| two | 200-300 | 0.0000 (0.0000) | 0.0133 | 0.0008 | 16.47 | 201.87 | +0.0109 [+0.0098, +0.0120] |

Gates (spec 0d): code check True; R0 band-limited <= 2x truth {'25-100': False, '100-200': True, '200-300': True}; R0 raw >= 3x band-limited (and CI of raw - bl above zero) {'25-100': False, '100-200': False, '200-300': False} -> R1-P launch NOT allowed.
