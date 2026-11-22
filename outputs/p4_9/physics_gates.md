# P4-9 0b: physics residual gates (D180; D181 A9)

## (i) Interior Helmholtz residual (D180 0b(i))

Relative residual ||lap F + k^2 (F - <F>)|| / ||k^2 (F - <F>)||, the S1 definition (`scripts/p4_8_s1_gate.py`), band means 25-100 / 100-200 / 200-300 Hz, mean over rooms. Projected = the render-time projection (|k| <= 1.5 k0). FDTD truth is the reference level (Rule 12).

| field | all 31 | in-family 15 | unseen 16 | U 8 | C3 8 | x FDTD's (all 31) |
|---|---|---|---|---|---|---|
| FDTD raw | 0.114 / 0.044 / 0.029 | 0.107 / 0.044 / 0.029 | 0.120 / 0.043 / 0.029 | 0.117 / 0.046 / 0.028 | 0.123 / 0.040 / 0.030 | -- |
| FDTD proj | 2.427 / 1.005 / 0.535 | 2.322 / 1.056 / 0.545 | 2.526 / 0.958 / 0.525 | 2.238 / 0.782 / 0.440 | 2.815 / 1.134 / 0.610 | 21.3x / 23.0x / 18.6x |
| R0 120K raw | 10.138 / 3.902 / 1.675 | 9.762 / 3.942 / 1.660 | 10.491 / 3.863 / 1.689 | 10.225 / 3.765 / 1.666 | 10.758 / 3.961 / 1.711 | 88.9x / 89.4x / 58.3x |
| R0 120K proj | 2.618 / 1.135 / 0.593 | 2.431 / 1.184 / 0.600 | 2.794 / 1.089 / 0.586 | 2.550 / 0.945 / 0.521 | 3.037 / 1.232 / 0.650 | 23.0x / 26.0x / 20.6x |
| S2 120K raw | 17.091 / 6.562 / 2.872 | 16.230 / 6.549 / 2.833 | 17.898 / 6.574 / 2.908 | 16.930 / 6.379 / 2.748 | 18.866 / 6.769 / 3.067 | 149.8x / 150.3x / 100.0x |
| S2 120K proj | 2.655 / 1.192 / 0.627 | 2.469 / 1.235 / 0.629 | 2.830 / 1.152 / 0.625 | 2.613 / 1.047 / 0.574 | 3.047 / 1.256 / 0.676 | 23.3x / 27.3x / 21.8x |

**Pre-registered reading (D180):** projected R0 <= 2x FDTD's residual in every band -> **DOES NOT HOLD**. Projection does NOT remove the interior term's target in every band: the interior Helmholtz loss is not closed by this reading.

Per-room values: `outputs/p4_9/gates/interior.json`.

**Confound (D182(f), labelled):** the same projection applied to FDTD truth raises its residual 21x; against projected truth, projected R0 is 1.08x / 1.13x / 1.11x and S2 1.09x / 1.19x / 1.17x. The pre-registered reference (raw truth) cannot separate the projection's own residual from the network's.

## (ii) Robin boundary residual and gate G-BC (D180 0b(ii); D181 A9)

rel = ||d_n,out p + j k beta p|| / ||k p|| at wall stencils every 5 cm (3-point one-sided normal derivative), beta = 0.0406; band means 25-100 / 100-200 / 200-300 Hz. At this beta the Robin term is <= 4% of k|p|: the residual effectively tests near-Neumann behaviour (A9).

| field | all 31 | in-family 15 | unseen 16 | U 8 | C3 8 |
|---|---|---|---|---|---|
| FDTD raw | 0.004 / 0.003 / 0.002 | 0.004 / 0.003 / 0.002 | 0.004 / 0.003 / 0.002 | 0.004 / 0.003 / 0.002 | 0.004 / 0.003 / 0.002 |
| FDTD proj | 0.609 / 0.563 / 0.687 | 0.587 / 0.548 / 0.675 | 0.629 / 0.578 / 0.698 | 0.625 / 0.573 / 0.713 | 0.634 / 0.583 / 0.683 |
| R0_120K raw | 6.951 / 7.492 / 4.486 | 7.234 / 7.791 / 4.670 | 6.686 / 7.212 / 4.314 | 6.506 / 7.135 / 4.121 | 6.867 / 7.289 / 4.507 |
| R0_120K proj | 0.885 / 0.963 / 1.022 | 0.782 / 0.916 / 0.987 | 0.981 / 1.008 / 1.054 | 0.990 / 1.059 / 1.120 | 0.973 / 0.957 / 0.988 |
| S2_120K raw | 8.996 / 9.001 / 6.704 | 9.029 / 9.156 / 6.573 | 8.965 / 8.856 / 6.827 | 8.640 / 8.650 / 6.521 | 9.290 / 9.062 / 7.133 |
| S2_120K proj | 0.920 / 1.064 / 1.214 | 0.821 / 1.013 / 1.164 | 1.013 / 1.113 / 1.261 | 1.028 / 1.231 / 1.373 | 0.998 / 0.995 / 1.150 |

**Validity of the projection's extension to the wall (A9):** FDTD truth, projected and extended the same way: 0.609 at 25-100 Hz (must be <= 0.15) -> NOT valid.

**Gate G-BC (R0 @120K; S2 in the JSON):** (1) FDTD truth 0.004 (max room 0.005) <= 0.15: holds; (2) projected residual unseen / in-family = 1.25 >= 1.5: FAILS; (3) Spearman(residual, 25-100 Hz r) = -0.81 <= -0.4: holds. **PL1 is NOT launched.**

Per-room values: `outputs/p4_9/gates/robin.json`.

**Reading (labelled):** raw R0's wall residual is ~1,700x FDTD's (6.95 vs 0.004 at 25-100 Hz). The gate fails on raw fields too (unseen / in-family 0.92), so condition (2) does not depend on the projection's extension, which FAILS its own validity check (projected truth 0.61 > 0.15). Neither physics residual targets the unseen-family failure (interior ~1.1x, Robin 0.9-1.25x unseen / in-family): Q-D's free gates are closed, PL1 is not launched. Across rooms the Robin residual does track accuracy (Spearman -0.81).
