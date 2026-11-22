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
