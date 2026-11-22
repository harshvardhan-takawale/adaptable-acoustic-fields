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
