# P4-10 progress board

Built 2026-11-23 17:52 (from JSON on disk; every read is a scored checkpoint).

**Running jobs:** aaf_confirm x1, aaf_r1 x1, aaf_r1s x1

## R1 -- in-family moved-source gap, band mean, second and third source (bar: each band <= 0.10)

| read | HF (test 200-300) | UF 25-100 / 100-200 / 200-300 | delta HF vs ref [CI] | delta UF 25-100 [CI] |
|---|---|---|---|---|
| R1_125K | 0.536 | 0.483 / 0.393 / 0.377 | -0.047 [-0.054, -0.041] | +0.016 [+0.003, +0.028] |
| R1_130K | 0.550 | 0.492 / 0.398 / 0.393 | -0.033 [-0.039, -0.027] | +0.025 [+0.011, +0.040] |

Moved-source gap = static r on the original room - r at the moved source (in-family lite rooms; lower is better). Bar: <= 0.10 in every band.

| read | source | 25-100 | 100-200 | 200-300 |
|---|---|---|---|---|
| R1_125K | second | 0.351 [0.252, 0.458] | 0.405 [0.349, 0.461] | 0.378 [0.349, 0.417] |
| R1_125K | third | 0.243 [0.199, 0.284] | 0.305 [0.274, 0.334] | 0.328 [0.307, 0.346] |
| R1_130K | second | 0.312 [0.213, 0.418] | 0.390 [0.321, 0.461] | 0.347 [0.321, 0.383] |
| R1_130K | third | 0.187 [0.151, 0.220] | 0.264 [0.235, 0.289] | 0.269 [0.254, 0.285] |

## R1S -- r_delta on Stretch (lite) cells, in-family, paired vs R1 at the same step (bar: +0.05, CI > 0, >= 2 of 3 bands)

| read | HF (test 200-300) | UF 25-100 / 100-200 / 200-300 | delta HF vs ref [CI] | delta UF 25-100 [CI] |
|---|---|---|---|---|
| R1S_125K | 0.534 | 0.474 / 0.396 / 0.391 | -0.002 [-0.009, +0.005] | -0.009 [-0.016, -0.002] |

