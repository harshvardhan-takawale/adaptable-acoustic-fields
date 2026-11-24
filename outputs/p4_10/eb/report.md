# P4-10 E-B: capture then edit (R1-M, M2, K = 32, 3 seeds, 15 of 16 unseen rooms)

Missing rooms: ['C3L6.22_W5.40_NWd0.52w0.96_NEd0.28w1.86_SEd0.32w2.02']. Rooms with fewer than 3 seeds: {'C3L5.06_W5.40_NWd0.38w0.92_NEd0.36w1.60_SEd0.88w0.52': 1}.

Mean over rooms (each room = mean over 3 seeds). C = common receivers, D = the edited room's own dense off-lattice receivers. The 32 measured receivers are excluded from C (and from D when D = C) for every method.

Captured static r (original room, non-drawn off-lattice): 0.821 / 0.586 / 0.550; zero-shot static: 0.506 / 0.431 / 0.408 (25-100 / 100-200 / 200-300 Hz).

## StretchU|1.10 (15 rooms)

| field (C) | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| zero-shot edit (R1-M, no capture) | 0.416 [0.360, 0.475] | 0.374 [0.325, 0.421] | 0.337 [0.300, 0.374] |
| captured and edited | 0.449 [0.413, 0.486] | 0.263 [0.225, 0.297] | 0.205 [0.179, 0.231] |
| captured, not edited | 0.285 [0.262, 0.312] | 0.110 [0.095, 0.123] | 0.171 [0.151, 0.191] |
| KRR on the same 32 microphones | 0.263 [0.238, 0.292] | 0.057 [0.047, 0.065] | 0.136 [0.118, 0.152] |
| ignore the edit (original truth) | 0.267 [0.242, 0.297] | 0.068 [0.057, 0.079] | 0.157 [0.137, 0.175] |
| captured and edited, on D | 0.431 | 0.240 | 0.183 |
| zero-shot edit, on D | 0.389 | 0.346 | 0.312 |

Survival (D): 0.13 [-0.02, 0.24] / -0.68 [-0.93, -0.49] / -0.91 [-1.15, -0.68] (25-100 / 100-200 / 200-300 Hz)
Same scale on C: captured and edited 0.10 [-0.04, 0.20] / -0.72 [-0.95, -0.54] / -0.93 [-1.18, -0.70]; captured and NOT edited (post-hoc reading aid, not pre-registered) -0.41 [-0.72, -0.22] / -1.71 [-2.26, -1.28] / -1.17 [-1.54, -0.85]
Captured and edited minus captured and not edited (C): +0.163 [+0.134, +0.192] / +0.154 [+0.120, +0.184] / +0.034 [-0.001, +0.069]; minus KRR: +0.186 [+0.156, +0.214] / +0.207 [+0.173, +0.236] / +0.070 [+0.041, +0.098]
g_delta captured 1.25 / 1.13 / 1.05 ; zero-shot 1.23 / 1.25 / 1.19

## Wall|L+0.3 (15 rooms)

| field (C) | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| zero-shot edit (R1-M, no capture) | 0.490 [0.417, 0.575] | 0.442 [0.404, 0.483] | 0.420 [0.388, 0.454] |
| captured and edited | 0.726 [0.691, 0.757] | 0.501 [0.472, 0.529] | 0.388 [0.358, 0.417] |
| captured, not edited | 0.673 [0.659, 0.689] | 0.449 [0.420, 0.476] | 0.242 [0.220, 0.265] |
| KRR on the same 32 microphones | 0.741 [0.727, 0.755] | 0.445 [0.414, 0.473] | 0.204 [0.190, 0.217] |
| ignore the edit (original truth) | 0.758 [0.744, 0.772] | 0.483 [0.447, 0.514] | 0.227 [0.211, 0.242] |
| captured and edited, on D | 0.720 | 0.488 | 0.382 |
| zero-shot edit, on D | 0.481 | 0.429 | 0.414 |

Survival (D): 0.76 [0.69, 0.81] / 0.38 [0.29, 0.46] / -0.22 [-0.34, -0.10] (25-100 / 100-200 / 200-300 Hz)
Same scale on C: captured and edited 0.75 [0.68, 0.80] / 0.38 [0.28, 0.46] / -0.23 [-0.36, -0.10]; captured and NOT edited (post-hoc reading aid, not pre-registered) 0.58 [0.43, 0.69] / 0.05 [-0.15, 0.17] / -1.25 [-1.64, -0.94]
Captured and edited minus captured and not edited (C): +0.053 [+0.024, +0.074] / +0.052 [+0.036, +0.068] / +0.146 [+0.109, +0.182]; minus KRR: -0.015 [-0.055, +0.023] / +0.056 [+0.041, +0.072] / +0.184 [+0.153, +0.212]
g_delta captured 1.14 / 1.04 / 1.11 ; zero-shot 1.11 / 1.04 / 1.14

## Material|all0.30 (15 rooms)

| field (C) | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| zero-shot edit (R1-M, no capture) | 0.631 [0.564, 0.701] | 0.590 [0.551, 0.628] | 0.571 [0.543, 0.602] |
| captured and edited | 0.834 [0.815, 0.854] | 0.667 [0.644, 0.689] | 0.663 [0.643, 0.681] |
| captured, not edited | 0.805 [0.790, 0.820] | 0.654 [0.631, 0.676] | 0.634 [0.616, 0.652] |
| KRR on the same 32 microphones | 0.879 [0.871, 0.886] | 0.781 [0.770, 0.791] | 0.704 [0.691, 0.717] |
| ignore the edit (original truth) | 0.890 [0.882, 0.897] | 0.844 [0.836, 0.851] | 0.818 [0.814, 0.823] |
| captured and edited, on D | 0.834 | 0.667 | 0.663 |
| zero-shot edit, on D | 0.631 | 0.590 | 0.571 |

Survival (D): 0.65 [0.58, 0.69] / 0.50 [0.42, 0.57] / 0.64 [0.55, 0.72] (25-100 / 100-200 / 200-300 Hz)
Same scale on C: captured and edited 0.65 [0.58, 0.69] / 0.50 [0.42, 0.57] / 0.64 [0.55, 0.72]; captured and NOT edited (post-hoc reading aid, not pre-registered) 0.55 [0.46, 0.62] / 0.41 [0.32, 0.49] / 0.44 [0.34, 0.53]
Captured and edited minus captured and not edited (C): +0.029 [+0.021, +0.036] / +0.013 [+0.007, +0.017] / +0.028 [+0.025, +0.031]; minus KRR: -0.045 [-0.067, -0.022] / -0.114 [-0.137, -0.093] / -0.041 [-0.061, -0.025]
g_delta captured 1.33 / 1.07 / 1.15 ; zero-shot 1.24 / 1.08 / 1.08

## Material|long0.50 (15 rooms)

| field (C) | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| zero-shot edit (R1-M, no capture) | 0.562 [0.489, 0.639] | 0.520 [0.473, 0.568] | 0.491 [0.463, 0.521] |
| captured and edited | 0.810 [0.786, 0.835] | 0.634 [0.604, 0.664] | 0.607 [0.581, 0.630] |
| captured, not edited | 0.792 [0.770, 0.815] | 0.623 [0.594, 0.650] | 0.593 [0.569, 0.616] |
| KRR on the same 32 microphones | 0.886 [0.879, 0.893] | 0.795 [0.785, 0.804] | 0.712 [0.698, 0.726] |
| ignore the edit (original truth) | 0.903 [0.895, 0.910] | 0.868 [0.862, 0.874] | 0.841 [0.831, 0.851] |
| captured and edited, on D | 0.810 | 0.634 | 0.607 |
| zero-shot edit, on D | 0.562 | 0.520 | 0.491 |

Survival (D): 0.79 [0.74, 0.83] / 0.74 [0.69, 0.78] / 0.82 [0.76, 0.87] (25-100 / 100-200 / 200-300 Hz)
Same scale on C: captured and edited 0.79 [0.74, 0.83] / 0.74 [0.69, 0.78] / 0.82 [0.76, 0.87]; captured and NOT edited (post-hoc reading aid, not pre-registered) 0.73 [0.67, 0.78] / 0.67 [0.59, 0.72] / 0.72 [0.67, 0.76]
Captured and edited minus captured and not edited (C): +0.018 [+0.011, +0.025] / +0.011 [+0.007, +0.016] / +0.014 [+0.010, +0.017]; minus KRR: -0.076 [-0.098, -0.052] / -0.161 [-0.192, -0.131] / -0.105 [-0.131, -0.082]
g_delta captured 1.17 / 1.02 / 1.05 ; zero-shot 1.09 / 1.02 / 1.01

## Source|third (15 rooms)

| field (C) | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| zero-shot edit (R1-M, no capture) | 0.546 [0.500, 0.594] | 0.373 [0.355, 0.392] | 0.299 [0.289, 0.308] |
| captured and edited | 0.394 [0.349, 0.441] | 0.246 [0.225, 0.267] | 0.200 [0.188, 0.211] |
| captured, not edited | 0.277 [0.235, 0.318] | 0.142 [0.128, 0.155] | 0.102 [0.088, 0.117] |
| KRR on the same 32 microphones | 0.327 [0.273, 0.378] | 0.268 [0.244, 0.289] | 0.173 [0.152, 0.195] |
| ignore the edit (original truth) | 0.336 [0.284, 0.386] | 0.289 [0.264, 0.313] | 0.211 [0.187, 0.235] |
| captured and edited, on D | 0.394 | 0.246 | 0.200 |
| zero-shot edit, on D | 0.546 | 0.373 | 0.299 |

Survival (D): -0.48 [-0.69, -0.33] / -0.82 [-0.96, -0.70] / -0.69 [-0.82, -0.58] (25-100 / 100-200 / 200-300 Hz)
Same scale on C: captured and edited -0.48 [-0.69, -0.33] / -0.82 [-0.96, -0.70] / -0.69 [-0.82, -0.58]; captured and NOT edited (post-hoc reading aid, not pre-registered) -0.86 [-1.35, -0.53] / -1.50 [-1.88, -1.22] / -1.39 [-1.65, -1.17]
Captured and edited minus captured and not edited (C): +0.117 [+0.050, +0.188] / +0.105 [+0.076, +0.133] / +0.098 [+0.081, +0.116]; minus KRR: +0.067 [-0.010, +0.149] / -0.021 [-0.059, +0.017] / +0.027 [+0.007, +0.049]
g_delta captured 1.31 / 1.22 / 1.15 ; zero-shot 1.57 / 1.41 / 1.35

## CE1 (survival >= 0.5 at 25-100 Hz, each family on its own)

- StretchU 1.10: 0.13 [-0.02, 0.24] -> FAILS
- Wall L+0.3: 0.76 [0.69, 0.81] -> HOLDS
- Material all0.30: 0.65 [0.58, 0.69] -> HOLDS

## Change in 1/EDT, predicted on true (Material all0.30 and long0.50; 3 octaves x 32 cells)

- zero-shot: slope 0.84, Pearson 0.85, n 90 cell-bands; edited |EDT error| 26.7% vs unedited 29.2%
- captured: slope 0.36, Pearson 0.71, n 90 cell-bands; edited |EDT error| 20.3% vs unedited 25.1%
