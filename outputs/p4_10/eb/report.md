# P4-10 E-B: capture then edit (R1-M, M2, K = 32, 3 seeds, 16 of 16 unseen rooms)

Missing rooms: none. Rooms with fewer than 3 seeds: none.

Mean over rooms (each room = mean over 3 seeds). C = common receivers, D = the edited room's own dense off-lattice receivers. The 32 measured receivers are excluded from C (and from D when D = C) for every method.

Captured static r (original room, non-drawn off-lattice): 0.825 / 0.585 / 0.548; zero-shot static: 0.522 / 0.429 / 0.407 (25-100 / 100-200 / 200-300 Hz).

## StretchU|1.10 (16 rooms)

| field (C) | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| zero-shot edit (R1-M, no capture) | 0.422 [0.367, 0.479] | 0.365 [0.316, 0.413] | 0.330 [0.292, 0.366] |
| captured and edited | 0.446 [0.411, 0.481] | 0.258 [0.222, 0.291] | 0.201 [0.176, 0.226] |
| captured, not edited | 0.278 [0.251, 0.305] | 0.109 [0.096, 0.121] | 0.174 [0.154, 0.194] |
| KRR on the same 32 microphones | 0.256 [0.229, 0.286] | 0.057 [0.049, 0.065] | 0.135 [0.118, 0.150] |
| ignore the edit (original truth) | 0.260 [0.232, 0.290] | 0.068 [0.058, 0.078] | 0.156 [0.138, 0.173] |
| captured and edited, on D | 0.428 | 0.234 | 0.179 |
| zero-shot edit, on D | 0.395 | 0.337 | 0.305 |

Survival (D): 0.11 [-0.05, 0.22] / -0.66 [-0.88, -0.49] / -0.90 [-1.12, -0.68] (25-100 / 100-200 / 200-300 Hz)
Same scale on C: captured and edited 0.08 [-0.08, 0.19] / -0.69 [-0.91, -0.52] / -0.91 [-1.14, -0.69]; captured and NOT edited (post-hoc reading aid, not pre-registered) -0.48 [-0.81, -0.26] / -1.65 [-2.17, -1.25] / -1.10 [-1.46, -0.77]
Captured and edited minus captured and not edited (C): +0.168 [+0.140, +0.195] / +0.149 [+0.116, +0.179] / +0.027 [-0.009, +0.062]; minus KRR: +0.189 [+0.161, +0.216] / +0.201 [+0.168, +0.230] / +0.067 [+0.039, +0.094]
g_delta captured 1.25 / 1.12 / 1.04 ; zero-shot 1.23 / 1.23 / 1.17

## Wall|L+0.3 (16 rooms)

| field (C) | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| zero-shot edit (R1-M, no capture) | 0.508 [0.430, 0.590] | 0.443 [0.407, 0.482] | 0.420 [0.390, 0.451] |
| captured and edited | 0.730 [0.697, 0.761] | 0.500 [0.472, 0.527] | 0.387 [0.359, 0.415] |
| captured, not edited | 0.675 [0.662, 0.690] | 0.447 [0.420, 0.473] | 0.240 [0.218, 0.262] |
| KRR on the same 32 microphones | 0.740 [0.727, 0.753] | 0.445 [0.416, 0.472] | 0.204 [0.190, 0.217] |
| ignore the edit (original truth) | 0.757 [0.744, 0.771] | 0.483 [0.449, 0.512] | 0.227 [0.213, 0.242] |
| captured and edited, on D | 0.725 | 0.487 | 0.382 |
| zero-shot edit, on D | 0.499 | 0.430 | 0.415 |

Survival (D): 0.75 [0.68, 0.80] / 0.37 [0.28, 0.44] / -0.23 [-0.35, -0.12] (25-100 / 100-200 / 200-300 Hz)
Same scale on C: captured and edited 0.74 [0.67, 0.79] / 0.37 [0.28, 0.44] / -0.23 [-0.36, -0.12]; captured and NOT edited (post-hoc reading aid, not pre-registered) 0.55 [0.39, 0.67] / 0.03 [-0.15, 0.15] / -1.28 [-1.64, -0.97]
Captured and edited minus captured and not edited (C): +0.056 [+0.028, +0.076] / +0.053 [+0.037, +0.068] / +0.147 [+0.113, +0.180]; minus KRR: -0.010 [-0.049, +0.026] / +0.055 [+0.041, +0.069] / +0.184 [+0.155, +0.209]
g_delta captured 1.15 / 1.04 / 1.11 ; zero-shot 1.12 / 1.04 / 1.14

## Material|all0.30 (16 rooms)

| field (C) | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| zero-shot edit (R1-M, no capture) | 0.645 [0.576, 0.714] | 0.589 [0.553, 0.625] | 0.572 [0.546, 0.600] |
| captured and edited | 0.837 [0.818, 0.857] | 0.668 [0.647, 0.689] | 0.659 [0.640, 0.678] |
| captured, not edited | 0.805 [0.791, 0.818] | 0.655 [0.633, 0.676] | 0.633 [0.616, 0.650] |
| KRR on the same 32 microphones | 0.877 [0.870, 0.885] | 0.780 [0.770, 0.790] | 0.703 [0.691, 0.716] |
| ignore the edit (original truth) | 0.887 [0.879, 0.895] | 0.842 [0.835, 0.850] | 0.818 [0.814, 0.822] |
| captured and edited, on D | 0.837 | 0.668 | 0.659 |
| zero-shot edit, on D | 0.645 | 0.589 | 0.572 |

Survival (D): 0.64 [0.57, 0.69] / 0.51 [0.43, 0.58] / 0.62 [0.53, 0.70] (25-100 / 100-200 / 200-300 Hz)
Same scale on C: captured and edited 0.64 [0.57, 0.69] / 0.51 [0.43, 0.58] / 0.62 [0.53, 0.70]; captured and NOT edited (post-hoc reading aid, not pre-registered) 0.53 [0.41, 0.61] / 0.43 [0.34, 0.50] / 0.44 [0.33, 0.53]
Captured and edited minus captured and not edited (C): +0.032 [+0.023, +0.043] / +0.013 [+0.008, +0.017] / +0.026 [+0.022, +0.030]; minus KRR: -0.040 [-0.062, -0.017] / -0.113 [-0.134, -0.092] / -0.044 [-0.063, -0.027]
g_delta captured 1.36 / 1.07 / 1.14 ; zero-shot 1.23 / 1.08 / 1.08

## Material|long0.50 (16 rooms)

| field (C) | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| zero-shot edit (R1-M, no capture) | 0.575 [0.502, 0.650] | 0.515 [0.472, 0.561] | 0.487 [0.460, 0.516] |
| captured and edited | 0.814 [0.791, 0.839] | 0.631 [0.603, 0.660] | 0.601 [0.575, 0.625] |
| captured, not edited | 0.796 [0.774, 0.818] | 0.620 [0.594, 0.647] | 0.587 [0.562, 0.611] |
| KRR on the same 32 microphones | 0.889 [0.881, 0.897] | 0.796 [0.786, 0.805] | 0.710 [0.697, 0.725] |
| ignore the edit (original truth) | 0.904 [0.897, 0.912] | 0.868 [0.861, 0.874] | 0.838 [0.828, 0.849] |
| captured and edited, on D | 0.814 | 0.631 | 0.601 |
| zero-shot edit, on D | 0.575 | 0.515 | 0.487 |

Survival (D): 0.79 [0.74, 0.83] / 0.75 [0.70, 0.78] / 0.81 [0.76, 0.86] (25-100 / 100-200 / 200-300 Hz)
Same scale on C: captured and edited 0.79 [0.74, 0.83] / 0.75 [0.70, 0.78] / 0.81 [0.76, 0.86]; captured and NOT edited (post-hoc reading aid, not pre-registered) 0.73 [0.67, 0.77] / 0.67 [0.60, 0.73] / 0.71 [0.67, 0.76]
Captured and edited minus captured and not edited (C): +0.018 [+0.012, +0.025] / +0.011 [+0.007, +0.016] / +0.014 [+0.010, +0.017]; minus KRR: -0.075 [-0.095, -0.053] / -0.164 [-0.194, -0.135] / -0.109 [-0.135, -0.086]
g_delta captured 1.18 / 1.02 / 1.05 ; zero-shot 1.08 / 1.02 / 1.01

## Source|third (16 rooms)

| field (C) | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| zero-shot edit (R1-M, no capture) | 0.555 [0.508, 0.602] | 0.373 [0.357, 0.391] | 0.296 [0.286, 0.307] |
| captured and edited | 0.415 [0.366, 0.467] | 0.246 [0.227, 0.265] | 0.200 [0.189, 0.210] |
| captured, not edited | 0.274 [0.234, 0.313] | 0.143 [0.130, 0.156] | 0.099 [0.085, 0.114] |
| KRR on the same 32 microphones | 0.324 [0.274, 0.372] | 0.266 [0.244, 0.286] | 0.171 [0.150, 0.193] |
| ignore the edit (original truth) | 0.332 [0.283, 0.379] | 0.287 [0.264, 0.310] | 0.208 [0.185, 0.232] |
| captured and edited, on D | 0.415 | 0.246 | 0.200 |
| zero-shot edit, on D | 0.555 | 0.373 | 0.296 |

Survival (D): -0.46 [-0.63, -0.32] / -0.82 [-0.96, -0.71] / -0.68 [-0.81, -0.57] (25-100 / 100-200 / 200-300 Hz)
Same scale on C: captured and edited -0.46 [-0.63, -0.32] / -0.82 [-0.96, -0.71] / -0.68 [-0.81, -0.57]; captured and NOT edited (post-hoc reading aid, not pre-registered) -0.93 [-1.45, -0.58] / -1.48 [-1.84, -1.23] / -1.40 [-1.66, -1.19]
Captured and edited minus captured and not edited (C): +0.141 [+0.068, +0.217] / +0.103 [+0.077, +0.129] / +0.102 [+0.083, +0.120]; minus KRR: +0.091 [+0.009, +0.178] / -0.020 [-0.055, +0.016] / +0.029 [+0.007, +0.052]
g_delta captured 1.35 / 1.22 / 1.16 ; zero-shot 1.60 / 1.41 / 1.35

## CE1 (survival >= 0.5 at 25-100 Hz, each family on its own)

- StretchU 1.10: 0.11 [-0.05, 0.22] -> FAILS
- Wall L+0.3: 0.75 [0.68, 0.80] -> HOLDS
- Material all0.30: 0.64 [0.57, 0.69] -> HOLDS

## Change in 1/EDT, predicted on true (Material all0.30 and long0.50; 3 octaves x 32 cells)

- zero-shot: slope 0.82, Pearson 0.85, n 96 cell-bands; edited |EDT error| 26.9% vs unedited 29.3%
- captured: slope 0.35, Pearson 0.70, n 96 cell-bands; edited |EDT error| 20.5% vs unedited 25.2%
