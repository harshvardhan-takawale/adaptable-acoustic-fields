# P4-10 P-2 (D217, D219): zero-training physics audit

15 in-family rooms, R0 @120K, R1 / R1-S / R1-M @150K against the simulated truth. CIs: paired bootstrap over rooms (10k).

## Wave equation: single-radius mean-value residual at held-out centres (RMS residual / RMS |p|; lower is more physical)

| band | truth (max room) | R0 raw | R1 raw | R1-S raw | R1-M raw | R0 band-limited | R1 band-limited | R1-S band-limited | R1-M band-limited |
|---|---|---|---|---|---|---|---|---|---|
| 25-100 | 0.0074 (0.0102) | 0.0307 (x4.1) | 0.0354 (x4.8) | 0.0349 (x4.7) | 0.0365 (x4.9) | 0.0206 (x2.8) | 0.0221 (x3.0) | 0.0227 (x3.1) | 0.0220 (x3.0) |
| 100-200 | 0.0369 (0.0583) | 0.0803 (x2.2) | 0.0853 (x2.3) | 0.0832 (x2.3) | 0.0812 (x2.2) | 0.0633 (x1.7) | 0.0692 (x1.9) | 0.0653 (x1.8) | 0.0593 (x1.6) |
| 200-300 | 0.0301 (0.0432) | 0.0632 (x2.1) | 0.0707 (x2.4) | 0.0659 (x2.2) | 0.0680 (x2.3) | 0.0464 (x1.5) | 0.0520 (x1.7) | 0.0495 (x1.6) | 0.0428 (x1.4) |

On-lattice centres among the 15 x 8: 0 (must be 0).

## Reciprocity: normal and swapped renders (r against the same truth)

| model | set | normal 25-100 / 100-200 / 200-300 | swapped | swapped - normal |
|---|---|---|---|---|
| R0 | in_family15 (15 rooms) | 0.918 / 0.701 / 0.582 | 0.167 / 0.050 / 0.027 | -0.751 [-0.776, -0.727] / -0.651 [-0.675, -0.626] / -0.554 [-0.576, -0.533] |
| R0 | unseen16 (16 rooms) | 0.468 / 0.380 / 0.371 | 0.122 / 0.054 / 0.027 | -0.346 [-0.434, -0.267] / -0.326 [-0.362, -0.293] / -0.344 [-0.379, -0.311] |
| R1 | in_family15 (15 rooms) | 0.909 / 0.683 / 0.571 | 0.759 / 0.344 / 0.222 | -0.150 [-0.174, -0.126] / -0.339 [-0.354, -0.320] / -0.349 [-0.366, -0.333] |
| R1 | unseen16 (16 rooms) | 0.506 / 0.409 / 0.399 | 0.465 / 0.289 / 0.238 | -0.041 [-0.076, -0.007] / -0.119 [-0.150, -0.090] / -0.160 [-0.189, -0.131] |
| R1-S | in_family15 (15 rooms) | 0.908 / 0.682 / 0.572 | 0.766 / 0.351 / 0.237 | -0.142 [-0.164, -0.120] / -0.331 [-0.346, -0.312] / -0.335 [-0.351, -0.319] |
| R1-S | unseen16 (16 rooms) | 0.515 / 0.419 / 0.411 | 0.468 / 0.300 / 0.253 | -0.046 [-0.078, -0.019] / -0.119 [-0.150, -0.089] / -0.159 [-0.188, -0.130] |
| R1-M | in_family15 (15 rooms) | 0.904 / 0.672 / 0.568 | 0.744 / 0.343 / 0.254 | -0.160 [-0.182, -0.138] / -0.330 [-0.344, -0.312] / -0.314 [-0.329, -0.300] |
| R1-M | unseen16 (16 rooms) | 0.522 / 0.429 / 0.407 | 0.468 / 0.304 / 0.262 | -0.054 [-0.089, -0.020] / -0.126 [-0.161, -0.091] / -0.145 [-0.173, -0.117] |

## Causality: early excess before the direct sound / direct-sound energy (25-300 Hz, receivers >= 3 m)

Rooms used 15 (excluded 0), receivers per room 1473 to 2186; onset check (median offset of the truth's arrival from t_direct) +0.84 ms, OK.

| reading | truth | R0 | R1 | R1-S | R1-M |
|---|---|---|---|---|---|
| pooled over receivers (primary) | +0.1087 [+0.0992, +0.1176] | +0.1150 [+0.1048, +0.1239]; 1.06x truth; minus truth +0.0063 [+0.0043, +0.0085]; minus 2x truth -0.1025 [-0.1116, -0.0930]; condition not met | +0.1203 [+0.1089, +0.1305]; 1.11x truth; minus truth +0.0116 [+0.0082, +0.0148]; minus 2x truth -0.0972 [-0.1055, -0.0885]; condition not met | +0.1170 [+0.1060, +0.1269]; 1.08x truth; minus truth +0.0083 [+0.0052, +0.0110]; minus 2x truth -0.1005 [-0.1091, -0.0917]; condition not met | +0.1128 [+0.1031, +0.1213]; 1.04x truth; minus truth +0.0041 [+0.0008, +0.0073]; minus 2x truth -0.1047 [-0.1149, -0.0944]; condition not met |
| median over receivers | +0.1007 [+0.0890, +0.1123] | +0.1072 [+0.0953, +0.1187]; 1.06x truth; minus truth +0.0065 [+0.0037, +0.0095]; minus 2x truth -0.0942 [-0.1067, -0.0819]; condition not met | +0.1140 [+0.0998, +0.1280]; 1.13x truth; minus truth +0.0133 [+0.0093, +0.0173]; minus 2x truth -0.0874 [-0.0979, -0.0770]; condition not met | +0.1095 [+0.0963, +0.1225]; 1.09x truth; minus truth +0.0088 [+0.0057, +0.0120]; minus 2x truth -0.0919 [-0.1029, -0.0808]; condition not met | +0.1047 [+0.0927, +0.1165]; 1.04x truth; minus truth +0.0040 [-0.0005, +0.0084]; minus 2x truth -0.0967 [-0.1100, -0.0836]; condition not met |
| pooled, late window [1.6, 1.9) s | +0.1140 [+0.1044, +0.1230] | +0.1203 [+0.1101, +0.1293]; 1.06x truth; minus truth +0.0063 [+0.0043, +0.0085]; minus 2x truth -0.1077 [-0.1171, -0.0982]; condition not met | +0.1259 [+0.1144, +0.1362]; 1.10x truth; minus truth +0.0119 [+0.0085, +0.0151]; minus 2x truth -0.1022 [-0.1107, -0.0934]; condition not met | +0.1225 [+0.1115, +0.1325]; 1.07x truth; minus truth +0.0085 [+0.0055, +0.0113]; minus 2x truth -0.1055 [-0.1143, -0.0965]; condition not met | +0.1182 [+0.1084, +0.1267]; 1.04x truth; minus truth +0.0041 [+0.0009, +0.0074]; minus 2x truth -0.1099 [-0.1204, -0.0995]; condition not met |

Parity of this script's swapped render (601-bin path) against R1's stored scores on 2 rooms: {"worst_abs_difference": 2.740411508828089e-08, "tol": 0.001, "pass": true, "rooms": 2}

Missing renders (counted): wave equation none; reciprocity rooms scored are in the table (15 in family, 16 unseen when complete).
