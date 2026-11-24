# P4-10 editability scorecard (full): R1M_full

Off-lattice dense receivers (edited-field r / LSD) or the common receivers (r_delta, g_delta). Mean over rooms of each room's average over magnitudes; CIs = paired bootstrap over rooms (10k, seed 20261013). Bands 25-100 / 100-200 / 200-300 Hz. In-family 15 is primary, unseen beside. 'ignore the edit' = the original room's truth scored against the edited truth. m1 = 10 cm / 350 Hz physics (reference, not a baseline to beat).

## StretchU -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.827 / 0.599 / 0.494 | 0.796 / 0.562 / 0.473 | 2.569 / 1.569 / 1.458 |
| R1_150K | 0.825 / 0.599 / 0.492 | 0.801 / 0.573 / 0.478 | 2.729 / 1.598 / 1.475 |
| R1S_150K | 0.863 / 0.629 / 0.522 | 0.828 / 0.590 / 0.492 | 2.805 / 1.622 / 1.517 |
| R0_120K | 0.833 / 0.606 / 0.493 | 0.818 / 0.589 / 0.486 | 2.988 / 1.646 / 1.489 |
| ignore the edit (original truth) | 0.410 / 0.171 / 0.119 | -- | 1 (by definition) |
| m1 physics (reference) | 0.982 / 0.924 / 0.762 | 0.977 / 0.922 / 0.751 | -- |
| naive warp of the original truth | 0.608 / 0.323 / 0.190 | -- | -- |

## StretchU -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.461 / 0.382 / 0.347 | 0.296 / 0.249 / 0.281 | 1.177 / 1.179 / 1.206 |
| R1_150K | 0.442 / 0.356 / 0.340 | 0.287 / 0.230 / 0.278 | 1.181 / 1.171 / 1.216 |
| R1S_150K | 0.492 / 0.409 / 0.389 | 0.325 / 0.264 / 0.308 | 1.227 / 1.208 / 1.261 |
| R0_120K | 0.402 / 0.325 / 0.304 | 0.253 / 0.203 / 0.249 | 1.149 / 1.143 / 1.188 |
| ignore the edit (original truth) | 0.440 / 0.153 / 0.103 | -- | 1 (by definition) |
| m1 physics (reference) | 0.969 / 0.903 / 0.754 | 0.953 / 0.903 / 0.743 | -- |
| naive warp of the original truth | 0.631 / 0.313 / 0.188 | -- | -- |

## StretchA -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.834 / 0.608 / 0.499 | 0.747 / 0.522 / 0.309 | 2.284 / 1.478 / 1.228 |
| R1_150K | 0.832 / 0.609 / 0.494 | 0.751 / 0.536 / 0.317 | 2.380 / 1.503 / 1.241 |
| R1S_150K | 0.872 / 0.633 / 0.527 | 0.788 / 0.552 / 0.328 | 2.482 / 1.524 / 1.273 |
| R0_120K | 0.835 / 0.613 / 0.495 | 0.766 / 0.552 / 0.327 | 2.542 / 1.545 / 1.245 |
| ignore the edit (original truth) | 0.514 / 0.250 / 0.281 | -- | 1 (by definition) |
| m1 physics (reference) | 0.984 / 0.932 / 0.769 | 0.971 / 0.920 / 0.712 | -- |
| naive warp of the original truth | 0.663 / 0.401 / 0.309 | -- | -- |

## StretchA -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.477 / 0.394 / 0.368 | 0.270 / 0.193 / 0.150 | 1.169 / 1.124 / 1.101 |
| R1_150K | 0.461 / 0.373 / 0.359 | 0.267 / 0.180 / 0.149 | 1.171 / 1.116 / 1.105 |
| R1S_150K | 0.499 / 0.410 / 0.396 | 0.301 / 0.199 / 0.163 | 1.202 / 1.141 / 1.125 |
| R0_120K | 0.416 / 0.347 / 0.327 | 0.224 / 0.164 / 0.135 | 1.144 / 1.103 / 1.090 |
| ignore the edit (original truth) | 0.544 / 0.255 / 0.255 | -- | 1 (by definition) |
| m1 physics (reference) | 0.969 / 0.905 / 0.757 | 0.945 / 0.893 / 0.700 | -- |
| naive warp of the original truth | 0.689 / 0.403 / 0.291 | -- | -- |

## Wall -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.888 / 0.651 / 0.554 | 0.713 / 0.471 / 0.440 | 1.867 / 1.382 / 1.410 |
| R1_150K | 0.891 / 0.658 / 0.557 | 0.721 / 0.486 / 0.448 | 1.970 / 1.407 / 1.432 |
| R1S_150K | 0.899 / 0.667 / 0.565 | 0.732 / 0.491 / 0.454 | 1.969 / 1.412 / 1.444 |
| R0_120K | 0.900 / 0.673 / 0.562 | 0.743 / 0.506 / 0.458 | 2.143 / 1.454 / 1.448 |
| ignore the edit (original truth) | 0.735 / 0.421 / 0.233 | -- | 1 (by definition) |
| m1 physics (reference) | 0.984 / 0.928 / 0.764 | 0.959 / 0.899 / 0.732 | -- |

## Wall -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.499 / 0.410 / 0.380 | 0.225 / 0.166 / 0.229 | 1.075 / 1.103 / 1.150 |
| R1_150K | 0.483 / 0.392 / 0.372 | 0.218 / 0.158 / 0.223 | 1.079 / 1.102 / 1.157 |
| R1S_150K | 0.505 / 0.417 / 0.401 | 0.244 / 0.175 / 0.245 | 1.093 / 1.116 / 1.182 |
| R0_120K | 0.438 / 0.363 / 0.347 | 0.179 / 0.144 / 0.210 | 1.064 / 1.091 / 1.144 |
| ignore the edit (original truth) | 0.752 / 0.413 / 0.213 | -- | 1 (by definition) |
| m1 physics (reference) | 0.972 / 0.912 / 0.762 | 0.922 / 0.876 / 0.724 | -- |

## Notch -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.901 / 0.672 / 0.557 | 0.602 / 0.301 / 0.207 | 1.361 / 1.209 / 1.176 |
| R1_150K | 0.906 / 0.681 / 0.560 | 0.614 / 0.310 / 0.209 | 1.424 / 1.228 / 1.183 |
| R1S_150K | 0.905 / 0.682 / 0.561 | 0.617 / 0.310 / 0.213 | 1.421 / 1.229 / 1.187 |
| R0_120K | 0.917 / 0.697 / 0.568 | 0.637 / 0.328 / 0.216 | 1.526 / 1.257 / 1.193 |
| ignore the edit (original truth) | 0.885 / 0.715 / 0.534 | -- | 1 (by definition) |
| m1 physics (reference) | 0.984 / 0.928 / 0.771 | 0.915 / 0.810 / 0.656 | -- |

## Notch -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.520 / 0.424 / 0.407 | 0.074 / 0.023 / 0.019 | 1.016 / 1.013 / 1.020 |
| R1_150K | 0.503 / 0.409 / 0.399 | 0.077 / 0.022 / 0.018 | 1.019 / 1.014 / 1.020 |
| R1S_150K | 0.512 / 0.419 / 0.411 | 0.077 / 0.023 / 0.018 | 1.021 / 1.015 / 1.021 |
| R0_120K | 0.464 / 0.380 / 0.372 | 0.067 / 0.019 / 0.019 | 1.026 / 1.014 / 1.017 |
| ignore the edit (original truth) | 0.916 / 0.760 / 0.624 | -- | 1 (by definition) |
| m1 physics (reference) | 0.972 / 0.912 / 0.759 | 0.854 / 0.767 / 0.584 | -- |

## Source -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.733 / 0.380 / 0.271 | 0.768 / 0.516 / 0.411 | 2.013 / 1.327 / 1.245 |
| R1_150K | 0.743 / 0.377 / 0.260 | 0.776 / 0.518 / 0.405 | 2.102 / 1.330 / 1.247 |
| R1S_150K | 0.755 / 0.383 / 0.266 | 0.783 / 0.520 / 0.409 | 2.148 / 1.330 / 1.241 |
| R0_120K | 0.231 / 0.083 / 0.044 | 0.504 / 0.391 / 0.304 | 0.974 / 0.994 / 0.935 |
| ignore the edit (original truth) | 0.305 / 0.217 / 0.215 | -- | 1 (by definition) |
| m1 physics (reference) | 0.985 / 0.947 / 0.785 | 0.979 / 0.934 / 0.756 | -- |

## Source -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.550 / 0.330 / 0.262 | 0.478 / 0.381 / 0.329 | 1.483 / 1.295 / 1.306 |
| R1_150K | 0.546 / 0.321 / 0.246 | 0.469 / 0.367 / 0.317 | 1.500 / 1.294 / 1.316 |
| R1S_150K | 0.547 / 0.328 / 0.250 | 0.473 / 0.378 / 0.326 | 1.507 / 1.297 / 1.295 |
| R0_120K | 0.212 / 0.110 / 0.062 | 0.283 / 0.253 / 0.226 | 1.024 / 1.038 / 1.001 |
| ignore the edit (original truth) | 0.344 / 0.246 / 0.209 | -- | 1 (by definition) |
| m1 physics (reference) | 0.972 / 0.907 / 0.765 | 0.961 / 0.903 / 0.744 | -- |

## Material -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.894 / 0.713 / 0.618 | 0.442 / 0.180 / 0.122 | 1.509 / 1.119 / 1.059 |
| R1_150K | 0.749 / 0.567 / 0.476 | 0.094 / 0.023 / 0.017 | 0.842 / 0.857 / 0.867 |
| R1S_150K | 0.749 / 0.558 / 0.484 | 0.091 / 0.026 / 0.020 | 0.843 / 0.855 / 0.877 |
| R0_120K | 0.772 / 0.583 / 0.469 | 0.067 / 0.026 / 0.017 | 0.843 / 0.843 / 0.850 |
| ignore the edit (original truth) | 0.911 / 0.891 / 0.880 | -- | 1 (by definition) |
R1M_150K: MM1 EDT slope 0.54 (Pearson 0.95, n 180); T20 slope 0.17 (n 129; 429477 of 543192 receiver-bands dropped by the model-floor rule); MM2 |EDT error| edited 22.5% vs unedited 24.6% (margin -2.1 points); level-change slope 0.45 (Pearson 0.96); 0 cell-bands dropped
R1_150K: MM1 EDT slope 0.01 (Pearson 0.06, n 180); T20 slope -0.19 (n 97; 443158 of 543192 receiver-bands dropped by the model-floor rule); MM2 |EDT error| edited 36.5% vs unedited 23.9% (margin +12.6 points); level-change slope 0.21 (Pearson 0.70); 0 cell-bands dropped
R1S_150K: MM1 EDT slope 0.01 (Pearson 0.08, n 180); T20 slope -0.03 (n 100; 460655 of 543192 receiver-bands dropped by the model-floor rule); MM2 |EDT error| edited 35.7% vs unedited 23.9% (margin +11.8 points); level-change slope 0.23 (Pearson 0.70); 0 cell-bands dropped
R0_120K: MM1 EDT slope -0.04 (Pearson -0.26, n 180); T20 slope -0.12 (n 118; 386162 of 543192 receiver-bands dropped by the model-floor rule); MM2 |EDT error| edited 37.8% vs unedited 23.4% (margin +14.4 points); level-change slope 0.09 (Pearson 0.33); 0 cell-bands dropped

## Material -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R1M_150K | 0.558 / 0.483 / 0.460 | 0.199 / 0.112 / 0.071 | 1.083 / 1.026 / 1.027 |
| R1_150K | 0.447 / 0.376 / 0.373 | 0.044 / -0.001 / -0.006 | 0.937 / 0.935 / 0.938 |
| R1S_150K | 0.455 / 0.385 / 0.397 | 0.045 / -0.002 / 0.002 | 0.938 / 0.928 / 0.944 |
| R0_120K | 0.428 / 0.357 / 0.347 | 0.027 / -0.006 / -0.000 | 0.959 / 0.949 / 0.948 |
| ignore the edit (original truth) | 0.921 / 0.897 / 0.884 | -- | 1 (by definition) |
R1M_150K: MM1 EDT slope 0.66 (Pearson 0.94, n 192); T20 slope 0.43 (n 135; 370742 of 505819 receiver-bands dropped by the model-floor rule); MM2 |EDT error| edited 29.0% vs unedited 29.4% (margin -0.4 points); level-change slope 0.44 (Pearson 0.96); 0 cell-bands dropped
R1_150K: MM1 EDT slope -0.05 (Pearson -0.18, n 192); T20 slope -0.11 (n 109; 420396 of 505819 receiver-bands dropped by the model-floor rule); MM2 |EDT error| edited 34.4% vs unedited 27.1% (margin +7.3 points); level-change slope 0.11 (Pearson 0.34); 0 cell-bands dropped
R1S_150K: MM1 EDT slope 0.00 (Pearson 0.02, n 192); T20 slope -0.21 (n 104; 441933 of 505819 receiver-bands dropped by the model-floor rule); MM2 |EDT error| edited 31.6% vs unedited 28.3% (margin +3.4 points); level-change slope 0.15 (Pearson 0.51); 0 cell-bands dropped
R0_120K: MM1 EDT slope -0.09 (Pearson -0.36, n 192); T20 slope -0.10 (n 126; 358031 of 505819 receiver-bands dropped by the model-floor rule); MM2 |EDT error| edited 34.8% vs unedited 27.0% (margin +7.8 points); level-change slope 0.01 (Pearson 0.03); 0 cell-bands dropped

