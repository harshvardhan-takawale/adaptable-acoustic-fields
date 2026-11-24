# P4-10 editability scorecard (full): R0_full

Off-lattice dense receivers (edited-field r / LSD) or the common receivers (r_delta, g_delta). Mean over rooms of each room's average over magnitudes; CIs = paired bootstrap over rooms (10k, seed 20261013). Bands 25-100 / 100-200 / 200-300 Hz. In-family 15 is primary, unseen beside. 'ignore the edit' = the original room's truth scored against the edited truth. m1 = 10 cm / 350 Hz physics (reference, not a baseline to beat).

## StretchU -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.833 / 0.606 / 0.493 | 0.818 / 0.589 / 0.486 | 2.988 / 1.646 / 1.489 |
| ignore the edit (original truth) | 0.410 / 0.171 / 0.119 | -- | 1 (by definition) |
| m1 physics (reference) | 0.982 / 0.924 / 0.762 | 0.977 / 0.922 / 0.751 | -- |
| naive warp of the original truth | 0.608 / 0.323 / 0.190 | -- | -- |

## StretchU -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.402 / 0.325 / 0.304 | 0.253 / 0.203 / 0.249 | 1.149 / 1.143 / 1.188 |
| ignore the edit (original truth) | 0.440 / 0.153 / 0.103 | -- | 1 (by definition) |
| m1 physics (reference) | 0.969 / 0.903 / 0.754 | 0.953 / 0.903 / 0.743 | -- |
| naive warp of the original truth | 0.631 / 0.313 / 0.188 | -- | -- |

## StretchA -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.835 / 0.613 / 0.495 | 0.766 / 0.552 / 0.327 | 2.542 / 1.545 / 1.245 |
| ignore the edit (original truth) | 0.514 / 0.250 / 0.281 | -- | 1 (by definition) |
| m1 physics (reference) | 0.984 / 0.932 / 0.769 | 0.971 / 0.920 / 0.712 | -- |
| naive warp of the original truth | 0.663 / 0.401 / 0.309 | -- | -- |

## StretchA -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.416 / 0.347 / 0.327 | 0.224 / 0.164 / 0.135 | 1.144 / 1.103 / 1.090 |
| ignore the edit (original truth) | 0.544 / 0.255 / 0.255 | -- | 1 (by definition) |
| m1 physics (reference) | 0.969 / 0.905 / 0.757 | 0.945 / 0.893 / 0.700 | -- |
| naive warp of the original truth | 0.689 / 0.403 / 0.291 | -- | -- |

## Wall -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.900 / 0.673 / 0.562 | 0.743 / 0.506 / 0.458 | 2.143 / 1.454 / 1.448 |
| ignore the edit (original truth) | 0.735 / 0.421 / 0.233 | -- | 1 (by definition) |
| m1 physics (reference) | 0.984 / 0.928 / 0.764 | 0.959 / 0.899 / 0.732 | -- |

## Wall -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.438 / 0.363 / 0.347 | 0.179 / 0.144 / 0.210 | 1.064 / 1.091 / 1.144 |
| ignore the edit (original truth) | 0.752 / 0.413 / 0.213 | -- | 1 (by definition) |
| m1 physics (reference) | 0.972 / 0.912 / 0.762 | 0.922 / 0.876 / 0.724 | -- |

## Notch -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.917 / 0.697 / 0.568 | 0.637 / 0.328 / 0.216 | 1.526 / 1.257 / 1.193 |
| ignore the edit (original truth) | 0.885 / 0.715 / 0.534 | -- | 1 (by definition) |
| m1 physics (reference) | 0.984 / 0.928 / 0.771 | 0.915 / 0.810 / 0.656 | -- |

## Notch -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.464 / 0.380 / 0.372 | 0.067 / 0.019 / 0.019 | 1.026 / 1.014 / 1.017 |
| ignore the edit (original truth) | 0.916 / 0.760 / 0.624 | -- | 1 (by definition) |
| m1 physics (reference) | 0.972 / 0.912 / 0.759 | 0.854 / 0.767 / 0.584 | -- |

## Source -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.231 / 0.083 / 0.044 | 0.504 / 0.391 / 0.304 | 0.974 / 0.994 / 0.935 |
| ignore the edit (original truth) | 0.305 / 0.217 / 0.215 | -- | 1 (by definition) |
| m1 physics (reference) | 0.985 / 0.947 / 0.785 | 0.979 / 0.934 / 0.756 | -- |

## Source -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.212 / 0.110 / 0.062 | 0.283 / 0.253 / 0.226 | 1.024 / 1.038 / 1.001 |
| ignore the edit (original truth) | 0.344 / 0.246 / 0.209 | -- | 1 (by definition) |
| m1 physics (reference) | 0.972 / 0.907 / 0.765 | 0.961 / 0.903 / 0.744 | -- |

## Material -- in-family (15 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.772 / 0.583 / 0.469 | 0.067 / 0.026 / 0.017 | 0.843 / 0.843 / 0.850 |
| ignore the edit (original truth) | 0.911 / 0.891 / 0.880 | -- | 1 (by definition) |
R0_120K: MM1 EDT slope -0.04 (Pearson -0.26, n 180); T20 slope -0.12 (n 118; 386162 of 543192 receiver-bands dropped by the model-floor rule); MM2 |EDT error| edited 37.8% vs unedited 23.4% (margin +14.4 points); level-change slope 0.09 (Pearson 0.33); 0 cell-bands dropped

## Material -- unseen (16 rooms)

| field | edited r | r_delta | g_delta |
|---|---|---|---|
| R0_120K | 0.428 / 0.357 / 0.347 | 0.027 / -0.006 / -0.000 | 0.959 / 0.949 / 0.948 |
| ignore the edit (original truth) | 0.921 / 0.897 / 0.884 | -- | 1 (by definition) |
R0_120K: MM1 EDT slope -0.09 (Pearson -0.36, n 192); T20 slope -0.10 (n 126; 358031 of 505819 receiver-bands dropped by the model-floor rule); MM2 |EDT error| edited 34.8% vs unedited 27.0% (margin +7.8 points); level-change slope 0.01 (Pearson 0.03); 0 cell-bands dropped

