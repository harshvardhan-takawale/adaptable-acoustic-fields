# P4-10 editability scorecard (full): R0_source

Off-lattice dense receivers (edited-field r / LSD) or the common receivers (r_delta, g_delta). Mean over rooms of each room's average over magnitudes; CIs = paired bootstrap over rooms (10k, seed 20261013). Bands 25-100 / 100-200 / 200-300 Hz. In-family 15 is primary, unseen beside. 'ignore the edit' = the original room's truth scored against the edited truth. m1 = 10 cm / 350 Hz physics (reference, not a baseline to beat).

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

