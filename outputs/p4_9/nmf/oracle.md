# P4-9 2c: NMF oracle decomposition (D187; descriptive)

Off-lattice r at 25-100 Hz (frozen scorer), mean over rooms. Each column substitutes m1's exact eigenpairs for one half of the NMF (24 modes, paired by sort order). 'recovered' = share of the gap pred -> all-true closed by that substitution. D187(c): the larger single share names the bottleneck; if both are below 0.5, the errors interact and neither is named.

| arm | rooms | pred | true shapes + pred k | pred shapes + true k | all true (24 modes) | recovered: shapes / k |
|---|---|---|---|---|---|---|
| NMFP_30K | in_family15 | 0.828 | 0.446 | 0.415 | 0.983 | -2.46 / -2.65 |
| NMFP_30K | unseen16 | 0.438 | 0.444 | 0.414 | 0.974 | 0.01 / -0.04 |
| NMF_30K | in_family15 | 0.848 | 0.544 | 0.475 | 0.983 | -2.25 / -2.77 |
| NMF_30K | unseen16 | 0.451 | 0.484 | 0.413 | 0.974 | 0.06 / -0.07 |
