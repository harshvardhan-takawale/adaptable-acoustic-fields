# P4-9 X3 diagnostic (D193): reciprocity gap

Off-lattice interior receivers; r per band vs the SAME truth: normal render (receiver r, source s) and swapped render (receiver s, source r). A reciprocal model scores the same both ways; last column = r(swapped, normal).

| model | set | normal | swapped | swapped vs normal |
|---|---|---|---|---|
| CTL_150K | in_family15 | 0.920 / 0.708 / 0.583 | 0.170 / 0.050 / 0.034 | 0.173 / 0.069 / 0.033 |
| CTL_150K | unseen16 | 0.467 / 0.376 / 0.373 | 0.121 / 0.052 / 0.028 | 0.189 / 0.079 / 0.042 |
| R0_120K | in_family15 | 0.918 / 0.701 / 0.582 | 0.167 / 0.050 / 0.027 | 0.172 / 0.070 / 0.030 |
| R0_120K | unseen16 | 0.468 / 0.380 / 0.371 | 0.122 / 0.054 / 0.027 | 0.189 / 0.082 / 0.038 |
