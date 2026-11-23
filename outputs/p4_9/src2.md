# P4-9 S-3: a second source position on the 16 frozen unseen rooms (D181 E8, D182(b); descriptive)

Every corpus so far uses ONE source, (0.5, 1.6). Here each frozen unseen room is re-simulated with a second source (D182(b)) on the same dense receivers; off-lattice r (Rule 11), mean over 16 rooms, band-mean per-bin Pearson of dB. Delta = second - original source, paired bootstrap over rooms (10k, seed 20261013).

**raw** (r: original source -> second source; delta [95% CI]; LSD dB original / second)

| field | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| m1 | 0.978 -> 0.978; +0.001 [-0.001, +0.003]; 0.9 / 0.9 | 0.972 -> 0.969; -0.003 [-0.006, -0.000]; 0.9 / 1.0 | 0.960 -> 0.960; -0.000 [-0.003, +0.003]; 1.1 / 1.1 |
| R0_120K | 0.468 -> 0.172; -0.295 [-0.390, -0.205]; 5.3 / 8.5 | 0.380 -> 0.077; -0.303 [-0.347, -0.259]; 5.7 / 7.7 | 0.371 -> 0.055; -0.316 [-0.363, -0.270]; 5.4 / 6.9 |
| S2_120K | 0.501 -> 0.386; -0.115 [-0.182, -0.041]; 5.1 / 7.1 | 0.419 -> 0.193; -0.226 [-0.264, -0.186]; 5.5 / 7.3 | 0.404 -> 0.155; -0.249 [-0.275, -0.220]; 5.2 / 6.0 |

**projected** (r: original source -> second source; delta [95% CI]; LSD dB original / second)

| field | 25-100 Hz | 100-200 Hz | 200-300 Hz |
|---|---|---|---|
| m1 | 0.974 -> 0.974; -0.001 [-0.003, +0.002]; 1.0 / 1.0 | 0.969 -> 0.965; -0.004 [-0.007, -0.001]; 1.0 / 1.0 | 0.957 -> 0.957; -0.000 [-0.003, +0.002]; 1.1 / 1.1 |
| R0_120K | 0.468 -> 0.173; -0.295 [-0.389, -0.204]; 5.3 / 8.5 | 0.379 -> 0.077; -0.302 [-0.347, -0.258]; 5.7 / 7.7 | 0.372 -> 0.055; -0.317 [-0.363, -0.270]; 5.4 / 6.9 |
| S2_120K | 0.501 -> 0.387; -0.114 [-0.182, -0.041]; 5.1 / 7.1 | 0.421 -> 0.195; -0.225 [-0.264, -0.185]; 5.5 / 7.3 | 0.408 -> 0.158; -0.250 [-0.276, -0.221]; 5.2 / 6.0 |

**Reference fields** (`scripts/p4_9_src2_diag.py`; raw, off-lattice, scored against the SECOND-source truth): R0's ORIGINAL-source render 0.174 / 0.031 / 0.052 ; the original-source TRUTH (a field that ignores the move) 0.357 / 0.204 / 0.210 (25-100 / 100-200 / 200-300 Hz).

Per room (sources, families, both variants): `outputs/p4_9/src2/<name>.json`.
