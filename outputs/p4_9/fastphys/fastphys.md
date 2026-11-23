# P4-9 X2 (D190): fast physics for the geometry edit

m1 (the FDTD's own operator in modal form) re-solved on a coarser grid h with modes to f_max; polygon snapped by the grid; exact beta. Off-lattice r per band (25-100 / 100-200 / 200-300 Hz), frozen scorer. Times on legacy01.umiacs.umd.edu (Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz), shape 1000, 4 threads, medians (I1 protocol); geometry-edit latency = operator + eigen-solve + field.

**Parity (h = 2 cm, 600 Hz vs physics_sensitivity's m1 exact, 31 rooms): max |dr| = 8.8e-09 -> PASSES.**

| h | f_max | modes (shape 1000) | single listener | full grid (3753 pts) | unseen 16 r | in-family 15 r | unseen LSD dB | unseen level offset dB |
|---|---|---|---|---|---|---|---|---|
| 2 cm | 350 Hz | 100 | 11.7 s | 11.8 s | 0.977 / 0.968 / 0.950 | 0.987 / 0.971 / 0.950 | 0.9 / 1.0 / 1.2 | -0.1 / -0.1 / -0.3 |
| 2 cm | 450 Hz | 161 | 21.5 s | 21.6 s | 0.977 / 0.971 / 0.958 | 0.988 / 0.974 / 0.959 | 0.9 / 0.9 / 1.1 | -0.1 / -0.1 / -0.3 |
| 2 cm | 600 Hz | 282 | 46.9 s | 47.0 s | 0.978 / 0.972 / 0.960 | 0.988 / 0.976 / 0.960 | 0.9 / 0.9 / 1.1 | -0.1 / -0.1 / -0.3 |
| 4 cm | 350 Hz | 100 | 2.6 s | 2.7 s | 0.978 / 0.959 / 0.918 | 0.988 / 0.967 / 0.933 | 0.9 / 1.1 / 1.7 | -0.1 / -0.1 / -0.3 |
| 4 cm | 450 Hz | 162 | 5.1 s | 5.2 s | 0.979 / 0.962 / 0.925 | 0.989 / 0.969 / 0.940 | 0.9 / 1.1 / 1.6 | -0.1 / -0.1 / -0.3 |
| 4 cm | 600 Hz | 284 | 12.2 s | 12.4 s | 0.979 / 0.963 / 0.927 | 0.989 / 0.971 / 0.942 | 0.9 / 1.1 / 1.6 | -0.1 / -0.1 / -0.3 |
| 6 cm | 350 Hz | 100 | 889 ms | 960 ms | 0.977 / 0.945 / 0.886 | 0.987 / 0.949 / 0.891 | 0.9 / 1.4 / 2.1 | -0.1 / -0.1 / -0.4 |
| 6 cm | 450 Hz | 162 | 1.9 s | 2.0 s | 0.977 / 0.947 / 0.892 | 0.987 / 0.951 / 0.897 | 0.9 / 1.4 / 2.0 | -0.1 / -0.1 / -0.4 |
| 6 cm | 600 Hz | 285 | 5.2 s | 5.4 s | 0.978 / 0.948 / 0.894 | 0.987 / 0.952 / 0.898 | 0.9 / 1.4 / 2.0 | -0.1 / -0.1 / -0.4 |
| 8 cm | 350 Hz | 100 | 459 ms | 537 ms | 0.975 / 0.924 / 0.826 | 0.986 / 0.945 / 0.855 | 1.0 / 1.6 / 2.6 | -0.1 / -0.1 / -0.4 |
| 8 cm | 450 Hz | 165 | 926 ms | 1.0 s | 0.976 / 0.926 / 0.832 | 0.987 / 0.948 / 0.862 | 1.0 / 1.6 / 2.6 | -0.1 / -0.1 / -0.4 |
| 8 cm | 600 Hz | 294 | 2.9 s | 3.0 s | 0.976 / 0.927 / 0.833 | 0.987 / 0.949 / 0.863 | 1.0 / 1.6 / 2.6 | -0.1 / -0.1 / -0.4 |
| 10 cm | 350 Hz | 100 | 307 ms | 384 ms | 0.972 / 0.913 / 0.759 | 0.984 / 0.931 / 0.774 | 1.0 / 1.8 / 3.3 | -0.1 / -0.0 / -0.2 |
| 10 cm | 450 Hz | 166 | 620 ms | 714 ms | 0.973 / 0.915 / 0.764 | 0.985 / 0.933 / 0.780 | 1.0 / 1.8 / 3.2 | -0.1 / -0.0 / -0.2 |
| 10 cm | 600 Hz | 302 | 1.8 s | 1.9 s | 0.973 / 0.916 / 0.765 | 0.985 / 0.934 / 0.781 | 1.0 / 1.8 / 3.2 | -0.1 / -0.0 / -0.2 |
| network R0 @120K | -- | -- | 12 ms | 32.1 s | 0.468 / 0.380 / 0.371 | 0.918 / 0.701 / 0.582 | -- | -- |

## D190(c) verdict: does any configuration dominate the network (faster AND higher r in all 3 bands)?

* **unseen16 / single listener** (network 12 ms): no
* **unseen16 / full grid** (network 32.1 s): YES -- 14 of 15 configs; fastest h0.10_f350 at 384 ms
* **in_family15 / single listener** (network 12 ms): no
* **in_family15 / full grid** (network 32.1 s): YES -- 14 of 15 configs; fastest h0.10_f350 at 384 ms

Timing check on largest_in_family (L6.70_W5.38_d0.36_w0.72.h5, 4056 pts): h0.02_f350 21.3 s / 21.4 s; h0.02_f450 40.6 s / 40.7 s; h0.02_f600 93.1 s / 93.3 s; h0.04_f350 5.0 s / 5.1 s; h0.04_f450 10.0 s / 10.1 s; h0.04_f600 24.3 s / 24.5 s; h0.06_f350 1.9 s / 2.0 s; h0.06_f450 4.2 s / 4.3 s; h0.06_f600 11.5 s / 11.7 s; h0.08_f350 894 ms / 986 ms; h0.08_f450 2.2 s / 2.3 s; h0.08_f600 6.8 s / 7.0 s; h0.10_f350 567 ms / 650 ms; h0.10_f450 1.3 s / 1.4 s; h0.10_f600 4.6 s / 4.8 s

Timing check on largest_unseen (C3L6.22_W5.40_NWd0.52w0.96_NEd0.28w1.86_SEd0.32w2.02.h5, 3850 pts): h0.02_f350 16.3 s / 16.4 s; h0.02_f450 30.8 s / 30.9 s; h0.02_f600 69.9 s / 70.0 s; h0.04_f350 3.8 s / 3.9 s; h0.04_f450 7.5 s / 7.6 s; h0.04_f600 18.1 s / 18.2 s; h0.06_f350 1.4 s / 1.5 s; h0.06_f450 3.0 s / 3.1 s; h0.06_f600 8.3 s / 8.5 s; h0.08_f350 660 ms / 744 ms; h0.08_f450 1.5 s / 1.6 s; h0.08_f600 4.8 s / 4.9 s; h0.10_f350 428 ms / 511 ms; h0.10_f450 906 ms / 1.0 s; h0.10_f600 3.1 s / 3.2 s

## D190(e) scaling (timing only; C3L6.22_W5.40_NWd0.52w0.96_NEd0.28w1.86_SEd0.32w2.02.h5 on brigid17.umiacs.umd.edu; h = c / (10 f_max); one run, 30 min cap)

| f_max | h | air nodes | modes | eigen-solve |
|---|---|---|---|---|
| 300 Hz | 11.43 cm | 2490 | 88 | 306 ms |
| 600 Hz | 5.72 cm | 9899 | 336 | 3.2 s |
| 1000 Hz | 3.43 cm | 27223 | 904 | 62.3 s |
| 2000 Hz | 1.71 cm | -- | -- | > 30 min |
