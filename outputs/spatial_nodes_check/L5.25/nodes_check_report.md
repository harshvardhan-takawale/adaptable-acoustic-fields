## Verdict: GREEN — 6 of 6 modes have spatial correlation ≥ 0.7

- run: `C2_latent_jitter` (inner loop: `B6`)
- L = 5.25 m, W = 4.00 m, fs = 4096 Hz, n_freq_bins = 4097
- modes inspected: first 6 distinct eigenfrequencies in (1, 150] Hz

| (n_x, n_y) | f (Hz) | spatial corr | node match | pred shape SNR (dB) | ISM shape SNR (dB) |
|---|---:|---:|---:|---:|---:|
| (1,0) | 32.7 | 0.949 | 0.33 | 1.2 | 7.5 |
| (0,1) | 42.9 | 0.994 | 1.00 | 3.2 | 3.5 |
| (1,1) | 53.9 | 0.969 | 0.83 | 1.6 | 5.7 |
| (2,0) | 65.3 | 0.870 | 0.75 | -7.2 | 2.0 |
| (2,1) | 78.1 | 0.894 | 0.92 | -4.5 | 3.1 |
| (0,2) | 85.8 | 0.964 | 0.82 | -3.4 | -1.9 |

Figures:
  - `figures/mode_10.png`
  - `figures/mode_01.png`
  - `figures/mode_11.png`
  - `figures/mode_20.png`
  - `figures/mode_21.png`
  - `figures/mode_02.png`
  - `figures/all_modes_overview.png`