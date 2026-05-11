## Verdict: GREEN — 6 of 6 modes have spatial correlation ≥ 0.7

- run: `C2_latent_jitter` (inner loop: `B6`)
- L = 5.75 m, W = 4.00 m, fs = 4096 Hz, n_freq_bins = 4097
- modes inspected: first 6 distinct eigenfrequencies in (1, 150] Hz

| (n_x, n_y) | f (Hz) | spatial corr | node match | pred shape SNR (dB) | ISM shape SNR (dB) |
|---|---:|---:|---:|---:|---:|
| (1,0) | 29.8 | 0.950 | 0.00 | 1.4 | 8.4 |
| (0,1) | 42.9 | 0.984 | 0.50 | 1.5 | 3.1 |
| (1,1) | 52.2 | 0.963 | 0.80 | 0.6 | 5.0 |
| (2,0) | 59.7 | 0.846 | 0.40 | -7.0 | 1.8 |
| (2,1) | 73.5 | 0.900 | 0.92 | -3.7 | 3.5 |
| (0,2) | 85.8 | 0.952 | 0.69 | -4.2 | -2.7 |

Figures:
  - `figures/mode_10.png`
  - `figures/mode_01.png`
  - `figures/mode_11.png`
  - `figures/mode_20.png`
  - `figures/mode_21.png`
  - `figures/mode_02.png`
  - `figures/all_modes_overview.png`