## Verdict: GREEN — 6 of 6 modes have spatial correlation ≥ 0.7

- run: `C2_latent_jitter` (inner loop: `B6`)
- L = 4.75 m, W = 4.00 m, fs = 4096 Hz, n_freq_bins = 4097
- modes inspected: first 6 distinct eigenfrequencies in (1, 150] Hz

| (n_x, n_y) | f (Hz) | spatial corr | node match | pred shape SNR (dB) | ISM shape SNR (dB) |
|---|---:|---:|---:|---:|---:|
| (1,0) | 36.1 | 0.936 | 1.00 | -0.2 | 6.3 |
| (0,1) | 42.9 | 0.995 | 0.80 | 3.1 | 3.5 |
| (1,1) | 56.1 | 0.948 | 0.91 | 0.8 | 6.1 |
| (2,0) | 72.2 | 0.850 | 0.50 | -7.5 | 1.6 |
| (2,1) | 84.0 | 0.876 | 0.69 | -7.8 | 1.0 |
| (0,2) | 85.8 | 0.930 | 0.18 | -2.5 | -3.4 |

Figures:
  - `figures/mode_10.png`
  - `figures/mode_01.png`
  - `figures/mode_11.png`
  - `figures/mode_20.png`
  - `figures/mode_21.png`
  - `figures/mode_02.png`
  - `figures/all_modes_overview.png`