## Verdict: GREEN — 6 of 6 modes have spatial correlation ≥ 0.7

- run: `C2_latent_jitter` (inner loop: `B6`)
- L = 4.25 m, W = 4.00 m, fs = 4096 Hz, n_freq_bins = 4097
- modes inspected: first 6 distinct eigenfrequencies in (1, 150] Hz

| (n_x, n_y) | f (Hz) | spatial corr | node match | pred shape SNR (dB) | ISM shape SNR (dB) |
|---|---:|---:|---:|---:|---:|
| (1,0) | 40.4 | 0.888 | 0.40 | -5.8 | 2.9 |
| (0,1) | 42.9 | 0.977 | 0.67 | 2.3 | 1.9 |
| (1,1) | 58.9 | 0.890 | 0.55 | -2.0 | 6.3 |
| (2,0) | 80.7 | 0.862 | 0.83 | -9.3 | 0.4 |
| (0,2) | 85.8 | 0.926 | 0.40 | -2.9 | -1.5 |
| (2,1) | 91.4 | 0.838 | 0.40 | -8.7 | 0.2 |

Figures:
  - `figures/mode_10.png`
  - `figures/mode_01.png`
  - `figures/mode_11.png`
  - `figures/mode_20.png`
  - `figures/mode_02.png`
  - `figures/mode_21.png`
  - `figures/all_modes_overview.png`