## Verdict: GREEN — 6 of 6 modes have spatial correlation ≥ 0.7

- run: `C2_latent_jitter` (inner loop: `B6`)
- L = 3.75 m, W = 4.00 m, fs = 4096 Hz, n_freq_bins = 4097
- modes inspected: first 6 distinct eigenfrequencies in (1, 150] Hz

| (n_x, n_y) | f (Hz) | spatial corr | node match | pred shape SNR (dB) | ISM shape SNR (dB) |
|---|---:|---:|---:|---:|---:|
| (0,1) | 42.9 | 0.933 | 0.17 | 2.9 | 3.2 |
| (1,0) | 45.7 | 0.851 | 0.25 | -6.6 | 1.5 |
| (1,1) | 62.7 | 0.848 | 0.50 | -3.1 | 6.2 |
| (0,2) | 85.8 | 0.955 | 0.33 | -1.9 | 0.3 |
| (2,0) | 91.5 | 0.817 | 0.33 | -14.4 | -1.5 |
| (1,2) | 97.2 | 0.871 | 0.40 | -5.0 | 0.2 |

Figures:
  - `figures/mode_01.png`
  - `figures/mode_10.png`
  - `figures/mode_11.png`
  - `figures/mode_02.png`
  - `figures/mode_20.png`
  - `figures/mode_12.png`
  - `figures/all_modes_overview.png`