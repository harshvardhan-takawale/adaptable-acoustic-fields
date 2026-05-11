## Verdict: GREEN — 6 of 6 modes have spatial correlation ≥ 0.7

- run: `C2_latent_jitter` (inner loop: `B6`)
- L = 3.25 m, W = 4.00 m, fs = 4096 Hz, n_freq_bins = 4097
- modes inspected: first 6 distinct eigenfrequencies in (1, 150] Hz

| (n_x, n_y) | f (Hz) | spatial corr | node match | pred shape SNR (dB) | ISM shape SNR (dB) |
|---|---:|---:|---:|---:|---:|
| (0,1) | 42.9 | 0.960 | 0.00 | 3.0 | 6.8 |
| (1,0) | 52.8 | 0.754 | 0.50 | -9.0 | 3.5 |
| (1,1) | 68.0 | 0.839 | 0.36 | -4.1 | 6.0 |
| (0,2) | 85.8 | 0.932 | 0.00 | -1.7 | 1.6 |
| (1,2) | 100.7 | 0.871 | 0.60 | -5.7 | 1.6 |
| (2,0) | 105.5 | 0.807 | 0.22 | -15.2 | -2.9 |

Figures:
  - `figures/mode_01.png`
  - `figures/mode_10.png`
  - `figures/mode_11.png`
  - `figures/mode_02.png`
  - `figures/mode_12.png`
  - `figures/mode_20.png`
  - `figures/all_modes_overview.png`