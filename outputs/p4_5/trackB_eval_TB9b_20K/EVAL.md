# P3-3-FAST Track 2b evaluation

72 test configs | held-out band a in [0.9, 1.1] (n=18) vs seen (n=48) | 6 sealed excluded from every fit and aggregate

checkpoint `outputs/p4_5/arms/TB9b/ckpt_iter0020000.pt` (iter 20000) | in-dist val LSD 5.355 dB

Band 20.0-300.0 Hz, df 0.5 Hz. EDC fit window [-5.0, -25.0] dB (inherited from FT-B). Sub-room mode basis: highest f_max in [160.0, 140.0, 120.0, 100.0, 80.0, 60.0] Hz with cond(Phi) <= 5.0.

## 1. The continuity claim: level difference vs sqrt(a)

sqrt(a) is FT-B's measured linearizing coordinate (pooled r^2 = 0.9870 on GT in the single 8.0 x 4.0 domain).

| fit | n | sqrt(a) span | slope (dB per sqrt m) | intercept dB | r | r^2 |
|---|---|---|---|---|---|---|
| GT pooled (6 domains) | 66 | 1.710 | 7.611 | -15.289 | 0.9737 | 0.9481 |
| GT held_out_band | 18 | 0.050 | 8.717 | -16.079 | 0.3249 | 0.1055 |
| GT seen_apertures | 48 | 1.710 | 7.648 | -15.449 | 0.9782 | 0.9569 |
| PRED pooled | 66 | 1.710 | 3.924 | -9.813 | 0.7948 | 0.6317 |
| PRED held_out_band | 18 | 0.050 | 4.112 | -10.153 | 0.1003 | 0.0101 |
| PRED seen_apertures | 48 | 1.710 | 3.906 | -9.736 | 0.8089 | 0.6544 |

> **Read the held-out row's r^2 with care.** The band is a in [0.9, 1.1], so sqrt(a) spans only 0.050 against 1.710 for the seen apertures. A regression run INSIDE the band is near-degenerate by construction -- its r^2 measures across-domain (L, W, x0) scatter, not the aperture law. The two tests that actually answer the question are the seen-line residuals and (with a model) the pred-vs-GT regression, both below.

Per-domain fits (the pooled fit carries (L, W, x0) scatter the aperture law does not own): GT r^2 0.9693 +/- 0.0093 over 6 domains, GT slope 7.636 +/- 0.701.

**Do the held-out points sit on the seen line?** Fit sqrt(a) on the SEEN apertures only, then score the held-out points against that line.

| side | seen-line slope | RMS resid seen dB | RMS resid held-out dB | ratio |
|---|---|---|---|---|
| GT | 7.648 | 0.851 | 0.679 | 0.798 |
| PRED | 3.906 | 1.487 | 0.859 | 0.578 |

- **GROUND TRUTH: YES -- the held-out points sit on the seen line (held-out residuals are 0.80x the seen residuals)**
- **PREDICTION: YES -- the held-out points sit on the seen line (held-out residuals are 0.58x the seen residuals)**

| pred vs GT level difference | n | slope | r | r^2 |
|---|---|---|---|---|
| held_out_band | 18 | 0.068 | 0.0444 | 0.0020 |
| seen_apertures | 48 | 0.507 | 0.8205 | 0.6732 |

## 2. Observables, held-out vs seen

| observable | held_out_band | seen_apertures |
|---|---|---|
| n configs | 18 | 48 |
| level difference GT (dB) | -7.364 +/- 0.564 | -7.134 +/- 4.138 |
| LD usable-cell frac | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| level difference PRED (dB) | -6.042 +/- 0.861 | -5.490 +/- 2.556 |
| |LD error| (dB) | 1.349 +/- 0.970 | 2.086 +/- 2.149 |
| LD pearson pred vs GT | 0.0444 | 0.8205 |
| mode split GT (Hz) | 0.000 +/- 0.000 | 0.107 +/- 0.553 |
| mode |migration| GT (Hz) | 1.172 +/- 0.400 | 1.356 +/- 0.666 |
| modes / config | 26.5 +/- 2.6 | 26.5 +/- 2.5 |
| usable modes / config | 7.7 +/- 1.1 | 6.9 +/- 1.8 |
| frac_modes_dropped | 0.5101 +/- 0.0469 | 0.5577 +/- 0.1015 |
| n degenerate modes / config | 16.33 +/- 2.89 | 16.33 +/- 2.84 |
| Kuttruff linewidth (Hz) | 3.83 +/- 0.27 | 3.83 +/- 0.26 |
| mode split PRED (Hz) | 0.958 +/- 0.637 | 0.988 +/- 0.782 |
| |split error| (Hz) | 0.958 +/- 0.637 | 1.095 +/- 0.889 |
| mode |migration| PRED (Hz) | 1.829 +/- 0.244 | 1.827 +/- 0.526 |
| |peak position error| (Hz) | 1.678 +/- 0.397 | 1.903 +/- 0.744 |
| migration pearson | 0.1933 +/- 0.5136 | 0.1698 +/- 0.4848 |
| T60 room A GT (s) | 3.320 +/- 0.004 | 3.317 +/- 0.009 |
| T60 room B GT (s) | 3.323 +/- 0.004 | 3.321 +/- 0.009 |
| room B double-slope (GT) | 0/18 | 0/48 |
| T60 room A PRED (s) | 1.094 +/- 0.042 | 1.101 +/- 0.049 |
| T60 room B PRED (s) | 1.131 +/- 0.039 | 1.133 +/- 0.051 |
| room B double-slope (PRED) | 0 | 0 |
| LSD all (dB) | 7.826 +/- 0.933 | 7.692 +/- 1.151 |
|   GT dynamic range all (dB) | 84.5 +/- 5.9 | 83.5 +/- 6.1 |
|   LSD / dynamic range all | 0.0932 +/- 0.0141 | 0.0925 +/- 0.0141 |
|   usable-cell frac all | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| LSD room_A (dB) | 8.253 +/- 1.233 | 8.099 +/- 1.628 |
|   GT dynamic range room_A (dB) | 77.4 +/- 5.8 | 76.2 +/- 5.9 |
|   LSD / dynamic range room_A | 0.1071 +/- 0.0173 | 0.1074 +/- 0.0254 |
|   usable-cell frac room_A | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| LSD room_B (dB) | 7.323 +/- 0.705 | 7.209 +/- 0.868 |
|   GT dynamic range room_B (dB) | 77.0 +/- 7.2 | 75.6 +/- 6.5 |
|   LSD / dynamic range room_B | 0.0960 +/- 0.0138 | 0.0961 +/- 0.0140 |
|   usable-cell frac room_B | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

## 3. Level difference per ISO third-octave band

| fc Hz | GT held-out | PRED held-out | |err| | GT seen | PRED seen | |err| | usable frac |
|---|---|---|---|---|---|---|---|
| 25 | -6.54 | -5.08 | 1.64 | -6.34 | -4.66 | 2.83 | 1.000 |
| 32 | -10.50 | -8.14 | 3.90 | -10.48 | -8.10 | 3.89 | 1.000 |
| 40 | -6.81 | -6.63 | 2.70 | -6.52 | -6.12 | 3.14 | 1.000 |
| 50 | -7.77 | -7.04 | 1.54 | -7.58 | -6.38 | 2.33 | 1.000 |
| 63 | -13.46 | -8.54 | 4.92 | -11.98 | -7.07 | 5.33 | 1.000 |
| 80 | -11.64 | -8.11 | 3.52 | -10.07 | -6.50 | 3.68 | 1.000 |
| 100 | -3.80 | -5.61 | 2.03 | -4.08 | -5.57 | 1.85 | 1.000 |
| 125 | -4.29 | -4.93 | 1.02 | -5.38 | -4.91 | 1.67 | 1.000 |
| 160 | -6.01 | -4.23 | 1.78 | -6.95 | -4.35 | 3.44 | 1.000 |
| 200 | -7.62 | -4.04 | 3.58 | -7.89 | -3.75 | 4.14 | 1.000 |
| 250 | -7.92 | -3.58 | 4.34 | -8.09 | -3.66 | 4.45 | 1.000 |

## 4. Sealed (a = 0): topological reference, NOT a point on the curve

a = 0 disconnects room B EXACTLY, so H_B == 0 and the level difference is -inf. sqrt(0) = 0 is also the limit of a vanishing doorway, so the conditioning cannot separate the two cases; sealed configs were excluded from training and are excluded from every continuous fit and aggregate here.

- sealed configs: 6 | GT room-B field identically zero in all of them: True
- GT level difference: -inf in every sealed config (room B is disconnected)
- PRED level difference: -5.35 +/- 3.60 dB -- finite by construction. The model's coordinate is sqrt(a), and sqrt(0) = 0 is also the limit of a vanishing doorway, so this number is the size of the gap between the continuous law and the topological truth, not an error the model could have avoided.
- room-A LSD on sealed configs: 10.24 +/- 2.66 dB (room-B LSD is undefined -- the target is identically zero)

## 5. Caveats that condition every number above

1. **Mode split is poorly conditioned in near-square sub-rooms.** When a sub-room's (1,0) and (0,1) fall within one linewidth of each other (FT-B measured 0.107 Hz apart inside a ~3.07 Hz linewidth in its 8.0 x 4.0 domain), the peak-search window contains a DIFFERENT mode rather than the split partner and the observable is unidentifiable. Such modes are flagged `degenerate` per mode and counted per config (`n_degenerate`); they are excluded from the split statistics but the count is reported so the exclusion is visible. The other three observables are unaffected.
2. **The two-peak split is resolution-limited; the peak POSITION is not.** Resolving a doublet needs a separation of order the linewidth (3.83 +/- 0.26 Hz median here, and FT-B measured that the splitting only exceeds the linewidth for a >= 1.66 m), so `split_hz` is legitimately 0 across most of the aperture range. FT-B could see sub-linewidth splitting because its divider sat at x0 = L/2 and the two sub-rooms were exact mirror images, which let it decompose the field into even/odd branches BEFORE peak fitting. Track 2b varies x0, the sub-rooms are not congruent, and that decomposition does not exist -- so the well-conditioned modal observable here is the peak MIGRATION (`migration_*_hz`, `mean_abs_peak_error_hz`), and `split_hz` is reported beside it rather than leaned on.
3. **Absolute LSD is not comparable to earlier ISM chunks.** this FDTD corpus spans ~75 dB in band against the P3-2b ISM corpus's ~22 dB, so absolute LSD is NOT comparable to P3-2b's ~1.0 dB; use lsd_over_dynamic_range for any cross-chunk statement
4. **Every table is conditioned on a resolvable subset.** `frac_modes_dropped` and `frac_usable` sit next to the numbers they gate; a mode whose window is narrower than 3 bins, or a cell below the 1e-08 log floor, never enters a statistic.
5. **The 8x8 receiver grid leaves 3-5 columns per sub-room**, so the sub-room mode basis blows up (cond ~1e16) above ~160 Hz. The f_max ladder backs off until cond(Phi) <= 5.0; the chosen f_max and cond are recorded per sub-room in `per_config[*].modal.bases`.
6. **The record is 2.0 s and T60 is longer**, so the EDC below about -30 dB is backward-integration truncation rather than decay. The fit window stops at -25.0 dB for exactly that reason (FT-B's finding, inherited unchanged).

