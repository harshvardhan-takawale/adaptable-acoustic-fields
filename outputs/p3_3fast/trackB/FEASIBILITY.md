# FT-B — doorway / divider aperture: feasibility

**Verdict: GO** — (i) smoothness GO, (ii) linearizing coordinate GO, (iii) effect size GO.

Domain 8.0x4.0 m, alpha = 0.15 on all four outer walls and on the divider at x = 4.0; dx = 0.01, fs = 61440.0, n = 122880 (T = 2.0 s, df = 0.5 Hz, lambda = 0.55827, anisotropic CFL = 0.78951 against a bound of 1.0). Grid [801, 401]. Source at [0.5, 0.5].

`a = 0.05` was **dropped as under-resolved**: under-resolved: 0.05 m is 5 cells at dx = 0.01 and `_apply_slab` needs >= 3 open nodes for the two edge nodes to carry the boundary condition, so the staircased tips occupy 2 of 5 cells. A0c already measured the aperture observable moving 10.4x the floor between dx 0.02 and 0.01, so a 5-cell aperture is inside the un-converged regime by construction.

Receiver grids: the 8x8 is carried for pipeline compatibility only — cond(Phi) on the full-domain basis is **1.46e+16** against the `modal_projection` gate of 5.0. All measurement uses the 16x8 (cond = 1.61). The sub-room even/odd basis has cond = 1.588.


## 1. Sub-room modes and splitting

each sub-room's 64 receivers are projected onto the 3.99 x 4.0 sub-room mode shapes (room B mirrored through x = 4.0 into the same local frame), giving per-sub-room modal frequencies and -3 dB bandwidths; the even/odd decomposition H_+- = (H(x) +- H(8-x))/2 separates the two members of each near-degenerate pair BEFORE peak fitting, so splitting = |f_odd - f_even| for the same (n_x, n_y)

> **CAVEAT.** sub-room (1,0) = 42.98 Hz and (0,1) = 42.87 Hz are 0.107 Hz apart inside a ~3.07 Hz linewidth, so the SPLITTING observable is poorly conditioned in THIS 8.0 x 4.0 domain for the axial pair -- reported, but do not lean on it. Mode (1,1) at 60.71 Hz is isolated (nearest sub-room neighbours 17.7 and 25.0 Hz away) and is the tracked mode. The other three observables are unaffected.

Tracked mode (1,1), analytic 60.710 Hz, followed by continuity with a +/-6.0 Hz window. Points where the peak landed within 2 bins of the tracking window edge (and are therefore the only ones where the branch could have been lost): **[3.0]**.

| a (m) | f_even (Hz) | f_odd (Hz) | BW even (Hz) | BW odd (Hz) | splitting (Hz) | split/BW | migration | edge? |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 0.00 | 60.668 | 60.668 | 4.44 | 4.44 | 0.000 | 0.00 | 0.00 |  |
| 0.10 | 60.670 | 60.679 | 4.44 | 4.44 | 0.009 | 0.00 | 0.00 |  |
| 0.20 | 60.672 | 60.719 | 4.44 | 4.44 | 0.046 | 0.01 | 0.00 |  |
| 0.30 | 60.675 | 60.788 | 4.44 | 4.44 | 0.113 | 0.03 | 0.01 |  |
| 0.50 | 60.680 | 61.007 | 4.44 | 4.44 | 0.327 | 0.07 | 0.02 |  |
| 0.70 | 60.684 | 61.348 | 4.43 | 4.44 | 0.664 | 0.15 | 0.04 |  |
| 1.00 | 60.689 | 62.084 | 4.42 | 4.44 | 1.395 | 0.32 | 0.08 |  |
| 1.40 | 60.694 | 63.555 | 4.37 | 4.42 | 2.861 | 0.65 | 0.17 |  |
| 2.00 | 60.694 | 66.936 | 4.24 | 4.29 | 6.242 | 1.46 | 0.37 |  |
| 3.00 | 60.667 | 72.500 | 3.86 | 6.31 | 11.833 | 2.33 | 0.71 | O |
| 4.00 | 60.610 | 77.383 | 3.33 | 3.32 | 16.774 | 5.05 | 1.00 |  |

Per-sub-room modal frequencies and bandwidths for every one of the 21 sub-room modes below 200 Hz (rooms A and B measured independently on the same basis) are in `aperture_sweep.json` under `observable_1_modal.per_a_full_table`. Room-A vs room-B mean -3 dB bandwidth per aperture:

| a (m) | mean BW room A (Hz) | mean BW room B (Hz) | n modes resolved A / B |
|---:|---:|---:|---:|
| 0.00 | 4.03 | nan | 21 / 0 |
| 0.10 | 5.16 | 4.07 | 21 / 21 |
| 0.20 | 4.95 | 4.02 | 21 / 19 |
| 0.30 | 4.84 | 4.21 | 21 / 21 |
| 0.50 | 4.59 | 4.61 | 21 / 21 |
| 0.70 | 4.55 | 4.50 | 21 / 21 |
| 1.00 | 4.71 | 4.39 | 21 / 20 |
| 1.40 | 4.40 | 4.75 | 20 / 21 |
| 2.00 | 4.67 | 4.74 | 20 / 20 |
| 3.00 | 5.03 | 4.60 | 19 / 21 |
| 4.00 | 3.38 | 3.40 | 21 / 21 |

## 2. Inter-room level difference

20 log10( mean_|H| over the 64 room-B receivers / mean_|H| over the 64 room-A receivers ), per ISO third-octave band

| a (m) | broadband 20-300 Hz | 25 | 31.5 | 40 | 50 | 63 | 80 | 100 | 125 | 160 | 200 | 250 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | -inf | -inf | -inf | -inf | -inf | -inf | -inf | -inf | -inf | -inf | -inf | -inf |
| 0.10 | -12.61 | -12.6 | -17.2 | -9.5 | -9.8 | -26.4 | -18.9 | -9.0 | -11.0 | -12.9 | -14.8 | -12.3 |
| 0.20 | -11.27 | -11.5 | -15.9 | -8.8 | -8.0 | -25.8 | -18.4 | -7.5 | -9.0 | -11.7 | -13.6 | -10.8 |
| 0.30 | -10.43 | -10.7 | -15.0 | -8.4 | -7.0 | -24.8 | -18.1 | -6.6 | -7.8 | -10.8 | -12.5 | -9.9 |
| 0.50 | -9.25 | -9.4 | -13.6 | -7.8 | -5.7 | -21.8 | -17.7 | -5.7 | -6.2 | -9.3 | -10.2 | -8.8 |
| 0.70 | -8.32 | -8.4 | -12.6 | -7.3 | -4.9 | -18.1 | -17.4 | -5.1 | -5.2 | -7.7 | -8.3 | -7.9 |
| 1.00 | -7.15 | -7.2 | -11.3 | -6.3 | -4.1 | -13.0 | -16.9 | -4.7 | -4.6 | -5.5 | -6.6 | -6.7 |
| 1.40 | -5.76 | -5.9 | -10.0 | -4.8 | -3.3 | -7.6 | -15.4 | -4.2 | -4.6 | -3.4 | -5.5 | -5.5 |
| 2.00 | -4.04 | -4.3 | -8.4 | -3.1 | -2.7 | -2.3 | -10.2 | -3.3 | -4.7 | -2.1 | -5.1 | -4.7 |
| 3.00 | -2.11 | -2.6 | -6.9 | -2.1 | -1.7 | -0.5 | -3.1 | -1.2 | -2.3 | -1.9 | -5.3 | -2.8 |
| 4.00 | -1.45 | -1.9 | -6.5 | -1.7 | -1.0 | -0.7 | -1.9 | -1.0 | -1.4 | -1.1 | -2.3 | -1.6 |

## 3. Coupled decay

Schroeder integration of the 0-300 Hz band-limited IR energy summed over each sub-room's 64 receivers; single-slope and two-segment (searched-breakpoint) fits over the SAME -5..-25 dB window

Double-slope is accepted only if the two-segment RMS is <= 0.5x the single-segment RMS AND T60_late/T60_early >= 1.3 (fixed before the runs).

> **This observable FAILED to produce a usable signal, and one earlier version of it produced a false positive.** T60 is ~3.3 s at alpha = 0.15 but the frozen record is T = 2.0 s, so the EDC below about -30 dB is backward-integration truncation, not decay. The first pass used a -5..-45 dB window and reported an apparently clean 'double slope' with the late segment 4x STEEPER than the early one and the SAME breakpoint (1.70 s / -21.5 dB) for every aperture INCLUDING the sealed divider -- the tell that it was the record end, not room coupling. The window was cut to -5..-25 dB, which the record does support. The double- slope question is therefore answered NEGATIVELY within the supported range and is INCONCLUSIVE below -25 dB; resolving it needs a longer record or a more absorptive room, not a different fit.

> classical coupled-room double slope requires the two sub-volumes to have DIFFERENT decay rates. Here the sub-rooms are exact mirror images with identical alpha = 0.15, so they share one decay constant by construction and no aperture width can split it. Measured: T60 moves only 3.324 -> 3.301 s (0.7%) across the whole sweep and the two sub-rooms agree to < 0.01 s. This axis's decay observable only becomes informative once the sub-rooms differ in absorption -- a real design note for any FT-B follow-up.

| a (m) | room | T60 single (s) | T60 early (s) | T60 late (s) | RMS ratio | double? | transition (s / dB) |
|---:|:--|---:|---:|---:|---:|:--|---:|
| 0.00 | A | 3.324 | 4.395 | 2.063 | 0.26 | no | 1.426 / -14.2 |
| 0.00 | B | — | — | — | — | — | no energy in this sub-room (sealed divider) |
| 0.10 | A | 3.324 | 4.395 | 2.063 | 0.26 | no | 1.424 / -14.2 |
| 0.10 | B | 3.331 | 4.406 | 2.067 | 0.26 | no | 1.428 / -14.2 |
| 0.20 | A | 3.324 | 4.394 | 2.063 | 0.26 | no | 1.425 / -14.2 |
| 0.20 | B | 3.330 | 4.405 | 2.067 | 0.26 | no | 1.428 / -14.2 |
| 0.30 | A | 3.324 | 4.394 | 2.062 | 0.26 | no | 1.425 / -14.2 |
| 0.30 | B | 3.328 | 4.402 | 2.066 | 0.26 | no | 1.428 / -14.2 |
| 0.50 | A | 3.323 | 4.393 | 2.061 | 0.26 | no | 1.427 / -14.2 |
| 0.50 | B | 3.327 | 4.400 | 2.064 | 0.26 | no | 1.429 / -14.2 |
| 0.70 | A | 3.322 | 4.391 | 2.060 | 0.26 | no | 1.429 / -14.2 |
| 0.70 | B | 3.326 | 4.398 | 2.063 | 0.26 | no | 1.430 / -14.2 |
| 1.00 | A | 3.320 | 4.388 | 2.058 | 0.26 | no | 1.431 / -14.2 |
| 1.00 | B | 3.324 | 4.394 | 2.061 | 0.26 | no | 1.432 / -14.2 |
| 1.40 | A | 3.318 | 4.385 | 2.056 | 0.26 | no | 1.434 / -14.2 |
| 1.40 | B | 3.321 | 4.390 | 2.058 | 0.26 | no | 1.435 / -14.2 |
| 2.00 | A | 3.315 | 4.379 | 2.052 | 0.26 | no | 1.438 / -14.3 |
| 2.00 | B | 3.317 | 4.384 | 2.054 | 0.26 | no | 1.439 / -14.3 |
| 3.00 | A | 3.309 | 4.371 | 2.046 | 0.26 | no | 1.446 / -14.3 |
| 3.00 | B | 3.311 | 4.375 | 2.047 | 0.26 | no | 1.447 / -14.3 |
| 4.00 | A | 3.301 | 4.365 | 2.042 | 0.26 | no | 1.453 / -14.2 |
| 4.00 | B | 3.303 | 4.369 | 2.043 | 0.26 | no | 1.454 / -14.2 |

## 4. When does it stop being two rooms?

**Criterion.** STATED IN TERMS OF MODAL FREQUENCY MIGRATION. The odd branch of sub-room mode (1,1) starts at the sealed-divider sub-room frequency f_sealed and ends, with the divider absent, at the full-domain frequency f_open; define migration(a) = |f_odd(a) - f_sealed| / |f_open - f_sealed|. The domain STOPS BEHAVING AS TWO ROOMS at the a where migration crosses 0.5, i.e. where the mode is closer in frequency to the one-room eigenvalue than to the two-room one. Two supporting crossings are reported but do not define the verdict: (a) the first a at which the migration exceeds one modal linewidth, i.e. the smallest coupling that is spectrally RESOLVABLE at all, and (b) the a at which the even/odd splitting exceeds the linewidth.

Migration is |f_odd(a) - f_odd(sealed)| / |f_odd(no wall) - f_odd(sealed)| for the tracked (1,1) odd branch. The tracked odd branch runs from 60.668 Hz (sealed) to 77.383 Hz (no divider), a span of 16.715 Hz against a median linewidth of 4.440 Hz.

**Endpoint check.** the odd branch of sub-room (1,1) must land on full-domain (3,1) once the divider is gone. It does, which is the independent proof that the continuity tracker never hopped branches -- including at a = 3.0, the one point flagged near the tracking-window edge. Measured 77.383 Hz vs analytic 77.294 Hz, 0.12% relative error.

**Stops behaving as two rooms at a = 2.376 m (0.594 of the divider width).** First spectrally resolvable coupling (migration > one linewidth) at a = 1.680 m; even/odd splitting exceeds the linewidth at a = 1.658 m.


## GO / NO-GO

**(i) Smoothness — GO.** Largest adjacent jump 1.93 dB between a = 2.0 and a = 3.0, which is **17.3%** of the total range (11.16 dB); threshold < 0.25.

> computed over the FINITE points a > 0. a = 0 is not a limit point: a sealed one-node divider disconnects room B exactly, so H_B == 0 and the level difference is -inf. That is a topological discontinuity, not a large jump, and no continuous coordinate can include it -- the trainable range is a in (0, 4]. The largest jump also falls across the WIDEST sampling gap (a: 2.0 -> 3.0, the only 1.0 m step in the sweep), so it measures the sweep spacing as much as the physics -- it is not a discontinuity in the response.

**(ii) Linearizing coordinate — GO.** Best pooled coordinate is **sqrt a** with r^2 = 0.9870 (threshold >= 0.95).

| coordinate | r^2 pooled (a>0) | r^2 (a <= 2.0) | r^2 (a >= 1.0) |
|:--|---:|---:|---:|
| a | 0.9048 | 0.9533 | 0.9343 |
| a^2 | 0.7043 | 0.7964 | 0.8327 |
| area (= a * 1 m, affine copy of a) | 0.9048 | 0.9533 | 0.9343 |
| log a | 0.9695 | 0.9689 | 0.9919 |
| sqrt a | 0.9870 | 0.9979 | 0.9715 |

Best on a <= 2.0: **sqrt a** (r^2 = 0.9979); best on a >= 1.0: **log a** (r^2 = 0.9919).

> 2D slit physics is logarithmic at small a and area-like at large a, so the restricted fits a <= 2.0 and a >= 1.0 are reported alongside the pooled fit. In 2D the aperture 'area' is the clear width times unit depth = a, an affine copy of the `a` coordinate, so it necessarily shares its r^2.

**(iii) Effect size — GO.** Effect 11.16 dB vs a measured floor of 0.454 dB = **24.6x** (threshold >= 10.0x).

Floor method: mean |delta| of the broadband level difference between two runs of the SAME geometry differing only in source position ((0.5,0.5) vs (0.7,0.9) m), measured at a = 0.3 and a = 1.0 -- measured, not assumed. Per-config: a=0.3: |-10.432 - -11.230| = 0.799 dB; a=1.0: |-7.148 - -7.256| = 0.108 dB.


## Required addition — how densely must `a` be sampled?

linear interpolation between adjacent training samples errs by at most |y''| h^2 / 8; requiring that to stay under the MEASURED estimator floor gives h* = sqrt(8 * floor / |y''|). With the selected coordinate **sqrt a** over a range of 1.684, the worst-case curvature is 16.1 dB/unit^2 (median 3.69), and the tolerance is the measured floor 0.454 dB.

**Delta\* = 0.282 of the coordinate's range** (worst-case curvature); 0.589 using the median curvature. P3-2d measured Delta* ~ 0.275 of the linearizing coordinate's range on the absorption axis.

