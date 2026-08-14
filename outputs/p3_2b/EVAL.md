# P3-2b — cross-arm ablation

**S2 (unseen geometry x HELD-OUT m-slab): PASSED, first by arm B (`p3_2b_B_cont_fourier`).**
**The change that produced it: continuous alpha sampling (B - A).**

## The S2 gate

Spec `p3_2b.accept/1`, thresholds sha256 `a8479c5e1dcc` (frozen in `aaf/eval/p3_2b_accept.py` before any arm was evaluated). All four criteria must hold on **S2_unseen_geom_slab** and no blocker may fire.

| criterion | op | threshold | A | B | C | D |
|---|---|---|---|---|---|---|
| `edit_bw_slope` | >= | 0.80 | 0.153 FAIL | 1.147 PASS | 0.959 PASS | 1.038 PASS |
| `edit_bw_pearson` | >= | 0.80 | 0.499 FAIL | 0.871 PASS | 0.868 PASS | 0.871 PASS |
| `edit_gain` | > | 1.00 | 0.868 FAIL | 1.084 PASS | 1.087 PASS | 1.082 PASS |
| `abs_rho_minus_1` | <= | 0.25 | 0.491 FAIL | 0.039 PASS | 0.053 PASS | 0.031 PASS |
| **verdict** |  |  | FAIL | **PASS** | **PASS** | **PASS** |


## Ablation table

`slope` is `edit_bw_slope` (predicted delta-BW regressed on GT delta-BW) and `gain` is `edit_gain` (>1 means the edited render beats the model's own baseline render as an explanation of the edited ground truth). `rho` is `a_fit / a_theory` on the slab walls with **kappa-scaled** theory, kappa = 1.660756; the raw-Lorentzian comparison is in the column after it, for transparency only.

| arm | cond | val LSD (dB) | S1 slope | S1 gain | S2 slope | S2 gain | S3 slope | S3 gain | S4 slope | S4 gain | S5 slope | S5 gain | rho (slab) | rho vs raw | S2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **P3-2** | geom_alpha | 2.687 | 1.087 | 1.035 | 0.133 | 0.874 | 0.044 | 0.940 | 0.379 | 0.691 | n/a | n/a | n/a | n/a | n/a |
| **A** | geom_alpha_fourier/64 | 0.931 | 1.077 | 1.050 | 0.153 | 0.868 | 0.042 | 0.946 | 0.347 | 0.658 | 0.939 | 1.011 | 0.509 | 1.473 | FAIL |
| **B** | geom_alpha_fourier/64 | 0.998 | 1.016 | 1.052 | 1.147 | 1.084 | 0.741 | 2.679 | 0.697 | 0.997 | 1.123 | 1.171 | 1.039 | 1.735 | **PASS** |
| **C** | m_linear/60 | 1.013 | 0.997 | 1.053 | 0.959 | 1.087 | 0.720 | 2.734 | 0.789 | 1.001 | 1.010 | 1.176 | 0.947 | 1.612 | **PASS** |
| **D** | m_linear/60 | 1.464 | 0.969 | 1.048 | 1.038 | 1.082 | 0.721 | 2.258 | 0.664 | 0.998 | 1.158 | 1.146 | 1.031 | 1.712 | **PASS** |

Marker: arm **B** is the first arm in ladder order to clear the S2 thresholds.

## Attribution ladder

| step | comparison | S2 slope | S2 pearson | S2 gain | rho (slab) | val LSD |
|---|---|---|---|---|---|---|
| renderer + eval protocol | A vs P3-2 baseline | +0.020 | -0.034 | -0.006 | n/a | -1.756 |
| continuous alpha sampling | B - A | +0.994 | +0.372 | +0.216 | +0.530 | +0.068 |
| m = -ln(1-alpha) coordinate | C - B | -0.188 | -0.004 | +0.003 | -0.092 | +0.015 |
| multi-wall training | C - D | -0.079 | -0.003 | +0.005 | -0.084 | -0.450 |

Deltas are `to - from`; for val LSD lower is better, so a negative delta is an improvement. For every other column higher is better and rho is best at 1.0.

- **renderer + eval protocol** (A vs P3-2 baseline): arm A inherits the P3-2 dataset, in which one holdout was an EXTRAPOLATION (alpha above every trained value on that wall), so a null here cannot separate 'the renderer did not help' from 'that holdout was unfair'.
- **multi-wall training** (C - D): a near-zero delta here means multi-wall training was NOT necessary, which is a positive finding about data cost, not a failure.

## Honesty notes

1. **Arm A cannot cleanly isolate the renderer.** It inherits the P3-2 dataset, in which one holdout was an EXTRAPOLATION rather than an interpolation: the held-out alpha lay above every alpha that wall was trained on. A null result on arm A is therefore ambiguous between *the renderer did not help* and *that holdout was unfair*. Arms B/C/D use the P3-2b manifest, whose held-out slabs (west m in [0.62, 0.77], north m in [1.13, 1.28]) are strictly INTERIOR to the sampled range m in [0.02, 1.61], so only they test interpolation.
2. **The P3-2 baseline row is not a like-for-like comparison.** Its splits are matched to P3-2b's by experimental role, not by construction, and P3-2 ran no slope fit, so its rho column is empty. Read it as context for the size of the failure, not as a fifth arm.
3. **rho is reported against kappa-scaled theory.** The bandwidth estimator measures a calibrated -3 dB width, not the raw Lorentzian width; the gate's T5 fit gives `BW = 0.302 + 1.6608 * (gamma/pi)`. The intercept cancels in a paired delta, the slope does not, so `a_theory = kappa * c / (4 pi D)`. Scoring against the raw value would hand a perfect model rho ~ 0.60 and fail it.
4. **A high S1/S3 with a dead S2 is exactly the P3-2 result.** Do not read a strong in-distribution or seen-geometry column as progress on the chunk's question.

## Per-split detail

### S1 unseen geom, non-slab

| arm | n cfg | n cells | frac modes dropped | band LSD (dB) | E_BW (Hz) | slope | pearson | gain | GT effect (Hz) | model effect (Hz) |
|---|---|---|---|---|---|---|---|---|---|---|
| P3-2 | 110 | None | n/a | 3.553 | 1.205 | 1.087 | 0.899 | 1.035 | 3.348 | 3.658 |
| A | 100 | 835 | 0.458 | 2.943 | 1.314 | 1.077 | 0.884 | 1.050 | n/a | n/a |
| B | 100 | 827 | 0.463 | 3.007 | 1.253 | 1.016 | 0.894 | 1.052 | n/a | n/a |
| C | 100 | 802 | 0.479 | 2.979 | 1.244 | 0.997 | 0.877 | 1.053 | n/a | n/a |
| D | 100 | 813 | 0.472 | 3.029 | 1.153 | 0.969 | 0.906 | 1.048 | n/a | n/a |

### S2 unseen geom, HELD-OUT slab

| arm | n cfg | n cells | frac modes dropped | band LSD (dB) | E_BW (Hz) | slope | pearson | gain | GT effect (Hz) | model effect (Hz) |
|---|---|---|---|---|---|---|---|---|---|---|
| P3-2 | 20 | None | n/a | 3.877 | 4.982 | 0.133 | 0.533 | 0.874 | 5.159 | 0.775 |
| A | 20 | 178 | 0.422 | 3.250 | 4.964 | 0.153 | 0.499 | 0.868 | n/a | n/a |
| B | 20 | 171 | 0.445 | 2.652 | 1.825 | 1.147 | 0.871 | 1.084 | n/a | n/a |
| C | 20 | 168 | 0.455 | 2.669 | 1.560 | 0.959 | 0.868 | 1.087 | n/a | n/a |
| D | 20 | 167 | 0.458 | 2.709 | 1.707 | 1.038 | 0.871 | 1.082 | n/a | n/a |

### S3 seen geom, HELD-OUT slab

| arm | n cfg | n cells | frac modes dropped | band LSD (dB) | E_BW (Hz) | slope | pearson | gain | GT effect (Hz) | model effect (Hz) |
|---|---|---|---|---|---|---|---|---|---|---|
| P3-2 | 80 | None | n/a | 2.014 | 5.450 | 0.044 | 0.554 | 0.940 | 5.454 | 0.262 |
| A | 80 | 1178 | 0.000 | 1.842 | 5.476 | 0.042 | 0.571 | 0.946 | n/a | n/a |
| B | 80 | 1176 | 0.002 | 0.627 | 2.302 | 0.741 | 0.939 | 2.679 | n/a | n/a |
| C | 80 | 1176 | 0.002 | 0.617 | 2.219 | 0.720 | 0.944 | 2.734 | n/a | n/a |
| D | 80 | 1178 | 0.000 | 0.778 | 2.279 | 0.721 | 0.931 | 2.258 | n/a | n/a |

### S4 unseen geom, alpha=0.30

| arm | n cfg | n cells | frac modes dropped | band LSD (dB) | E_BW (Hz) | slope | pearson | gain | GT effect (Hz) | model effect (Hz) |
|---|---|---|---|---|---|---|---|---|---|---|
| P3-2 | 40 | None | n/a | 5.135 | 2.254 | 0.379 | 0.323 | 0.691 | 1.254 | 1.126 |
| A | 40 | 356 | 0.422 | 4.499 | 2.219 | 0.347 | 0.314 | 0.658 | n/a | n/a |
| B | 40 | 353 | 0.427 | 3.041 | 0.607 | 0.697 | 0.636 | 0.997 | n/a | n/a |
| C | 40 | 347 | 0.437 | 3.000 | 0.606 | 0.789 | 0.612 | 1.001 | n/a | n/a |
| D | 40 | 336 | 0.455 | 3.054 | 0.571 | 0.664 | 0.692 | 0.998 | n/a | n/a |

### S5 unseen geom, two walls

| arm | n cfg | n cells | frac modes dropped | band LSD (dB) | E_BW (Hz) | slope | pearson | gain | GT effect (Hz) | model effect (Hz) |
|---|---|---|---|---|---|---|---|---|---|---|
| P3-2 | — | — | — | — | — | — | — | — | — | — |
| A | 40 | 344 | 0.442 | 2.969 | 4.075 | 0.939 | 0.659 | 1.011 | n/a | n/a |
| B | 40 | 337 | 0.453 | 2.617 | 2.277 | 1.123 | 0.859 | 1.171 | n/a | n/a |
| C | 40 | 326 | 0.471 | 2.632 | 2.286 | 1.010 | 0.828 | 1.176 | n/a | n/a |
| D | 40 | 325 | 0.472 | 2.709 | 2.491 | 1.158 | 0.870 | 1.146 | n/a | n/a |

## Provenance

| arm | checkpoint | iter | cond | train configs | manifest sha |
|---|---|---|---|---|---|
| P3-2 | `outputs/p3_2/p3_2_main/ckpt_iter0060000.pt` | 60000 | geom_alpha | — | `—` |
| A | `outputs/p3_2/p3_2b_A_preset_fourier/ckpt_iter0060000.pt` | 60000/60000 | geom_alpha_fourier(64) | 440 | `ecf0ee6e620d` |
| B | `outputs/p3_2/p3_2b_B_cont_fourier/ckpt_iter0060000.pt` | 60000/60000 | geom_alpha_fourier(64) | 960 | `ecf0ee6e620d` |
| C | `outputs/p3_2/p3_2b_C_cont_mlinear/ckpt_iter0060000.pt` | 60000/60000 | m_linear(60) | 960 | `ecf0ee6e620d` |
| D | `outputs/p3_2/p3_2b_D_single_mlinear/ckpt_iter0060000.pt` | 60000/60000 | m_linear(60) | 480 | `ecf0ee6e620d` |

Held-out slabs (m = -ln(1-alpha)): west m in [0.62, 0.77] (brackets alpha=0.50), north m in [1.13, 1.28] (brackets alpha=0.70). Both interior to the sampled range m in [0.02, 1.61].

## Reproduction

```bash
export PYTHONPATH="$PWD"
python scripts/p3_2b_ablation.py          # this table (CPU, reads eval JSONs)
python scripts/make_p3_2b_figures.py      # the meeting pack, incl. figure E
```

