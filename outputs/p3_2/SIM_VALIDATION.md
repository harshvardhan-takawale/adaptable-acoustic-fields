# P3-2 simulator validation — BLOCKING GATE: **PASS**

pyroomacoustics 0.9.0, fs=4096, N=8192, max_order=60, source=[0.5, 0.5], 8x8 receivers (margin 0.3 m).

Measured on ISM ground truth via the 64-receiver modal projection (`aaf.eval.modal_projection`), so each measurement is attributable to a single (n_x, n_y) mode rather than to whatever dominates one receiver's spectrum.

## Provenance asserts (G0)

| check | result | detail |
|---|---|---|
| wall convention (image lattice) | PASS | west image damping 0.5000 = sqrt(1-a) 0.5000; other walls undamped |
| mirror equivariance | PASS | rel err 4.17e-08 (control 1.98e-01) |
| max_order 60 converged | PASS | BW 2.404 -> 2.404 Hz (0.0%), level 0.050 dB |

## Selectivity (room 4.5x4.0, cond(Phi)=1.539)

Bandwidth deltas vs that room's own baseline, per mode family:

| edit | dBW x-axial | dBW y-axial | dBW tangential | dLevel x | dLevel y |
|---|---:|---:|---:|---:|---:|
| west -> M1 (a=0.05) | -1.160 | -0.023 | -0.721 | +1.68 | +0.17 |
| west -> M2 (a=0.5) | +5.355 | +0.110 | +3.312 | -4.08 | -0.74 |
| west -> M3 (a=0.7) | +10.183 | +0.207 | +6.262 | -5.95 | -1.34 |
| east -> M1 (a=0.05) | -1.164 | -0.062 | -0.738 | +1.67 | +0.06 |
| east -> M2 (a=0.5) | +5.542 | +0.309 | +3.553 | -4.02 | -0.23 |
| east -> M3 (a=0.7) | +10.982 | +0.636 | +7.113 | -5.88 | -0.40 |
| south -> M1 (a=0.05) | -0.039 | -1.296 | -0.910 | +0.18 | +1.70 |
| south -> M2 (a=0.5) | +0.180 | +6.014 | +4.183 | -0.80 | -4.04 |
| south -> M3 (a=0.7) | +0.341 | +11.464 | +7.943 | -1.45 | -5.81 |
| north -> M1 (a=0.05) | -0.064 | -1.301 | -0.923 | +0.09 | +1.69 |
| north -> M2 (a=0.5) | +0.317 | +6.199 | +4.407 | -0.37 | -4.06 |
| north -> M3 (a=0.7) | +0.653 | +12.257 | +8.796 | -0.66 | -5.93 |

## Decision

| test | result | n |
|---|---|---|
| T1_direction | PASS | 8/8 |
| T2_selectivity | PASS | 8/8 |
| T4_orthogonal_flip | PASS | 8/8 |
| T3_bidirectional | PASS | 8/8 |
| T3_monotonic | PASS | 8/8 |
| T5_theory_fit | PASS | 1/1 |

**Bandwidth selectivity** = 29.1 (95% CI [20.0, 39.3]), threshold 5.0.

**Which damping law?** ISM-ray fit R^2 = 0.9981 (BW = 0.302 + 1.661*gamma/pi); Kuttruff R^2 = 0.9820. dAIC = 73 favouring **ism_ray**.

> **Scoping (D48).** ISM uses angle-independent reflection and so has no grazing-incidence absorption: a purely axial mode is damped only by the wall pair it bounces between. Real locally-reacting walls follow Kuttruff and would give ~2:1 selectivity with no invariant family. The claim this chunk can support is therefore *the model learns the simulator's per-wall law*.

Sources: `outputs/p3_2/gate/gate.json`, `outputs/p3_2/gate/fig_A_gate.png`.