# Arm C demo pack v2 — figure manifest

Richer spatial patterns, difference maps, and a ground-truth doorway motivator. **v1 figures are
untouched**; everything here is additive.

Every number below is read from `mode_screen.json`, `figures_v2.json` or
`doorway/doorway_meta.json`. No figure recomputes a metric, and nothing was re-simulated for
Figs A2/D/E — the cached v1 `.npz` dumps hold the full `(4096, 601)` complex spectra to 300 Hz,
so all three run on CPU from the same checkpoint as v1.

## Provenance

| item | value |
|---|---|
| checkpoint | `outputs/p3_2/p3_2b_C_cont_mlinear/ckpt_iter0060000.pt` (iter 60000), `m_linear` 60-d |
| geometry (Figs A2/D/E) | median frozen test geometry **5.93 × 3.18 m** |
| receivers | 64 × 64 = 4096, margin 0.15 m — training used **8 × 8** at 0.30 m |
| GT (Figs A2/D/E) | `aaf.sim.ism_2d`, fs 4096, n 8192, max_order 60, source (0.5, 0.5) |
| mode enumeration | `aaf.eval.modal_projection.enumerate_modes(L, W, f_max=200.0)`, excludes (0,0) |
| Pearson estimator | `aaf.eval.p3_2_eval._pearson` (the estimator v1 used) |
| linewidth | `damping_to_bandwidth_hz(modal_damping_2d(..., model="kuttruff"))` at α = 0.15 |
| cached fields | `outputs/armC_demo/fields/median_*.npz` (v1, unmodified) |
| doorway sim | `outputs/armC_demo/v2/doorway/`, FDTD, **ground truth only** |

## Files

| file | px | what |
|---|---|---|
| `figD_mode_screen.png` | 3247 × 1569 | accuracy across all 24 modes ≤ 200 Hz |
| `figA2_multimode_fields.png` | 3089 × 2696 | 3 modes × 4 scenarios, pred/ISM stacked |
| `figE_difference_maps.png` | 3077 × 3217 | Δ = edited − baseline, 2 mode blocks × 3 edits |
| `figF_doorway_physics.png` | see below | doorway motivator — **simulation, not prediction** |
| `mode_screen.json`, `figures_v2.json`, `figF_doorway_physics.json` | — | every plotted number |

---

## 1. Mode-accuracy screen (`figD`, `mode_screen.json`)

24 analytic modes ≤ 200 Hz, baseline scenario, spatial Pearson on the 64×64 grid.

**The abort rule did not trip.** It required reporting-instead-of-proceeding if *no* mode above
60 Hz reached 0.85; **6 of the 21 modes above 60 Hz clear it**, the highest at 198.9 Hz.

| | dB fields | linear \|H\| |
|---|---|---|
| min | +0.655 | +0.578 |
| mean | **+0.822** | +0.681 |
| max | +0.991 | +0.885 |
| ≥ 0.70 | **23 / 24** | 9 / 24 |
| ≥ 0.85 | 9 / 24 | 1 / 24 |

The two columns disagree sharply and both are reported. The dB form compresses dynamic range and
rewards reproducing the nodal *pattern*; the linear form weights the loud regions and declines
steadily with frequency (+0.885 at 28.9 Hz → +0.590 at 198.9 Hz). Quoting only the dB column
would overstate high-frequency performance.

Full per-mode table (dB / linear, baseline):

| mode | f Hz | R dB | R lin | linewidth Hz | nn gap Hz | isolated |
|---|---|---|---|---|---|---|
| (1,0) | 28.92 | +0.991 | +0.885 | 2.67 | 25.01 | yes |
| (0,1) | 53.93 | +0.960 | +0.818 | 3.27 | 3.91 | yes |
| (2,0) | 57.84 | +0.921 | +0.838 | 2.67 | 3.35 | yes |
| (1,1) | 61.20 | +0.885 | +0.834 | 3.96 | 3.35 | **no** |
| (2,1) | 79.08 | +0.797 | +0.765 | 3.96 | 7.68 | yes |
| (3,0) | 86.76 | +0.881 | +0.738 | 2.67 | 7.68 | yes |
| (3,1) | 102.16 | +0.771 | +0.712 | 3.96 | 5.70 | yes |
| (0,2) | 107.86 | +0.856 | +0.724 | 3.27 | 3.81 | yes |
| (1,2) | 111.67 | +0.870 | +0.732 | 3.96 | 3.81 | **no** |
| (4,0) | 115.68 | +0.756 | +0.680 | 2.67 | 4.01 | yes |
| (2,2) | 122.39 | +0.736 | +0.675 | 3.96 | 5.24 | yes |
| (4,1) | 127.64 | +0.712 | +0.677 | 3.96 | 5.24 | yes |
| (3,2) | 138.43 | **+0.655** | +0.654 | 3.96 | 6.18 | yes |
| (5,0) | 144.60 | +0.800 | +0.634 | 2.67 | 6.18 | yes |
| (5,1) | 154.33 | +0.792 | +0.619 | 3.96 | 3.83 | **no** |
| (4,2) | 158.17 | +0.730 | +0.621 | 3.96 | 3.63 | **no** |
| (0,3) | 161.79 | +0.852 | +0.619 | 3.27 | 2.56 | **no** |
| (1,3) | 164.36 | +0.817 | +0.611 | 3.96 | 2.56 | **no** |
| (2,3) | 171.82 | +0.797 | +0.591 | 3.96 | 1.70 | **no** |
| (6,0) | 173.52 | +0.790 | +0.587 | 2.67 | 1.70 | **no** |
| (5,2) | 180.40 | +0.823 | +0.582 | 3.96 | 1.31 | **no** |
| (6,1) | 181.71 | +0.835 | +0.578 | 3.96 | 1.31 | **no** |
| (3,3) | 183.59 | +0.812 | +0.580 | 3.96 | 1.88 | **no** |
| (4,3) | 198.90 | **+0.891** | +0.590 | 3.96 | 15.31 | yes |

**Accuracy is not monotone in frequency.** It falls to +0.655 at (3,2) 138 Hz and recovers to
+0.891 at (4,3) 199 Hz. Reporting a single "quality above X Hz" number would misdescribe this.

**11 of 24 modes are NOT isolated** — a neighbour lies inside their Kuttruff linewidth, so the
field at those frequencies is a genuine superposition and the mode label is nominal. Marked
hollow on `figD` and flagged per mode in the JSON. The 64×64 grid resolves to `n_x ≤ 33,
n_y ≤ 34` against a maximum of (6,3) in band, so nothing here is aliasing.

**Secondary robustness check** (not requested; nearly free from cache, and the large room is
notably worse — reported rather than omitted):

| geometry | modes ≤ 200 Hz | min | mean | ≥0.70 | ≥0.85 | not isolated |
|---|---|---|---|---|---|---|
| small 3.44 × 3.14 | 15 | +0.678 | +0.821 | 13 | 6 | 4 |
| **median 5.93 × 3.18** | 24 | +0.655 | **+0.822** | 23 | 9 | 11 |
| large 5.56 × 4.90 | 35 | +0.610 | **+0.735** | 20 | 4 | 22 |

---

## 2. New Fig A (`figA2_multimode_fields.png`)

Rows = modes, columns = the four v1 scenarios, each cell a stacked predicted / ISM pair, colour
scale **shared across all eight panels of a row**.

Mode selection followed the spec's rule — highest frequency still clearing 0.85, preferring both
indices non-zero:

| row | mode | f Hz | bin | vmin/vmax dB | why |
|---|---|---|---|---|---|
| low | (1,0) | 28.92 | 58 | −22.36 / +17.64 | kept for continuity with v1 |
| mid | (1,2) | 111.67 | 223 | −25.15 / +14.85 | best both-indices-non-zero mid candidate |
| high | (4,3) | 198.90 | 398 | −29.15 / +10.85 | highest mode clearing 0.85; both indices non-zero |

Per-cell spatial Pearson (dB):

| mode | baseline | east 0.50 | north 0.70 ★ | two-wall |
|---|---|---|---|---|
| (1,0) 28.9 Hz | +0.991 | +0.984 | +0.991 | +0.984 |
| (1,2) 111.7 Hz | +0.870 | +0.874 | +0.885 | +0.780 |
| (4,3) 198.9 Hz | +0.891 | +0.902 | +0.919 | +0.872 |

★ `north@0.70` (m = 1.204) lies inside Arm C's held-out slab `north (1.13, 1.28)` and appears in
**0** training configs — unseen geometry *and* unseen material placement. It scores highest in
both higher-mode rows.

**Stated on the figure, not buried:** the mid row's (1,2) at 111.67 Hz overlaps (0,2) at
107.86 Hz within its 3.96 Hz linewidth. That field is a superposition of the two and the label
is nominal. Only (4,3) among the both-indices-non-zero modes clearing 0.85 is isolated.

---

## 3. Difference maps (`figE_difference_maps.png`) — the stricter test

Raw-field correlation partly rewards getting the **room** right, because prediction and truth
share the same room. Subtracting the baseline cancels that and leaves only the response to the
**edit**, which is what an editable representation actually claims.

**Δ correlation is materially lower than raw-field correlation (0.78–0.99) at every mode.**

Plotted blocks — Δ Pearson, linear then dB:

| mode | edit | Δr linear | Δr dB | map scale (± × baseline RMS) |
|---|---|---|---|---|
| (1,1) 61.2 Hz | east 0.50 | +0.654 | +0.671 | 0.200 |
| | north 0.70 ★ | **+0.888** | +0.873 | 0.585 |
| | two-wall | +0.804 | +0.829 | 0.629 |
| (1,2) 111.7 Hz | east 0.50 | +0.366 | **+0.144** | 0.093 |
| | north 0.70 ★ | +0.846 | +0.720 | 0.444 |
| | two-wall | +0.821 | +0.501 | 0.485 |

Baseline RMS |H| used for normalisation: 2.902 at 61.2 Hz, 1.805 at 111.7 Hz.

Across all 24 modes:

| edit | linear mean | lowest 6 modes | highest 6 modes | dB mean |
|---|---|---|---|---|
| east curtain 0.50 | +0.586 | +0.871 | **+0.315** | +0.333 |
| north absorber 0.70 ★ | +0.614 | +0.822 | **+0.391** | +0.427 |
| two-wall | **+0.835** | +0.885 | **+0.876** | +0.528 |

**The decay is not universal, and saying "Δ decays with frequency" without qualification would
be wrong.** Both *single-wall* edits fall steeply from the lowest six modes to the highest six
(+0.871 → +0.315 and +0.822 → +0.391), while the *two-wall* edit — a larger perturbation — holds
essentially flat in linear terms (+0.885 → +0.876). Δ recovery tracks **edit magnitude** as well
as frequency.

### Δ convention, and why this ordering

Maps and the leading number are **linear**, normalised by the baseline RMS so the units are
"fraction of baseline amplitude". The **dB** value is printed beside every panel and tabulated
above.

The ordering is chosen on one argument: dB differences diverge near pressure nulls, where the
field is negligible and the quantity is numerically unstable. That artifact alone moves east
curtain at 107.9 Hz from **+0.521 linear to +0.010 dB**. For a *difference* map the physical
question is how much the field amplitude moved, and the dB form overweights points where there
is almost no sound.

Recorded plainly: **this ordering is the more favourable one.** It was selected on the
null-domination argument above, not on that, and both numbers appear on every panel, in every
table, and in `figures_v2.json`. v1's raw-field Pearson remains dB-based, so the two are not
interchangeable — Δ-linear and raw-dB are different quantities and are never compared directly.

---

## 4. Doorway physics (`figF_doorway_physics.png`) — SIMULATION, NOT MODEL OUTPUT

No neural network appears in this figure. It motivates the next phase (doorway aperture as a
trainable edit axis) and is not a result. Labelled as such in the title, per panel, and in the
caption.

Domain is **FT-B's frozen setup** so the panels sit on an already-validated configuration:
`L = 8.0, W = 4.0`, one-node divider at `x0 = 4.0`, α = 0.15 on every surface including the
divider, source (0.5, 0.5) in room A, `dx = 0.01`, `fs = 61440`.

Three apertures, built with the canonical recipe (`aaf/data/aperture_configs.py:156`): sealed =
slab with **no** `apertures` key; mid = slab + one centred aperture; fully open = the **empty
list** (not a W-wide aperture, which would leave staircased tips).

Mode: **sub-room (1,1) at 60.71 Hz**, FT-B's own `PRIMARY_MODE`, whose even branch ends on
full-domain (2,1) at 60.63 Hz — so sealed shows two independent sub-room fields and open shows
one field spanning the domain. Its nearest sub-room neighbour is 17.7 Hz away.

Dense pass: 8192 receivers (128 × 64), `n = 30720` (T = 0.5 s, df = 2.0 Hz). Memory, not time,
is the constraint — `simulate` allocates `ir_t`, `ir`, `H_complex` and `H_deconv` all at
`n_rx × n` and `H_deconv` has no skip flag; measured peak 15.9 GB, ~123 s per aperture.

**Room B is exactly zero in the sealed case** (`room_b_exactly_zero: true`), not merely quiet —
a full-span slab disconnects the domain, so `H_B ≡ 0` and the level difference is −inf, matching
the published `DATASET_GATE.json`. Displayed clipped to the floor and labelled as exact.

*(Level-difference reproduction against FT-B's published values — see the section appended
below.)*

---

## Reproduce

```bash
sbatch scripts/slurm/armC_v2_doorway.sh    # CPU, ~30 min: dense fields + FT-B reproduction
python scripts/armC_v2_mode_screen.py      # CPU, ~20 s  -> mode_screen.json
python scripts/armC_v2_figures.py          # CPU, ~60 s  -> figD, figA2, figE
python scripts/armC_v2_doorway_figure.py   # CPU, ~10 s  -> figF
```

`armC_v2_mode_screen.py` exits non-zero if no mode above 60 Hz reaches 0.85.
