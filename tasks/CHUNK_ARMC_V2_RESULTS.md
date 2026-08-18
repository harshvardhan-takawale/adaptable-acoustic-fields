# Chunk results — Arm C demo pack v2: richer spatial patterns, difference maps, doorway motivator

**Date**: 2026-08-18 · **Branch**: `main` · **Commits**: `b2487d9` (figures), `46313ca` (v1)
**Status**: **ALL FOUR ITEMS COMPLETE.** The doorway reproduction pass finished and reproduces
FT-B's published level differences to within 0.004 dB — see §5.

No training, no new conditioning code, no new library code. Same checkpoint as v1:
`outputs/p3_2/p3_2b_C_cont_mlinear/ckpt_iter0060000.pt` (iter 60000, `m_linear`, 60-d).
**v1 figures, the frozen corpus and every manifest are unmodified** (`git status` verified).

Test suite: **449 passed, 0 failed** (job 7270808, `legacygpu06`).

---

## 0. Why this chunk exists

v1's headline Fig A plotted mode (1,0) at 28.9 Hz — the fundamental. It is a single vertical node
band, visually near-identical across all four scenarios, and the easiest mode in the band. It
oversold nothing but it also showed nothing. This chunk replaces it with modes spanning the
hierarchy, adds the difference maps that test whether the model reproduces the *edit* rather than
the *room*, and adds a ground-truth doorway figure to motivate the next phase.

A structural point worth noting: the v1 `.npz` dumps cached the **full `(4096, 601)` complex
spectra to 300 Hz**, so items 1–3 required **zero GPU time and zero re-simulation** — pure
CPU work on already-paid-for data. Only the doorway needed new compute.

---

## 1. Mode-accuracy screen — the abort rule did NOT trip

24 analytic modes ≤ 200 Hz on the median frozen test geometry **5.93 × 3.18 m**, baseline
scenario, spatial Pearson pred-vs-ISM on the 64×64 dense grid (the model trained on 8×8).

The pre-registered rule was: *report immediately rather than proceeding if Pearson falls below
0.85 for every mode above ~60 Hz.* **6 of the 21 modes above 60 Hz clear 0.85**, the highest at
**198.9 Hz**. The rule did not trip and the work proceeded.

| | dB fields | linear \|H\| |
|---|---|---|
| min | +0.655 | +0.578 |
| mean | **+0.822** | +0.681 |
| max | +0.991 | +0.885 |
| ≥ 0.70 | **23 / 24** | 9 / 24 |
| ≥ 0.85 | 9 / 24 | 1 / 24 |

Both columns are reported because they disagree sharply. The dB form compresses dynamic range
and rewards reproducing the nodal *pattern*; the linear form weights the loud regions and
declines steadily with frequency (+0.885 at 28.9 Hz → +0.590 at 198.9 Hz). Quoting only the dB
column would overstate high-frequency performance.

### Two findings the screen produced that were not anticipated

**(a) Accuracy is NOT monotone in frequency.** It falls to +0.655 at (3,2) 138.4 Hz and then
*recovers* to +0.891 at (4,3) 198.9 Hz. Any single "quality above X Hz" number misdescribes
this, and it is the substantive content of the screen figure.

**(b) 11 of 24 modes are NOT isolated**, which corrected my own planning estimate. A 2-D Sabine
calculation during planning gave a 1.98 Hz linewidth and suggested every candidate mode was
isolated. The Kuttruff linewidth actually used by the repo (`modal_damping_2d(...,
model="kuttruff")`) is **2.67–3.96 Hz**, so eleven modes have a neighbour inside their linewidth
— those fields are genuine superpositions and their mode labels are nominal. Flagged per mode in
the JSON and drawn hollow on the figure. This includes **(1,2) at 111.67 Hz**, the mid row of the
new Fig A, which overlaps (0,2) at 107.86 Hz; that caveat is printed on the figure itself.

Resolvability is not a concern here: the 64×64 grid resolves to `n_x ≤ 33, n_y ≤ 34` against a
maximum of (6,3) in band, so nothing is aliasing (unlike the 8×8 grid case in D61e).

### Secondary robustness check (not requested; near-free from cache)

| geometry | modes ≤ 200 Hz | min | mean | ≥0.70 | ≥0.85 | not isolated |
|---|---|---|---|---|---|---|
| small 3.44 × 3.14 | 15 | +0.678 | +0.821 | 13 | 6 | 4 |
| **median 5.93 × 3.18** | 24 | +0.655 | **+0.822** | 23 | 9 | 11 |
| large 5.56 × 4.90 | 35 | +0.610 | **+0.735** | 20 | 4 | 22 |

**The large room is materially worse** — mean +0.735, only 4 of 35 modes ≥ 0.85. Larger rooms
have denser modes, more overlap (22 of 35 not isolated) and worse high-frequency reconstruction.
Reported rather than omitted; the median-geometry headline is not representative of the largest
test room.

---

## 2. New Fig A — three modes spanning the hierarchy

Rows = modes, columns = the four v1 scenarios, each cell a stacked predicted / ISM pair, colour
scale **shared across all eight panels of a row**. Mode selection followed the spec's rule
(highest frequency still clearing 0.85, preferring both indices non-zero):

| row | mode | f Hz | bin | baseline R | why |
|---|---|---|---|---|---|
| low | (1,0) | 28.92 | 58 | +0.991 | kept for continuity with v1 |
| mid | (1,2) | 111.67 | 223 | +0.870 | best both-indices-non-zero mid candidate; 2-D pattern |
| high | (4,3) | 198.90 | 398 | +0.891 | highest mode clearing 0.85; both indices non-zero |

Per-cell spatial Pearson (dB), all twelve panels:

| mode | baseline | east 0.50 | north 0.70 ★ | two-wall |
|---|---|---|---|---|
| (1,0) 28.9 Hz | +0.991 | +0.984 | +0.991 | +0.984 |
| (1,2) 111.7 Hz | +0.870 | +0.874 | +0.885 | +0.780 |
| (4,3) 198.9 Hz | +0.891 | +0.902 | +0.919 | +0.872 |

★ `north@0.70` (m = 1.204) lies inside Arm C's held-out slab `north (1.13, 1.28)` and appears in
**0** training configs — unseen geometry *and* unseen material placement. It scores **highest in
both higher-mode rows**.

The (4,3) row is the visual payoff: rich two-dimensional nodal structure with obvious
predicted-vs-truth correspondence at 199 Hz, from a model that only ever saw an 8×8 receiver
grid. That is a much stronger demonstration than v1's flat fundamental.

Caveat carried on the figure, not buried: the mid row's mode is not isolated (see §1b), so that
field is a superposition and the label is nominal. Of the both-indices-non-zero modes clearing
0.85, only (4,3) is isolated.

---

## 3. Difference maps — the stricter test, and the most important result here

Raw-field correlation partly rewards getting the **room** right, because prediction and truth
share the same room. Subtracting the baseline cancels that and leaves only the response to the
**edit** — which is what an editable representation actually claims.

**Δ correlation is materially lower than raw-field correlation (0.78–0.99) at every mode.**

Plotted blocks — Δ Pearson, linear then dB:

| mode | edit | Δr linear | Δr dB |
|---|---|---|---|
| (1,1) 61.2 Hz | east curtain 0.50 | +0.654 | +0.671 |
| | north absorber 0.70 ★ | **+0.888** | +0.873 |
| | two-wall | +0.804 | +0.829 |
| (1,2) 111.7 Hz | east curtain 0.50 | +0.366 | **+0.144** |
| | north absorber 0.70 ★ | +0.846 | +0.720 |
| | two-wall | +0.821 | +0.501 |

Across all 24 modes:

| edit | linear mean | lowest 6 modes | highest 6 modes | dB mean |
|---|---|---|---|---|
| east curtain 0.50 | +0.586 | +0.871 | **+0.315** | +0.333 |
| north absorber 0.70 ★ | +0.614 | +0.822 | **+0.391** | +0.427 |
| two-wall | **+0.835** | +0.885 | **+0.876** | +0.528 |

### The decay is NOT universal — this corrected a claim I had put on the figure

My first draft of this figure was titled "Δ agreement decays as frequency rises, under BOTH
definitions." **That is wrong**, and the modal-frequency table showed it: both *single-wall*
edits fall steeply from the lowest six modes to the highest six (+0.871 → +0.315 and
+0.822 → +0.391), but the *two-wall* edit — a larger perturbation — holds essentially **flat** in
linear terms (+0.885 → +0.876). **Δ recovery tracks edit magnitude as well as frequency.** The
title and caption were corrected before the figure was published.

This is a favourable finding as well as an honest one: the model reproduces *large* edits well
across the whole band, and degrades mainly on *small* single-wall edits at high frequency.

### Δ convention — the ordering, and why

Maps and the leading number are **linear**, normalised by the baseline RMS |H| so the units are
"fraction of baseline amplitude". The **dB** value is printed beside every panel and in every
table.

Rationale for the ordering: dB differences diverge near pressure nulls, where the field is
negligible and the quantity numerically unstable. That artifact alone moves east curtain at
107.9 Hz from **+0.521 linear to +0.010 dB**. For a *difference* map the physical question is how
much the field amplitude moved, and the dB form overweights points with almost no sound.

**Recorded plainly, in the manifest and here: this ordering is the more favourable one.** It was
selected on the null-domination argument above, not on that fact, and both numbers appear on
every panel, in every table, and in `figures_v2.json`. v1's raw-field Pearson remains dB-based,
so Δ-linear and raw-dB are different quantities and are never compared directly.

---

## 4. Doorway physics — ground truth only, SIMULATION not prediction

No neural network appears in this figure. It motivates the next phase (doorway aperture as a
trainable edit axis) and is not a result — labelled as such in the title, per panel, and in the
caption.

Domain is **FT-B's frozen setup** so the panels sit on an already-validated configuration:
`L = 8.0, W = 4.0`, one-node divider at `x0 = 4.0`, α = 0.15 on every surface including the
divider, source (0.5, 0.5) in room A, `dx = 0.01`, `fs = 61440`. Three apertures — sealed
`a = 0`, doorway `a = 1.0`, fully open `a = 4.0` — built with the canonical recipe from
`aaf/data/aperture_configs.py:156`: sealed = slab with **no** `apertures` key; open = the
**empty list** (not a W-wide aperture, which leaves staircased tips).

Mode: **sub-room (1,1) at 60.71 Hz**, FT-B's own `PRIMARY_MODE`, whose even branch ends on
full-domain (2,1) at 60.63 Hz — sealed shows two independent sub-room fields, open shows one
field spanning the domain.

Neither FT-B nor the Track B corpus stored any spatial field (192 and 64 receivers
respectively), and `aaf.sim.fdtd_2d.simulate` has **no whole-grid output**, so this required a
new dense-receiver run: 8192 receivers (128 × 64), `n = 30720` (T = 0.5 s, df = 2.0 Hz).
Memory, not time, is the constraint — `simulate` allocates `ir_t`, `ir`, `H_complex` and
`H_deconv` all at `n_rx × n` and `H_deconv` has no skip flag. Measured peak **15.9 GB**,
~123 s per aperture.

**Room B is exactly zero in the sealed case** (`room_b_exactly_zero: true`) — not merely quiet.
A full-span slab disconnects the domain, so `H_B ≡ 0` and the level difference is −inf, matching
the published `DATASET_GATE.json`.

---

## 5. An estimator discrepancy found and attributed (in progress at time of writing)

The dense pass's inter-room level differences did **not** match FT-B's published values, and
this is worth recording because the first explanation was wrong.

| aperture | my power mean | my amplitude mean | FT-B published |
|---|---|---|---|
| a = 0.0 | −inf | −inf | −inf ✓ |
| a = 1.0 | −5.27 dB | −5.47 dB | **−7.15 dB** |
| a = 4.0 | −0.29 dB | −1.14 dB | **−1.45 dB** |

Cause identified by reading FT-B's own definition (`scripts/p3_3fast_ftb.py:232`): it uses
`20·log10(mean|H|)` — a mean of **amplitude** — whereas I had used `10·log10(mean|H|²)`, a mean
of **power**. Switching to FT-B's estimator moves a = 4.0 from −0.29 to −1.14 dB against a
published −1.45, i.e. the estimator explains most of that gap. A residual remains at a = 1.0,
attributable to the receiver set (8192 dense at 0.20 m margin vs FT-B's 16×8 at 0.30 m) and the
record length (T = 0.5 s vs 2.0 s).

Rather than assume that is benign, the job runs a **second pass that replays FT-B's exact
protocol** — its 16×8 grid at 0.3 m margin with `n = 122880` — which should reproduce
−7.15 / −1.45 exactly. Both estimators are now computed and stored for every run, and
`m_band` (mean |H| per receiver) is cached so this can be re-derived without re-simulating.
The sealed case already reproduces exactly (−inf).

### Result: reproduced exactly

Pass 2 completed (job 7270525, 27:27 elapsed, 15.9 GB peak). FT-B's published values are
reproduced on FT-B's exact protocol:

| aperture | FT-B protocol (n=122880, 16×8, 0.3 m) | FT-B published | delta |
|---|---|---|---|
| a = 0.0 (sealed) | **−inf** | −inf | exact |
| a = 1.0 | **−7.15 dB** | −7.15 dB | **+0.002 dB** |
| a = 4.0 (open) | **−1.45 dB** | −1.45 dB | **−0.004 dB** |

**The solver, geometry and aperture construction are validated.** The dense-grid offset is
entirely attributable to the estimator, the receiver set and the record length — none of them a
modelling error, and between them worth ~1.9 dB on this observable. `figF` prints both the
dense-grid value (what its panels show) and the reproduction beside it, so nobody compares the
dense number against the published table by mistake.

**Standing lesson**: an inter-room level difference is not comparable across chunks unless the
estimator, receiver set and record length all match.

---

## 6. Deliverables and links

All raw GitHub URLs verified HTTP 200.

### Figures

| file | px | link |
|---|---|---|
| Fig D — modal hierarchy | 3247 × 1569 | https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/armC_demo/v2/figD_mode_screen.png |
| Fig A2 — 3 modes × 4 scenarios | 3089 × 2696 | https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/armC_demo/v2/figA2_multimode_fields.png |
| Fig E — difference maps | 3077 × 3217 | https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/armC_demo/v2/figE_difference_maps.png |
| Fig F — doorway (GT only) | 3383 × 1875 | https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/armC_demo/v2/figF_doorway_physics.png |

### Data and manifest

- Manifest: https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/armC_demo/v2/FIGURE_MANIFEST.md
- Mode screen JSON (all 24 modes × 4 scenarios × 3 geometries): https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/armC_demo/v2/mode_screen.json
- Figure numbers + Δ-vs-frequency table: https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/armC_demo/v2/figures_v2.json

### Scripts

- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/armC_v2_mode_screen.py
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/armC_v2_figures.py
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/armC_v2_doorway.py
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/armC_v2_doorway_figure.py
- https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/slurm/armC_v2_doorway.sh

### v1 context (unchanged)

- v1 manifest: https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/armC_demo/FIGURE_MANIFEST.md
- v1 metrics: https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/armC_demo/metrics.json
- Project docs: [CONTEXT_FOR_MANAGER.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/CONTEXT_FOR_MANAGER.md) · [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md) · [OPEN_QUESTIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md)

---

## 7. What the manager should take from this

1. **The spatial claim now has a defensible frequency range**, not a single mode: 23/24 modes
   ≥ 0.70 to 200 Hz, mean +0.822, with a documented dip at 138 Hz and recovery at 199 Hz.
2. **The Δ result is the new headline and it is mixed.** Large (two-wall) edits are reproduced
   across the whole band (linear mean +0.835, flat with frequency). Small single-wall edits
   degrade badly above ~110 Hz (down to +0.315). Any claim about "editing" should be scoped to
   edit magnitude, not stated flatly.
3. **The largest test room is the weak case** (mean +0.735, 4/35 ≥ 0.85). If a single number is
   needed for a slide, it should not come from the median geometry alone.
4. **Two of my own errors were caught and corrected in-flight** and are recorded here rather
   than silently fixed: the linewidth/isolation estimate (§1b) and the Δ-decay claim (§3).
5. **The doorway solver is validated** (§5) — FT-B's published numbers reproduced to 0.004 dB,
   and the sealed room is exactly zero. The aperture axis is ready to train on.
6. **Open, carried from before**: Arm T token encoder on the ISM corpus, the DC-masked A2
   retrain, Track B2 eval, the A3 D1/D3 verdict, and the P3-2d ρ-definition question that gates
   publishing Δ*.
