# CHUNK RESULTS — FIG 5, the zero-shot edit demo (P3-3-FAST, 2026-08-18)

**Deliverables**
- `scripts/p3_3fast_demo_figure.py` (re-runnable)
- `scripts/slurm/p3_3fast_demo_figure.sh` (`sbatch` it; needs a GPU — tinycudann has no CPU path)
- `outputs/p3_3fast/meeting_assets/fig5_topological_edits.png` (>= 1920 x 1080)
- `outputs/p3_3fast/meeting_assets/fig5_topological_edits.json` (sidecar, every plotted number)
- FIG 5 section appended to `outputs/p3_3fast/meeting_assets/FIGURE_MANIFEST.md`
- `outputs/p3_3fast/floored_lsd_30k.json` (corpus LSD at the demo checkpoint, all 120 test configs)
- `data/track_p3_3fast_A_demo/` (2 new FDTD configs; the training corpus is untouched)

Decisions in `DECISIONS.md` **D61**. New question in `OPEN_QUESTIONS.md` **Q18**.

## What the figure shows

Checkpoint `outputs/p3_3fast/p3_3fast_trackA2/ckpt_iter0030000.pt` (`m_token`, cond_dim 448,
400 training configs). One held-out test geometry — **geom_id 9, L = 5.25 m, W = 3.60 m** — and
five panels, each a **single zero-shot forward pass**: no optimisation, no per-room fitting, no
measurement of the edited room. Test geometries share no `(L, W)` with the 20 training
geometries, so the geometry is zero-shot too.

| panel | edit | GT dE (dB) | pred dE (dB) | recovered | LSD raw / modal / floored (dB) | map r |
|---|---|---:|---:|---:|---|---:|
| (a) | baseline, all 16 at alpha 0.15 | 0.000 (ref) | 0.000 | n/a | 6.68 / 7.28 / 1.61 | +0.24 |
| (b) | `west_2` alpha 0.50 — TRAINED position | -1.294 | -1.536 | **+1.187** | 6.51 / 7.15 / 1.48 | +0.44 |
| (c) | `east_3` alpha 0.50 — HELD-OUT position | -1.321 | -1.182 | **+0.895** | 6.58 / 7.18 / 1.71 | +0.43 |
| (d) | `east_3` alpha 0.95 (window open) — HELD-OUT | -4.835 | -4.880 | **+1.009** | 6.32 / 7.00 / 1.61 | +0.60 |
| (e) | window closed, back to alpha 0.15 | 0.000 (ref) | 0.000 | n/a | 6.68 / 7.28 / 1.61 | +0.24 |

`dE` = in-band (0-300 Hz) total energy relative to the baseline room, via
`scripts.p3_3fast_trackA_diag.band_energy_db` — the same estimator behind the published
aggregate. Recovery = pred dE / GT dE. `map r` = spatial Pearson correlation between the GT and
predicted dB field maps at the plotted mode.

**Panel (e) is a separate forward pass and is bitwise identical to (a)** — the standing
`renderer.eval()` determinism check (D49 C3) passes.

**The plotted mode is the same for all five panels**: (1,0) at 32.5 Hz, the argmax of
|GT delta vs baseline| over the baseline modal peaks the 8 x 8 receiver grid can resolve.
Panel (b)'s unrestricted argmax is 294.0 Hz at -1.82 dB, a **0.01 dB** tie with 32.5 Hz's
-1.81 dB; both are recorded.

## The result

**Positive.** The model recovers 0.895-1.187 of the ground-truth in-band energy change for
three different edits, including two at a segment position (`east_3`) that is at the baseline
in **all 400** training configs. Panel (d)'s +1.009 reproduces the Track A2 aggregate
(+1.010 held-out, +1.106 seen) on a single geometry. Closing the window restores the baseline
field exactly.

**Negative 1 — the spatial mode shape is wrong.** At 32.5 Hz the GT map is a clean (1,0)
standing wave (node at x = L/2); the prediction stays source-centred. Spatial correlation
**r = +0.24 to +0.60**. Consistent with it, an LSD over the 27 baseline modal-peak bins
(7.00-7.28 dB) is *worse* than raw LSD (6.32-6.68 dB). The zero-shot claim this figure supports
is about the **energy** response to an edit, not about spatial structure at a single mode. The
negative is printed on the figure rather than cropped out.

**Negative 2 — the circulated "floored LSD" measures the near-DC term.** See **Q18**. Each
config's peak cell is the bin-0 (0,0) compliance term, ~46 dB above the strongest room mode, so
a -40 dB peak-relative floor keeps only bins 0-12 (0-6 Hz).

## Corrections to circulated numbers

- **Corpus LSD at iter 30000 over ALL 120 test configs: 5.460 dB raw / 0.703 dB floored.**
  The circulated 4.956 / 0.613 came from `scripts/p3_3fast_floored_lsd.py`'s default
  `--limit 24`, i.e. test geometries 0-1 only, at iter **28000**. The slurm wrapper now passes
  `--limit 120`.
- The FIG-5 geometry is **harder** than the corpus mean (6.32-6.68 dB raw vs 5.460), so the
  median-rule geometry pick did not flatter the model.

## Choices that had to be made

1. **Ground truth for the alpha = 0.50 panels did not exist** — the Track A test split only
   enumerates 0.70 and 0.95. It is simulated here by calling the corpus builder itself
   (`scripts.build_p3_3fast_trackA.build_one`; same solver, dx, fs, receiver snapping) into
   **`data/track_p3_3fast_A_demo/`**, never into `data/track_p3_3fast_A/`. ~20 s per config,
   idempotent. `rows_sha256` and every dataset gate are untouched.
2. **Geometry pick**: lower median of the ten test geometries by `t_window_holdout` energy
   recovery in `DIAGNOSTIC_30K.json` (rank 5 of 10, recovery 1.0095). Full ranking is in the
   sidecar and the manifest entry.
3. **Mode candidate set** referenced to the strongest LOCAL MAXIMUM, not to `s.max()` (the DC
   term makes the peak-relative set empty), and restricted to modes the 8 x 8 grid resolves
   (`n_x <= 3`, `n_y <= 4`) because the bottom rows are mode-shape maps.
4. **Map colour scaling**: each row referenced to its own panel-(a) peak (removes the constant
   +1.67 dB model-vs-GT level offset, keeps every per-panel difference) and then scaled to its
   own range. Stated on the figure.

## Re-running

```bash
sbatch scripts/slurm/p3_3fast_demo_figure.sh
```

`scripts/p3_3fast_figures.py` **rewrites** `FIGURE_MANIFEST.md` from scratch; re-run this
script afterwards to restore the FIG 5 section. The append is idempotent and replaces its own
section in place.
