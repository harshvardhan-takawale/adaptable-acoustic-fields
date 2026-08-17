# P3-3-FAST Track A -- localization diagnostic

**Status: PARTIAL. The ground-truth half is computed and final; the prediction half is
staged but NOT run.** The three diagnostics all need model renders, tinycudann has no CPU
path, and the submit node (`nexusgroup01`) has no GPU (`torch.cuda.is_available() == False`,
no `/dev/nvidia*`). Rendering therefore requires a SLURM allocation, and this task was run
under an explicit instruction not to launch or cancel any SLURM job. Nothing was submitted.

Everything else is done: `scripts/p3_3fast_trackA_diag.py` implements all three
diagnostics, its prediction paths are validated against synthetic oracles (below), and
`scripts/slurm/p3_3fast_trackA_diag.sh` is ready. One command finishes it (~10 min):

```
sbatch scripts/slurm/p3_3fast_trackA_diag.sh \
  --checkpoint outputs/p3_3fast/p3_3fast_trackA/ckpt_iter0012000.pt
```

That writes `outputs/p3_3fast/trackA/DIAGNOSTIC.json` with every field named below. The
GT-only artifact produced here is `outputs/p3_3fast/trackA/DIAGNOSTIC_GT_ONLY.json`. No
`DIAGNOSTIC.json` was written, deliberately: a file of that name holding nulls would be
misread as a measured result.

Checkpoint intended for the run: **`ckpt_iter0012000.pt`** -- the newest at the time of
writing (the trainer is still running and writes every 2000 iters). Passing it explicitly
avoids racing the trainer's `ckpt_every` window; `find_checkpoint` would otherwise pick
whatever is newest at job start, possibly mid-write.

---

## The resolvability problem, which is a result on its own

This is measured, GT-only, and it conditions how every prediction number should be read.

| GT quantity, 10 test geometries, 0-300 Hz | mean | sd | n | frac |
|---|---|---|---|---|
| Edit magnitude: LSD(single-segment @0.70, baseline) | **1.607 dB** | 0.071 | 10 | 1.00 |
| Positional spread across the 4 edit positions (`gt_spread_db`) | **1.199 dB** | 0.071 | 10 | 1.00 |
| Positional fraction of the edit (spread / edit magnitude) | **0.746** | 0.025 | 10 | 1.00 |
| Mean pairwise LSD between the 4 GT fields | **1.456 dB** | 0.085 | 10 | 1.00 |
| Scale-free spread, std(&#124;H&#124;)/mean(&#124;H&#124;) | 0.048 | 0.020 | 10 | 1.00 |

Two things follow.

1. **The physics is strongly positional.** 75% of the entire effect of a single-segment
   absorber is *where* it is, not *how much* it absorbs. So the test is well-posed: there
   is a large positional signal for the model to either capture or miss.
2. **The model's fit error is ~3x the whole signal.** In-distribution val LSD is 4.54-4.68 dB
   (iters 8000-12000) against a total edit magnitude of 1.61 dB and a positional signal of
   1.20 dB. A model whose per-config error is 3x the between-config difference cannot be
   *expected* to show a spread ratio near 1, and if it does show a low ratio, "cannot
   localize" and "has not yet fit anything" are not separated by this measurement alone.
   The ratio is still worth having -- a ratio near 1 would be a clean positive even at this
   fit level, and a ratio near 0 is consistent with the FiLM-global-modulation limitation --
   but the negative form of the result must be reported as *confounded with the poor fit*,
   not as a clean architectural finding. Do not upgrade it to one.

The right disambiguator, if the ratio comes back low, is the same diagnostic on a
checkpoint that has actually converged, or on a P3-2b-style 4-wall model rerun through this
script; that comparison is not available today.

`frac` is 1.000 everywhere on the GT side: all 64 receivers x 601 bins of all 120 test
files are finite and above the 1e-8 log floor, so no GT number here is conditioned on a
subset. Prediction-side `frac` is computed per table by the script and is not yet known.

---

## 1. Segment discrimination -- **NOT RUN** (needs GPU)

Per geometry, the four `t_single_segment` configs (alpha = 0.70 on segment 3 of west /
east / south / north; nothing else differs). Denominators above; numerator pending.

Fields the run will fill: `pred_spread_db`, `spread_ratio_db` per geometry,
`spread_ratio_db_pooled` (sum of numerators / sum of denominators, so no geometry with a
small denominator can dominate), `pred_pairwise_lsd_db`, `pairwise_lsd_ratio_pooled`, and
`verdict`.

**The threshold is fixed in code, before any number exists** (`_discrimination_verdict`):
ratio < 0.15 -> `does_not_localize`; 0.15-0.6 -> `partial`; >= 0.6 -> `localizes`; and
`undetermined` if `gt_spread` < 0.5 dB. It is 1.199 dB, so the test resolves.

The estimator is validated at both ends of the answer space with synthetic predictions on
3 geometries:

| synthetic model | pooled spread ratio | pairwise LSD ratio | verdict |
|---|---|---|---|
| oracle (`pred = gt`) | 1.0000 | 1.0000 | `localizes` |
| position-blind (`pred` = 0.6 x mean over the 4 configs) | 0.0000 | 0.0000 | `does_not_localize` |

The position-blind case is exactly the FiLM failure mode being tested (identical field
whatever the position, plus a gross amplitude error), and the dB-based spread ratio reads
0.0 through that amplitude error -- which is why dB is the primary unit here rather than
linear magnitude.

## 2. `east_3` hold-out -- **NOT RUN** (needs GPU)

Groups are formed and the pairing is implemented; no GT-only form of this test exists,
because it compares prediction error at a held-out position against prediction error at
seen positions. Group structure (10 geometries):

| group | configs/geom | touches `east_3` |
|---|---|---|
| `single_seen` (west/south/north seg 3 @0.70) | 3 | no |
| `single_holdout` (east seg 3 @0.70) | 1 | **yes** |
| `uniform_seen` (west/south/north whole wall @0.70) | 3 | no |
| `uniform_holdout` (east whole wall @0.70) | 1 | **yes** |
| `window_seen` (`west_2` @0.95) | 1 | no |
| `window_holdout` (`east_3` @0.95) | 1 | **yes** |
| `patch3_holdout` (`east_2,3,4` @0.70) | 1 | **yes** |

Two caveats already fixed in the grouping. `t_uniform_wall` on east covers all four east
segments including `east_3`, so it is held-out too and is *not* a seen comparator.
`t_patch3_holdout` has no position-matched seen counterpart anywhere in the test split, so
it is reported on its own and contributes to no paired contrast.

Each config is scored as band-limited LSD against its own GT and as
`delta_vs_baseline_db` = that LSD minus the same geometry's `baseline` LSD, which removes
the per-geometry fit level. The headline is `paired_holdout_minus_seen`: per geometry, the
held-out group's delta minus its seen group's delta, aggregated over the 10 geometries,
with `distinguishable` = |mean| > 2 x sem. On a 10-geometry paired contrast that is a weak
test; a null there means "not resolved at n=10", not "identical".

## 3. Window configs (alpha = 0.95) -- **GT HALF DONE**, prediction pending

| slot | edited segment | GT in-band energy vs baseline | sd | n | frac |
|---|---|---|---|---|---|
| `t_window_seen` | `west_2` | **-5.273 dB** | 0.665 | 10 | 1.00 |
| `t_window_holdout` | `east_3` (held out) | **-5.281 dB** | 0.686 | 10 | 1.00 |

This confirms the expected -4 to -6 dB drop, and the seen and held-out positions produce
statistically identical drops in GT (-5.273 vs -5.281 dB, well inside the 0.21 dB sem) --
so for this observable the hold-out contrast is a fair one: any predicted asymmetry is the
model, not the physics.

The run will add `d_energy_pred_db`, `energy_recovered_frac`
(= `d_energy_pred_db` / `d_energy_gt_db`), and `lsd_db` per slot.

---

## Method notes

- Ground truth for Track A is stored already truncated to the 601 supervised bins
  (0-300 Hz, `band_hi_hz` attr = 300), while `FreqRenderer2D` emits 4097. Both sides go
  through the frozen `band_limit` and are then cut to `hi_idx` = 601; the truncation is
  identical on both. This differs from P3-2/P3-2b, where GT files carried the full 4097.
- `load_model` puts model **and renderer** in `eval()`. Load-bearing, not hygiene:
  `FreqRenderer2D` jitters ray azimuths while `self.training` (D49 C3), which would inject
  noise into precisely the across-config spread diagnostic 1 measures.
- `render_config_arm` (P3-2b) is used rather than P3-2's `render_config` because it
  dispatches the conditioning encoder on `cond_source`; the checkpoint is asserted to be
  `m_segment` / 144-d before anything is rendered.
- `_lsd_db` is imported from `aaf.eval.band_limited`, not reimplemented. `usable_mask`
  drops any cell not finite or at/below the 1e-8 log floor in *every* member of the
  comparison, and `frac` is reported next to every statistic it gates.
- Spreads use the sample std (`ddof=1`) across the 4 configs.
