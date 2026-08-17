# P3-3-FAST Track 2b — doorway-aperture dataset, conditioning arm, and dataset gate

**Status (2026-08-17)**: dataset **BUILDING** on scavenger (60-way array, job `7265361`);
everything else — manifest, config module, conditioning arm, trainer branch, training config,
unit tests, gate script — is **DONE and verified**. No GPU training was launched.

## What the axis is

A shoebox `L x W` split by one interior slab at `x = x0`, alpha = 0.15 on the slab and on all
four outer walls. The edit is the **doorway width `a`**, centred on the divider at `y = W/2`.

* **Conditioning coordinate: `sqrt(a)`** — FT-B's measured linearizer, pooled r^2 = **0.9870**
  for the inter-room level difference (raw `a` 0.905, `a^2` 0.704).
  `outputs/p3_3fast/trackB/aperture_sweep.json`.
* **`a = 0` is a topological discontinuity, not the small-aperture limit.** A sealed one-node
  divider disconnects room B *exactly*: `H_B == 0`, level difference `-inf`. Sealed rooms are
  kept in the dataset, flagged `sealed: true`, given their own `kind`, and **excluded from
  training** (`config_kinds: ["open", "aperture"]`) and from every continuous-coordinate fit.
* **The hold-out is an exact BAND**: no training aperture in `[0.9, 1.1]`; 18 test apertures
  (0.95 / 1.00 / 1.05 x 6 domains) sit inside it.

## Dataset

| | value |
|---|---|
| train | 20 domains x 20 configs = **400** (1 sealed, 1 fully open, 18 drawn on [0.1, 2.5]) |
| test | 6 frozen domains x 12 = **72** (sealed, fully open, 3 in-band, 7 out-of-band) |
| total | **472** rooms, all filenames unique |
| geometry box | L in [7.0, 9.0], W in [3.5, 4.5], x0 in [0.4L, 0.6L] (test domains strictly interior) |
| solver | FDTD `fdtd_2d_slf_kw`, **dx = 0.01, fs = 61440, n = 122880** (lambda 0.55827, T = 2.000 s, df = 0.5 Hz) |
| source / receivers | (0.5, 0.5); 8x8 grid, 0.3 m margin, spanning the FULL domain |
| stored | `ism/H_complex` [64, 601] complex64, 0-300 Hz (legacy key, FDTD data — D56) |
| cost | ~200-250 s/room on an uncontended core; ~22x Track A |

`dx = 0.01` is forced (FT-1b A0c: the aperture observable moves 10.4x the estimator floor
between dx 0.02 and 0.01) and **fs must scale with 1/dx or CFL fails** — a fixed fs = 12288 at
dx = 0.01 raises a `ValueError` from the solver.

**Receivers are nudged off the divider per DOMAIN, not per config**: any receiver whose x-node
lands on or beside the divider column is pushed two nodes clear. Without it `simulate` raises
"snaps onto a solid node" for the sealed and narrow-aperture configs but succeeds for wide
ones. Because the nudge depends only on `(L, W, x0)`, all 20 configs of a domain share one
receiver array.

## Conditioning arm: `aperture`, 55 dims

```
[ 0:16] L      [16:32] W      [32:48] x0      [48:55] aperture
```

Geometry: `(L-8)/1`, `(W-4)/0.5`, `(x0/L - 0.5)/0.1`, each 8 octaves of sin/cos.
Aperture: identity `sqrt(a)/sqrt(4.0)` + 3 octaves (pi, 2pi, 4pi).
Registered in `cond_dim_for` **and** in `INR2D_AutoDecoder`'s independent whitelist.
`build_cond_vector_2d` gained keyword-only `x0` / `a` (every other arm ignores them); the
trainer passes them with `getattr(c, "x0", None)`.

## Dataset gate

`python scripts/gate_p3_3fast_trackB_dataset.py` -> `outputs/p3_3fast/trackB/DATASET_GATE.json`.
Thresholds fixed before the numbers were seen.

State at 15:26 on 2026-08-17 (**392 of 472 rooms built, 0 malformed**; the array was still
running, so the JSON records item (i) as FAIL on completeness alone):

| item | test | threshold | result |
|---|---|---|---|
| (i) | every manifest config built, opens, shape (64, 601), attrs match, `a_realized` within 2 cells | 0 missing / 0 malformed | **FAIL — 392/472 built, 80 missing, 0 malformed** (build in flight; re-run) |
| (ii) | filenames unique | 0 duplicates | **PASS** (472 rows, 472 unique) |
| (iii) | training apertures in [0.9, 1.1] / test apertures in band | 0 / >= 3 | **PASS** (0 / 18, at a = 0.95, 1.00, 1.05) |
| (iv) | sealed room-B / room-A amplitude ratio; fully-open \|level difference\| | <= 1e-6 / <= 3.0 dB | **PASS** (22 sealed rooms: ratio **exactly 0.0**; 20 open rooms: max \|LD\| **2.03 dB**) |

Item (iv) is the only one that tests the simulator rather than the bookkeeping: if
`extra_walls` were silently dropped, (i)-(iii) would still pass.

**Physics sanity beyond the gate.** On the one fully-built domain (L=7.06, W=4.11, x0=3.71),
the inter-room level difference is monotone in `a` and a straight line in `sqrt(a)`:
-11.53 dB at a = 0.227 rising to -2.58 dB at a = 2.45 and -1.27 dB fully open, with a
within-domain fit **slope 7.62, intercept -14.63, r^2 = 0.966** over the 18 finite points —
i.e. FT-B's sqrt(a) law reproduces on the actual training geometry, and the sealed room sits
at `-inf`, off that line entirely.

**Build throughput (measured, worth knowing before the next wave-track dataset).** The FDTD
loop is memory-bandwidth bound, so per-room cost depends on how many array tasks the scheduler
packs onto one node: 20 of 59 tasks landed on `cbcb21`/`brigid`/`quics` and finished all 8
rooms in under 30 min (~225 s/room), while the 39 tasks packed 5-10 to a node on the
`legacygpu*` machines ran ~25-30 min/room — an **8x spread**. The array's 3 h wall limit was
sized on the fast figure and could not be raised after submission (`scontrol update` -> access
denied); the script now asks for 8 h. Any task that does hit the limit costs nothing but time:
`.done` sentinels make a resubmit skip everything already built.

## Files

* `aaf/data/aperture_configs.py` — `ApertureConfig`, sampler, frozen test domains, manifest.
* `scripts/build_p3_3fast_trackB.py` / `..._manifest.py` / `scripts/slurm/build_p3_3fastB_array.sh`
* `configs/sweeps_2d_mat/p3_3fast_trackB_manifest.json` (schema `p3_3fast.trackB/1`)
* `configs/sweep_2d_mat/P3_3FAST_trackB.yaml` (cond_source `aperture`, cond_dim 55, 30k iters)
* `aaf/models/conditioning_2d.py`, `aaf/models/inr_2d.py`, `aaf/train/multi_room_2d_mat.py`
* `tests/test_aperture_configs.py` (9 tests, CPU-only, all pass)
* `scripts/gate_p3_3fast_trackB_dataset.py`

## How to finish the build

```bash
python scripts/build_p3_3fast_trackB.py --plan          # n_pending + array range
sbatch --array=0-58%60 scripts/slurm/build_p3_3fastB_array.sh   # idempotent; .done sentinels
python scripts/gate_p3_3fast_trackB_dataset.py          # rewrite DATASET_GATE.json
```

The worklist is the FULL stable list and is never filtered on `.done` — filtering would shrink
it as the build progresses and race the array index against the config mapping, which silently
left 79 of 479 P3-2c configs unbuilt.

---

# Track 2b, part 2: the evaluation harness (2026-08-17)

**Status: harness built, tested, and dry-run on ground truth. Training (job 7266375) is still
running, so no scored EVAL.json exists yet — the eval is one `sbatch` away.**

```bash
sbatch scripts/slurm/p3_3fast_trackB_eval.sh \
    --checkpoint outputs/p3_3fast/p3_3fast_trackB/ckpt_iter0030000.pt
```

Pin the checkpoint explicitly while training is live: without `--checkpoint` the driver takes
the newest `ckpt_iter*.pt` at job start and races the trainer's `ckpt_every` window. The job
writes `outputs/p3_3fast/trackB/EVAL.json` + `EVAL.md` and then the demo figure. Measured cost
on an RTX A5000: **~6.6 s per config**, so ~8 min for the 72 test configs.

## What it measures

Four observables, predicted vs ground truth, per test config; **every one reported held-out
band (`a in [0.9, 1.1]`, n = 18) vs seen apertures (n = 48) separately**, with the 6 sealed
configs excluded from all of it and reported alone.

1. `level_difference` — `20 log10(<|H|>_roomB / <|H|>_roomA)` per ISO third-octave band inside
   20-300 Hz and pooled. Room membership by receiver x against the domain's `x0`.
2. `mode_split` — each sub-room projected onto its OWN analytic basis (room B shifted into its
   local frame); peak POSITION (`migration_*_hz`) and two-peak SPLITTING (`split_hz`).
3. `decay` — band-limited Schroeder EDC per sub-room, single- vs two-segment slope over FT-B's
   frozen -5..-25 dB window, so double-slope verdicts stay comparable to FT-B's.
4. `lsd` — band-limited, whole-domain and per sub-room, always beside `gt_dynamic_range_db`.

`frac_modes_dropped` / `frac_usable` sit next to every number they gate. Design rationale and
the caveats that bound each observable are D59.

## The GT-only result (a real result, and the harness's own validation)

`python scripts/p3_3fast_trackB_eval.py --gt-only` runs on CPU in **18 s** and writes
`outputs/p3_3fast/trackB/EVAL_GT_ONLY.{json,md}`. The `sqrt(a)` fit of the inter-room level
difference over the 66 non-sealed test configs:

| fit | n | sqrt(a) span | slope (dB per sqrt m) | intercept dB | r | r^2 |
|---|---|---|---|---|---|---|
| GT pooled, 6 domains | 66 | 1.710 | 7.611 | -15.289 | 0.9737 | 0.9481 |
| GT held-out band | 18 | **0.050** | 8.717 | -16.079 | 0.3249 | 0.1055 |
| GT seen apertures | 48 | 1.710 | 7.648 | -15.449 | 0.9782 | 0.9569 |

Per-domain: r^2 **0.9693 +/- 0.0093**, slope **7.636 +/- 0.701**.

**The held-out row's r^2 is not a failure — it is the x-range.** Three aperture values inside
the band means `sqrt(a)` spans 0.050 against 1.710, so a within-band regression is degenerate
by construction. The test that answers the question is the seen-line residual: fit `sqrt(a)`
on the seen apertures only, then score the held-out points against that line.

| side | seen-line slope | RMS resid seen | RMS resid held-out | ratio |
|---|---|---|---|---|
| GROUND TRUTH | 7.648 | 0.851 dB | **0.679 dB** | **0.798** |

The held-out band sits ON the seen line in the ground truth (ratio < 1), which is what makes
the same test meaningful when the model's numbers land in the same table. FT-B's single-domain
slope was 6.808 (r^2 0.9870) and CONTEXT's one-training-domain check gave 7.62 / 0.966 — three
independent measurements of the same law agreeing.

Other GT-side facts the model will be scored against: T60 ~**3.32 s** in both sub-rooms, **no**
double-slope decay anywhere (0/66 in room B — expected, the two sub-rooms have identical
absorption so there is no slow/fast pair), in-band dynamic range **~75 dB**, and 6/6 sealed
configs with room-B energy **exactly zero**.

## Caveats, all enforced in code rather than only stated

* **The modal split is resolution-limited and mostly reads 0.** FT-B resolved sub-linewidth
  splitting via an even/odd decomposition that only exists when `x0 = L/2`; Track 2b varies
  `x0`, so that route is closed and a two-peak search needs a separation of order the linewidth
  (Kuttruff median **3.83 Hz**). The usable modal observable is peak migration. **16.3 of 26.5
  modes per config** are flagged `degenerate` and excluded, leaving ~8 usable;
  `frac_modes_dropped` runs **0.46-0.50**.
* **Absolute LSD is NOT comparable to P3-2b's ~1.0 dB.** ~75 dB of dynamic range here against
  ~22 dB on the ISM corpus; use `lsd_over_dynamic_range` for anything cross-chunk.
* **The 8x8 grid leaves 3-5 columns per sub-room**, so the projection basis blows up above
  ~160 Hz; an f_max ladder backs off to the first rung with `cond(Phi) <= 5` (140 Hz clears all
  twelve sub-rooms) and records the rung it used.
* **`a = 0` never enters a fit or an aggregate** (D57b), and every dB is guarded against log(0).

## Files

* `scripts/p3_3fast_trackB_eval.py` — the harness (`--gt-only` runs without a GPU).
* `scripts/p3_3fast_trackB_demo_fig.py` — 1920x1200 predicted-vs-GT spectra at
  a = 0 (sealed, labelled topological) / 0.3 / **1.0 (HELD OUT)** / 2.0, both sub-rooms.
* `scripts/slurm/p3_3fast_trackB_eval.sh` — runs both; forwards `"$@"` to each.
* `tests/test_p3_3fast_trackB_eval.py` — 20 CPU-only tests: the held-out/seen split, the
  room-A/room-B assignment (including on-divider receivers, which belong to neither), the
  third-octave banding, the sealed guards, and the GT/renderer rfft bin alignment.
