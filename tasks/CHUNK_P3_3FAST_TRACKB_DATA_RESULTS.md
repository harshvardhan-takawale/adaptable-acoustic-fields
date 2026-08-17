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
