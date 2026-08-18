# P3-3-FAST Track B2 — the A2/A3 TOKEN encoder on the aperture axis

**Date**: 2026-08-18 · **Status**: training RUNNING, job **7268013** (RTX A5000, scavenger,
30K iters, ~24 h wall) · Output `outputs/p3_3fast/p3_3fast_trackB2/` · Decision **D60**.

## Why this chunk exists

Track B (`aperture`, 55-d) **did not learn the aperture law at all** — it is not a transfer
failure. Its predicted inter-room level difference fits `sqrt(a)` at r^2 **0.172** / slope
**2.46** against ground truth's **0.948** / **7.61**, and it is *equally* wrong on both sides
of the hold-out:

| | seen-line residual | held-out residual | ratio |
|---|---|---|---|
| Track B prediction | 2.463 dB | 2.298 dB | **0.933** |
| ground truth (D59a) | 0.851 dB | 0.679 dB | 0.798 |

Mean held-out residual **0.019 dB**. A model that learned the law and failed to extrapolate
would be *asymmetric*; this is flat. The representational diagnosis: a **global** 55-d vector
carrying a **scalar** aperture is being asked to induce a **spatial** barrier at `x0` with a
gap of width `a`, and FiLM has no mechanism to localize the gap. Track A2/A3 fixed the
identical pathology on the absorption axis by tokenizing the boundary — held-out window
recovery went **-0.069 -> +1.010** at convergence.

## What was built

Everything except the conditioning arm is byte-identical to Track B. **No new simulation**:
the same 472-config FDTD corpus (`data/track_p3_3fast_B/`, schema `p3_3fast.trackB/1`), the
same 380 training configs, the same `manifest_sha
50d01b696a27afe30e309c8e5dd04d6ff271ab16f788f7fc16849b7d08d58c2d`, the same band (0-300 Hz),
optimizer (Adam 2e-4), schedule (30K, ckpt every 2K) and `n_pts_per_ray=64`. The two yamls
differ in exactly three lines: `run_id`, `cond_source`, `cond_dim`.

### `aperture_token`, cond_dim **464**

`[48 geometry (L, W, x0) | 16 divider tokens x 26]`, reduced by the shared encoder to
**112 = 48 + 64** before the FiLM generator.

- Tokens use **A2's featurization verbatim** (`D_TOK = 26`: position Fourier k=0..3 (16),
  normal raw (2), extent raw (1), m_hat identity + Fourier k=0..2 (7)), so
  `INR2D_AutoDecoder.segment_encoder` is **reused, not duplicated**.
- `cx = x0/L` for every token, `cy = (i+0.5)/16`, face normal `(+1, 0)`, extent `1/16`.
- The geometry prefix widens **32 -> 48** to carry `x0` (D57a), and those 48 dims are asserted
  **byte-identical** to Track B's first 48 — the arms differ *only* in the doorway encoding.
- Outer walls are deliberately not tokenized: they are baseline in every config of this corpus.

### Continuity in `a`: fractional `m_hat` (NOT `extent`)

`f_i` = fraction of segment `i`'s y-extent covered by the centred doorway;
`m_hat_i = m_hat(baseline) + f_i * (1 - m_hat(baseline))`, with open = `m_hat 1.0`
(m = 3.0, alpha = 0.9502 — Track A's open-window value). `extent` stays `1/16`.

Rationale: an integer open-**count** steps by `W/16 ~ 0.25 m`, **coarser than the 0.2 m
hold-out band**, which would quantize the very axis the track tests (D52: continuity in the
linearizing coordinate is the operative variable). `extent` was rejected because segments must
**partition** the divider (P3-3 Part-A), and shrinking a solid segment breaks the partition.

Verified: `sum_i f_i * (W/16) == a` exactly; `sum_i m_hat_i` strictly increasing and jump-free
over 201 samples of [0.1, 2.5] with **201 distinct values**; `a = 0.26` separates from
`a = 0.25` despite opening the same segment count.

**FT-B's `sqrt(a)` coordinate is no longer supplied.** The encoder must recover it from token
geometry — that is the claim under test.

### Delta pooling, and the sealed room

A3's `sum_i [phi(t_i) - phi(t_i^baseline)]`, not A2's mean (which diluted a single edit by 1/16
and dropped discrimination to 0.222). The divider's alpha **is** `ALPHA_BASELINE`, so a sealed
divider has all 16 tokens at baseline and the aggregate is **exactly zero** — correct, and
matching the topology: a sealed divider is the un-edited room, the doorway is the edit. Sealed
configs stay out of training (`config_kinds: ["open", "aperture"]`, D57b) and are reported
separately.

## Files

| Path | Change |
|---|---|
| `aaf/models/conditioning_2d.py` | `COND_SOURCE_BTOK`, `divider_open_fraction`, `divider_token_geometry`, `divider_m_hat`, `aperture_token_features_2d`; registered in `cond_dim_for` + `build_cond_vector_2d` |
| `aaf/models/inr_2d.py` | `aperture_token` added to the **independent** whitelist; encoder branch generalized to a per-arm (expected width, reduced width) so `_tok_geom` is 48 for B2 and 32 for A2/A3 |
| `configs/sweep_2d_mat/P3_3FAST_trackB2.yaml` | new |
| `tests/test_aperture_tokens.py` | 13 tests, CPU-only / tcnn-free |

Trainer and eval unchanged — `configs_from_rows` already dispatches on
`p3_3fast.trackB/1`, and `x0`/`a` already reach `build_cond_vector_2d` via `getattr`.

## Test results

`tests/test_aperture_tokens.py`: **13 passed**. Full suite on a GPU node: **437 passed, 0
failed** (pytest exit 0), so A2/A3's `m_token` / `m_token_delta` paths are unregressed.
GPU construction smoke: B2 `_tok_geom=48`, `cond_dim=112`, `pool=delta`; A3 `_tok_geom=32`,
`cond_dim=96`, `pool=delta`; A2 `pool=mean`, `cond_dim=96`; forward finite at
`a in {0, 0.5, 1.0, 2.5, 4.0}`.

## Honest caveats

1. **The sealed aggregate is ~1e-7, not bitwise zero, in the shipped model.** `inr_2d` builds
   `_tok_baseline_m` with python `math` in float64 while `conditioning_2d` builds the same
   features with `torch.sin` in float32 — a <= 1 ULP disagreement. Six orders of magnitude
   below any open-aperture aggregate, and asserted at both precisions in the test rather than
   papered over. **Left unfixed on purpose**: Track A3 (job 7267594) shares that code and was
   mid-flight.
2. **`--gres` names the GPU type** (`gpu:rtxa5000:1`) via the reused
   `scripts/slurm/p3_3fast_train.sh` header; no new launcher was written.
3. Nothing is committed (task constraint). `git status` shows the four changed/added files
   plus the doc updates.

## What to do when 7268013 finishes

Run `scripts/p3_3fast_trackB_eval.py` (unchanged, D59) against
`outputs/p3_3fast/p3_3fast_trackB2/` and compare to Track B on the same four observables. The
single decision number is the **seen-line residual ratio**: Track B's 0.933 with a slope of
2.46 means "never learned it". A B2 slope near GT's 7.61 with r^2 near 0.948 would mean the
tokenization supplied what the scalar could not; a slope near GT with a residual ratio well
above 1 would be the *good* failure — learned but did not transfer — and would move the
question to the hold-out band rather than to the encoder.
