# Chunk results — P4-1: shape editing, Stage 0 and Stage 1

**BOTH GATES PASS, and the DC hypothesis is PARTIAL.** Gate 0: token-only geometry reaches
spatial Pearson **0.982** and band LSD **1.722 dB**, *better than* Arm C's 0.951 / 2.268 on the
identical protocol — the global `(L, W)` features are not merely replaceable, they were costing
something. Gate 1: the architecture fits a non-convex L-room with NLOS spatial Pearson **+0.974**
and a LOS−NLOS gap of **+0.020** (held-out receivers: +0.890 / +0.074). The DC diagnostic came
back at **23.7%** of the training objective, not the majority the spec assumed, and the masked
retrain is a **trade-off**, not a win: spatial Pearson +0.593 → +0.726 but modal-peak LSD
5.73 → 6.21 dB.

**Two caveats that change what the passes mean**, both detailed below: Gate 1's headline is a
**fit** metric (87.5% of receivers were supervised; 1.00 dB trained vs 3.25 dB held out), and
the learned σ field did **not** discover the occluder in any physically meaningful sense. And one
counter-result: Arm T-geo's **edit-response slope is worse than Arm C's in every split**.

**Date**: 2026-09-11 · **Branch**: `main` · Tests: **471 passed, 0 failed**
**Numbering**: the spec said to log the coordinate convention as D63, but D63 was taken by the
2026-09-10 repo-sync entry. This chunk is **D64–D65**.

---

## 0. Three places the spec was wrong about this codebase

All three verified by reading and executing the code, not inferred. None blocked the work; two
changed what got built and one changed how Stage 1 is interpreted.

**1. There is no source→receiver ray.** The spec motivates Stage 1 with "the renderer integrates
along source→receiver rays; in an L-room that path crosses solid wall." `FreqRenderer2D` fans 64
rays *outward from the receiver* to the room AABB and feeds the source in as a network input
(`freq_2d.py:166-172`); no ray is traced from `tx` and `|rx − tx|` is never computed. The real
consequence is sharper: transmittance accumulates **only along the receiver→point leg**, so a
wall between the *source* and a sample point has **no structural representation at all**. This
reframed the σ diagnostic (§3).

**2. `wall_segments` cannot describe an L-room, and fails silently.** Executed on the notched
room, it raises nothing and reports `tiles_exactly: True` with extents summing to 5.0 m for an
east wall that physically exists for 2.5 m, writing half the absorption onto inert solid nodes —
and that `True` is exactly what the builder stores into the HDF5 attrs. The L-room is cut with
the `mask` escape hatch instead (D65).

**3. Positions were never bbox-normalized; the TOKENS were.** `segment_geometry` returns
unit-square coordinates, hard-codes `extent = 0.25`, and ignores its own `L, W` arguments — so
two rooms of different size produced byte-identical token positions. A token arm could not have
carried geometry regardless of training. That is the gap D64 closes.

---

## 1. GATE 0 — token-only geometry (PASS, and better than the baseline)

Arm **T-geo**: no global geometry features at all. The room is 4 boundary tokens
`[cx, cy, nx, ny, extent, m̂]` in **absolute metres / WORLD_SCALE**, through the shared encoder,
masked-mean pooled, into FiLM at the same injection points. Identical to Arm C in corpus
(960 configs, manifest sha `ecf0ee6e…`), recipe, optimizer and 60000-iteration budget; the only
differences are `cond_source` and the `world_scale`/`base_resolution` pair that must move
together (§1.2).

### Demo protocol — 3 frozen test geometries × 4 scenarios

| metric | **Arm T-geo** | Arm C | gate |
|---|---|---|---|
| spatial Pearson (mean) | **0.982** | 0.951 | ≥ 0.85 ✅ |
| spatial Pearson (worst) | **0.969** | 0.920 | — |
| band LSD (mean) | **1.722 dB** | 2.268 dB | ≤ 2.768 ✅ |
| in-distribution val LSD | **0.9914** | 1.0132 | — |

T-geo wins **all 12 scenarios on both gated metrics** (mean ΔR +0.026, mean ΔLSD −0.51 dB).

**A trajectory warning worth recording.** T-geo started *behind* — +0.18 dB at iter 8000 — drew
level around 22000, and finished ahead. An early-checkpoint comparison would have concluded the
opposite, and the chunk would have reported that tokens cost accuracy.

### The frozen P3-2b splits (thresholds unchanged, `thr a8479c5e1dcc`)

**S2 verdict: PASS** — slope 0.871 ≥ 0.80, pearson 0.900 ≥ 0.80, gain 1.285 > 1.00,
|ρ−1| 0.054 ≤ 0.25, no blockers.

| split | LSD T-geo | LSD Arm C | slope T | slope C | pearson T | pearson C |
|---|---|---|---|---|---|---|
| S1 unseen geom, 1-wall | **2.315** | 2.979 | 0.784 | **0.997** | **0.911** | 0.877 |
| S2 unseen geom, slab ★ | **2.049** | 2.669 | 0.871 | **0.959** | **0.900** | 0.868 |
| S3 seen geom, slab | 0.657 | **0.617** | 0.720 | 0.720 | 0.937 | **0.944** |
| S4 unseen α = 0.30 | **2.365** | 3.000 | **0.464** | 0.789 | 0.556 | **0.612** |
| S5 unseen geom, 2-wall | **1.947** | 2.632 | 0.913 | **1.010** | **0.891** | 0.828 |

### The counter-result, stated plainly

T-geo reconstructs the **field** better — lower LSD in every unseen-geometry split — but responds
to an **edit** less linearly: `edit_bw_slope` is lower than Arm C in every split, while
`edit_gain` is higher everywhere (it overshoots edit magnitude). **S4 is the sharpest drop,
0.464 vs 0.789**, well under 0.80, though S4 is not a gated split.

So the honest summary is not "tokens are better". It is: **tokens reconstruct better and edit
less linearly.** If Stage 2 needs edit linearity as well as reconstruction, the token `m̂` channel
is the first place to look — it is the only part of the token whose representation changed
without its Fourier treatment changing.

`mode_shape_invariance` (pred): T-geo 0.9898, Arm C 0.9920 — a different quantity from the
pointwise spatial Pearson above, and not merged with it.

---

## 2. The world-scale convention (D64)

`WORLD_SCALE = 10.0` in `aaf/walls.py`, applied to tokens, source, receivers and sample points
alike through `normalize_position(x, world_scale)`. `world_scale=None` reproduces the legacy
`(x+1)/2` **exactly**, so the switch is opt-in per checkpoint.

**`base_resolution` 16 → 80 moves with it, and that coupling is load-bearing.** At world_scale 10
a 6 m room spans 0.6 hash units instead of 3.0 — 5× fewer cells across the room. Without the
compensation (16 × 1.5⁴ = 80 restores metres-per-cell), a Gate 0 failure could not have been
attributed to the tokens rather than to lost resolution, which is the entire question Stage 0
exists to answer.

**Verified safe for every existing checkpoint**: Arm C re-renders **bit-identically** — 12 of 12
cached scenarios, worst |delta| **0.000e+00**.

The map lives in `walls` and not in the model because `aaf.models.inr_2d` imports tinycudann and
needs a GPU; the spec's required "source and receiver use identical scaling" test has to run in
the ordinary CPU pass.

---

## 3. GATE 1 — the L-room (PASS, but the mechanism is not occlusion)

One L-room, 6.0 × 5.0 with a 3.0 × 2.5 notch, six edges, reflex corner at (3.0, 2.5), α = 0.15
everywhere. FDTD dx = 0.02, fs = 30720, **1903 receivers (1426 LOS / 477 NLOS, 25.1%)**.
Per-scene fit, 30000 iterations, six polygon-edge tokens through the *same* module as Stage 0.

| | LOS | NLOS | gap | band LSD |
|---|---|---|---|---|
| all receivers (**the gate**) | +0.994 | **+0.974** | **+0.020** | 1.24 / 1.41 dB |
| held-out only (238) | +0.964 | **+0.890** | +0.074 | 3.27 / 3.18 dB |

**GATE 1: PASS** — NLOS +0.974 ≥ 0.70, gap +0.020 ≤ 0.15. It also passes on the held-out split.

**I predicted this would likely fail** on the Track B precedent (D60a), where a solid interior
structure the renderer could not see produced a representation failure. That was wrong: Track B's
failure was *conditioned generalization*, not representational capacity.

### Caveat 1 — the gate metric is a FIT number

1665 of 1903 receivers were supervised. Band LSD is **1.00 dB on trained receivers vs 3.25 dB
held out (3.2×)**. The spec scopes Stage 1 to exactly this question ("only whether the
representation can fit one such room when given dense data"), so gating on it is correct — but
+0.974 would be badly misread as generalization. Both splits ship in `GATE1.json`.

The `+1.000` Pearsons are **not** an affine artifact: Pearson is affine-invariant, so a correct
shape with a wrong level scores 1.0, but at the (1,0) mode the offset is −0.037 dB, slope 1.0017
and mean |error| **0.058 dB**. The residual level error is now recorded per mode so a high R can
never be read as a perfect fit without it.

### Caveat 2 — σ did not discover the occluder

Probed along a ray from an NLOS receiver through the notch — the one direction in which σ is
*structurally capable* of representing the wall:

| | value |
|---|---|
| mean σ inside the notch | 0.7292 |
| mean σ in air | 0.6783 |
| ratio | **1.075** |
| Mann-Whitney (solid > air) | p = **2.5e-05** |
| Cohen's d | **0.736** (medium) |
| transmittance across the 3.68 m notch crossing | **0.068** |
| …same path length in air | **0.082** |

σ is **statistically** elevated but **physically negligible** — the field is near-uniform (1.5×
total spread) and the notch crossing attenuates only 17% more than the same distance of air. σ is
doing generic distance attenuation; the room's shape is carried by the `signal` field.

**So Gate 1 shows the architecture can FIT a non-convex room, not that it found a geometric
occlusion mechanism** — and a localized mechanism is precisely what Track B lacked when it
failed. Stage 2's generalization question is therefore still open on its merits.

---

## 4. DC-masked retrain — PARTIAL, and a trade-off

**Diagnostic** (A2 checkpoint, iter 30000). A prediction was recorded *before* running it,
because it makes the result interpretable either way, and it held exactly:

| term | DC share | per-bin enrichment | |
|---|---|---|---|
| `L_spec_real` | 39.0% | 9.0× | linear → exposed |
| `L_spec_imag` | 24.0% | 4.4× | linear → exposed |
| `L_amp` | 1.1% | 0.2× | log10 → immune |
| `L_phase` | 0.1% | 0.0× | cosine → immune |
| **weighted total** | **23.7%** | | |

**The spec's framing is too strong.** Bins 0–39 command about a *quarter* of the gradient, not a
majority, because two of the four terms are structurally immune. The spec's claim that
"spec_real, spec_imag, and amp are linear-domain losses" is wrong about `L_amp`, which is log10 —
and that is exactly why the effect is 24% rather than the ~87% the in-band power share suggests.

**Retrain, scored fairly** — both checkpoints through one code path on identical bins at the only
iteration they share (12000, where the masked run early-stopped):

| metric | unmasked | DC-masked | delta |
|---|---|---|---|
| modal-band LSD | 5.2352 | 5.2125 | −0.023 (negligible) |
| **modal-peak LSD** | 5.7339 | 6.2070 | **+0.473 worse** |
| **spatial Pearson** | +0.5932 | +0.7256 | **+0.132 better** |
| full-band LSD | 4.9545 | 7.2686 | +2.314 *(worse by construction)* |

**A trade-off, not a win**: masking substantially improves spatial structure and costs modal
magnitude accuracy. Every FDTD stage in this chunk used the **unmasked** loss; Stage 0 and
Stage 1 are ISM and FDTD respectively but neither used a low cut (`loss_band_lo_hz = 0`).

---

## 5. Bugs found and fixed during the chunk

Three were mine, three were latent in the codebase.

| # | what | how it surfaced |
|---|---|---|
| 1 | Notch mask built in the **wrong corner** | my own analytic cross-check fired on run 1 — 37500 nodes = exactly twice the block |
| 2 | `VAL_RX` hard-coded to `range(3, 64, 8)` | would have held out 8 of 1903 receivers, all in the first two grid columns — and early stopping reads that metric |
| 3 | Stage 1 OOM: 32 rows per backward | trainer's own comment says 8 is the proven footprint |
| 4 | **Band-masked run scored on the full band** | masked run looked 1.65 dB *worse* while every term it was trained on was *better*; would have early-stopped it and reported "masking hurts" |
| 5 | **`load_model` ignored `base_resolution`/`world_scale`** | Gate 1 crashed with a bare tensor size mismatch naming no config key; **would have taken Gate 0 down the same way** |
| 6 | `demo_edit_2d.py` had the identical gap | found by grep after #5 |

#4 and #5 are the ones worth the manager's attention: both would have produced a *confidently
wrong published number* rather than an error.

---

## 6. Deliverables

`outputs/p4_1/{stage0,stage1,dc_fix}/`, `PHASE4_SUMMARY.md`, this file, plus D64/D65 in
`DECISIONS.md` and Q19–Q20 in `OPEN_QUESTIONS.md`.

Figures: `figG_lroom_fields.png` (3062×1204, LOS/NLOS field map), `figH_sigma_profile.png`
(3208×1242, the σ probe).

Scripts: `build_p4_1_lroom.py`, `gate_p4_1_lroom.py`, `p4_1_gate0.py`, `p4_1_stage1_eval.py`,
`p4_1_stage1_figures.py`, `p4_1_dc_diagnostic.py`, `p4_1_dc_compare.py`,
`p4_1_armc_regression.py`.

---

## 7. What the manager should take from this

1. **The last shoebox-specific component is gone, at no cost** — in fact at a gain. Stage 2 is
   unblocked on the conditioning side.
2. **Gate 1's pass is about capacity, not mechanism.** The architecture can fit a non-convex
   room; it has *not* been shown to represent occlusion geometrically. Stage 2 (multi-shape
   generalization) is the test that would separate those, and the Track B precedent argues it is
   still genuinely at risk.
3. **Tokens trade edit linearity for reconstruction.** S4's slope drops 0.789 → 0.464. If the
   phase goal needs both, this needs attention before the Z-corridor.
4. **The DC hypothesis is a quarter-effect, and masking is a trade-off.** Worth keeping as a
   tunable, not worth adopting by default on this evidence.
5. **Do not use `wall_segments` or `patch` on any non-rectangular room** until they raise on
   solid nodes (D65e, a 3-line change deferred here).
