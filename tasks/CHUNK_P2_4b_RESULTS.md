# Chunk P2-4b — Bounding the convergence confound — RESULTS ✅ COMPLETE

**Verdict: coverage effect CONFIRMED at matched convergence — but the raw P2-4 curve overstated
its magnitude ~3×.** At equal in-distribution convergence (~4.3 dB), 250 training rooms beat 45 on
magnitude-band LSD, phase, and RIR — and because both sides are at matched convergence, blur is
equalized on both, so these deltas isolate coverage — so
densification genuinely helps and the P2-4 curve's **direction** is trustworthy. But a
decomposition shows **~68% of the raw P2-4 magnitude-correlation climb was the convergence/blur
confound, only ~32% genuine coverage**. Cite the matched-convergence deltas below, **not** the raw
P2-4 curve's slope/magnitude. Headline deliverable: `outputs/coverage_curve/CONFOUND_CHECK.md`.

## Why this chunk

P2-4 confounded two variables: as room count grew, the fixed iteration budget meant per-room
exposure fell, so in-distribution val LSD degraded monotonically (45→2.17, 90→3.31, 150→3.84,
250→4.30 dB). Since P2-3 showed an **under-converged decoder inflates zero-shot magnitude
correlation** (a blurry "average-room" prediction correlates better with an arbitrary unseen room
than a sharp, specifically-wrong one), the P2-4 climb could be partly a blur artifact. This chunk
isolates coverage from convergence with matched-convergence endpoints on the **same frozen test
set** (15 interior rooms), full metric suite.

## Design (and the key finding that shaped it)

**250 rooms is capacity-plateaued at the frozen recipe.** density_250's in-dist val LSD flattened
at ~4.3 dB (constant LR, slope ~0.008 dB/1k over the last 13K iters, bouncing 4.28–4.38) — reaching
45's 2.17 dB would need ~250K+ more iters and it is asymptoting well above that. **At fixed latent_dim
16 / hashgrid 18/16/1.38 capacity, 250 rooms cannot converge as tightly as 45** — itself a capacity
signal. So convergence was held constant at **~4.3 dB** by *under-training 45 down to 250's plateau*
(user-approved minimal-compute path, DECISIONS D42), not the infeasible reverse. The 250 point reuses
density_250@85K (4.30 dB); the 45 endpoint is a fresh frozen-recipe retrain (`density_45_conv`) with
dense retained checkpoints, evaluated at the checkpoint nearest 4.30 dB (11K @ **4.333 dB** — a 0.03 dB
match). A blur sweep (45 at 4.55/4.33/3.80/3.52 dB) traces the confound directly.

## The result — full-suite comparison (known-geometry zero-shot, mean over 15 frozen rooms)

| point | in-dist LSD | mag full | mag modal | held LSD full | LSD 0–250 | phase (mw) | RIR | modal recall | modal MAE (Hz) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **45 · converged** (P3 @60K) | 2.17 | 0.273 | 0.409 | 7.70 | 9.28 | 0.125 | 0.130 | 0.104 | 0.97 |
| **45 · matched** @4.33 | 4.33 | 0.401 | 0.698 | 6.25 | 5.30 | 0.218 | 0.287 | 0.125 | 0.92 |
| **250 · plateau** @4.30 | 4.30 | **0.461** | **0.811** | **6.04** | **4.71** | **0.348** | **0.421** | 0.089 | 1.00 |
| 45 · @4.55 | 4.55 | 0.414 | 0.721 | 6.35 | 5.25 | 0.225 | 0.302 | 0.143 | 0.86 |
| 45 · @3.80 | 3.80 | 0.357 | 0.617 | 6.24 | 5.77 | 0.188 | 0.233 | 0.127 | 0.86 |
| 45 · @3.52 | 3.52 | 0.331 | 0.576 | 6.37 | 6.07 | 0.174 | 0.208 | 0.121 | 0.89 |

*Held-out LSD lower = better; correlations + recall higher = better; modal MAE lower = better. Modal band = 0–250 Hz (sub-Schroeder ≈217 Hz).*

### 1. Matched-convergence verdict — CONFIRMED
At ~4.3 dB, **250 beats 45** on: mag full **+0.060**, mag modal **+0.113**, held LSD full **+0.21 dB**,
modal-band LSD **+0.59 dB**, phase **+0.131**, RIR **+0.133**. It does **not** win on modal *peak
placement* (recall −0.036, MAE −0.08 Hz favour 45) — but those are weak metrics (recall ~0.1 for both;
MAE differences ~0.1 Hz). The largest matched deltas are **phase and RIR** (broadband + time-domain
structure) and the **modal-band LSD** — and because blur is equalized on both sides at matched
convergence, these deltas isolate coverage. Coverage genuinely helps.

### 2. The confound was large — decompose the raw P2-4 gap
Split the raw P2-4 mag-corr gap (45@2.17 → 250@4.30) at the matched-convergence midpoint:
- **full-band: raw +0.188 = blur +0.128 (68%) + coverage +0.060 (32%)**
- **modal (0–250): raw +0.402 = blur +0.289 (72%) + coverage +0.113 (28%)**

So **~⅔ of the raw P2-4 magnitude-correlation climb was the convergence/blur confound.** The genuine
coverage effect is real but ~3× smaller than the raw curve implied.

### 3. Blur inflates LSD too — not just the soft metric
Under-training the *same* 45 rooms (2.17 → 4.33 dB) improves **both** mag corr (0.273 → 0.401) **and**
held-out LSD (7.70 → 6.25 dB) on unseen rooms — the "sharp decoder renders a specific wrong room; blur
renders a safe average" effect (P2-3) hits the **level** metric, not only correlation. The blur sweep
is monotonic: 45 mag corr rises 0.273 → 0.331 → 0.357 → 0.401 → 0.414 as convergence worsens
2.17 → 4.55 dB. Consequence: LSD alone is *not* a confound-proof metric; the matched comparison plus
**phase/RIR** (least blur-gameable) carry the clean signal. Note the paradox this exposes — the
*converged* 45 (0.273) is a **worse** zero-shot renderer than the *under-trained* 45 (0.401), because
blur generalises "safer"; a sharp model is punished for a wrong latent on an unseen room.

## Implications

- **For the P2-4 curve**: its **direction** (more rooms → better zero-shot) is trustworthy; its
  **magnitude/slope is not** — "modal closes 76% of the gap by 250" reflects raw numbers that are
  ~72% confound in the modal band. The clean modal coverage effect at matched convergence is **+0.113**,
  not +0.402. SCALING.md is annotated accordingly.
- **For P3-1 (geometry conditioning) — the motivation strengthens**: densification is a *real but
  modest* lever, and it comes with a **capacity penalty** (250 rooms can't even converge at fixed
  capacity → 4.3 dB plateau). Explicit (L,W,H) conditioning is now doubly motivated: it may reach
  better fidelity **without** the data cost **and without** the convergence penalty of piling on rooms.
  The baseline P3-1 must beat is the **matched-convergence 250 point** (mag 0.461/0.811, phase 0.348,
  RIR 0.421 at 4.30 dB), and ideally P3-1 should also report at matched convergence to avoid re-importing
  this confound.

## Deliverables & compute
- `outputs/multi_room_3d/density_45_conv/` — matched-convergence 45 retrain (frozen recipe, 22K, all 22
  checkpoints retained), scalars + convergence sweep. (The spec's `density_250_conv` was **superseded**
  by the approved design: the 250 endpoint is density_250@85K = the capacity plateau; matching is done by
  under-training 45. One 4-GPU run + 4 scavenger evals ≈ ~40 GPU-h — the minimal-compute path.)
- `outputs/coverage_curve/eval_conv45_lsd{43,45,38,34}/` — per-room full-suite metrics + provenance
  (which checkpoint/LSD each used).
- `outputs/coverage_curve/CONFOUND_CHECK.md` — **the verdict deliverable** (verdict + matched deltas +
  decomposition + 6-point table).
- Decision **D42** in DECISIONS.md; Q14 updated; SCALING.md annotated; CONTEXT_FOR_MANAGER + PHASE2_SUMMARY updated.

## Verification
Numbers independently re-derived from the per-room `metrics.json` (n=15 each, no NaN): matched deltas,
the blur/coverage decomposition (exact — it splits the raw gap at the matched midpoint), and the
checkpoint provenance (11K @ 4.333 dB is the matched point) all confirmed. A checkpoint-pruning bug
(trainer kept only the last 3 ckpts, deleting the 4.3 dB point) was caught and fixed (`ckpt_keep_last`,
D42) and the run redone clean before these numbers.

## Manager actions
- Read `outputs/coverage_curve/CONFOUND_CHECK.md`. The P2-4 curve is de-risked for **direction** and
  quantified for **magnitude**: cite the matched-convergence deltas, not the raw slope.
- Finalize the P3-1 spec (geometry conditioning: latent-only vs raw-(L,W,H) vs eigenfrequency-featurized)
  on the same frozen test set, benchmarked against the **matched-convergence 250** baseline, and reported
  at matched convergence.
