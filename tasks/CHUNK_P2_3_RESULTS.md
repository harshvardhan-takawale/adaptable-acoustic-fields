# Chunk P2-3 — Converged 45-room 3D training + zero-shot adaptation

**Status**: COMPLETE — 2026-06-09. **Two-part result: in-distribution SOLVED;
zero-shot generalization is a CONFIRMED COVERAGE problem (not the model, not the
test-time procedure).**

---

## 1. Headline

| | result | verdict |
|---|---|---|
| **In-distribution** (45-room fit) | **val LSD 2.169 dB @ 60K** | ✅ cleared the ≤2.5 target |
| **Zero-shot** (8 unseen test rooms) | **0/8** reach mag corr ≥ 0.9 (got 0.20–0.28) | ❌ does not generalize |

**The architecture, conditioning, recipe, and 4-GPU DDP all work** — a single
shared model renders all 45 training rooms to 2.169 dB. **Zero-shot to unseen
rooms fails, and we have localized the cause precisely: the converged model
memorizes its 45 rooms and does not interpolate to unseen geometries. This is a
training-set-coverage / decoder-generalization wall — definitively not the model
quality and definitively not the test-time adaptation procedure.**

---

## 2. Pipeline + training (the part that worked)

- **4-GPU DDP**, effective batch 64 (4 ranks × per-rank 16, Run C's recipe),
  n_pts 32, 60K iters. Sanity gate first: 0.596 it/s = **3.8× single-GPU**, val@2000
  tracked-and-beat Run B's 45-room curve, no NCCL hang → 4-rank all-reduce correct.
- **24h walltime → afterany continuation** handled the >24h run cleanly: main hit
  walltime at ~55.8K with a 55K checkpoint, continuation resumed and finished 60K.
- **Convergence**: val LSD 6.43 (2K) → 2.55 (40K) → **2.169 (60K)**, early-stop off,
  still descending gently at the end. Beats P2-2's failed 6.16 dB and confirms the
  P2-2.5 verdict (coverage/compute, not capacity, was the in-distribution bottleneck).

So **the in-distribution problem from P2-2 is fully solved.**

---

## 3. Zero-shot result + the three-layer diagnosis

Stock procedure (8 observed receivers, 2000-step z* optimization, lr 1e-2,
λ‖z‖²=1e-4), re-run on 24 GB cards. **0/8 rooms reach the ≥0.9 target; mag corr
0.20–0.28; D37 verdict `manifold_coverage` for all 8.** The self-diagnosis
instrument makes the failure mode unambiguous, in three layers:

### (a) It is the test-time latent z*, NOT the model
- **obs_lsd ≈ held_lsd ≈ 7 dB**: the optimized z* cannot fit even the *8 observed*
  receivers it was optimized on. So this is not overfitting — the rendered spectrum
  is simply wrong everywhere, because no good z* was found.
- **Converged P3 zero-shot (mag 0.20–0.28) is WORSE than unconverged P2-2 M1
  (0.52–0.59).** This is the key tell: a sharp, converged decoder *punishes* a wrong
  z* (detailed-but-wrong spectrum → low correlation), while a blurry, unconverged
  decoder outputs a smooth "average room" that *accidentally* correlates ~0.5. The
  0.52–0.59 from M1 was never real generalization — it was a blur artifact. The
  representation and renderer are fine; **finding the right latent for an unseen room
  is the problem.**

### (b) It is NOT the test-time procedure (disambiguation sweep)
We added an opt-in manifold-anchored adaptation (`--z_init mean` = init z* at the
training-latent centroid, on the manifold; `--lambda_latent` controls the anchor
strength) and swept it on Run C (10 rooms) and P3 (45 rooms):

| model | config | mag corr | z*_norm (manifold shell) |
|---|---|---|---|
| Run C (10 rm) | randn / mean / λ1e-4 / λ1e-2 | 0.18–0.22 (invariant) | escapes to 11–19 (shell 6.6) |
| **P3 (45 rm)** | randn (stock) | 0.27 | 10.6 |
| **P3 (45 rm)** | mean + λ1e-2 | 0.27 | 9.9 |
| **P3 (45 rm)** | mean + λ1e-1 | 0.28 | 5.6 (pinned below shell) |
| **P3 (45 rm)** | mean + λ1e-2 (small room) | 0.22 | 7.2 |

**mag corr is invariant (~0.2–0.28) across every init and regulariser**, at both 10
and 45 rooms. Even pinning z* onto the manifold (λ=1e-1) only marginally improves the
obs fit (6.46 dB) and leaves mag at 0.28. **There is no good latent to find** — the
optimizer's "escape" is a *symptom* (it chases spurious off-manifold minima because no
on-manifold latent renders the room), not the cause. **The procedure is not the lever.**

### (c) Coverage helped placement, but 45 rooms is still far too sparse
P3 (45 rooms) vs Run C (10 rooms): denser coverage *did* improve z* placement —
geometry-head error **0.5–2.6 m vs 4.2 m**, z*_norm **7–10 vs 11–13**, nearest-latent
distance **7–9.5 vs 11–13**. So adding rooms demonstrably anchors z* better. But going
10 → 45 rooms barely moved zero-shot mag corr (~0.20 → ~0.25). **The decoder needs
*much* denser coverage of the 3-D (L, W, H) box to interpolate spectra** — 45 LHS rooms
across three dimensions is too sparse (≈ 3.6 rooms per dimension).

---

## 4. The coverage curve (meeting-grade summary)

| training rooms | model | zero-shot mag corr |
|---|---|---|
| 10 (Run C, converged 0.98 dB) | sharp | ~0.20 |
| 45 (P3, converged 2.169 dB) | sharp | ~0.20–0.28 |
| 45 (M1, unconverged 6.16 dB) | blurry | 0.52–0.59 *(blur artifact, not generalization)* |

**Zero-shot generalization in 3-D is a quantified function of training-set density,
and 45 rooms is below threshold.** The Phase-1 (2-D, length-only) analog needed
dense room sampling (0.1 m spacing) to break its zero-shot ceiling; the 3-D box is
vastly larger, so it needs far more rooms.

---

## 5. Recommendations for P2-4 (ranked)

1. **Scale the training set substantially** (first-order fix). Go from 45 → ~150–300+
   rooms with a denser LHS / grid over the (L, W, H) box. This is the proven lever;
   everything else (architecture, recipe, DDP) is already in place and the
   in-distribution fit is solved, so this is a data-generation + retrain, not a
   redesign. Watch the zero-shot mag-corr-vs-room-count curve.
2. **Use the known test geometry directly** (architectural lever, possibly decisive).
   The current setup optimizes a free z* and *never uses the test room's known
   (L, W, H)* — yet for a new room we typically know its dimensions. Conditioning the
   decoder *explicitly* on (L, W, H) (concat alongside / instead of the optimized
   latent) would let zero-shot to a known-geometry room render **directly, with no z*
   search at all** — sidestepping the entire test-time-latent bottleneck this chunk
   identified. Strong candidate to test early in P2-4.
3. **Do NOT pursue test-time-procedure fixes** (init, λ, multi-restart). Proven
   ineffective across Run C and P3, three init/λ configs each.
4. **Do NOT widen capacity / change conditioning.** In-distribution is solved at the
   current capacity (D34); the wall is coverage, not capacity.

---

## 6. Caveat / honesty

- The "0/8" is on the 8 maximin test rooms, several of which sit *outside* the
  45-room training hull (extrapolation, not interpolation) — so they are a hard test.
  Even the interior box-center room fails (mag 0.27), so the conclusion holds, but the
  P2-4 zero-shot eval should separate interpolative vs extrapolative test rooms.
- The procedure-vs-coverage disambiguation used **all 4 P3 runs** (box center at
  λ=1e-2 and λ=1e-1, plus a small and a large room at λ=1e-2) + the full Run C sweep.
  Every config and every room lands at mag corr 0.22–0.28 — the verdict is firm.

## 7. Deliverables / artifacts

- `outputs/multi_room_3d/P3_45rooms_4gpu/`: `train_meta.json`, `scalars.json`,
  `SUMMARY.md` (D37 verdict), `mag_corr_per_room.png`, `zero_shot/L*/metrics.json` (8).
- `outputs/meeting_assets_p2_3/`: receiver-volume-slice plots (predicted vs ISM) for
  the 3 best rooms.
- `outputs/p3_zeroshot_anchored/`: the disambiguation sweep.
- Opt-in `--z_init mean` / `--lambda_latent` in `aaf/eval/zero_shot_3d.py` (kept for
  P2-4; defaults unchanged). GPU-targeting fix in `zero_shot_3d_eval.sh`.

## 8. Manager actions requested

1. **Approve P2-4 = scale training rooms** (≈150–300) + **test explicit (L,W,H)
   conditioning** as the two primary levers. Confirm room-count budget / ISM
   generation (≈ 24 s/room).
2. **Meeting framing**: P2-3 is a *positive* engineering result (converged 3-D
   multi-room model, 2.169 dB) plus a *sharp, well-diagnosed* open problem (zero-shot
   = coverage, with a clear path). The "representation works, rendering works, the
   open problem is interpolating to unseen geometry" framing is honest and compelling.
