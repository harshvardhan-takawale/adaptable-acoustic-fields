# Run C zero-shot probe — RESULT (Spec A, early signal)

**Date**: 2026-06-08. **Verdict**: ❌ **zero-shot FAILS on the converged Run C model** —
the cause is **not** in-distribution convergence, **not** the model, and (validated
below) **not** the test-time procedure. The converged 10-room model **memorizes its
10 rooms and does not interpolate to unseen geometries** — a coverage /
decoder-generalization wall. **The decisive variable for P3 is room count (45 vs 10),
not the adaptation procedure.**

> **VALIDATION (added 2026-06-08, per "validate on Run C only").** I swept the
> test-time procedure across 3 configs on the 2 cleanest interior rooms:
> baseline (z* init ‖z‖≈1, λ‖z‖²=1e-4), manifold-anchored (init at training-latent
> centroid, λ‖z*−z̄‖²=1e-2), and init-only (manifold init, weak λ=1e-4).
>
> | room | baseline | anchored λ1e-2 | init-only λ1e-4 |
> |---|---|---|---|
> | box center | mag 0.217 / obsLSD 7.14 | 0.202 / 6.84 | 0.211 / 7.10 |
> | L3.65 | 0.184 / 7.50 | 0.179 / 7.01 | 0.192 / 7.54 |
>
> **mag corr is invariant (~0.2) across all three**; obs LSD stays ~7 dB; z* fails to
> fit even the 8 *observed* receivers under any init/regulariser. init-only z* still
> escaped (L3.65 → ‖z‖ 19.4) because the obs gradient actively drives z* off-manifold
> — there is **no good on-manifold latent to settle at**. So the manifold-escape seen
> in the baseline is a *symptom* (the optimiser chasing spurious minima because no
> latent renders the room), **not the cause**. The opt-in `--z_init mean` /
> `--lambda_latent` knobs added to `zero_shot_3d.py` are kept (defaults unchanged) but
> **do not fix this** — coverage does.
>
> **Implication for P3 (revised):** leave P3's zero-shot **stock** — the procedure is
> fine. P3's result is a clean test of *"is 45 rooms (4.5× denser) enough for 3D
> zero-shot interpolation?"* If P3 zero-shot also fails, the fix is **more training
> rooms (P2-4)**, not a procedure change. (Consistent: P2-2's 45-room model already
> scored mag 0.47–0.64 — better than this 10-room 0.2 — so coverage is the axis, and
> 45 helps, though it may still fall short of ≥0.9.)

---

### Original discovery narrative (how we got to the verdict)

The first signal pointed at the **test-time latent adaptation escaping the training
manifold**; the validation sweep above showed that escape is a symptom of the deeper
coverage wall, not the root cause.

## Setup

- **Model**: Run C from P2-2.5 (`outputs/diag_p2_2_5/C_10rm_b64`), 10 rooms,
  eff-batch 64, **converged to 0.98 dB in-distribution** (the architecture's 3D
  multi-room ceiling). Loaded `ckpt_iter0030000.pt`; geometry head present.
- **Test rooms** (3): the box center (4.50, 4.00, 3.25) — the only one of the 8
  maximin test rooms interior to Run C's 10-room convex hull — plus two **freshly
  generated interpolative interior rooms** (3.65, 4.04, 3.24) and (4.36, 4.14, 3.96),
  built as convex combinations of training-room vertices and **verified strictly
  inside the hull** (scipy `Delaunay.find_simplex ≥ 0`), simulated with the same
  physics (α 0.15, source 0.5, fs 4096). None coincide with a training room.
- **Adaptation**: stock `aaf/eval/zero_shot_3d.py` — freeze network, optimize z*
  from 8 observed receivers, 2000 steps, Adam lr 1e-2, λ‖z‖²=1e-4. Predict held-out.

## Result — all 3 rooms fail badly

| room | mag corr | phase mw | RIR ρ | env corr | modal LSD | branch (D37) |
|------|---:|---:|---:|---:|---:|---|
| box center 4.50/4.00/3.25 | **0.217** | 0.056 | 0.063 | 0.752 | 6.93 | manifold_coverage |
| 3.65/4.04/3.24 | **0.184** | 0.067 | 0.074 | 0.758 | 7.46 | manifold_coverage |
| 4.36/4.14/3.96 | **0.179** | 0.047 | 0.052 | 0.742 | 7.19 | manifold_coverage |

Target was mag corr ≥ 0.9. We are at ~0.2. (For reference, P2-2's *unconverged*
45-room model gave 0.47–0.64 — i.e. worse coverage-per-room but more rooms did
*better* than this converged 10-room model. Consistent with a coverage axis.)

## Self-diagnosis — the mechanism is z* escaping the manifold

The geometry-head + manifold-distance instrument (D37) localizes the failure
precisely. It is **not** test-time overfitting and **not** the model:

1. **obs_lsd ≈ held_out_lsd (~7.1–7.6 dB on both).** z* does not even fit the 8
   *observed* receivers — so it is not overfitting them; the rendered spectrum is
   simply wrong everywhere.
2. **z*_norm blows up to 11.5–12.9**, while the **10 training latents sit at norm
   ~6.6** (range 5.9–7.4; mean inter-latent distance 5.2). The inner-loop trace
   shows ‖z‖ growing **1.16 → 12.5**. λ=1e-4 is far too weak to hold z* on the
   manifold.
3. **z* is initialized at norm ~1.16** — *far below* the manifold shell (~6.6).
   So z* starts off-manifold (too small), and the optimizer overshoots to
   off-manifold (too big), never passing through the region where the decoder is
   constrained.
4. **The geometry head on z* returns nonsense** — e.g. (0.51, **−0.18**, 2.32) m
   (negative width) — confirming z* lands where the decoder/geom-head were never
   trained. The "4 m geometry error" is the *symptom* of off-manifold z*, not a
   mis-sized room.

So the D37 branch is `manifold_coverage`, but the sharper statement is:
**the test-time latent search leaves the trained latent manifold** — under-
regularized (λ too small) and badly initialized (z* starts at ‖z‖≈1.16 vs a
manifold at ‖z‖≈6.6), compounded by a sparse 10-point manifold with few anchors.

## What this means for P3 (the load-bearing implication)

- **It does NOT doom P3's in-distribution result** (the primary deliverable),
  which is converging well (3.96 dB @ 15K, heading toward ~2 dB). Representation
  and rendering are fine — this is purely the *adaptation* step.
- **It DOES threaten P3's zero-shot.** P3's pending zero-shot jobs call the *same*
  `zero_shot_3d.py` with the same weak λ and the same ‖z‖≈1.16 init. If z* escapes
  the (denser, 45-room) manifold there too, P3's zero-shot could fail for a
  **procedure** reason and we'd misattribute it to the model.
- **Two non-exclusive fixes**, both testable before P3's zero-shot runs (~19 h away):
  1. **Fix the test-time optimization**: initialize z* at the training-latent
     *mean* (‖z‖≈6.6, on the manifold) instead of small-random, and raise λ (e.g.
     1e-2) and/or add a norm constraint / early stop — keep z* on the manifold.
  2. **Rely on denser coverage**: P3's 45 rooms give a much denser manifold (more
     anchors). This helps, but alone may not stop z* escaping if the init/λ stay
     as-is.

## Recommendation (flagged for immediate decision)

Before P3's zero-shot fires, **fix the test-time latent adaptation** (init at
manifold mean + stronger λ). The Run C model is a clean, ready testbed to validate
the fix in ~10 min: if a manifold-anchored z* recovers mag corr on these 3 interior
rooms, the fix transfers to P3; if it does not, the bottleneck is genuinely
coverage (→ more rooms, P2-4). Either way we learn the right thing *before*
spending P3's zero-shot on a broken procedure.

Backup-result value: even as a negative, this is a clean, well-diagnosed result
for the meeting — "the representation and renderer work; the open problem is
test-time adaptation onto the latent manifold."

Artifacts: `outputs/runC_zeroshot_probe/L*/metrics.json` + per-room signal plots.
