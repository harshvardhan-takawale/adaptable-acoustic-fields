# Phase 4 — Status Summary for Manager Review

**Scope**: shape editing. The phase goal is to train on rectangles and L-rooms and render a
Z-shaped corridor zero-shot. It is deliberately staged, each stage behind a gate, so that when
something breaks we know *which* thing broke.

**Status as of 2026-09-14**: **P4-1, P4-2, P4-3, P4-4 all complete.** Gate 2 now passes on five
arms. The Z-corridor is **not** yet licensed, and the phase now knows precisely why.

---

## TL;DR

**Shape generalization is a data problem, corpus scaling saturates fast, and the last barrier to
Z is tokenization fragility — not corners, not occlusion.**

* **Gate 0 PASS** (P4-1). Boundary tokens beat global `(L, W)`: 0.982 vs 0.951.
* **Gate 1 PASS** (P4-1). Per-scene fit of one non-convex L-room, NLOS Pearson +0.974.
* **Gate 2 FAIL** (P4-2), all four arms; best 0.7978, σ ≈ 1.0 everywhere.
* **Gate 2 PASS** (P4-3), two arms: attention-residual **0.8438**, more-data **0.8021**.
* **Gate 2 PASS** (P4-4), all three arms; best out-of-slab **0.8359**.
* **Scaling saturates at ~250 shapes** and reverses after (D80).
* **Tokenization fragility exceeds the whole corner-count effect** (D84). ← the Z blocker
* **σ ≈ 1.0 on every arm ever measured.** No learned occluder, and none needed (D77).

---

## The numbers that matter (keep these regimes distinct)

| regime | number | what it is |
|---|---|---|
| zero-shot, unseen rectangle | **0.982** | P4-1 Gate 0 |
| per-scene fit, non-convex | **+0.974** | P4-1 Gate 1 — *not* generalization |
| zero-shot, unseen SHAPE | **0.8438** | best Gate 2 in the phase (P4-3 X1) |
| corner count 0 → 2 | 0.847 → 0.712 | P4-4 Task 3, gentle and decelerating |
| **same room, re-tokenized** | **0.847 → 0.603** | **P4-4 Task 3 — the largest effect measured** |
| mechanism | σ **≈ 1.0** | no learned occluder, on any arm |

---

## The three results that redirect the phase

**1. Corpus scaling saturates, because compute was held fixed (D80).** 92 → 500 shapes at matched
60K iterations moved the gate by −0.0001 and cost 0.22 on the depth sweep. Draws-per-shape falls
5,217 → 1,920 → 960; the sweep needs per-room fidelity, the gate needs breadth. This *bounds*
D75's "density is first-order" rather than contradicting it: breadth was limiting at 60→92, and
steps-per-shape is limiting by 250→500. **Corpus size and iteration budget must scale together.**

**2. Tokenization, not corners, is what Z hinges on (D84).** Re-describing an *identical* room
with collinear vertices costs −0.13 to −0.24; adding two reflex corners costs −0.135. A Z has 10
tokens and every training room has 4, 6 or 8. The honest framing: training used **one**
tokenization convention, so a redundantly tokenized room is out of distribution in the
conditioning — and the non-monotonicity (0.603 at 8 tokens, 0.716 at 12) rules out a simple
per-token cost law. **The conditioning depends on how a room is described, not only which room.**

**3. Two metric families disagree in sign, reproducibly (D81).** Val LSD (in-distribution,
training receivers) and Gate 2 (unseen shapes) inverted in P4-3 and again in P4-4, for the same
architectural reason. P4-4 *predicted* the second inversion before the gate ran. Quote both or
neither; a shape-generalization claim belongs on Gate 2.

---

## Honesty notes

1. **P4-4 missed both its targets** — 0.8420 against 0.88, 4.42 dB against 3.0 (a 47% miss).
   Flagged as a risk when planning and again mid-training. The gate moved; the targets did not.
2. **The attention residual is the whole architectural effect.** The extent base adds ~0.003 dB;
   `attn_residual` and `attn_residual_extent` are one arm, not two.
3. **D69 was retracted** (D73): `load_model` ignored `token_pool`, so P4-2's extent arms were
   evaluated as mean-pooled — silently, because extent adds no parameters.
4. **The σ probe was counting out-of-room samples** — 82% of them (D74). Conclusions survived; a
   *localized* contrast was being diluted.
5. **`enumerate_modes` is the analytic rectangular mode list**, used on non-convex rooms by four
   chunks with no written justification until D79. Sound, but the argument existed nowhere.
6. **The token-count control measures robustness to an unseen tokenization**, not an intrinsic
   per-token cost. Stated in D84(c) and in the results doc.
7. **`tx` has been a single fixed source in 100% of the 2D corpus.** The source-position encoders
   remain essentially unconstrained.

---

## Artifacts (raw GitHub, `…/main/<path>`)

`tasks/CHUNK_P4_4_RESULTS.md` · `tasks/CHUNK_P4_3_RESULTS.md` ·
`outputs/p4_4/p4_4_{C92,C250,C500}_eval/GATE2.json` ·
`outputs/p4_4/sweep/{C92,C250,C500}/sweep_metrics.json` + `figI/figJ/figK*.png` ·
`outputs/p4_4/p4_4_FAM_eval/FAMILY_EVAL.json` · `outputs/p4_4/family/DATASET_GATE.json` ·
`outputs/p4_4/demo/demo_metrics.json` + `fig{L_A,L_B,M_A,M_B,N}*.png` ·
`configs/sweeps_2d_mat/p4_4_shapes_{250,500}.json` ·
`configs/sweeps_2d_mat/p4_4_family_manifest.json` · `DECISIONS.md` (D79–D84)

---

## Recommended next step

**Tokenization augmentation, then re-run the control, then Z.** Present each room at several
equivalent tokenizations during training so the conditioning is pushed toward invariance. It is
**one training run and zero new simulation** — the control reuses existing `.h5` unchanged — and
it converts the Z-corridor from a gamble into a test with a pre-registered predictor.

**Do not scale the corpus further without scaling iterations.** 500 shapes at matched
draws/shape (~500K iterations) is the untested cell and the only way to separate "more data does
not help" from "more data at fixed compute does not help".

**Do not rewrite the renderer** (D77) and **do not reach for DC-masked loss** (D70). Both closed.
