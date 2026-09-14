# Chunk results — P4-4: lock the shape-edit demo, then ramp

**The ramp did not ramp.** Scaling the corpus 5.4× (92 → 500 shapes, strictly nested, identical
test set, matched 60K iterations) moved the gated metric by **−0.0001** and *cost* **0.22** on the
depth sweep. The curve saturates by 250 shapes and reverses after. **Both of the chunk's targets
were missed by every arm** — best in-slab **0.8420** against 0.88, best band LSD **4.42 dB**
against 3.0. What did move: all three P4-4 arms **PASS** the frozen Gate-2 criterion where all
four P4-2 arms failed, and out-of-slab generalization reached **0.8359**, the best in the phase.

**The spec's premise that the three wins are orthogonal does not hold.** Attention-on-an-extent-
base lands within 0.003 dB of attention-on-a-mean-base: the attention residual dominates and the
extent base it rides on is very nearly inert.

**Date**: 2026-09-14 · **Branch**: `main` · Tests: **563 passed**
**Decisions**: D79–D82 · Task 3 (two-reflex-corner families) training at time of writing.

---

## 1. Task 2 — the scaling curve and the architecture ablation

Gate 2, 15 **frozen** test shapes, identical for every arm. The `gate` column is the frozen 0.80
criterion carried from P4-2 so the number stays comparable; P4-4's stricter 0.88 target is
reported separately below.

| arm | corpus | in-slab R | out-slab R | LSD dB | RIR r | σ | sweep mean | gate |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| BASE `mean_unmasked` | 60 | 0.7483 | 0.7369 | 4.84 | 0.9989 | 1.036 | 0.7615 | FAIL |
| P4-3 X1 `attn_residual` | 60 | 0.8438 | 0.7612 | 4.61 | 0.9988 | 1.130 | 0.7836 | PASS |
| P4-3 D `masked_mean` | 92 | 0.8021 | 0.7641 | **4.42** | 0.9992 | 0.933 | **0.8811** | PASS |
| **P4-4 C92** `attn+ext` | 92 | 0.8337 | 0.8060 | 4.46 | 0.9989 | 0.946 | 0.8276 | **PASS** |
| **P4-4 C250** `attn+ext` | 250 | **0.8420** | **0.8359** | 4.43 | 0.9989 | 1.006 | 0.6857 | **PASS** |
| **P4-4 C500** `attn+ext` | 500 | 0.8336 | 0.8292 | 4.51 | 0.9988 | 1.001 | 0.6036 | **PASS** |

### The scaling curve (D80)

| corpus | draws/shape | in-slab R | LSD dB | U amplitude | sweep mean |
|---:|---:|---:|---:|---:|---:|
| 92 | 5,217 | 0.8337 | 4.46 | 0.0619 | 0.8276 |
| 250 | 1,920 | 0.8420 | 4.43 | 0.1027 | 0.6857 |
| 500 | 960 | 0.8336 | 4.51 | 0.1206 | 0.6036 |

92→250 is **+0.0082**; 250→500 is **−0.0083**. Over 5.4× more data the gated metric moves
**−0.0001** while the depth sweep falls **0.828 → 0.604** and the U amplitude nearly doubles.

**The mechanism is arithmetic, not mysterious.** The batch draws 8 configs per iteration
regardless of corpus size, so 60K iterations is 480K config-draws however many rooms exist:
**5,217 / 1,920 / 960** draws per shape. The sweep probes **one** bounding box across 20 depths
and is sensitive to how well each individual room was fit. Gate 2 probes **15 varied** shapes and
is sensitive to breadth. Falling steps-per-shape hurts the first and not the second — which is
exactly the pattern, and the sweep degradation is **uniform across d̂** rather than concentrated
anywhere, the signature of a global fidelity loss rather than a regime-specific failure.

**This bounds D75 rather than contradicting it.** 60→92 shapes eliminated the U-curve for five
minutes of CPU because *breadth* was limiting there. By 250–500 at a fixed budget,
*steps-per-shape* is limiting and the trade reverses. Corpus size and iteration budget have to
scale together; P4-4 deliberately held iterations fixed to isolate data, so it measured the
trade rather than escaping it. A 500-shape corpus at matched draws/shape (~500K iterations) is
untested and is the obvious next question.

**Checked, not assumed**: zero sweep rooms appear in any training corpus. The balanced sampler
did spread the corpus out — the fraction within 0.3 of the sweep bounding box fell 7.6% → 3.8%
even as the absolute count rose 7 → 19 — but that is a second-order effect next to a 5.4× drop
in steps per shape.

### The architecture ablation (D81)

C92 and P4-3's D are the **same 92-shape corpus** and the **same 15 test shapes**, differing only
in `token_pool`:

| | in-slab | out-slab | val LSD | sweep mean | band LSD |
|---|---:|---:|---:|---:|---:|
| `masked_mean` (D) | 0.8021 | 0.7641 | **2.508** | **0.8811** | 4.42 |
| `attn_residual_extent` (C92) | **0.8337** | **0.8060** | 4.311 | 0.8276 | 4.46 |
| delta | **+0.0316** | **+0.0419** | +1.80 | −0.0535 | +0.04 |

The two metric families **disagree in sign**: the new pool generalizes better *across varied
shapes* and fits any *given* room worse. That inversion is not new — P4-3's X1 had the phase's
worst val LSD (4.273) and its best Gate-2 score (0.8438). P4-4 predicted the repeat from the
val-LSD trajectory **before** Gate 2 ran (C92 converged to 4.347 against X1's 4.344 at 52K) and
Gate 2 confirmed it. Two chunks, same sign, same explanation: val LSD is in-distribution on
training-corpus receivers, Gate 2 is 15 unseen shapes, and a shape-generalization chunk is gated
on the second.

**The extent base is nearly inert.** C92 sits within 0.003 dB of X1 on val LSD and within 0.01 on
Gate 2. The spec's premise that attention-residual and extent pooling are two orthogonal wins
that combine does not hold — attention dominates and the base barely matters. (D69's retraction
already showed extent_sum's *apparent* advantage was an evaluation artifact; this shows its real
advantage, measured correctly, vanishes under attention.)

### Targets (D82)

| target | best achieved | verdict |
|---|---|---|
| in-slab ≥ 0.88 | 0.8420 (C250) | **MISS** |
| band LSD < 3.0 dB | 4.42 (P4-3 D) | **MISS by 47%** |

The LSD miss was flagged as a risk when planning and again mid-training, before the evaluation
ran. It is reported as a miss; the targets are not retroactively relaxed.

---

## 2. Task 1 — the shape-edit demo pack (COMPLETE)

Two checkpoints, each named in its captions, because they are good at different things.

| | spatial R (3 modes) | LSD dB | modal RIR r |
|---|---|---|---|
| room A — X1 | 0.957 / 0.935 / 0.931 / 0.872 | 5.5–5.8 | 0.82–0.85 |
| room B — X1 | 0.936 / 0.963 / 0.938 / 0.879 | 5.7–5.9 | 0.74–0.81 |
| morph — D, **full d̂ range** | **0.894 → 0.980** | 3.49–4.70 | 0.86–0.94 |

The spec asked for the morph from X1 "now that the U is flattened", but X1 is not the arm that
flattened it — D is (sweep amplitude 0.075 vs 0.163; shallow half +0.889 vs +0.750). Panels come
from X1 (best on the frozen test shapes) and the full-range morph from D, whose **best** frame is
the pure rectangle at d̂ = 0. Running X1 over the full range would have visibly degraded the
shallow frames.

`n_modes: 3` is recorded in the metrics JSON: this pack averages over 3 resolvable modes while
Gate 2 and the sweep use 6, so the demo reads higher for the same checkpoint. That is a protocol
difference, not a discrepancy, and it is now self-describing.

---

## 3. Task 3 — two reflex corners (corpus COMPLETE, training in flight)

350 rooms built, 70 per family — rect (0 reflex corners, 4 tokens), L (1, 6), U (2, 8), T (2, 8),
DN (2, 8) — 60 train + 10 test each, 3 in-slab test rooms per notched family.
**Dataset gate: 14/14 PASS.**

Nothing in the model, conditioner, renderer or solver changed: `MAX_SEG_POLY = 12` already covers
4/6/8 and leaves room for a 10-vertex Z, `cond_dim` stays 324, existing checkpoints stay loadable.

The gate's one **simulator-level** item is the one bookkeeping cannot fake: an L and a U with the
**same bounding box** differ by **4.31 dB mean**. Every other item — unique filenames, balanced
families, matching token counts, an empty hold-out slab, grid alignment, reflex-corner counts —
would pass on a corpus whose second notch had been silently dropped.

**The token-count control, and a constraint worth having found.** Comparing a rectangle (4
tokens) with a U (8) confounds token count with geometry. The control re-tokenizes the **same**
room with collinear vertices: identical physics, identical `.h5`, identical receivers, only the
conditioning changes. `MAX_SEG_POLY = 12` bounds what is reachable — rect at 4/8/12, L at 6/12,
and the 8-token families not at all — which makes the **rectangle the cleanest control in the
corpus: one room, three token counts, literally fixed geometry.** The U/T/DN rows carry the
across-*geometry* comparison instead, so the spec's single question splits into two clean
measurements rather than one confounded one.

Training runs on `attn_residual_extent`, chosen on Task 2's gated evidence rather than assumed.

---

## 4. What the manager should take from this

1. **Corpus scaling is not free at fixed compute.** 5.4× more data bought nothing on the gate and
   cost 0.22 on the sweep, because draws-per-shape fell 5.4×. Scale the corpus and the iteration
   budget together, or state which one you held fixed. The next experiment is 500 shapes at
   matched draws/shape (~500K iterations), which is untested.
2. **Two metric families disagree in sign here, reproducibly.** Val LSD (in-distribution, training
   receivers) and Gate 2 (15 unseen shapes) have now inverted in two consecutive chunks for the
   same architectural reason. Quote both or neither; a shape-generalization claim belongs on
   Gate 2.
3. **The attention residual is the whole architectural effect.** The extent base adds ~0.003 dB.
   Treat `attn_residual` and `attn_residual_extent` as one arm, not two.
4. **The targets were missed.** 0.842 against 0.88, 4.42 dB against 3.0. The gate moved; the
   targets did not.
5. **`enumerate_modes` is the analytic rectangular mode list and three chunks used it on
   non-convex rooms without writing down why** (D79). The practice is sound — it only picks which
   bins to correlate at, and prediction and truth are read at the same bin — but the argument
   existed nowhere and now does.
