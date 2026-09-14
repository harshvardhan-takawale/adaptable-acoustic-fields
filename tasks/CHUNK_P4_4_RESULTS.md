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
**Decisions**: D79–D84 · All three tasks complete.

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

## 3. Task 3 — two reflex corners, and the token-count control (COMPLETE)

One model, 300 training rooms — 60 each of rect (0 reflex corners, 4 tokens), L (1, 6), U (2, 8),
T (2, 8), DN (2, 8) — evaluated on 50 held-out rooms, 10 per family. **Dataset gate 14/14 PASS**,
including the one item bookkeeping cannot fake: an L and a U with the **same bounding box** differ
by **4.31 dB mean**. Nothing in the model, conditioner, renderer or solver changed.

### Q1 — corner count degrades monotonically, and decelerates (D83)

| reflex corners | family | tokens | spatial R | LSD dB | RIR r |
|---:|---|---:|---:|---:|---:|
| 0 | rect | 4 | **+0.8469** | 4.44 | 0.856 |
| 1 | L | 6 | +0.7663 | 4.69 | 0.839 |
| 2 | T | 8 | +0.7582 | 4.64 | 0.806 |
| 2 | DN | 8 | +0.7039 | 4.83 | 0.807 |
| 2 | U | 8 | **+0.6727** | 4.82 | 0.758 |
| **2 (pooled)** | | 8 | **+0.7116** | 4.77 | |

0 → 1 costs −0.081; 1 → 2 costs −0.055. Decelerating, not collapsing: 2-corner rooms retain 84%
of the rectangle's accuracy. **U is hardest**, which is geometrically sensible — two notches on
the *same* wall leave a narrow north stem and the most severely non-convex interior.

σ reads **1.0132** over 32 probeable rooms. Adding reflex corners does not make the model
discover occlusion — unchanged from every measurement since P4-2.

### Q2 — and corner count is NOT the dominant term (D84)

The same rooms, re-tokenized with **collinear** vertices. Identical physics, identical `.h5`,
identical receivers, identical polygon (verified: `polygon_contains` agrees on 4000 random
points). Only the conditioning vector changes.

| room | tokens | spatial R | Δ vs native |
|---|---:|---:|---:|
| rect | 4 (native) | +0.8469 | — |
| rect | 8 | **+0.6029** | **−0.2440** |
| rect | 12 | +0.7163 | −0.1306 |
| L | 6 (native) | +0.7663 | — |
| L | 12 | **+0.5642** | **−0.2021** |

**Re-describing an identical room costs more than adding two reflex corners does** (−0.13 to
−0.24 against −0.135). The rectangle is a room the model reconstructs at 0.847 when described
minimally and 0.603 when described redundantly — same room, same physics, same receivers.

**The honest reading is not "token count costs accuracy".** The model was trained with one
tokenization convention, so in training 8 tokens *always* meant a 2-corner room. A redundantly
tokenized rectangle is **out of distribution in the conditioning**, and this measures robustness
to that shift, not an intrinsic per-token cost. **The non-monotonicity proves it**: the rectangle
scores 0.603 at 8 tokens but 0.716 at 12, which no monotone "more tokens is worse" law produces.
What the data supports is weaker and more useful: **the learned conditioning depends on HOW a
room is described, not only on WHICH room it is.**

### What this means for the Z-corridor

A Z has **10 tokens**; every training room has 4, 6 or 8. It is out of distribution in exactly
the dimension this control shows the model is fragile in, and that fragility is worth more
accuracy than the extra corner. Corner count alone does not forbid Z — the per-step loss is
shrinking — but the tokenization result, not the corner table, is the right predictor.

**Recommended before Stage 3**: train with **tokenization augmentation** — present each room at
several equivalent tokenizations so the conditioning is pushed toward invariance — then re-run
this control. One training run, **no new simulation** (the control reuses existing `.h5`
unchanged), and it converts Z from a gamble into a test.

---

## 4. What the manager should take from this

1. **Corpus scaling is not free at fixed compute.** 5.4x more data bought −0.0001 on the gate and
   cost 0.22 on the sweep, because draws-per-shape fell 5.4x. Scale corpus and iteration budget
   together, or state which you held fixed. 500 shapes at matched draws/shape is untested.
2. **Tokenization fragility, not corner count, is what Z hinges on.** Re-describing an identical
   room costs −0.13 to −0.24; adding two reflex corners costs −0.135. Train with tokenization
   augmentation before Stage 3 — one run, no new simulation.
3. **Two metric families disagree in sign, reproducibly.** Val LSD and Gate 2 have now inverted
   in two consecutive chunks for the same reason. A shape-generalization claim belongs on Gate 2.
4. **The attention residual is the whole architectural effect.** The extent base adds ~0.003 dB;
   treat `attn_residual` and `attn_residual_extent` as one arm.
5. **Both targets were missed.** 0.842 against 0.88; 4.42 dB against 3.0. The gate moved (all
   three P4-4 arms pass where all four P4-2 arms failed); the targets did not.
6. **`enumerate_modes` is the analytic rectangular mode list** and three chunks used it on
   non-convex rooms without writing down why (D79). Sound, but the argument existed nowhere.
