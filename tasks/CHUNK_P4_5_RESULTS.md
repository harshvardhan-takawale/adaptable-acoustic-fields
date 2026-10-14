# Chunk P4-5 — High-frequency diagnosis and fixes: RESULTS

**Status: IN PROGRESS (2026-10-14).** Diagnostics A1-A3 and the record corrections are complete. The arms are training: B1-B4 on tron, B1b/B5/B6/B7 on scavenger. 30K evaluations land 03:00-06:00 and 60K ~10 h later. Sections marked PENDING fill in as each lands. Decisions D92-D112 in [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md).

---

**Shape editing is not at parity with material editing below 100 Hz. The earlier apparent parity was shape measured favourably against material measured honestly (D106).** The material corpora drew receivers per room, at positions no training room shares (`scripts/build_2d_mat_dataset.py:39-42`, `scripts/build_p3_3fast_trackA.py:32-35`), so their 1.7-2.3 dB band LSD is a field measurement. Shape's 2.32 dB was measured on the absolute receiver lattice every P4 room shares, which the attention-pooling models memorised. Measured as a field (dense, off-lattice), shape below 100 Hz is **3.41 dB (Arm D, mean pooling) to 3.89 dB (C250)**.

**The chunk's question, answered in the spec's terms.** The pre-registered prediction holds as scored: bins >= 100 Hz carry **0.9%** of the parameter-gradient projection at C250 @ 60K, against "< 15%" (D97). The mechanism it named is refuted. At initialisation the same share is **65.7%**, the uniform 66.7% that an L1 loss gives. The starvation EMERGES in training, and at 60K the 0-10 Hz band carries ~93% of the step. The baseline (C250 @ 60K, dense, OFF the training lattice, mean over +-2.5 Hz) scores spatial Pearson **+0.209 at 100 Hz and +0.257 at 200 Hz**. On the lattice it scores +0.547 and +0.455, and the gap between those is memorisation (D99), not acoustics. **Which arm moved the tail most: PENDING** (criterion: arm minus baseline at 200 Hz, off-lattice, beyond the receiver-bootstrap 95% CI of the difference).

**The bigger finding was not about frequency.** Every P4 room places its receivers on one absolute lattice, x, y = 0.12 + 0.06 k. A query-position-dependent conditioning path (attention pooling) gave the network a position-indexed storage channel, and on that fixed lattice it used the channel for memorisation before physics (D99, D110). The standard protocol of the field, held-out positions drawn from the training measurement grid, cannot detect this. Every Gate-2 verdict, sweep curve and family number in P4 was measured that way.

---

## 1. Correction to the record

Dense re-score of Gate 2 with the frozen evaluator (exact on all 14 fields when pointed at the original data; D102), with across-shape bootstrap 95% CIs (D105). OFF-lattice is the primary figure (Rule 11):

| checkpoint | pooling | published in-slab | dense **off-lattice** in-slab | on-minus-off GAP (in-slab) |
|---|---|---|---|---|
| P4-3 D | masked_mean | 0.8021 [0.765, 0.839] at threshold | **0.8063 [0.766, 0.845] at threshold** | +0.008 [+0.001, +0.015] |
| P4-3 X1 | attn_residual | 0.8438 [0.794, 0.886] at threshold | **0.7581 [0.715, 0.811] at threshold** | +0.085 [+0.044, +0.121] |
| P4-4 C92 | attn_residual_extent | 0.8337 [0.794, 0.869] at threshold | **0.7545 [0.710, 0.795] FAIL** | +0.076 [+0.054, +0.098] |
| P4-4 C250 | attn_residual_extent | 0.8420 [0.804, 0.878] PASS | **0.6109 [0.580, 0.656] FAIL** | +0.230 [+0.191, +0.277] |
| P4-4 C500 | attn_residual_extent | 0.8336 [0.782, 0.875] at threshold | **0.5641 [0.544, 0.587] FAIL** | +0.270 [+0.226, +0.308] |

(Level CIs are D105's. Gaps come from `scripts/p4_5_gate2_ci.py`, paired and slab-stratified, which reproduces D105's Arm D row exactly. It uses a fresh generator per checkpoint, so its level CIs for other rows can differ from D105's in the third decimal, with no verdict changed.) With 6 in-slab shapes the gate cannot resolve +-0.04 (D105(e)), so the robust findings are the SHIFTS and the POOLING contrast, not a count of verdicts.

* **Pooling (D104, D105).** On the identical 92-shape corpus, mean pooling (D) beats attention (C92) off-lattice by **+0.122 / +0.324 / +0.237** at 25-100 / 100-200 / 200-300 Hz, in 14 / 15 / 15 of 15 shapes, with paired CIs excluding zero.
* **Physics (D107, D108).** The attention arms' high-frequency fields lack the Helmholtz ring. Excess supra-physical fraction (|k| > 1.5 k0, 100-300 Hz): C250 +0.272, C92 +0.39, Arm D +0.023 (at the floor). The defect is **point-like discontinuities at the lattice points, carrying forbidden wavenumbers at ~10x the physical floor (1.50 vs 0.14), plus an elevated floor everywhere else (~3.5x)** (D108(f)). This is now a standing metric. It is **not a proxy for accuracy**: it ranks C92 worse than C250 while off-lattice Gate-2 Pearson ranks it better, and Arm D's field is physical yet only 0.36-correlated at 100-200 Hz (D108(g)).
* **Family eval (D111).** Off the lattice it drops by **0.22** (range 0.17-0.28, 5 of 5 rooms). D83's corner-count ordering is unsupported as a field. D84's token cost shrinks (rect -0.11 -> -0.04, L -0.30 -> -0.21): real, but smaller than stated.
* **D80's corpus-size decline** (C92 0.7545, C250 0.6109, C500 0.5641 off-lattice) is **observed without mechanism**. The offered per-point-count mechanism is refuted: at matched counts, C92 and C250 differ 3.4x with disjoint CIs (D109). It must not be read as "more data hurts". B7 vs D is the test.
* **Material comparison: inverted** (lead paragraph; D106).

## 2. Part A — diagnostics

**A1 (D97).** Summarised in the second paragraph above. Details: spec terms carry 100% of the net step at 60K (spec_real 0.673, spec_imag 0.329). Bins >= 100 Hz hold 74% of the remaining loss VALUE, but only 9.9% of the gradient norm and 0.9% of the projection. Their gradient is both smaller than proportional and largely orthogonal to the step. Output-gradient share >= 100 Hz is exactly 0.6672 (L1 gives 401/601), so the concentration lives in the Jacobian. Inference, labelled: the network outputs ~1000x more at DC.

**A2, baseline curve (C250 @ 60K, 3 frozen dense shapes, every 0.5 Hz bin, 25-300 Hz).** Off / on the lattice, mean over +-2.5 Hz:

| | 50 Hz | 100 Hz | 150 Hz | 200 Hz | 250 Hz |
|---|---|---|---|---|---|
| C250, **off-lattice** | **+0.614** | **+0.209** | **+0.064** | **+0.257** | **+0.105** |
| C250, on-lattice (diagnostic) | +0.854 | +0.547 | +0.419 | +0.455 | +0.296 |
| Arm D, **off-lattice** (reference) | +0.805 | +0.427 | +0.401 | +0.221 | +0.254 |
| C92, **off-lattice** (reference) | +0.771 | +0.224 | +0.088 | +0.237 | +0.060 |

Band LSD (C250, off-lattice): 3.89 / 7.13 / 6.93 dB at 0-100 / 100-200 / 200-300 Hz. RIR Pearson (C250): Gate-2 style 0.9989, modal 0.8624.

**A3 (D94): ground truth is valid across the whole band.** dx 0.02 vs 0.01 on the in-slab A2 room. The 5-Hz-window dB Pearson never falls below 0.99; the worst window is **0.99919 at 265 Hz**. No GT-validity line falls inside 0-300 Hz.

## 3. Part B — arms

PENDING. Reading order (manager): **B7, B6** first; **B5** third; **B1 / B1b / B2 / B3** last and conditioned on pooling. **B4** is a supervision-density diagnostic only, and its on-lattice number appears in no headline. For every arm: off-lattice first, with its CI, and the excess supra-physical fraction beside it. No Gate-2 verdict without its bootstrap interval. Per D100(c): if B6's 800-receiver Gate-2 falls while its off-lattice rises, the arm has succeeded.

## 4. Decisions

D92 pre-registration · D93 B4 needs FDTD · D94 GT-validity criterion · D95 B2 null · D96 B1 weights / DC risk · D97 A1 result · D98 energy pre-registration · D99 lattice memorisation · D100 B6 · D101 B5 · D102 Gate-2 correction · D103 B4 OOM · D104 pooling · D105 bootstrap CIs · D106 material inversion · D107/D108 k-space · D109 D80 mechanism refuted · D110 paper framing · D111 family bound · D112 pre-registration of the pooling-on-family / token-invariance tests.
