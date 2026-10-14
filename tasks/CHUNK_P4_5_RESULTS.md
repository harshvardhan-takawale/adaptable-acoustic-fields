# Chunk P4-5 — High-frequency diagnosis and fixes: RESULTS

**Status: IN PROGRESS (2026-10-14).** Diagnostics A1-A3 and the record corrections are complete. The arms are training: B1-B4 on tron, B1b/B5/B6/B7 on scavenger. 30K evaluations land 03:00-06:00 and 60K ~10 h later. Sections marked PENDING fill in as each lands. Decisions D92-D134 in [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md).

---

**Below 100 Hz, measured as a field on unseen rooms, shape editing with the DC fix reaches 1.6-1.9 dB band LSD. That is ahead of the existing FDTD material-editing models (best 3.39 dB [2.79, 3.99]) and at parity with the ISM material model Arm C (1.57 [1.31, 1.85]) (D134).** Off-lattice, 15 frozen shapes: B5 @30K 1.58 [1.42, 1.75]; B9b @30K 1.70 [1.52, 1.91]; B4 @60K 1.88 [1.73, 2.04]; B1 @60K 1.92 [1.74, 2.12]. The baseline recipe is at 3.67, which is parity with the FDTD material models. The ISM T-geo model (0.99 [0.65, 1.41]) remains ahead. The comparison is between pipelines: the FDTD material models were trained before the DC finding, so the task-level statement needs material editing re-trained with the fix. The earlier apparent parity (D99(h)) compared a 0-300 Hz single-receiver material figure with on-lattice shape numbers (D106, D127).

**The chunk's question, answered in the spec's terms.** The pre-registered prediction holds as scored: bins >= 100 Hz carry **0.9%** of the parameter-gradient projection at C250 @ 60K, against "< 15%" (D97). The mechanism it named is refuted. At initialisation the same share is **65.7%**, the uniform 66.7% that an L1 loss gives. The starvation EMERGES in training, and at 60K the 0-10 Hz band carries ~93% of the step. The baseline (C250 @ 60K, dense, OFF the training lattice, mean over +-2.5 Hz) scores spatial Pearson **+0.209 at 100 Hz and +0.257 at 200 Hz**. On the lattice it scores +0.547 and +0.455, and the gap between those is memorisation (D99), not acoustics. **Which arm moved the tail most: provisionally B1** (manager's verdict: "B1 is the result of the chunk"). At 30K it scores off-lattice r 0.682 at 100 Hz and 0.451 at 200 Hz against the baseline's 0.209 / 0.257, with dense Gate-2 off-lattice in-slab **0.896 [0.881, 0.914]** and gap -0.002, the project's first unambiguous Gate-2 pass as a field. The pre-registered criterion (arm minus baseline at 200 Hz, off-lattice, beyond the receiver-bootstrap 95% CI of the difference) is scored at 60K.

**The bigger finding was not about frequency.** Every P4 room places its receivers on one absolute lattice, x, y = 0.12 + 0.06 k. A position-indexed encoding (the hash grid) is a storage channel. On that fixed lattice, some conditioning schemes let the network use it for memorisation before physics (D99, D110, amended by D115). The standard protocol of the field, held-out positions drawn from the training measurement grid, cannot detect this. Every Gate-2 verdict, sweep curve and family number in P4 was measured that way. Which models memorise is not "attention" as first stated. extent_sum, which has no query-position dependence, memorises too (+0.164; D115). Across all 12 models measured, memorisation appears only where the field network fits the DC term AND pooling is extent-weighted or attention-based. Removing either factor removed it in every case (D116, pre-registered against the running arms).

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
* **Family eval (D111, D113).** Off the lattice FAM drops by **0.226** (range 0.12-0.32, 10 of 10 rooms; two median rooms per family). D83's corner-count ordering is unsupported as a field. **Mean pooling RECOVERS the published family numbers as a field on rect and L**: Arm D off-lattice 0.808 against FAM's published 0.799 and FAM's own field at 0.562, in 4 of 4 rooms. For T, DN and U the family claim is **unresolved, not gone**: Arm D never saw those polygon types. The deciding experiment is a non-attention model trained on the family corpus.
* **Tokenization fragility (D84, D113, D115) is real and general.** Off-lattice re-tokenisation cost, mean |cost| over 6 controls: Arm D 0.299, P4-2 extent_sum 0.221, P4-2 masked_mean 0.181, **FAM 0.087**. The manager's prediction that a mean-pooled set encoder is near-invariant fails by 9x. So does my alternative, that extent-weighting is the partition-consistent pooling (withdrawn, D115). Only FAM, the one model trained on 4-, 6- and 8-token rooms, is robust. Whether that is its corpus or its attention delta is untested.
* **P4-2's four arms re-scored (D114, D115)** after I wrongly recorded their checkpoints as deleted. extent_unmasked 0.7916 -> **0.631** off-lattice (gap +0.164); the other three show no gap. The published verdict (all FAIL) stands.
* **D80's corpus-size decline** (C92 0.7545, C250 0.6109, C500 0.5641 off-lattice) is **observed without mechanism**. The offered per-point-count mechanism is refuted: at matched counts, C92 and C250 differ 3.4x with disjoint CIs (D109). The surviving candidate, that harder rooms lean on the lattice more, runs against the family data: the rectangle loses the most in both rooms, and Spearman(reflex count, shift) is +0.33 where the candidate predicts negative (D109(g), D113(d)). It must not be read as "more data hurts". B7 vs D is the test.
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

### Early reads (gap and supra-physical fraction, D112(d) quantities)

Dense 15-shape Gate-2, OFF-lattice in-slab level and on-minus-off gap with paired bootstrap CIs (`GAP_CI_EARLY.json`); 3-shape A2 off-lattice r at 100 / 200 Hz; excess supra-physical fraction at 100-300 Hz.

| arm @ iter | off-lattice in-slab [95% CI] | gap in-slab [95% CI] | r 100 / 200 Hz | excess S | RIR, Gate-2 style / modal |
|---|---|---|---|---|---|
| B7 @6K (mean, 250) | 0.435 [0.381, 0.477] | +0.032 [+0.014, +0.046] | 0.242 / 0.180 | +0.013 | 0.997 / 0.642 |
| B5 @6K (DC head) | 0.852 [0.818, 0.883] | -0.003 [-0.008, +0.003] | — | +0.027 | 0.999 / 0.873 |
| B6 @10K (off-lattice rx) | 0.602 [0.535, 0.662] | +0.009 [+0.002, +0.016] | 0.309 / 0.235 | +0.028 | 0.997 / 0.683 |
| B5 @10K | 0.896 [0.886, 0.902] | -0.004 [-0.010, +0.001] | 0.658 / 0.344 | +0.018 | 0.999 / 0.902 |
| B1 @10K (1/RMS weights) | 0.872 [0.854, 0.888] | -0.004 [-0.010, +0.002] | 0.658 / 0.450 | +0.013 | 0.994 / 0.909 |
| B4 @10K (4x rx, same lattice) | 0.805 [0.786, 0.827] | +0.001 [-0.004, +0.005] | 0.514 / 0.391 | +0.065 | 0.999 / 0.855 |
| B1 @20K | 0.887 [0.873, 0.900] | -0.003 [-0.007, +0.001] | 0.687 / 0.456 | +0.007 | 0.999 / 0.924 |
| B4 @20K | 0.904 [0.884, 0.919] | +0.002 [-0.002, +0.006] | 0.687 / 0.388 | +0.031 | 0.999 / 0.902 |
| *C250 @60K (baseline)* | *0.611 [0.580, 0.656]* | *+0.230 [+0.191, +0.277]* | *0.209 / 0.257* | *+0.272* | *0.999 / 0.862* |

**What the early reads show.** (1) **No attention arm on the shared lattice shows a gap through 20K** (B1, B4, B5). Memorisation, if it comes, comes later, so a 10-12K gap read is not the early signal hoped for, and B6's zero gap at 10K is not yet evidence. (2) B1, B4 and B5 are far ahead of C250's 60K field at 10-20K. B6 and B7, whose training validation tracks C250's (5.65 / 5.34 vs 5.50 dB at 10K), are not. The likely reading is that the three interventions (1/RMS weights, 4x receivers, DC head) genuinely improve the field, rather than that the field later degrades; there is no C250 checkpoint before 60K to check. (3) B5's DC head shows up at 26-300 Hz, where the head outputs exactly zero, so it is the field network that improved, as D97 predicts once the DC no longer takes ~93% of the step.

### 30K (all eight original arms), the manager's reporting standard

OFF-lattice first. Dense Gate-2 in-slab level and gap from 15 frozen shapes with paired, slab-stratified bootstrap CIs (`GAP_CI_30K.json`). High-frequency gap at 100-200 Hz on the same 15 shapes (`HF_GAP.json`). It has a non-memorisation floor of about +0.01 to +0.035, seen in Arm D and in B6, which never trained on the lattice. Excess supra-physical fraction at 100-300 Hz. Off-lattice LSD below 100 Hz on 15 shapes [CI] (`RISK_METRICS.json`). Full-band RIR (all bins, DC included, every off-lattice receiver) and modal RIR. Training validation LSD at 30K; C250 is 4.86 at 30K.

| arm @30K | off-lattice in-slab [CI] | gap in-slab [CI] | HF gap 100-200 | excess S | LSD<100 [CI] | full / modal RIR | val LSD | r @100 / 200 Hz |
|---|---|---|---|---|---|---|---|---|
| **B5** DC head | **0.920 [0.909, 0.931]** | -0.002 [-0.007, +0.003] | +0.014 | +0.009 | **1.58 [1.42, 1.75]** | 0.9998 / **0.932** | 2.95 | 0.706 / 0.356 |
| **B1** 1/RMS weights | **0.896 [0.881, 0.914]** | -0.002 [-0.007, +0.002] | +0.016 | +0.004 | **1.95 [1.79, 2.13]** | 0.9997 / 0.926 | **2.82** | 0.682 / **0.451** |
| B1b calibrated weights | 0.850 [0.825, 0.875] | -0.003 [-0.007, +0.001] | +0.015 | +0.004 | 2.31 [2.11, 2.52] | 0.9996 / 0.916 | 2.96 | 0.696 / 0.415 |
| B4 4x rx *(diagnostic)* | 0.921 [0.907, 0.932] | -0.001 [-0.004, +0.002] | +0.015 | +0.023 | 2.03 [1.89, 2.18] | 0.9997 / 0.918 | 3.76 | 0.749 / 0.414 |
| B6 off-lattice rx | 0.763 [0.724, 0.793] | -0.008 [-0.016, +0.001] | +0.035 | +0.004 | 3.05 [2.91, 3.20] | 0.9994 / 0.796 | 4.98 | 0.342 / 0.336 |
| B7 mean, 250 shapes | 0.664 [0.647, 0.683] | **+0.118 [+0.078, +0.158]** | **+0.244** | **+0.295** | 3.54 [3.41, 3.68] | 0.9992 / 0.819 | 4.86 | 0.229 / 0.228 |
| B2 coarse-to-fine | 0.696 [0.658, 0.739] | **+0.083 [+0.059, +0.106]** | **+0.230** | **+0.207** | 3.15 [3.02, 3.31] | 0.9994 / 0.831 | 4.73 | 0.210 / 0.149 |
| B3 curriculum* | 0.671 [0.641, 0.695] | **+0.135 [+0.120, +0.148]** | +0.147 | +0.042 | 3.59 [3.42, 3.77] | 0.9881 / 0.293 | 9.15 | 0.192 / 0.062 |
| *C250 @60K (baseline)* | *0.611 [0.580, 0.656]* | *+0.230 [+0.191, +0.277]* | *+0.281* | *+0.272* | *3.67 [3.56, 3.79]* | *0.9988 / 0.862* | *4.42 (60K)* | *0.209 / 0.257* |
| *Arm D @60K (reference)* | *0.806 [0.766, 0.845]* | *+0.008* | *+0.017* | *+0.023* | *2.79 [2.52, 3.11]* | *0.9996 / —* | — | *0.427 / 0.221* |

\*B3's 200-300 Hz band is untrained until 35K by design, so it is not judged before 60K.

**What 30K says.**
1. **The DC-gradient interventions win, by two routes.** B1 (DC down-weighted in the gradient) and B5 (DC moved into its own head) both give zero gap, forbidden energy at the floor, material-level LSD below 100 Hz, the best time responses measured, and the high-frequency lift: r at 100 Hz goes from 0.209 to 0.68-0.71. B5 reweights nothing, so the shared factor is the DC, not spectral reweighting (D119(d), D121; holds at 30K, completed at 60K).
2. **Memorisation needs a shared lattice AND the baseline (DC-dominated) loss, and given both it appears in the slow fitters** (D119(b), D117). Every arm whose training validation tracks C250's and that sits on the lattice memorises (B2, B3, B7). Every fast fitter does not (B1, B1b, B4, B5). B6 tracks C250 but trains off the lattice, and is clean. D117's 30K predictions scored **4 of 5**: B7's gap was predicted not to fall and fell from +0.151 to +0.118, within CI, while its high-frequency gap and forbidden energy grew.
3. **Pooling is not protective at 250 shapes** (B7, D117). The 92-shape pooling contrast (D104/D105) holds only under the baseline loss at 92 shapes (D119(c)). B8 (mean pooling + B1's weights) completes the 2x2 at 30K/60K.
4. **B4 is a diagnostic only.** More supervision on the same lattice did not feed memorisation through 30K; D99(g)'s false-positive risk did not materialise.

### Still running

B8 (2x2 fourth cell), B9 (bin-0-only ablation), B9b (near-DC-band ablation) and C250r (replica for C250 @30K); 30K reads ~10:30-11:30. All eight original arms at 60K ~13:00-17:00. The pre-registered "moved the tail" criterion and D116 / D119(d) are completed at 60K.

## 4. Decisions

D92 pre-registration · D93 B4 needs FDTD · D94 GT-validity criterion · D95 B2 null · D96 B1 weights / DC risk · D97 A1 result · D98 energy pre-registration · D99 lattice memorisation · D100 B6 · D101 B5 · D102 Gate-2 correction · D103 B4 OOM · D104 pooling · D105 bootstrap CIs · D106 material inversion · D107/D108 k-space · D109 D80 mechanism refuted · D110 paper framing (amended by D115) · D111 family bound · D112 pre-registration of the pooling-on-family / token-invariance tests · D113 their results · D114 correction (P4-2 checkpoints exist) + pre-registration · D115 extent_sum memorises; D113(c) withdrawn · D116 pre-registration: DC x pooling pattern · D117 B7 memorises; slow-fitter prediction · D118 30K B1-B4 · D119 manager's reading; D110 amended; B5 pre-registered; B8/B9/B9b/C250r · D120 B1 risk metrics, LSD parity, matched val · D121 B5 @30K holds · D122 B7 @30K.
