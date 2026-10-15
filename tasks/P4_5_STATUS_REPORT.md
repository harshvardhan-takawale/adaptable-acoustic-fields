# P4-5 status report for the manager — 2026-10-15, 11:55 EDT

**State: nothing is running.** Every arm launched in this round has trained to its endpoint and has its final read recorded (D148–D159). There are no SLURM jobs, no evaluation watchers and no pending launches. GitHub `main` matches the working tree; every link below resolves.

Base for every link: `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/`

- Decisions (D148–D159 are this round): https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md
- Chunk write-up (section 5 = this round): https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_P4_5_RESULTS.md
- Re-orientation doc: https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/CONTEXT_FOR_MANAGER.md
- Open questions (Q23 still open): https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md

**Protocol for every number below** (unless marked):
- Dense receivers on the 15 frozen test shapes, OFF the training lattice (Rule 11).
- 95% CIs come from a shape bootstrap (10k resamples, seed 20261013). Differences are paired.
- "Gate-2" is the frozen 6-mode in-slab spatial Pearson; "gap" is on-lattice minus off-lattice.
- Band values are off-lattice per-bin column Pearson of dB, averaged over the band.
- Following the standing instruction, the 200–300 Hz band value is quoted, not the 295–300 Hz edge point.

---

## 1. Running now

**Nothing.**
- **Training:** the last job, B5r, finished at 60K at 09:56.
- **Evaluation:** the last evaluation jobs (B5r 60K speckle and 2 cm figure) finished at 11:00.
- **Watchers:** all exited after launching their items.
- **Health monitor:** stopped deliberately, because nothing remains to watch.

**Resources.**
- **Storage is tight:** the project export has **32 GB free of 500 GB (94%)**. The quota incident of D143 happened at 0 GB. Cleanup is advisable before the next launch (section 7).
- **Cluster:** no tron or scavenger slots are held.

---

## 2. Finished — shape-editing arms, final reads at 60K

| Arm | Recipe (one change from its parent) | Gate-2 OFF in-slab [CI] | Gap (in) | Test r 25–100 / 100–200 / **200–300** | Train-room r / LSD, 200–300 | Decision |
|---|---|---|---|---|---|---|
| **B5** (default) | C250 + DC head (0–20 Hz) | 0.924 [0.909, 0.939] | +0.001 | 0.887 / 0.631 / **0.467** | 0.648 / 3.83 dB | D136–D137 |
| **PB3** | B5 + 3 per-band signal heads (173 / 245 Hz) | **0.931 [0.921, 0.942]** | −0.002 | 0.895 / 0.647 / **0.496** | **0.772 / 2.88** | D157 |
| **W3** | B5 + one 896-wide signal MLP (parameter-matched to PB3) | 0.921 [0.910, 0.931] | +0.002 | 0.897 / 0.646 / 0.472 | 0.746 / 3.12 | D157 |
| **B5r** (a1) | B5 on B6's per-room receiver corpus | 0.921 [0.907, 0.934] | −0.001 | 0.881 / 0.627 / 0.464 | 0.661 / 3.74 | D156, D159 |
| **B9b** | B1 weights on 0–10 Hz only | 0.924 [0.915, 0.935] | +0.002 | 0.885 / 0.634 / 0.478 | — | D151 |
| **B9** | bin 0 alone suppressed | 0.922 [0.911, 0.931] | +0.001 | 0.883 / 0.618 / 0.450 | — | D151 |
| **B4** | C250 + 4× receivers | 0.922 [0.913, 0.930] | +0.001 | 0.890 / 0.622 / 0.442 | 0.527 / 4.59 | D158 |
| **B1** | C250 + 1/RMS spec weights | 0.895 [0.882, 0.908] | −0.001 | 0.854 / 0.609 / 0.448 | 0.727 / 3.32 | D158 |
| **B10** | B4 + B1 weights | 0.870 [0.858, 0.882] | −0.000 | 0.848 / 0.602 / 0.443 | 0.722 / 3.38 | D158 |
| **B8** | B7 (mean pooling) + B1 weights | 0.870 [0.854, 0.882] | −0.000 | 0.844 / 0.596 / 0.427 | — | D151 |
| **B6** | C250 on per-room receivers (no head) | 0.824 [0.794, 0.853] AT THRESHOLD | −0.006 | 0.770 / 0.363 / 0.267 | — | D152, D159 |
| **C250r** | replica of C250 | 0.775 [0.741, 0.812] AT THRESHOLD | **+0.099** | 0.648 / 0.162 / 0.106 | — | D151 |
| C250 (published reference) | baseline | 0.611 [0.580, 0.656] FAIL | +0.230 | 0.512 / 0.149 / 0.118 | — | D105, D151 |

**Physics checks on the DC-fixed arms at 60K.**
- **Full-band RIR:** 0.9997–0.9998 for every one.
- **Off-lattice modal RIR:** 0.922 (B10) to 0.939 (PB3).
- **Lattice gap:** none of these arms memorises; every gap CI includes or hugs 0.
- Files: [RISK_METRICS_PB3_W3_60K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/RISK_METRICS_PB3_W3_60K.json), [RISK_METRICS_B10_60K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/RISK_METRICS_B10_60K.json), [RISK_METRICS_B5r_60K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/RISK_METRICS_B5r_60K.json), [GAP_CI_PB3_W3_60K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/GAP_CI_PB3_W3_60K.json), [GAP_CI_B10_60K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/GAP_CI_B10_60K.json), [GAP_CI_B5r_60K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/GAP_CI_B5r_60K.json), [GAP_CI_60K_round3.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/gate2_dense/GAP_CI_60K_round3.json).

### The four findings that matter for planning

1. **At 200–300 Hz the deficit is now transfer, not fit (D157).**
   - On training rooms, extra signal-pathway capacity raises fit: r 0.648 → 0.746 (W3) and 0.772 (PB3).
   - Only band separation carries that into unseen rooms: PB3 − B5 +0.029 [+0.018, +0.041], while W3 − B5 is +0.005 (not significant).
   - PB3 − W3 = +0.024 [+0.013, +0.035], stable at 20K, 30K and 60K.
   - The fit–transfer gap widened from 0.18 to 0.28. This revises D145's "~70% fit" for these recipes.
2. **Receiver density only keeps paying without B1's weights (D158).**
   - From 30K to 60K at 200–300 Hz: B4 +0.054 [+0.038, +0.068]; B1 −0.013; B10 −0.016.
   - B10 equals B1 in every band (density + B1's loss is a substitute, not a complement).
3. **The DC fix is settled.** B5 (DC head) and B9b (0–10 Hz suppression) are tied at 0.924. B9b is slightly ahead at 200–300 Hz (+0.011 [+0.004, +0.017]).
4. **C250r does not reproduce C250 (D151).**
   - In-slab score 0.775 against 0.611; gap +0.099 against +0.230.
   - The whole difference is at 25–100 Hz (+0.137). At 200–300 Hz the two agree within 0.012, so the 200 Hz tail criterion stands.
   - A memorising baseline carries run-to-run variance of about 0.16 in-slab; non-memorising arms replicate within 0.002.

Per-arm training-room fits: `outputs/p4_5/train_offlat_fit/<ARM>_<iter>.json`, e.g. [PB3_60K](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/train_offlat_fit/PB3_60K.json), [B5r_60K](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/train_offlat_fit/B5r_60K.json).

---

## 3. Finished — doorway (Track B corpus)

All rows are scored on 72 unseen test configurations (6 geometries).

| Model | Pooled slope (GT 7.61) | Slope / GT | r² (GT 0.948) | \|level-difference error\| at a = 0.15 m / 1.0 m / open | Held-out RMS residual (GT 0.68) | Field-only A1 share 0–10 / 10–20 / ≥100 Hz |
|---|---|---|---|---|---|---|
| Track B @20K (published) | 2.46 | 0.32 | 0.172 | 7.72 / 3.32 / 0.80 dB | 2.30 dB | 0.810 / 0.072 / 0.055 (no head) |
| **TBdc** @14K | 4.57 | 0.60 | 0.757 | 4.82 / 1.34 / 0.42 | 0.67 | 0.033 / 0.197 / 0.305 |
| **TBdc** @20K (matched pair) | 4.57 | 0.60 | 0.739 | 4.86 / 1.43 / 0.36 | 0.79 | 0.038 / 0.117 / 0.340 |
| **TBdc** @30K | **4.75** | **0.62** | 0.746 | 4.77 / 1.41 / 0.39 | 0.85 | 0.140 / 0.135 / 0.343 |
| TB9b @20K | 3.92 | 0.52 | 0.632 | 5.39 / 1.35 / 0.88 | 0.86 | 0.084 / 0.176 / 0.277 (trained objective) |
| TB9b @30K | 3.92 | 0.51 | 0.604 | 5.66 / 1.46 / 0.57 | 1.01 | 0.069 / 0.203 / 0.299 |

- **The DC head is the doorway recipe.** It halves the level-difference error at every aperture and doubles the slope ratio (0.32 → 0.62). It plateaued by 14–20K; 16K more iterations bought +0.18 of slope.
- **TB9b** (0–10 Hz suppression, no head) is weaker. It leaves more of the room model's update in 10–20 Hz, the band where narrow doorways break DC uniformity (D146).
- **The two-region DC head is not built.** Its gate was a field-only 10–20 Hz share ≥ 0.15 at 30K, and TBdc reads 0.135. It was close, and the 0–10 Hz share rose late, so the combined 0–20 Hz field share is 0.275. Flagged in D155(b), not acted on.
- **Why TBdc and TB9b were resumed.** Their original early stops used validation on a held-out row of the same 8×8 grid, with a jittering ray fan (trainer bug, D150(d)). Both were resumed to 30K.
- Files: [DELIVERABLE.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/trackB_dc_eval/DELIVERABLE.md) (per-aperture table), [TBdc @30K EVAL.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/trackB_eval_TBdc_30K/EVAL.json), [TBdc @20K EVAL.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/trackB_eval_TBdc_20K/EVAL.json), [TB9b @30K EVAL.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/trackB_eval_TB9b_30K/EVAL.json), [A1 TBdc @30K field-only](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/diagnostics/a1_TBdc_30K_fieldonly_C250_60K.json), [A1 TB9b @30K](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/diagnostics/a1_TB9b_30K_C250_60K.json).

---

## 4. Finished — speckle thread (D150 pre-registration, D152 diagnostics, D156 / D159 a1)

**What the speckle is.** Supra-physical content (|k| > 1.5 k0):
- It is under 1% of the field's energy, at the 2–4 cm scale, and sits in the loud regions (+0.4–0.5 dB per neighbour step).
- B5's nulls are shallower than FDTD's (−2 dB per step), not amplified.
- The 64×64 test grid (about 8 cm spacing) aliases it away. B5's d1 is 0.97 on that grid, against 1.45 / 1.27 at 2 cm at 200 / 280 Hz.

**What removes it.**
- **Physical projection at render time (least-squares band-limit) removes all of it.** 2 cm d1 drops to 0.89 / 0.94, and FDTD's own d1 stays at 0.99 / 1.00, so nodal lines are intact. Accuracy rises slightly: r +0.0023 [+0.0017, +0.0030].
- **Nothing else measured removes it:**
  - hash-level masking costs 0.12–0.24 r;
  - fan-rotation averaging removes about a fifth of it;
  - per-room supervision (B5r) does not reduce it at 60K.

**Accuracy is lost elsewhere.** B5's 200–300 Hz error is a wrong pattern at the field's own scale.

| Diagnostic | Result | File |
|---|---|---|
| d1 speckle ratio + quintiles (B5, C250, B6) | B5 0.968; Q1 → Q5 0.61 / 1.09 / 1.28 / 1.42 / 1.46 | [B5_60K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/speckle/B5_60K.json), [B6_60K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/speckle/B6_60K.json), [P4-4_C250.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/speckle/P4-4_C250.json) |
| d2 fine-scale coherence | B5 \|c\| 0.455; fine-scale energy 2.55× FDTD | same files |
| d3 hash-level masking | levels ≥ 6 masked: r −0.235 / −0.204 / −0.124 → **a2 NEGATIVE** | [B5_60K_lv6.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/speckle/B5_60K_lv6.json) |
| d4 fan-rotation average | d1 −0.054, r +0.003 (a minor aliasing part) | [B5_60K_rotavg.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/speckle/B5_60K_rotavg.json) |
| d5 physical projection | r +0.0023; 2 cm d1 1.45 → 0.89 | [d5_d1_after_projection.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/speckle/d5_d1_after_projection.json) |
| d6 2 cm render, shape 1000 (FDTD exact to 2e-10) | B5 1.45 / 1.27; B5r 1.54 / 1.25 | [B5 figure](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/speckle/d6_shape1000_2cm.png), [B5r figure](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/speckle/d6_shape1000_2cm_B5r_60K.png) |
| a1 B5r vs B5 @60K | accuracy equal; coherence +0.071 [+0.051, +0.092]; speckle not reduced | [B5r_60K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/speckle/B5r_60K.json) |
| PB3 / W3 speckle @60K | fine-scale energy 2.15× / 1.84× FDTD (B5 2.55×); no added speckle | [PB3_60K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/speckle/PB3_60K.json), [W3_60K.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_5/speckle/W3_60K.json) |

---

## 5. Pre-registration scorecard (this round)

| Pre-registration | Outcome |
|---|---|
| D139 — TBdc moves the coupling slope toward 7.61 and r² far above 0.172 | **HOLDS** (D147); confirmed at the matched 20K and at 30K (D153, D155) |
| D144 — PB3: "band separation is the lever" | **NOT MET** — (i) training-room fit holds, (ii) test r ≥ 0.52 fails (0.496), (iii) W3 recovers < ½ of PB3's gain: splits (true on transfer, false on fit) (D157) |
| D137 — B10 reset prediction 0.38–0.42 at 295–300 Hz | **IN RANGE** — 0.380 [0.319, 0.440]; the band value to quote is 0.443 (D158) |
| D150 — a3 gate (speckle concentrated in the lowest-amplitude quintile) | **FAIL** — Q1 has the smallest ratio; not run (D152) |
| D150 — a2 gate (level masking cuts speckle at ≤ 0.02 r cost) | **NEGATIVE** — r cost 0.12–0.24; not run (D152). A trained 6-level cap is untested, not refuted |
| D150 — a1 prediction (lowest speckle ratio, accuracy ≥ B5) | **Accuracy HOLDS** (equal); **speckle FAILS** (D159) |
| D150 — Helmholtz gate (d2 \|c\| ≤ 0.1 AND a1/a2 do not remove speckle) | Condition 1 **FAILS** (\|c\| 0.46–0.53); condition 2 met → **not queued** (D159) |
| D150(g) — two-region DC head (TBdc @30K field-only 10–20 Hz ≥ 0.15) | **NOT MET** — 0.135; flagged as close (D155) |

---

## 6. Corrections to the record this round

- **D149's "Track B @14K" A1 row was Track B2**, a different architecture (D150(f)). The matched baseline is Track B @20K; Track B has no 14K checkpoint.
- **C250r is not a stand-in for C250** below 100 Hz or in-slab (D151).
- **The A1 script decomposed the baseline objective for spec-weighted arms** (D153(d)). Only TB9b @20K was ever affected, and it was rerun on its trained objective. The script now applies the checkpoint's `loss_spec_bin_weights` by default.
- **"S4 shows the right amount of fine-scale energy" was mask-edge leakage.** Measured mask-aware, B5 has 2.5× FDTD's (D152(c)).
- **"B6 has the lowest excess fine-scale energy (−0.023)" is B6 @20K.** B6 @60K reads +0.030 (D159).
- **Trainer bug, found and not yet fixed:** `validate()` in `aaf/train/multi_room_2d_mat.py` never calls `renderer.eval()`, so the trainer's val LSD uses a jittering ray fan (D150(d)). No reported number depends on it; P4-5 shape arms run with early stopping off. The fix is deferred to chunk end so that running arms log consistently. There are none running now, so it can be fixed whenever you approve.

---

## 7. Awaiting the manager's decision (nothing launched)

1. **Density + DC head.** This is B4's corpus with B5's head, and is untested. B4 is the only arm still climbing at 60K (+0.054 at 200–300 Hz), and B1's weights cancel that climb (D158). Config work only; no new code.
2. **The 200–300 Hz ceiling is now transfer-limited** (fit–transfer gap 0.28, D157). Measured transfer levers: band separation (+0.029) and, per D75, corpus density along the shape parameters. PB3 + density, or PB3 + B9b, are natural combinations; none has been run.
3. **Speckle.** Recommend the physical projection at render time for figures and deliverables (free and lossless on FDTD). Whether to add a training-time physics term (Helmholtz) is your call; its pre-registered gate does not fire.
4. **A 6-level hash cap trained from scratch** (`P4_5_B5L6.yaml`, ready). Untested, because the inference-time proxy was confounded by input distribution shift.
5. **Two-region DC head** (near-miss, D155(b)). The design is recorded; it is not implemented.
6. **Q23 — the "8×" point of the density sweep** (OPEN_QUESTIONS.md).
7. **Housekeeping before any new launch.** 32 GB is free. Regenerable, gitignored candidates (their results are committed as JSON):
   - dense 15-shape render caches: 25 GB, 51 files (`outputs/p4_5/gate2_dense/cache/`);
   - A2 render caches: 9.2 GB (`outputs/p4_5/eval/cache/`);
   - speckle projection cache: 1.3 GB;
   - older checkpoints of finished arms: 57 GB under `outputs/p4_5/arms/`, e.g. B5r keeps 30 checkpoints at 6.8 GB.

   Nothing has been deleted; this needs your or the user's go-ahead.

**Ready but unlaunched configs:**
- [P4_5_B5F.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_5_B5F.yaml) — floored L_amp, gate failed;
- [P4_5_B5L6.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_5_B5L6.yaml) — 6-level hash cap.

**Code added this round** (32 tests across `tests/test_p4_5_{speckle,amp_floor,pb3,hf_arms}.py` pass, in two GPU runs):
- [scripts/p4_5_speckle.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_5_speckle.py)
- [scripts/p4_5_speckle_render.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_5_speckle_render.py)
- [scripts/p4_5_speckle_2cm.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_5_speckle_2cm.py)
- [scripts/p4_5_speckle_lp_d1.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_5_speckle_lp_d1.py)
- [scripts/p4_5_doorway_deliverable.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_5_doorway_deliverable.py)
- [scripts/p4_5_a1_bin_gradient.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_5_a1_bin_gradient.py) (now `--objective trained`)
- [tests/test_p4_5_speckle.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tests/test_p4_5_speckle.py)
- [tests/test_p4_5_amp_floor.py](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tests/test_p4_5_amp_floor.py)

The trainer change is the floored-L_amp option: default off, branch-guarded, and bit-identical by test.
