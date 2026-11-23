# P4-10: the editability scorecard, the source fix, and four single-change arms (PRE-REGISTRATION)

Commit this spec verbatim as the next free D-number before computing any P4-10 number.

You may start the FDTD jobs and the Step 0 engineering at once. Send amendments as A1-An. Launches wait for approval only if an amendment changes a pre-registered bar or kill rule.

## Why this round

The project's end-point is a claim about the representation: the field is editable in latent space. Rooms can be stretched and reshaped, given new materials, have their source moved, and later be joined. P4-9 established three facts that shape this round.
* **Moving the source is broken.** Every corpus fixes the source at (0.5, 1.6). At the second source R0 reads 0.227 / 0.067 / 0.040 in family, below "ignore the move" (0.311 / 0.211 / 0.247). Swaps plus flips help: AFAM keeps 0.499 / 0.398 / 0.412.
* **We have never scored an edit family against re-simulated truth.** The one probe (P4-7 p3, D165(c)) found the change fields of B5 and PB3 uncorrelated with the true one (-0.10 to +0.18) after a corner removal.
* **Physics wins full-room 2D rendering at <= 300 Hz (X2).** 2D is the testbed. This round tests editability, not speed.

This round does three things, in order:
1. Measures R0's editability on six edit families. Eval only; first numbers on Day 1.
2. Fixes the source (R1).
3. Tests four single changes on R1, each aimed at one edit family: stretch (R1-S), materials (R1-M), renderer geometry (R1-C), and a geometry-free physics regulariser (R1-P).

The methodology does not change: R0's recipe, the frozen sets, the same scorer, paired reads.

## Progress contract (standing rule from P4-10 on; record it as its own D-number)

Amisha's requirement: every experiment produces scored numbers at regular intervals. No job runs for days with a verdict only at the end.
* **Reads.** No training run goes more than about 2 hours of wall time without a scored read. Every run is read at +5K (smoke), +10K, +20K and +30K steps.
* **Latency.** Every read lands in the progress board within 1 hour of its checkpoint. Keep one or two GPUs as eval watchers.
* **Stopping.** Every run has a kill rule (at +10K; +20K for R1-C) and a pre-registered extension rule at +30K. Nothing else extends a run. No run goes past +50K this round.
* **Eval-only work ships first.** Post each scorecard family as it finishes. Do not wait for the whole scorecard.
* **Daily digest.** At 09:00 ET, a dated entry at the top of the status report: three to five plain sentences plus the board's Table B. Amisha must be able to forward it as is.
* **Slips.** If a read slips (pre-emption, queue), the board says so and gives the new ETA.

**Progress board.** `outputs/p4_10/progress/board.md` and `board.json`, regenerated after every read:
* What is running: arm, step, GPU, ETA of the next read.
* Table A, static accuracy per arm and read: HF and UF per band, delta vs the arm's reference with 95% CI.
* Table B, each arm's primary metric per read: level, delta vs reference [CI], the bar, status (running / killed / extended / final).
* Table C, the latest scorecard headline per model and family.
* F2: trajectories (each arm's primary metric, HF and UF vs step, with R0 / CTL / RCP marks), redrawn at every read.

## Conventions (unchanged)

* Off-lattice everywhere (Rule 11). HF = in-family 200-300 Hz r on the 15 test shapes. UF = unseen r per band on the 16 rooms (8 U + 8 C3).
* Paired bootstrap over rooms, 10k resamples, seed 20261013. Every table prints the reference's absolute level (Rule 12).
* The D160 standard report for every trained arm.
* New code is default-off and Rule-8 tested. New YAML keys go in the resume guard.
* Name the GPU type in `--gres`. Keep a checkpoint every 2K steps.
* Adopt `aaf_sm89` for all P4-10 training and eval (S-4: parity exact, 1.33x faster).
* Add every new PNG to the `.gitignore` allowlist. Check every raw link for HTTP 200.

## Finish P4-9 (does not block P4-10)

* Record DIR-G @150K and RCP @150K as pre-registered.
* **S2X, new manager stop rule.** Continue past 180K only if HF @180K >= 0.527, the linear pace to 0.56 at 240K. Otherwise stop, record 180K as 2a's final read, and give its GPUs to P4-10.

## Closed, do not rerun

* NMF / NMF-P (N1 and N2 fail). DIR (E2 fails under D188); DIR-G closes on its own 150K read. PL1 (G-BC fails). CTL beyond 150K.
* LS1 / LS2 / LS3. The hash-level cap (g4). Receiver density (G3). Wall sigma-contrast injection (P4-3 Arm S).

**Reopened for a different target:**
* **Reciprocity swaps.** m2 (D171) closed them for unseen-family accuracy. X3a (D191) showed swaps plus flips largely fix source moves; RCP isolates the swaps. R1 keeps them.
* **An interior Helmholtz regulariser on unlabelled rooms.** S1 (D175(d)) closed it as a fix for unseen-family accuracy, and that stands. R1-P asks a different question (physical raw fields, derivatives, edited rooms) with a different form (the mean-value identity, no derivatives).

---

## Step 0: assets and engineering (Day 0)

### 0a. Edit manifest (freeze before any score)

**Rooms.** The 31 frozen rooms. In-family 15 is the primary stratum; report unseen 16 beside it.

**Geometry edits keep the SW corner at the origin and the source at (0.5, 1.6)**, as in every training room. This separates geometry edits from source moves and from translations.

| family | edit | magnitudes | per room |
|---|---|---|---|
| Stretch-U | every length x s | s = 0.90, 0.95, 1.05, 1.10 | 4 |
| Stretch-A | x-lengths x s, or y-lengths x s | x 0.90, x 1.10, y 0.90, y 1.10 | 4 |
| Wall | east wall (L) or north wall (W) moved; notches keep their parameters | L +/- 0.3 m, W +/- 0.3 m | 4 |
| Notch | the room's largest notch | depth +/- 0.3 m, width +/- 0.3 m | 4 |
| Source | move the source; geometry unchanged | second source (existing sets); third interior source (new) | 2 |
| Material | wall absorption | longest wall alpha 0.30, 0.50; all walls alpha 0.10, 0.30 | 4 |

* **Validity.** Skip a cell if the room has no notch (Notch family), a notch dimension falls below 0.2 m, the room narrows below 1.0 m, or the source comes within 0.5 m of a wall. Log every skip.
* **Third source.** The interior point nearest (0.55 L, 0.5 W) with >= 1.0 m wall clearance and >= 1.5 m from the original and second sources. Freeze it in the manifest.
* **Receivers.** Dense receivers in each edited room by the frozen sets' rule. Edit-delta metrics use common receivers: the original room's off-lattice receivers that lie inside the edited room, >= 0.1 m from its walls. Simulate truth at both sets.
* **Truth.** Dense FDTD for every cell: about 650 runs at ~58 s, on a scavenger CPU array.
* **Scorecard-lite (for periodic reads).** 8 in-family rooms (the 15 sorted by name, every other one from the first) and 8 unseen (4 U + 4 C3, same rule). Magnitudes:
  * Stretch-U 0.90 / 1.10; Stretch-A x 1.10 / y 0.90.
  * Wall L +0.3 / W -0.3; Notch depth +0.3 / width -0.3.
  * Source second / third.
  * Material longest wall 0.50 / all walls 0.30.

### 0b. Training corpora (scavenger CPU arrays)

The MS corpus is the critical path. Build it first.
* **MS (R1).** The 1,000 GX rooms with 4 sources each: (0.5, 1.6) plus 3 drawn uniformly inside the polygon with >= 0.5 m wall clearance, at least one with >= 1.0 m. Same training receivers.
* **MAT (R1-M).** The same rooms, sources and receivers. Per-wall alpha drawn per room, uniform in m = -ln(1 - alpha) over alpha in [0.08, 0.60]. Continuous sampling on the physical axis is the P3-2 lesson. If the FDTD lacks per-wall beta, add it and validate it against a second method on 3 rooms (D54).
* **STR (R1-S).** Two stretched variants per GX room: one Stretch-U (s uniform in [0.85, 1.15], |s - 1| >= 0.03) and one Stretch-A (random axis, same range). SW-anchored. Keep each source's absolute position if it stays >= 0.5 m from the walls; otherwise redraw it by MS's rule.
  * These are simulated, not derived from the scaling law. p2 failed on the FDTD's static compliance term (D165), and the FDTD is cheap (~0.9 CPU-h per 250 rooms, D93).
* **Memory.** Four sources quadruple the targets. Keep targets in host memory and transfer per batch, or subsample receivers for the added sources. Rule-8: batches bit-identical to the preload path on a small corpus. Record the choice and the disk use.

### 0c. Engineering (default-off, Rule-8 tested)

* **Batch stream.** A multi-source batch generator with swaps at RCP's p = 0.25. One deterministic stream that every arm shares. An arm's replaced rows come from the same RNG draws, so reads stay paired.
* **Per-wall m in the tokens.** If R0's tokens already carry m, use it. Otherwise add it zero-initialised. Test: with alpha = 0.15 everywhere, R1-M at step 0 equals R1 exactly.
* **Polygon ray clipping (R1-C).** t_max = the ray's first hit on the room polygon. Tests: on rectangles the render is bit-identical to AABB clipping; no ray sample lies outside the polygon (100 rooms x 1,000 rays).
* **The mean-value regulariser (R1-P),** defined in 2e.
* **The scorecard harness** (metrics in Step 1) **and the progress board.**

### 0d. R1-P calibration (no training; gates the R1-P launch)

Compute the 2e residual on the 15 in-family rooms for FDTD truth (re-run storing 2 cm fields at the sampled bins), R0 raw and R0 band-limited. Use disks clear of walls and the source.
* **Code check.** Truth's normalised residual <= 0.05 in every band. If the room-uniform component pushes it higher, switch to 2e's two-radius form and re-check.
* **Blind to physical content.** R0 band-limited within 2x of truth.
* **Something to remove.** R0 raw >= 3x R0 band-limited, paired CI above zero, in every band. If not, report it; R1-P is not launched.
* The mean-value form measures fine-scale error at its own amplitude. The Laplacian amplifies it by (kappa / k)^2. So do not expect S1's 89x ratios here.

---

## Step 1: the editability scorecard on R0 (eval only; Day 1)

**The edit operator.**
* Geometry: re-encode the edited polygon's tokens.
* Material: change the wall's m in its token.
* Source: change the source input.

Render with the 601-bin path.

**Metrics, per cell and band (25-100 / 100-200 / 200-300 Hz):**
* **Edited-field r and LSD.** The frozen scorer on the edited room's off-lattice receivers.
* **r_delta.** Per-bin spatial Pearson of delta-dB = dB(edited) - dB(original), clipped to +/- 20 dB, on the common receivers; band mean. Reuse p3's change-field code (D165(c)) where it matches.
* **g_delta (Phase 3's edit gain).** LSD(the model's original render vs edited truth) / LSD(the model's edited render vs edited truth). Above 1 means the edit moved the model toward the truth.
* **Mode-shift slope (geometry families).** Take the first 10 peaks of the receiver-averaged power spectrum in 25-150 Hz, Hungarian-matched between original and edited. Report the slope of predicted on true shifts, pooled per family, and the MAE in Hz.
* **Decay (primary for Material, descriptive for the others).**
  * T30, EDT and C50 per octave band (63 / 125 / 250 Hz), from the band-limited RIR at off-lattice receivers >= 1 m from the source, after removing the static compliance offset (D165(b)).
  * Decay-rate edit slope: delta(1/T30) predicted on true.
  * Report against the JNDs: T30 and EDT 5% (ISO 3382-1; measured: RT about 15%, EDT about 18%); C50 1 dB.

**Baselines and references, per cell:**
* **Ignore the edit:** the original room's truth scored against the edited truth on the common receivers.
* **The model's own original render:** the null for g_delta.
* **Naive warp (Stretch-U and Stretch-A only):** the original truth mapped through the geometric warp, at the same frequency.
* **Physics reference, not a baseline to beat:** m1 at 10 cm / 350 Hz on the edited room (0.38 s). Use per-wall beta only if m1 supports it; otherwise omit m1 for Material and say so.

**Summary.** Per family and band: the mean over rooms of each room's average over magnitudes, with paired CIs. Report in-family and unseen separately.

**F1.** A family x band panel of edited-field r and r_delta for R0, with ignore-the-edit and m1 marks. Add one example room per family (truth delta vs R0 delta).

**Also score CTL @150K and RCP @150K on the Source family.** This splits R1's source gain into swaps and data.

**Manager's predictions for R0** (record them; scored, not gating):
* P-a: Stretch-U, Stretch-A and Wall, in family: edited-field r within 0.05 of R0's static in-family r, every band.
* P-b: in family, g_delta > 1 for Stretch-U and Wall at 100-300 Hz.
* P-c: Notch r_delta is below the Stretch and Wall r_delta, in family (p3: right amount, wrong place).
* P-d: Source: R0 is below ignore-the-move in every band (S-3, X3a).
* P-e: Material: g_delta near 1 and decay-rate slope near 0. R0 never saw alpha vary.
* P-f: Unseen: every family at or below R0's static unseen r.

---

## Step 2: training (R1 and four single-change arms)

**Common to every run:**
* **Start.** Warm start from R0 @120K with CTL's closed-form LR restart; 30K steps. All runs share one batch stream.
* **Reads.**
  * +5K smoke read: HF, UF and the primary metric on scorecard-lite.
  * +10K and +20K: the same on scorecard-lite.
  * +30K: the full scorecard.
* **References.** R1 is read against CTL @150K (the same steps from R0). Each arm is read against R1 at the matched step.
* **Harm kill** (+10K; +20K for R1-C): HF or any UF band below the reference by more than 0.05, with the CI entirely below -0.02. Any non-finite value stops the run (D186).
* **Extension rule at +30K.** If the primary bar is missed by <= 0.05 and the primary metric improved by >= 0.02 from +20K to +30K, extend to +50K with reads at +40K and +50K. +50K is final.
* **R1 is never stopped early.** It is every arm's reference. If it fails a harm kill or its learning check, mark the test failed, report it at once, and keep training.

### 2a. R1: the source fix

R0's recipe on MS, with swaps at p = 0.25. Nothing else changes.

Pre-registered (+30K; in family unless stated):
* **RS1 (primary).** Moved-source gap = r(original source) - r(moved source). It is <= 0.10 in every band, at the second AND the third source. R0's gap at the second source is 0.69 / 0.63 / 0.54.
* **RS2.** Unseen 16: moved-source r >= ignore-the-move r + 0.05 in every band, at both sources.
* **RS3.** Reciprocity: swapped-render r >= normal r - 0.05. R0: 0.167 against 0.918 at 25-100 Hz.
* **RC1 (no harm).** At the original source, HF and each UF band >= CTL @150K - 0.02.
* **Learning check (+10K).** In-family second-source r at 25-100 Hz >= 0.38 (R0's 0.227 + 0.15). If it fails, RS1 is marked failed and reported.
* **Prediction.** RS1 holds at the second source. At the third source it holds at 25-100 Hz and misses at 200-300 Hz. RC1 may cost up to 0.03 at 200-300 Hz, because the original source now gets about a quarter of its former exposure (Q25).
* **Board metric.** The in-family moved-source gap, band mean, both sources.
* **Descriptive.** R1 - RCP: what the multi-source data adds beyond swaps.
* **If RS1 holds,** R1 is the meeting model and the base for P4-11, with RC1's cost reported beside it. Score it on the 16-room confirmation set at its final read.

### 2b. R1-S: stretched rooms in training

R1 plus STR rows. In each batch, p = 0.25 of the rows are replaced by the same room's stretched variant, with the same source slot.
* **SS1 (primary).** Stretch-U and Stretch-A, in family: R1-S - R1 on r_delta >= +0.05 with CI > 0, in at least 2 of 3 bands. If R1 already reads r_delta >= 0.85 on these cells, the bar cannot be met; the unseen stretch cells become the primary read, with the same bar.
* **SS2 (descriptive).** Mode-shift slope against R1's (closer to 1 is better); the unseen stretch cells.
* **SC1 (no harm).** HF and UF >= R1 - 0.02.
* **Board metric.** r_delta on the lite stretch cells, band mean.

### 2c. R1-M: per-wall materials

R1 plus MAT labels. p = 0.5 of the rows take the MAT labels for the same room, source and receivers.
* **MM1 (primary).** Material family, in family: decay-rate edit slope in [0.8, 1.2] with Pearson >= 0.8. For reference, the Phase 3 material arms reached bandwidth-edit slopes of 0.87-1.15 with Pearson 0.87-0.90.
* **MM2.** Median |T30 error| on the edited rooms <= 15% (the measured RT JND), in the 63 / 125 / 250 Hz octave bands.
* **MC1 (no harm, alpha = 0.15).** HF and UF >= R1 - 0.02.
* **Learning check (+10K).** Decay-rate slope >= 0.3. Otherwise stop: the material input is not being used.
* **Board metric.** Decay-rate edit slope on the lite cells.

### 2d. R1-C: rays clipped at the room outline

R1 with polygon ray clipping. Nothing else changes.
* **Also read at +0K:** R0's weights with clipped rays, no training. It shows what clipping alone does.
* **CC1 (primary).** Unseen UF 25-100 Hz: R1-C - R1 >= +0.03 with CI > 0. The hypothesis: the renderer carries geometry the encoder does not use. R0 reads 0.468 on unseen rooms with their real shapes and 0.408 given only bounding boxes.
* **CC2.** r_delta on Wall and Notch, in family: R1-C - R1 >= +0.05 with CI > 0, in at least one band below 200 Hz.
* **CN1 (no harm).** HF >= R1 - 0.02 at +30K. The harm kill starts at +20K, because clipping changes notched rooms at step 0.
* **Board metric.** Unseen UF 25-100 Hz.

### 2e. R1-P: geometry-free Helmholtz regulariser (launch only if 0d passes)

R1 plus a mean-value penalty. It needs only k and a conservative interior mask.
* **Identity.** If p satisfies Helmholtz in a disk of radius r around x, the circle mean is M_r[p](x) = J0(kr) p(x).
* **Residual.** R = M_r[p](x) - J0(kr) p(x), with M_r from 8 points under a random rotation.
  * If 0d shows a room-uniform component, use the two-radius form, which cancels constants: R = (M_r1 - p)(J0(k r2) - 1) - (M_r2 - p)(J0(k r1) - 1).
* **Normalisation.** Divide by the RMS of |p| over the step's centres at that bin, detached, so shrinking the field cannot lower the loss.
* **Radii.** r = min(1.2 / k, 0.35 m); r2 = 0.6 r.
* **Centres.** Fresh uniform draws each step, never on the receiver lattice. At least r + 0.05 m from every wall and r + 0.3 m from the source.
* **Bins.** 16 per step, stratified over the three bands. Ray jitter is shared within a stencil.
* **Collocation.** Half in the batch's training rooms. Half in unlabelled edited versions of them: a Stretch-U with s in [0.85, 1.15], or one wall moved +/- 0.3 m.
* **Weight.** Calibrated at step 0 so the term is about 20% of the total loss (DIR's machinery; DIR used 30%), then fixed and checkpointed. Step-time overhead <= 30%; report it.
* **Standing principle.** This is a soft prior, and the standing principle says physics must enter structurally (the G+ bypass). The bypass monitor below is this round's test of that principle.

Pre-registered (+30K, against R1):
* **PP1 (primary, physicality).** Two conditions:
  * The raw fine-scale energy ratio at 200-300 Hz (LS, in-family 15) is <= 1.2x FDTD and <= 0.75x R1's.
  * The 2 cm d1 on shape 1000 lies within [0.9, 1.1] at 200 and 280 Hz. R0 raw reads 1.41 / 1.23.
* **PP2 (derivatives).** Raw finite-difference gradient direction score (I5's scorer, h = 4 cm) >= R1's projected-gradient score - 0.02, every band, in family. R0: raw 0.848 / 0.739 / 0.747, projected 0.943 / 0.844 / 0.803.
* **PC1 (no harm).** HF and UF >= R1 - 0.02.
* **PE1 (exploratory).** r_delta on every geometry family, raw render: R1-P - R1 >= +0.03.
* **Bypass monitor.** The residual at held-out random points stays within 1.5x of the residual at the training centres.
* **Learning check (+10K).** Raw fine-scale energy ratio <= 0.9x R1's. Otherwise stop: the penalty is too weak or bypassed.
* **Prediction.** PP1 holds. PP2 holds at 100-300 Hz and is uncertain at 25-100 Hz. Static r moves by less than 0.01.
* **Board metric.** The raw fine-scale energy ratio at 200-300 Hz.

### 2f. Capture then edit (eval only; lowest priority; Day 3)

Does a measurement fit survive an edit?
* **Fit.** Run M2 (D183 settings, K = 32, seeds 0-2) in the original unseen room.
* **Edit.** Re-encode the edited room, add the fitted code offset (fitted code - encoded original), and keep the fitted FiLM heads.
* **Families.** Stretch-U 0.90 / 1.10 and Wall L +/- 0.3. Add Source (second) for R1 only.
* **Survival** = (fitted + edit - zero-shot edit) / (fitted static - zero-shot static), at 25-100 Hz.
* **Pre-registered.** Survival >= 0.5 for R0 on the geometry families.

---

## Scheduling and priority

* **tron:** R1 first. Everything is read against it.
* **L40S / A6000 pool** (after S2X's stop rule): R1-S and R1-C.
* **Scavenger A5000s:** R1-M and R1-P (checkpoints every 2K for pre-emption), plus one or two eval watchers.
* **CPU scavenger arrays:** all FDTD, the m1 references and the decay metrics.
* **If GPUs are short,** launch in priority order: R1 > R1-S > R1-C > R1-M > R1-P. A late arm still pairs with R1 at matched steps, because the batch stream is deterministic.

**Planned increments** (targets; pre-emption can shift a read, not the order):

| # | Increment | Target |
|---|---|---|
| I0 | Spec recorded. P4-9 reads (DIR-G, RCP, S2X @180K) recorded. Manifest frozen | Day 0 |
| I1 | Corpora and scorecard truth simulated. 0c tests pass. 0d calibration | Day 0 |
| I2 | Scorecard on R0, posted per family (Source first: most of its truth exists); CTL / RCP on Source -> F1 | Day 1 |
| I3 | Launches. +5K smoke reads. R1-C's +0K read | Day 1 |
| I4 | +10K reads; kill and learning checks | Day 1 |
| I5 | +20K reads | Day 1-2 |
| I6 | +30K verdicts and full scorecards -> F2-F7 | Day 2 |
| I7 | Extensions to +50K, if triggered | Day 3 |
| I8 | Capture then edit | Day 3 |
| I9 | Meeting pack, primary table, results doc | Day 3-4 |

## Reporting

* `P4_10_STATUS_REPORT.md`: live log, newest first, with a plain-language headline per entry.
* Also keep: the progress board, the daily digest, DECISIONS, `CONTEXT_FOR_MANAGER.md`, and the primary table (rows for R1 and each arm, plus scorecard columns).
* Meeting figures in `outputs/p4_10/meeting/`, with a FIGURE_MANIFEST.md:
  * F1: R0's scorecard.
  * F2: trajectories (the board's figure).
  * F3: moved-source maps, truth / R0 / RCP / R1, second and third source, 50 and 200 Hz.
  * F4: a stretch edit, truth delta vs R1 delta vs R1-S delta, plus the mode-shift plot.
  * F5: a material edit, Schroeder decays for truth / R1 / R1-M.
  * F6: R1-P, 2 cm speckle and direction maps: R1 raw / R1 band-limited / R1-P raw / truth.
  * F7: the scorecard after: R0 vs R1 vs each arm, in F1's layout.

**Not in this round:** the doorway join (P4-11, on whichever base passes), combining arms, 3D, and Q25's capacity options.
