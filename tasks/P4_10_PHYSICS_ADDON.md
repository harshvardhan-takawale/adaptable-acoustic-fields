From: the manager. To: the coding agent, relayed by Amisha. 2026-11-24, about 08:45 ET; revised 08:55 (E3 numbers; the causality measure in P-2; a note in the expectation). Add-on to the meeting-day brief.

# Physics add-on: one quick regulariser run and a zero-training physics audit

A collaborator asked to see the physics-regulariser idea tried. R1-P was gated out (D205), so nothing was trained. Today we run one short, controlled fine-tune, plus a cheap audit. Record this as one D-entry before any number.
* This is an exploratory re-registration. D205's verdict stands: the pre-registered R1-P is not launched.
* The new target comes from 0d itself. R0's band-limited residual was 2.8x the truth's at 25-100 Hz. That points to a non-physical low-frequency component that the band-limit does not remove. E3 points the same way: at 63 Hz the room-common part of the late energy is 2.2x the truth's for R0, 2.4x for R1, and 1.9-3.2x across all seven models.
* Same cut-off as the brief: everything posted by 14:30 ET. Later reads are marked "after the cut-off".

## P-1. R1-P(t): a 10K fine-tune with the mean-value penalty, with a matched control

**The penalty:**
* D204's single-radius stencil:
  * r = min(1.2 / k, 0.35 m); 8 circle points under a random rotation.
  * Centres drawn fresh each step, never on the receiver lattice. At least r + 0.05 m from every wall and r + 0.3 m from the source.
  * Bins stratified over the three bands, skipping the DC-head bins (< 41).
* Normalisation per A13: the RMS residual divided by the RMS of |p| at the centres, NOT detached.
* Training rooms only today. No unlabelled edited rooms, so no collapse monitor is needed: the single-radius form penalises constants.
* Centres per step: as many as keep step-time overhead at or under 60%. Record the number.
* Weight: calibrated at step 0 so the term is about 20% of the total loss, then fixed and checkpointed.

**The runs:**
* **R1-P(t):** R1 @150K plus the penalty.
* **R1-ctl:** the same start, schedule and batch stream, with no penalty.
* **Schedule:** 10K steps each. Pick the LR: either constant at R1's final LR, or a short restart of CTL's closed form over 10K steps. Record the choice before the first number.
* **Rule-8:** with weight 0, R1-P(t) reproduces R1-ctl bit for bit.

**Reads** at +2.5K, +5K, +7.5K and +10K. Each read is paired against R1-ctl at the same step:
1. Static HF and UF, with CIs.
2. The single-radius residual at held-out centres on the 15 in-family rooms, per band, beside the truth's (from 0d).
3. The band-limited residual over the truth's at 25-100 Hz.
4. E3's 63 Hz room-common ratio, plus D1's floor and EDT error per octave (in-family).
5. The raw finite-difference direction score (I5's scorer, h = 4 cm), per band.
6. Bypass check: the residual at held-out centres against the training centres (should be at most 1.5x).

**Stop rule:** HF or any UF band more than 0.02 below R1-ctl, with the CI entirely below zero, at any read.

**Recorded expectation (manager):**
* The residual falls toward the truth's.
* Static r stays within 0.01 of the control.
* Raw direction improves most at 25-100 Hz.
* EDT does not change, because its main cause is sharing (E1).
* Whether the 63 Hz room-common energy and floor fall is the open question.
  * The penalty sees a spatially uniform offset only through 1 - J0(kr). At 63 Hz with r = 0.35 m, that is about 4% of the offset.
  * If the floor falls anyway, the excess is not a pure offset.
  * If it does not fall, the next variant (P4-11, not today) uses r = 1.2/k at low frequency, about 1 m at 63 Hz. That sees an offset 8x more strongly.

## P-2. Physics audit, zero training (about 1 GPU-hour, mostly reuse)

For R0, R1, R1-S and R1-M, against the truth, on the 15 in-family rooms:
* **Wave equation:** the single-radius mean-value residual per band, raw and band-limited (0d machinery, held-out centres).
* **Reciprocity:** swapped against normal r per band (RS3's measure; compute it for R1-S and R1-M).
* **Causality:** on the full 25-300 Hz band-limited IR.
  * Not per octave. An octave filter at 63 Hz smears the direct sound by about 20 ms, which is longer than the window.
  * Use receivers at least 3 m from the source, so the window [0, t_direct - 5 ms] is at least 3.7 ms long.
  * The spectrum is periodic, so the late floor wraps into this window. Subtract it:
    * early excess = (energy in the window) - (window length x energy per ms in the last 0.3 s of the 2 s period).
    * Report it divided by the direct-sound energy (t_direct +/- 5 ms), model against truth.
  * This tests the revisit condition of the 2026-05-09 decision that left out INFER's Kramers-Kronig term: "revisit if reconstructions show non-causal phase artifacts".
    * INFER's KK term enforces causality on the sigma + j beta attenuation, which our model inherits from INFER.
    * The condition is met if the model's early excess is at least 2x the truth's, with the 95% CI over rooms of (model - 2 x truth) above zero.
    * Report the result. There is no KK run today.

**Figure** "How physical are our fields?": one panel per constraint, model against truth, in plain-language labels.

## Priority
* **GPUs:** P-1 needs two GPUs for about 4 h after about 1 h of build, so launch by about 10:00.
* **If GPUs are short:** delay E-C(c), the R1 per-room fine-tune, first. Do not stop R1-SM, E-B, E-C(a, b) or the V-figures.
* **P-2:** runs on the eval watcher between reads.
