From: the manager. To: the coding agent, relayed by Amisha. 2026-11-23, evening.

# P4-10 manager rulings: Q26-Q33 and the standing questions

Record this as one D-entry before R1's +30K read. Every ruling here is made before the +30K numbers exist.

## Q27 (time-critical): reading RS1 and SS1 at +30K

* **RS1: the strict reading.** It holds only if all six in-family gaps (two sources x three bands) are <= 0.10.
* **R1's extension:** only if the worst of the six gaps is <= 0.15 AND the mean of the six fell by >= 0.02 from +20K to +30K. Otherwise +30K is final.
* **SS1:** pool Stretch-U and Stretch-A per room, as implemented; 2 of 3 bands.
* **R1-S's extension:** only if the second-best band's R1-S - R1 difference is >= 0.00 (within 0.05 of the bar) AND the band-mean difference rose by >= 0.02 from +20K to +30K.
* **Matched reference:** if any arm extends, R1 continues to the same step.
* **RS2 and RS3** are read as written. Report RS1's six gaps with CIs, and the unseen gaps beside them.

## Q28: base model for P4-11 and the meeting

* **R1 at its final read is the base,** reported with RS1's result and RC1's cost. The reasons:
  * It is the only model that moves the source. At +15K: third source, in family, 0.748 / 0.407 / 0.283 against ignore-the-move 0.285 / 0.231 / 0.177 (lite set). R0 reads 0.235 / 0.099 / 0.048 (full set).
  * Its unseen accuracy at +15K (0.497 / 0.412 / 0.398) already matches S2X @240K (0.495 / 0.403 / 0.398). R1 got there with 30K steps on top of R0; S2X took 240K steps on twice the rooms.
  * Its HF cost against CTL @150K is falling: -0.047, -0.033, -0.028.
* **Fallback, fixed now:** R0 stays the base if, on the lite set, R1's final third-source r is below its +15K value by more than 0.03 in any band, or if its HF cost against CTL @150K exceeds 0.04.
* **Arms:** if an arm passes its primary and its no-harm bar, the P4-11 base is that arm (R1 plus one validated change). Combining two passing arms is P4-11 work.
* **S2X:** record its pass (HF 0.5635, UF 0.4951) and keep it as a reference row. No S2X + R1 combination.

## Q26: decay metrics; release R1-M now

* **Release R1-M** (`scontrol release 8542917`). Its training does not depend on the metric.
* **Approve D200** (the DC compliance gain). It is the zero-mode law, exactly 1 at alpha 0.15, and tested.
* **Re-registration.** T30 on model fields measures the late floor, not the decay. Record this before any R1-M number:
  * **MM1 (primary):** decay-rate edit slope from EDT, i.e. delta(1/EDT) predicted on true. In family, pooled over the Material cells and the 63 / 125 / 250 Hz octaves. The bar is unchanged: slope in [0.8, 1.2], Pearson >= 0.8.
    * Report the T20-based slope beside it. Use only receiver-bands where the model's decay clears its own floor by >= 10 dB below the T20 end point (your option a). Print the dropped counts.
  * **MM2:** median |EDT error| on the edited rooms <= R1-M's own median |EDT error| on the unedited rooms + 5 points. That is, editing adds no decay error. Print beside it (Rule 12):
    * the absolute levels;
    * R0's and R1's unedited errors;
    * the JNDs (5% in ISO 3382-1; about 18% measured for EDT).
  * **Descriptive:** the band-level change slope (room-averaged level change per octave, predicted on true; JND 1 dB), plus g_delta and r_delta on Material.
  * **T30 on model fields** is no longer a headline anywhere. Truth T30 stays.
* **Re-score R0's Material decay** with these definitions and replace the P-e decay line.

## Decay is a model problem, not only a metric problem

The 18:36 diagnosis shows R0's impulse responses decay too fast (EDT 15-28% short, T20 35-59% short) and then sit on a floor near -40 dB. Both exceed the measured JNDs, and every edit inherits them. Eval only, no training:
* **D1, standard-report addition.** Report EDT, masked T20 and the floor level per octave on the original in-family and unseen rooms. Models: R0, CTL, RCP, S2X, R1 and R1-S, each at its final checkpoint. Add these to the D160 standard report from now on.
* **D2, mechanism.** Take the first peaks of the receiver-averaged power spectrum in 25-100 Hz on the 15 in-family rooms. Compare the half-power linewidths, model against truth, using A8's peak code.
  * Broader model peaks would mean the model averages over uncertain mode frequencies. The time domain then shows that averaging as over-damping.
  * Also report the floor's level against the model's per-bin error.
* These feed P4-11. Do not start a decay loss in P4-10.

## Q29: R1-C. No override; R1-C is closed

Rule 10 stands. Below 300 Hz the wavelength (1.1-13.7 m) is at or above the notch size. First-hit ray termination would impose geometric shadows the field does not have, and D77 showed an injected occluder makes the fields worse. The spec's R1-C conflicted with rule 10, and the manager withdraws it. Do not build the clipping code. Q21 closes with it, including the path-targeted sigma residue.

## Q30: R1-P. Keep it, lowest priority

* **Accept A13** (the non-detached ratio). The spec's detached denominator still lets the gradient shrink the field. **Accept A9 and A10.**
* **Add a collapse monitor.** In the unlabelled edited rooms, compute the field's spatial standard deviation over the stencil centres, per bin, divided by R1's at the same latents. It must stay in [0.8, 1.25]; if a read falls outside, stop. The reason: the two-radius form has constants in its null space, so a flat field would zero the penalty.
* **Write 0d now** (CPU FDTD at stencil points, plus about one GPU-hour). Launch only if 0d passes, on the first GPU free after R1-M.

## Q31: environment

Train and score every P4-10 arm in `aaf`. Correct A3's text in the record. `aaf_sm89` stays unused until a training-parity check exists.

## Q32: confirmation set

* Score R1 once at its final checkpoint, beside S2X @240K and R0 @120K, because R1 is the base (Q28). No P4-10 choice is made from that read.
* Freeze a fresh 16-room confirmation set for P4-11 now (same rule, new draws), before any P4-11 number.

## Q33: meeting date

Amisha will answer.

## Board and figures

* Build the board's Table C, the next-read ETA and F2 before the 09:00 digest. They are part of D196.
* Then F1 (R0's scorecard) and the A8 mode-shift slope. F3-F7 follow the final reads.

## Source-move diagnostic (eval plus 32 CPU FDTD runs; after the final reads)

R1's second source (0.5 m from the east wall) trails the interior third source by 0.13 / 0.14 / 0.10 at +15K. Test whether wall clearance is the cause:
* In the 8 in-family lite rooms, place sources on a line at y = 1.6, at 0.5, 0.75, 1.0 and 1.5 m from the east wall.
* Report R1's moved-source r by clearance, and the clearance histogram of the MS training sources.

## Standing questions

* **Q11, Q13, Q14:** close as superseded. They concern Phase 1-2 ISM and 3D targets; the current protocol is 2D FDTD, with per-band r and LSD from 25 Hz.
* **Q12:** remove; it is already closed.
* **Q15:** keep open as a maintenance item. Confirm by grep that no P4-10 metric (scorecard, octave_decay, A8) calls the three functions.
* **Q16:** close. The FDTD's locally reacting per-edge walls (D199) are the wave-based law Q16 asked for, and R1-M answers it.
* **Q17:** defer to the P4-11 join spec.
* **Q18:** close. The headline is per-band r and LSD from 25 Hz, which excludes the compliance term by construction. Floored LSD is retired.
* **Q21:** close, with Q29.
* **Q23:** close. Receiver density was closed in P4-9.
* **Q24:** answer for now. The INR's case is:
  * (i) edits without re-simulation, scored by this scorecard;
  * (ii) a prior for sparse measurements (X1);
  * (iii) 3D and wider bands, where re-simulation is expensive.
  Keep it open as a standing question for 3D.
* **Q25:** keep open. R1's HF cost and its moved-source ceiling above 100 Hz look like the same sharing cap. With the decay work, it is the leading P4-11 candidate.
