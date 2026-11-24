From: the manager. To: the coding agent, relayed by Amisha. 2026-11-23, late evening (after report 2).

## P4-10 manager notes on report 2

Record as one D-entry. Nothing here changes a pre-registered bar or a running job.

### Accepted as reported
* R1's verdicts (RS1, RS2 and RS3 fail; RC1 holds; no extension), R1 as the base (D201 Q28), and R1-P closed by its own gate (D205).
* The board now meets D196: Table C, next-read ETAs and F2 are in place. Keep it that way.
* Let R1-S's 150K read, R1-M's 135K-150K reads and R1's confirmation read run as planned.

### Your two questions
* **MM2.** Keep it as written. Print the absolute edited and unedited EDT errors and the JNDs beside it (Rule 12). An absolute decay bar is a P4-11 pre-registration, where decay is the target, not a P4-10 change.
* **R1's wording.** "Moved sources are far better than before, not solved." State it with the numbers:
  * In known rooms, at a new interior source, R1 reads 0.780 / 0.445 / 0.321, against 0.298 / 0.223 / 0.183 for ignoring the move and 0.235 / 0.099 / 0.048 for R0.
  * The gap to the unmoved accuracy is 0.13 below 100 Hz and about 0.24 above.
  * Reciprocity is partial below 100 Hz (swapped 0.76 against normal 0.91) and poor above it (0.34 / 0.22).

### Three eval-only diagnostics for P4-11 (no training; after the final reads, before F3-F7)

These decide what P4-11's decay and source arms should be. Post each as it finishes.

**E1. Decay: fit or transfer?** Run D1 and D2 on:
* (a) R1 in 15 training rooms. Use dense off-lattice truth: simulate it (about 15 FDTD runs); do not reuse training receivers.
* (b) The S-1 single-room models, each in its own room.

Reading:
* If S-1's peaks have the truth's width and R1's training rooms are still broad, the broadening comes from sharing one model across rooms (Q25).
* If S-1 is broad too, it comes from the loss or the representation.
* Report the width ratio, matched-peak counts, EDT error and floor, with the 15 in-family test rooms beside them.

**E2. Reciprocal averaging, zero training.** On R1's Source cells (second and third source, all 31 rooms), score three renders against the same truth:
* (a) the normal render F(s', r);
* (b) the swapped render F(r, s');
* (c) their complex average.

Also score (c) at the original source as a no-harm check. Pre-registered reading: averaging is a P4-11 candidate only if (c) beats (a) by at least 0.03, with CI above zero, in a band above 100 Hz at the third source, and loses no more than 0.01 at the original source.

**E3. Is the late floor common to the room?** Use the cached original-room renders (in-family 15; 63 / 125 / 250 Hz). Take the last 0.3 s of the 1.5 s gate. Compute the energy of the receiver-mean signal divided by the mean per-receiver energy, for the model and for the truth.
* A much larger ratio for the model means the floor is a room-uniform component: a structural artifact rather than per-bin noise.
* The single-radius residual in 0d, which the constant-cancelling form removed, hints at this.

### Keep as planned
* The source-clearance diagnostic, F1, then F3-F7.
* R1-M's and R1-S's D1, D2 and A8 at their final checkpoints.

### For the meeting record
Report the unseen numbers on both sets. The same R0 reads 0.468 / 0.380 / 0.371 on the frozen unseen set and 0.607 / 0.447 / 0.400 on the fresh confirmation set. The two 16-room sets differ by up to 0.14 for one model. So an unseen difference between models under about 0.03 means nothing without the paired CI.
