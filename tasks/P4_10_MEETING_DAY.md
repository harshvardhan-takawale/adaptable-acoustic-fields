From: the manager. To: the coding agent, relayed by Amisha. 2026-11-24, morning.

# Meeting day: experiments, evaluations and figures, start to 14:30 ET

Record this as one D-entry (rulings plus pre-registrations) before any number below. The collaborator meeting is at about 16:00 ET.

## Ground rules for today
* **Hard cut-off at 14:30 ET.** After 12:30, launch nothing new except reads already scheduled. Post every item as it finishes; do not batch.
* **Every figure is slide-ready:**
  * plain-language titles (accuracy, pattern of the change, ignore the edit, decay rate);
  * 14 pt or larger, 16:9;
  * PNG at 200 dpi plus PDF, with a JSON sidecar of the plotted values;
  * MP4 (H.264, at most 15 s) plus a GIF fallback for animations.
* **Display and scoring stay separate.** Maps are drawn on a full grid: models rendered at every grid point, truth from FDTD 2 cm fields. Every printed number comes from the off-lattice scorer (Rule 11). Captions say so. No white lattice gaps (as in F4).
* **Colours and marks.** Keep the manifest's entity colours, viridis for level maps and RdBu for change maps. Room outline in black, source as a white star, listener path as a white dashed line.
* **Selection is disclosed.** The showcase room is L5.36_W4.34_d1.88_w1.40 (the F3 and F5 room). For each figure, print its rank among the 15 in-family rooms on that figure's metric. If it ranks in the top 3, add the median room as a second example.
* **Unseen backups.** V1 and V2 also get one unseen backup: the unseen room with the median R1 static r.

## Rulings on Q34-Q38
* **Q34:** (a) approved, run as E-C below. E-C adds single-room and R1-fine-tune trajectories so that sharing and per-room exposure are compared at matched per-room rows. (b) and (c) wait for E-C. Agreed: not (e) first.
* **Q35:** (b) approved. R1-SM launches now (pre-registered below).
* **Q36:** (a) for the meeting. (c) runs as optional E-E today. No (b) or (d) before (c).
* **Q37:** D208's reading is confirmed (lite slope 0.62 against the 0.75 needed). (d) is approved.
  * A manager note for the record (reasoning, not a measurement): blur in frequency multiplies the impulse response by a time window, and that compresses the change in 1/EDT.
  * At the observed ~25% EDT shortening this predicts a slope near 0.8. Blur explains part of 0.54, not all.
  * The rest matches the mode-shift slopes (0.5-0.7). Working hypothesis: the shared model under-scales its edit responses. E-C(c) gives a first look.
* **Q38:**
  * Wording approved with these finals:
    * **R1-M:** "The first model that responds to a change of wall material through its inputs: right direction, nearly proportional (Pearson 0.95), about half the true size. After the edit its render is closer to the truth than before. Its remaining decay error is the same as every shared model's; the edit does not add to it (MM2)."
    * **E1:** "Models trained on one room reproduce their rooms' resonance widths and early decay (EDT within 2%). The model shared across 1,000 rooms blurs resonances by 1.4-1.7 times, even in rooms it trained on. The architecture can express the right decay; sharing one model across rooms blurs it." E-C's sentence follows when it lands.
  * **2f is not dropped.** It runs today as E-B, on R1-M. It is the roadmap's Phase 4 demonstration: ground a measured room, then edit it.

## Priority 0 (launch first): R1-SM, the combined model
* **Recipe:** R1 plus STR rows plus MAT labels, composed from R1-S's and R1-M's code.
  * STR replaces p = 0.25 of rows, as in R1-S.
  * MAT labels go on p = 0.5 of the non-STR rows, as in R1-M; STR rows stay at alpha 0.15.
  * Swaps and the D200 DC gain as in R1 and R1-M.
  * Warm start from R0 @120K with CTL's LR restart, 30K steps, on the shared batch stream.
* **Rule-8 tests:** the composed stream reproduces R1-S's rows bit for bit with MAT off, and R1-M's with STR off.
* **Reads:** lite reads at +5K, +10K, +15K, +20K and +25K (descriptive, posted before the meeting). The full scorecard, D1, D2 and A8 at +30K, after the meeting.
* **Kill at +10K:** the standard harm rule against R1.
* **Pre-registered at +30K,** against the matched-step checkpoints. "The gains add" if all four hold:
  * **CS1:** stretch pattern of change (in-family, Stretch-U and Stretch-A pooled) >= R1-S - 0.01, every band.
  * **CM1:** material EDT slope >= R1-M - 0.05, and Pearson >= 0.9.
  * **CSrc1:** third-source in-family r >= R1 - 0.02, every band.
  * **CN1:** HF and each UF band >= R1 - 0.02, CI not entirely below -0.02.

## Priority 1: meeting visuals (V1-V8)

FDTD for all of these is small (about 35 runs); run it on CPU arrays at once.

**V1. Stretching a room.**
* Filmstrip: s = 0.90, 0.95, 1.00, 1.05, 1.10. Rows: truth and R1-S. Level maps at 50 Hz and 125 Hz, with each panel's r printed.
* Animation: 9 anchor scales, 0.90 to 1.10 in steps of 0.025, truth and R1-S side by side.
* FDTD: 9 runs with 2 cm fields at both frequencies, plus V4's path receivers.

**V2. Moving the source.**
* Positions: 6 along the straight line from the original source (0.5, 1.6) to the third source; 12 for the animation.
* Rows: truth, R1, R0. Separate filmstrips at 50 Hz and 125 Hz, each panel's r printed.
* FDTD: 12 runs.

**V3. Changing wall materials.**
* (a) Decay rate 1/EDT against all-walls absorption alpha = 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, per octave (63 / 125 / 250 Hz), for truth, R1 and R1-M. Showcase room; 6 FDTD runs.
* (b) The same for one wall only (the longest; the "soft wall" edit): alpha = 0.15, 0.30, 0.40, 0.50. 3 new FDTD runs.
* (c) Early energy decay curves (0 to -30 dB) at alpha 0.10 and 0.50, for truth and R1-M, at 63 and 125 Hz.
* (d) The MM1 scatter over all in-family Material cells: predicted against true change in 1/EDT, for R1 and R1-M, with the y = x line and the slope and Pearson in the legend. Existing data.

**V4. The listener's walk.**
* Path: a straight line along the room's long axis through the interior, at least 0.5 m from walls and the source, 100 points.
* Position-by-frequency level images (25-300 Hz): truth, R1, and their difference. One set for the original room and one for s = 1.10 (truth from V1's runs).
* Three marked positions with their frequency responses, truth against R1.

**V5. Where the energy flows.**
* Active-intensity arrows over the level map at 50 Hz and 100 Hz, original source, truth against R1.
* Model: band-limited gradient (Q-C's method). Truth: FDTD 2 cm gradient.
* Print the direction agreement in the title.

**V6. Why decay is off.** In the S-1 room L5.74_W4.78_d0.62_w2.32:
* (a) The receiver-averaged power spectrum over 25-100 Hz for truth, S-1 and R1, on the same axes. This is the resonance-width picture.
* (b) Energy decay curves at 63 and 125 Hz for truth, S-1 and R1.

**V7. The late floor under standard compensation (descriptive).**
* Apply ISO 3382-1 noise compensation (Lundeby truncation plus a correction for the energy after the truncation point) to the model impulse responses.
* Report EDT, T20 and T30 errors before and after, for R1, R1-M and S-1, on the 2 S-1 rooms and the 15 in-family rooms.
* Panel: R1 raw, R1 compensated and truth at 125 Hz.
* Purpose: separate the late floor from the early-decay error. It is not a pass bar.

**V8. Slide scorecards.** A 2 x 2 figure: known and new rooms, at 25-100 Hz and at 100-200 Hz.
* Per family: an R0 bar, and a "this round" bar labelled with its model (Stretch: R1-S; Wall, Notch, Source: R1; Material: R1-M).
* A red tick for "ignore the edit".
* F7 stays as the backup.

## Priority 1: E-B, capture then edit (Phase 4; pre-registered)
* **Model and capture.** R1-M. M2 with D183's settings (K = 32 microphones, seeds 0-2), in each of the 16 frozen unseen rooms.
* **Edit operator.** 2f's: re-encode the edited room and add the fitted code offset; keep the fitted FiLM heads.
* **Cells** (truth already exists): Stretch-U 1.10, Wall L +0.3, Material all walls 0.30, Material longest wall 0.50, Source third.
* **Compared against:**
  * the zero-shot edit (R1-M, no capture);
  * the captured model without the edit, scored against the edited truth;
  * KRR (D183's M5) on the same 32 microphones, scored against the edited truth. Interpolating the measurements cannot edit.
  * The truth's "ignore the edit" as the reference.
* **Survival** = (captured-and-edited r - zero-shot edited r) / (captured static r - zero-shot static r). Take it on room means, with a bootstrap CI over rooms.
* **CE1 (primary):** survival >= 0.5 at 25-100 Hz for Stretch-U, Wall and Material (all walls 0.30), each family on its own.
* **Descriptive:**
  * every band;
  * Source;
  * on the Material cells after capture, the change in 1/EDT predicted against true (slope, Pearson) and g_delta.
* **Figure:** bars per family for zero-shot edit, captured and edited, captured unedited, and KRR, with the "ignore the edit" tick. Add one Material decay panel after capture.

## Priority 1: E-C, how much per-room training fixes the decay (descriptive; the plot that decides P4-11)

Rooms: the 2 S-1 rooms. Same metrics at every point: width ratio, matched peaks, EDT error at 63 / 125 / 250 Hz, floor, and static off-lattice r. The x-axis is per-room training rows, on a log scale. Three trajectories:
* **(a) Shared.** R0 @30K / 60K / 90K / 120K (whichever checkpoints survive; say which), CTL @150K, S2X @240K and R1 @150K.
* **(b) Single room from scratch.** S-1's recipe re-run with checkpoints at 30, 60, 120, 250, 500, 1000 and 2000 steps. At about 16 rows per step, about 120 steps matches R0's rows per room.
* **(c) R1 fine-tuned on the room.** All weights, checkpoints at the same steps. Record the LR choice before the first number.
  * Also score Stretch-U s = 1.10 on these 2 rooms at each checkpoint: edited r and mode-shift. It shows whether editability survives specialisation. 2 FDTD runs.

How to read it:
* If (b) is sharp at R0's per-room rows while (a) is broad, sharing causes the broadening.
* If (b) also needs many more rows, exposure does.
* (c) shows how fast a shared start specialises.

Post one figure of width ratio and EDT error against rows, with (a), (b) and (c) on the same axes.

## Priority 3 (only if a GPU is idle before 12:30): E-E, source coverage
* Test positions: 20 source positions per room in the 8 in-family lite rooms, at least 0.5 m from the walls (about 160 FDTD runs).
* Plot R1's moved-source r per band against the local density of MS training sources in absolute coordinates: the count within 0.3 m, 0.5 m and 1.0 m.
* Descriptive: does accuracy follow training coverage?

## Schedule (targets; post as each lands)

| time (ET) | item |
|---|---|
| on receipt | D-entry; R1-SM launched; all FDTD submitted |
| about 10:15 | R1-SM +5K smoke read |
| 10:30-11:30 | V1, V2, V3, V6 (in that order) |
| about 11:30 | R1-SM +10K; E-B |
| 11:30-12:30 | V4, V5, V7, V8 |
| about 12:45 | R1-SM +15K; E-C |
| about 14:00 | R1-SM +20K; E-E if run |
| 14:30 | MEETING_PACK.md (below); R1-SM +25K is posted when it lands, marked as after the cut-off |

**GPU plan if four are free:**
* one: R1-SM;
* one: V-renders, then E-C(a);
* one: E-C(b) and (c);
* one: E-B.

With three, run E-B after the V-renders. FDTD, decay metrics and KRR go on CPU.

## MEETING_PACK.md (by 14:30)
1. **Figure list.** Each figure with a one-sentence takeaway and the numbers behind it, linked to its JSON.
2. **Numbers sheet.** About 20 headline numbers, each with its source file, written so they can go straight onto a slide.
3. **Do not claim:**
   * that clearance explains the second source;
   * that material edits beat the original room's true field;
   * that geometry edits transfer to new rooms (unseen pattern of change <= 0.33; notches not predicted);
   * that R1-SM's gains add, before its +30K verdict;
   * that anything renders in real time except single listeners (12 ms).
4. **Anything that failed today, or did not finish, in one line each.**
