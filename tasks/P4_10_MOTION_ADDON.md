From: the manager. To: the coding agent, relayed by Amisha. 2026-11-24, 11:35 ET. Add-on to the meeting-day brief.

# Motion figures for the meeting

The meeting now centres on motion: moving the source (one source per room, then reciprocity swaps, then four source positions) and moving the listener. The capture (E-B) and physics (P-2, P-1) results come after that. Materials and stretch go to backup. Collaborators respond best to spatial maps, so we need a few more in the V2 / V4 / V5 style.

Rules:
* Plot only fields and renders that already exist or are quick to render. No new training. P-1 and R1-SM keep their GPUs.
* Post each figure as it lands. Hard stop 14:30 ET. If time runs short, drop the figures from the bottom of the list first.
* Same style as V2 / V4 / V5: 16:9, 14 pt or more, PNG plus PDF plus JSON sidecar, r printed under every model map, and the room's rank stated.
* Say "in-family (held-out)", not "known rooms", in new titles and captions.
* List every new figure in MEETING_PACK.md and rebuild the presentation report after each.

## Figures, in priority order

1. **M1, moving the source with all three training recipes.** Use V2's room, line and six positions, at 50 Hz and 125 Hz.
   * Rows, top to bottom: truth, R0 (one source per room), RCP (one source per room plus source-listener swaps), R1 (four source positions per room plus swaps).
   * In the JSON, give each model's source-family r over the 15 in-family rooms. The scorecard has 0.231, 0.531 and 0.743 below 100 Hz.

2. **M2, reciprocity maps.** Use V2's room at 50 Hz and 125 Hz.
   * Columns: truth; R0 normal; R0 swapped; R1 normal; R1 swapped.
   * Normal: the source at the room's source position, mapped over listener positions.
   * Swapped: at each map point x, the model's render with the source at x and the listener at the original source position. By reciprocity it should equal the normal map.
   * Print r against the truth under each map.

3. **M3, the listener's walk with the source moved.** V4's layout and walk, with the source at the third position.
   * Rows: truth, R1, R0 (level against position and frequency), plus spectra at the three walk points.
   * Print accuracy per band for R1 and R0.

4. **P1F, the P-1 result.** Post it as soon as the +10K read lands, and earlier with whatever reads exist by 13:00.
   * Left panel: the single-radius residual over the truth's at 25-100 Hz against step, for R1-P(t) and R1-ctl, with a line at 1 = truth.
   * Right panel: HF and UF against step for both.
   * The 63 Hz room-common ratio and EDT go in the JSON and the caption.

5. **M4, capture then move the source.** Use the median unseen room by R1-M zero-shot accuracy, at 50 Hz, first draw.
   * Top row, original source: truth with the 32 microphone positions marked; zero-shot; captured.
   * Bottom row, source at the third position: truth; zero-shot; captured, then moved; KRR on the 32 microphones.
   * Print r under each map.

6. **M5, energy flow with the source moved.** V5's layout at 50 Hz with the source at the third position.
   * Panels: truth, R1, R0, with the direction agreement for each.
