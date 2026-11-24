# P4-10 handoff (written 2026-11-23 19:25 ET) -- read first if a new session takes over

The user asked for work to continue if the SSH connection breaks. This page says what runs by itself, what does not, and what to do next. Rules in force: D195 (spec, `tasks/P4_10_SPEC.md`), D197 (amendments), **D201 (manager rulings, `tasks/P4_10_RULINGS.md`)**, D202 (decay metrics), D203 (P4-11 confirmation set, not scored), D204 (R1-P 0d). Live log: `tasks/P4_10_STATUS_REPORT.md`.

## State at 2026-11-24 03:26 ET: no training run is in progress

All arms are final (R1, R1-S, R1-M at 150K); no arm was extended; R1-P's gate (0d) failed (D205) and R1-C is closed. Verdicts: `outputs/p4_10/verdict_R1_150K.md`, `verdict_R1S_150K.md`, `verdict_R1M_150K.md`. All reads, E1 / E2 / E3, the source-clearance diagnostic, D1 / D2 / A8 for every final model and the one-time R1 confirmation read are posted (status report, newest first; summary `tasks/CHUNK_P4_10_RESULTS.md`).

**Nothing more runs by itself** (the backstop job exited; the read watchers finished). In-session timers die with the session (cron ids in the session: the tick and the 08:43 digest).

## To do (in order)
1. **Daily digest at 09:00 ET on 2026-11-24** at the top of `tasks/P4_10_STATUS_REPORT.md`: 3-5 plain sentences the user can forward plus Table B (the arms' verdicts).
2. **Figures: done** (F1, F2, F3, F4, F5, F7 in `outputs/p4_10/meeting/` and `outputs/p4_10/progress/F2_trajectories.png`; F6 void; `outputs/p4_10/meeting/FIGURE_MANIFEST.md`). Remaining only if asked: a capture-then-edit step (2f, lowest priority).
3. **Manager-facing**: `tasks/P4_10_MANAGER_REPORT_3.md` (final results and decisions) and `tasks/CHUNK_P4_10_RESULTS.md` (written). Open: Q34-Q38 (manager, P4-11), Q33 (meeting date, the user's answer), Q24, Q25, Q17, Q15.
4. If the manager asks for anything after the final report, treat it as a new D-entry before running numbers.

## Cautions
* Failed-job checks must use `sacct -S <start> -E now -X -s FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL`: a state filter without `-E now` prints nothing, so an empty list is not evidence (found 2026-11-23 21:24, after the sweeps before then had been using the blind form; a correct query of the day shows four setup failures at 13:12-14:28 (manifest builder, unit test, two R1 smoke runs, all fixed before R1 launched) and the R1 reciprocity run at 21:21). A `tail` on the output hid the older ones.
* Do not re-launch a read that already has a watcher log line. Kill processes by PID / job id only. Do not delete corpora or the dense sets. Commit or push only text, code, configs, JSON, and PNGs under the .gitignore allowlist (`outputs/p4_10/**/*.png` is allowed).
* R1-C is closed. CLAUDE.md rule 10 is unchanged. Nothing about the meeting date is known (Q33 is open: the user answers).
