# P4-10 handoff (rewritten 2026-11-24 09:14 ET, meeting day) -- read first if a new session takes over

The user asked (2026-11-24 morning) that work not stop if the connection breaks. Instructions in force: `tasks/P4_10_MEETING_DAY.md` (D211), `tasks/P4_10_PHYSICS_ADDON.md` (D217). **Hard cut-off 14:30 ET; launch nothing new after 12:30 except reads already scheduled. Meeting about 16:00 ET.** Designs: D212 (R1-SM), D213 (V1-V5), D214 (E-B), D215 (E-C), D216 (V6-V8, V3d), D218 (P-1, when written), D219 (P-2), D220 (E-E). Live log: `tasks/P4_10_STATUS_REPORT.md`.

## What runs by itself (Slurm, independent of any session)
* **R1-SM** training, job 8555361 (tron, requeue): +5K ... +30K checkpoints at 125K ... 150K; +30K is due about 15:30.
* **R1-SM reads**: detached watcher on the login node (`scripts/p4_10_watch_arm.sh R1SM`, log `logs/p4_10_watch_R1SM.log`), and the **backstop job `aaf_backstop`** (launches any read the watcher missed 30 min after its checkpoint; log `logs/p4_10_backstop.log`). Each read ends with a board rebuild.
* **Dead-man publisher `aaf_autopub`** (`scripts/p4_10_autopublish.sh`): does nothing while `logs/p4_10_heartbeat` is younger than 50 min; after that it rebuilds `outputs/p4_10/meeting/MEETING_PACK.md` (marked unreviewed), commits an allowlist of result files and pushes, every 10 min until 17:00. **A live session should `touch logs/p4_10_heartbeat` whenever it works.**
* **E-B, E-C, P-2, P-1** were built by forked agents that were told to chain their remaining steps as Slurm jobs (names aaf_eb*, aaf_ec*, aaf_p2*, aaf_pt*). P-1's reads run from a watcher job. E-E's truth runs: `aaf_eefd`; its renders (`scripts/p4_10_ee_coverage.py --render --idx 0..7`, GPU) start only if a GPU is free before 12:30.

## To do, in order (what still needs an agent)
1. Post each result as it lands (status report, newest first; a read within one hour of its checkpoint): R1-SM +5K (about 09:42), +10K (about 11:00), +15K, +20K, +25K (descriptive; the +25K read is marked after the cut-off). **Kill check at +10K**: the standard harm rule against R1. The +30K verdict (CS1, CM1, CSrc1, CN1; D211) is after the meeting.
2. E-B (CE1), E-C (trajectories), P-2 (audit), P-1 (reads, stop rule) as they finish; figures `EB_capture_then_edit`, `EC_exposure`, `P2_how_physical`. Look at every PNG before posting.
3. **MEETING_PACK.md by 14:30**: `python scripts/p4_10_meeting_pack.py` (reads every number from JSON; add lines to `outputs/p4_10/meeting/PACK_NOTES.md` for anything that failed, in one line each; add takeaways for E-B / E-C / P-2 / P-1 in the script). Commit, push, check links return 200.
4. After the meeting: R1-SM's +30K full scorecard, D1, D2, A8, verdict (`CS1`, `CM1`, `CSrc1`, `CN1`); `CONTEXT_FOR_MANAGER.md` and a results entry (Rule 1); OPEN_QUESTIONS if anything is undecided.

## Cautions
* Do not edit `aaf/train/multi_room_2d_mat.py` while R1-SM may requeue (it re-imports the file); P-1's code is in `aaf/train/p410_pt.py`.
* `sacct` failure checks need `-S <start> -E now -X -s FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL`; many nodes (legacy*, legacygpu*, cml23, gammagpu02-06/09) failed this morning: exclude them.
* Kill by job id or PID only (never `pkill -f`). Do not delete corpora or the dense sets. Never force-push. Gates that say "needs the manager" are real.
* Do not claim (meeting): clearance explains the second source; material edits beat the original room's true field; geometry edits transfer to new rooms; R1-SM's gains add before its verdict; anything real-time except a single listener (12 ms); "12-34% too fast" for all of the decay (EDT only).
