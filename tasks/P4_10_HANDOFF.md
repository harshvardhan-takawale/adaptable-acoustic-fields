# P4-10 handoff (written 2026-11-23 19:25 ET) -- read first if a new session takes over

The user asked for work to continue if the SSH connection breaks. This page says what runs by itself, what does not, and what to do next. Rules in force: D195 (spec, `tasks/P4_10_SPEC.md`), D197 (amendments), **D201 (manager rulings, `tasks/P4_10_RULINGS.md`)**, D202 (decay metrics), D203 (P4-11 confirmation set, not scored), D204 (R1-P 0d). Live log: `tasks/P4_10_STATUS_REPORT.md`.

## Runs by itself (Slurm, survives a dropped connection)
| what | id / where |
|---|---|
| R1 training, to 150K | job 8539621; `outputs/p4_10/arms/R1` |
| R1-S training, to 150K | job 8540809; `outputs/p4_10/arms/R1S` |
| R1-M training, to 150K | job 8542917; `outputs/p4_10/arms/R1M` (started 19:08:31 on 2026-11-23) |
| reads at every 5K checkpoint | login-node watchers, PIDs 1730074 (R1S), 1730075 (R1M) (R1 watcher retired: its 150K read was launched by the agent at 21:19), detached (parent 1); logs `logs/p4_10_watch_<ARM>.log`. UMIACS may kill login-node processes at any time |
| **backstop for those watchers** | Slurm job 8543973 (`scripts/p4_10_backstop.sh`): launches a read if a checkpoint is > 45 min old and nobody launched it; log `logs/p4_10_backstop.log` |
| R1-P gate 0d | chain 8543903 (FDTD array) -> 8543925 (R0 render) -> 8543926 (score) -> `outputs/p4_10/r1p_0d/report.md` |
| board rebuild | a Slurm job after every scored read, writes `outputs/p4_10/progress/` (it does NOT commit or push) |

## Does NOT run by itself (needs an agent)
Posting status entries, commit / push, the pre-registered verdicts, extensions, the daily digest, figures, and any new launch. The in-session timers (cron ids 746eb3b5 and 4092ab02) and waits die with the session.

## To do, in order (D201)
1. **Post each read within 1 h of its checkpoint** (status entry, board, commit, push, check raw links are 200). Pending reads at 2026-11-23 19:25: R1 @140K (+20K) and R1-S @135K were rendering; R1-M's first read is at 125K.
2. **R1 @+30K (150K): judge RS1 (strict: all six in-family gaps <= 0.10), RS2, RS3, RC1 (HF and UF >= CTL @150K - 0.02), with six gaps and CIs, unseen beside.** Extension to 50K steps more only if the worst gap <= 0.15 AND the mean of the six fell by >= 0.02 from +20K to +30K. Extending an arm means extending R1 to the same step too. Training code to 150K only: an extension needs a config with a longer horizon (the LR schedule ends at 150000).
3. **R1-S:** SS1 pooled over the two stretch families, 2 of 3 bands; extension only if the second-best band's difference >= 0.00 AND the band-mean difference rose >= 0.02 from +20K to +30K.
4. **Q28 base rule:** R1 is the base unless, on the lite set, its final third-source r is more than 0.03 below its +15K value (0.748 / 0.407 / 0.283) in any band, or its HF cost against CTL @150K exceeds 0.04; then R0. An arm that passes its primary and no-harm bar becomes the base.
5. **R1 on the P4-9 confirmation set (16 rooms, sha 688e319f18f6dd32): ONCE, at R1's final checkpoint (after the extension question is settled), beside S2X @240K and R0 @120K.** `scripts/p4_9_confirm_eval.py`.
6. **R1-M:** MM1 = EDT edit slope [0.8, 1.2], Pearson >= 0.8; MM2 = edited |EDT error| <= its own unedited + 5 points; MC1; learning check at +10K (slope >= 0.3). `scripts/p4_10_scorecard.py` (`material_mm`).
7. **R1-P: NOT launched** (gate 0d failed, D205). Its mean-value mode is not built.
8. **After final reads:** D1 (EDT, masked T20, floor per octave on original rooms: R0, CTL, RCP, S2X, R1, R1-S), D2 (half-power linewidths of the first 25-100 Hz peaks, A8 code, model vs truth), the source-move diagnostic (8 lite rooms, y = 1.6, 0.5 / 0.75 / 1.0 / 1.5 m from the east wall, 32 FDTD runs), F1 and the A8 mode-shift slope (before F3-F7).
9. **Daily digest at 09:00 ET** at the top of the status report: 3-5 plain sentences plus the board's Table B.
10. Docs at the end: `CONTEXT_FOR_MANAGER.md` P4-10 entry, `tasks/CHUNK_P4_10_RESULTS.md`, a D-entry for the MS / STR / MAT draws, FIGURE_MANIFEST.

## Cautions
* Do not re-launch a read that already has a watcher log line. Kill processes by PID / job id only. Do not delete corpora or the dense sets. Commit or push only text, code, configs, JSON, and PNGs under the .gitignore allowlist (`outputs/p4_10/**/*.png` is allowed).
* R1-C is closed. CLAUDE.md rule 10 is unchanged. Nothing about the meeting date is known (Q33 is open: the user answers).
