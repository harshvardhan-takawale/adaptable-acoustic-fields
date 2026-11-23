# P4-10 status report (live log) — the editability scorecard, the source fix, four single-change arms

**Pre-registration:** D195 (spec, verbatim; also `tasks/P4_10_SPEC.md`), D196 (progress contract), D197 (amendments A1-A14), D198 (edit manifest frozen), D199 (per-edge absorption validated). The P4-9 erratum (D194) corrects a premise the spec inherited.
**Board:** [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md). One dated entry per read, newest first, entry times from `date`. Raw links checked for 200.

## 2026-11-23 (16:08) — I3 (smoke read): R1 (the source fix) at +5K steps already beats the swap-only model at the moved source

**Headline:** R1, the model trained with several source positions per room, reached its first scored checkpoint after only 5K steps. At the moved source it is already as good as the swap-only model (RCP) after six times as much training, and better above 100 Hz: at the clean third source, in-family, 0.648 / 0.342 / 0.202 against RCP's 0.604 / 0.266 / 0.147 (25-100 / 100-200 / 200-300 Hz), and against "leave the source alone" 0.285 / 0.231 / 0.177 it is ahead in every band. Unseen, 0.473 / 0.329 / 0.199 against RCP 0.439 / 0.273 / 0.154. This is a small-sample reading: the 16-room "lite" set and one early checkpoint.

Off-lattice, frozen scorer; scorecard-lite rooms (8 in-family, 8 unseen) at the second and third source; paired bootstrap over rooms (10k, seed 20261013). "gap" = the model's static r on the original room minus its r at the moved source.

| source | set | R1 @+5K (125K) | RCP @+30K | CTL @+30K | ignore the edit | R1 - CTL (25-100 / 100-200 / 200-300) | R1 - RCP |
|---|---|---|---|---|---|---|---|
| second | in-family | 0.539 / 0.241 / 0.152 | 0.503 / 0.189 / 0.102 | 0.234 / 0.067 / 0.041 | 0.336 / 0.197 / 0.249 | +0.305 / +0.175 / +0.111 | +0.036 [-0.033, +0.106] / +0.053 [+0.033, +0.076] / +0.050 [+0.033, +0.066] |
| **third** | **in-family** | **0.648 / 0.342 / 0.202** | 0.604 / 0.266 / 0.147 | 0.220 / 0.071 / 0.028 | 0.285 / 0.231 / 0.177 | +0.427 [+0.358, +0.490] / +0.271 / +0.174 | +0.044 [+0.017, +0.071] / +0.076 / +0.055 |
| second | unseen | 0.454 / 0.204 / 0.168 | 0.448 / 0.173 / 0.116 | 0.158 / 0.074 / 0.042 | 0.381 / 0.229 / 0.229 | +0.296 / +0.130 / +0.126 | +0.007 [-0.057, +0.071] / +0.032 / +0.052 |
| **third** | **unseen** | **0.473 / 0.329 / 0.199** | 0.439 / 0.273 / 0.154 | 0.241 / 0.135 / 0.041 | 0.314 / 0.276 / 0.200 | +0.232 / +0.193 / +0.157 | +0.034 [-0.001, +0.072] / +0.056 / +0.044 |

- **Still at 125K steps (+5K).** Moved-source gap (R1's static r on the original room minus its r at the moved source; RS1's bar is <= 0.10 in every band at BOTH sources, read at +30K): in-family third source 0.243 / 0.305 / 0.328, second 0.351 / 0.405 / 0.378. R0's was 0.68 / 0.60 / 0.53 (third). R1 is not at the bar yet; whether it reaches it by +30K is the pre-registered read.
- **No static accuracy yet:** the HF / UF read for R1 @125K (standard report) is still rendering on the cluster; it follows this entry. The +10K learning check (in-family second-source r at 25-100 Hz >= 0.38) is read at 130K.
- **RCP's lite numbers differ slightly from its full-set numbers** because the lite set has 8 rooms per group (RCP full in-family third source 0.575 / 0.266 / 0.153).

Links: [scorecard harness](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/scripts/p4_10_scorecard.py) · [P4_10_R1.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_10_R1.yaml) · [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md)

## 2026-11-23 (15:36) — I2 cont.: Source, three models; the swap-only model (RCP) does move the source, at low frequencies, in both sources

**Headline:** Scoring the source-swap model (RCP) beside the plain continuation (CTL) and the meeting model (R0) at the same two moved positions. RCP's gain over CTL is clear at both sources: at the clean third source, +0.34 / +0.18 / +0.11 (25-100 / 100-200 / 200-300 Hz, in-family) and +0.20 / +0.13 / +0.11 unseen. It beats "leave the source alone" at 25-100 Hz in both room sets (third source 0.575 vs 0.298 in-family, 0.443 vs 0.332 unseen), and at 100-200 Hz in-family (0.266 vs 0.223), but not at 200-300 Hz. Physics still sits at 0.99 / 0.95 / 0.79. The R1 arm adds multi-source training on top of RCP's swaps.

Off-lattice, paired over rooms, bootstrap 10k (seed 20261013); CTL's and RCP's own static accuracy is unchanged (0.92 / 0.71 / 0.58; D194, earlier reads).

| source | set | R0 edited r | CTL edited r | **RCP edited r** | ignore the edit | RCP - CTL |
|---|---|---|---|---|---|---|
| second | in-family 15 | 0.227 / 0.067 / 0.040 | 0.221 / 0.068 / 0.048 | 0.487 / 0.191 / 0.098 | 0.311 / 0.211 / 0.247 | +0.266 [+0.228, +0.306] / +0.123 / +0.051 |
| **third** | **in-family 15** | 0.235 / 0.099 / 0.048 | 0.232 / 0.090 / 0.041 | **0.575 / 0.266 / 0.153** | 0.298 / 0.223 / 0.183 | **+0.343 [+0.291, +0.397] / +0.176 / +0.112** |
| second | unseen 16 | 0.172 / 0.077 / 0.055 | 0.176 / 0.068 / 0.052 | 0.444 / 0.163 / 0.114 | 0.357 / 0.204 / 0.210 | +0.267 [+0.217, +0.312] / +0.094 / +0.062 |
| **third** | **unseen 16** | 0.252 / 0.144 / 0.069 | 0.244 / 0.135 / 0.056 | **0.443 / 0.266 / 0.164** | 0.332 / 0.287 / 0.208 | **+0.199 [+0.164, +0.236] / +0.131 / +0.107** |

- **Moved-source gap** (static r on the original room minus moved r; RS1's metric for R1, bar <= 0.10 in every band; read here as a reference for R1): R0 0.683 / 0.602 / 0.534 in-family third source (CTL 0.689 / 0.618 / 0.542); RCP 0.346 / 0.439 / 0.429. RCP is a large step towards, but not at, the bar of 0.10 in every band.
- **Reading, replacing the D194 wording for RCP:** RCP's gain at the clean third source is larger than at the (mirror) second one at every band, so it does not come from the flips. Swaps alone carry RCP's gain at 25-200 Hz; at 200-300 Hz RCP stays below ignore-the-move (0.153 vs 0.183 in-family; 0.164 vs 0.208 unseen). Whether multi-source training closes the rest is R1's question.
- **A corruption fixed:** nine score files (CTL and RCP, Source) were corrupted by two jobs writing the same cell at once. They were regenerated; the scorer now writes atomically, and 0 corrupt files remain. R0's published table was not affected.

Links: [scorecard_source_R0_CTL_RCP.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_source_R0_CTL_RCP.md) · [json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_source_R0_CTL_RCP.json)

## 2026-11-23 (15:28) — I2 (first read): R0's Source family; R0 cannot move the source to a NEW position either

**Headline:** This is the first scored sound-field edit. We moved the source to two new positions: the "second" one (the mirror position from P4-9) and a "third" one that neither the training nor the mirror flips ever used. For known room shapes R0's accuracy at the moved source is 0.23 / 0.08 / 0.04 (25-100 / 100-200 / 200-300 Hz), which is worse than leaving the source alone (0.30 / 0.22 / 0.22) and far below cheap physics re-solved for the new position (0.99 / 0.95 / 0.79). The third, cleanest position gives the same answer as the second, so the failure is not special to the mirror position.

Off-lattice, 601-bin renders; off-lattice receivers and the frozen scorer; paired bootstrap over rooms (10k, seed 20261013). "edited r" = the model at the new source against the re-simulated field; "ignore" = the ORIGINAL field scored against the moved-source truth; m1 = physics re-solved at the new source (10 cm grid, the reference). 'second' = the 4 source position of D182/D191; 'third' = the new one (D198).

| set | source | R0 edited r | ignore the edit | R0 r_delta | R0 g_delta | m1 physics |
|---|---|---|---|---|---|---|
| in-family 15 | second | 0.227 [0.183, 0.270] / 0.067 / 0.040 | 0.311 / 0.211 / 0.247 | 0.476 / 0.383 / 0.296 | 0.76 / 0.95 / 0.94 | 0.984 / 0.949 / 0.778 |
| in-family 15 | **third** | **0.235 [0.169, 0.304] / 0.099 / 0.048** | 0.298 / 0.223 / 0.183 | 0.532 / 0.400 / 0.313 | 1.18 / 1.04 / 0.93 | 0.985 / 0.945 / 0.792 |
| unseen 16 | second | 0.172 / 0.077 / 0.055 | 0.357 / 0.204 / 0.210 | 0.265 / 0.246 / 0.224 | 0.86 / 1.02 / 1.02 | 0.972 / 0.899 / 0.761 |
| unseen 16 | **third** | 0.252 [0.216, 0.291] / 0.144 / 0.069 | 0.332 / 0.287 / 0.208 | 0.300 / 0.260 / 0.228 | 1.19 / 1.06 / 0.99 | 0.972 / 0.916 / 0.770 |

- **Parity gate PASSES:** R0's scorecard render of the ORIGINAL rooms reproduces its stored static accuracy (in-family 0.9185 / 0.7009 / 0.5818; unseen 0.4676 / 0.3798 / 0.3711).
- **Manager's prediction P-d (R0 below ignore-the-move in every band): HOLDS,** both sources, both sets (the 'second' row repeats S-3).
- **r_delta** (does the predicted change follow the true change in pattern?): 0.3-0.5 in-family, 0.22-0.30 unseen, so the change map has some truth in it even though the moved field itself is wrong. **g_delta** (is the new render closer to the moved truth than the old render was): above 1 only for the third source at 25-100 Hz (1.18 in-family, 1.19 unseen).
- **Cells:** Source is complete for R0 (62 of 62). CTL @150K and RCP @150K Source are rendering and follow; R0's other four geometry families and materials follow as they finish.
- **R1** (the source fix) is at 123.2K of 150K steps at the time of writing and 0.78 s per step; its first scored checkpoint (+5K) is due around 15:50.

Links: [scorecard_R0_source.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R0_source.md) · [scorecard_R0_source.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R0_source.json) · [edit_manifest.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/p4_10/edit_manifest.json)

## 2026-11-23 (14:54) — I0 / I1: pre-registration, gates, corpora; R1 is training

**Headline:** Everything needed to score sound-field edits against re-simulated truth is in place and verified. The source fix (R1, multi-source training) started training this afternoon; its first scored checkpoint is due around 16:00, and it reads every 5K steps (~1.1 hours) after that.

- **Edit manifest frozen (D198):** 675 cells over 31 rooms (in-family 329, unseen 346), 7 logged skips, sha 7ce02a5fb5ec. The third source is feasible in all 31 rooms.
- **Truth-parity gate PASSES (no number scored before it):** the dense-receiver rule reproduces all 31 stored rooms' receivers exactly; re-simulating original rooms matches the stored field to 2e-10 (relative L2 over 25-300 Hz, normalised by the band). **All 551 geometry + source truth cells and all 124 material cells are simulated.**
- **Per-edge absorption gate PASSES (D199):** against an exact rectangle solution (complex Newton, no small-alpha expansion), FDTD decay rates match within 0.4% for every mode up to alpha 0.70, including the case where the old gate failed. The old reference was approximate, not the solver.
- **Corpora:** MS (1000 rooms x 4 sources; slot 0 = the room's own source (A1); 142 sources redrawn for the 1.0 m rule, 1 room with no 1.0 m node; every new slot re-simulates exactly, slot 0 matches the corpus to 6e-10). STR (2,000 stretched variants x 4 sources) assembled. MAT (per-edge alpha, continuous on m = -ln(1 - alpha) over [0.08, 0.60]) is simulating.
- **Trainer (default-off, Rule 8 tested):** multi-source memmap rows with one per-row resolver (slot, source, target, swap mask, jitter), per-row jitter from dedicated checkpointed generators; streams are identical across arms with and without replaced rows; slots and targets verified to pair correctly. **Amendment A3 stands: training in `aaf` (fp32), scoring in `aaf`.**
- **R1 launched** (tron, high-memory queue): R0 @120K warm start, MS corpus, swaps p 0.25, reads at +5K ... +30K every 5K. 0.76 s per step.
- **Not yet scored:** R0's scorecard Source family (GPU queue), CTL and RCP Source.

