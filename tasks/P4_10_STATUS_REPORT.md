# P4-10 status report (live log) — the editability scorecard, the source fix, four single-change arms

**Pre-registration:** D195 (spec, verbatim; also `tasks/P4_10_SPEC.md`), D196 (progress contract), D197 (amendments A1-A14), D198 (edit manifest frozen), D199 (per-edge absorption validated). The P4-9 erratum (D194) corrects a premise the spec inherited.
**Board:** [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md). One dated entry per read, newest first, entry times from `date`. Raw links checked for 200.

## 2026-11-23 (17:52) — I4 (R1 +10K, R1-S +5K): R1 passes its learning check, keeps gaining on the moved source, and is paying a little static accuracy

**Headline:** After 10K steps the source-fix model (R1) is still gaining on source moves: at the clean third source in known rooms it reaches 0.713 / 0.391 / 0.279 (25-100 / 100-200 / 200-300 Hz), up from 0.648 / 0.342 / 0.202 at +5K, and the gap to its own static accuracy has narrowed to 0.19 / 0.26 / 0.27 (bar: at most 0.10 in every band, read at +30K). Its accuracy at the original source, the price of training on several positions, is 0.024 below the plain continuation at the same step count (HF 0.550 against 0.575), while unseen rooms are up. The second arm, R1-S (stretched rooms mixed in), reached its first checkpoint; at +5K it is level with R1 on stretch edits, as expected that early.

Off-lattice; lite set (8 in-family, 8 unseen rooms) for the source cells; frozen scorer.

- **Learning check (pre-registered, +10K): PASSES.** In-family second-source r at 25-100 Hz 0.587 (bar 0.38; R0 0.227; RCP at +30K 0.503).
- **Harm kill (+10K; HF or any UF band below the reference by more than 0.05 with the whole CI below -0.02): NOT triggered.** R1 @130K - CTL @130K (matched step): HF -0.024 [-0.031, -0.018]; UF +0.028 [+0.015, +0.041] / +0.013 [+0.001, +0.025] / +0.013 [+0.002, +0.025]. Against CTL @150K (the final reference): HF -0.033 [-0.039, -0.027], UF +0.025 / +0.023 / +0.020. Static levels (Rule 12): R1 @130K HF 0.550, UF 0.492 / 0.398 / 0.393 (R1 @125K 0.536, 0.483 / 0.393 / 0.377; CTL @130K 0.575, 0.464 / 0.385 / 0.380). The RC1 rule at +30K allows R1's HF to be at most 0.02 below CTL @150K.
- **Source cells at +10K** (edited r / gap to the original room's static r): second source in-family 0.587 / 0.265 / 0.201 (gap 0.312 / 0.390 / 0.347), unseen 0.518 / 0.235 / 0.198 (gap 0.008 / 0.202 / 0.204); **third source in-family 0.713 / 0.391 / 0.279 (gap 0.187 / 0.264 / 0.269), unseen 0.525 / 0.356 / 0.255 (gap 0.000 / 0.080 / 0.148).** RS1's bar (<= 0.10 in every band, both sources, at +30K) is not met yet but every gap shrank from +5K (third source in-family 0.243 / 0.305 / 0.328 at +5K).
- **R1-S @+5K (descriptive; its reference R1 is at a different step and is labelled).** Stretch-U r_delta (8 in-family rooms) 0.809 / 0.558 / 0.370; Stretch-A 0.732 / 0.482 / 0.277; edited r Stretch-U 0.787 / 0.539 / 0.437 (R0 0.761 / 0.532 / 0.417). Against R1's +10K (a step ahead): r_delta +0.009 [-0.014, +0.035] / -0.012 [-0.022, -0.003] / -0.010 [-0.023, +0.001]. The paired read against R1 at the same step starts at +10K.

Links: [board.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/progress/board.md) · [P4_10_R1.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_10_R1.yaml) · [P4_10_R1S.yaml](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweep_2d_mat/P4_10_R1S.yaml)

## 2026-11-23 (16:27) — I2 complete: R0's full editability scorecard (675 cells, 6 families) -> it edits geometry well in rooms it knows, not in new ones; it cannot edit the source or materials

**Headline:** We scored every edit the pre-registration lists against re-simulated truth. In rooms the model knows, changing the room's geometry works well: a stretched room is reproduced with accuracy 0.83 / 0.61 / 0.49 (25-100 / 100-200 / 200-300 Hz), and the moved wall and the notch edit with 0.90 / 0.67 / 0.56 and 0.92 / 0.70 / 0.57. Its predicted changes follow the true ones closely (pattern correlation 0.82 for stretches, 0.74 for moved walls). In unseen rooms the same edits are far worse (0.40 / 0.33 / 0.30). Moving the source and changing absorption do not work: the moved source scores 0.23 / 0.08 / 0.04, and changing the wall absorption moves the model the wrong way (the new render is farther from the truth than the old one). Cheap physics, re-solved for the edit, scores 0.97-0.98 in every family, so this is a score sheet, not a claim of superiority.

In-family 15 rooms (off-lattice; mean over rooms of each room's average over magnitudes; m1 = physics at the edited geometry, 10 cm grid, a reference):

| family | R0 edited r | ignore the edit | R0 r_delta | R0 g_delta | m1 physics | unseen: R0 edited r / r_delta |
|---|---|---|---|---|---|---|
| Stretch-U | 0.833 / 0.606 / 0.493 | 0.410 / 0.171 / 0.119 | 0.818 / 0.589 / 0.486 | 2.99 / 1.65 / 1.49 | 0.982 / 0.924 / 0.762 | 0.402 / 0.325 / 0.304 ; 0.253 / 0.203 / 0.249 |
| Stretch-A | 0.835 / 0.613 / 0.495 | 0.514 / 0.250 / 0.281 | 0.766 / 0.552 / 0.327 | 2.54 / 1.55 / 1.25 | 0.984 / 0.932 / 0.769 | 0.416 / 0.347 / 0.327 ; 0.224 / 0.164 / 0.135 |
| Wall | 0.900 / 0.673 / 0.562 | 0.735 / 0.421 / 0.233 | 0.743 / 0.506 / 0.458 | 2.14 / 1.45 / 1.45 | 0.984 / 0.928 / 0.764 | 0.438 / 0.363 / 0.347 ; 0.179 / 0.144 / 0.210 |
| Notch | 0.917 / 0.697 / 0.568 | 0.885 / 0.715 / 0.534 | 0.637 / 0.328 / 0.216 | 1.53 / 1.26 / 1.19 | 0.984 / 0.928 / 0.771 | 0.464 / 0.380 / 0.372 ; 0.067 / 0.019 / 0.019 |
| Source (second + third) | 0.231 / 0.083 / 0.044 | 0.305 / 0.217 / 0.215 | 0.504 / 0.391 / 0.304 | 0.97 / 0.99 / 0.94 | 0.985 / 0.947 / 0.785 | 0.212 / 0.110 / 0.062 ; 0.283 / 0.253 / 0.226 |
| Material | 0.772 / 0.583 / 0.469 | 0.911 / 0.891 / 0.880 | 0.067 / 0.026 / 0.017 | 0.84 / 0.84 / 0.85 | omitted (no per-wall beta) | 0.428 / 0.357 / 0.347 ; 0.027 / -0.006 / -0.000 |

(g_delta > 1 means the edited render is closer to the edited truth than the original render was. r_delta = per-bin spatial correlation of the dB change over the common receivers, clipped at +/-20 dB.)

- **P-a (edited r within 0.05 of static r, in-family): FAILS for Stretch, HOLDS for Wall.** Stretch-U -0.085 [-0.129, -0.047] / -0.094 [-0.125, -0.066] / -0.089 [-0.121, -0.059]; Stretch-A -0.084 [-0.124, -0.048] / -0.088 [-0.112, -0.064] / -0.086 [-0.118, -0.057]; Wall -0.019 [-0.035, -0.005] / -0.028 [-0.043, -0.012] / -0.020 [-0.037, -0.004].
- **P-b (g_delta > 1 for Stretch-U and Wall at 100-300 Hz, in-family): HOLDS** (Stretch-U 1.65 / 1.49, Wall 1.45 / 1.45).
- **P-c (Notch r_delta below Stretch and Wall r_delta, in-family): HOLDS.** Paired: Notch - Stretch-U -0.181 [-0.217, -0.146] / -0.261 / -0.271; - Stretch-A -0.129 [-0.166, -0.092] / -0.224 / -0.111; - Wall -0.106 [-0.140, -0.074] / -0.177 / -0.242.
- **P-d (Source below "ignore the move" in every band): HOLDS** (0.231 / 0.083 / 0.044 against 0.305 / 0.217 / 0.215, and the clean third source).
- **P-e (Material: g_delta near 1, decay-rate slope near 0): mostly.** g_delta 0.84 in every band (the edited render is FARTHER from the truth, not near 1). Decay-rate edit slope -0.11 with Pearson -0.64 (pooled over octaves and cells, n 180 in-family): near zero and anti-correlated with the truth (the model's small predicted change goes the wrong way). R0's corpus never varied absorption, so this is extrapolation (D200 shows its head ignores material tokens).
- **P-f (unseen: every family at or below R0's static unseen r): HOLDS.** Edited minus static unseen: Stretch-U -0.065 [-0.099, -0.036] / -0.055 / -0.067, Stretch-A -0.052 / -0.032 / -0.044, Wall -0.030 [-0.045, -0.015] / -0.017 / -0.024, Notch -0.004 / +0.001 / +0.001 (equal within CI), Source -0.255 / -0.270 / -0.309, Material -0.039 / -0.023 / -0.024 (all CIs below or at 0).
- **Reading:** R0 is editable for geometry inside the family it was trained on, with a real pattern of change (r_delta up to 0.82), and degrades quickly outside it. The source and material edits are not there, which is what R1 and R1-M target. **Not yet scored:** mode-shift slope (A8), F1 figure, and the second-source/third-source split for R1 at +10K.

Links: [scorecard_R0_full.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R0_full.md) · [json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_10/scorecard_R0_full.json)

## 2026-11-23 (16:08) — I3 (smoke read): R1 (the source fix) at +5K steps already beats the swap-only model at the moved source

**Headline:** R1, the model trained with several source positions per room, reached its first scored checkpoint after only 5K steps. At the moved source it is already as good as the swap-only model (RCP) after six times as much training, and better above 100 Hz: at the clean third source, in-family, 0.648 / 0.342 / 0.202 against RCP's 0.604 / 0.266 / 0.147 (25-100 / 100-200 / 200-300 Hz), and against "leave the source alone" 0.285 / 0.231 / 0.177 it is ahead in every band. Unseen, 0.473 / 0.329 / 0.199 against RCP 0.439 / 0.273 / 0.154. This is a small-sample reading: the 16-room "lite" set and one early checkpoint.

Off-lattice, frozen scorer; scorecard-lite rooms (8 in-family, 8 unseen) at the second and third source; paired bootstrap over rooms (10k, seed 20261013). "gap" = the model's static r on the original room minus its r at the moved source.

| source | set | R1 @+5K (125K) | RCP @+30K | CTL @+30K | ignore the edit | R1 - CTL (25-100 / 100-200 / 200-300) | R1 - RCP |
|---|---|---|---|---|---|---|---|
| second | in-family | 0.539 / 0.241 / 0.152 | 0.503 / 0.189 / 0.102 | 0.234 / 0.067 / 0.041 | 0.336 / 0.197 / 0.249 | +0.305 / +0.175 / +0.111 | +0.036 [-0.033, +0.106] / +0.053 [+0.033, +0.076] / +0.050 [+0.033, +0.066] |
| **third** | **in-family** | **0.648 / 0.342 / 0.202** | 0.604 / 0.266 / 0.147 | 0.220 / 0.071 / 0.028 | 0.285 / 0.231 / 0.177 | +0.427 [+0.358, +0.490] / +0.271 / +0.174 | +0.044 [+0.017, +0.071] / +0.076 / +0.055 |
| second | unseen | 0.454 / 0.204 / 0.168 | 0.448 / 0.173 / 0.116 | 0.158 / 0.074 / 0.042 | 0.381 / 0.229 / 0.229 | +0.296 / +0.130 / +0.126 | +0.007 [-0.057, +0.071] / +0.032 / +0.052 |
| **third** | **unseen** | **0.473 / 0.329 / 0.199** | 0.439 / 0.273 / 0.154 | 0.241 / 0.135 / 0.041 | 0.314 / 0.276 / 0.200 | +0.232 / +0.193 / +0.157 | +0.034 [-0.001, +0.072] / +0.056 / +0.044 |

- **[Added 16:58] Static accuracy at +5K (standard report): HF 0.536, UF 0.483 / 0.393 / 0.377**, against CTL @150K 0.583 and 0.467 / 0.376 / 0.373 (R1 - CTL: HF -0.047 [-0.054, -0.041], UF 25-100 Hz +0.016 [+0.003, +0.028]). In-family static accuracy is paying for the original-source exposure (each row sees the original source about a quarter as often as before): HF -0.047 at +5K. The pre-registered harm kill at +10K is a drop of more than 0.05 (R1's HF is -0.039 against CTL @130K, the same step count from R0). RC1 (the no-harm rule at +30K) allows up to 0.02; the +10K read is running.
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

