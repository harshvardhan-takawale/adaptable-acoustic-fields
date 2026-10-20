# P4-8 status report for the manager — 2026-10-19, 22:45 EDT

**State.**
- **Wave 1 is complete:** all eight arms have their final reads (D178).
- **S2 has its 60K interim read.** It reached 85.4K before being preempted and is requeued on scavenger. Its 120K standard report launches automatically when it finishes; this report does not wait for it.
- **Nothing else is training.**
- **Decisions:** D173–D178 in [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md).
- **Write-up:** [CHUNK_P4_8_RESULTS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_P4_8_RESULTS.md).
- **Primary table:** [primary_table.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/primary_table.md).

**Protocol for every number below.**
- Receivers: off-lattice (Rule 11).
- **HF** = in-family test r at 200–300 Hz on the 15 frozen shapes.
- **UF** = unseen-family r at 25–100 / 100–200 / 200–300 Hz on the frozen 16-room set (8 U + 8 C3 = NW+NE+SE; [manifest](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweeps_2d_mat/p4_8_unseen_manifest.json)).
- CIs: paired bootstrap (10k, seed 20261013) over shapes or rooms, against GX @60K.
- Network models only (no modal-prior or solver results, per scope).

---

## 1. Headline

1. **R0 (GX at matched exposure, 120K) passes and is the meeting model.**
   - HF 0.582 equals G1's 0.581 (+0.001 [−0.005, +0.006]).
   - UF 0.468 / 0.380 / 0.371 beats G1 by +0.084 [+0.049, +0.123] / +0.080 / +0.073, all CIs above 0.
   - GX's in-family cost was exposure, not dilution.
2. **No multi-band or loss arm passes.** Wave 2 is skipped as pre-registered, and the LS2 control is not needed.
   - **MB1 comes closest:** UF +0.033 [+0.013, +0.055], HF −0.024. Its wavelength-matched low-band encoder helps the 25–100 Hz unseen field, but by less than the +0.05 threshold.
3. **More diverse rooms move the unseen-family score; more training steps do not.**
   - **S2** (2000 diverse rooms) at its 60K midpoint has the largest UF gain of any arm: +0.042 [+0.024, +0.058], with the best C3 (0.485). Its HF (0.448) is still below its 0.484 floor mid-schedule; the decisive 120K read is pending.
   - **R0's UF is flat from 30K to 120K** (0.480 → 0.468) while its HF climbs 0.440 → 0.582.

## 2. Primary table (final reads; S2 interim)

| model | HF | UF 25–100 / 100–200 / 200–300 | U | C3 | test LSD dB | gap 200–300 | modal RIR | verdict |
|---|---|---|---|---|---|---|---|---|
| G1 @60K (ref) | 0.581 | 0.384 / 0.300 / 0.298 | 0.444 | 0.323 | 1.53 / 3.67 / 4.28 | 0.094 | 0.956 | — |
| GX @60K (ref) | 0.504 | 0.460 / 0.389 / 0.375 | 0.480 | 0.439 | 1.91 / 4.22 / 4.67 | 0.025 | 0.936 | — |
| **R0 @120K** | **0.582** | 0.468 / 0.380 / 0.371 | 0.481 | 0.454 | 1.72 / 3.83 / 4.25 | 0.063 | 0.951 | **HOLDS** |
| MB1 @60K | 0.480 | **0.492** / 0.405 / 0.390 | **0.518** | 0.467 | 1.99 / 4.32 / 4.81 | 0.015 | 0.930 | fails (closest) |
| MB2 @60K | 0.503 | 0.457 / 0.389 / 0.391 | 0.487 | 0.427 | 1.85 / 4.22 / 4.69 | 0.026 | 0.937 | fails (= GX) |
| LS1A @60K | 0.524 | 0.443 / 0.354 / 0.347 | 0.465 | 0.420 | 1.89 / 4.20 / 4.79 | 0.037 | 0.939 | fails |
| LS1B @60K | 0.527 | 0.391 / 0.362 / 0.367 | 0.439 | 0.343 | 2.01 / 4.29 / 4.93 | 0.031 | 0.937 | fails |
| LS3 @60K | 0.519 | 0.458 / 0.379 / 0.367 | 0.483 | 0.433 | 1.87 / 4.16 / 4.66 | 0.033 | 0.939 | fails (= GX) |
| LS2A @30K end | 0.368 | 0.452 / 0.384 / 0.336 | 0.452 | 0.451 | 2.64 / 5.05 / 5.32 | −0.003 | 0.883 | fails |
| LS2B @30K end | 0.295 | 0.460 / 0.379 / 0.272 | 0.473 | 0.447 | 2.74 / 5.13 / 5.55 | −0.011 | 0.856 | fails |
| S2 @60K (interim) | 0.448 | **0.502 / 0.419 / 0.400** | **0.518** | **0.485** | 2.20 / 4.52 / 4.94 | 0.000 | 0.922 | 120K pending |

**Paired differences vs GX @60K** (mean [95% CI]):

| arm | HF | UF 25–100 | UF 100–200 | UF 200–300 |
|---|---|---|---|---|
| R0 @120K | +0.078 [+0.070, +0.087] | +0.008 [+0.001, +0.015] | −0.009 [−0.019, +0.001] | −0.004 [−0.025, +0.015] |
| MB1 | −0.024 [−0.029, −0.019] | +0.033 [+0.013, +0.055] | +0.016 [−0.005, +0.037] | +0.015 [−0.008, +0.035] |
| MB2 | −0.001 [−0.004, +0.002] | −0.002 [−0.014, +0.009] | −0.000 [−0.013, +0.013] | +0.015 [+0.004, +0.025] |
| LS1A | +0.020 [+0.016, +0.024] | −0.017 [−0.041, +0.006] | −0.035 [−0.050, −0.020] | −0.028 [−0.054, −0.006] |
| LS1B | +0.023 [+0.017, +0.029] | −0.069 [−0.111, −0.027] | −0.028 [−0.040, −0.016] | −0.009 [−0.025, +0.009] |
| LS3 | +0.015 [+0.011, +0.020] | −0.002 [−0.016, +0.011] | −0.010 [−0.020, +0.001] | −0.008 [−0.021, +0.004] |
| LS2A | −0.136 [−0.144, −0.127] | −0.008 [−0.030, +0.011] | −0.006 [−0.030, +0.017] | −0.040 [−0.064, −0.014] |
| LS2B | −0.209 [−0.225, −0.195] | +0.001 [−0.021, +0.020] | −0.011 [−0.038, +0.015] | −0.103 [−0.127, −0.079] |
| S2 @60K | −0.056 [−0.062, −0.049] | +0.042 [+0.024, +0.058] | +0.030 [+0.008, +0.055] | +0.025 [+0.003, +0.047] |

## 3. Readings per lever (D178)

- **Exposure (R0).**
  - GX at 120K reaches G1's in-family accuracy and keeps GX's unseen-family advantage.
  - Over the run: HF 0.440 → 0.528 → 0.567 → 0.582 (30 / 60 / 90 / 120K); UF 0.480 → 0.477 → 0.468 → 0.468; the 200–300 Hz train-to-test gap 0.003 → 0.063.
  - Longer training improves in-family accuracy, not unseen-family transfer.
- **Separate per-band models.**
  - **MB2 − GX ≈ 0:** full separation beyond PB3's shared trunk adds nothing.
  - **MB1 − MB2 (wavelength-matched grids):** UF +0.035 [+0.013, +0.060] at 25–100 Hz, +0.016 at 100–200, −0.001 at 200–300; HF −0.023.
  - The pre-registered mechanism (a coarse low-band encoder must represent smooth, global low-frequency fields) holds in direction and in band, but below threshold and at a small in-family cost.
- **LS1 (neighbour differences)** buys in-family accuracy (HF +0.02, CI above 0) at an unseen-family cost (UF −0.017 / −0.069 at 25–100 Hz for λ 0.3 / 1.0).
  - Per-band log ([D176(c)](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md)): the term fell in every band and was stiffest at 200–300 Hz (neighbour phase), not in the low band.
- **LS2 (per-room correlation; one room × 32 receivers per step, 30K) is harmful.**
  - HF −0.14 / −0.21; UF at 200–300 Hz −0.04 / −0.10; test LSD +0.65–0.91 dB; modal RIR 0.88 / 0.86.
  - Training-room fit drops as far as transfer, so the arms under-fit.
  - The λ=0 batching control runs only on a pass, so it was not run. **Whether the damage comes from the loss or from the one-room-per-step batching is unresolved.**
- **LS3 (floored log-amplitude)** matches GX.
- **S2 (twice the diverse rooms; interim).**
  - At its 60K midpoint it beats R0 @60K (same step count) on UF: +0.025 [+0.010, +0.040] / +0.024 / +0.017.
  - It costs HF −0.080 at that point. R0's HF rose +0.054 from 60K to 120K; the pre-registered read is at 120K.

## 4. Step 0, gates and stretch (D175)

- **Topology audit:** CX500 contains T-like (174) and DN (19) rooms, so T and DN are in-distribution for every GX-corpus arm. P4-7's GX "+0.110 out of family" splits into T +0.051, DN +0.206 and U +0.072, and only U was unseen. ([audit](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/topology_audit.md))
- **Correction to the record:** the two-room U numbers of earlier rounds overstated U (G1 0.718). On the 8-room set G1 reads **0.444**; quote that from now on.
- **S1 Helmholtz gate: FAILED, so S1 was not launched.** ([s1_gate.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/s1_gate.md))
  - The FDTD residual is 0.057.
  - The networks' Helmholtz residual is 4–5× the k² term, but the same in family and out (ratio 1.05 / 1.02, needed ≥ 2).
- **S2 corpus:** CX1000 is built (1000 rooms, 3.4 GB).

## 5. Meeting figures ([manifest](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/meeting/FIGURE_MANIFEST.md))

**Models shown:** G1 @60K, GX @60K, R0 @120K, the best wave-1 arm by the fixed rule, and S2 @60K, labelled "interim read" (the figures could not wait for 120K). Best arm: no arm passed, so it is the largest ΔHF + ΔUF, LS3 (+0.013, with MB1 at +0.009), labelled "no arm passed".

- [F1](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/meeting/F1_freq_curve_in_family.png): in-family off-lattice r vs frequency.
- [F2](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/meeting/F2_freq_curve_unseen.png): unseen-family r vs frequency, U and C3 separately.
- **F3 field maps:**
  - [in family](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/meeting/F3_fields_in_family.png), [U](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/meeting/F3_fields_U.png), [C3](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/meeting/F3_fields_C3.png).
  - Layout: rows 50 / 100 / 200 / 280 Hz; columns FDTD | G1 | R0 | best.
  - The render-time projection is applied and labelled.
  - Rooms: the lower-median by G1.
- [F4](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/meeting/F4_scatter_unseen.png): per-room paired scatter on the unseen set, best arm vs G1. The best arm, LS3, beats G1 in 15 / 15 / 14 of the 16 rooms across the three bands. That reflects the GX corpus, since LS3 ≈ GX, not LS3's loss.

## 6. Incidents (recorded)

- **LS2A / LS2B: preempted, then out of memory on resume** (D176(g), D177).
  - **Fix:** renderer chunks of 4, memory only (both passes identical; loss and batching unchanged); both resumed from 12K / 10K.
  - The resume path now loads checkpoints to CPU (tested, exact).
- **MB1 / MB2 were preempted.** They were held until S2 started, so S2 took the first L40S, as the manager directed; then they completed.

## 7. Awaiting the manager's decision (nothing launched)

1. **S2's 120K read** lands automatically (in about 6 h of running time after it resumes).
   - If it passes (UF ≥ 0.510 and HF ≥ 0.484), diversity is the lever for unseen families at matched per-room exposure.
   - A larger diverse corpus or a longer S2 schedule are the natural next steps.
2. **MB1's low-band mechanism is real but small** (+0.035 UF at 25–100 Hz vs MB2). Options:
   - pair it with S2's data (wave 2's slot was unused); or
   - retune the low-band grid (e.g. a coarser finest cell).
3. **LS2's damage is unattributed** (loss vs batching). The λ=0 control (`configs/sweep_2d_mat/P4_8_LS2C.yaml`, ready) would settle it, if that matters for the record.
4. **M3-fam** (D173) is recorded for the next meeting, as instructed.
