# Chunk P4-6 — Geometry transfer: RESULTS (live)

**Status: IN PROGRESS (2026-10-15, 18:05).** Step 0 and Step 1 are complete. The arms g1, g2 and g4 are training on tron; g3 is queued for an L40S. Decisions D160–D162 are in [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md).

**The chunk's premise** (the manager's correction, D160(a)): at 200–300 Hz the binding limit is geometry generalisation, not fit. Extra capacity raised training-room fit (r 0.648 → 0.772) but widened the train-to-test gap (0.18 → 0.28; D157). Every arm reports in-family AND out-of-family metrics.

---

## Step 0 — housekeeping (D160)

| item | result |
|---|---|
| Checkpoint keep-list | Written first ([ckpt_keeplist.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_6/housekeeping/ckpt_keeplist.md)). All 47 kept checkpoints load through `load_model`; 40.3 GB deleted. |
| Caches | Dense render 25 GB, A2 9.2 GB and speckle 1.3 GB deleted. The B5 / PB3 @60K dense caches were kept for t1, then deleted. Free space 32 → 102 GB. |
| `validate()` | Renders in eval mode and restores training mode afterwards. Tested: two validations at fixed weights agree exactly. |
| Retention (new arms) | Every 10K + final, plus 4K for D135's kill signal (tested). |
| Physical projection | Library default (`aaf/eval/physical_projection.py`), labelled post-processing; raw and projected are reported side by side (tested). |
| g2 | FAM had no held-out family, so the user decided to hold out U: 240 train (rect / L / T / DN), 70 U out-of-family. |

## Step 1 — diagnostics on existing checkpoints

**t3, narrow doorway (D161).** For TBdc @30K at a = 0.15 m, the mean |level-difference error| is:
- 0–20 Hz: 0.30 dB (2% of the per-band error);
- 20–100 Hz: 4.00 dB;
- 100–200 Hz: 4.76 dB;
- 200–300 Hz: **8.16 dB (47%)**.

The gate fails, so **the two-region DC head is closed**. The deficit is the aperture representation in the acoustic band. The model also leaks too much sound through the 15 cm doorway: it predicts −4 to −13 dB where FDTD has −10 to −16.5 dB.

**t1, lag test (D162(a)).** **Not mode misplacement.**
- Test rooms show large per-receiver spectral misalignment (3.7–4.2 Hz at 250 Hz, about 4× the training rooms), but its sign is random within each room.
- 0/15 test rooms are systematic at 250 Hz.
- The transfer error is field shape, so the inverse-span features are not launched.

**t2, out-of-family baseline (D162(b)).** Off-lattice, raw (projection changes nothing, ±0.003):

| model | in family (rect, L): r 25–100 / 100–200 / 200–300 | out of family (T, DN, U) | U at 100 Hz |
|---|---|---|---|
| PB3 @60K | 0.902 / 0.658 / 0.455 | **0.600 / 0.347 / 0.329** | +0.11 |
| B5 @60K | 0.897 / 0.639 / 0.458 | **0.624 / 0.367 / 0.314** | +0.24 |

**Out of family, the collapse is at LOW frequency** (25–100 Hz r −0.30, LSD +3 dB), the band that is excellent in family. Band separation does not transfer across families.

## Step 2 — arms (config-only)

| arm | what | pre-registered (manager) | status |
|---|---|---|---|
| g1 | PB3 on the 500-shape corpus, 60K | test 200–300 Hz r >= 0.53 AND train-to-test gap < 0.22 | training |
| g2 | PB3 on the family corpus, U held out, 60K | report held-out U per band beside the in-family families | training |
| g3 | PB3 + 4× receivers, 120K cosine | test 200–300 Hz r >= 0.52 by 120K; reads at 30K / 60K / 120K | queued (L40S) |
| g4 | B5 + hash grid capped at 6 levels, 60K | training-room fit falls, test r within CI of B5 or higher, so the gap shrinks; speckle d1 | training |

**Gated, not launched:** the two-region DC head (t3 closed it) and inverse-span geometry features (t1 closed it).

**Standard report per arm:**
- off-lattice Gate-2 with CI, and the gap;
- per-band test r and LSD;
- training-room fit and the training-to-test gap per band;
- out-of-family per band (`scripts/p4_6_family_oof.py`);
- modal RIR;
- speckle d1, raw and projected.
