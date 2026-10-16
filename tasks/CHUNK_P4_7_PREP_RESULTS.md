# P4-7 prep — physics priors for unseen geometry: RESULTS

**Status (2026-10-16): p1, p2, p3 and p5 are done (D165). p2 FAILS as pre-registered, and its cause is diagnosed. p4 is complete (phases A + B merged, D166); phase C (scale) is dropped because p2 failed.** No training was launched. Pre-registration is D164; results are D165 in [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md).

## p1 — orientation audit
[p1_orientation.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/p1_orientation.md)

| corpus | notch sides (train) | source | receivers on the SW-anchored lattice | wall alpha |
|---|---|---|---|---|
| C250 | **NW 242** / rect 8 | (0.5, 1.6) in 250 / 250 | 100% | 0.15 everywhere |
| C500 | **NW 492** / rect 8 | (0.5, 1.6) in 500 / 500 | 100% | 0.15 |
| 15 test shapes | **NW 15** | (0.5, 1.6) | 100% | 0.15 |
| family | NW 180, NE 60, midN 60, SE 60 / rect 60 (no SW, W-mid, E-mid or S-mid) | (0.5, 1.6) in 300 / 300 | 100% | 0.15 |

- **Every L-corpus notch is in the NW corner.**
- **Every room has the source at (0.5, 1.6)**, in the SW quadrant, at least 0.66 m from any reflex corner (median 2.4–2.5 m).
- **Every receiver lattice is anchored at the SW origin**, and L > W in 94–100% of rooms.

**Recorded as a likely contributor to the out-of-family failure (D162).** An L-corpus model has never seen a reflex corner outside NW, nor a source-relative geometry other than "source SW, notch NW". T (midN), DN (NW + SE) and U (NW + NE) all put a notch where it has never seen one.

## p2 — scale invariance: FAILS as pre-registered (diagnosis below)
[p2_scale.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/p2_scale.json)

**FAILS above 20 Hz in all 3 rooms × 2 α:** relative L2 1.22–2.01 against thresholds of 0.016–0.046, and dB r about 0.2–0.36. The α = 1 control is exactly 0. **Scale augmentation is NOT implemented**, as pre-registered.

**Why** ([p2_scale_diagnostics.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/p2_scale_diagnostics.json)):
- **The compliance term never decays.** The pulse-free FDTD response carries a spatially uniform static offset (c0·N ≈ 920–1100, the near-DC compliance term) that does not decay within the 2 s record. It holds 27–41% of the energy beyond αT; the propagating field alone has only 0.1–0.6% there.
- **The stretch truncates it.** The zero-padded stretch cuts the offset off at αT, and the resulting step leaks as roughly 1/f across every band.
- **Removing the offset** leaves 25–36% error.
- **Matching the time windows** leaves 16–18% at α = 0.9 (r 0.93–0.95) and 29–34% at α = 0.8 (r 0.85–0.88). The residual grows with the scale change, consistent with the compliance level not following the geometric law (labelled inference).
- **Any future attempt** must remove the compliance term first (e.g. the DC head's 0–20 Hz part) and re-verify on the propagating field.

## p3 — geometry sensitivity
[p3_sensitivity.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/p3_sensitivity.json)

**Setup.** D111's U room versus the same room with its 14 cm NE notch removed (8 → 6 boundary tokens). Both are simulated at the U's 3680 receivers; 3285 are off the lattice.

| | 50 Hz | 100 Hz | 200 Hz |
|---|---|---|---|
| ground-truth change, mean abs dB | 0.54 | 1.58 | 2.67 |
| PB3 change (ratio to GT; change-field r) | 1.67 (**3.10**; +0.18) | 1.78 (1.13; −0.01) | 2.04 (0.76; +0.09) |
| B5 change (ratio; change-field r) | 0.93 (1.73; +0.10) | 2.46 (1.56; +0.04) | 3.25 (1.22; −0.10) |

- **Verdict (D164 rule: ratio ≥ 3 at ≥ 2 of 3 frequencies): NOT segment-count sensitive** for either model. PB3 over-reacts only at 50 Hz.
- **The finding is the change-field correlation, which is about 0** (−0.10 to +0.18). The models respond to the edit with roughly the right *amount* of change, but in the wrong *places*. This is the same "field shape" signature as t1 and t2 (D162).
- **Arc-length boundary sampling is therefore not indicated** by this test.
- **Ground truth is more sensitive than the notch's size suggests.** Removing a 14 × 48 cm notch moves the field by 1.6–2.7 dB at 100–200 Hz, spread over the whole room, not near the notch.
- **Confound:** the L version is in C250's held-out d_hat slab, and has L < W.

## p4 — loader augmentations (phase A done)
- **`aaf/data/augment_2d.py`:**
  - D4 maps with the wall-alpha table;
  - CCW polygon images in canonical form (the identity on every native room);
  - `is_interior_node`;
  - `transpose_holdout_leaks` (88 / 500 in C500 and 43 / 240 in holdU would leak into the held-out slab under the transposing elements; the flips leak none);
  - `precompute_d4_tables`.
- **`tests/test_augment_2d.py`, 24 / 24 pass, including the manager's test:**
  - **D4 exactness:** all 8 D4 images of an L-room with non-uniform walls reproduce its FDTD field (identity and fx bit-exact; the rest ≤ 1e-11). Teeth: unpermuted walls, or the wrong receiver map, fail.
  - **Reciprocity:** exact in a rectangle; bounded but nonzero (the D65 residual) in a notched room; a factor of 2 at wall nodes.
- **Phase B is merged (c9e49da, D166).**
  - **Flags:** `aug_d4_p`, `aug_d4_elems` (default fx,fy,r180; transposes are refused on slab corpora), `aug_recip_p` and `aug_seed`, all default-off.
  - **Tests:** 72 / 73 pass. They include a frozen-copy check that the OFF `_step` is bit-identical (losses, parameters, RNG states).
  - **Smoke run:** 200 iterations with augmentation on, at 0.76 s/it (no throughput cost).
  - **Reciprocity at corpus resolution:** exact in a rectangle; ≤ 1% in notched rooms ([p4_reciprocity.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/p4_reciprocity.json)).
- **Phase C (scale augmentation): not implemented**, because p2 failed.

## p5 — wall impedance
[p5_wall_impedance.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/p5_wall_impedance.md)

- **The FDTD wall is frequency-independent.** It is a locally reacting Kowalczyk–van Walstijn boundary with β = ξ = (1 − R)/(1 + R) = **tanh(m/4)**, where R = √(1 − α) and m = −ln(1 − α).
- **Robin condition** in the repo's rfft convention: **∂p/∂n_out = −i k β p**.
- **Values:** β = 0.0406 for every P4 wall (α 0.15). Concrete 0.0128, curtain 0.172, absorber 0.292, window 0.58–0.63.
