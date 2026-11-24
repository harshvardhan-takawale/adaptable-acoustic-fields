# R1-S verdicts at R1S_150K (reference R1_150K)

- in-family (SS1; full set): 15 rooms; r_delta 0.808 / 0.571 / 0.410 vs 0.776 / 0.554 / 0.398; delta +0.032 [+0.018, +0.047] / +0.017 [+0.010, +0.025] / +0.012 [+0.007, +0.018]
- unseen (descriptive): 16 rooms; r_delta 0.313 / 0.231 / 0.236 vs 0.277 / 0.205 / 0.213; delta +0.036 [+0.024, +0.050] / +0.026 [+0.020, +0.032] / +0.022 [+0.016, +0.029]
SS1 (>= +0.05 with CI > 0 in >= 2 of 3 bands): **FAILS** (0 bands met)
SC1 (HF and UF >= R1 - 0.02): **HOLDS**; HF 0.572 vs 0.571 +0.001 [-0.003, +0.004]
Extension (lite): second-best band +0.0225 (>= 0.00?), band mean +0.0304 against +0.0215 at R1S_140K (rise +0.0089, needs >= 0.02) -> **not triggered**
