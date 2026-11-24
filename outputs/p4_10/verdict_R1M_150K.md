# R1-M verdicts at R1M_150K (reference R1_150K)

MM1 (EDT slope in [0.8, 1.2], Pearson >= 0.8; in family, 60 cells, 180 cell-bands): **FAILS** -- slope 0.54, Pearson 0.95
- in-family: 60 cells; slope per octave 63 0.48 / 125 0.58 / 250 0.54; level-change slope 0.45; T20 rx-bands dropped 429477 of 543192; cell-bands dropped 0
- unseen: 64 cells; slope per octave 63 0.48 / 125 0.77 / 250 0.59; level-change slope 0.44; T20 rx-bands dropped 370742 of 505819; cell-bands dropped 0
MM2 (edited |EDT error| <= own unedited + 5 points; absolute levels and JNDs printed): **HOLDS** -- in family edited 22.5% vs unedited 24.6% (margin -2.1 points); unseen 29.0% vs 29.4% (-0.4); JNDs 5% (ISO 3382-1), about 18% measured
MC1 (HF and each UF band >= R1 @150K - 0.02): **HOLDS**; HF 0.568 vs 0.571 delta -0.003 [-0.007, +0.001]; UF 25-100 0.522 vs 0.506; 100-200 0.429 vs 0.409; 200-300 0.407 vs 0.399
Extension reading (D208): lite slope 0.62 Pearson 0.85 against 0.53 / 0.77 at R1M_140K; rise +0.09 -> not met
