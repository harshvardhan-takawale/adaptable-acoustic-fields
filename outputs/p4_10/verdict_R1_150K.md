# R1 verdicts at R1_150K

RS1 (strict, every in-family gap <= 0.10; 15 rooms, off-lattice, paired bootstrap 10k): **FAILS**
- second source gaps (25-100 / 100-200 / 200-300 Hz): 0.203 [0.163, 0.247]; 0.373 [0.321, 0.427]; 0.372 [0.339, 0.410]
- third source gaps (25-100 / 100-200 / 200-300 Hz): 0.128 [0.110, 0.146]; 0.238 [0.216, 0.256]; 0.250 [0.234, 0.266]
Extension rule (lite): worst 0.366 (<= 0.15?), mean 0.2550 against 0.2677 at R1_140K (fell +0.0127, needs >= 0.02) -> not triggered
RS2 (unseen, moved r >= ignore + 0.05 in every band, both sources): **FAILS**
- second: moved r minus ignore: 0.181 [0.109, 0.249]; 0.069 [0.025, 0.113]; -0.010 [-0.038, 0.017]
- third: moved r minus ignore: 0.222 [0.136, 0.313]; 0.082 [0.052, 0.116]; 0.084 [0.058, 0.111]
RS3 (swapped >= normal - 0.05, in family): **FAILS**
- in_family15: normal / swapped 0.909/0.759; 0.683/0.344; 0.571/0.222
- unseen16: normal / swapped 0.506/0.465; 0.409/0.289; 0.399/0.238
RC1 (HF and each UF band >= CTL @150K - 0.02): **HOLDS**; HF 0.571 vs 0.583, delta -0.013 [-0.019, -0.007]; UF 25-100 0.506 vs 0.467; 100-200 0.409 vs 0.376; 200-300 0.399 vs 0.373
Q28 fallback to R0: False
