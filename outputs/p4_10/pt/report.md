# P4-10 P-1: R1-P(t) (R1 @150K + mean-value penalty, 10K steps) against R1-ctl (D217 / D218)

Updated 2026-11-24 11:31 ET. Reads done: +2500.  **STOPPED: stop rule triggered at 152K (step 152500): HF more than 0.02 below R1-ctl with the CI entirely below zero; R1-P(t) cancelled at 11:30 ET
**

# P-1 read at +2500 steps (R1 @150K + 2500 ; R1-P(t) against R1-ctl at the same step)

1. Static: HF 0.534 vs 0.572, -0.0379 [-0.0476, -0.0285]; UF 25-100 0.508 vs 0.507, +0.0008 [-0.0076, +0.0090]; UF 100-200 0.408 vs 0.409, -0.0015 [-0.0089, +0.0067]; UF 200-300 0.391 vs 0.400, -0.0081 [-0.0145, -0.0012]
2. Mean-value residual 25-100 Hz (single radius, held-out centres, 15 rooms): truth 0.0074; raw R1-P(t) 0.0290 vs R1-ctl 0.0356 (difference -0.0067 [-0.0087, -0.0047]); raw over truth 3.90 vs 4.80
2. Mean-value residual 100-200 Hz (single radius, held-out centres, 15 rooms): truth 0.0369; raw R1-P(t) 0.0592 vs R1-ctl 0.0852 (difference -0.0260 [-0.0313, -0.0206]); raw over truth 1.60 vs 2.31
2. Mean-value residual 200-300 Hz (single radius, held-out centres, 15 rooms): truth 0.0301; raw R1-P(t) 0.0537 vs R1-ctl 0.0707 (difference -0.0169 [-0.0212, -0.0126]); raw over truth 1.79 vs 2.35
3. Band-limited residual over the truth's at 25-100 Hz: R1-P(t) 2.77, R1-ctl 2.89 (100-200: 1.25 / 1.83; 200-300: 1.23 / 1.69; 0d for R0 was 2.78 at 25-100)
4. Late floor common to the room at 63 Hz (E3): level R1-P(t) 0.1116 vs R1-ctl 0.0850 (truth 0.0386; difference +0.0267 [+0.0128, +0.0405]); model / truth as a ratio of the means (the published E3 form; R1 2.4) 2.89 vs 2.20; mean of room ratios 3.96 vs 3.06
   D1 63 Hz: EDT error 17.2% vs 14.6% (difference +2.59% [+1.80%, +3.37%]); floor -30.4 vs -35.0 dB (difference +4.60 [+4.19, +4.99])
   D1 125 Hz: EDT error 33.4% vs 24.3% (difference +9.06% [+7.82%, +10.37%]); floor -24.5 vs -29.4 dB (difference +4.87 [+4.67, +5.09])
   D1 250 Hz: EDT error 53.6% vs 29.3% (difference +24.27% [+23.29%, +25.15%]); floor -22.9 vs -27.1 dB (difference +4.23 [+3.93, +4.51])
5. Raw finite-difference direction (in-family, h = 4 cm, Re rho): 25-100 Hz 0.845 vs 0.829 (+0.0162 [+0.0133, +0.0189]); 100-200 Hz 0.730 vs 0.726 (+0.0037 [-0.0002, +0.0076]); 200-300 Hz 0.724 vs 0.734 (-0.0101 [-0.0146, -0.0059])
6. Bypass check (held-out test-room centres / training-rule centres in 15 training rooms; <= 1.5x): 25-100 Hz R1-P(t) 0.68 (R1-ctl 0.79); 100-200 Hz R1-P(t) 0.56 (R1-ctl 0.55); 200-300 Hz R1-P(t) 0.92 (R1-ctl 0.71)

Stop rule (HF or any UF band more than 0.02 below R1-ctl with the CI entirely below zero): **TRIGGERED** ['HF']

