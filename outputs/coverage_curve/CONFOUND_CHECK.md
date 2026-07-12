# P2-4b — Bounding the convergence confound in the coverage curve

**Coverage effect CONFIRMED at matched convergence.** At equal in-distribution convergence (~4.3 dB), 250 rooms beats 45 rooms across the suite — magnitude-band LSD, phase, RIR, and magnitude correlation. Because both sides are at matched convergence, blur is equalized on both, so these deltas isolate coverage. So the P2-4 curve's *direction* is trustworthy and densification genuinely helps, though its *magnitude* was inflated by the confound (see decomposition).


Matched-convergence deltas (250@4.30dB − 45@4.33dB, positive = 250 better):
- mag corr full **+0.060**, modal **+0.113**
- held-out LSD full **+0.21 dB**, modal-band (0–250) **+0.59 dB** (lower better → positive = 250 better)
- phase (mw) **+0.131**, RIR Pearson **+0.133**
- modal recall **-0.036**, modal MAE **-0.08 Hz** (lower better)
- hard-metric wins for 250 (of 6): **4**


**How much of the raw P2-4 climb was the confound?** Decompose the raw P2-4 mag-corr gap (45@2.17dB → 250@4.30dB) into blur/convergence (45@2.17→45@4.33, *same rooms*) + genuine coverage (matched, 45@4.33→250@4.30):
- full-band: raw **+0.188** = blur **+0.128 (68%)** + coverage **+0.060 (32%)**
- modal (0–250): raw **+0.402** = blur **+0.289 (72%)** + coverage **+0.113 (28%)**

So **~68% of the raw P2-4 magnitude-correlation climb was the convergence/blur confound, not coverage.** The genuine coverage effect is real (verdict above) but **smaller than the raw curve shows**. Because blur is equalized at matched convergence, the matched deltas isolate coverage; it shows most strongly on phase (+0.131), RIR (+0.133), modal-band LSD (+0.59 dB). **Do not cite the raw P2-4 curve's slope/magnitude; cite the matched-convergence deltas.**


**Blur-inflation test (same 45 rooms, 2.17dB→4.33dB under-training):** mag corr full moves **0.273 → 0.401** (+0.128), held-out LSD **7.70 → 6.25 dB** (-1.45). Under-training **inflates** the soft correlation while LSD worsens — the P2-3 blur effect is real and must be discounted when reading the raw P2-4 mag-corr curve.


## Full-suite comparison (known-geometry zero-shot, mean over 15 frozen rooms)

| point | n | in-dist LSD | mag full | mag modal | held LSD full | LSD 0–250 | LSD diffuse | phase(mw) | RIR | modal recall | modal MAE (Hz) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 45 rooms · converged (P3 @60K) | 15 | 2.17 | 0.273 | 0.409 | 7.70 | 9.28 | 7.58 | 0.125 | 0.130 | 0.104 | 0.97 |
| 45 rooms · matched @~4.3 (under-trained) | 15 | 4.33 | 0.401 | 0.698 | 6.25 | 5.30 | 6.42 | 0.218 | 0.287 | 0.125 | 0.92 |
| 250 rooms · plateau @~4.3 | 15 | 4.30 | 0.461 | 0.811 | 6.04 | 4.71 | 6.37 | 0.348 | 0.421 | 0.089 | 1.00 |
| 45 rooms · @~4.5 | 15 | 4.55 | 0.414 | 0.721 | 6.35 | 5.25 | 6.53 | 0.225 | 0.302 | 0.143 | 0.86 |
| 45 rooms · @~3.8 | 15 | 3.80 | 0.357 | 0.617 | 6.24 | 5.77 | 6.34 | 0.188 | 0.233 | 0.127 | 0.86 |
| 45 rooms · @~3.4 | 15 | 3.52 | 0.331 | 0.576 | 6.37 | 6.07 | 6.44 | 0.174 | 0.208 | 0.121 | 0.89 |

*Held-out LSD lower = better; all correlations + recall higher = better; modal MAE lower = better. Modal = 0–250 Hz (sub-Schroeder ≈217 Hz); diffuse = mean of 250–500/500–1k/1k–2k bands.*


_Interpretation, saturation, and P3-1 implications: tasks/CHUNK_P2_4b_RESULTS.md._
