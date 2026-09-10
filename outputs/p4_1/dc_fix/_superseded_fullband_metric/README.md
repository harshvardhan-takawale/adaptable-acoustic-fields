# Superseded: the first DC-masked run, scored on the wrong band

This run's WEIGHTS were fine -- the loss mask worked correctly from iteration 1. What was wrong
was the metric used to judge it.

`validate()` computed `lsd_db` over the FULL band, including bins 0-39, which this run
deliberately never trains on. In the FDTD corpus those bins sit ~37 dB above the strongest room
mode, so the full-band number is dominated by them:

| iter | lsd_db full-band vs original A2 | L_amp (DC-immune) | L_phase |
|---|---|---|---|
| 2000 | +1.017 worse | -0.0419 **better** | 0.610 vs 0.702 **better** |
| 4000 | +1.747 worse | -0.0082 **better** | 0.513 vs 0.630 **better** |
| 6000 | +1.648 worse | -0.0142 **better** | 0.485 vs 0.576 **better** |

The masked run was improving on every term it was actually being asked to fit, while the metric
that drives early stopping and checkpoint selection said it was getting worse. Its trend was
flat/up (6.299 -> 6.591 -> 6.440), so the 0.3%-over-10k early-stop rule would very likely have
killed it at iter 20000 -- and even without that, "best checkpoint" was being chosen on a
criterion the model was not optimizing.

Fixed by making `lsd_db` respect the loss band whenever a low cut is active, with the full-band
value still recorded as `lsd_db_full_band`. Strict no-op when `loss_band_lo_hz` is 0, so every
other run is unaffected. Kept here as the evidence for that decision.
