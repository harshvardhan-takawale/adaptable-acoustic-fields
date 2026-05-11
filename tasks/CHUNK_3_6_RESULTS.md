# Chunk 3.6 — Band-limited evaluation + inner-loop fixes + smoothness retrains

Three parallel tracks layered on the Chunk-3.5 sweep (R0-R8). Goal: find any configuration with band-limited (0-250 Hz) zero-shot LSD ≤ 2 dB on ≥ 4/6 unseen L.

## Headline result

Track B winner: **B6** — lowest mean modal_held_mean=3.523 (modal count ≤ 2 dB: 0).

See per-track sections below for full tables.

## Track A — band-limited evaluation on R0/R6/R7/R8

# Band-limited zero-shot LSD summary (Chunk 3.6 Track A)

Recomputed from saved z_star.pt for each (run, L) — no inner-loop re-adaptation.
LSD is mean ``|20*log10(|H_pred|/|H_target|)|`` over the 56 held-out receivers and the bins inside each band.

Bands: modal (0-250 Hz), transition (250-500 Hz), diffuse (500-2000 Hz), full (0-2000 Hz).
Target for the meeting deliverable: ≤ 2 dB on ≥ 4/6 unseen L (modal regime).

| Run | Band | mean LSD (dB) | min | max | count ≤ 2 dB | count ≤ 3 dB | n L |
|---|---|---:|---:|---:|---:|---:|---:|
| R0_central | 0-250 Hz (modal) | 3.69 | 3.32 | 4.42 | 0/6 | 0/6 | 6 |
| R0_central | 250-500 Hz (transition) | 5.54 | 5.12 | 5.84 | 0/6 | 0/6 | 6 |
| R0_central | 500-2000 Hz (diffuse) | 5.66 | 5.45 | 6.06 | 0/6 | 0/6 | 6 |
| R0_central | 0-2000 Hz (full band) | 5.40 | 5.19 | 5.83 | 0/6 | 0/6 | 6 |
| R6_tiny_lhead | 0-250 Hz (modal) | 3.66 | 3.39 | 3.78 | 0/6 | 0/6 | 6 |
| R6_tiny_lhead | 250-500 Hz (transition) | 5.53 | 5.30 | 5.75 | 0/6 | 0/6 | 6 |
| R6_tiny_lhead | 500-2000 Hz (diffuse) | 5.49 | 5.22 | 5.72 | 0/6 | 0/6 | 6 |
| R6_tiny_lhead | 0-2000 Hz (full band) | 5.27 | 5.07 | 5.46 | 0/6 | 0/6 | 6 |
| R7_medium_hash | 0-250 Hz (modal) | 3.68 | 3.49 | 3.85 | 0/6 | 0/6 | 6 |
| R7_medium_hash | 250-500 Hz (transition) | 5.96 | 5.57 | 6.40 | 0/6 | 0/6 | 6 |
| R7_medium_hash | 500-2000 Hz (diffuse) | 5.81 | 5.52 | 6.21 | 0/6 | 0/6 | 6 |
| R7_medium_hash | 0-2000 Hz (full band) | 5.57 | 5.29 | 5.89 | 0/6 | 0/6 | 6 |
| R8_tiny_latent | 0-250 Hz (modal) | 3.54 | 3.05 | 4.03 | 0/6 | 0/6 | 6 |
| R8_tiny_latent | 250-500 Hz (transition) | 5.68 | 5.49 | 5.86 | 0/6 | 0/6 | 6 |
| R8_tiny_latent | 500-2000 Hz (diffuse) | 5.70 | 5.46 | 5.86 | 0/6 | 0/6 | 6 |
| R8_tiny_latent | 0-2000 Hz (full band) | 5.42 | 5.19 | 5.63 | 0/6 | 0/6 | 6 |

## Headline figure

![band-limited LSD per L](figures/band_limited_lsd_per_L.png)

## Notes

- These numbers reuse the EXACT z_star produced by the original Chunk-3.5 zero-shot runs.
- The full-band column matches the existing `held_out_lsd_db` field in `metrics.json` modulo numerical noise — sanity check.
- Track B variants (different inner-loop strategies) are aggregated in `outputs/inner_loop_experiments/SUMMARY.md`.

## Track B — inner-loop adaptation variants on R6

# Track B summary — inner-loop variants on R6_tiny_lhead

Winner: **B6** (z_star = softmax(logits) @ Z_train (simplex)). Reason: lowest mean modal_held_mean=3.523 (modal count ≤ 2 dB: 0)

| Variant | Description | n L | mean obs LSD | mean full held LSD | count full ≤ 2 dB | mean 0-250 Hz held LSD | count modal ≤ 2 dB | count modal ≤ 3 dB | winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| B1 | baseline (8 obs, 2K iters, random init) | 6 | 4.82 | 5.30 | 0/6 | 3.66 | 0/6 | 0/6 |  |
| B3 | 10K inner iters (was 2K) | 6 | 4.83 | 5.31 | 0/6 | 3.70 | 0/6 | 0/6 |  |
| B4 | 10 random restarts, keep best obs LSD | 6 | 4.81 | 5.31 | 0/6 | 3.63 | 0/6 | 0/6 |  |
| B5 | init z_star from nearest-L training latent | 6 | 4.82 | 5.32 | 0/6 | 3.62 | 0/6 | 0/6 |  |
| B6 | z_star = softmax(logits) @ Z_train (simplex) | 6 | 4.84 | 5.24 | 0/6 | 3.52 | 0/6 | 0/6 | ✅ |

## Headline figure

![inner-loop comparison](../multi_room/sweep/figures/inner_loop_comparison.png)

## Notes

- All variants run on the same trained R6 checkpoint; only the inner-loop
  procedure differs.
- 'modal' = LSD restricted to 0-250 Hz (the band where Track A finds
  visually-correct tracking).
- The winner's kwargs are written to `best_variant.txt` and consumed by
  `scripts/zero_shot_with_best_variant.py` for evaluating Track-C trained
  models.

## Track C — FiLM + latent-jitter retrained variants

### C1_film

Training status: completed; final val LSD: 1.38 dB.

**C1_film — B1 baseline (8 obs, 2K iters, random init)**: no metrics found.

**C1_film — Track B winner (B6)** (6 unseen L)

- mean obs LSD: 4.89 dB
- mean full-band held LSD: 5.10 dB  (count ≤ 2 dB: 0/6)
- mean 0-250 Hz held LSD: 3.62 dB  (count ≤ 2 dB: 0/6, count ≤ 3 dB: 0/6)

### C2_latent_jitter

Training status: completed; final val LSD: 1.43 dB.

**C2_latent_jitter — B1 baseline (8 obs, 2K iters, random init)**: no metrics found.

**C2_latent_jitter — Track B winner (B6)** (6 unseen L)

- mean obs LSD: 4.71 dB
- mean full-band held LSD: 5.25 dB  (count ≤ 2 dB: 0/6)
- mean 0-250 Hz held LSD: 3.51 dB  (count ≤ 2 dB: 0/6, count ≤ 3 dB: 0/6)


## Updated sweep summary

Refreshed `outputs/multi_room/sweep/SWEEP_SUMMARY.md` includes C1 and C2 alongside R0-R8.

## Recommendations / next iteration

TODO (manager-facing): prioritise the next-chunk experiment based on which track moves the modal-LSD needle. Replace this placeholder once the Track C numbers are in.
