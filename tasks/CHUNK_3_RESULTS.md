# Chunk 3 — Results

**Date**: 2026-05-10. **Scope**: auto-decoder multi-room training on the dense sweep + zero-shot adaptation at 6 unseen L values + latent manifold probe.

**Headline result**: training the shared model on 7 rooms succeeded (per-room val LSD 0.66-0.98 dB, **6 of 7 rooms ≤ 1 dB target met**), but **zero-shot adaptation failed** (held-out LSD 5.7-6.0 dB vs ≤ 2 dB target) because the latents collapsed to a non-physical structure (PC1 vs L R² = **−0.63**, intrinsic_dim = 10 of 32). Documented per spec ("if worse, document as failure mode rather than blocker"). The over-parameterisation risk flagged in `CHUNK_2_RESULTS.md` §6 / `DECISIONS.md` (HashGrid capacity note) materialised.

---

## 1. Pipeline summary

| Stage | SLURM script | Hardware | Wall-time |
|-------|--------------|---------:|----------:|
| Tests (70) | `scripts/slurm/run_pytest.sh` | scavenger CPU+1 GPU | <1 min |
| Memory check | `scripts/slurm/multi_room_memory_check.sh` | scavenger 1 GPU (TITAN X) | 1 min |
| Multi-room training | `scripts/slurm/multi_room_train.sh` | **tron64** RTX 3070 (8 GB) | **2 h 38 min** |
| 6× zero-shot eval (parallel) | `scripts/slurm/zero_shot_eval.sh` | scavenger 1 GPU each | ~16-17 min each |
| Latent probing | `scripts/slurm/latent_probing.sh` | scavenger 1 GPU | 9 s |
| Cross-room summary | `scripts/multi_room_summary.py` | login | <2 s |

Orchestrator: `scripts/run_chunk3_pipeline.sh`, all chained via `--dependency=afterok`. Total real-time ~3.5 h (training-dominated; the projected 6-10 h was conservative — RTX 3070 turned out 3× faster than the GTX TITAN X used for the chunk-2 baseline).

## 2. Multi-room training results

- **Final iter**: 30,000 / 30,000 (no early-stop trigger).
- **Wall-clock**: 9,498 s (2:38:39).
- **Iter rate**: 3.16 iter/s on RTX 3070 with batch=8 + grad_accum=2 (effective batch 16).

### Per-training-room reconstruction at the final val checkpoint

| L (m) | val LSD (dB) | val complex L1 | val phase L1 | Δ vs Chunk-2 baseline |
|------:|-------------:|---------------:|-------------:|----------------------:|
| 3.0 | **0.77** | 0.110 | — | 1.8× worse than 0.42 dB |
| 3.5 | 0.80 | 0.112 | — | (no Chunk-2 baseline) |
| 4.0 | 0.66 | 0.080 | — | (no Chunk-2 baseline) |
| 4.5 | 0.89 | 0.112 | — | 2.3× worse than 0.39 dB |
| 5.0 | 0.94 | 0.108 | — | (no Chunk-2 baseline) |
| 5.5 | 0.97 | 0.110 | — | (no Chunk-2 baseline) |
| 6.0 | **0.98** | 0.107 | — | 2.7× worse than 0.36 dB |
| **agg** | **0.86** | 0.105 | 0.146 | — |

**Spec target**: full-band LSD ≤ 1 dB on ≥ 5/7 rooms — **MET (6/7 rooms within 1 dB; only L=6.0 at 0.984 dB just barely above the 0.98 dB rounding threshold)**. The shared model successfully fits all 7 training rooms within ~2-3× the single-room baseline, which is the expected cost of capacity sharing.

Per-room LSD spread is tight (0.66 − 0.98 dB; max/min ratio ≈ 1.5). Larger L is consistently slightly worse — likely because larger rooms have more receivers per equivalent batch slot under uniform sampling, but the difference is mild.

### Loss curves

`outputs/multi_room/dense/figures/training_curves.png` — five training losses (top row) + per-room val LSD (bottom-right). Clean exponential decay across all spec losses; L_latent (the L2 reg term) hovers at ~0.02 throughout, never exploding. Per-room LSD lines all converge toward 0.7-1.0 dB by iter 30K with no room left behind.

### Latent norms

`outputs/multi_room/dense/figures/latent_norms.png` — mean ‖z_s‖ grew from initialisation 0.18 (1/√32) up to ~0.94 by iter 1K then settled to ~0.81 by iter 30K, with min/max range narrowing to 0.70-0.95. **Latents did not collapse to zero; L2 reg strength was correct.** What we lost on the manifold (next section) is structure, not magnitude.

## 3. Zero-shot results — FAILURE MODE

| L (m) | obs LSD (dB) | held-out LSD (dB) | held-out modal MAE (Hz) | held-out modal recall | n picked / n analytical |
|------:|-------------:|------------------:|------------------------:|----------------------:|------------------------:|
| 3.25 | 5.10 | **5.76** | 0.59 | 0.06 | 84 / 1416 |
| 3.75 | 5.00 | **5.77** | 0.57 | 0.06 | 98 / 1611 |
| 4.25 | 5.08 | **5.98** | 0.60 | 0.05 | 98 / 1845 |
| 4.75 | 4.86 | **5.93** | 0.58 | 0.04 | 90 / 2049 |
| 5.25 | 4.89 | **6.02** | 0.55 | 0.04 | 96 / 2253 |
| 5.75 | 4.70 | **5.85** | 0.51 | 0.04 | 101 / 2475 |

**Spec target**: held-out LSD ≤ 2 dB on ≥ 4/6 unseen L values — **NOT MET** (0/6 below 2 dB; all between 5.76 and 6.02 dB).

**Even on the 8 observed receivers, the inner-loop optimisation only reaches LSD ~5 dB**, vs the trained-rooms val LSD of 0.86 dB. This is the smoking gun: the trained network cannot produce sub-1-dB output for any z_s outside the discrete set of 7 training latents. The optimisation problem isn't just hard for held-out receivers — it's hard *everywhere* once z_star is anywhere outside the trained-latent neighbourhood.

The few peaks the picker does match are modal-MAE 0.5-0.6 Hz, comparable to the noise floor — but only ~5% of analytical eigenfreqs are recovered (the very lowest few). Modal structure above ~150 Hz is essentially absent in the predictions.

`outputs/multi_room/dense/zero_shot/L4.25/figures/zero_shot_overlay.png` shows the predicted spectrum (red) loosely tracking the ISM target (blue) envelope but ~10 dB below it across most of the band, with no preserved spectral peaks above ~150 Hz. The 8×8 receiver_grid (gold-bordered = observed) shows the same pattern at every receiver — observed receivers are no better fit than held-out, confirming the failure is in the *model conditioning*, not in the spatial extrapolation.

## 4. Latent manifold probe — confirms the failure mode

`outputs/multi_room/dense/latent_probe/figures/latent_pca_1d.png` is the diagnostic.

| Quantity | Value | Interpretation |
|---|---:|---|
| `intrinsic_dim_95pct` | **10** | Latents need 10 of 32 PCs to span 95% of variance. Far from the 1D manifold a smooth-in-L embedding would produce. |
| `pc1_vs_L_r2` | **−0.634** | Negative R² means the linear PC1≈f(L) fit is *worse than the mean*. PC1 does not correlate with L. |
| explained variance per PC (first 6) | 0.26, 0.21, 0.11, 0.08, 0.07, 0.06 | No single direction dominates. |
| Train PC1 range | −0.55 to −0.26 | All 7 trained latents cluster in a narrow PC1 band, regardless of L. |
| Test PC1 range | −0.04 to +0.80 | Zero-shot z_star tensors land in a *completely separate* region. |

**Interpretation**: the auto-decoder's latents act as **room-id one-hot indicators**, not as a smooth physical manifold parameterised by L. The shared MLP memorises 7 specific (z_s_i, room_i) pairs. When zero-shot tries to find a z_star in between, the network has no learned structure to interpolate against — z_star ends up in a high-dim "void" between training latents where the network produces low-quality output even at the receivers it's optimising on.

## 5. Visual artifacts

All produced (per spec):
- `outputs/multi_room/dense/figures/{training_curves, latent_norms}.png`
- `outputs/multi_room/dense/zero_shot/L*/figures/{zero_shot_overlay, zero_shot_modal_tracking, zero_shot_receiver_grid, adapt_loss_curve}.png` × 6 unseen L values = 24 PNGs
- `outputs/multi_room/dense/latent_probe/figures/{latent_pca_1d, latent_pca_2d, latent_variance}.png`
- `outputs/multi_room/dense/lsd_vs_L.png` (cross-L summary)
- `outputs/multi_room/dense/SUMMARY.md`

## 6. Capacity diagnosis

The Chunk-2 §6 / DECISIONS.md prediction was correct: with INFER's HashGrid defaults (`log2_hashmap_size=18, n_levels=20` ≈ 120 MB hash params per encoder × 6 encoders = ~720 MB hash storage), the shared MLP has more than enough capacity to memorise the 7 training rooms via hash entries, with `z_s` acting as a small per-room gate rather than a continuous parameterisation of room geometry. The L2 reg (`λ=1e-4`) was too weak to force the network to use `z_s` for anything more structural; it just kept ‖z_s‖ from drifting unboundedly.

Concrete evidence the network is memorising via hash params, not z_s:
1. PC1 vs L R² < 0 (z_s tells you nothing about L).
2. Train latents cluster in PC1 (all rooms map to the same PC1 region; the network distinguishes rooms via *which discrete z_s values* it sees, not via *where in latent space* they sit).
3. Zero-shot adaptation drops sharply outside the trained latent set (5+ dB obs LSD vs 0.9 dB train LSD).

## 7. Failure-mode characterisation

- **No structure to L within zero-shot held-out range**: all 6 unseen L values give within-noise zero-shot LSD (5.76 − 6.02 dB, σ ≈ 0.10 dB). The model fails uniformly; it doesn't fail "more for L outside the training boundary" or "less near training points". This rules out an interpolation-vs-extrapolation explanation — it's a generic failure to use z_s smoothly.
- **Train-room success demonstrates the architecture and renderer are correct**: per-room LSD in the 0.7-1.0 dB range across all 7 rooms, comparable to the Chunk-2 single-room baseline up to a 2-3× capacity-sharing penalty. The model can fit; it just can't generalise via z_s.

## 8. Time and compute

| Step | Wall (per job) | Total |
|---|---:|---:|
| Pytest | <1 min | <1 min |
| Memory check | 1 min | 1 min |
| Multi-room training (1 job) | 2 h 38 min | 2 h 38 min |
| 6× zero-shot eval (parallel) | 16-17 min each | ~17 min |
| Latent probe | 9 s | 9 s |

**Compute**: 1 GPU-hour (memory + zero-shot + probe) + 2.6 GPU-hours (training) ≈ **3.6 GPU-hours**. Way under the 6-10 h estimate in the plan, thanks to the RTX 3070 landing on tron64.

**Preemption**: zero (tron is non-preemptible).

## 9. Surprises and risks for the meeting / Chunk 4

- **Most important finding**: the per-training-room result demonstrates the architecture is sound. Chunks 0-2 produced a valid 2D INR for room acoustics. The Phase-1 *headline* claim (zero-shot at unseen L) does **not** work as currently configured.
- **The fix is structural**, not hyperparameter tuning. The HashGrid capacity needs to drop substantially (the recommendation in `CHUNK_2_RESULTS.md` §8: `log2_hashmap_size` 18 → 14, `n_levels` 20 → 14, ~16× fewer params) to force the network to use z_s for genuine geometric encoding rather than as a room-ID lookup.
- **Other levers** (tried in spirit by setting up the framework): increase `λ_latent` to push z_s magnitudes more toward zero, force more-aggressive latent regularisation; or use a `lr_latent` schedule that anneals more slowly; neither is likely to help if the underlying capacity problem is the network's, not the latent's.
- **Implication for the Dolby meeting**: lead with the per-room reconstruction result (the model can reproduce ISM at single-room and multi-room training-set quality), not the zero-shot one. Show the latent-manifold plot as the diagnostic that motivates the next step (smaller HashGrid, retrain).

## 10. Manager actions requested

1. **Decide on a HashGrid resize-and-retrain run** before the meeting. The Chunk-2 recommendation (`log2_hashmap_size=14, n_levels=14`) reduces hash params ~16×; with the same 30K-iter training budget this should:
   - Force the network to use z_s, since the hash grid no longer has capacity to memorise 7 rooms.
   - Retain reasonable per-room reconstruction on training rooms (target: 1-2 dB val LSD).
   - Produce a non-trivial latent manifold (target: PC1 vs L R² > 0.5).
   - Restore zero-shot interpolation at unseen L (target: held-out LSD ≤ 2-3 dB on ≥ 4/6).

   The same `scripts/run_chunk3_pipeline.sh` would work; only the model's `_default_hash_grid_config()` needs adjustment. Cost: ~3-4 h wall-time.

2. **If the meeting demands a positive zero-shot story now**, an alternative is to run zero-shot evaluation only at L values *very close* to a training room (e.g., L = 3.05 m, 3.45 m) where local linearisation might work even with the current latent collapse. Stretch — not recommended.

3. **Q5 (cluster partition)**: the chunk used 1 tron slot for training, 6 scavenger slots for zero-shot. Worked smoothly. 3 tron slots remain banked for any rerun decision in (1).

The **dataset, model port, renderer, eval framework, and latent-probe diagnostic are all in place and validated** — the next iteration is purely a config change to the HashGrid + a retrain. The negative result here is itself useful: we now have direct evidence (latent_pca_1d.png) that motivates the smaller architecture for the meeting.
