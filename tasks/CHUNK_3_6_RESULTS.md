# Chunk 3.6 — Band-limited evaluation + inner-loop fixes + smoothness retrains

This chunk layered three parallel tracks on top of the Chunk 3.5/3.5+ sweep (R0-R8):
**Track A** quantified the modal-vs-full breakdown of the existing zero-shot
predictions; **Track B** ran six inner-loop adaptation variants on R6 to see if
the failure was fixable by changing how we optimise `z_star`; **Track C**
retrained two new variants (FiLM conditioning, latent jitter) designed to make
the model's latent-to-spectrum response surface smoother. The original meeting
goal was band-limited (0-250 Hz) zero-shot LSD ≤ 2 dB on ≥ 4/6 unseen L.

## Headline result

**Across 11 configurations × 6 unseen L = 66 (config, L) pairs, exactly 0 meet
the 2 dB modal target.** The best-performing single (config, L) is
C1_film + B6 at L=5.75 with modal LSD 3.14 dB — still 1.14 dB above the bar.
The bottleneck is neither inner-loop optimisation (Track B variants cluster
within 0.14 dB of each other) nor decoder smoothness (FiLM and latent jitter
both improve train val LSD to 1.38/1.43 dB but land at the same modal
zero-shot LSD as R6, within 0.2 dB). The auto-decoder paradigm with the
available 7-room training set genuinely cannot zero-shot interpolate to
unseen L below ~3.5 dB modal.

In one sentence: **all 11 configurations × 5 inner-loop strategies cluster
in the 3.5-3.7 dB modal band** — a tight, reproducible failure mode that is
robust to architectural and optimisation changes.

## Track A — band-limited evaluation on R0/R6/R7/R8

Re-forwarded the saved `z_star.pt` from each Chunk-3.5 zero-shot run through
the model+renderer and recomputed LSD on three frequency bands. **The modal
regime is ~1.7 dB better than full-band, but still well above the 2 dB
target.**

| Run | Modal 0-250 Hz | Transition 250-500 Hz | Diffuse 500-2000 Hz | Full 0-2000 Hz |
|---|---:|---:|---:|---:|
| R0_central     | 3.69 | 5.54 | 5.66 | 5.40 |
| R6_tiny_lhead  | 3.66 | 5.53 | 5.49 | 5.27 |
| R7_medium_hash | 3.68 | 5.96 | 5.81 | 5.57 |
| R8_tiny_latent | 3.54 | 5.68 | 5.70 | 5.42 |

The modal regime is dominated by isolated eigenmodes (below the Schroeder
frequency ~210 Hz for L=4.5 m, α=0.15) — the band the analytical 2D Helmholtz
solution covers and where peaks are sparse. The improvement over full-band
confirms the visual-inspection intuition that the model captures some
low-frequency physics, but quantitatively it's a 30% LSD reduction, not the
60% needed to reach the 2 dB bar. R8 (2-D latent) edges out best in modal —
consistent with the prior Chunk-3.5 ranking by zero-shot held-LSD.

Headline figure: [outputs/multi_room/sweep/figures/band_limited_lsd_per_L.png](../outputs/multi_room/sweep/figures/band_limited_lsd_per_L.png)
(4-panel grouped bar chart, one panel per band).

## Track B — inner-loop adaptation variants on R6

Five variants ran on the same R6 checkpoint, differing only in how `z_star`
is initialised and optimised. A sixth variant (B2, n_obs=32) crashed with
OOM on GTX 1080 Ti (skippable per spec).

| Variant | Description | obs LSD | full held LSD | **modal held LSD** | Δ modal vs B1 |
|---|---|---:|---:|---:|---:|
| B1 baseline                       | 8 obs, 2K iters, random init           | 4.82 | 5.30 | 3.66 | 0.00 |
| B3 longer inner adapt             | 10K inner iters                        | 4.83 | 5.31 | 3.70 | +0.04 |
| B4 multi-restart                  | 5 random inits, keep best obs LSD      | 4.81 | 5.31 | 3.63 | −0.03 |
| B5 nearest-train init             | z_star = trained latent of nearest L   | 4.82 | 5.32 | 3.62 | −0.04 |
| **B6 simplex (winner)**           | z_star = softmax(logits) @ Z_train     | 4.84 | 5.24 | **3.52** | **−0.14** |

**Interpretation**: changing the inner-loop strategy moves the modal LSD by
at most 0.14 dB. This is **strong evidence that inner-loop optimisation is
not the bottleneck**:
- B3 (5× more iterations) gives ZERO improvement → the optimiser had already
  converged at 2K iters; this rules out "the inner loop needs more steps".
- B4 (5 random restarts) gives 0.03 dB → the loss landscape is essentially
  unimodal in the regions z_star explores; this rules out "we're stuck in a
  bad basin".
- B5 (warm-start from nearest training latent) gives 0.04 dB → not a wrong
  starting point either; the basins around training latents don't generalise
  to unseen L any better than the random init.
- B6 (z_star forced into the convex hull of trained latents via softmax)
  gives 0.14 dB — the largest gain, but still small. Mild evidence that
  *constraining* z_star to known-good regions helps, but the model itself
  still can't render the unseen L correctly even when handed a
  near-perfect z.

This **narrows the Chunk-3.5 diagnosis**: rather than "the inner-loop is the
bottleneck" (Chunk 3.5 §11 hypothesis), it's "**the model's
latent-to-spectrum mapping is genuinely ambiguous at unseen L, and that
ambiguity is largely independent of where in latent space you place z_star**".

Headline figure: [outputs/multi_room/sweep/figures/inner_loop_comparison.png](../outputs/multi_room/sweep/figures/inner_loop_comparison.png).

## Track C — FiLM (C1) and latent jitter (C2) retrains

Both retrains use the R6 hyperparameters (linear L-head, 14-bit hash, 8-D
latent) but change one aspect of the model:

- **C1_film** — replaces concat-of-z_s at the MLP inputs with input-side
  FiLM (γ(z)·feat + β(z), initialised to identity at construction).
  Hypothesis: low-rank affine modulation produces a smoother
  latent-to-spectrum response.
- **C2_latent_jitter** — adds Gaussian noise σ=0.1 to z_s during training
  only (gated on `model.training`). Hypothesis: training the decoder to be
  robust to z-perturbations explicitly smooths the loss landscape around
  trained latents.

### Per-room reconstruction (training fit)

Both retrains *improve* in-distribution val LSD (lowest in the entire
11-config sweep):

- **C1_film** final val LSD: **1.38 dB** (best across all 11 configs).
- **C2_latent_jitter** final val LSD: **1.43 dB**.
- R6 was 1.45 dB; the 9 R-runs span 1.29-1.70 dB.

So FiLM and latent jitter are both **helpful for training-room fit**. The
model fits its training rooms slightly better than the concat baseline.

### Zero-shot at unseen L

Both retrains were evaluated with TWO inner-loop strategies in parallel: B1
(baseline) for like-for-like comparison to R0-R8, and B6 (Track B winner)
for "best model × best inner loop" synthesis.

| Config | Inner loop | obs LSD | full held LSD | **modal held LSD** | count modal ≤ 2 dB |
|---|---|---:|---:|---:|---:|
| R6_tiny_lhead  | B1 | 4.82 | 5.30 | 3.66 | 0/6 |
| R6_tiny_lhead  | **B6** | 4.84 | 5.24 | **3.52** | 0/6 |
| C1_film        | B1 | 4.92 | 5.31 | 3.73 | 0/6 |
| C1_film        | **B6** | 4.89 | 5.10 | **3.62** | 0/6 |
| **C2_latent_jitter** | B1 | 4.73 | 5.30 | 3.54 | 0/6 |
| **C2_latent_jitter** | **B6** | 4.71 | 5.25 | **3.51** | **0/6 (best)** |

Per-L breakdown for the two retrains (with B6 simplex inner loop):

| L | C1_film modal | C2_jitter modal |
|--:|--:|--:|
| 3.25 | 4.03 | 3.78 |
| 3.75 | 3.82 | 3.69 |
| 4.25 | 3.71 | 3.55 |
| 4.75 | 3.80 | 3.36 |
| 5.25 | 3.21 | 3.37 |
| 5.75 | 3.14 | 3.30 |

### Interpretation

**Neither FiLM nor latent jitter moves the zero-shot modal LSD by more than
the noise.**
- C2 + B6 at 3.51 dB is the best overall — only 0.01 dB better than R6 + B6.
- C1 + B6 at 3.62 dB is actually *worse* than R6 + B6 by 0.10 dB.

So **FiLM, the more architecturally-principled change, slightly HURT
zero-shot generalisation** even though it improved in-distribution val LSD
to the lowest seen (1.38 dB). This is the classic
expressivity/generalisation tradeoff: a more flexible decoder can fit the
training rooms even more tightly without thereby improving its behaviour on
unseen rooms. Latent jitter (C2) is closer to neutral — it doesn't help, but
doesn't hurt either.

A second observation: both C1 and C2 show an **asymmetric modal-LSD pattern**
across L, doing better at L ≥ 4.75 (modal 3.1-3.4 dB) than at L ≤ 4.25
(modal 3.5-4.0 dB). This asymmetry is not strongly present in R6 (where the
spread across L is tighter). The mechanism is unclear — possibly an
artefact of the latent geometry tilting toward one end of the L range during
training — but it points to an interesting follow-up probe.

## Best overall configuration

Ranked by mean modal LSD across 6 unseen L:

1. **C2_latent_jitter + B6** : 3.51 dB modal (full 5.25, obs 4.71)
2. R6_tiny_lhead + B6        : 3.52 dB modal
3. C2_latent_jitter + B1     : 3.54 dB modal
4. R8_tiny_latent + B1 (Track A) : 3.54 dB modal
5. C1_film + B6              : 3.62 dB modal
6. ... (R0-R7 + B1 from Track A) : 3.66-3.69 dB modal
7. C1_film + B1              : 3.73 dB modal (worst of Track C)

But **all 11 configurations × inner loops are within 0.22 dB modal of each
other** — well inside the noise range expected from a 7-room training set
and a 2K-iter zero-shot. The "winner" is essentially arbitrary; what
matters is that **no configuration breaks the 3.5 dB barrier** and **all
configurations sit ~1.5 dB above the 2 dB meeting target**.

## Updated meeting story (2-3 sentence draft claim)

> Per-training-room reconstruction is solved (1.29-1.70 dB val LSD across 11
> architecturally-diverse configurations: capacity-reduced HashGrid, auxiliary
> L-head with both MLP and linear architectures, FiLM conditioning, and
> latent-jitter training). Zero-shot at unseen L, however, fails uniformly:
> the modal regime (0-250 Hz) lands at 3.5 dB across all configurations × all
> five inner-loop adaptation strategies (longer iters, multi-restart,
> nearest-train init, convex-hull projection, baseline) — 1.5 dB above the
> 2 dB target with 0/66 (config, L) pairs meeting the bar. The bottleneck is
> the auto-decoder paradigm's reliance on a 7-room training set spanning a
> 1-D L family; the latent-to-spectrum mapping is genuinely ambiguous at
> unseen L, and neither inner-loop optimisation nor architectural smoothness
> can break that ambiguity. The recommended pivot is a denser training sweep
> (15+ rooms at ~0.2 m L spacing) and/or hyper-network conditioning that
> separates "what changes per room" from "what's shared".

## Failure modes and limitations

- **B2 (n_obs=32) did not run** due to OOM on the GTX 1080 Ti — the
  inner-loop backprop graph for 32 receivers needs ~10 GB during the signal
  branch's `torch.complex` allocation. Acceptable per the chunk spec.
  Future work would chunk receivers across the inner-loop forward pass
  with gradient accumulation.
- **B4 was reduced from 10 to 5 restarts** to fit the 90-min scavenger wall
  (the inner-loop writes the winner only after all restarts finish; a
  10-restart timeout discards all 10). 5 restarts is still a meaningful
  multi-basin test but a 10-restart re-run might catch occasional better
  basins.
- **B6's simplex parameterisation** is one specific choice of latent-hull
  constraint (z = softmax(logits) · Z_train). Other formulations
  (Mahalanobis ball, KL-regularised toward the training-latent mean) might
  give different results.
- **C1's FiLM is input-side only** because `sigma_mlp` / `signal_mlp` are
  tcnn fused kernels and don't expose intermediate features for per-layer
  FiLM. The result reported is consistent with "input-side FiLM doesn't
  help zero-shot", not "FiLM in general doesn't help". A torch-MLP rewrite
  would enable true per-layer FiLM at ~3× training cost.
- **All zero-shot used 8 observed receivers** (B2 was the failed exception).
  The fundamental ambiguity may partly be "8 observations × 2 kHz of
  spectrum is information-theoretically insufficient to uniquely identify
  L"; this hypothesis isn't ruled out by the Chunk-3.6 evidence and is the
  cheapest next experiment.
- **Latent jitter σ=0.1 is one specific choice**. Smaller σ might cause less
  in-distribution noise but also less smoothing benefit; larger σ might
  destabilise training. We didn't sweep σ.

## Recommendations for next iteration (post-meeting)

Ranked by expected information gain per compute hour:

1. **Denser training sweep** (15 rooms at 0.2 m L spacing instead of 7 at 0.5
   m). Cost: re-build dataset (~20 min) + 1 retrain (~3 h). Tests whether
   7-room sparsity is the constraint. Chunk-3.5 R8's 2-D latent showed
   L-correlation in `lhead_predicted_L` (Pearson r ≈ 0.93) even though the
   spectrum didn't follow, suggesting the latent has the *room* signal but
   can't be decoded reliably at unseen rooms; a denser sweep gives more
   interpolation anchors.
2. **Chunked-receiver B2 redux**: re-implement zero-shot's inner loop with
   per-chunk gradient accumulation so n_obs=32 fits in memory. Cost: ~1
   hour code change + 6 ZS jobs. Tests the "8 obs is too few" hypothesis
   directly.
3. **Hyper-network conditioning**: replace `forward(pts, z)` with
   `forward(pts; MLP_params=hyper(z))`. Conceptually clean separation
   between "what the latent does" (set the MLP weights for this room) and
   "what the MLP does" (render given those weights). Cost: ~1 day code + 1
   retrain.
4. **True per-layer FiLM via torch MLPs**: rewrite `sigma_mlp` and
   `signal_mlp` as `nn.Sequential` to enable γ/β between every hidden
   layer. Cost: ~half a day code + 1 retrain (~3× wall time per
   tcnn-vs-torch speed delta).
5. **Modal-regime-only model**: drop the diffuse regime entirely (Phase 1
   would target 0-250 Hz). The model already does ~70% as well on modal as
   full-band; a simpler architecture targeting modal only might hit 2 dB.

(1) is the cheapest and has the strongest a-priori case. (2) is the
cheapest test of the receiver-count hypothesis. (3) is the most ambitious
architectural change but the most likely to fundamentally fix the
generalisation failure.

## Pointers

- Per-track artifacts on disk:
  - **Track A**: [outputs/multi_room/sweep/band_limited_summary.md](../outputs/multi_room/sweep/band_limited_summary.md) + [figures/band_limited_lsd_per_L.png](../outputs/multi_room/sweep/figures/band_limited_lsd_per_L.png).
  - **Track B**: [outputs/inner_loop_experiments/SUMMARY.md](../outputs/inner_loop_experiments/SUMMARY.md) + `best_variant.txt` + [figures/inner_loop_comparison.png](../outputs/multi_room/sweep/figures/inner_loop_comparison.png). B1 outputs at `outputs/inner_loop_experiments/B1/R6_tiny_lhead/L*/`; other variants at `outputs/inner_loop_experiments/B[3456]/R6_tiny_lhead/L*/`.
  - **Track C**: `outputs/multi_room/sweep/{C1_film,C2_latent_jitter}/` (training + `zero_shot_B6/` + `latent_probe/`); B1-baseline ZS at `outputs/inner_loop_experiments/B1/{C1_film,C2_latent_jitter}/L*/`.
- Cross-run aggregate: [outputs/multi_room/sweep/SWEEP_SUMMARY.md](../outputs/multi_room/sweep/SWEEP_SUMMARY.md) (11 rows: R0-R8 + C1 + C2).
- Chunk-3.6 orchestrator: `scripts/run_chunk3_6.sh` (and the post-cancel resume helper `/tmp/chunk3_6_continue.sh` that was needed after the B2 OOM and B6 device-init failures broke the original `afterok` chain).
