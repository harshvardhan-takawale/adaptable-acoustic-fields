# OPEN_QUESTIONS.md

Numbered, append-only ledger of ambiguities, blockers, and research-direction questions. When a question is answered, move the resolution to `DECISIONS.md` and remove the question here (so this file always reflects what's still open).

Asker: agent that wrote it. Owner: who can answer (manager, user, or "research call — needs experiment").

---

### Q11 — How to unblock zero-shot adaptation? *(refined in Chunk 3.6)*

**Asker**: chunk-3.5+ agent. **Owner**: manager / research call.

Chunk 3.5 + 3.5+ swept 4 architecturally-diverse multi-room auto-decoders (R0/R6/R7/R8: mlp_32 vs linear L-head, 14-bit vs 16-bit hash, 2-D vs 8-D latent). All 4 fully-complete runs **fail zero-shot at 5.21-5.91 dB held-out LSD** full-band (target was ≤ 2 dB on ≥ 4/6 unseen L). The architecture choices made no material difference; the bottleneck has shifted from latent collapse (Chunk 3) to inner-loop adaptation.

**Chunk 3.6 Track A refines the picture**: modal-regime (0-250 Hz) LSD is ~1.7 dB lower than full-band — R0/R6/R7/R8 land at 3.54-3.69 dB modal vs 5.27-5.57 dB full. So the failure is *not* uniform across frequency; the diffuse regime (>250 Hz) dominates. Still, modal is 0/24 (run, L) below the 2 dB target — Chunk 3.6 Tracks B and C aim to close that gap.

Diagnostic evidence (Chunk 3.5+):
- **Even on observed receivers** the inner-loop optimizer can't drive obs LSD below 4.5-5.4 dB (vs training val LSD of 1.3-1.7 dB). 2K iters of Adam(lr=1e-2) on a low-dim z_star can't navigate the model's latent-to-spectrum response surface outside the trained-latent neighbourhood.
- R6's `latent_pca_1d` shows **train latents nearly monotonic with L** (linear L-head IS shaping z_s), but **zero-shot z_star tensors collapse to one region of latent space** (PC1 ≈ -0.5 for all 6 unseen L). The inner loss surface has a single attractor outside the training region.
- R8 (2-D latent) has zero-shot z_star with some L-correlation (Pearson r ≈ 0.93 between true L and lhead_predicted_L from optimized z_star) — but the actual reconstruction quality doesn't improve. The latent moves in the right direction but the model doesn't follow.

Chunk 3.6 (in flight) attacks two of the four ranked strategies from below:
- **Track B** runs all 6 inner-loop variants in parallel on R6: B1 baseline, B2 n_obs=32, B3 10K iters, B4 10 random restarts, B5 nearest-training-latent init, B6 simplex of training latents (convex hull). The winner is identified by mean modal-regime LSD.
- **Track C** retrains two new models with smoothness-promoting changes: C1 FiLM conditioning (input-side γ·feat+β, identity-init), C2 latent jitter σ=0.1 at training time. Each is evaluated with both B1-baseline ZS and the Track B winner.

Concrete experimental paths still on the table (cheap, ranked by cost):
1. **More observed receivers** (Track B/B2; running): 32 instead of 8.
2. **Multi-restart inner adaptation** (Track B/B4; running): 10 random z_star inits.
3. **Longer inner adaptation** (Track B/B3; running): 10K iters instead of 2K.
4. **Latent-hull-constrained adaptation** (Track B/B6; running): SimplexLatent over trained latents.
5. **Nearest-train-latent init** (Track B/B5; running): warm-start from closest training L.
6. **Smoother decoder** (Track C/C1 FiLM, C2 jitter; queued): retrained models with structurally smoother latent-to-spectrum mapping.

After Tracks B/C land, this question can close (winner identified) or be refined (if neither track resolves it, the next candidates are hyper-networks, larger hash + per-layer FiLM via torch MLPs, or rethinking the auto-decoder paradigm).

---

(Q1 ray sampling, Q2 frequency grid, Q3 L sweep, Q4 latent dim, Q5 cluster partition, Q6 vendoring, Q7 pyroomacoustics 2D sanity, Q8 geometric attenuation, Q9 modal-degeneracy convention, Q10 HashGrid resize have been resolved; see `DECISIONS.md`.)
