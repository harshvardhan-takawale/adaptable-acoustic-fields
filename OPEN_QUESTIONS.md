# OPEN_QUESTIONS.md

Numbered, append-only ledger of ambiguities, blockers, and research-direction questions. When a question is answered, move the resolution to `DECISIONS.md` and remove the question here (so this file always reflects what's still open).

Asker: agent that wrote it. Owner: who can answer (manager, user, or "research call — needs experiment").

---

### Q11 — How to unblock zero-shot adaptation? *(new in Chunk 3.5+)*

**Asker**: chunk-3.5+ agent. **Owner**: manager / research call.

Chunk 3.5 + 3.5+ swept 4 architecturally-diverse multi-room auto-decoders (R0/R6/R7/R8: mlp_32 vs linear L-head, 14-bit vs 16-bit hash, 2-D vs 8-D latent). All 4 fully-complete runs **fail zero-shot at 5.21-5.91 dB held-out LSD** (target was ≤ 2 dB on ≥ 4/6 unseen L). The architecture choices made no material difference; the bottleneck has shifted from latent collapse (Chunk 3) to inner-loop adaptation.

Diagnostic evidence:
- **Even on observed receivers** the inner-loop optimizer can't drive obs LSD below 4.5-5.4 dB (vs training val LSD of 1.3-1.7 dB). 2K iters of Adam(lr=1e-2) on a low-dim z_star can't navigate the model's latent-to-spectrum response surface outside the trained-latent neighbourhood.
- R6's `latent_pca_1d` shows **train latents nearly monotonic with L** (linear L-head IS shaping z_s), but **zero-shot z_star tensors collapse to one region of latent space** (PC1 ≈ -0.5 for all 6 unseen L). The inner loss surface has a single attractor outside the training region.
- R8 (2-D latent) has zero-shot z_star with some L-correlation (Pearson r ≈ 0.93 between true L and lhead_predicted_L from optimized z_star) — but the actual reconstruction quality doesn't improve. The latent moves in the right direction but the model doesn't follow.

Concrete experimental paths to try (cheap, ranked by cost):
1. **More observed receivers**: 32 instead of 8. If this works, we know "8 is too few" and the architecture/training is fine.
2. **Multi-restart inner adaptation**: 10 random z_star inits, pick the best. Characterizes whether the loss landscape has multiple basins.
3. **Longer inner adaptation**: 10K iters instead of 2K. May just need more steps.
4. **Latent-hull-constrained adaptation**: project z_star onto the convex hull of trained latents at each step. Forces interpolation rather than escape.

Each of (1)-(4) reuses the existing infrastructure (model checkpoints from R0/R6/R7/R8 are on disk). Cost: ~10-30 min per (config, strategy) on scavenger.

The **manager needs to decide** which strategy(ies) the next chunk pursues. Recommended order: (1) → (2) → (3) → (4).

This question can stay open until the next-chunk plan is written; it doesn't block other work.

---

(Q1 ray sampling, Q2 frequency grid, Q3 L sweep, Q4 latent dim, Q5 cluster partition, Q6 vendoring, Q7 pyroomacoustics 2D sanity, Q8 geometric attenuation, Q9 modal-degeneracy convention, Q10 HashGrid resize have been resolved; see `DECISIONS.md`.)
