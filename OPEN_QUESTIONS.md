# OPEN_QUESTIONS.md

Numbered, append-only ledger of ambiguities, blockers, and research-direction questions. When a question is answered, move the resolution to `DECISIONS.md` and remove the question here (so this file always reflects what's still open).

Asker: agent that wrote it. Owner: who can answer (manager, user, or "research call — needs experiment").

---

### Q10 — Should the next iteration shrink HashGrid before another training run? *(new in Chunk 3)*

**Asker**: chunk-3 agent. **Owner**: manager.

Chunk 3 confirmed the over-parameterisation prediction from Chunk-2 §6/§8: with INFER's default HashGrid (`log2_hashmap_size=18, n_levels=20`), the multi-room model memorised 7 training rooms via hash params and the latents collapsed to room-ID indicators (PC1 vs L R² = −0.63). Per-training-room reconstruction met the spec target (6/7 rooms ≤ 1 dB val LSD); zero-shot at unseen L did not (held-out LSD 5.7-6.0 dB vs ≤ 2 dB target).

Recommended next config: `log2_hashmap_size=14, n_levels=14` (~16× fewer hash params, ~4× fewer hash levels). Same z_s injection (candidate A), same loss weights, same 30K iters. Estimated cost: ~3 h training + ~17 min for 6 parallel zero-shot evals.

**Decision the manager needs to make**: do we re-run Chunk 3 with the smaller HashGrid before the meeting, or do we present the current results as-is (with the latent_pca_1d.png as the diagnostic motivating a future smaller-architecture run)?

The infrastructure is in place — only `_default_hash_grid_config()` in `aaf/models/inr_2d.py` needs editing to flip the capacity, then re-run `scripts/run_chunk3_pipeline.sh`.

---

(Q1 ray sampling, Q2 frequency grid, Q3 L sweep, Q4 latent dim, Q5 cluster partition, Q6 vendoring, Q7 pyroomacoustics 2D sanity, Q8 geometric attenuation, and Q9 modal-degeneracy convention have been resolved; see `DECISIONS.md`.)
