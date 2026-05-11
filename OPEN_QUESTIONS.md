# OPEN_QUESTIONS.md

Numbered, append-only ledger of ambiguities, blockers, and research-direction questions. When a question is answered, move the resolution to `DECISIONS.md` and remove the question here (so this file always reflects what's still open).

Asker: agent that wrote it. Owner: who can answer (manager, user, or "research call — needs experiment").

---

### Q11 — How to close the last 0.55 dB to the 2 dB modal target? *(refined in Chunk 3.7)*

**Asker**: chunk-3.5+ agent. **Owner**: manager / research call.

**Status as of Chunk 3.7**: substantially de-mystified, and partly answered. The 2 dB modal target is NOT yet met on any single L, but the gap is now 0.28-0.81 dB depending on L (was 1.5+ dB through Chunk 3.6) — and the mechanism is identified.

**Headline progress**: Chunk 3.7's I1 denser-training sweep (`D1_dense15`, 15 rooms at 0.2 m vs the original 7 at 0.5 m) drops modal-regime zero-shot LSD from 3.51 to **2.55 dB** averaged over the 6 unseen L. Per-L: 2.81 (L=3.25), 2.58 (3.75), 2.69 (4.25), 2.63 (4.75), **2.33 (5.25)**, **2.28 (5.75)** — at the upper-half L's the model is within 0.3 dB of the target.

**Mechanism identified**: Chunk 3.7 Track I gives a clean partial ordering of the bottleneck. The 3.5 dB modal ceiling from Chunk 3.6 was:
- NOT info-bound — Chunk 3.7 I3 (n_obs=32 via chunked inner loop) gives modal 3.53 dB, unchanged from C2 + B6's 3.51.
- NOT capacity-bound (in the FiLM-output-LoRA direction) — Chunk 3.7 I2 (rank-8 LoRA additive correction at decoder output) gives 3.35 dB, marginally better but well short of D1.
- DATA-DENSITY-BOUND — Chunk 3.7 I1 (denser sweep) breaks the ceiling by ~1 dB.

So more interpolation anchors smooth the latent-to-spectrum mapping the decoder learns, without changing the architecture.

**Concrete experimental paths still on the table (ranked by expected gain / cost)**:

1. **Push I1 further — denser sweep at 0.1 m and/or extended range** (cheapest, highest-prior). D1's lower-L gap (modal 2.81 at L=3.25) is mostly the boundary effect of having the 3.0 m endpoint. Adding training rooms at L = 2.6, 2.8 m and/or halving the spacing to 0.1 m (∼30 rooms covering [3.0, 5.8]) is the cleanest test of "modal LSD scales with training density". Cost: 12-15 new ISM rooms (~10 min) + 1 retrain (~3 h).
2. **Wider decoder** (n_levels=18 or sigma_encoder_dim=512). D1's val LSD 2.37 dB (vs 7-room C2's 1.43) suggests the same 8-D-latent + 14-bit HashGrid is now over-saturated by 15 rooms. Widening should lift both in-distribution and zero-shot. Cost: ~3 h.
3. **True hyper-network conditioning**: replace static FiLM with a small MLP that takes z and emits the weights of a per-room signal-branch decoder. Architectural risk + ~3-5× training cost. Cost: ~1 day implementation + 1 retrain.
4. **Modal-regime-only model**: drop diffuse training and target just 0-250 Hz. With D1 already at 2.55, a simpler model on the modal band might cleanly hit 2 dB.

This question can close definitively once (1) or (2) lands the 2 dB target. If neither does, the door re-opens for (3).

---

(Q1 ray sampling, Q2 frequency grid, Q3 L sweep, Q4 latent dim, Q5 cluster partition, Q6 vendoring, Q7 pyroomacoustics 2D sanity, Q8 geometric attenuation, Q9 modal-degeneracy convention, Q10 HashGrid resize have been resolved; see `DECISIONS.md`.)
