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

This question can close definitively once (1) or (2) lands the 2 dB target. If neither does, the door re-opens for (3). *Status update Chunk P2-1: this question is now Phase-1 history; Phase 2 runs are 3D and the 2 dB modal target is reinterpreted in the new 3D modal-density regime (see Q12, Q14).*

---

### Q12 — Is the 3D HashGrid 18/16/1.38 the right starting capacity for P2-2 multi-room conditioning?

**Asker**: Chunk P2-1 agent. **Owner**: P2-2 manager / research call.

DECISIONS.md D10 picks the 3D HashGrid defaults (`log2_hashmap_size=18`, `n_levels=16`, `per_level_scale=1.38`) based on collision-rate-corrected scaling from Phase 1's 2D-validated 14/14. The 5 de-risk single-room overfits in this chunk give the first empirical signal:

- If modal MAE < 3 Hz on f<f_Schroeder for ≥4 of 5 rooms → capacity is roughly right; P2-2 inherits these defaults.
- If modal MAE > 3 Hz on a clear majority → undercapacity; iterate downward via `per_level_scale=1.34` (finer finest level) before bumping `log2_hashmap_size`.
- If single-room val LSD plateaus very early (e.g., flat after 5K iters) → overcapacity; P2-2 should reduce.

This question closes when the single-room 3D summary lands and the diagnosis is clear. The chunk-3.7 Phase-1 lesson was that auto-decoder capacity choices are easier to make once we've seen single-room overfit behaviour first.

---

### Q13 — ISM `max_order=17` tail truncation tolerance

**Asker**: Chunk P2-1 agent. **Owner**: research call after first 3D evals.

DECISIONS.md D6 hard-caps `max_order` at 17 for 3D tractability. The resulting IR covers ~108 ms of decay (4× the 50 ms early-reflection envelope) but truncates the diffuse tail. For 5 de-risk rooms the `ir_truncated` warning will fire (4·T60·fs ≈ 14000-15000 > 8192).

**Sub-question**: does the truncation noticeably affect the signal-level eval suite's late-corr / EDC-error metrics? The early band (≤50 ms) is well-resolved; the late band (50 ms – n_time/fs ≈ 2.0 s) is in the truncated regime.

If late-corr is consistently near zero across de-risk rooms → truncation matters; budget time for max_order=17 + ray-tracing fallback in Phase 3. If late-corr stays comparable to early-corr → truncation is acceptable for P2-2 / future work.

---

### Q14 — Reinterpreting the 2 dB modal target in 3D

**Asker**: Chunk P2-1 agent. **Owner**: research call (manager + user).

The Phase-1 "modal LSD ≤ 2 dB" target was defined on the 2D 0-250 Hz band. In 3D:
- Modal density is ~3× higher in the 0-250 Hz band (~30 modes vs ~10 for 2D).
- The Schroeder frequency varies per room (typical f_Schroeder ≈ 150-200 Hz for the ranges in D1).
- Above f_Schroeder, modal density exceeds the RFFT resolution Δf = 0.5 Hz (D18) — LSD remains a valid metric but modal MAE doesn't.

**Sub-question**: should the Phase-2 target be:
(a) modal LSD ≤ 2 dB on f < f_Schroeder (variable per-room band)?
(b) modal LSD ≤ 2 dB on a fixed 0-250 Hz band?
(c) signal-level magnitude-corr ≥ 0.9 in 0-500 Hz (matches Dolby's stated language)?

Decision needed for P2-2 zero-shot eval. Single-room overfit results from P2-1 will inform: e.g., if all 5 de-risk rooms hit modal LSD ≤ 1 dB on f<f_Schroeder, then (a) is the right scale for the zero-shot target.

---

(Q1 ray sampling, Q2 frequency grid, Q3 L sweep, Q4 latent dim, Q5 cluster partition, Q6 vendoring, Q7 pyroomacoustics 2D sanity, Q8 geometric attenuation, Q9 modal-degeneracy convention, Q10 HashGrid resize have been resolved; see `DECISIONS.md`.)
