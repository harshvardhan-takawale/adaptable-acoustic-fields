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

**P2-1 status (single-room overfit, 5 rooms)**: Capacity is **roughly right**. All 5 de-risk rooms converged 6-7 dB → 1.3-1.8 dB val LSD over 15K iters; modal MAE 0.61-1.18 Hz on f<f_Schroeder (well under the 3 Hz spec target). No early plateau. P2-2 inherits these defaults (D23). P2-2 runs a d=16 vs d=32 hedge (D29) to test whether the latent table — not the HashGrid — is the binding capacity constraint. This question closes when the M1/M2 in-distribution val LSD comparison lands in CHUNK_P2_2_RESULTS.md.

If multi-room M1 in-distribution val LSD > 2.5 dB and M2 (d=32) is meaningfully lower, the latent dim is the lever — P2-3 should adopt d=32 (or larger). If both reach similar LSD, d=16 is preferred for downstream analysis.

**CLOSED (P2-2.5, 2026-06-06)** — capacity is NOT the wall; **per-iter coverage / total compute is**. The P2-2.5 diagnostic (DECISIONS.md D34) showed: the 10-room set fits to ~1.0-1.8 dB (runs A & C), and the 45-room set improved P2-2 M1's 6.16 → 2.61 dB purely from 8× coverage + 60K iters (run B) — all at the *same* `latent_dim=16` / HashGrid 18/16/1.38 / FiLM. The HashGrid/latent capacity is right; **do not widen**. (Note: P2-2's batch=4 ceiling was itself partly a GPU-misallocation artifact — M1/M2 silently ran on an 11 GB 2080 Ti via a bare `--gres=gpu:1`; CLUSTER_INFO.md now documents naming the GPU type.) P2-3 lever = scale compute (eff-batch 64, n_pts 32, 80-100K iters on A6000+DDP), not architecture.

---

### Q13 — ISM `max_order=12` tail truncation tolerance (was 17, revised in P2-1)

**Asker**: Chunk P2-1 agent. **Owner**: research call after Phase 2 evals.

**Status update**: D6 was revised in P2-1 from `max_order=17` to `max_order=12` (the budget check showed 17 gave 30+ min/room while the modal sum was Python-bound). At cap=12 the IR covers ~175 ms (3.5× the 50 ms early window). P2-1's signal-level eval showed late-corr 0.94-0.98 across de-risk rooms — the structure is captured, but **T20/T30 deltas blow up to 1.7-3.4 s** because the ISM ground-truth itself is truncated past ~175 ms.

**P2-2 deliberately defers EDC/T20/T30 calibration** (per spec). P2-2 will report late-corr and env-corr as the late-field metrics; EDC-derived T20/T30 are flagged "not yet calibrated" wherever they appear.

This question closes in Phase 3 with a deliberate ISM+ray-tracing hybrid pass that produces well-calibrated ground-truth EDCs.

---

### Q14 — Reinterpreting the 2 dB modal target in 3D

**Asker**: Chunk P2-1 agent. **Owner**: research call (manager + user).

**P2-1 outcome**: Modal density turned out to be ~11× higher than 2D in the 0-250 Hz band (P2-1 §5: 136 distinct modes on the box-center room vs ~12 in 2D). Per-band LSD also **inverts direction** in 3D — 0-250 Hz is the HARDEST band (2.1 dB on box-center single-room), 1000-2000 Hz the easiest (1.3 dB).

**P2-2 adopted answer (c)**: signal-level magnitude correlation ≥ 0.9 in 0-500 Hz on ≥ 5/8 unseen test rooms. Matches Dolby's stated language; relevant for downstream applications. Modal LSD is reported as supporting (not pass/fail).

**STILL OPEN (status as of P2-2.5, 2026-06-06)** — pending a valid P2-3 zero-shot run. P2-2's zero-shot hit 0/8 (mag corr 0.47-0.64), but that result is **not informative for Q14** because P2-2's *in-distribution* fit never converged (val LSD 6.16 dB; the model couldn't reconstruct even training rooms — see P2-2.5: the cause was coverage/compute + a GPU-misallocation, not the target metric). The target-metric question only becomes answerable once P2-3 trains a model that fits in-distribution (P2-2.5 shows that's reachable: ~1 dB on 10 rooms, 2.6 dB→ on 45 rooms) and THEN runs zero-shot. Carry forward the three candidate framings: (a) modal LSD ≤ 2 dB on f<f_Schroeder; (b) modal LSD ≤ 2 dB on a fixed 0-100/0-250 Hz band; (c) signal-level mag corr ≥ 0.9 in 0-500 Hz on ≥ 5/8 rooms (Dolby's language). **Decide in P2-3 once real zero-shot numbers exist.**

**UPDATE (P2-3, 2026-06-09)** — real zero-shot numbers now exist on a *converged* model (P3, in-dist 2.169 dB): **0/8 rooms, mag corr 0.20-0.28** (full spectrum). The target metric (c) is **not met**, but P2-3 showed *why*, and it is **not a metric-definition issue** — it is **training-set coverage**: the 45-room model memorizes and does not interpolate to unseen geometry (proven invariant to the test-time procedure across init/λ; see CHUNK_P2_3_RESULTS.md + DECISIONS D38). So Q14 stays open but is **reframed**: the metric choice is moot until zero-shot actually generalizes; that requires P2-4 (more rooms and/or explicit (L,W,H) conditioning). Re-evaluate the metric once P2-4 produces a model whose zero-shot is in a meaningful range.

---

(Q1 ray sampling, Q2 frequency grid, Q3 L sweep, Q4 latent dim, Q5 cluster partition, Q6 vendoring, Q7 pyroomacoustics 2D sanity, Q8 geometric attenuation, Q9 modal-degeneracy convention, Q10 HashGrid resize have been resolved; see `DECISIONS.md`.)

(P2-1's first budget check failed at per-room wall 2034.7s; resolved in-chunk by vectorizing `modal_rir_3d` to a single complex matmul and lowering `MAX_ORDER_CAP` from 17 → 12. Re-run wall 23.6s/room — see DECISIONS.md D6 revised.)
