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

**UPDATE (P2-4, 2026-07-04)** — zero-shot is now in a **meaningful, climbing range** and lever (1) (densification) is validated: on the frozen interior test set, known-geometry zero-shot mag corr scales **0.273→0.461 full / 0.409→0.811 modal** as rooms 45→250 (CHUNK_P2_4_RESULTS.md). The modal band (0–250 Hz) at 250 rooms is **0.811 — 76% of the way to the 0.938 training-density ceiling and still climbing**, so a modal-band framing of the target is within reach of pure densification (rough extrapolation ~0.9 modal near ~400–500 rooms), while the **full-band** metric (0.461) lags and is far from any ≥0.9 target. Q14 stays open pending P3-1: (a) does explicit (L,W,H) conditioning hit the modal target with fewer rooms; (b) settle whether pass/fail is modal-band (near-reachable) or full-band (far). Note all P2-4 points are fixed-budget lower bounds (in-dist LSD degraded 2.17→4.30 dB as exposure/room fell). *(Correction, see P2-4b below: these are NOT conservative floors — the mag-corr numbers are inflated by the convergence/blur confound.)*

**UPDATE (P2-4b, 2026-07-06)** — the P2-4 curve is confound-corrected (CHUNK_P2_4b_RESULTS.md + CONFOUND_CHECK.md). Coverage genuinely helps at matched convergence, but **~⅔ of the raw P2-4 mag-corr climb was the convergence/blur confound** (full-band raw +0.188 = 68% blur + 32% coverage; modal +0.402 = 72% / 28%). So the "0.811 modal, 76% of the gap, ~0.9 near 400–500 rooms" reading **overstated it**: the genuine matched-convergence modal coverage effect is **+0.113** (not +0.402), and pure densification alone will reach any ≥0.9 target **much more slowly** than the raw curve implied — *and* 250 rooms already can't converge at fixed capacity (capacity plateau ~4.3 dB). This **reframes the metric decision toward geometry conditioning (P3-1)** rather than densification as the route to the target. Q14 resolves once P3-1 shows whether explicit (L,W,H) conditioning clears a modal-band mag-corr / LSD target at matched convergence with tractable room counts.

**UPDATE (P3-1 in progress, 2026-07-14)** — data-only, no verdict yet; matched-convergence comparison not run (dense L checkpoints removed under the disk quota; G+ still training). Three arms under a band-limited 0–300 Hz protocol on the frozen 15-room interior test set. In-dist val LSD: L 0.72 dB (40K iters), G 1.14 dB (28K), G+ 2.02 dB (iter 10,900, ongoing). Zero-shot modal placement recall@250: L 0.104, G 0.101, G+ 0.164/0.114/0.089 at ckpt 2K/6K/11K; recall@300: L 0.075, G 0.075, G+ 0.129/0.084/0.069. G+ learned resonance weight w by ckpt: 0.03/0.14/0.35/0.41/0.34 (1K/2K/4K/6K/11K); per-room, G+ recall@250 is lower at 6K than 2K for 14/15 rooms. Numbers: `outputs/p3_1/HEADTOHEAD.md`, `tasks/CHUNK_P3_1_RESULTS.md`. Q14 stays open pending the matched-convergence comparison and G+ convergence.

---

(Q1 ray sampling, Q2 frequency grid, Q3 L sweep, Q4 latent dim, Q5 cluster partition, Q6 vendoring, Q7 pyroomacoustics 2D sanity, Q8 geometric attenuation, Q9 modal-degeneracy convention, Q10 HashGrid resize have been resolved; see `DECISIONS.md`.)

(P2-1's first budget check failed at per-room wall 2034.7s; resolved in-chunk by vectorizing `modal_rir_3d` to a single complex matmul and lowering `MAX_ORDER_CAP` from 17 → 12. Re-run wall 23.6s/room — see DECISIONS.md D6 revised.)

---

### Q15 — Three latent bugs in shared evaluation code, deliberately isolated rather than fixed (P3-2)

**Status: OPEN, deferred by explicit decision.** Found while building P3-2's measurement stack. All three sit in code behind already-published P2/P3-1 numbers, so P3-2 works around them in new modules and leaves the shared code untouched; reproducibility of published results wins over tidiness until a dedicated fix chunk can re-verify the affected numbers.

1. **`aaf/eval/spatial_modes.py:21` `MARGIN_DEFAULT = 0.5` contradicts the datasets' `margin=0.3`** (`scripts/build_datasets.py:66,90`). `receiver_grid_xy`, `analytical_mode_shape` and `mode_shape_fit_error` therefore default to receiver coordinates that do not match `data/track_a`. Measured impact: `cond(Phi)` degrades 1.54 -> 3.02 and peak level picks up a **+0.66 dB** bias. This very plausibly explains the negative shape-fit SNRs flagged in `tasks/CHUNK_3_7_RESULTS.md:224-230` ((1,0) and (2,0) at L=4.25 reported -5.8 dB and -9.3 dB). The unit test at `tests/test_spatial_node_extraction.py:109` passes `margin=0.5` to *both* sides, so it cannot catch this. **P3-2 avoids it** by always reading `receiver_pos` from the HDF5 attrs and never reconstructing a grid.

2. **`aaf/eval/spatial_modes.py:154` `pick_first_modes` keeps only `sorted(e.pairs)[0]`**, silently discarding the degenerate partner of any frequency-degenerate pair. On a near-square room that throws away half the modes and mislabels mode families. **P3-2 avoids it** with `modal_projection.enumerate_modes`, which expands `EigenFreq.pairs`.

3. **`aaf/eval/modal_verifier.py:96-109` computes a -3 dB bandwidth and then discards it**, keeping only `q_factor`. The outward walk has **no distance cap** (on real 2D ground truth it strides across neighbouring modes and returns widths of 13-172 Hz), does no sub-bin interpolation (quantizing every width to 0.5 Hz), and **fabricates `bw = df` when the walk collapses**, turning an unresolvable peak into a confident number. **P3-2 avoids it** with `aaf/eval/modal_bandwidth.py`, which caps the walk, interpolates the peak and the crossings, and returns `nan` plus a flag rather than a fabricated width.

**Owner**: a future maintenance chunk. **What resolving it requires**: fix (1)-(3) in place, then re-run the affected Phase-1/Phase-2 shape-fit and modal-placement numbers and note any changes in DECISIONS. **Do not** silently fix them alongside unrelated work — the point of deferring was that published numbers move.

---

### Q16 — Does wall-selective editing survive a wave-based (Kuttruff) absorption law? (P3-2)

P3-2's ground truth is pyroomacoustics ISM, whose angle-independent reflection coefficient gives `gamma ~ cos(theta)` and therefore **no grazing-incidence absorption**: an axial mode is damped only by the wall pair it bounces between, and measured selectivity is ~29:1 (D48; `dAIC = 73` favouring the ray law over Kuttruff). A real locally-reacting wall follows Kuttruff, where every wall sits at a pressure antinode of every mode, giving only **~2:1** selectivity and **no invariant family**.

So P3-2 can support *"the model learns the simulator's per-wall law"* and cannot, on its own evidence, support *"wall-selective editing works in real rooms"*. **Open question for the manager**: is the 2:1 regime the target worth pursuing (a materially harder learning problem, since the edit signal is ~15x weaker relative to the non-edited family), and if so should it be approached by (a) a wave-based / FEM 2D solver, (b) pyroomacoustics with `ray_tracing=True` and angle-dependent materials, or (c) measured RIRs? This decides whether P3-3 is "scale the same claim to 3D" or "re-establish the claim under realistic wall physics". Nothing in P3-2 needs to change to keep this option open — the Kuttruff law is already implemented and reported alongside every prediction.

---

### Q17 — Should the SEALED room get its own conditioning channel, or stay out of the training set? (P3-3-FAST Track 2b)

Track 2b's edit axis is a doorway width `a`, and `a = 0` is **not** the small-aperture limit: a sealed one-node divider disconnects room B exactly, so `H_B == 0` and the inter-room level difference is `-inf` (FT-B). The conditioning coordinate `sqrt(a)/2` sends both a vanishing doorway and a sealed wall to 0, so the arm **cannot** represent the discontinuity: the two inputs are identical and the two targets are not.

The dataset therefore builds and keeps the 26 sealed rooms (20 train domains + 6 test domains, flagged `sealed` in the manifest, used by dataset-gate item (iv) to prove the divider plumbing reaches the solver) but **excludes them from training** via `config_kinds: ["open", "aperture"]` (D57). That is a defensible default — training on an unrepresentable discontinuity would smear every narrow aperture near it — but it is a choice, and the alternative is a design question for the manager rather than a bug:

* **(a) keep the exclusion** (current): the model learns a continuous law on `a in (0, W]`, and "what happens when the door is bricked up?" is out of scope.
* **(b) add a binary `sealed` channel** to the conditioning (56-d instead of 55-d), making the topology explicit and letting one model cover both regimes. Costs a dimension whose value is 0 in 380 of 400 training rooms, and invites the network to route the whole aperture response through a switch.
* **(c) predict the discontinuity from the physics instead**, i.e. keep the model continuous and special-case `a = 0` downstream in the renderer/eval.

**What resolving it requires**: only (b) changes the arm, and it must be decided *before* the Track 2b model is trained, since the cond_dim is baked into the checkpoint. Nothing in the dataset needs to change either way — the sealed rooms are on disk under all three options.
