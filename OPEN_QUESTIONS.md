# OPEN_QUESTIONS.md

Numbered, append-only ledger of ambiguities, blockers, and research-direction questions. When a question is answered, move the resolution to `DECISIONS.md` and remove the question here (so this file always reflects what's still open).

Asker: agent that wrote it. Owner: who can answer (manager, user, or "research call — needs experiment").

---

(Q1 ray sampling, Q2 frequency grid, Q3 L sweep, Q4 latent dim, Q5 cluster partition, Q6 vendoring, Q7 pyroomacoustics 2D sanity, Q8 geometric attenuation, Q9 modal-degeneracy convention, Q10 HashGrid resize have been resolved; see `DECISIONS.md`.)


(P2-1's first budget check failed at per-room wall 2034.7s; resolved in-chunk by vectorizing `modal_rir_3d` to a single complex matmul and lowering `MAX_ORDER_CAP` from 17 → 12. Re-run wall 23.6s/room — see DECISIONS.md D6 revised.)


## Resolved by the manager on 2026-11-23 (D201; removed from this file, 2026-11-23 (19:11))

Q11, Q13, Q14 (superseded: Phase 1-2 ISM and 3D targets), Q12 (already closed), Q16 (the FDTD's locally reacting per-edge walls, D199), Q18 (headline = per-band r and LSD from 25 Hz; floored LSD retired), Q21 (closed with R1-C), Q23 (density closed in P4-9), Q26-Q32 (see D201 and D202). Full texts with resolutions: [tasks/P4_10_RULINGS.md](tasks/P4_10_RULINGS.md). Kept open: Q15 (maintenance: the grep the manager asked for is clean, no P4-10 metric calls the three functions), Q17 (deferred to the P4-11 join spec), Q24 (standing, for 3D), Q25 (leading P4-11 candidate together with the decay work), Q33.

### Q15 — Three latent bugs in shared evaluation code, deliberately isolated rather than fixed (P3-2)

**Status: OPEN, deferred by explicit decision.** Found while building P3-2's measurement stack. All three sit in code behind already-published P2/P3-1 numbers, so P3-2 works around them in new modules and leaves the shared code untouched; reproducibility of published results wins over tidiness until a dedicated fix chunk can re-verify the affected numbers.

1. **`aaf/eval/spatial_modes.py:21` `MARGIN_DEFAULT = 0.5` contradicts the datasets' `margin=0.3`** (`scripts/build_datasets.py:66,90`). `receiver_grid_xy`, `analytical_mode_shape` and `mode_shape_fit_error` therefore default to receiver coordinates that do not match `data/track_a`. Measured impact: `cond(Phi)` degrades 1.54 -> 3.02 and peak level picks up a **+0.66 dB** bias. This very plausibly explains the negative shape-fit SNRs flagged in `tasks/CHUNK_3_7_RESULTS.md:224-230` ((1,0) and (2,0) at L=4.25 reported -5.8 dB and -9.3 dB). The unit test at `tests/test_spatial_node_extraction.py:109` passes `margin=0.5` to *both* sides, so it cannot catch this. **P3-2 avoids it** by always reading `receiver_pos` from the HDF5 attrs and never reconstructing a grid.

2. **`aaf/eval/spatial_modes.py:154` `pick_first_modes` keeps only `sorted(e.pairs)[0]`**, silently discarding the degenerate partner of any frequency-degenerate pair. On a near-square room that throws away half the modes and mislabels mode families. **P3-2 avoids it** with `modal_projection.enumerate_modes`, which expands `EigenFreq.pairs`.

3. **`aaf/eval/modal_verifier.py:96-109` computes a -3 dB bandwidth and then discards it**, keeping only `q_factor`. The outward walk has **no distance cap** (on real 2D ground truth it strides across neighbouring modes and returns widths of 13-172 Hz), does no sub-bin interpolation (quantizing every width to 0.5 Hz), and **fabricates `bw = df` when the walk collapses**, turning an unresolvable peak into a confident number. **P3-2 avoids it** with `aaf/eval/modal_bandwidth.py`, which caps the walk, interpolates the peak and the crossings, and returns `nan` plus a flag rather than a fabricated width.

**Owner**: a future maintenance chunk. **What resolving it requires**: fix (1)-(3) in place, then re-run the affected Phase-1/Phase-2 shape-fit and modal-placement numbers and note any changes in DECISIONS. **Do not** silently fix them alongside unrelated work — the point of deferring was that published numbers move.

---

### Q17 — Should the SEALED room get its own conditioning channel, or stay out of the training set? (P3-3-FAST Track 2b)

*(Manager, 2026-11-23, D201: deferred to the P4-11 join spec.)*

Track 2b's edit axis is a doorway width `a`, and `a = 0` is **not** the small-aperture limit: a sealed one-node divider disconnects room B exactly, so `H_B == 0` and the inter-room level difference is `-inf` (FT-B). The conditioning coordinate `sqrt(a)/2` sends both a vanishing doorway and a sealed wall to 0, so the arm **cannot** represent the discontinuity: the two inputs are identical and the two targets are not.

The dataset therefore builds and keeps the 26 sealed rooms (20 train domains + 6 test domains, flagged `sealed` in the manifest, used by dataset-gate item (iv) to prove the divider plumbing reaches the solver) but **excludes them from training** via `config_kinds: ["open", "aperture"]` (D57). That is a defensible default — training on an unrepresentable discontinuity would smear every narrow aperture near it — but it is a choice, and the alternative is a design question for the manager rather than a bug:

* **(a) keep the exclusion** (current): the model learns a continuous law on `a in (0, W]`, and "what happens when the door is bricked up?" is out of scope.
* **(b) add a binary `sealed` channel** to the conditioning (56-d instead of 55-d), making the topology explicit and letting one model cover both regimes. Costs a dimension whose value is 0 in 380 of 400 training rooms, and invites the network to route the whole aperture response through a switch.
* **(c) predict the discontinuity from the physics instead**, i.e. keep the model continuous and special-case `a = 0` downstream in the renderer/eval.

**What resolving it requires**: only (b) changes the arm, and it must be decided *before* the Track 2b model is trained, since the cond_dim is baked into the checkpoint. Nothing in the dataset needs to change either way — the sealed rooms are on disk under all three options.

---

### Q24 — With a per-room modal solve at r 0.96-0.99, what is the INR for? *(P4-7 m1, D170(f); the manager's call)*

m1 found that the solver's own operator, solved once per room in a modal basis (to 600 Hz, first-order Robin damping, frozen C, exact zero mode), scores r 0.97-0.99 / 0.97-0.98 / 0.96 in AND out of family. The best network scores 0.93 / 0.72 / 0.58 in family and 0.63 / 0.37 / 0.36 out of family. Three facts bound what that comparison means:
* **Near-tautological on this data.** The ground truth is the same discrete operator, so physics "transferring" to unseen families is not evidence about real rooms: the baseline has every input the FDTD has (exact polygon, source, wall beta).
* **Cost.** The eigen-solve costs about as much as the FDTD itself: 102 s vs 58 s median per dense room. Neither is real-time. But one solve serves EVERY source and receiver position, and the FDTD needs one run per source.
* **What it does not cover.** Phase 2 targets are frequency-dependent or unknown materials, measured (not simulated) rooms, and partial geometry knowledge. A modal prior needs the boundary condition and the geometry; an INR fitted to measurements does not.

**Options:**
* (a) The INR becomes the residual / correction over a physics prior (m3 as pre-registered). The deliverable is then "physics + learned residual", and its value must be measured where the prior is WRONG: unknown beta, approximate geometry, measured data.
* (b) The modal prior is used only as a feature / regulariser for an INR that must work without exact geometry.
* (c) The INR's target shifts to what physics cannot give cheaply (e.g. real-time rendering of many rooms, inverse problems).

**Not acted on beyond m3, as pre-registered.** One concrete probe would make m3 informative under (a): perturb the prior (beta x 0.5 / x 2, or geometry snapped to 10 cm) and measure whether the residual network recovers the truth. m1 already shows what the unaided prior loses under a beta error of 2x: 0.02-0.09 r and 1-2.4 dB.

### Q25 — The 200-300 Hz fit cap comes from sharing one model across rooms (S-1, D185). Is it shared capacity or per-room exposure? *(P4-9 S-1; the manager's call)*

**Finding:** R0's recipe trained on ONE room reaches 200-300 Hz off-lattice r 0.931 / 0.953 on it (LSD 1.3 / 1.0 dB) at 10K, converged. R0, which shares one model across 1000 rooms, reads 0.581 / 0.645 (4.0 / 3.8 dB) on the same rooms. So the renderer can represent these fields; sharing is what caps them. S-1 cannot say WHICH consequence of sharing does it. The one-room run gets about 83x R0's per-room exposure, and it also has the whole model to itself.

**What already speaks to exposure (free, from P4-9's own reads):** R0's fit to the same 23 training rooms at 200-300 Hz rises 0.443 / 0.556 / 0.612 / 0.645 at 30K / 60K / 90K / 120K. CTL, which is R0 continued, reads 0.659 at 140K: still rising, now about +0.015 per 10K steps and slowing. So exposure helps. But even at the current rate, without further slowing, closing the remaining 0.29 to S-1's level would take about 190K more steps. Exposure alone is not a practical lever.

**Options for P4-10 (not acted on):**
* (a) Capacity at fixed exposure: R0 with a wider decoder (2x hidden) on the same 1000 rooms, read against R0's trajectory at matched steps (60K / 120K).
* (b) Per-room adapters (FiLM heads per room, as in Step 1's M2) on 1000 rooms. This is the deployable form of "more capacity per room"; Step 1 already shows M2 lifting unseen 25-100 Hz from 0.47 to 0.81 with 32 measurements.
* (c) Fewer rooms at fixed steps (R0's recipe on 200 and 50 GX rooms, 120K). This confounds capacity and exposure, but it maps the fit-versus-sharing curve between R0 (1000) and S-1 (1).

The S-4 build (1.33x throughput; score parity <= 7.2e-6, but fp16 MLPs and training never compared -- D194) could make these cheaper from P4-10.

## Resolved by the manager on 2026-11-24 (D211; removed from this file)

Q33 (meeting date: 2026-11-24, about 16:00 ET), Q34 (E-C, eval and small trainings; (b) and (c) wait for it), Q35 (R1-SM launched, pre-registered in D211), Q36 ((a) for the meeting, (c) as E-E), Q37 (D208 confirmed; folded into the decay work), Q38 (wording approved; 2f runs as E-B). Full texts of Q34-Q38: commit ec71f58, tasks/P4_10_MANAGER_REPORT_3.md. Still open: Q15, Q17, Q24, Q25.
