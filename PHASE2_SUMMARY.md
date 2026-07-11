# Phase 2 — Status Summary for Manager Review

> **⏩ P2-4 COMPLETE (2026-07-04) — the "decision for P2-4" below has been executed.** The
> coverage-scaling curve (lever 1) is measured: known-geometry zero-shot mag corr scales
> **0.273→0.461 full / 0.409→0.811 modal** as rooms 45→90→150→250. Full writeup:
> `tasks/CHUNK_P2_4_RESULTS.md` + `outputs/coverage_curve/SCALING.md`.
>
> **⏩ P2-4b COMPLETE (2026-07-06) — the P2-4 curve is confound-corrected.** The curve conflated
> coverage with convergence (in-dist LSD degraded 2.17→4.30 dB as rooms rose), and under-training
> inflates zero-shot mag corr. A matched-convergence check (`outputs/coverage_curve/CONFOUND_CHECK.md`)
> found: coverage is **real** — at ~4.3 dB, 250 beats 45 on LSD/phase/RIR (verdict **CONFIRMED**) — but
> **~⅔ of the raw P2-4 mag-corr climb was the confound, only ~⅓ genuine coverage**. So the curve's
> *direction* is trustworthy, its *magnitude* was inflated ~3×; cite the matched deltas, not the raw
> slope. Also: **250 rooms can't converge at fixed capacity (~4.3 dB plateau)**, so densification carries
> a capacity penalty. **Next: P3-1 (lever 2, explicit (L,W,H) conditioning)** — now doubly motivated —
> benchmarked against the matched-convergence 250 point on the same frozen test set, reported at matched
> convergence. `tasks/CHUNK_P2_4b_RESULTS.md`. The section below is the pre-P2-4 decision snapshot.

**Date**: 2026-06-09 (P2-4 outcome note added 2026-07-04). **Scope**: everything completed since
the P2-2 multi-room conditioning attempt — the diagnostic (P2-2.5), the converged training +
zero-shot (P2-3), the coverage validations (Run C probe, anchored disambiguation), the
known-geometry + oracle-ceiling study (P2-3.5), and two meeting-figure packs. **Purpose**: give you
the full picture to decide P2-4. Every number below is traceable to a file on disk (links at the end).

---

## TL;DR (read this first)

Phase 2 went from "multi-room conditioning fails" to a **converged 3D multi-room model
with a sharp, honest characterization of what works and what's left**:

1. **In-distribution is SOLVED.** A single shared model renders all 45 training rooms to
   **val LSD 2.169 dB** (4-GPU DDP). The architecture, conditioning, and recipe all work.
2. **Known-geometry rendering WORKS at training density.** Predict a room's latent from its
   (L,W,H) and render — **no measurements** — and you get **0.89 magnitude correlation /
   2.6 dB** on held-out rooms (leave-one-out, n=45). The representation and renderer are sound.
3. **Generalization to genuinely-unseen rooms is the open problem, and it is
   ceiling-proven a TRAINING-COVERAGE limit** — not the architecture, not the method, not the
   test-time search. On rooms far from the 45 training samples, every route (8-measurement
   search, known-geometry lookup, and the best-possible "oracle" latent) collapses to
   **~0.27**. The fix is **more training rooms**, and the LOO result is the evidence it will work.

**The decision for P2-4**: scale the training set (45 → ~150–300 rooms) and/or condition the
decoder explicitly on (L,W,H). Details + rationale in the final section.

---

## The three numbers that matter (keep these regimes distinct)

| Regime | What it measures | Metric | Source |
|---|---|---|---|
| **In-distribution upper bound** | a model overfit to ONE room (the architecture's ceiling) | mag corr **0.95–0.98**, LSD **1.3–1.8 dB** | single-room (5 de-risk rooms) |
| **In-distribution, 45-room** | the shared model on its own training rooms | val LSD **2.169 dB** | P3 |
| **Generalization at training density (LOO)** | known-geometry render of a held-out room at training spacing | mag corr **0.89 / 0.94 (modal)**, LSD **2.6 dB** | P2-3.5 LOO |
| **Zero-shot to unseen rooms** | rooms deliberately far from training (maximin) | mag corr **~0.27** | P2-3 / P2-3.5 |

These four rows are the spine of the story: the architecture can render rooms (top two),
and the known-geometry route generalizes **when coverage is adequate** (row 3); it only
fails in the sparse gaps between 45 rooms (row 4).

---

## What we built (engineering deliverables, all reusable)

- **4-GPU manual-all-reduce DDP** in `aaf/train/multi_room_3d.py` (generic in `world_size`;
  sanity-gated at **0.596 it/s = 3.8×** single-GPU, gradients verified correct). Made the
  60K-iter converged run feasible (~28 h with one auto-resume past the 24 h cap).
- **Signal-level eval suite** (`aaf/eval/signal_level.py`): magnitude/phase/RIR/EDC/early-late/
  envelope correlation + per-band LSD + per-band mag corr — the Dolby-requested surface.
- **Self-diagnosing zero-shot** (`aaf/eval/zero_shot_3d.py` + `zero_shot_diagnosis.py`):
  geometry-head placement check + manifold-distance, and an opt-in manifold-anchored
  adaptation (`--z_init mean`, `--lambda_latent`, norm-clip) used to rule out the test-time
  procedure as the cause.
- **Known-geometry route** (`aaf/eval/known_geometry.py`): (L,W,H)→latent maps (RBF + linear)
  + an "oracle" latent search (optionally norm-clipped to the trained shell) + leave-one-out
  evaluation, all reusing the frozen decoder + renderer.
- **Two meeting-figure packs** (11 figures, 1920×1080, every number traceable) +
  `FIGURE_MANIFEST.md`.

---

## The scientific narrative (how the pieces fit)

1. **Capacity is not the bottleneck** (P2-2.5). A controlled diagnostic showed a 10-room
   model reaches ~1.0 dB in-distribution (Run C, 0.98 dB) — the architecture has ample
   representational headroom. The P2-2 failure (6.16 dB) was **per-iteration coverage/compute**.
2. **In-distribution training solved** (P2-3). Scaling compute on the full 45 rooms (4-GPU DDP,
   effective batch 64, 60K iters) converged to **2.169 dB**, clearing the ≤2.5 target. The
   representation also organizes cleanly: a linear probe recovers (L,W,H) from the latent at
   **R² 0.991 / 0.967 / 0.974**.
3. **Zero-shot from sparse measurements fails — and it's coverage, not the search** (P2-3 +
   validations). Optimizing a latent from 8 measured receivers gives mag corr **0.20–0.28** on
   the 8 maximin test rooms. We ruled out the test-time procedure two independent ways:
   - **Run C probe**: on the *converged* 10-room model, zero-shot on interior rooms is ~0.2,
     and a 3-config sweep (random vs manifold-mean init; λ ∈ {1e-4, 1e-2}) leaves it invariant.
   - **P3 anchored disambiguation**: on the 45-room model, the same sweep (init × λ ∈ {1e-4,
     1e-2, 1e-1}) is invariant at **0.22–0.28**. A telling sign: the converged model's zero-shot
     (0.27) is *worse* than the unconverged P2-2 model's (0.47–0.64), because a sharp decoder
     punishes a wrong latent while a blurry one outputs a generic "average room." → The bottleneck
     is **finding/representing the latent for an unseen room**, not the optimizer.
4. **Known-geometry rendering works at training density** (P2-3.5). Skipping measurements
   entirely — predict the latent from (L,W,H) and render — gives **0.89 mag corr / 2.6 dB** on
   held-out rooms (leave-one-out over the 45). The render reproduces the magnitude spectrum,
   phase, RIR, and even the spatial mode shapes. The route and the decoder are sound.
5. **Generalization to unseen geometries is ceiling-proven coverage-bound** (P2-3.5). On the
   maximin test rooms (chosen far from training), **every** route lands at ~0.27: the
   8-measurement search, the known-geometry lookup (RBF and linear), and the **oracle** — the
   best latent found by optimizing against the room's own measurements. We verified this three
   ways, closing every loophole: (i) an on-manifold lookup latent → 0.27; (ii) an unconstrained
   oracle (which drifts off-manifold) → 0.27; (iii) a **norm-clipped on-manifold oracle** (best
   latent constrained to the trained shell) → **still 0.27**. **No latent renders an unseen
   room** at 45-room coverage → the decoder memorizes its 45 rooms and does not interpolate.
6. **The coverage relationship** (P2-3.5). Two measured anchors: at training density
   (~0.34 m nearest-neighbor spacing) the route gives 0.89; in the sparse test gaps (~0.61 m)
   it gives 0.27. We present this as **two anchors with an explicitly unmeasured gap** — we
   have NOT measured the intermediate scaling (that is exactly what P2-4 would map).

---

## Honesty notes (so nothing gets conflated)

- The **three regimes are distinct** and never averaged together: in-distribution upper bound
  (single-room overfit) vs leave-one-out generalization at training density vs zero-shot on
  unseen rooms. (One room, L4.50/4.00/3.25, appears as a single-room overfit *and* as the
  zero-shot box center — they are different experiments; captions keep them separate.)
- The LOO 0.89 is **at training density** — it is not a claim that an *arbitrary* new room
  renders at 0.89 today. It is the demonstration that the machinery works given adequate coverage.
- The coverage plot is two **measured anchors**, not a fitted curve — we explicitly do not
  draw a trend between them.
- Distinct-mode count for 3D is **135** (≤250 Hz, excluding the DC term), ~11× the 2D ~12.

---

## Chunk-by-chunk (what each produced)

- **P2-2.5 — diagnostic** (`tasks/CHUNK_P2_2_5_RESULTS.md`, `outputs/diag_p2_2_5/DIAGNOSIS.md`).
  Verdict: coverage/compute, not capacity. Runs: A (10 rm, eff-16) 1.84 dB; C (10 rm, eff-64)
  **0.98 dB** (the architecture's 3D ceiling); B (45 rm, eff-32, 60K) 2.61 dB. Built the 2-GPU
  DDP (verified vs single-GPU, Δ ≤ 0.03 dB).
- **P2-3 — converged training + zero-shot** (`tasks/CHUNK_P2_3_RESULTS.md`,
  `outputs/multi_room_3d/P3_45rooms_4gpu/SUMMARY.md`). 4-GPU DDP → **2.169 dB** in-distribution.
  Zero-shot 0/8 (mag 0.20–0.28); diagnosed as coverage, not procedure.
- **Run C zero-shot probe** (`outputs/runC_zeroshot_probe/PROBE_RESULT.md`). Confirmed on the
  converged 10-room model that the test-time procedure is not the cause (3-config sweep).
- **P2-3.5 — known-geometry + oracle ceiling** (`tasks/CHUNK_P2_3_5_RESULTS.md`,
  `outputs/known_geometry/RESULTS.md`). LOO 0.89; oracle ceiling-proves coverage 3 ways.
- **Meeting figure packs** (`tasks/CHUNK_P2_VIZ_RESULTS.md`, `CHUNK_P2_VIZ2_RESULTS.md`,
  `outputs/phase2_meeting_assets/FIGURE_MANIFEST.md`). 11 figures with traceable numbers +
  honest captions + backup slides (single-room fidelity table, train-rooms list, IR/dataset spec).

---

## Dataset / methods (verified on disk)

45 training rooms (Latin-hypercube over L,W,H) + 8 maximin test rooms (interpolative-interior
by design, but in practice mostly just past the LHS hull edges). Each room: **fs = 4096 Hz,
8192 samples (2.0 s), 4097 frequency bins (Δf ≈ 0.5 Hz), ISM max_order = 12, α = 0.15, 512
receivers (8×8×8 grid), 1 source.** Frequency-domain σ+jβ renderer; per-room latent
auto-decoder (latent_dim 16) with FiLM conditioning + a linear geometry head.

---

## The decision for P2-4 (recommended, ranked)

The in-distribution problem is closed; the open problem is **generalization to unseen
geometries**, and it is unambiguously **training-set coverage**. Two levers, both compatible:

1. **Scale the training set: 45 → ~150–300 rooms** (denser Latin-hypercube over the L,W,H box).
   This is the **evidenced** fix — the LOO result shows the known-geometry route already reaches
   0.89 at training density; denser sampling shrinks the gaps so an arbitrary room is always
   near a trained geometry. Cost is data generation (~24 s/room ISM) + one retrain on the proven
   4-GPU recipe. The natural first experiment is the **coverage-scaling curve** (e.g. 90 / 150 /
   250 rooms), which directly maps the region our two anchors leave unmeasured.
2. **Condition the decoder explicitly on (L,W,H)** (feed geometry directly, not only via the
   optimized latent). For a new room we *know* its dimensions; letting the decoder *compute*
   modal structure from geometry — rather than memorize it per-latent — could let known-geometry
   rendering work with no latent search at all. Potentially decisive; complementary to (1).

**What NOT to pursue** (ruled out by the data):
- Better test-time search or a better (L,W,H)→latent map — the oracle proves neither is the
  bottleneck at 45 rooms.
- Widening capacity / changing conditioning — in-distribution is already solved at the current
  capacity; the wall is coverage, not capacity.

**Open question for you**: which of the two P2-4 levers to run first (or both in parallel),
and the room-count budget for the scaling curve.

---

## Artifacts & links (raw GitHub, `…/main/<path>`)

Base: `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/`

**Results docs**
- `tasks/CHUNK_P2_2_5_RESULTS.md` — diagnostic (coverage not capacity)
- `tasks/CHUNK_P2_3_RESULTS.md` — converged training + zero-shot
- `tasks/CHUNK_P2_3_5_RESULTS.md` — known-geometry + oracle ceiling
- `outputs/known_geometry/RESULTS.md` — full P2-3.5 comparison table + verdict
- `outputs/runC_zeroshot_probe/PROBE_RESULT.md` — Run C zero-shot probe + 3-config validation
- `outputs/multi_room_3d/P3_45rooms_4gpu/SUMMARY.md` — P3 zero-shot summary
- `outputs/diag_p2_2_5/DIAGNOSIS.md` — P2-2.5 diagnostic

**Meeting figures** (11 figures, 1920×1080)
- `outputs/phase2_meeting_assets/FIGURE_MANIFEST.md` — every figure, its source, exact numbers,
  honest caption (the index to the deck)

**Standing docs**
- `CONTEXT_FOR_MANAGER.md` — re-orientation
- `DECISIONS.md` — decisions D32–D38 (DDP, iteration budget, 3-way self-diagnosis, P2-3 verdict,
  4-GPU eff-batch-64, P2-3.5 coverage verdict)
- `OPEN_QUESTIONS.md` — Q12 (capacity) closed; Q14 (target metric) reframed pending denser-coverage zero-shot

---

## Risks / caveats

- The maximin test rooms are a **hard** test (several sit just past the LHS hull edges); the
  P2-4 zero-shot eval should split interpolative vs extrapolative test rooms.
- The "more rooms fixes it" claim is **motivated and consistent with the LOO**, but the
  intermediate scaling is **not yet measured** — P2-4's coverage curve is what turns it from a
  well-supported hypothesis into a measured result. (We were careful not to imply a curve we
  haven't run.)
- Cluster note for any future agent: always **name the GPU type** in `--gres` (e.g.
  `--gres=gpu:rtxa5000:1`) — a bare `gpu:1` lands on an 11 GB card and OOMs the n_pts=32
  renderer. This was the cause of two avoidable failures.
