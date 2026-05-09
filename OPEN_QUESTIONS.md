# OPEN_QUESTIONS.md

Numbered, append-only ledger of ambiguities, blockers, and research-direction questions. When a question is answered, move the resolution to `DECISIONS.md` and remove the question here (so this file always reflects what's still open).

Asker: agent that wrote it. Owner: who can answer (manager, user, or "research call — needs experiment").

---

### Q1 — 2D ray sampling strategy

**Asker**: chunk-0 agent. **Owner**: research call (likely manager + first ablation in Chunk 2).

In 3D, AVR/INFER do stochastic spherical ray sampling (n_azi × n_ele directions, jittered each step). In 2D, the analogue is just `n_azi` rays uniformly on `[0, 2π)` (single elevation). But: pyroomacoustics ISM gives us *exact* image-source paths analytically — we know every reflection up to order N. Should our 2D renderer:

- (a) keep stochastic uniform-azimuth sampling (faithful to AVR/INFER, scales to non-rectangular rooms in later phases),
- (b) sample only along image-source paths (exploits known geometry; no wasted rays through walls),
- (c) deterministic uniform grid + small jitter (a compromise)?

Option (a) is least-surprising; (b) might be much more sample-efficient for shoebox but doesn't generalize to Phase 3+ structured edits. Decision affects Chunk 1 dataset format (do we store paths?) and Chunk 2 renderer signature.

---

### Q2 — Frequency grid for Phase 1

**Asker**: chunk-0 agent. **Owner**: manager / user.

Phase 1 targets 0–2 kHz. AVR uses fs=16 kHz (Nyquist 8 kHz) with `signal_output_dim=1600` (so n_freq_bins ≈ 800 single-sided, Δf ≈ 10 Hz). For 0–2 kHz with a margin we could use `fs=4096`, `n_time_samples=4096`, `n_freq_bins=2049` (Δf = 1 Hz) — much finer modal resolution but bigger model output. Or `fs=4096`, `n_time_samples=1024`, `n_freq_bins=513` (Δf = 4 Hz). The eigenfrequency probe needs Δf small enough to resolve modes at ~50–100 Hz spacing.

Proposed default: **fs=4096, n_time_samples=2048, n_freq_bins=1025, Δf=2 Hz.** Confirms ≥10 bins per modal spacing, modest output dim.

---

### Q3 — L sweep range and density for Phase 1

**Asker**: chunk-0 agent. **Owner**: user / roadmap.

Roadmap mentions ≥10 rooms for Phase 2; doesn't specify Phase 1 sweep. Reasonable starting range: L ∈ [3 m, 8 m] uniformly at 0.25 m spacing → 21 rooms. Or sparser at 0.5 m → 11 rooms. Width W and height H held constant (for Phase 1 in 2D, W is the only other dim; pick W=4 m). What's the user's intended density?

---

### Q4 — Auto-decoder latent dimension and prior

**Asker**: chunk-0 agent. **Owner**: research call.

DeepSDF uses 256-dim latents with an N(0, σ²I) prior. For a 1-D family (L only), 4–8 dims should suffice. For Phase-2 (multiple structured axes), 32–64 dims. Defaulting to 32 with a unit-Gaussian prior + L2 regularizer in the latent loss is a reasonable opener. Confirm before Chunk 2 implementation.

---

### Q5 — Cluster partition for long-running training

**Asker**: chunk-0 agent. **Owner**: user.

Default is `scavenger` (preemptible, 3-day wall, unlimited resources). For final eval runs, do we have access to `tron` with `account=nexus` (the standard UMIACS account), or do we need a sponsor account? Affects how we structure long jobs in Chunk 3+.

---

### Q6 — Vendoring INFER's reference code

**Asker**: chunk-0 agent. **Owner**: user.

The INFER reference files (`unified_models.py`, `unified_renderers.py`) sit in `project_files/` (gitignored). When we port the relevant pieces into `aaf/models/` and `aaf/renderers/`, do we want a `aaf/_legacy_inference/` tree that vendor-copies the relevant classes verbatim for diffing during the port, or do we keep the references purely outside the repo? Vendoring eases code-review of the 2D adaptation; not vendoring keeps the repo trim. Probably want to copy *just* the chosen main classes (one model, one renderer) into a `_inference_ref/` module with a header comment about provenance.

---

### Q7 — pyroomacoustics 2D ISM sanity

**Asker**: chunk-0 agent. **Owner**: chunk-1 agent (do as a quick experiment).

Pyroomacoustics is 3D-first. `pra.ShoeBox(p=[L, W])` should produce 2D rooms but the library may auto-extrude or reject 2D internally. Verify before generating the dataset sweep: simulate one 2D shoebox and check (a) that the IR is causal, (b) that early image-source delays match analytical 2D ISM, (c) that frequency response shows the expected 2D modal spacing `f_{m,n} = (c/2)·sqrt((m/L)² + (n/W)²)`.

---

### Q8 — Greens-function / 2D path-loss convention

**Asker**: chunk-0 agent. **Owner**: research call.

In 3D, AVR's renderer applies 1/r geometric amplitude attenuation (free-field point-source pressure). The 2D analogue is **cylindrical**: pressure decays as 1/√r in the far field, with an additional Hankel-function near-field term. Three options:

- (a) keep 1/r (matches 3D code; physically wrong in 2D but might be empirically fine if the model absorbs the discrepancy),
- (b) switch to 1/√r (cylindrical free-field),
- (c) full 2D Green's function with a Hankel function (most rigorous; numerically annoying near r=0).

Probably (b) for Phase 1; flag if it causes systematic amplitude errors at small r.
