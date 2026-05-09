# OPEN_QUESTIONS.md

Numbered, append-only ledger of ambiguities, blockers, and research-direction questions. When a question is answered, move the resolution to `DECISIONS.md` and remove the question here (so this file always reflects what's still open).

Asker: agent that wrote it. Owner: who can answer (manager, user, or "research call — needs experiment").

---

### Q1 — 2D ray sampling strategy *(deferred to Chunk 2)*

**Asker**: chunk-0 agent. **Owner**: research call (likely manager + first ablation in Chunk 2).

In 3D, AVR/INFER do stochastic spherical ray sampling (n_azi × n_ele directions, jittered each step). In 2D, the analogue is just `n_azi` rays uniformly on `[0, 2π)` (single elevation). But: pyroomacoustics ISM gives us *exact* image-source paths analytically — we know every reflection up to order N. Should our 2D renderer:

- (a) keep stochastic uniform-azimuth sampling (faithful to AVR/INFER, scales to non-rectangular rooms in later phases),
- (b) sample only along image-source paths (exploits known geometry; no wasted rays through walls),
- (c) deterministic uniform grid + small jitter (a compromise)?

Chunk-1 deferred this. Chunk-2 should default to (a) and ablate against (c) before considering (b).

---

### Q5 — Cluster partition for long-running training

**Asker**: chunk-0 agent. **Owner**: user.

Default is `scavenger` (preemptible, 3-day wall, unlimited resources). For final eval runs, do we have access to `tron` with `account=nexus` (the standard UMIACS account), or do we need a sponsor account? Affects how we structure long jobs in Chunk 3+.

---

### Q9 — Modal verifier handling of degenerate eigenfrequencies *(new in Chunk 1)*

**Asker**: chunk-1 agent. **Owner**: manager / research call.

At symmetric room geometries (L=W=4 m), modes (n_x, n_y) and (n_y, n_x) coincide. The peak picker sees one peak; the analytical mode list contains both modes at the same frequency. Two possible matcher behaviours:

- **(a) one-to-one (current)**: each pick attaches to exactly one mode; a degenerate pair counts as 1 match + 1 missed. Recall ceiling at L=W is `n_distinct_freqs / n_modes` ~ 0.4.
- **(b) many-to-one**: a single pick within tolerance of multiple modes credits all of them. Recall ceiling restored to 1.0 but we're double-counting one peak.
- **(c) deduplicate analytical list**: collapse degenerate modes to single entries in the comparison list. Cleanest; but loses information for "how many modes does this peak represent?".

For the noise-floor report we left option (a) and noted the L=W=4 ceiling explicitly. The Chunk-2 eigenfrequency MAE metric needs a definitive choice — affects whether L=W rooms appear "harder" than L≠W in the model evaluation.

Recommendation: **(c) for the recall metric, (a) for the per-peak audit table**. Get manager sign-off before Chunk 2's eval.

---

(Q2 frequency grid, Q3 L sweep, Q4 latent dim, Q6 vendoring, Q7 pyroomacoustics 2D sanity, and Q8 geometric attenuation were resolved during Chunk 1; see `DECISIONS.md`.)
