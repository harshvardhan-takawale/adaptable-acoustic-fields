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

(Q2 frequency grid, Q3 L sweep, Q4 latent dim, Q6 vendoring, Q7 pyroomacoustics 2D sanity, Q8 geometric attenuation, and Q9 modal-degeneracy convention have been resolved; see `DECISIONS.md`.)
