# OPEN_QUESTIONS.md

Numbered, append-only ledger of ambiguities, blockers, and research-direction questions. When a question is answered, move the resolution to `DECISIONS.md` and remove the question here (so this file always reflects what's still open).

Asker: agent that wrote it. Owner: who can answer (manager, user, or "research call — needs experiment").

---

### Q5 — Cluster partition for long-running training

**Asker**: chunk-0 agent. **Owner**: user.

Default is `scavenger` (preemptible, 3-day wall, unlimited resources). Chunk 2 uses scavenger with 2,500-iter checkpointing — works, no preemption events seen at the time of writing (all three Chunk-2 jobs landed on the same `legacygpu06` node and ran without interruption).

For Chunks 3-4 (auto-decoder training across multiple rooms), training time grows; the user has 4 `tron` slots banked for non-preemptible runs. Use `tron` with `account=nexus` for the headline auto-decoder run; keep ablations on scavenger.

This question can stay open until Chunk 3 needs to commit to a specific allocation strategy.

---

(Q1 ray sampling, Q2 frequency grid, Q3 L sweep, Q4 latent dim, Q6 vendoring, Q7 pyroomacoustics 2D sanity, Q8 geometric attenuation, and Q9 modal-degeneracy convention have been resolved; see `DECISIONS.md`.)
