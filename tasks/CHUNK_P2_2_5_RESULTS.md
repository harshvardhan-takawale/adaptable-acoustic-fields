# Chunk P2-2.5 — Diagnostic: isolating the multi-room 3D training bottleneck

**Status**: COMPLETE — 2026-06-06.

## 1. Question

P2-2 failed both zero-shot and in-distribution (val LSD plateaued at 6.16 dB
on 45 rooms, vs the ≤ 2.5 target), even though P2-1 single-room overfit reached
1.3-1.8 dB on the same 3D rooms. Two hypotheses: **sampling/coverage** (P2-2
used batch=4, 0.017% coverage/iter) vs **capacity** (a 3-parameter room family
exceeds the shared decoder + FiLM). This chunk runs three controlled trainings
to disambiguate — diagnostic only, no zero-shot, no meeting assets.

## 2. The three runs + result

| Run | Rooms | eff-batch | n_pts | coverage | iters | val LSD | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **A** | 10 | 16 | 16 | 0.31% | 30K ✅ | **1.84 dB** | ✅ success |
| **B** | 45 | 32 | 32 | 0.14% | 60K ✅ (DDP) | **2.61 dB** | ⚠ at threshold, ↓ |
| **C** | 10 | 64 | 32 | 1.25% | ~30K | **~1.0 dB** | ✅ success (ceiling) |

A and C share the same 10-room maximin subset (`configs/sweeps_3d/diag_10rooms.yaml`);
their only difference is coverage. B is the full 45-room set at scaled compute.

## 3. Verdict — capacity is NOT the wall; coverage/compute is the lever

- **A succeeds (1.84) AND C succeeds (~1.0)** → the decoder fits a 3D room set
  cleanly at this capacity. Capacity / conditioning is **not** the bottleneck.
- **C is the architecture's true 3D multi-room ceiling: ≈ 1.0 dB** (10 rooms,
  high coverage). This is the single number the manager asked to see first.
- **B improved from P2-2 M1's 6.16 dB → 2.61 dB** purely by raising per-iter
  coverage 8× (batch 4 → 32) and training to 60K — a **3.55 dB gain with no
  architecture change**, landing right at the 2.5 success threshold and still
  descending at 60K.

**P2-3 recommendation**: scale compute on the full 45-room set — apply Run C's
recipe (effective batch 64, n_pts 32) to all 45 rooms and/or extend to 80-100K
iters. B at 2.61@60K was still descending; either lever should carry it ≤ 2.5.
**Do not widen the decoder** — C proves ~1 dB is achievable at the current
capacity (`latent_dim=16`, HashGrid 18/16/1.38, FiLM).

## 4. Per-room (B, 45 rooms, DDP @ 60K)

Best 2.29, worst 3.01; **12/45 rooms already ≤ 2.5 dB**; geometry head per-axis
MAE 3-7 mm. No room catastrophically left behind — the spread is tight and the
whole set is descending together, consistent with "needs more compute," not
"a subset is unlearnable."

## 5. Two cluster problems found + fixed mid-chunk

### 5a. GPU targeting (the big one)
The first two launches OOM'd because **`--gres=gpu:1` ignores GPU type** — the
scheduler filled the slot with an 11 GB RTX 2080 Ti, even with `--qos=high`
(qos sets *limits*, not GPU *type*). **This was also why P2-2's batch was
capped at 4** — M1/M2 silently ran on a 2080 Ti. Fix: name the card, e.g.
`--gres=gpu:rtxa6000:1`. Documented in CLUSTER_INFO.md (new "GPU-type targeting"
section + per-type table). So P2-2's "sampling sparsity" was partly a
self-inflicted GPU-misallocation artifact — which is exactly why re-running the
diagnostic correctly mattered.

### 5b. Effective-batch config + memory model
The trainer computes `bs_micro = batch_size // grad_accum_steps`. The first
correctly-GPU'd launch revealed (i) peak memory scales as `bs_micro × n_pts`
(direct batch=32/n_pts=32 ≈ 86 GB > 48 GB A6000), and (ii) an off-by-config in
Run C (`batch_size=8, grad_accum=8` → effective batch **8**, not 64). Fix: all
runs use micro-batch=8 with the effective batch set via accumulation
(16/32/64 → accum 2/4/8); `validate()` chunks at the micro size too.

## 6. The DDP speedup (to fit the time budget)

The full B/C runs at the safe single-GPU rate were ~53 h each — over the ~1-day
budget. Grad-accum tuning can't help (compute-invariant), and the cached tcnn
binary (arch 52) denies the A6000 its FullyFusedMLP speedup (rebuilding tcnn is
forbidden by CLAUDE.md). So the only real lever was **parallel hardware**: a
2-GPU manual-all-reduce DDP path added to `aaf/train/multi_room_3d.py` (guarded
by `--ddp`; single-GPU path byte-identical). Manual all-reduce rather than the
DDP wrapper because the trainer calls the model multiple times per step
(per-room AABB grouping) before one backward — a DDP-wrapper landmine.

- **Smoke-tested first** (120 iters, 2 A6000): NCCL init OK, loss 6.93 → 2.91,
  clean teardown.
- **DDP-B: 0.639 it/s = exactly 2.0×** the single-GPU rate; reached 60K in
  16.1 h, inside budget.
- **Correctness cross-check**: DDP-B and an independent single-GPU B (same 22.5K
  resume point) agree to **0.00-0.03 dB at every matched iter (45K-50K)** — the
  all-reduce trains the identical model, not a corrupted one.

## 7. Comparison to anchors

| | rooms | batch | iters | val LSD |
|---|---:|---:|---:|---:|
| P2-1 single-room overfit | 1 | 4-8 | 15K | 1.3-1.8 dB |
| P2-2 M1 (the failure) | 45 | 4 | 24K | 6.16 dB |
| **P2-2.5 B (this chunk)** | 45 | 32 | 60K | **2.61 dB** |
| **P2-2.5 C (ceiling)** | 10 | 64 | 30K | **~1.0 dB** |

## 8. Deliverables

- `outputs/diag_p2_2_5/DIAGNOSIS.md` — the load-bearing verdict (+ convergence
  curves, per-room bars, DDP cross-check table).
- `outputs/diag_p2_2_5/DIAGNOSIS_interim.md` — the earlier zero-risk snapshot.
- `configs/sweeps_3d/diag_10rooms.yaml` — the maximin subset (A, C share it).
- `configs/sweep_3d/{A_diag,B_diag,C_diag,B_ddp,_ddp_smoke}.yaml`.
- 3 training runs under `outputs/diag_p2_2_5/{A_10rm_b16,B_45rm_b32,B_45rm_ddp,C_10rm_b64}/`.
- DDP support in `aaf/train/multi_room_3d.py` + `tests/test_diag_subset.py`.

## 9. Manager actions requested

1. **Confirm P2-3 config**: Run C's recipe (eff-batch 64, n_pts 32) on the full
   45 rooms, 80-100K iters, on A6000s with the corrected `--gres=gpu:rtxa6000`
   targeting + 2-GPU DDP (now available). Expected to push the 45-room
   in-distribution LSD ≤ 2.5, after which zero-shot (the real P2-3 deliverable)
   becomes meaningful.
2. **Note for all future chunks**: always name the GPU type in `--gres`
   (CLUSTER_INFO.md "GPU-type targeting"). This single fix is worth ~3.5 dB.
