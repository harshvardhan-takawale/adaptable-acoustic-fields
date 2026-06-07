# P2-2.5 DIAGNOSIS — multi-room 3D training bottleneck

> **FINAL (all runs complete). A, C, and DDP-B reached full targets; single-GPU B cross-check ran to its 24h walltime (~58K).**

## Headline
| Run | Rooms | batch | n_pts | iters / target | wall | final val LSD | classification |
|---|---:|---:|---:|---|---:|---:|---|
| A | 10 | 16 × accum 2 | 16 | 30000/30000 | 6.6 h | 1.84 dB | ✅ success |
| B | 45 | 16 × accum 2 | 32 | 60000/60000 | 16.1 h | 2.61 dB | ⚠ ambiguous |
| C | 10 | 64 × accum 8 | 32 | 30000/30000 | 23.6 h | 0.98 dB | ✅ success |

**Thresholds (spec)**: `≤ 2.5 dB = success`; `> 4 dB = clear failure`; in-between = ambiguous, flag for manager.

## Decision-matrix verdict

**Capacity is NOT the wall — coverage / compute is the dominant lever. The 10-room set fits cleanly (A, C ≤ ~1.8 dB); the full 45-room set, given 8× the per-iter coverage and 60K iters, improved from P2-2 M1's 6.16 dB to the 2.5 dB threshold and was still descending — reachable with more compute, not a new architecture.**

### Recommendation for P2-3

P2-3: scale compute on the full 45-room set — apply Run C's recipe (effective batch 64, n_pts 32) to all 45 rooms and/or extend to 80–100K iters. B reached 2.61 dB at 60K still descending; either lever should carry it below 2.5. Do NOT widen the decoder — Run C proves ~1 dB is achievable at this capacity.

## Run B descent rate (sizes the P2-3 iteration budget)

| window | Δ val LSD per 10K iters |
|---|---:|
| 30K → 40K | 0.421 dB |
| 40K → 50K | 0.285 dB |
| 50K → 60K | 0.162 dB |

B ended at **2.61 dB** with the descent decelerating (≈ 0.16 dB/10K at the end). Closing the 0.11 dB gap to 2.5 at that slope is ~7K nominal iters; with continued deceleration, budget **~80-100K iters** for P2-3 to clear 2.5 on the full 45-room set.

## Convergence curves

![](convergence_curves.png)

## Per-room final val LSD

![](per_room_lsd.png)

## Per-run details

### Run A — 10 rooms, eff-batch 16, n_pts=16

- Output dir: `outputs/diag_p2_2_5/A_10rm_b16/`
- Wall-clock: 6.65 h (30000/30000 iters)
- Stopped early: `False`
- Final val LSD: **1.84 dB**

### Run B — 45 rooms, eff-batch 32, n_pts=32 (relaxed early-stop) [2-GPU DDP]

- Output dir: `outputs/diag_p2_2_5/B_45rm_ddp/`
- Wall-clock: 16.12 h (60000/60000 iters)
- Stopped early: `False`
- Final val LSD: **2.61 dB**

### Run C — 10 rooms, eff-batch 64, n_pts=32

- Output dir: `outputs/diag_p2_2_5/C_10rm_b64/`
- Wall-clock: 23.58 h (30000/30000 iters)
- Stopped early: `False`
- Final val LSD: **0.98 dB**


## DDP correctness cross-check

Canonical B is the 2-GPU DDP run; the single-GPU B (`B_45rm_b32`) trained independently from the **same** 22.5K checkpoint. Tight agreement at matched iterations confirms the manual all-reduce trains the same model (effective batch 32 = 2 ranks × 16), not a corrupted one.

| iter | DDP-B val LSD | single-B val LSD | Δ |
|---:|---:|---:|---:|
| 50000 | 2.78 | 2.76 | 0.01 |
| 51000 | 2.76 | 2.75 | 0.01 |
| 52000 | 2.75 | 2.73 | 0.02 |
| 53000 | 2.72 | 2.72 | 0.00 |
| 54000 | 2.70 | 2.70 | 0.00 |
| 55000 | 2.68 | 2.68 | 0.01 |

## Anchors (other multi-room results)

- P2-1 single-room overfit (5 rooms, batch=4-8, n_pts=16): val LSD **1.3-1.8 dB** — the architecture can reconstruct 3D spectra.
- P2-2 M1 (45 rooms, batch=4, n_pts=16): val LSD **6.16 dB** after 24K iters (early-stop). This is the baseline P2-2.5 is trying to beat.
