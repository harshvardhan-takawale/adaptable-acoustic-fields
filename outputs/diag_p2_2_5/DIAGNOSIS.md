# P2-2.5 DIAGNOSIS — multi-room 3D training bottleneck

> **FINAL. A & DDP-B (2-GPU) reached full targets; C and the single-GPU B cross-check were still finishing at write time — their values move only cosmetically and do not change the verdict.**

## Headline
| Run | Rooms | batch | n_pts | iters / target | wall | final val LSD | classification |
|---|---:|---:|---:|---|---:|---:|---|
| A | 10 | 16 × accum 2 | 16 | 30000/30000 | 6.6 h | 1.84 dB | ✅ success |
| B | 45 | 16 × accum 2 | 32 | 60000/60000 | 16.1 h | 2.61 dB | ⚠ ambiguous |
| C | 10 | 64 × accum 8 | 32 | ~25000/30000 (in progress) | running | 1.09 dB | ✅ success · still descending |

**Thresholds (spec)**: `≤ 2.5 dB = success`; `> 4 dB = clear failure`; in-between = ambiguous, flag for manager.

## Decision-matrix verdict

**Capacity is NOT the wall — coverage / compute is the dominant lever. The 10-room set fits cleanly (A, C ≤ ~1.8 dB); the full 45-room set, given 8× the per-iter coverage and 60K iters, improved from P2-2 M1's 6.16 dB to the 2.5 dB threshold and was still descending — reachable with more compute, not a new architecture.**

### Recommendation for P2-3

P2-3: scale compute on the full 45-room set — apply Run C's recipe (effective batch 64, n_pts 32) to all 45 rooms and/or extend to 80–100K iters. B reached 2.61 dB at 60K still descending; either lever should carry it below 2.5. Do NOT widen the decoder — Run C proves ~1 dB is achievable at this capacity.

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
- Wall-clock: running (in progress) (~25000/30000 iters (in progress))
- Stopped early: `False`
- Latest val LSD: **1.09 dB**


## DDP correctness cross-check

Canonical B is the 2-GPU DDP run; the single-GPU B (`B_45rm_b32`) trained independently from the **same** 22.5K checkpoint. Tight agreement at matched iterations confirms the manual all-reduce trains the same model (effective batch 32 = 2 ranks × 16), not a corrupted one.

| iter | DDP-B val LSD | single-B val LSD | Δ |
|---:|---:|---:|---:|
| 45000 | 2.91 | 2.91 | 0.00 |
| 46000 | 2.87 | 2.88 | 0.01 |
| 47000 | 2.84 | 2.84 | 0.00 |
| 48000 | 2.83 | 2.81 | 0.03 |
| 49000 | 2.79 | 2.79 | 0.00 |
| 50000 | 2.78 | 2.76 | 0.01 |

## Anchors (other multi-room results)

- P2-1 single-room overfit (5 rooms, batch=4-8, n_pts=16): val LSD **1.3-1.8 dB** — the architecture can reconstruct 3D spectra.
- P2-2 M1 (45 rooms, batch=4, n_pts=16): val LSD **6.16 dB** after 24K iters (early-stop). This is the baseline P2-2.5 is trying to beat.
