# GPU memory smoke check — multi-room (Chunk 3)

**GPU**: NVIDIA GeForce GTX TITAN X  
**Total memory**: 12.8 GB

## Configurations tried
| n_azi | n_pts | batch | status | peak GB | fwd+bwd s |
|------:|------:|------:|--------|--------:|----------:|
| 64 | 32 | 16 | oom | — | — |
| 64 | 32 | 8 | pass | 8.10 | 0.76 |

## Chosen configuration
**PASS** — `n_azi=64`, `n_pts_per_ray=32`, `batch=8` (effective batch 16 via `grad_accum_steps=2`). Peak working set **8.10 GB** (fwd+bwd 0.76s).
