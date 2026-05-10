# GPU memory smoke check

**GPU**: NVIDIA GeForce GTX TITAN X  
**Total memory**: 12.8 GB

## Configurations tried
| n_azi | n_pts_per_ray | batch | status | peak GB | fwd+bwd s |
|------:|--------------:|------:|--------|--------:|----------:|
| 64 | 64 | 8 | oom | — | — |
| 64 | 32 | 8 | pass | 8.09 | 0.75 |

## Chosen configuration
**PASS** — `n_azi=64`, `n_pts_per_ray=32`, `batch=8`. Peak working set **8.09 GB** (fwd+bwd 0.75s).
