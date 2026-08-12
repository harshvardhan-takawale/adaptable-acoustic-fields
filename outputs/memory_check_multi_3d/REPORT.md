# 3D GPU memory smoke check

**GPU**: NVIDIA GeForce GTX TITAN X  
**Total memory**: 12.8 GB

## Configurations tried
| n_azi | n_ele | n_rays | n_pts | batch | status | peak GB | fwd+bwd s |
|------:|------:|-------:|------:|------:|--------|--------:|----------:|
| 16 | 16 | 258 | 32 | 8 | oom | — | — |
| 16 | 16 | 258 | 16 | 8 | oom | — | — |
| 16 | 16 | 258 | 32 | 4 | oom | — | — |
| 16 | 16 | 258 | 16 | 4 | pass | 8.30 | 0.84 |

## Chosen configuration
**PASS** — `n_azi=16`, `n_ele=16`, `n_pts_per_ray=16`, `batch=4` (258 rays). Peak working set **8.30 GB** (fwd+bwd 0.84s).
