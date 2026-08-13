# P3-2b dataset gate — **PASS**

Manifest `configs/sweeps_2d_mat/p3_2b_manifest.json` (seed 20260813, rows_sha256 `ecf0ee6e620dc56e`): **960 train + 210 test**.

| check | result |
|---|---|
| G1 no training draw inside a held-out slab | PASS (0 violations) |
| G2 no training alpha equals a demo preset (1e-6) | PASS (0 collisions) |
| G3 per-wall m coverage uniform outside slabs, empty inside | PASS |
| G4 block-diagonal signature reproduces on new-generator configs | PASS |

## Per-wall m coverage (12 bins over [0.02, 1.61]; `X` = slab bin)

| wall | n draws | histogram | slab bins | in-slab |
|---|---:|---|---|---:|
| west | 431 | [57, 42, 39, 33, 13, 11, 41, 36, 46, 43, 35, 35] | `.....X......` | 0 |
| east | 430 | [37, 39, 42, 29, 28, 51, 38, 31, 36, 34, 34, 31] | `............` | 0 |
| south | 430 | [42, 30, 38, 32, 30, 30, 43, 34, 27, 41, 40, 43] | `............` | 0 |
| north | 429 | [46, 30, 49, 39, 48, 37, 32, 27, 16, 17, 43, 45] | `........XX..` | 0 |

## Physics signature on new-generator configs

| config | wall | alpha | ΔBW own | ΔBW other | selectivity |
|---|---|---:|---:|---:|---:|
| L3.68_W4.03_west0.690 | west | 0.6903 | +12.238 | +0.368 | 33.3 |
| L3.68_W4.03_east0.716 | east | 0.7157 | +14.137 | +0.757 | 18.7 |

Because alpha is drawn continuously, the demo presets (0.05 / 0.50 / 0.70) have probability zero of appearing exactly in training — **every preset evaluation is at an unseen exact value by construction**.
