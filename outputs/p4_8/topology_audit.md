# P4-8 Step 0a -- topology audit (D174(c))

Classified from each room's polygon (m2 classifier, generalised). Counts per split.

| corpus | split | rect | L | T-like | DN | U | LT | TT | C3 | other |
|---|---|---|---|---|---|---|---|---|---|---|
| C250 | train | 8 | 242 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C250 | test | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C500 | train | 8 | 492 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C500 | test | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CX500 | train | 0 | 143 | 174 | 19 | 0 | 110 | 54 | 0 | 0 |
| CX1000 | train | 0 | 268 | 342 | 41 | 0 | 230 | 119 | 0 | 0 |
| family (FAM) | train | 60 | 60 | 60 | 60 | 60 | 0 | 0 | 0 | 0 |
| family (FAM) | test | 10 | 10 | 10 | 10 | 10 | 0 | 0 | 0 | 0 |
| family holdU (G2) | train | 60 | 60 | 60 | 60 | 0 | 0 | 0 | 0 | 0 |
| family holdU (G2) | test_family | 0 | 0 | 0 | 0 | 70 | 0 | 0 | 0 | 0 |
| family holdU (G2) | test | 10 | 10 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |

**GX's P4-7 '+0.110 out of family' restated per family** (25-100 / 100-200 / 200-300 Hz, off-lattice r):

| family | GX @60K | G1 @60K | GX - G1 | in GX's training corpus? |
|---|---|---|---|---|
| T | 0.603/0.441/0.399 | 0.552/0.287/0.349 | +0.051/+0.154/+0.050 | **yes** (CX500 has this topology) |
| DN | 0.819/0.516/0.396 | 0.614/0.395/0.322 | +0.206/+0.121/+0.073 | **yes** (CX500 has this topology) |
| U | 0.790/0.495/0.463 | 0.718/0.434/0.402 | +0.072/+0.061/+0.061 | no (truly unseen) |
