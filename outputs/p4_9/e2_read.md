# P4-9 I11: DIR / DIR-G at 150K -- E2 (D188), C2, C3

Directional r = Re rho vs FDTD grad p / (jk), receivers >= 0.5 m from the source; in-family 15 shapes decide, unseen 16 reported beside. Paired per-room bootstrap (10k, seed 20261013).

## DIR_150K vs CTL_150K

| band | arm readout (level) | CTL bar (level) | delta in-family [95% CI] | delta unseen [95% CI] | band |
|---|---|---|---|---|---|
| 25-100 Hz | proj-FD (0.947) | proj-FD (0.945) | +0.002 [+0.001, +0.003] | +0.008 [+0.004, +0.011] | -- |
| 100-200 Hz | proj-FD (0.852) | proj-FD (0.849) | +0.003 [+0.001, +0.005] | -0.001 [-0.004, +0.001] | -- |
| 200-300 Hz | proj-FD (0.808) | proj-FD (0.806) | +0.002 [+0.001, +0.003] | -0.003 [-0.007, +0.000] | -- |

**E2 (D188, primary): FAILS** (0 of 3 bands). Literal reading (CTL bar = raw-FD only, labelled): passes (3 of 3).

**C2 (no harm): HOLDS** -- HF +0.003 [+0.002, +0.005]; UF_25-100 +0.004 [-0.002, +0.010]; UF_100-200 -0.006 [-0.011, -0.002]; UF_200-300 -0.006 [-0.009, -0.003].
**C3 (exploratory): does not hold** -- UF 25-100 Hz +0.004 [-0.002, +0.010].
Levels (Rule 12): DIR_150K HF 0.587, UF 0.471 / 0.369 / 0.367; CTL_150K HF 0.583, UF 0.467 / 0.376 / 0.373

**DIRG_150K**: no directional render yet.

