# P4-7 prep p5 -- wall impedance per material (D164)

FDTD boundary: Kowalczyk-van Walstijn, locally reacting, **frequency-independent** real admittance. beta = xi = (1-R)/(1+R) = tanh(m/4), R = sqrt(1-alpha), m = -ln(1-alpha). Robin condition in the repo's rfft convention: **dp/dn_out = -i k beta p**.

| material | alpha | m | R | beta (= xi, rho c / Z) | Z / (rho c) |
|---|---|---|---|---|---|
| M0 brick / ALPHA_BASELINE (every P4 wall, divider, notch mask) | 0.1500 | 0.1625 | 0.92195 | 0.04061 | 24.626 |
| M1 concrete | 0.0500 | 0.0513 | 0.97468 | 0.01282 | 77.987 |
| UNSEEN material | 0.3000 | 0.3567 | 0.83666 | 0.08893 | 11.244 |
| M2 curtain | 0.5000 | 0.6931 | 0.70711 | 0.17157 | 5.828 |
| M3 absorber panel | 0.7000 | 1.2040 | 0.54772 | 0.29222 | 3.422 |
| P3-2c preset | 0.7500 | 1.3863 | 0.50000 | 0.33333 | 3.000 |
| P3-2c preset / continuous-range top | 0.8000 | 1.6094 | 0.44721 | 0.38197 | 2.618 |
| window (segment corpus) | 0.9300 | 2.6593 | 0.26458 | 0.58156 | 1.720 |
| window top (segment corpus) | 0.9500 | 2.9957 | 0.22361 | 0.63451 | 1.576 |
| continuous-range bottom | 0.0198 | 0.0200 | 0.99005 | 0.00500 | 200.015 |
