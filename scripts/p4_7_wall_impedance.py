"""P4-7 prep p5 (D164): wall impedance per material, extracted from the FDTD boundary model, for a later
wall-condition loss.

The FDTD boundary (`aaf/sim/fdtd_2d.py`, Kowalczyk-van Walstijn, solver "fdtd_2d_slf_kw") is locally
reacting and FREQUENCY-INDEPENDENT: one real scalar per wall face,
    R  = sqrt(1 - alpha)                  normal-incidence pressure reflection   (`wall_reflection`)
    xi = (1 - R) / (1 + R)                normalised specific ADMITTANCE rho c / Z (`wall_admittance`)
    Z / (rho c) = 1 / xi                                                         (`wall_impedance`)
In the project's absorption parameter m = -ln(1 - alpha) (`aaf/walls.py`), R = exp(-m/2) and
    xi = tanh(m / 4)                      (closed form; checked numerically below).
Boundary condition (time domain, outward normal n): dp/dn = -(xi / c) dp/dt. In the repo's rfft
convention X(f) = sum_t x(t) exp(-2 pi i f t) (so d/dt -> +i omega), the frequency-domain Robin condition is
    dp/dn_out = -i k xi p,   k = 2 pi f / c,
with xi independent of f (no reactance). Discrete note (labelled inference, not used): the centred KW
update realises xi * cos(omega dt / 2) / cos(k dx / 2), about +0.1% at 300 Hz for dx 0.02. Validation already
in the repo: modal damping FDTD 23.945 vs exact 24.175 (-1.0%, `scripts/p3_3_boundary_validation.py`).
Writes outputs/p4_7/p5_wall_impedance.{json,md}.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from aaf.sim.fdtd_2d import wall_admittance, wall_impedance, wall_reflection

OUT = Path("outputs/p4_7")
MATERIALS = [
    ("M0 brick / ALPHA_BASELINE (every P4 wall, divider, notch mask)", 0.15, "aaf/walls.py:41-49; shape_configs.ALPHA"),
    ("M1 concrete", 0.05, "aaf/walls.py:41-49"),
    ("UNSEEN material", 0.30, "aaf/data/mat_configs.py:55"),
    ("M2 curtain", 0.50, "aaf/walls.py:41-49"),
    ("M3 absorber panel", 0.70, "aaf/walls.py:41-49"),
    ("P3-2c preset", 0.75, "aaf/data/mat_configs_p3_2c.py"),
    ("P3-2c preset / continuous-range top", 0.80, "aaf/data/mat_configs_p3_2c.py; mat_configs_cont m<=1.61"),
    ("window (segment corpus)", 0.93, "aaf/data/seg_configs.py"),
    ("window top (segment corpus)", 0.95, "aaf/data/seg_configs.py"),
    ("continuous-range bottom", 0.0198, "aaf/data/mat_configs_cont.py m>=0.02"),
]


def main():
    rows = []
    for name, a, src in MATERIALS:
        R, xi, Z = wall_reflection(a), wall_admittance(a), wall_impedance(a)
        m = -math.log(1.0 - a)
        assert abs(xi - math.tanh(m / 4.0)) < 1e-12, (a, xi, math.tanh(m / 4.0))
        rows.append({"material": name, "alpha": a, "m": m, "R": R, "beta_xi_admittance": xi,
                     "Z_over_rho_c": Z, "source": src})
    rep = {"model": "Kowalczyk-van Walstijn locally reacting, frequency-independent real admittance",
           "formulas": {"R": "sqrt(1-alpha)", "xi": "(1-R)/(1+R) = tanh(m/4), m = -ln(1-alpha)",
                        "Z/(rho c)": "1/xi",
                        "robin_time": "dp/dn_out = -(xi/c) dp/dt",
                        "robin_freq_rfft_convention": "dp/dn_out = -i k xi p, k = 2 pi f / c"},
           "frequency_dependence": "none (real scalar per face; no mass/spring/resistance filter)",
           "discrete_note": "centred KW realises xi*cos(w dt/2)/cos(k dx/2), ~+0.1% at 300 Hz, dx 0.02 (inference)",
           "materials": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p5_wall_impedance.json").write_text(json.dumps(rep, indent=1))
    md = ["# P4-7 prep p5 -- wall impedance per material (D164)", "",
          "FDTD boundary: Kowalczyk-van Walstijn, locally reacting, **frequency-independent** real admittance. "
          "beta = xi = (1-R)/(1+R) = tanh(m/4), R = sqrt(1-alpha), m = -ln(1-alpha). "
          "Robin condition in the repo's rfft convention: **dp/dn_out = -i k beta p**.", "",
          "| material | alpha | m | R | beta (= xi, rho c / Z) | Z / (rho c) |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {} | {:.4f} | {:.4f} | {:.5f} | {:.5f} | {:.3f} |".format(
            r["material"], r["alpha"], r["m"], r["R"], r["beta_xi_admittance"], r["Z_over_rho_c"]))
    (OUT / "p5_wall_impedance.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
