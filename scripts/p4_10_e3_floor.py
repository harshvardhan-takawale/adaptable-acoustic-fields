"""P4-10 E3 (D206, D207): is the late floor common to the room? On the cached original-room renders (in-family 15): for each octave (63 / 125 / 250 Hz)
take the band-limited impulse responses (octave_ir, no onset alignment, so the receivers share an absolute time axis), the window [1.2, 1.5) s of the record (the last 0.3 s of a 1.5 s
gate), off-lattice receivers >= 1 m from the source, and compute ratio = energy of the receiver-mean signal / mean per-receiver energy (1 = every receiver has the same signal; 1 / N_rx
for independent noise). Model and truth, per room; mean over rooms with a bootstrap CI (10k, seed 20261013). CPU; reads the caches.
  python scripts/p4_10_e3_floor.py R0_120K CTL_150K RCP_150K S2X_240K R1_150K -> outputs/p4_10/e3/report.{json,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from aaf.eval.octave_decay import FS, octave_ir
from scripts.p4_10_d1_decay import rooms_of
from scripts.p4_10_scorecard import boot, on_lattice

SRC = np.array([0.5, 1.6])
BANDS3 = ("63", "125", "250")


def ratio(H, band):
    h = octave_ir(np.asarray(H), band)
    a, b = int(round(1.2 * FS)), int(round(1.5 * FS))
    w = h[:, a:b]
    return float((w.mean(0) ** 2).sum() / max((w ** 2).sum(1).mean(), 1e-300)), int(len(h))


def main(models):
    OUT = Path("outputs/p4_10/e3")
    OUT.mkdir(parents=True, exist_ok=True)
    rep, L = {}, ["# P4-10 E3: how much of the late floor is common to the room (in-family 15)", "", "ratio = energy of the receiver-mean signal in [1.2, 1.5) s of the band-limited IR / mean per-receiver energy (1 = common to all receivers; 1 / N for independent noise)", "",
                  "| model | octave | model ratio [95% CI] | truth ratio [95% CI] | model / truth | N_rx (typical) |", "|---|---|---|---|---|---|"]
    for name in models:
        rep[name] = {}
        for b in BANDS3:
            m, t, n = [], [], []
            for P, T, rx in rooms_of(name, "test"):
                sel = ~on_lattice(rx) & (np.hypot(rx[:, 0] - SRC[0], rx[:, 1] - SRC[1]) >= 1.0)
                r1, k = ratio(np.asarray(P)[sel], b)
                r2, _ = ratio(np.asarray(T)[sel], b)
                m.append(r1)
                t.append(r2)
                n.append(k)
            rep[name][b] = {"model": boot(m), "truth": boot(t), "n_rx_median": int(np.median(n)), "n_rooms": len(m)}
            q = rep[name][b]
            L.append("| {} | {} | {:.4f} [{:.4f}, {:.4f}] | {:.4f} [{:.4f}, {:.4f}] | {:.1f} | {} |".format(
                name, b, *q["model"], *q["truth"], q["model"][0] / max(q["truth"][0], 1e-12), q["n_rx_median"]))
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:] or ["R1_150K"])
