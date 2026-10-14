"""P4-5: is the model a continuous field, or a lookup table at the training-receiver positions?

Every training and test room in the P4 corpora places its 800 receivers on ONE absolute lattice,
x, y = 0.12 + 0.06 k (`build_p4_2_shapes.receivers`), and the hash encoding is indexed by
absolute position -- so the same lattice points are ray origins in every one of the 250 training
rooms. The dense 64x64 test lattice, snapped to the 0.02 m FDTD nodes, puts ~1 point in 9 back on
that lattice and the rest 2-3 cm off it. That makes a matched comparison possible with no new
simulation: same rooms, same FDTD run, same density, split only by lattice membership.

A 2 cm move is 1/57 of a wavelength at 300 Hz. Physics cannot make accuracy depend on it; a
model that has memorised the receiver positions can. So accuracy is also binned by distance to
the nearest lattice point: memorisation shows as a SPIKE at distance 0, not a smooth falloff.

Reports, per band: spatial Pearson (dB), band LSD, energy ratio dE and contrast ratio k (D98's
quantities), for on-lattice / off-lattice / all, plus r by distance to the lattice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.build_p4_5_rooms import test_shapes_for_a2
from scripts.p4_5_eval import DENSE_DIR, _db, _pearson_cols

LAT0, LAT_STEP = 0.12, 0.06
BANDS = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
LSD_BANDS = {"0-100": (0, 200), "100-200": (200, 400), "200-300": (400, 601), "0-300": (0, 601)}
DIST = [(0.0, 1e-6, "0"), (0.019, 0.021, "0.020"), (0.027, 0.029, "0.028"),
        (0.039, 0.041, "0.040"), (0.044, 0.046, "0.045")]


def dist_to_lattice(P):
    k = np.round((P - LAT0) / LAT_STEP)
    return np.hypot(*(P - (LAT0 + LAT_STEP * k)).T)


def stats(P, T):
    dP, dT = _db(P), _db(T)
    r = _pearson_cols(dP, dT)
    dE = 10 * np.log10((np.abs(P) ** 2).mean(0) / (np.abs(T) ** 2).mean(0))
    k = dP.std(0) / np.maximum(dT.std(0), 1e-12)
    out = {"n_rx": int(P.shape[0])}
    for nm, (a, b) in BANDS.items():
        out[nm] = {"r_db": float(np.nanmean(r[a:b])), "dE_db": float(np.mean(dE[a:b])),
                   "k": float(np.mean(k[a:b]))}
    out["lsd_db"] = {nm: float(np.mean(np.abs(dP[:, a:b] - dT[:, a:b])))
                     for nm, (a, b) in LSD_BANDS.items()}
    return out


def _mean(dicts):
    """Average the per-room stats (each room weighs equally)."""
    out = {"n_rx": int(sum(d["n_rx"] for d in dicts))}
    for nm in BANDS:
        out[nm] = {q: float(np.mean([d[nm][q] for d in dicts])) for q in ("r_db", "dE_db", "k")}
    out["lsd_db"] = {nm: float(np.mean([d["lsd_db"][nm] for d in dicts])) for nm in LSD_BANDS}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/p4_5/eval/cache/render_baseline_60K.npz")
    ap.add_argument("--name", default="baseline_60K")
    ap.add_argument("--out", default="outputs/p4_5/diagnostics")
    a = ap.parse_args()
    z = np.load(a.cache, allow_pickle=False)
    per = {"on": [], "off": [], "all": []}
    dist = {lab: [] for _lo, _hi, lab in DIST}
    nd = {lab: 0 for _lo, _hi, lab in DIST}
    for i, (tag, c) in enumerate(test_shapes_for_a2()):
        with h5py.File(Path(DENSE_DIR) / c.filename) as f:
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        P, T = z["dP%d" % i], z["dT%d" % i]
        if P.shape[0] != rx.shape[0]:
            raise AssertionError("cache and data disagree on receiver count for " + c.filename)
        d = dist_to_lattice(rx)
        on = d < 1e-6
        per["on"].append(stats(P[on], T[on]))
        per["off"].append(stats(P[~on], T[~on]))
        per["all"].append(stats(P, T))
        for lo, hi, lab in DIST:
            s = (d >= lo) & (d <= hi)
            if s.sum() >= 50:
                r = _pearson_cols(_db(P[s]), _db(T[s]))
                dist[lab].append([float(np.nanmean(r[a_:b_])) for a_, b_ in BANDS.values()])
                nd[lab] += int(s.sum())
    rep = {"name": a.name, "cache": a.cache, "lattice": "x, y = 0.12 + 0.06 k",
           "on_lattice": _mean(per["on"]), "off_lattice": _mean(per["off"]),
           "all": _mean(per["all"]),
           "r_by_distance_to_lattice_m": {lab: {"n_rx": nd[lab],
                                                **dict(zip(BANDS, np.mean(v, 0).tolist()))}
                                          for lab, v in dist.items() if v}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out / "lattice_{}.json".format(a.name), "w"), indent=1)
    print("{}  (3 frozen test shapes, dense)".format(a.name))
    print("                 n_rx  | r_db 25-100 100-200 200-300 | dE dB 25-100 100-200 200-300 | "
          "k 25-100 100-200 200-300 | LSD 0-100 100-200 200-300")
    for nm in ("on_lattice", "all", "off_lattice"):
        s = rep[nm]
        print("  {:12s} {:5d}  |      {}  |       {}  |   {}  |    {}".format(
            nm, s["n_rx"], " ".join("{:+.3f}".format(s[b]["r_db"]) for b in BANDS),
            " ".join("{:+6.2f}".format(s[b]["dE_db"]) for b in BANDS),
            " ".join("{:5.2f}".format(s[b]["k"]) for b in BANDS),
            " ".join("{:5.2f}".format(s["lsd_db"][b]) for b in ("0-100", "100-200", "200-300"))))
    print("  r by distance to the nearest training-lattice point:")
    for lab, v in rep["r_by_distance_to_lattice_m"].items():
        print("    {:6s} m  n_rx {:5d}  r {}".format(lab, v["n_rx"],
                                                     " ".join("{:+.3f}".format(v[b]) for b in BANDS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
