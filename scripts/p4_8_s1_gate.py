"""P4-8 S1 gate (D174(g)): the free diagnostic that decides whether a Helmholtz-residual regulariser is launched.

On each dense room's 64x64 lattice (the off-lattice evaluation receivers), per bin f in 25-300 Hz:
    R = lap_h F + k^2 (F - <F>),     rel(f) = ||R|| / ||k^2 (F - <F>)||   over INTERIOR stencils
where <F> is the complex spatial mean (the room-uniform record component, D170(c), removed as the manager specifies;
a constant has zero Laplacian, so only the k^2 term carries it), and lap_h is the 5-point stencil with UNEQUAL
spacings taken from the stored (solver-snapped) receiver coordinates: d2F/dx2 = 2[(F+ - F0)/h+ - (F0 - F-)/h-] /
(h+ + h-), likewise in y. A stencil is interior when its 4 neighbours are on the room's lattice; stencils within
0.3 m of the source are dropped (the field is singular there).
Gate (i): the FDTD ground truth's band-mean rel residual <= 0.10 (D174(g)); (ii) on G1 / GX predictions, the
residual on unseen-topology rooms is >= 2x the in-family one. Reported either way. Writes outputs/p4_8/s1_gate.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

C0, DF, LO, HI = 343.0, 0.5, 50, 601
SRC = np.array([0.5, 1.6])
OUT = Path("outputs/p4_8")


def stencils(keep, rx):
    """Index arrays (c, xm, xp, ym, yp) into the receiver list, and the spacings, for interior lattice nodes."""
    idx = -np.ones(keep.shape, dtype=np.int64)
    idx[keep] = np.arange(int(keep.sum()))
    rows = []
    for i in range(1, 63):
        for j in range(1, 63):
            c, a, b, d, e = idx[i, j], idx[i - 1, j], idx[i + 1, j], idx[i, j - 1], idx[i, j + 1]
            if min(c, a, b, d, e) < 0:
                continue
            if np.hypot(*(rx[c] - SRC)) < 0.3:
                continue
            rows.append((c, a, b, d, e))
    s = np.asarray(rows, dtype=np.int64)
    hxm = rx[s[:, 0], 0] - rx[s[:, 1], 0]
    hxp = rx[s[:, 2], 0] - rx[s[:, 0], 0]
    hym = rx[s[:, 0], 1] - rx[s[:, 3], 1]
    hyp = rx[s[:, 4], 1] - rx[s[:, 0], 1]
    return s, hxm, hxp, hym, hyp


def residual(F, keep, rx):
    """Band-mean (25-300 Hz) relative Helmholtz residual of a field F [n_rx, >=601]."""
    s, hxm, hxp, hym, hyp = stencils(keep, rx)
    f = np.arange(LO, HI) * DF
    k2 = (2 * np.pi * f / C0) ** 2
    X = F[:, LO:HI]
    Xc = X - X.mean(0)
    c, a, b, d, e = (X[s[:, q]] for q in range(5))
    lap = (2 * ((b - c) / hxp[:, None] - (c - a) / hxm[:, None]) / (hxp + hxm)[:, None]
           + 2 * ((e - c) / hyp[:, None] - (c - d) / hym[:, None]) / (hyp + hym)[:, None])
    R = lap + k2[None] * Xc[s[:, 0]]
    rel = np.linalg.norm(R, axis=0) / np.linalg.norm(k2[None] * Xc[s[:, 0]], axis=0)
    return float(rel.mean()), {"25-100": float(rel[:150].mean()), "100-200": float(rel[150:350].mean()),
                                "200-300": float(rel[350:].mean())}, int(len(s))


def rooms_test():
    from aaf.data.shape_configs import configs_from_rows
    cs = configs_from_rows(json.load(open("configs/sweeps_2d_mat/p4_2_shapes_manifest.json"))["configs"], split="test")
    return [("test", Path("data/track_p4_5_testdense") / c.filename) for c in cs]


def rooms_unseen():
    m = json.load(open("configs/sweeps_2d_mat/p4_8_unseen_manifest.json"))["configs"]
    return [(r["family"], Path("data/track_p4_8_unseen") / r["filename"]) for r in m]


def main():
    rep = {"definition": __doc__.split("\n\n")[1].strip(), "fdtd": [], "models": {}}
    groups = {"test": rooms_test(), "unseen": rooms_unseen()}
    fields = {}
    for g, rs in groups.items():
        for fam, p in rs:
            with h5py.File(p) as f:
                T = np.asarray(f["ism/H_complex"])[:, :HI]
                rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
                keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
            fields[(g, p.name)] = (keep, rx)
            m, bands, n = residual(T, keep, rx)
            rep["fdtd"].append({"group": g, "family": fam, "room": p.name, "rel": m, "bands": bands, "n_stencils": n})
            print("FDTD {:6s} {:3s} {:60s} rel {:.3f} {}".format(g, fam, p.name[:60], m,
                                                                    {k: round(v, 3) for k, v in bands.items()}), flush=True)
    for name in ("G1_60K", "GX_60K"):
        res = {"test": [], "unseen": [], "unseen_U": [], "unseen_C3": []}
        zt = np.load("outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz".format(name))
        zu = np.load("outputs/p4_8/cache/unseen_{}.npz".format(name))
        for i, (_fam, p) in enumerate(groups["test"]):
            keep, rx = fields[("test", p.name)]
            assert np.allclose(zt["rx%d" % i], rx)
            res["test"].append(residual(zt["P%d" % i], keep, rx)[0])
        for i, (fam, p) in enumerate(groups["unseen"]):
            keep, rx = fields[("unseen", p.name)]
            assert np.allclose(zu["rx%d" % i], rx)
            v = residual(zu["P%d" % i], keep, rx)[0]
            res["unseen"].append(v)
            res["unseen_" + fam].append(v)
        summ = {k: float(np.mean(v)) for k, v in res.items()}
        summ["ratio_unseen_over_test"] = summ["unseen"] / summ["test"]
        rep["models"][name] = {"per_room": res, "mean": summ}
        print(name, {k: round(v, 3) for k, v in summ.items()}, flush=True)
    fd = float(np.mean([r["rel"] for r in rep["fdtd"]]))
    ratios = [rep["models"][n]["mean"]["ratio_unseen_over_test"] for n in rep["models"]]
    rep["gate"] = {"i_fdtd_rel_mean": fd, "i_threshold": 0.10, "i_holds": bool(fd <= 0.10),
                   "ii_ratios": dict(zip(rep["models"], ratios)), "ii_threshold": 2.0,
                   "ii_holds": bool(all(r >= 2.0 for r in ratios)),
                   "launch_S1": bool(fd <= 0.10 and all(r >= 2.0 for r in ratios))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s1_gate.json").write_text(json.dumps(rep, indent=1))
    L = ["# P4-8 S1 gate -- finite-difference Helmholtz residual (D174(g))", "", rep["definition"], "",
         "| quantity | value |", "|---|---|",
         "| (i) FDTD relative residual, band-mean 25-300 Hz, mean over 31 rooms | {:.3f} (threshold 0.10) |".format(fd)]
    for n, d in rep["models"].items():
        mm = d["mean"]
        L.append("| (ii) {}: in-family (15 test shapes) / unseen (16) / U / C3 | {:.3f} / {:.3f} / {:.3f} / {:.3f}; "
                 "ratio {:.2f} (threshold 2) |".format(n, mm["test"], mm["unseen"], mm["unseen_U"], mm["unseen_C3"],
                                                       mm["ratio_unseen_over_test"]))
    L += ["", "**Gate:** " + json.dumps(rep["gate"])]
    (OUT / "s1_gate.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-4:]))


if __name__ == "__main__":
    main()
