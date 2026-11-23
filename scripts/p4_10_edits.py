"""P4-10 Step 0a: the edit manifest (D195, D197 A6) -- every edit cell of the scorecard, frozen before any score.

Rooms: the 15 in-family test shapes (primary) and the 16 unseen rooms. Geometry edits keep the SW corner at the origin and
the source at (0.5, 1.6); every edited dimension is quantised to 0.02 m (A6) and the REALISED scale is stored.
  StretchU   L, W, notch d, w x s        s = 0.90 0.95 1.05 1.10
  StretchA   x: L and notch w x s ; y: W and notch d x s      x/y 0.90 1.10
  Wall       L +/- 0.3 (east wall) ; W +/- 0.3 (north wall); notches keep their parameters
  Notch      the largest notch (d x w): depth +/- 0.3, width +/- 0.3
  Source     second (the existing set's) and third source; geometry unchanged
  Material   longest wall alpha 0.30 / 0.50 ; all walls alpha 0.10 / 0.30 (needs per-edge absorption, A5)
Skips (logged): no notch (Notch); a notch dimension that FALLS below 0.2 m; local width < 1.0 m; source strictly < 0.5 m from a wall.
Third source: the FDTD node nearest (0.55 L, 0.5 W) with >= 1.0 m wall clearance and >= 1.5 m from the first and second
sources (checked feasible in all 31 rooms).
  PYTHONPATH=$PWD python scripts/p4_10_edits.py    -> configs/p4_10/edit_manifest.json (rows_sha256) + skips
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from aaf.data.cx_rooms import edge_distance, q, rect_gap
from aaf.sim.polygon_geom import polygon_contains

OUT = Path("configs/p4_10/edit_manifest.json")
SRC = (0.5, 1.6)
LITE = {"StretchU": ("0.90", "1.10"), "StretchA": ("x1.10", "y0.90"), "Wall": ("L+0.3", "W-0.3"),
        "Notch": ("d+0.3", "w-0.3"), "Source": ("second", "third"), "Material": ("long0.50", "all0.30")}


def rect_of(n, L, W):
    s, d, w = n["side"], n["d"], n["w"]
    return {"NW": (0.0, w, W - d, W), "NE": (L - w, L, W - d, W), "SE": (L - w, L, 0.0, d), "SW": (0.0, w, 0.0, d)}[s]


def verts_of(L, W, notches):
    """the ring exactly as the frozen sets build it: `verts_for` for one NW notch, `verts_for_notches` otherwise."""
    from aaf.data.multi_notch import Notch, verts_for_notches
    from aaf.data.shape_configs import verts_for
    if len(notches) == 1 and notches[0]["side"] == "NW":
        return [tuple(map(float, v)) for v in verts_for(L, W, notches[0]["d"], notches[0]["w"])]
    return [tuple(map(float, v)) for v in verts_for_notches(L, W, [Notch(side=n["side"], d=n["d"], w=n["w"], x0=n.get("x0", 0.0))
                                                                     for n in notches])]


def base_rooms():
    out = []
    t = json.load(open("configs/sweeps_2d_mat/p4_2_shapes_manifest.json"))["configs"]
    for r in [x for x in t if x["split"] == "test"]:
        out.append({"group": "test", "family": "L", "filename": r["filename"], "L": r["L"], "W": r["W"],
                    "notches": [{"side": "NW", "d": r["d"], "w": r["w"], "x0": 0.0}], "verts": r["verts"],
                    "src2_dir": "data/track_p4_9_test_src2", "data": "data/track_p4_5_testdense"})
    u = json.load(open("configs/sweeps_2d_mat/p4_8_unseen_manifest.json"))["configs"]
    for r in u:
        out.append({"group": "unseen", "family": r["family"], "filename": r["filename"], "L": r["L"], "W": r["W"],
                    "notches": [dict(side=n["side"], d=n["d"], w=n["w"], x0=n.get("x0", 0.0)) for n in r["notches"]],
                    "verts": r["verts"], "src2_dir": "data/track_p4_9_unseen_src2", "data": "data/track_p4_8_unseen"})
    for r in out:                                       # our ring must be the frozen ring, vertex for vertex
        got = verts_of(r["L"], r["W"], r["notches"])
        want = [tuple(map(float, v)) for v in r["verts"]]
        assert len(got) == len(want) and np.allclose(got, want), (r["filename"], got, want)
    return out


def valid(L, W, notches, src, check_src=True, orig=None):
    """None, or the reason an edited room is skipped. A notch dimension must not FALL below 0.2 m (D195's word): three
    unseen rooms start with one under 0.2 m (d 0.14 / 0.08 / 0.06), and an edit that does not shrink it is valid."""
    for k, n in enumerate(notches):
        for dim in ("d", "w"):
            if n[dim] < 0.2 - 1e-9 and (orig is None or n[dim] < orig[k][dim] - 1e-9):
                return "notch dimension falls below 0.2 m"
    if any(W - n["d"] < 1.0 - 1e-9 or L - n["w"] < 1.0 - 1e-9 for n in notches):
        return "room narrows below 1.0 m"
    rs = [rect_of(n, L, W) for n in notches]
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            if rect_gap(rs[i], rs[j]) < 1.0 - 1e-9:
                return "gap between notches < 1.0 m"
    try:
        V = verts_of(L, W, notches)
    except Exception as e:
        return "invalid ring: {}".format(e)
    if not bool(polygon_contains(V, np.array([src]))[0]):
        return "source outside the room"
    if check_src and edge_distance(V, np.array([src]))[0] < 0.5 - 1e-6:
        return "source < 0.5 m from a wall"
    return None


def third_source(V, L, W, s2):
    xs, ys = np.arange(int(round(L / 0.02)) + 1) * 0.02, np.arange(int(round(W / 0.02)) + 1) * 0.02
    P = np.stack([a.ravel() for a in np.meshgrid(xs, ys, indexing="ij")], 1)
    P = P[polygon_contains(V, P)]
    ok = (edge_distance(V, P) >= 1.0) & (np.hypot(P[:, 0] - SRC[0], P[:, 1] - SRC[1]) >= 1.5) \
        & (np.hypot(P[:, 0] - s2[0], P[:, 1] - s2[1]) >= 1.5)
    c = P[ok]
    return None if not len(c) else tuple(float(x) for x in np.round(c[np.argmin(np.hypot(c[:, 0] - 0.55 * L,
                                                                                        c[:, 1] - 0.5 * W))], 6))


def cells():
    import h5py
    out, skips = [], []
    for r in base_rooms():
        L, W, N, stem = r["L"], r["W"], r["notches"], Path(r["filename"]).stem
        with h5py.File(Path(r["src2_dir"]) / r["filename"], "r") as f:
            s2 = tuple(float(x) for x in json.loads(f.attrs["source_pos"]))
        V0 = verts_of(L, W, N)
        base = {"group": r["group"], "room_family": r["family"], "room": stem, "filename": r["filename"],
                "data": r["data"]}

        def add(fam, mag, Lx, Wx, Nx, src=SRC, alpha_edges=None, extra=None, check_src=True, Vx=None):
            why = None if fam in ("Source", "Material") else valid(Lx, Wx, Nx, src, check_src, N)
            cell = dict(base, family=fam, mag=mag, id="{}|{}|{}".format(stem, fam, mag))
            if why:
                skips.append({"cell": cell["id"], "why": why})
                return
            V = Vx if Vx is not None else verts_of(Lx, Wx, Nx)
            cell.update(L=round(Lx, 6), W=round(Wx, 6), notches=Nx, verts=[list(v) for v in V], src=list(src),
                        src_orig=list(SRC), lite=mag in LITE.get(fam, ()),
                        edge_alphas=alpha_edges or [0.15] * len(V), orig_L=L, orig_W=W, orig_notches=N,
                        orig_verts=[list(v) for v in V0])
            cell.update(extra or {})
            out.append(cell)

        def scaled(Nx, sx, sy):
            return [dict(n, w=q(n["w"] * sx), d=q(n["d"] * sy)) for n in Nx]
        for s in (0.90, 0.95, 1.05, 1.10):
            Lx, Wx, Nx = q(L * s), q(W * s), scaled(N, s, s)
            add("StretchU", "{:.2f}".format(s), Lx, Wx, Nx, extra={"sx": Lx / L, "sy": Wx / W})
        for ax, s in (("x", 0.90), ("x", 1.10), ("y", 0.90), ("y", 1.10)):
            sx, sy = (s, 1.0) if ax == "x" else (1.0, s)
            Lx, Wx = q(L * sx), q(W * sy)
            Nx = [dict(n, w=q(n["w"] * sx), d=q(n["d"] * sy)) for n in N]
            add("StretchA", "{}{:.2f}".format(ax, s), Lx, Wx, Nx, extra={"sx": Lx / L, "sy": Wx / W})
        for nm, dL, dW in (("L+0.3", 0.3, 0), ("L-0.3", -0.3, 0), ("W+0.3", 0, 0.3), ("W-0.3", 0, -0.3)):
            add("Wall", nm, q(L + dL), q(W + dW), [dict(n) for n in N])
        if not N:
            for nm in ("d+0.3", "d-0.3", "w+0.3", "w-0.3"):
                skips.append({"cell": "{}|Notch|{}".format(stem, nm), "why": "no notch"})
        else:
            k = int(np.argmax([n["d"] * n["w"] for n in N]))
            for nm, dd, dw in (("d+0.3", 0.3, 0), ("d-0.3", -0.3, 0), ("w+0.3", 0, 0.3), ("w-0.3", 0, -0.3)):
                Nx = [dict(n) for n in N]
                Nx[k] = dict(Nx[k], d=q(Nx[k]["d"] + dd), w=q(Nx[k]["w"] + dw))
                add("Notch", nm, L, W, Nx, extra={"notch": N[k]["side"]})
        add("Source", "second", L, W, N, src=s2, check_src=False, Vx=V0)
        s3 = third_source(V0, L, W, s2)
        if s3 is None:
            skips.append({"cell": stem + "|Source|third", "why": "no third source"})
        else:
            add("Source", "third", L, W, N, src=s3, check_src=False, Vx=V0)
        lens = [float(np.hypot(V0[(i + 1) % len(V0)][0] - V0[i][0], V0[(i + 1) % len(V0)][1] - V0[i][1]))
                for i in range(len(V0))]
        longest = int(np.argmax(lens))
        for a in (0.30, 0.50):
            e = [0.15] * len(V0)
            e[longest] = a
            add("Material", "long{:.2f}".format(a), L, W, N, alpha_edges=e, Vx=V0, extra={"edge": longest})
        for a in (0.10, 0.30):
            add("Material", "all{:.2f}".format(a), L, W, N, alpha_edges=[a] * len(V0), Vx=V0)
    return out, skips


def main():
    C, skips = cells()
    blob = json.dumps(C, sort_keys=True).encode()
    sha = hashlib.sha256(blob).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"schema": "p4_10.edits/1", "rows_sha256": sha, "n_cells": len(C), "skips": skips,
                               "cells": C}, indent=0))
    from collections import Counter
    c = Counter((x["group"], x["family"]) for x in C)
    print("cells", len(C), "skips", len(skips), "sha", sha[:16])
    for g in ("test", "unseen"):
        print(g, {f: c[(g, f)] for f in LITE})
    print(Counter(s["why"] for s in skips))


if __name__ == "__main__":
    main()
