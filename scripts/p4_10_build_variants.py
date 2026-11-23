"""P4-10 variant corpora (D195 0b, D197 A5/A6): STR (two stretched variants per GX room, 4 sources) and MAT (per-edge
absorption, same rooms / sources / receivers).

STR   variant 2i   = Stretch-U: every length x s, s uniform in [0.85, 1.15] excluding |s - 1| < 0.03; variant 2i + 1 =
      Stretch-A: a random axis (x-lengths or y-lengths), same range. Vertices scaled about the SW corner and quantised to
      0.02 m (A6); realised scales are stored. Each of the room's 4 MS sources keeps its absolute position if it stays
      >= 0.5 m from every wall, else it is redrawn (uniform over nodes with >= 0.5 m, >= 0.04 m from a receiver).
      Receivers: 800 nodes drawn uniformly >= 0.12 m from walls and > 0.15 m from reflex corners, >= 0.04 m from sources.
MAT   one draw per room: m = -ln(1 - alpha) uniform over alpha in [0.08, 0.60] per polygon EDGE (continuous on the
      physical axis); the room's 4 MS sources and 800 training receivers unchanged. Needs the `edges` spec (A5).
  --kind str|mat --manifest / --idx I --chunk N / --assemble
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from scripts.p4_10_build_ms import FS, MAN as MS_MAN, MIX, N, NB, NRX, rows, room, sim

ROOT = Path("/fs/nexus-scratch/htakawal/aaf_p410")
MAN = {"str": Path("configs/p4_10/str_manifest.json"), "mat": Path("configs/p4_10/mat_manifest.json")}
OUT = {k: ROOT / k for k in ("str", "mat")}
MM = {k: ROOT / (k + "_mm") for k in ("str", "mat")}
SEED = 20261125
CORNER_CLEAR, WALL_CLEAR, SRC_RX = 0.15, 0.12, 0.04


def qv(x):
    return round(round(float(x) / 0.02) * 0.02, 6)


def scale_verts(verts, sx, sy):
    V = [(qv(x * sx), qv(y * sy)) for x, y in verts]
    out = [V[0]]
    for v in V[1:]:                                              # drop edges that quantise to zero length
        if v != out[-1]:
            out.append(v)
    if out[0] == out[-1]:
        out.pop()
    return out


def check_ring(V):
    a2 = sum(V[k][0] * V[(k + 1) % len(V)][1] - V[(k + 1) % len(V)][0] * V[k][1] for k in range(len(V)))
    ok = a2 > 0 and len(V) >= 4
    for k in range(len(V)):
        (x0, y0), (x1, y1) = V[k], V[(k + 1) % len(V)]
        ok &= (abs(x1 - x0) < 1e-9) != (abs(y1 - y0) < 1e-9)       # exactly one of dx, dy is zero
    return ok


def str_rooms():
    """[(i, kind, verts, sx, sy)] for the 2C variants, deterministic"""
    R, M = rows(), json.loads(MS_MAN.read_text())["configs"]
    out = []
    for i, row in enumerate(R):
        rng = np.random.default_rng([SEED, i])
        V0 = [tuple(map(float, v)) for v in row["verts"]]

        def draw_s():
            while True:
                s = float(rng.uniform(0.85, 1.15))
                if abs(s - 1) >= 0.03:
                    return s
        s = draw_s()
        out.append((i, "U", scale_verts(V0, s, s), s, s))
        s = draw_s()
        if rng.integers(0, 2):
            out.append((i, "A-x", scale_verts(V0, s, 1.0), s, 1.0))
        else:
            out.append((i, "A-y", scale_verts(V0, 1.0, s), 1.0, s))
    return out, R, M


def mat_alphas(i, n_edges):
    rng = np.random.default_rng([SEED, 10 ** 6 + i])
    lo, hi = -np.log(1 - 0.08), -np.log(1 - 0.60)
    return [float(np.round(1 - np.exp(-m), 4)) for m in rng.uniform(lo, hi, n_edges)]


def geometry_of(verts, L, W, alphas):
    """(extra_walls, air mask positions, polygon verts) for a possibly non-0.15 polygon"""
    from aaf.sim import fdtd_2d as F
    from aaf.sim.polygon_geom import polygon_contains
    nx, dxx = F._fit_axis(L, 0.02, "L")
    ny, dxy = F._fit_axis(W, 0.02, "W")
    assert abs(dxx - 0.02) < 1e-12 and abs(dxy - 0.02) < 1e-12, (L, W)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    P = np.stack([ii * 0.02, jj * 0.02], -1)
    solid = ~polygon_contains(verts, P)
    extra = [{"type": "mask", "solid": solid, "alpha": 0.15}] if solid.any() else []
    if alphas is not None:
        extra.append({"type": "edges", "verts": [tuple(v) for v in verts], "alphas": alphas})
    F.build_geometry(L, W, [0.15] * 4, dx=0.02, extra_walls=extra)     # refuses unpainted faces
    return extra or None, P.reshape(-1, 2)[~solid.ravel()]


def str_points(verts, base_sources, seed_i):
    from aaf.data.cx_rooms import edge_distance, reflex_vertices
    from scipy.spatial import cKDTree
    L, W = max(v[0] for v in verts), max(v[1] for v in verts)
    extra, P = geometry_of(verts, L, W, None)
    rng = np.random.default_rng([SEED, 2, seed_i])
    d = edge_distance(verts, P)
    keep = d >= WALL_CLEAR
    for cx, cy in reflex_vertices(verts):
        keep &= np.hypot(P[:, 0] - cx, P[:, 1] - cy) > CORNER_CLEAR
    pool = P[keep]
    src = []
    for s in base_sources:
        if edge_distance(verts, np.array([s]))[0] >= 0.5 - 1e-6 and any(np.abs(P - s).max(1) < 1e-9):
            src.append(list(map(float, s)))
        else:
            src.append(None)
    p05 = P[d >= 0.5]
    for k in range(len(src)):
        if src[k] is None:
            src[k] = list(map(float, p05[rng.integers(0, len(p05))]))
    tree = cKDTree(np.array(src))
    ok = tree.query(pool)[0] >= SRC_RX
    rx = pool[ok][np.sort(rng.choice(int(ok.sum()), size=NRX, replace=False))]
    return np.round(rx, 6), np.round(np.array(src), 6), extra, L, W


def write_manifest(kind):
    MAN[kind].parent.mkdir(parents=True, exist_ok=True)
    R, MS = rows(), json.loads(MS_MAN.read_text())["configs"]
    out = []
    if kind == "mat":
        for i, row in enumerate(R):
            out.append({"i": i, "filename": row["filename"], "edge_alphas": mat_alphas(i, len(row["verts"])),
                        "verts": row["verts"], "L": max(v[0] for v in row["verts"]), "W": max(v[1] for v in row["verts"])})
    else:
        srooms, _, M = str_rooms()
        bad = 0
        for v, (i, k, V, sx, sy) in enumerate(srooms):
            if not check_ring(V):
                bad += 1
                V = [tuple(map(float, x)) for x in R[i]["verts"]]          # unscaled fallback (recorded)
                sx = sy = 1.0
                k = "identity"
            base = [M[i]["slot0"]] + M[i]["new_sources"]
            rx, src, _, L, W = str_points(V, base, v)
            out.append({"v": v, "i": i, "kind": k, "verts": [list(x) for x in V], "L": L, "W": W, "sx": sx, "sy": sy,
                        "sources": src.tolist(), "redrawn": int(sum(1 for a, b in zip(base, src) if not np.allclose(a, b))),
                        "edge_alphas": [0.15] * len(V), "n_rx": len(rx), "rx_hash": hashlib.sha1(rx.tobytes()).hexdigest()[:16]})
            if v % 200 == 0:
                print(v, R[i]["filename"], k, out[-1]["redrawn"], "sources redrawn", flush=True)
        print("invalid rings:", bad)
    blob = json.dumps(out, sort_keys=True).encode()
    MAN[kind].write_text(json.dumps({"kind": kind, "seed": SEED, "rows_sha256": hashlib.sha256(blob).hexdigest(),
                                     "rule": __doc__.split("\n\n")[1].strip(), "configs": out}, indent=0))
    print("wrote", MAN[kind], hashlib.sha256(blob).hexdigest()[:16], len(out))


def build(kind, v, M, MS, R):
    c = M[v]
    i = c["i"]
    dst = OUT[kind] / "{:05d}.npz".format(v)
    if dst.exists():
        return "skip"
    r = room(R[i])
    ms = MS[i]
    t0 = time.time()
    if kind == "mat":
        verts = [tuple(x) for x in c["verts"]]
        extra, _ = geometry_of(verts, c["L"], c["W"], c["edge_alphas"])
        rx, src = r["rx"], np.array([ms["slot0"]] + ms["new_sources"])
        rr = {"L": c["L"], "W": c["W"]}
    else:
        verts = [tuple(x) for x in c["verts"]]
        rx, src, extra, L, W = str_points(verts, [ms["slot0"]] + ms["new_sources"], v)
        assert hashlib.sha1(rx.tobytes()).hexdigest()[:16] == c["rx_hash"], "receiver draw changed"
        rr = {"L": c["L"], "W": c["W"]}
    H = np.stack([sim(rr, extra, s, rx) for s in src])
    OUT[kind].mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, H=H.astype(np.complex64), rx=rx, src=np.asarray(src), secs=time.time() - t0)
    tmp.rename(dst)
    return "{:.0f}s".format(time.time() - t0)


def assemble(kind):
    M = json.loads(MAN[kind].read_text())["configs"]
    R = rows()
    n = len(M)
    S = 4
    MM[kind].mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(MM[kind] / "H.npy", mode="w+", dtype=np.complex64, shape=(n, S, NRX, NB))
    rx, src = np.zeros((n, NRX, 2)), np.zeros((n, S, 2))
    for v in range(n):
        z = np.load(OUT[kind] / "{:05d}.npz".format(v))
        mm[v] = z["H"]
        rx[v], src[v] = z["rx"], z["src"]
        if v % 200 == 0:
            mm.flush()
            print(v, flush=True)
    mm.flush()
    np.save(MM[kind] / "rx.npy", rx)
    np.save(MM[kind] / "src.npy", src)
    (MM[kind] / "index.json").write_text(json.dumps({"kind": kind, "base_filenames": [x["filename"] for x in R], "n_variants": n,
                                                     "shape": [n, S, NRX, NB], "manifest_sha": json.loads(MAN[kind].read_text())["rows_sha256"],
                                                     "variants": [{"L": c["L"], "W": c["W"], "verts": c["verts"], "edge_alphas": c["edge_alphas"]} for c in M]}))
    print("assembled", mm.shape, round(mm.nbytes / 1e9, 1), "GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["str", "mat"], required=True)
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=4)
    a = ap.parse_args()
    if a.manifest:
        write_manifest(a.kind)
    elif a.assemble:
        assemble(a.kind)
    elif a.idx is not None:
        M, MS, R = json.loads(MAN[a.kind].read_text())["configs"], json.loads(MS_MAN.read_text())["configs"], rows()
        for v in range(a.idx * a.chunk, min((a.idx + 1) * a.chunk, len(M))):
            print(v, build(a.kind, v, M, MS, R), flush=True)


if __name__ == "__main__":
    main()
