"""P4-9 (D180 0b / 0c / 2b; D181 A9, A11, E2): FDTD reruns that record extra nodes, for gradient and wall-stencil truth.

Every P4 room was simulated by the same call -- `aaf.sim.fdtd_2d.simulate(L, W, [0.15]*4, src, rx, dx=0.02,
fs=30720, n=61440, c=343, extra_walls=[polygon solid mask])`, each builder asserting the solver's air mask equals
`polygon_contains(verts, nodes)` node for node. So one generic rerun from the room's own h5 attributes reproduces it,
whatever builder made it. PROOF, not assumption: every chunk also records 50 of the stored receivers and must
reproduce the stored H to <= 1e-6 (relative to their max |H|) before anything is kept. Receivers sit on 2 cm nodes
(no interpolation), so stencil nodes are exact node offsets. At most 5000 recorded nodes per run (~2 MB each).

  --set dense --idx I     one of the 31 dense rooms (15 in-family test shapes, then the 16 unseen rooms, in the
                          S1-gate order). Records, per dense receiver, the +-1 and +-2 node neighbours along x and y
                          and stores central-difference gradients g1 (h = 1 node, ~2 cm) and g2 (h = 2 nodes, ~4 cm);
                          and WALL STENCILS for the Robin residual: points every 5 cm along each polygon edge, more than
                          0.15 m from its ends (the vertices) and >= 0.3 m from the source, with the field at depths 0,
                          1, 2 nodes along the inward normal (the depth-0 node is the wall's boundary node).
  --set gx --idx I [--chunk 8]   GX training rooms (configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json, manifest
                          order) I*chunk ...: g1 and g2 at each room's 800 stored receivers (labels for DIR / DIR-G).
  --assemble-gx           stack the per-room GX files into memory-mappable arrays data/track_p4_9_gx_grad/
                          {g1x,g1y,g2x,g2y}.npy, [1000, 800, 601] complex64, + index.json (labels, manifest sha).
Outputs (gitignored .npz / .npy): outputs/p4_9/fdtd_stencils/{dense,gx}/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.sim.polygon_geom import polygon_contains

ALPHA, DX, C, NB = 0.15, 0.02, 343.0, 601
MAX_NODES = 5000
N_CHECK = 50
OUT = Path("outputs/p4_9/fdtd_stencils")
GX_MAN = "configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"
GX_DIR = Path("data/track_p4_7_mix")
GX_GRAD = Path("data/track_p4_9_gx_grad")
WALL_STEP, WALL_END, SRC_CLEAR = 0.05, 0.15, 0.30


def dense_rooms():
    from scripts.p4_8_s1_gate import rooms_test, rooms_unseen
    return [("test", fam, p) for fam, p in rooms_test()] + [("unseen", fam, p) for fam, p in rooms_unseen()]


def load_room(path):
    with h5py.File(path) as f:
        a = f.attrs
        r = {"H": np.asarray(f["ism/H_complex"]), "rx": np.asarray(json.loads(a["receiver_pos"]), float),
             "src": tuple(float(x) for x in json.loads(a["source_pos"])),
             "verts": [tuple(map(float, v)) for v in json.loads(a["verts"])],
             # the famdense / unseen builders did not write these; every P4 corpus used dx 0.02, fs 30720, n 61440
             # (build_p4_2_shapes.FS_FOR_DX, T = 2 s). The exactness check would catch any other value.
             "dx_x": float(a["dx_x"]) if "dx_x" in a else None, "dx_y": float(a["dx_y"]) if "dx_y" in a else None,
             "fs": float(a["fs"]) if "fs" in a else 30720.0, "n": int(a["n_time_samples"]) if "n_time_samples" in a else 61440,
             "L": float(a["L"]), "W": float(a["W"])}
    assert abs(float(a_dx(path)) - DX) < 1e-12
    assert abs(r["L"] - max(v[0] for v in r["verts"])) < 1e-9 and abs(r["W"] - max(v[1] for v in r["verts"])) < 1e-9
    return r


def a_dx(path):
    with h5py.File(path) as f:
        return f.attrs["dx"]


def geometry(r):
    nx, dxx = F._fit_axis(r["L"], DX, "L")
    ny, dxy = F._fit_axis(r["W"], DX, "W")
    if r["dx_x"] is not None:
        assert abs(dxx - r["dx_x"]) < 1e-12 and abs(dxy - r["dx_y"]) < 1e-12, (dxx, r["dx_x"], dxy, r["dx_y"])
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    inside = polygon_contains(r["verts"], np.stack([ii * dxx, jj * dxy], axis=-1))
    solid = ~inside
    extra = [{"type": "mask", "solid": solid, "alpha": ALPHA}] if solid.any() else None
    geom = F.build_geometry(r["L"], r["W"], [ALPHA] * 4, dx=DX, extra_walls=extra)
    bad = int((geom.air != inside).sum())
    if bad:
        raise AssertionError("air mask disagrees with the polygon at {} nodes".format(bad))
    return geom, extra, dxx, dxy


def simulate(r, extra, nodes_ij, dxx, dxy):
    """Record nodes_ij [N, 2] (int) in chunks, each chunk also recording N_CHECK stored receivers for the exactness
    check. Returns H [N, 601] complex64 and the worst check error."""
    stored_ij = np.rint(r["rx"] / [dxx, dxy]).astype(int)
    assert np.abs(stored_ij * [dxx, dxy] - r["rx"]).max() < 1e-9, "stored receivers are not on nodes"
    chk = np.linspace(0, len(stored_ij) - 1, N_CHECK).round().astype(int)
    H = np.zeros((len(nodes_ij), NB), np.complex64)
    worst = 0.0
    per = MAX_NODES - N_CHECK
    for s in range(0, len(nodes_ij), per):
        ij = np.concatenate([stored_ij[chk], nodes_ij[s:s + per]])
        P = ij * np.array([dxx, dxy])
        t0 = time.time()
        res = F.simulate(r["L"], r["W"], [ALPHA] * 4, r["src"], P, dx=DX, fs=r["fs"], n=r["n"], c=C,
                         extra_walls=extra)
        h = np.asarray(res["H_complex"])[:, :NB].astype(np.complex64)
        snap = np.asarray(res["meta"]["rx_pos_snapped"])
        assert np.abs(snap - P).max() < 1e-9
        T = r["H"][chk]
        err = float(np.abs(h[:N_CHECK] - T).max() / np.abs(T).max())
        if err > 1e-6:
            raise AssertionError("rerun does NOT reproduce the stored H: max rel err {:.3e}".format(err))
        worst = max(worst, err)
        H[s:s + per] = h[N_CHECK:]
        print("  chunk {:6d}-{:6d}: {} nodes, check err {:.2e}, {:.0f}s".format(
            s, s + len(ij) - N_CHECK, len(ij), err, time.time() - t0), flush=True)
        del res, h, snap          # otherwise this chunk's ~10 GB of impulse responses stay alive through the next
        gc.collect()
    return H, worst


def neighbour_ij(stored_ij):
    """+-1 and +-2 node offsets along x and y: [8 n, 2], order (dx-1, dx+1, dy-1, dy+1, dx-2, dx+2, dy-2, dy+2)."""
    offs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-2, 0], [2, 0], [0, -2], [0, 2]])
    return (stored_ij[:, None, :] + offs[None]).reshape(-1, 2)


def gradients(Hn, n, dxx, dxy):
    """Central differences from neighbour spectra Hn [8 n, 601] -> g1x, g1y (h = 1 node), g2x, g2y (h = 2 nodes)."""
    X = Hn.reshape(n, 8, NB).astype(np.complex128)
    g1x = (X[:, 1] - X[:, 0]) / (2 * dxx)
    g1y = (X[:, 3] - X[:, 2]) / (2 * dxy)
    g2x = (X[:, 5] - X[:, 4]) / (4 * dxx)
    g2y = (X[:, 7] - X[:, 6]) / (4 * dxy)
    return [g.astype(np.complex64) for g in (g1x, g1y, g2x, g2y)]


def wall_stencils(r, geom, dxx, dxy):
    """Per polygon edge (counter-clockwise, so the interior is on the LEFT): points every 5 cm, > 0.15 m from both
    ends and >= 0.3 m from the source; depth-0/1/2 node indices along the inward normal."""
    V = r["verts"]
    area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(V, V[1:] + V[:1]))
    assert area2 > 0, "verts must be counter-clockwise"
    pts, meta = [], []
    for e, ((x0, y0), (x1, y1)) in enumerate(zip(V, V[1:] + V[:1])):
        d = np.array([x1 - x0, y1 - y0])
        ln = float(np.hypot(*d))
        d = d / ln
        assert abs(d[0]) < 1e-9 or abs(d[1]) < 1e-9, "edges must be axis-aligned"
        nrm = np.array([-d[1], d[0]]).round().astype(int)          # inward (left) unit normal, in node steps
        for s in np.arange(WALL_END + 1e-9, ln - WALL_END - 1e-9, WALL_STEP):
            p = np.array([x0, y0]) + s * d
            if np.hypot(p[0] - r["src"][0], p[1] - r["src"][1]) < SRC_CLEAR:
                continue
            ij0 = np.rint(p / [dxx, dxy]).astype(int)
            ijs = np.stack([ij0 + m * nrm for m in range(3)])
            if not all(geom.air[i, j] for i, j in ijs):
                raise AssertionError("wall stencil off the air mask at edge {} s {:.2f}".format(e, s))
            pts.append(ijs)
            meta.append((e, float(s), int(nrm[0]), int(nrm[1])))   # edge id, arc position (m), inward normal
    return np.stack(pts), np.asarray(meta, dtype=float)


def run_dense(idx):
    group, fam, path = dense_rooms()[idx]
    r = load_room(path)
    geom, extra, dxx, dxy = geometry(r)
    sij = np.rint(r["rx"] / [dxx, dxy]).astype(int)
    nb = neighbour_ij(sij)
    if not geom.air[nb[:, 0], nb[:, 1]].all():
        raise AssertionError("a receiver neighbour is not an air node")
    wij, wmeta = wall_stencils(r, geom, dxx, dxy)
    print("dense {:2d} {:6s} {:3s} {}: {} receivers -> {} neighbour nodes; {} wall points x 3".format(
        idx, group, fam, path.name, len(sij), len(nb), len(wij)), flush=True)
    nodes = np.concatenate([nb, wij.reshape(-1, 2)])
    H, worst = simulate(r, extra, nodes, dxx, dxy)
    g1x, g1y, g2x, g2y = gradients(H[:len(nb)], len(sij), dxx, dxy)
    Hw = H[len(nb):].reshape(len(wij), 3, NB)
    out = OUT / "dense"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / (path.stem + ".npz"), g1x=g1x, g1y=g1y, g2x=g2x, g2y=g2y, rx=r["rx"],
                        wall_ij=wij, wall_meta=wmeta, wall_H=Hw, dxx=dxx, dxy=dxy,
                        src=np.asarray(r["src"]), check_err=worst, group=group, family=fam)
    print("wrote {} (check err {:.2e})".format(out / (path.stem + ".npz"), worst), flush=True)


def run_gx(idx, chunk):
    man = json.load(open(GX_MAN))
    rows = man["configs"]
    out = OUT / "gx"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(idx * chunk, min(len(rows), (idx + 1) * chunk)):
        fn = rows[k]["filename"]
        dst = out / "{:04d}.npz".format(k)
        if dst.exists():
            print("skip", k, fn)
            continue
        r = load_room(GX_DIR / fn)
        geom, extra, dxx, dxy = geometry(r)
        sij = np.rint(r["rx"] / [dxx, dxy]).astype(int)
        nb = neighbour_ij(sij)
        if not geom.air[nb[:, 0], nb[:, 1]].all():
            raise AssertionError("{}: a receiver neighbour is not an air node".format(fn))
        print("gx {:4d} {}: {} receivers -> {} nodes".format(k, fn, len(sij), len(nb)), flush=True)
        H, worst = simulate(r, extra, nb, dxx, dxy)
        g1x, g1y, g2x, g2y = gradients(H, len(sij), dxx, dxy)
        tmp = out / "{:04d}.tmp.npz".format(k)
        np.savez(tmp, g1x=g1x, g1y=g1y, g2x=g2x, g2y=g2y, rx=r["rx"], src=np.asarray(r["src"]), filename=fn,
                 check_err=worst, dxx=dxx, dxy=dxy)
        tmp.replace(dst)


def assemble_gx():
    man = json.load(open(GX_MAN))
    rows = man["configs"]
    GX_GRAD.mkdir(parents=True, exist_ok=True)
    shape = (len(rows), 800, NB)
    arrs = {k: np.lib.format.open_memmap(GX_GRAD / (k + ".npy"), mode="w+", dtype=np.complex64, shape=shape)
            for k in ("g1x", "g1y", "g2x", "g2y")}
    worst, labels, dxs = 0.0, [], []
    rx_all = np.zeros((len(rows), 800, 2), np.float64)
    for k, row in enumerate(rows):
        z = np.load(OUT / "gx" / "{:04d}.npz".format(k))
        assert str(z["filename"]) == row["filename"]
        for key, a in arrs.items():
            a[k] = z[key]
        worst = max(worst, float(z["check_err"]))
        labels.append(row["filename"])
        dxs.append([float(z["dxx"]), float(z["dxy"])])
        rx_all[k] = z["rx"]
    for a in arrs.values():
        a.flush()
    np.save(GX_GRAD / "rx.npy", rx_all)
    idx = {"manifest": GX_MAN, "rows_sha256": man.get("rows_sha256"),
           "manifest_file_sha256": hashlib.sha256(Path(GX_MAN).read_bytes()).hexdigest(),
           "filenames": labels, "dx_xy": dxs, "shape": list(shape), "worst_check_err": worst,
           "definition": "central differences of FDTD spectra at each stored receiver: g1 = +-1 node (h = dx_x / dx_y "
                         "~ 2 cm), g2 = +-2 nodes (~4 cm); complex64, bins 0-600 (0-300 Hz)"}
    (GX_GRAD / "index.json").write_text(json.dumps(idx, indent=1))
    print("assembled", shape, "worst check err {:.2e}".format(worst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("dense", "gx"))
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--assemble-gx", action="store_true")
    a = ap.parse_args()
    if a.assemble_gx:
        assemble_gx()
    elif a.set == "dense":
        run_dense(a.idx)
    elif a.set == "gx":
        run_gx(a.idx, a.chunk)


if __name__ == "__main__":
    main()
