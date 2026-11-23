"""P4-10 MS corpus (D195 0b, D197 A1 / A2): four sources per GX room.

Slot 0 = the room's own existing source (its stored labels; nothing is re-simulated). Slots 1-3 = drawn uniformly from the
room's FDTD air nodes with >= 0.5 m clearance from every wall (seed [20261124, room index]); if none of the three has
>= 1.0 m, the last is redrawn from the >= 1.0 m pool. A candidate within 0.04 m of a training receiver is rejected.
Receivers = the room's 800 training receivers. One FDTD run per new source.
  --manifest          freeze the draws: configs/sweeps_2d_mat/p4_10_ms_manifest.json (rows_sha256)
  --idx I --chunk N   simulate rooms I*N .. I*N+N-1 -> /fs/nexus-scratch/htakawal/aaf_p410/ms/<room index>.npz
  --smoke             3 rooms (C500 rect, C500 notch, CX500): every slot re-simulated at 50 receivers against the
                      stored / built H, error normalised by the BAND's own norm (bins >= 50), never by max |H|
  --assemble          memmap [C, 4, 800, 601] complex64 + sources.npy + index.json in .../ms_mm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

MIX = Path("configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json")
DATA = Path("data/track_p4_7_mix")
MAN = Path("configs/sweeps_2d_mat/p4_10_ms_manifest.json")
OUT = Path("/fs/nexus-scratch/htakawal/aaf_p410/ms")
MM = Path("/fs/nexus-scratch/htakawal/aaf_p410/ms_mm")
SEED, DX, FS, N, C, ALPHA, NB, NRX = 20261124, 0.02, 30720.0, 61440, 343.0, 0.15, 601, 800


def rows():
    return json.loads(MIX.read_text())["configs"]


def room(row):
    """(H, rx, src, verts, L, W) of one stored GX room; asserts the file is the manifest row's room."""
    with h5py.File(DATA / row["filename"], "r") as f:
        a = f.attrs
        out = {"H": np.asarray(f["ism/H_complex"]), "rx": np.asarray(json.loads(a["receiver_pos"]), float),
               "src": np.asarray(json.loads(a["source_pos"]), float), "L": float(a["L"]), "W": float(a["W"]),
               "verts": [tuple(map(float, v)) for v in json.loads(a["verts"])]}
    assert out["verts"] == [tuple(map(float, v)) for v in row["verts"]], row["filename"]
    assert out["H"].shape == (NRX, NB) and out["rx"].shape == (NRX, 2), row["filename"]
    return out


def geometry(r):
    from aaf.sim import fdtd_2d as F
    from aaf.sim.polygon_geom import polygon_contains
    nx, dxx = F._fit_axis(r["L"], DX, "L")
    ny, dxy = F._fit_axis(r["W"], DX, "W")
    assert abs(dxx - DX) < 1e-12 and abs(dxy - DX) < 1e-12
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    P = np.stack([ii * dxx, jj * dxy], axis=-1)
    inside = polygon_contains(r["verts"], P)
    solid = ~inside
    extra = [{"type": "mask", "solid": solid, "alpha": ALPHA}] if solid.any() else None
    geom = F.build_geometry(r["L"], r["W"], [ALPHA] * 4, dx=DX, extra_walls=extra)
    bad = int((geom.air != inside).sum())
    assert bad == 0, bad
    return geom, extra, P.reshape(-1, 2)


def draw(i, r):
    from aaf.data.cx_rooms import edge_distance
    geom, _, Pf = geometry(r)
    air = geom.air.ravel()
    P = Pf[air]
    d = edge_distance(r["verts"], P)
    from scipy.spatial import cKDTree
    rxd = cKDTree(r["rx"]).query(P)[0]
    ok = rxd >= 0.04
    p05, p10 = P[(d >= 0.5) & ok], P[(d >= 1.0) & ok]
    rng = np.random.default_rng([SEED, i])
    s = p05[rng.choice(len(p05), size=3, replace=False)]
    clear = edge_distance(r["verts"], s)
    redrawn = False
    if clear.max() < 1.0:
        if len(p10):
            s[2] = p10[rng.integers(0, len(p10))]
            redrawn = True
        else:
            redrawn = "none >= 1.0 m"          # the room is too narrow (recorded; its three sources use the 0.5 m rule)
    return np.round(s, 6), redrawn, len(p05), len(p10)


def write_manifest():
    out, n_red, n_none = [], 0, 0
    for i, row in enumerate(rows()):
        r = room(row)
        s, red, a, b = draw(i, r)
        n_red += red is True
        n_none += red == "none >= 1.0 m"
        out.append({"i": i, "filename": row["filename"], "origin": row["origin"], "slot0": [float(x) for x in r["src"]],
                    "new_sources": s.tolist(), "pool_05": a, "pool_10": b,
                    "no_1m_node": red == "none >= 1.0 m"})
        if i % 200 == 0:
            print(i, row["filename"], s.tolist(), flush=True)
    blob = json.dumps(out, sort_keys=True).encode()
    MAN.write_text(json.dumps({"schema": "p4_10.ms/1", "seed": SEED, "rule": __doc__.split("\n\n")[1].strip(),
                               "rows_sha256": hashlib.sha256(blob).hexdigest(), "n_redrawn_1m": n_red,
                               "n_rooms_without_1m_node": n_none, "configs": out},
                              indent=0))
    print("wrote", MAN, hashlib.sha256(blob).hexdigest()[:16], "redrawn for 1.0 m:", n_red, "rooms without a 1.0 m node:", n_none)


def sim(r, extra, src, rx):
    from aaf.sim import fdtd_2d as F
    res = F.simulate(r["L"], r["W"], [ALPHA] * 4, tuple(src), rx, dx=DX, fs=FS, n=N, c=C, extra_walls=extra)
    assert np.abs(np.asarray(res["meta"]["rx_pos_snapped"]) - rx).max() < 1e-9
    assert np.abs(np.asarray(res["meta"]["src_pos_snapped"]) - np.asarray(src)).max() < 1e-9
    return np.asarray(res["H_complex"])[:, :NB].astype(np.complex64)


def rel_band(a, b):
    """relative L2 over 25-300 Hz (bins 50-600), normalised by the truth's own band norm."""
    return float(np.linalg.norm((a - b)[:, 50:]) / np.linalg.norm(b[:, 50:]))


def build(i, row, man):
    dst = OUT / "{:04d}.npz".format(i)
    if dst.exists():
        return "skip"
    r = room(row)
    geom, extra, _ = geometry(r)
    t0 = time.time()
    H = np.stack([sim(r, extra, s, r["rx"]) for s in man["new_sources"]])
    chk = float("nan")
    if i % 25 == 0:                                   # geometry proof: the original source must reproduce the stored H
        sel = np.linspace(0, NRX - 1, 50).round().astype(int)
        chk = rel_band(sim(r, extra, r["src"], r["rx"][sel]), r["H"][sel])
        assert chk < 1e-4, (i, chk)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, H=H, rx=r["rx"], src=np.array(man["new_sources"]), slot0=r["src"], check_rel_band=chk,
             secs=time.time() - t0)
    tmp.rename(dst)
    return "built {:.0f}s check {:.1e}".format(time.time() - t0, chk)


def smoke():
    R, M = rows(), json.loads(MAN.read_text())["configs"]
    pick = [next(i for i, x in enumerate(R) if x["origin"] == "C500" and x["kind"] == "rect"),
            next(i for i, x in enumerate(R) if x["origin"] == "C500" and x["kind"] != "rect"),
            next(i for i, x in enumerate(R) if x["origin"] == "CX500")]
    for i in pick:
        r = room(R[i])
        geom, extra, _ = geometry(r)
        z = np.load(OUT / "{:04d}.npz".format(i))
        assert np.allclose(z["rx"], r["rx"]) and np.allclose(z["src"], M[i]["new_sources"])
        sel = np.linspace(0, NRX - 1, 50).round().astype(int)
        print(i, R[i]["filename"], "slot 0 rel", rel_band(sim(r, extra, r["src"], r["rx"][sel]), r["H"][sel]), flush=True)
        for k in range(3):
            f = sim(r, extra, M[i]["new_sources"][k], r["rx"][sel])
            print("   slot", k + 1, "rel", rel_band(f, z["H"][k][sel]), "vs slot 0 rel", rel_band(f, r["H"][sel]), flush=True)


def assemble():
    R, M = rows(), json.loads(MAN.read_text())
    MM.mkdir(parents=True, exist_ok=True)
    C_ = len(R)
    mm = np.lib.format.open_memmap(MM / "H.npy", mode="w+", dtype=np.complex64, shape=(C_, 4, NRX, NB))
    src, rx = np.zeros((C_, 4, 2)), np.zeros((C_, NRX, 2))
    for i, row in enumerate(R):
        r = room(row)
        z = np.load(OUT / "{:04d}.npz".format(i))
        assert np.allclose(z["rx"], r["rx"])
        mm[i, 0] = r["H"]
        mm[i, 1:] = z["H"]
        src[i, 0], src[i, 1:] = r["src"], z["src"]
        rx[i] = r["rx"]
        if i % 100 == 0:
            mm.flush()
            print(i, flush=True)
    mm.flush()
    np.save(MM / "src.npy", src)
    np.save(MM / "rx.npy", rx)
    (MM / "index.json").write_text(json.dumps({"filenames": [x["filename"] for x in R], "shape": [C_, 4, NRX, NB],
                                               "manifest_sha": M["rows_sha256"], "base": str(MIX)}))
    print("assembled", mm.shape, float(mm.nbytes) / 1e9, "GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10)
    a = ap.parse_args()
    if a.manifest:
        write_manifest()
    elif a.smoke:
        smoke()
    elif a.assemble:
        assemble()
    elif a.idx is not None:
        R, M = rows(), json.loads(MAN.read_text())["configs"]
        for i in range(a.idx * a.chunk, min((a.idx + 1) * a.chunk, len(R))):
            print(i, R[i]["filename"], build(i, R[i], M[i]), flush=True)


if __name__ == "__main__":
    main()
