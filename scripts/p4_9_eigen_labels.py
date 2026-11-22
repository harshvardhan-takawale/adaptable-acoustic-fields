"""P4-9 2c (D180 NMF-LB; D181 A10): eigen-labels for the neural modal field, m1's solver (`scripts/p4_7_m1_modal.py`).

Per room (the 1000 GX training rooms in manifest order, then the 31 dense rooms): m1's discrete operator on the room's
polygon (the FDTD's 2 cm Laplacian, image-node Neumann walls, mu-weighted symmetrisation), the zero mode + the first 28
nonzero eigenpairs (NMF uses 24; modes 25-28 feed the soft projector cutoff, A10), m1's normalisation
sum_j mu_j psi_j^2 dA = 1. Stored (float32 except where noted):
  k2 [29], eta [29] (= sum_j mu_j D_j psi_j^2 dA, beta 0.0406), psi at the room's receivers [n_rx, 29] and source [29];
  quadrature: every 3rd air node in x and y (~6 cm), psi [n_q, 29] and weights w_q = 9 mu_j dA (sum w psi^2 ~ 1);
  walls: every air node with a blocked face, psi [n_w, 29], positions and m1's damping weights mu_j D_j dA
  (eta_n = sum_w weight * psi_n^2); the zero mode's exact record td0 [601] (complex128) at the source; area, perimeter, dA.
  python scripts/p4_9_eigen_labels.py --set gx --idx I [--chunk 8]      -> data/track_p4_9_eigen/gx/NNNN.npz
  python scripts/p4_9_eigen_labels.py --set dense --idx I               -> data/track_p4_9_eigen/dense/<room>.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from aaf.eval.modal_prior import pulse, zero_mode_record
from scripts.p4_7_m1_modal import operator

OUT = Path("data/track_p4_9_eigen")
GX_MAN = "configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"
GX_DIR = Path("data/track_p4_7_mix")
N_NONZERO = 28
Q_STRIDE = 3


def eig_lowest(op, n):
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    s = 1.0 / np.sqrt(op["mu"])
    Kstd = sp.diags(s) @ op["Ks"] @ sp.diags(s)
    vals, vecs = eigsh(Kstd.tocsc(), k=n + 4, sigma=-0.05, which="LM", tol=1e-10)
    o = np.argsort(vals)
    vals, vecs = np.maximum(vals[o][:n + 1], 0.0), vecs[:, o][:, :n + 1]
    psi = (vecs * s[:, None]) / np.sqrt(op["dA"])
    return vals, psi


def label_room(path, verts=None):
    with h5py.File(path) as f:
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
        src = np.asarray(json.loads(f.attrs["source_pos"]), float)
        vv = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])] if verts is None else verts
    L, W = max(v[0] for v in vv), max(v[1] for v in vv)
    t0 = time.time()
    op = operator(SimpleNamespace(L=L, W=W, verts=vv, label=Path(path).stem))
    k2, psi = eig_lowest(op, N_NONZERO)
    idx, dxx, dxy = op["idx"], op["dxx"], op["dxy"]

    def node(p):
        q = idx[np.rint(p[:, 0] / dxx).astype(int), np.rint(p[:, 1] / dxy).astype(int)]
        assert (q >= 0).all()
        return q
    jr, js = node(rx), node(src[None])[0]
    I, J = np.nonzero(idx >= 0)
    assert (idx[I, J] == np.arange(len(I))).all()
    qsel = (I % Q_STRIDE == 0) & (J % Q_STRIDE == 0)
    qn = idx[I[qsel], J[qsel]]
    wq = (Q_STRIDE ** 2) * op["mu"][qn] * op["dA"]
    wn = np.nonzero(op["D"] > 0)[0]
    ww = op["mu"][wn] * op["D"][wn] * op["dA"]
    eta = (ww[:, None] * psi[wn] ** 2).sum(0)
    s, S = pulse()
    td0 = zero_mode_record(float(eta[0]), float(op["mu"][js] * psi[js, 0] * op["dA"]), s)
    pos = np.stack([I * dxx, J * dxy], axis=1)
    return {"k2": k2, "eta": eta, "psi_r": psi[jr].astype(np.float32), "psi_s": psi[js].astype(np.float64),
            "q_pos": pos[qn].astype(np.float32), "psi_q": psi[qn].astype(np.float32), "w_q": wq.astype(np.float32),
            "w_pos": pos[wn].astype(np.float32), "psi_w": psi[wn].astype(np.float32), "w_w": ww.astype(np.float32),
            "td0": td0, "S": S, "rx": rx, "src": src, "area": op["area_q"], "perimeter": op["perimeter"],
            "dA": op["dA"], "dxx": dxx, "dxy": dxy, "beta": op["beta"], "solve_s": time.time() - t0,
            "quad_check": float((wq[:, None] * psi[qn] ** 2).sum(0)[1:].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("gx", "dense"), required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    if a.set == "gx":
        rows = json.load(open(GX_MAN))["configs"]
        out = OUT / "gx"
        out.mkdir(parents=True, exist_ok=True)
        for k in range(a.idx * a.chunk, min(len(rows), (a.idx + 1) * a.chunk)):
            dst = out / "{:04d}.npz".format(k)
            if dst.exists():
                continue
            lab = label_room(GX_DIR / rows[k]["filename"], [tuple(v) for v in rows[k]["verts"]])
            np.savez(str(dst).replace(".npz", ".tmp.npz"), filename=rows[k]["filename"], **lab)
            Path(str(dst).replace(".npz", ".tmp.npz")).replace(dst)
            print(k, rows[k]["filename"], "f_24 = {:.0f} Hz".format(343 * np.sqrt(lab["k2"][24]) / (2 * np.pi)),
                  "quad {:.3f}".format(lab["quad_check"]), "{:.1f}s".format(lab["solve_s"]), flush=True)
    else:
        from scripts.p4_9_fdtd_stencils import dense_rooms
        g, fam, p = dense_rooms()[a.idx]
        out = OUT / "dense"
        out.mkdir(parents=True, exist_ok=True)
        lab = label_room(p)
        np.savez(out / (p.stem + ".npz"), filename=p.name, group=g, family=fam, **lab)
        print(p.name, "quad {:.3f}".format(lab["quad_check"]), "{:.1f}s".format(lab["solve_s"]), flush=True)


if __name__ == "__main__":
    main()
