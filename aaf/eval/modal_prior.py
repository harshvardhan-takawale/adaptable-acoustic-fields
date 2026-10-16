"""P4-7 m3 (D170(g)): the physics modal prior H_modal, the modal-kernel features, and the renderer wrapper
that supplies both to every evaluator.

H_modal is m1's transfer function (D170) with its one diagnosed defect removed:
    H_modal(f) = C S(f) e^{i w dt} sum_{n >= 1, f_n <= 600 Hz} psi_n(x_r) psi_n(x_s) / (k_n^2 - k^2 + i k eta_n)
               + (C / C_pred) psi_0(x_r) a_0(f),
where a_0 is the zero mode's EXACT record over the 2 s window (its scalar leapfrog recursion driven by the
solver's pulse; the continuous zero-mode term is wrong above 20 Hz by ~96%, D170(c)), C is m1's frozen
calibration and C_pred = dA / (c dt)^2. The modes are the solver's own discrete Laplacian (`operator`,
`eigen`: copied from `scripts/p4_7_m1_modal.py`, which stays the frozen m1 record).

Features (per receiver): for the first K mode CLUSTERS above the zero mode (consecutive eigenvalues within
CLUSTER_RTOL are one cluster, so a degenerate pair is summed and the feature is invariant to the basis
chosen inside its eigenspace), s_c = A sum_{n in c} psi_n(x_r) psi_n(x_s) and kc_c = k_c / k_600.

`ModalPriorRenderer` wraps `FreqRenderer2D`. `load_model` returns it for any checkpoint trained with a
prior, so an evaluator that knows nothing about m3 still renders H_modal + H_net: each row's room is found
by its conditioning vector, its receiver by coordinates, in the precomputed files under ROOT; a row with
no precomputed prior raises (no silent zero -- the D73 hazard closed by construction).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

DX, FS, N, C0, ALPHA = 0.02, 30720.0, 61440, 343.0, 0.15
DT = 1.0 / FS
BINS = 601
DF = FS / N
F_PRIOR = 600.0
C_FIT = complex(3.2055259717326545, 0.038008477027395966)     # m1 calibration rectangle (D170(b)), frozen
K_CLUSTERS = 48
CLUSTER_RTOL = 1e-4
ROOT = Path("data/track_p4_7_modal")
K600 = 2 * np.pi * F_PRIOR / C0


# ------------------------------------------------------------------------------------------- physics
def operator(verts, L: float, W: float, n_air_expected: Optional[int] = None) -> dict:
    import scipy.sparse as sp

    import aaf.sim.fdtd_2d as F
    from aaf.sim.polygon_geom import polygon_contains
    L, W, verts = float(L), float(W), [tuple(map(float, v)) for v in verts]
    nx, dxx = F._fit_axis(L, DX, "L")
    ny, dxy = F._fit_axis(W, DX, "W")
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    inside = polygon_contains(verts, np.stack([ii * dxx, jj * dxy], axis=-1))
    extra = [{"type": "mask", "solid": ~inside, "alpha": ALPHA}] if (~inside).any() else None
    geom = F.build_geometry(L, W, [ALPHA] * 4, dx=DX, extra_walls=extra)
    assert (geom.air == inside).all()
    if n_air_expected is not None and int(geom.air.sum()) != int(n_air_expected):
        raise AssertionError("air nodes {} != the corpus file's {}".format(int(geom.air.sum()), n_air_expected))
    coef = F.build_coefficients(geom, C0 * DT / dxx, C0 * DT / dxy)
    air = geom.air
    idx = -np.ones((nx, ny), dtype=np.int64)
    idx[air] = np.arange(int(air.sum()))
    XM, XP, YM, YP = 0, 1, 2, 3
    op = (~geom.blocked).astype(np.float64)
    w = {XP: op[XP] + 1 - op[XM], XM: op[XM] + 1 - op[XP], YP: op[YP] + 1 - op[YM], YM: op[YM] + 1 - op[YP]}
    I, J = np.nonzero(air)
    me = idx[I, J]
    rows, cols, vals = [me], [me], [np.full(len(me), -2.0 / dxx ** 2 - 2.0 / dxy ** 2)]
    for d, (di, dj, h2) in {XP: (1, 0, dxx ** 2), XM: (-1, 0, dxx ** 2), YP: (0, 1, dxy ** 2),
                            YM: (0, -1, dxy ** 2)}.items():
        wd = w[d][I, J]
        m = (wd > 0) & ~geom.blocked[d][I, J]
        nb = idx[I[m] + di, J[m] + dj]
        assert (nb >= 0).all()
        rows.append(me[m]); cols.append(nb); vals.append(wd[m] / h2)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(me), len(me)))
    muv = coef["mu"][I, J]
    K = -(sp.diags(muv) @ A)
    adm = np.where(geom.blocked, geom.adm, 0.0)
    D = 2.0 * ((adm[XM] + adm[XP]) / dxx + (adm[YM] + adm[YP]) / dxy)
    n = len(verts)
    perim = sum(float(np.hypot(verts[(k + 1) % n][0] - verts[k][0], verts[(k + 1) % n][1] - verts[k][1]))
                for k in range(n))
    return {"Ks": 0.5 * (K + K.T), "mu": muv, "D": D[I, J], "idx": idx, "dxx": dxx, "dxy": dxy,
            "dA": dxx * dxy, "asym": float(sp.linalg.norm(K - K.T) / sp.linalg.norm(K)),
            "n_air": int(len(me)), "area": float(muv.sum() * dxx * dxy), "perimeter": perim}


def weyl(area: float, perim: float, f: float) -> float:
    k = 2 * np.pi * f / C0
    return area * k * k / (4 * np.pi) + perim * k / (4 * np.pi)


def eigen(op: dict, f_max: float = F_PRIOR):
    """Every eigenpair with f_n <= f_max of K psi = k^2 diag(mu) psi, normalised sum mu psi^2 dA = 1."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    s = 1.0 / np.sqrt(op["mu"])
    Kstd = (sp.diags(s) @ op["Ks"] @ sp.diags(s)).tocsc()
    k2max = (2 * np.pi * f_max / C0) ** 2
    n_req = int(np.ceil(1.15 * weyl(op["area"], op["perimeter"], f_max))) + 20
    t0 = time.time()
    while True:
        vals, vecs = eigsh(Kstd, k=n_req, sigma=-0.05, which="LM", tol=1e-10)
        o = np.argsort(vals)
        vals, vecs = vals[o], vecs[:, o]
        if vals[-1] >= k2max:
            break
        n_req = int(n_req * 1.3) + 10
    keep = vals <= k2max
    psi = (vecs[:, keep] * s[:, None]) / np.sqrt(op["dA"])
    return np.maximum(vals[keep], 0.0), psi, time.time() - t0


def nodes(op: dict, pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    k = op["idx"][np.rint(p[:, 0] / op["dxx"]).astype(int), np.rint(p[:, 1] / op["dxy"]).astype(int)]
    if (k < 0).any():
        raise ValueError("{} point(s) snap onto a solid node".format(int((k < 0).sum())))
    return k


def pulse():
    import aaf.sim.fdtd_2d as F
    s = F.band_limited_pulse(n=N, fs=FS).astype(np.float64)
    return s, np.fft.rfft(s)[:BINS]


def zero_mode_record(eta0: float, g0: float, s: np.ndarray) -> np.ndarray:
    """rfft (bins 0-600) of the zero mode's leapfrog recursion (lambda = 0, damping b = c dt eta0 / 2),
    a^{k+1} = [2 a^k - (1 - b) a^{k-1}] / (1 + b) + g0 s^k, recorded as ir[k] = a^{k+1}."""
    b = C0 * DT * eta0 / 2.0
    c1, c2 = 2.0 / (1.0 + b), (1.0 - b) / (1.0 + b)
    rec = np.empty(N, dtype=np.float64)
    a_prev = a_now = 0.0
    for t in range(N):
        a_next = c1 * a_now - c2 * a_prev + g0 * s[t]
        rec[t] = a_next
        a_prev, a_now = a_now, a_next
    return np.fft.rfft(rec)[:BINS]


def clusters(k2: np.ndarray, rtol: float = CLUSTER_RTOL):
    """Index arrays of consecutive near-equal eigenvalues, EXCLUDING the zero mode (index 0)."""
    out, cur = [], [1]
    for n in range(2, len(k2)):
        if abs(k2[n] - k2[cur[0]]) <= rtol * k2[n]:
            cur.append(n)
        else:
            out.append(np.asarray(cur)); cur = [n]
    out.append(np.asarray(cur))
    return out


def features(psi_r: np.ndarray, psi_s: np.ndarray, k2: np.ndarray, area: float, k: int = K_CLUSTERS):
    cl = clusters(k2)[:k]
    s = np.stack([area * (psi_r[:, c] * psi_s[c][None]).sum(1) for c in cl], axis=1)
    kc = np.asarray([np.sqrt(k2[c].mean()) / K600 for c in cl])
    return s.astype(np.float32), kc.astype(np.float32), [len(c) for c in cl]


def compute(verts, L, W, rx, src, n_air_expected=None, k: int = K_CLUSTERS) -> Dict[str, np.ndarray]:
    op = operator(verts, L, W, n_air_expected)
    k2, psi, t_eig = eigen(op, F_PRIOR)
    jr, js = nodes(op, rx), int(nodes(op, np.asarray(src)[None])[0])
    psi_r, psi_s = psi[jr], psi[js]
    eta = ((op["mu"] * op["D"])[:, None] * psi ** 2).sum(0) * op["dA"]
    s_sig, S = pulse()
    f = np.arange(BINS) * DF
    kk = 2 * np.pi * f / C0
    den = k2[1:, None] - kk[None] ** 2 + 1j * kk[None] * eta[1:, None]
    den[:, 0] = 1.0
    G = psi_r[:, 1:] @ (psi_s[1:, None] / den)
    G = G * (S * np.exp(1j * 2 * np.pi * f * DT))[None]
    G[:, 0] = 0.0
    c_pred = op["dA"] / (C0 * DT) ** 2
    a0 = zero_mode_record(float(eta[0]), float(op["mu"][js] * psi_s[0] * op["dA"]), s_sig)
    H = C_FIT * G + psi_r[:, :1] * a0[None] * (C_FIT / c_pred)
    s, kc, sizes = features(psi_r, psi_s, k2, op["area"], k)
    meta = {"n_modes": int(len(k2)), "weyl": weyl(op["area"], op["perimeter"], F_PRIOR), "eig_s": round(t_eig, 1),
            "asymmetry": op["asym"], "area": op["area"], "perimeter": op["perimeter"], "eta0": float(eta[0]),
            "n_air": op["n_air"], "cluster_sizes": sizes, "C_fit": [C_FIT.real, C_FIT.imag], "C_pred": c_pred}
    return {"H": H.astype(np.complex64), "s": s, "kc": kc, "meta": meta}


# ------------------------------------------------------------------------------------------- lookup
def cond_key(cond) -> str:
    c = np.round(np.asarray(cond, dtype=np.float64).ravel(), 5) + 0.0          # +0.0: no -0.0 bytes
    return hashlib.sha1(c.tobytes()).hexdigest()


def feat_rows(s: np.ndarray, kc: np.ndarray, k: int) -> np.ndarray:
    """[n_rx, 2k]: the first k cluster kernels, then their k / k_600 (the same for every receiver)."""
    if k > s.shape[1]:
        raise ValueError("asked for {} clusters, {} precomputed".format(k, s.shape[1]))
    return np.concatenate([s[:, :k], np.broadcast_to(kc[None, :k], (s.shape[0], k))], axis=1).astype(np.float32)


def load_room(path) -> dict:
    z = np.load(str(path), allow_pickle=False)
    return {k: z[k] for k in z.files}


class ModalPriorRenderer(torch.nn.Module):
    """`FreqRenderer2D` plus the m3 prior: rows are matched to precomputed rooms (by conditioning vector) and
    receivers (by coordinates). Explicit `h_prior` / `modal_feat` (the trainer's path) bypass the lookup."""

    def __init__(self, base: torch.nn.Module, root=ROOT, tol: float = 1e-4):
        super().__init__()
        self.base = base
        self._root, self._tol = Path(root), float(tol)
        self._index: Optional[Dict[str, list]] = None
        self._rooms: Dict[Path, dict] = {}

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["base"], name)

    def _candidates(self, key: str):
        """Every precomputed file of this room (a room can sit in several data sets with different receivers)."""
        if self._index is None:
            self._index = {}
            for p in sorted(self._root.rglob("*.npz")):
                if p.name.endswith(".tmp.npz"):                    # a precompute still writing it
                    continue
                with np.load(str(p), allow_pickle=False) as z:
                    self._index.setdefault(str(z["cond_key"]), []).append(p)
        if key not in self._index:
            raise KeyError("no precomputed modal prior for this room (cond key {}) under {}; run "
                           "scripts/p4_7_m3_precompute.py for its data set".format(key[:12], self._root))
        out = []
        for p in self._index[key]:
            if p not in self._rooms:
                from scipy.spatial import cKDTree
                d = load_room(p)
                d["_tree"] = cKDTree(np.asarray(d["rx"], dtype=np.float64))
                self._rooms[p] = d
            out.append(self._rooms[p])
        return out

    def lookup(self, model, z_s, rx_pos, tx_pos):
        z = z_s.detach().float().cpu().numpy()
        rx = rx_pos.detach().double().cpu().numpy()
        tx = tx_pos.detach().double().cpu().numpy()
        k = int(getattr(model, "modal_feat_dim", 0)) // 2
        H, Fe = [], []
        for b in range(len(z)):
            hit = None
            for d in self._candidates(cond_key(z[b])):
                dist, i = d["_tree"].query(rx[b])
                if dist <= self._tol and np.abs(d["src"] - tx[b]).max() <= 1e-3:
                    hit = (d, int(i))
                    break
            if hit is None:
                raise KeyError("row {}: receiver {} / source {} not among the precomputed receivers of this "
                               "room".format(b, rx[b], tx[b]))
            d, i = hit
            H.append(d["H"][i])
            if k:
                Fe.append(feat_rows(d["s"][i:i + 1], d["kc"], k)[0])
        dev = rx_pos.device
        h = torch.as_tensor(np.stack(H), device=dev)
        f = torch.as_tensor(np.stack(Fe), device=dev) if k else None
        return h, f

    def forward(self, model, rx_pos, tx_pos, room_min, room_max, z_s=None, h_prior=None, modal_feat=None):
        if h_prior is None:
            h_prior, modal_feat = self.lookup(model, z_s, rx_pos, tx_pos)
        return self.base(model, rx_pos, tx_pos, room_min, room_max, z_s=z_s, h_prior=h_prior,
                         modal_feat=modal_feat)
