"""P4-7 m1 (D169): a physics-only MODAL baseline -- no training, CPU only.

Per room (`--room K`, one SLURM array task each):
  1. Rebuild the solver's own discrete Laplacian from `fdtd_2d.build_geometry` (dx 0.02 fitted per axis,
     the solver's solid mask = nodes outside the polygon, exactly what every builder asserted): per-direction
     weights w_d (a blocked face doubles the opposite neighbour = the image-node Neumann wall),
          (A p)_j = (wE p_E + wW p_W - 2 p_j) / dx_x^2 + (wN p_N + wS p_S - 2 p_j) / dx_y^2 .
     K = -diag(mu) A is symmetric except at reflex-corner nodes (D65); the mu-weighted symmetrisation
     (K + K^T) / 2 is used and its relative asymmetry reported.
  2. Eigenpairs K psi = k^2 diag(mu) psi (shift-invert Lanczos), every mode to F_SOLVE (800 Hz; the
     primary uses <= 600 Hz, the extra modes are the truncation diagnostic), normalised
     sum_j mu_j psi_j^2 dA = 1. Eigen-solve wall time recorded.
  3. Damping eta_n = sum_j mu_j D_j psi_n(j)^2 dA with D_j = 2 (sum of blocked x-face xi / dx_x + y-face
     xi / dx_y): the grid form of beta * (boundary integral of psi_n^2), from the solver's own semi-discrete
     KW wall (p_tt + c D p_t = c^2 A p).
  4. Saved: psi at the receivers and the source node, k_n^2, eta_n, and the EXACT discrete time-domain modal
     synthesis (every mode's leapfrog recursion driven by the solver's actual pulse over the 2 s record,
     the diagonal damping b_n = c dt eta_n / 2, rfft, bins 0-600) -- a no-fit diagnostic that includes the
     record truncation of the slowly relaxing zero mode (D101(b), D165 p2).
`--score`: the manager's transfer function
     H(f) = C S(f) e^{i w dt} sum_n psi_n(x_r) psi_n(x_s) / (k_n^2 - k^2 + i k eta_n)
with S the solver's pulse spectrum and e^{i w dt} the documented one-step alignment (`ir[:, k] = p^{k+1}`);
ONE complex C fitted on the calibration rectangle (bins 40-600, all its receivers), frozen; the sign of the
damping term checked by fitting both. Scored off-lattice above 20 Hz exactly as `p4_6_family_oof.scores`,
beside PB3 / G1 @60K from their existing JSONs. Writes outputs/p4_7/m1_modal.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.sim.polygon_geom import polygon_contains

DX, FS, N, C0, ALPHA = 0.02, 30720.0, 61440, 343.0, 0.15
DT = 1.0 / FS
F_PRIMARY, F_SOLVE = 600.0, 800.0
BINS = 601
DF = FS / N
BANDS = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
OUT = Path("outputs/p4_7/m1")
TEST_MAN = "configs/sweeps_2d_mat/p4_2_shapes_manifest.json"
C250_MAN = "configs/sweeps_2d_mat/p4_4_shapes_250.json"


# ------------------------------------------------------------------------------------------- rooms
def rooms():
    """The 30 rooms, in a fixed order: 15 test shapes, 10 family rooms, 5 training rooms (D169 rule)."""
    from aaf.data.shape_configs import configs_from_rows
    from scripts.build_p4_5_rooms import OUT as ROOT, family_test_rooms
    out = []
    for c in configs_from_rows(json.load(open(TEST_MAN))["configs"], split="test"):
        out.append({"set": "test", "family": "L" if c.kind == "L" else c.kind, "cfg": c,
                    "path": str(Path(ROOT["testdense"]) / c.filename)})
    for j, (fam, c) in enumerate(family_test_rooms() + family_test_rooms(second=True)):
        out.append({"set": "family", "family": fam, "cfg": c, "pick": "first" if j < 5 else "second",
                    "path": str(Path(ROOT["famdense"]) / c.filename)})
    tr = configs_from_rows(json.load(open(C250_MAN))["configs"], split="train")
    rects = sorted([c for c in tr if c.kind == "rect"], key=lambda c: (c.L * c.W, c.filename))
    cal = rects[(len(rects) - 1) // 2]
    Ls = sorted([c for c in tr if c.kind == "L"], key=lambda c: (c.d_hat, c.filename))
    picks = [Ls[int(round(q * (len(Ls) - 1)))] for q in (0.10, 0.35, 0.65, 0.90)]
    for c, role in [(cal, "calibration")] + [(c, "train") for c in picks]:
        out.append({"set": "train", "family": c.kind, "cfg": c, "role": role,
                    "path": str(Path("data/track_p4_5_offlat") / c.filename)})
    return out


def load_room(r):
    with h5py.File(r["path"]) as f:
        T = np.asarray(f["ism/H_complex"])[:, :BINS]
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]), dtype=np.float64)
        src = np.asarray(json.loads(f.attrs["source_pos"]), dtype=np.float64)
        n_air = int(f.attrs["n_air_nodes"]) if "n_air_nodes" in f.attrs else None
    return T, rx, src, n_air


# ------------------------------------------------------------------------------------------- operator
def operator(c, n_air_expected=None):
    import scipy.sparse as sp
    L, W, verts = float(c.L), float(c.W), [tuple(v) for v in c.verts]
    nx, dxx = F._fit_axis(L, DX, "L")
    ny, dxy = F._fit_axis(W, DX, "W")
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    inside = polygon_contains(verts, np.stack([ii * dxx, jj * dxy], axis=-1))
    extra = [{"type": "mask", "solid": ~inside, "alpha": ALPHA}] if (~inside).any() else None
    geom = F.build_geometry(L, W, [ALPHA] * 4, dx=DX, extra_walls=extra)
    assert (geom.air == inside).all()
    if n_air_expected is not None:
        assert int(geom.air.sum()) == n_air_expected, (c.label, int(geom.air.sum()), n_air_expected)
    lam_x, lam_y = C0 * DT / dxx, C0 * DT / dxy
    coef = F.build_coefficients(geom, lam_x, lam_y)
    mu = coef["mu"]
    air = geom.air
    idx = -np.ones((nx, ny), dtype=np.int64)
    idx[air] = np.arange(int(air.sum()))
    XM, XP, YM, YP = 0, 1, 2, 3
    op = (~geom.blocked).astype(np.float64)
    w = {XP: op[XP] + 1 - op[XM], XM: op[XM] + 1 - op[XP], YP: op[YP] + 1 - op[YM], YM: op[YM] + 1 - op[YP]}
    rows, cols, vals = [], [], []
    I, J = np.nonzero(air)
    me = idx[I, J]
    rows.append(me); cols.append(me); vals.append(np.full(len(me), -2.0 / dxx ** 2 - 2.0 / dxy ** 2))
    for d, (di, dj, h2) in {XP: (1, 0, dxx ** 2), XM: (-1, 0, dxx ** 2), YP: (0, 1, dxy ** 2),
                            YM: (0, -1, dxy ** 2)}.items():
        wd = w[d][I, J]
        m = (wd > 0) & ~geom.blocked[d][I, J]
        nb = idx[I[m] + di, J[m] + dj]
        assert (nb >= 0).all()
        rows.append(me[m]); cols.append(nb); vals.append(wd[m] / h2)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(me), len(me)))
    muv = mu[I, J]
    K = -(sp.diags(muv) @ A)
    asym = float(sp.linalg.norm(K - K.T) / sp.linalg.norm(K))
    Ks = 0.5 * (K + K.T)
    adm = np.where(geom.blocked, geom.adm, 0.0)
    D = 2.0 * ((adm[XM] + adm[XP]) / dxx + (adm[YM] + adm[YP]) / dxy)
    perim = sum(np.hypot(verts[(k + 1) % len(verts)][0] - verts[k][0], verts[(k + 1) % len(verts)][1] - verts[k][1])
                for k in range(len(verts)))
    return {"Ks": Ks, "mu": muv, "D": D[I, J], "idx": idx, "dxx": dxx, "dxy": dxy, "dA": dxx * dxy,
            "asym": asym, "n_air": int(len(me)), "area_q": float(muv.sum() * dxx * dxy), "perimeter": perim,
            "beta": float(F.wall_admittance(ALPHA))}


def weyl(area, perim, f):
    k = 2 * np.pi * f / C0
    return area * k * k / (4 * np.pi) + perim * k / (4 * np.pi)


def eigen(op, f_max):
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    s = 1.0 / np.sqrt(op["mu"])
    Kstd = sp.diags(s) @ op["Ks"] @ sp.diags(s)
    k2max = (2 * np.pi * f_max / C0) ** 2
    n_req = int(np.ceil(1.15 * weyl(op["area_q"], op["perimeter"], f_max))) + 20
    t0 = time.time()
    while True:
        vals, vecs = eigsh(Kstd.tocsc(), k=n_req, sigma=-0.05, which="LM", tol=1e-10)
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        if vals[-1] >= k2max:
            break
        n_req = int(n_req * 1.3) + 10
    keep = vals <= k2max
    vals, vecs = np.maximum(vals[keep], 0.0), vecs[:, keep]
    psi = (vecs * s[:, None]) / np.sqrt(op["dA"])          # sum mu psi^2 dA = 1
    return vals, psi, time.time() - t0, n_req


def nodes(op, pts):
    i = np.rint(pts[:, 0] / op["dxx"]).astype(int)
    j = np.rint(pts[:, 1] / op["dxy"]).astype(int)
    k = op["idx"][i, j]
    assert (k >= 0).all()
    return k


def pulse():
    s = F.band_limited_pulse(n=N, fs=FS).astype(np.float64)
    return s, np.fft.rfft(s)[:BINS]


def td_synthesis(k2, eta, g, s):
    """Every mode's exact leapfrog recursion (diagonal damping), driven by the actual pulse; rfft of
    a_n^{k+1} over the 2 s record. Returns [n_modes, BINS]."""
    lam = (C0 * DT) ** 2 * k2
    b = C0 * DT * eta / 2.0
    c1, c2 = (2.0 - lam) / (1.0 + b), (1.0 - b) / (1.0 + b)
    a_prev = np.zeros_like(k2)
    a_now = np.zeros_like(k2)
    rec = np.empty((len(k2), N), dtype=np.float64)
    for t in range(N):
        a_next = c1 * a_now - c2 * a_prev + g * s[t]
        rec[:, t] = a_next
        a_prev, a_now = a_now, a_next
    return np.fft.rfft(rec, axis=1)[:, :BINS]


def run_room(k):
    r = rooms()[k]
    c = r["cfg"]
    T, rx, src, n_air = load_room(r)
    op = operator(c, n_air)
    vals, psi, t_eig, n_req = eigen(op, F_SOLVE)
    jr, js = nodes(op, rx), nodes(op, src[None])[0]
    eta = (op["mu"] * op["D"])[:, None] * psi ** 2
    eta = eta.sum(0) * op["dA"]
    s, S = pulse()
    g = op["mu"][js] * psi[js] * op["dA"]
    t0 = time.time()
    td = td_synthesis(vals, eta, g, s)
    t_td = time.time() - t0
    f_n = C0 * np.sqrt(vals) / (2 * np.pi)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "room{:02d}.npz".format(k), k2=vals, f_n=f_n, eta=eta,
                        psi_r=psi[jr].astype(np.float64), psi_s=psi[js], td=td.astype(np.complex128),
                        S=S, rx=rx, src=src)
    meta = {"k": k, "set": r["set"], "family": r["family"], "label": c.label, "role": r.get("role"),
            "pick": r.get("pick"), "n_air": op["n_air"], "n_rx": int(len(rx)), "asymmetry": op["asym"],
            "area": op["area_q"], "perimeter": op["perimeter"], "beta": op["beta"],
            "n_modes_600": int((f_n <= F_PRIMARY).sum()), "n_modes_800": int(len(f_n)),
            "weyl_600": weyl(op["area_q"], op["perimeter"], F_PRIMARY), "eig_s": round(t_eig, 1),
            "eig_requested": n_req, "td_s": round(t_td, 1), "eta0": float(eta[0]), "f0": float(f_n[0]),
            "psi0_sq_x_area": float(psi[js, 0] ** 2 * op["area_q"])}
    (OUT / "room{:02d}.json".format(k)).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


# ------------------------------------------------------------------------------------------- scoring
def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def colr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def scores(P, T):
    r = colr(db(P), db(T))
    out = {b: float(np.nanmean(r[lo:hi])) for b, (lo, hi) in BANDS.items()}
    out.update({"lsd_" + b: float(np.mean(np.abs(db(P[:, lo:hi]) - db(T[:, lo:hi])))) for b, (lo, hi) in BANDS.items()})
    out["r_100Hz"] = float(np.nanmean(r[195:206]))
    return out


def formula(z, f_max=F_PRIMARY, sign=+1.0, eta_scale=1.0):
    f = np.arange(BINS) * DF
    k = 2 * np.pi * f / C0
    m = z["f_n"] <= f_max
    den = z["k2"][m][:, None] - k[None] ** 2 + sign * 1j * k[None] * (eta_scale * z["eta"][m])[:, None]
    den[:, 0] = 1.0
    G = z["psi_r"][:, m] @ (z["psi_s"][m][:, None] / den)
    G = G * (z["S"] * np.exp(1j * 2 * np.pi * f * DT))[None]
    G[:, 0] = 0.0
    return G


def score_all():
    from scripts.p4_5_gate2_dense import on_lattice
    rs = rooms()
    metas = [json.loads((OUT / "room{:02d}.json".format(k)).read_text()) for k in range(len(rs))]
    kc = next(k for k, r in enumerate(rs) if r.get("role") == "calibration")
    zc = dict(np.load(OUT / "room{:02d}.npz".format(kc)))
    Tc = load_room(rs[kc])[0]
    lo = 40
    fit = {}
    for name, sign in (("plus", 1.0), ("minus", -1.0)):
        G = formula(zc, sign=sign)
        Cf = complex(np.vdot(G[:, lo:], Tc[:, lo:]) / np.vdot(G[:, lo:], G[:, lo:]))
        res = float(np.linalg.norm(Cf * G[:, lo:] - Tc[:, lo:]) / np.linalg.norm(Tc[:, lo:]))
        fit[name] = {"C": [Cf.real, Cf.imag], "abs_C": abs(Cf), "arg_C_deg": float(np.degrees(np.angle(Cf))),
                     "rel_residual": res}
    sign = 1.0 if fit["plus"]["rel_residual"] <= fit["minus"]["rel_residual"] else -1.0
    Cfit = complex(*fit["plus" if sign > 0 else "minus"]["C"])
    C_pred = DX * DX * FS * FS / C0 ** 2     # dA / (c dt)^2 at dx_x = dx_y = 0.02; per room below
    rep = {"calibration_room": metas[kc]["label"], "sign_check": fit, "sign_used": "+ i k eta" if sign > 0 else "- i k eta",
           "C_fit": [Cfit.real, Cfit.imag], "C_pred_nominal": C_pred, "rooms": []}
    for k, r in enumerate(rs):
        z = dict(np.load(OUT / "room{:02d}.npz".format(k)))
        T, rx, _src, _ = load_room(r)
        off = ~on_lattice(rx)
        H = Cfit * formula(z, sign=sign)
        row = dict(metas[k])
        row["n_off"] = int(off.sum())
        row["formula"] = scores(H[off], T[off])
        m6 = z["f_n"] <= F_PRIMARY
        row["td_nofit"] = scores((z["psi_r"][:, m6] @ z["td"][m6])[off], T[off])
        row["td_nofit_800"] = scores((z["psi_r"] @ z["td"])[off], T[off])
        row["formula_800"] = scores((Cfit * formula(z, f_max=F_SOLVE, sign=sign))[off], T[off])
        # the diagnosis of formula vs td: the same formula with ONLY the zero-mode term replaced by its exact
        # record (the slowly relaxing step truncated by the 2 s window), scaled by C_fit / C_pred
        z1 = dict(z, f_n=z["f_n"].copy())
        z1["f_n"][0] = np.inf                                  # drop mode 0 from the formula
        dA_r = (r["cfg"].L / round(r["cfg"].L / DX)) * (r["cfg"].W / round(r["cfg"].W / DX))
        H0 = z["psi_r"][:, :1] * z["td"][None, 0] * (Cfit / (dA_r * FS * FS / C0 ** 2))
        row["formula_zero_exact"] = scores((Cfit * formula(z1, sign=sign) + H0)[off], T[off])
        row["formula_eta_x0.5"] = scores((Cfit * formula(z, sign=sign, eta_scale=0.5))[off], T[off])
        row["formula_eta_x2"] = scores((Cfit * formula(z, sign=sign, eta_scale=2.0))[off], T[off])
        # level: the fitted C against this room's analytic dA / (c dt)^2
        dA = (r["cfg"].L / round(r["cfg"].L / DX)) * (r["cfg"].W / round(r["cfg"].W / DX))
        row["C_pred_room"] = dA * FS * FS / C0 ** 2
        # zero mode vs D101 and vs the solver: the FDTD spatial mean at 0-20 Hz against the modal one
        # (exact time-domain synthesis, no fit), and the zero mode's continuous-frequency term against
        # its exact record above 20 Hz (the truncation of the slowly relaxing step).
        mT = T[:, :41].mean(0)
        mP = (z["psi_r"].mean(0)[:, None] * z["td"][:, :41]).sum(0)
        row["zero_mode"] = {
            "eta0": float(z["eta"][0]), "eta0_over_beta_P_over_A": float(z["eta"][0] / (row["beta"] * row["perimeter"] / row["area"])),
            "decay_rate_c_eta0_per_s": float(C0 * z["eta"][0]),
            "dc_fdtd_abs": float(abs(mT[0])), "dc_modal_td_abs": float(abs(mP[0])),
            "rel_err_spatial_mean_0_20Hz": float(np.linalg.norm(mP - mT) / np.linalg.norm(mT)),
            "rel_err_spatial_mean_0_10Hz": float(np.linalg.norm(mP[:21] - mT[:21]) / np.linalg.norm(mT[:21])),
            "rel_err_spatial_mean_10_20Hz": float(np.linalg.norm(mP[21:] - mT[21:]) / np.linalg.norm(mT[21:])),
        }
        f = np.arange(BINS) * DF
        kk = 2 * np.pi * f / C0
        cont0 = row["C_pred_room"] * z["psi_s"][0] * z["S"] * np.exp(1j * 2 * np.pi * f * DT)
        cont0[1:] = cont0[1:] / (-kk[1:] ** 2 + sign * 1j * kk[1:] * z["eta"][0])
        cont0[0] = 0.0
        ex0 = z["td"][0]
        row["zero_mode"]["cont_vs_record_rel_diff_25_300Hz"] = float(
            np.median(np.abs(cont0[50:] - ex0[50:]) / np.abs(ex0[50:])))
        row["zero_mode"]["zero_mode_share_of_T_25_100Hz"] = float(
            np.linalg.norm(z["psi_r"][:, 0][:, None] * ex0[None, 50:200]) / np.linalg.norm(T[:, 50:200]))
        rep["rooms"].append(row)
        print("{:6s} {:5s} {:48s} formula r {:.3f}/{:.3f}/{:.3f} lsd {:.1f}/{:.1f}/{:.1f} | td r {:.3f}/{:.3f}/{:.3f} | "
              "modes {} eig {:.0f}s".format(r["set"], r["family"], row["label"][:48],
                                            *[row["formula"][b] for b in BANDS],
                                            *[row["formula"]["lsd_" + b] for b in BANDS],
                                            *[row["td_nofit"][b] for b in BANDS], row["n_modes_600"], row["eig_s"]),
              flush=True)
    rep["summary"] = summarise(rep["rooms"])
    rep["decision"] = decide(rep["summary"])
    (OUT.parent / "m1_modal.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["summary"], indent=1))
    print(json.dumps(rep["decision"], indent=1))
    print("wrote", OUT.parent / "m1_modal.json")


GROUPS = {"test15 (in family, L)": lambda r: r["set"] == "test",
          "family in-family ref (rect, L)": lambda r: r["set"] == "family" and r["family"] in ("rect", "L"),
          "family OUT of family (T, DN, U)": lambda r: r["set"] == "family" and r["family"] in ("T", "DN", "U"),
          "family T": lambda r: r["set"] == "family" and r["family"] == "T",
          "family DN": lambda r: r["set"] == "family" and r["family"] == "DN",
          "family U": lambda r: r["set"] == "family" and r["family"] == "U",
          "train (5 C250 rooms)": lambda r: r["set"] == "train"}
KEYS = list(BANDS) + ["lsd_" + b for b in BANDS] + ["r_100Hz"]


def network_rows(name):
    """PB3 / G1 per-room numbers from their existing JSONs, keyed like ours (same receivers, same metric)."""
    out = {}
    sp = json.load(open("outputs/p4_5/speckle/{}.json".format(name)))["per_shape"]
    from aaf.data.shape_configs import configs_from_rows
    for i, c in enumerate(configs_from_rows(json.load(open(TEST_MAN))["configs"], split="test")):
        out[("test", c.label)] = {**{b: sp["r"][b][i] for b in BANDS}, **{"lsd_" + b: sp["lsd"][b][i] for b in BANDS}}
    fo = json.load(open("outputs/p4_6/family_oof/{}.json".format(name)))
    for row in fo["rooms"]:
        out[("family", row["room"])] = row["raw"]
    tf = Path("outputs/p4_5/train_offlat_fit/{}.json".format(name))
    if tf.exists():
        for row in json.load(open(tf))["rooms"]:
            out[("train", row["room"])] = {"25-100": row["r_25_100"], "100-200": row["r_100_200"],
                                           "200-300": row["r_200_300"], "lsd_100-200": row.get("lsd_100_200"),
                                           "lsd_200-300": row.get("lsd_200_300")}
    return out


def summarise(rows):
    nets = {n: network_rows(n) for n in ("PB3_60K", "G1_60K")}
    out = {}
    for g, sel in GROUPS.items():
        rr = [r for r in rows if sel(r)]
        ent = {"n_rooms": len(rr)}
        for v in ("formula", "formula_zero_exact", "td_nofit", "td_nofit_800", "formula_800", "formula_eta_x0.5", "formula_eta_x2"):
            ent[v] = {k: float(np.mean([r[v][k] for r in rr])) for k in KEYS}
        for n, tab in nets.items():
            got = [tab.get((r["set"], r["label"])) for r in rr]
            if all(x is not None for x in got):
                ent[n] = {k: float(np.mean([x[k] for x in got])) for k in KEYS if all(x.get(k) is not None for x in got)}
            else:
                ent[n] = "n/a ({} of {} rooms scored)".format(sum(x is not None for x in got), len(got))
        out[g] = ent
    return out


def decide(s):
    oof = s["family OUT of family (T, DN, U)"]["formula"]["25-100"]
    inf = s["test15 (in family, L)"]["formula"]["25-100"]
    return {"out_of_family_25_100_r": oof, "threshold": 0.85, "m3_launches": bool(oof >= 0.85),
            "in_family_test15_25_100_r": inf, "poor_in_family (< 0.85)": bool(inf < 0.85)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, r in enumerate(rooms()):
            print(k, r["set"], r["family"], r.get("role") or r.get("pick") or "", r["cfg"].label, r["path"])
        return
    if a.score:
        score_all()
        return
    run_room(a.room)


if __name__ == "__main__":
    main()
