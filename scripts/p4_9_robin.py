"""P4-9 0b(ii) (D180; D181 A9): the Robin boundary residual and gate G-BC.

Wall stencils (`scripts/p4_9_fdtd_stencils.py --set dense`): points every 5 cm along each polygon edge, > 0.15 m from
its ends and >= 0.3 m from the source; the field at the wall's boundary node and 1, 2 nodes along the inward normal.
Normal derivative: one-sided, second order, d_n,in p = (-3 p0 + 4 p1 - p2) / (2 h), h = the node spacing along the
normal (D180; a 2-point difference would err by ~0.30 |p| at 300 Hz, A9). Operator: m1's first-order Robin condition
d_n,out p = -j k beta p, beta = tanh(m/4) = 0.0406 (alpha 0.15): the sign that gives m1's "+ i k eta" modal damping
(D170). Metric: rel(f) = ||d_n,out p + j k beta p|| / ||k p0|| over the room's wall points; band means. NOTE (A9): at
beta 0.04 the Robin term is <= 4% of k|p|, so the residual effectively tests near-Neumann behaviour.
Fields: FDTD truth; R0 / S2 @120K RAW (rendered at the stencil points, 601-bin path); and PROJECTED: the render-time
projection's least-squares band-limited field (|k| <= 1.5 k0, fitted on the dense lattice), evaluated at the stencil
points by its exact Fourier series -- 12 cm past the lattice (A9). The identical projection + extension applied to FDTD
truth must leave its residual <= 0.15 (the validity check).
Gate G-BC (D180; computed on R0, S2 beside it, A9): (1) FDTD truth residual <= 0.15 at 25-100 Hz; (2) R0's projected
residual unseen / in-family >= 1.5 at 25-100 Hz; (3) Spearman(per-room R0 projected residual, per-room R0 25-100 Hz r)
<= -0.4 over the 31 rooms. All three -> PL1 launches.
  --render   GPU: networks at the wall stencils -> outputs/p4_9/robin/cache/
  --score    GPU (projection): outputs/p4_9/gates/robin.json + the physics_gates.md section
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

STEN = Path("outputs/p4_9/fdtd_stencils/dense")
OUT = Path("outputs/p4_9/robin")
GATES = Path("outputs/p4_9/gates")
MODELS = {"R0_120K": "outputs/p4_8/arms/R0/ckpt_iter0120000.pt", "S2_120K": "outputs/p4_8/arms/S2/ckpt_iter0120000.pt"}
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
UN = "outputs/p4_8/cache/unseen_{}.npz"
C0, DF, LO = 343.0, 0.5, 50
BANDS = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
WALL_CLEAR = 0.12


def beta():
    import aaf.sim.fdtd_2d as F
    return float(F.wall_admittance(0.15))


def rooms():
    from scripts.p4_9_dir_zeroshot import dense_rooms
    return dense_rooms()


def stencil(st):
    ij = st["wall_ij"]                                                     # [n, 3, 2] node indices
    P = ij * np.array([float(st["dxx"]), float(st["dxy"])])
    nrm = st["wall_meta"][:, 2:4]
    h = np.where(np.abs(nrm[:, 0]) > 0.5, float(st["dxx"]), float(st["dxy"]))
    return P, h


def residual(W, h, b):
    """W [n, 3, F] (bins 0-600) -> rel(f) [F] and band means (bins >= 25 Hz)."""
    k = 2 * np.pi * np.arange(W.shape[-1]) * DF / C0
    dn_out = -(-3 * W[:, 0] + 4 * W[:, 1] - W[:, 2]) / (2 * h[:, None])
    R = dn_out + 1j * k[None] * b * W[:, 0]
    rel = np.linalg.norm(R, axis=0) / np.maximum(np.linalg.norm(k[None] * W[:, 0], axis=0), 1e-300)
    return rel, {bn: float(np.mean(rel[lo:hi])) for bn, (lo, hi) in BANDS.items()}


def render():
    import torch

    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    dev = torch.device("cuda")
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    for name, ck in MODELS.items():
        model, renderer, *_ = load_model(Path(ck), dev)
        for r in rooms():
            dst = OUT / "cache" / "{}__{}.npz".format(name, r["path"].stem)
            if dst.exists():
                continue
            st = np.load(STEN / (r["path"].stem + ".npz"))
            P, _h = stencil(st)
            with h5py.File(r["path"]) as f:
                verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
                L, W = float(f.attrs["L"]), float(f.attrs["W"])
            z = build_cond_vector_2d("geom_token", L, W, [0.15] * len(verts), verts=verts, device=dev)
            lo, hi = torch.zeros(2, device=dev), torch.tensor([L, W], device=dev, dtype=torch.float32)
            R = torch.tensor(P.reshape(-1, 2), dtype=torch.float32, device=dev)
            S = torch.tensor([[0.5, 1.6]], device=dev)
            out = []
            with torch.no_grad():
                for s0 in range(0, len(R), 32):
                    rr = R[s0:s0 + 32]
                    out.append(renderer(model, rr, S.expand(len(rr), -1), lo, hi,
                                        z_s=z.unsqueeze(0).expand(len(rr), -1), n_out=601).cpu().numpy())
            np.savez_compressed(dst, W=np.concatenate(out).reshape(P.shape[0], 3, 601).astype(np.complex64))
            print("rendered", dst.name, flush=True)


def projected_at(F, keep, L, Wd, rx, P):
    """The projection's band-limited field (fitted on the lattice) at the stencil points P [n, 3, 2] -> [n, 3, 601]
    (bins < 25 Hz zero: they are never scored)."""
    import torch

    from aaf.eval.physical_projection import CUT, bandlimited_at, k_grid, project_lowpass
    dx, dy = (L - 2 * WALL_CLEAR) / 63, (Wd - 2 * WALL_CLEAR) / 63
    ix, iy = np.nonzero(keep)
    nom = np.stack([WALL_CLEAR + ix * dx, WALL_CLEAR + iy * dy], axis=1)
    assert np.abs(nom - rx).max() < 0.011, "receivers are not on the nominal lattice (+- 1 cm snap)"
    K = k_grid(64, 64, dx, dy, pad=4)
    uv = np.stack([(P[..., 0].ravel() - WALL_CLEAR) / dx, (P[..., 1].ravel() - WALL_CLEAR) / dy], axis=1)
    bins = np.arange(LO, 601)
    kc = CUT * 2 * np.pi * bins * DF / C0
    out = np.zeros((len(uv), 601), np.complex128)
    for s in range(0, len(bins), 64):
        sl = slice(s, s + 64)
        g = project_lowpass(F[:, bins[sl]].T, keep, K, kc[sl], return_grid=True)
        v = bandlimited_at(g, K, float(kc[sl].max()), uv)
        out[:, bins[sl]] = v.T.cpu().numpy()
        del g, v
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out.reshape(P.shape[0], 3, 601)


def score():
    from scipy.stats import spearmanr

    from scripts.p4_5_gate2_dense import on_lattice
    from scripts.p4_7_m1_modal import scores
    b = beta()
    caches = {m: {"test": np.load(G2.format(m)), "unseen": np.load(UN.format(m))} for m in MODELS}
    rows = []
    for r in rooms():
        st = np.load(STEN / (r["path"].stem + ".npz"))
        P, h = stencil(st)
        with h5py.File(r["path"]) as f:
            T = np.asarray(f["ism/H_complex"])[:, :601]
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
            keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
            L, Wd = float(f.attrs["L"]), float(f.attrs["W"])
        row = {"room": r["path"].name, "group": r["group"], "family": r["family"], "n_wall": int(len(P))}
        row["FDTD raw"] = residual(st["wall_H"].astype(np.complex128), h, b)[1]
        row["FDTD proj"] = residual(projected_at(T, keep, L, Wd, rx, P), h, b)[1]
        off = ~on_lattice(rx)
        for m in MODELS:
            row[m + " raw"] = residual(np.load(OUT / "cache" / "{}__{}.npz".format(m, r["path"].stem))["W"]
                                       .astype(np.complex128), h, b)[1]
            Pm = caches[m][r["group"]]["P%d" % r["cache_i"]][:, :601]
            row[m + " proj"] = residual(projected_at(Pm, keep, L, Wd, rx, P), h, b)[1]
            row[m + " r_25_100"] = scores(Pm[off], T[off])["25-100"]
        rows.append(row)
        print(r["path"].name[:40], {k: round(v["25-100"], 3) for k, v in row.items() if isinstance(v, dict)},
              flush=True)
    keys = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    sets = {"all31": lambda x: True, "in_family15": lambda x: x["group"] == "test",
            "unseen16": lambda x: x["group"] == "unseen", "U8": lambda x: x["family"] == "U",
            "C3_8": lambda x: x["family"] == "C3"}
    summ = {s: {k: {bn: float(np.mean([x[k][bn] for x in rows if f(x)])) for bn in BANDS} for k in keys}
            for s, f in sets.items()}
    gate = {}
    for m in MODELS:
        res = [x[m + " proj"]["25-100"] for x in rows]
        rr = [x[m + " r_25_100"] for x in rows]
        sp = float(spearmanr(res, rr).correlation)
        ratio = summ["unseen16"][m + " proj"]["25-100"] / summ["in_family15"][m + " proj"]["25-100"]
        gate[m] = {"c1_fdtd_truth_25_100": summ["all31"]["FDTD raw"]["25-100"],
                   "c1_fdtd_truth_25_100_max_room": float(max(x["FDTD raw"]["25-100"] for x in rows)),
                   "c1_holds": bool(summ["all31"]["FDTD raw"]["25-100"] <= 0.15),
                   "c2_ratio_unseen_over_in_family": float(ratio), "c2_holds": bool(ratio >= 1.5),
                   "c3_spearman_residual_vs_r": sp, "c3_holds": bool(sp <= -0.4)}
        gate[m]["all_hold"] = bool(gate[m]["c1_holds"] and gate[m]["c2_holds"] and gate[m]["c3_holds"])
    validity = {"fdtd_projected_25_100": summ["all31"]["FDTD proj"]["25-100"],
                "holds": bool(summ["all31"]["FDTD proj"]["25-100"] <= 0.15)}
    rep = {"definition": __doc__.split("\n\n")[0].strip(), "beta": b, "summary": summ, "gate": gate,
           "decision": {"model": "R0_120K", "launch_PL1": bool(gate["R0_120K"]["all_hold"] and validity["holds"])},
           "projection_extension_validity": validity, "rows": rows}
    GATES.mkdir(parents=True, exist_ok=True)
    (GATES / "robin.json").write_text(json.dumps(rep, indent=1))

    def f3(d):
        return " / ".join("{:.3f}".format(d[bn]) for bn in BANDS)
    Lm = ["## (ii) Robin boundary residual and gate G-BC (D180 0b(ii); D181 A9)", "",
          "rel = ||d_n,out p + j k beta p|| / ||k p|| at wall stencils every 5 cm (3-point one-sided normal derivative), "
          "beta = {:.4f}; band means 25-100 / 100-200 / 200-300 Hz. At this beta the Robin term is <= 4% of k|p|: the "
          "residual effectively tests near-Neumann behaviour (A9).".format(b), "",
          "| field | all 31 | in-family 15 | unseen 16 | U 8 | C3 8 |", "|---|---|---|---|---|---|"]
    for k in keys:
        Lm.append("| {} | {} | {} | {} | {} | {} |".format(k, f3(summ["all31"][k]), f3(summ["in_family15"][k]),
                                                       f3(summ["unseen16"][k]), f3(summ["U8"][k]), f3(summ["C3_8"][k])))
    g = gate["R0_120K"]
    Lm += ["", "**Validity of the projection's extension to the wall (A9):** FDTD truth, projected and extended the same "
           "way: {:.3f} at 25-100 Hz (must be <= 0.15) -> {}.".format(validity["fdtd_projected_25_100"],
                                                                      "valid" if validity["holds"] else "NOT valid"),
           "", "**Gate G-BC (R0 @120K; S2 in the JSON):** (1) FDTD truth {:.3f} (max room {:.3f}) <= 0.15: {}; "
           "(2) projected residual unseen / in-family = {:.2f} >= 1.5: {}; (3) Spearman(residual, 25-100 Hz r) = {:.2f} "
           "<= -0.4: {}. **PL1 {}.**".format(g["c1_fdtd_truth_25_100"], g["c1_fdtd_truth_25_100_max_room"],
                                           "holds" if g["c1_holds"] else "FAILS", g["c2_ratio_unseen_over_in_family"],
                                           "holds" if g["c2_holds"] else "FAILS", g["c3_spearman_residual_vs_r"],
                                           "holds" if g["c3_holds"] else "FAILS",
                                           "LAUNCHES" if rep["decision"]["launch_PL1"] else "is NOT launched"),
           "", "Per-room values: `outputs/p4_9/gates/robin.json`."]
    (GATES / "robin.md").write_text("\n".join(Lm) + "\n")
    print("\n".join(Lm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.render:
        render()
    if a.score:
        score()


if __name__ == "__main__":
    main()
