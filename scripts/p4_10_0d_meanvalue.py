"""P4-10 0d (D195 spec 0d / 2e, D197 A10 / A13, D201 Q30): calibration of the mean-value identity for R1-P. NO training.

If p solves Helmholtz in a disk of radius r, the circle mean satisfies M_r[p](x) = J0(k r) p(x). Residuals, at every
sampled bin (3 per band: 40 60 85 | 120 150 180 | 215 250 285 Hz), over centres x drawn uniformly in the polygon at
>= r_max + 0.05 m from every wall and >= r_max + 0.3 m from the source, on 8 points under a random rotation:
  single-radius  R  = M_r1[p] - J0(k r1) p
  two-radius     R2 = (M_r1 - p)(J0(k r2) - 1) - (M_r2 - p)(J0(k r1) - 1),  r2 = 0.6 r1 (cancels constants, spec 2e)
  r1 = min(1.2 / k, 0.35 m). Normalised (D201 / spec): RMS(R) / RMS(|p| at the centres), per room and bin, then the mean over bins.
Fields on the 15 in-family rooms: FDTD truth (recorded at stencil nodes, 4x4 cubic interpolation on the 2 cm grid,
<= 5000 nodes per run, A10), R0 @120K RAW (rendered at the points), R0 BAND-LIMITED (D160 projection of R0's lattice
field, its band-limited Fourier series evaluated at the points).
Gates (spec 0d, read on the form chosen by the code check): truth <= 0.05 in every band (else the two-radius form is used
and re-checked); R0 band-limited <= 2 x truth; R0 raw >= 3 x R0 band-limited with a paired CI above zero, in every band.
  --fdtd --idx I   CPU: record truth at the stencil nodes of in-family room I -> outputs/p4_10/r1p_0d/fdtd/<room>.npz
  --render         GPU: R0 raw at the points -> outputs/p4_10/r1p_0d/r0/<room>.npz (band-limited evaluated there too)
  --score          CPU: outputs/p4_10/r1p_0d/report.json + .md
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import aaf.sim.fdtd_2d as F

DF, C0 = 0.5, 343.0
FREQS = {"25-100": (40.0, 60.0, 85.0), "100-200": (120.0, 150.0, 180.0), "200-300": (215.0, 250.0, 285.0)}
NC, NPT, SEED = 8, 8, 20261125
OUT = Path("outputs/p4_10/r1p_0d")
CK = "outputs/p4_8/arms/R0/ckpt_iter0120000.pt"
N_BOOT, BOOT_SEED = 10000, 20261013


def radii(f):
    k = 2 * np.pi * f / C0
    r1 = min(1.2 / k, 0.35)
    return k, r1, 0.6 * r1


def distinct_sets():
    """distinct r1 over the sampled frequencies, ascending (a set = the points for one r1: centre + 8 at r1 + 8 at r2)"""
    return sorted({round(radii(f)[1], 6) for fs in FREQS.values() for f in fs})


def poly_dist(p, verts):
    v = np.asarray(verts, float)
    a, b = v, np.roll(v, -1, 0)
    ab = b - a
    t = np.clip(((p[:, None, :] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(p[:, None, :] - (a + t[..., None] * ab), axis=-1).min(1)


def inside(p, verts):
    from aaf.sim.polygon_geom import polygon_contains
    return np.asarray(polygon_contains(verts, p), bool)


def room_points(r, room_i, sets):
    """[NC, S, 17, 2] point coordinates: per centre and set, the centre, NPT at r1, NPT at r2 (random rotation per set)"""
    rng = np.random.default_rng([SEED, room_i])
    rmax = max(sets)
    L, W = r["L"], r["W"]
    src = np.asarray(r["src"])
    P = []
    while len(P) < NC:
        c = rng.uniform([0, 0], [L, W], (200, 2))
        ok = inside(c, r["verts"]) & (poly_dist(c, r["verts"]) >= rmax + 0.05) & (np.linalg.norm(c - src, axis=1) >= rmax + 0.3)
        P += list(c[ok])
    ctr = np.array(P[:NC])
    ang = rng.uniform(0, 2 * np.pi, (NC, len(sets), 2))
    th = np.arange(NPT) * 2 * np.pi / NPT
    out = np.zeros((NC, len(sets), 1 + 2 * NPT, 2))
    for i in range(NC):
        for s, r1 in enumerate(sets):
            out[i, s, 0] = ctr[i]
            for j, rr in enumerate((r1, 0.6 * r1)):
                a = th + ang[i, s, j]
                out[i, s, 1 + j * NPT:1 + (j + 1) * NPT] = ctr[i] + rr * np.stack([np.cos(a), np.sin(a)], 1)
    return out


def interp_weights(P, dxx, dxy):
    """4-point Lagrange in each axis: P [M, 2] -> node indices [M, 16, 2], weights [M, 16]"""
    u = P / np.array([dxx, dxy])
    i0 = np.floor(u).astype(int)
    t = u - i0
    nodes = np.array([-1, 0, 1, 2])
    w1 = lambda x: np.stack([np.prod([(x - o) / (n - o) for o in nodes if o != n], axis=0) for n in nodes], -1)  # noqa: E731
    wx, wy = w1(t[:, 0]), w1(t[:, 1])
    ij = np.stack([np.repeat(i0[:, None, 0] + nodes[None], 4, 1), np.tile(i0[:, None, 1] + nodes[None], (1, 4))], -1).reshape(len(P), 16, 2)
    w = (wx[:, :, None] * wy[:, None, :]).reshape(len(P), 16)
    return ij, w


def truth_points(idx):
    from scripts.p4_9_fdtd_stencils import MAX_NODES, N_CHECK, dense_rooms, geometry, load_room, simulate
    rooms = [p for g, fam, p in dense_rooms() if g == "test"]
    p = rooms[idx]
    r = load_room(p)
    geom, extra, dxx, dxy = geometry(r)
    sets = distinct_sets()
    P = room_points(r, idx, sets)
    ij, w = interp_weights(P.reshape(-1, 2), dxx, dxy)
    uniq, inv = np.unique(ij.reshape(-1, 2), axis=0, return_inverse=True)
    assert geom.air[uniq[:, 0], uniq[:, 1]].all(), "an interpolation node is not air"
    print(p.stem, "points", P.shape, "unique nodes", len(uniq), "cap", MAX_NODES - N_CHECK, flush=True)
    Hn, err = simulate(r, extra, uniq, dxx, dxy)
    Hp = (Hn[inv.reshape(-1)].reshape(len(ij), 16, 601) * w[:, :, None]).sum(1)
    OUT.joinpath("fdtd").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "fdtd" / (p.stem + ".npz"), P=P, H=Hp.reshape(P.shape[:3] + (601,)).astype(np.complex64), sets=np.array(sets),
                        check_err=err)
    print("wrote", p.stem, "check err {:.1e}".format(err), flush=True)
    del Hn
    gc.collect()


def band_limited_at(F_lat, keep, L, Wd, rx, uv_m):
    """R0's lattice field -> its D160 band-limited projection evaluated at points uv_m [M, 2] (metres); bins < 25 Hz are zero"""
    import torch

    from aaf.eval.physical_projection import CUT, bandlimited_at, k_grid, project_lowpass
    WC = 0.12
    dx, dy = (L - 2 * WC) / 63, (Wd - 2 * WC) / 63
    K = k_grid(64, 64, dx, dy, pad=4)
    uv = np.stack([(uv_m[:, 0] - WC) / dx, (uv_m[:, 1] - WC) / dy], 1)
    out = np.zeros((len(uv), 601), np.complex128)
    bins = np.arange(50, 601)
    kc = CUT * 2 * np.pi * bins * DF / C0
    for s in range(0, len(bins), 64):
        sl = slice(s, s + 64)
        g = project_lowpass(F_lat[:, bins[sl]].T, keep, K, kc[sl], return_grid=True)
        out[:, bins[sl]] = bandlimited_at(g, K, float(kc[sl].max()), uv).T.cpu().numpy()
        del g
        torch.cuda.empty_cache()
    return out


def render():
    import h5py
    import torch

    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    from scripts.p4_9_dir_zeroshot import dense_rooms as dzr
    dev = torch.device("cuda")
    model, renderer, *_ = load_model(Path(CK), dev)
    lat = np.load("outputs/p4_5/gate2_dense/cache/R0_120K__track_p4_5_testdense.npz")
    (OUT / "r0").mkdir(parents=True, exist_ok=True)
    for r in [x for x in dzr() if x["group"] == "test"]:
        st = np.load(OUT / "fdtd" / (r["path"].stem + ".npz"))
        P = st["P"]
        M = P.reshape(-1, 2)
        with h5py.File(r["path"]) as f:
            verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
            L, W = float(f.attrs["L"]), float(f.attrs["W"])
            keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
        z = build_cond_vector_2d("geom_token", L, W, [0.15] * len(verts), verts=verts, device=dev)
        lo, hi = torch.zeros(2, device=dev), torch.tensor([L, W], device=dev, dtype=torch.float32)
        R = torch.tensor(M, dtype=torch.float32, device=dev)
        S = torch.tensor([[0.5, 1.6]], device=dev)
        o = []
        with torch.no_grad():
            for s0 in range(0, len(R), 32):
                rr = R[s0:s0 + 32]
                o.append(renderer(model, rr, S.expand(len(rr), -1), lo, hi, z_s=z.unsqueeze(0).expand(len(rr), -1), n_out=601).cpu().numpy())
        raw = np.concatenate(o)
        proj = band_limited_at(lat["P%d" % r["cache_i"]][:, :601], keep, L, W, rx, M)
        np.savez_compressed(OUT / "r0" / (r["path"].stem + ".npz"), raw=raw.reshape(P.shape[:3] + (601,)).astype(np.complex64),
                            proj=proj.reshape(P.shape[:3] + (601,)).astype(np.complex64))
        print("rendered", r["path"].stem, flush=True)


def residuals(X, sets):
    """X [NC, S, 17, 601] -> per band: single-radius and two-radius normalised residual (per centre-set collapsed over centres)
    and the plain |p| rms; the radius at each sampled bin decides which set it uses."""
    from scipy.special import j0
    out = {}
    for bn, fs in FREQS.items():
        single, two = [], []
        for f in fs:
            k, r1, r2 = radii(f)
            s = int(np.argmin(np.abs(np.asarray(sets) - r1)))
            b = int(round(f / DF))
            p = X[:, s, 0, b]
            m1 = X[:, s, 1:1 + NPT, b].mean(1)
            m2 = X[:, s, 1 + NPT:, b].mean(1)
            den = np.sqrt(np.mean(np.abs(p) ** 2))
            single.append(np.sqrt(np.mean(np.abs(m1 - j0(k * r1) * p) ** 2)) / max(den, 1e-30))
            R2 = (m1 - p) * (j0(k * r2) - 1) - (m2 - p) * (j0(k * r1) - 1)
            two.append(np.sqrt(np.mean(np.abs(R2) ** 2)) / max(den, 1e-30))
        out[bn] = {"single": float(np.mean(single)), "two": float(np.mean(two))}
    return out


def _boot(v, seed=BOOT_SEED, n=N_BOOT):
    v = np.asarray(v, float)
    m = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def score():
    from scripts.p4_9_dir_zeroshot import dense_rooms as dzr
    rows = []
    for r in [x for x in dzr() if x["group"] == "test"]:
        f, c = OUT / "fdtd" / (r["path"].stem + ".npz"), OUT / "r0" / (r["path"].stem + ".npz")
        if not f.exists() or not c.exists():
            print("missing", r["path"].stem)
            continue
        t, o = np.load(f), np.load(c)
        sets = t["sets"]
        rows.append({"room": r["path"].stem, "check_err": float(t["check_err"]),
                     "truth": residuals(t["H"], sets), "raw": residuals(o["raw"], sets), "proj": residuals(o["proj"], sets)})
    rep = {"definition": __doc__.split("\n\n")[0], "n_rooms": len(rows), "form": {}, "rooms": rows}
    for form in ("single", "two"):
        rep["form"][form] = {}
        for b in FREQS:
            g = lambda k: np.array([x[k][b][form] for x in rows])  # noqa: E731
            tr, rw, pj = g("truth"), g("raw"), g("proj")
            rep["form"][form][b] = {"truth_mean": _boot(tr), "truth_room_max": float(tr.max()), "raw_mean": _boot(rw), "proj_mean": _boot(pj),
                                    "proj_over_truth": float(pj.mean() / tr.mean()), "raw_over_proj": float(rw.mean() / pj.mean()),
                                    "raw_minus_3proj_ci": _boot(rw - 3 * pj), "raw_minus_proj_ci": _boot(rw - pj)}
    code_check_single = all(v["truth_mean"][0] <= 0.05 for v in rep["form"]["single"].values())
    form = "single" if code_check_single else "two"
    rep["code_check"] = {"single_truth_le_0.05_every_band": code_check_single, "form_used": form,
                         "truth_le_0.05_every_band_in_form_used": all(v["truth_mean"][0] <= 0.05 for v in rep["form"][form].values())}
    sel = rep["form"][form]
    rep["gates"] = {"blind_to_content": {b: bool(v["proj_mean"][0] <= 2 * v["truth_mean"][0]) for b, v in sel.items()},
                    "something_to_remove": {b: bool(v["raw_over_proj"] >= 3 and v["raw_minus_proj_ci"][1] > 0) for b, v in sel.items()},
                    "code_check": rep["code_check"]["truth_le_0.05_every_band_in_form_used"]}
    rep["passes"] = bool(rep["gates"]["code_check"] and all(rep["gates"]["blind_to_content"].values()) and all(rep["gates"]["something_to_remove"].values()))
    L = ["# P4-10 0d (R1-P calibration): mean-value residual on truth, R0 raw and R0 band-limited", "",
         "{} in-family rooms; 8 centres per room, 8 points per circle; normalised RMS residual per bin (3 bins per band), mean over bins; "
         "form used by the gate: {}".format(len(rows), form), "", "| form | band | truth mean (max room) | R0 raw | R0 band-limited | raw / bl | bl / truth | raw - 3 bl [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    for fm in ("single", "two"):
        for b, v in rep["form"][fm].items():
            L.append("| {} | {} | {:.4f} ({:.4f}) | {:.4f} | {:.4f} | {:.2f} | {:.2f} | {:+.4f} [{:+.4f}, {:+.4f}] |".format(
                fm, b, v["truth_mean"][0], v["truth_room_max"], v["raw_mean"][0], v["proj_mean"][0], v["raw_over_proj"], v["proj_over_truth"], *v["raw_minus_3proj_ci"]))
    L += ["", "Gates (spec 0d): code check {}; R0 band-limited <= 2x truth {}; R0 raw >= 3x band-limited (and CI of raw - bl above zero) {} -> R1-P launch {}.".format(
        rep["gates"]["code_check"], rep["gates"]["blind_to_content"], rep["gates"]["something_to_remove"], "ALLOWED" if rep["passes"] else "NOT allowed")]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    for k in ("--fdtd", "--render", "--score"):
        ap.add_argument(k, action="store_true")
    ap.add_argument("--idx", type=int, default=None)
    a = ap.parse_args()
    if a.fdtd:
        truth_points(a.idx)
    elif a.render:
        render()
    elif a.score:
        score()


if __name__ == "__main__":
    main()
