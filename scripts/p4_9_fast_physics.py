"""P4-9 X2 (D190): a fast-physics baseline for the GEOMETRY edit -- m1 re-solved on a coarser grid, sized for 0-300 Hz.

m1's operator (`scripts/p4_7_m1_modal.operator`, re-implemented here with a grid-spacing argument; the existing module is
untouched) and m1's formula (`aaf.eval.modal_prior.compute`: C_FIT, + i k eta damping, the zero mode by its exact
discrete record), on a grid of spacing h in {2, 4, 6, 8, 10} cm with every mode to f_max in {350, 450, 600} Hz. The
polygon is snapped by the operator's own grid fit (air = coarse nodes inside the polygon); beta is exact. Mode shapes
at the dense receivers and the source by bilinear interpolation of the coarse eigenvectors, renormalised over air
nodes (nearest air node if none of the four is air; counted). Normalisation sum mu psi^2 dA_h = 1 (continuous), so the
FDTD's own constant C_FIT (2 cm source cell) stays; the zero-mode term's dA cancels exactly.
Accuracy: one eigen-solve per (room, h) to 600 Hz, truncated to each f_max (the eigenpairs below f_max are the same
set; degenerate-pair bases do not change the field). Scored on the 31 dense rooms, off-lattice receivers, the frozen
scorer (`p4_7_m1_modal.scores`), plus a signed mean dB level offset per band (a wrong constant shows as a constant
offset). Parity gate: h = 2 cm, f_max = 600 Hz must reproduce `outputs/p4_9/physics_sensitivity.json`'s per-room
"physics exact" r (vs FDTD) within 2e-3 per band.
Timing (one CPU node, OMP / MKL / OPENBLAS = 4, the I1 protocol `p4_9_latency.timeit`): t_solve = operator build +
eigen-solve to f_max (a separate solve per f_max); t_field = interpolation + zero-mode record + modal sum for ONE
listener and for the room's full dense grid. Single listener = t_solve + t_field(1); full grid = t_solve + t_field(grid).
Rooms: shape 1000 (I1's room, where the network's geometry-edit times were measured), the largest in-family and the
largest unseen dense room (most receivers).
Scaling (D190(e), timing only, largest unseen room): f_max in {300, 600, 1000, 2000} Hz at h = c / (10 f_max), one run
per config in a child process, capped at 30 min.
  python scripts/p4_9_fast_physics.py --room I      accuracy, room I of the 31 -> outputs/p4_9/fastphys/rooms/
  python scripts/p4_9_fast_physics.py --timing      -> outputs/p4_9/fastphys/timing.json
  python scripts/p4_9_fast_physics.py --scaling     -> outputs/p4_9/fastphys/scaling.json
  python scripts/p4_9_fast_physics.py --aggregate   -> outputs/p4_9/fastphys/fastphys.{json,md}, meeting/F7_fast_physics.png
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

OUT = Path("outputs/p4_9/fastphys")
H_GRID = (0.02, 0.04, 0.06, 0.08, 0.10)
FMAX = (350.0, 450.0, 600.0)
SCALE_F = (300.0, 600.0, 1000.0, 2000.0)
BANDS = ("25-100", "100-200", "200-300")
BAND_BINS = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
ALPHA, C0, FS = 0.15, 343.0, 30720.0
DT = 1.0 / FS
SHAPE1000 = "L5.54_W5.28_d1.42_w1.52.h5"
NET_REF = "outputs/p4_9/physics_sensitivity.json"
LATENCY = "outputs/p4_9/latency"
FIG = Path("outputs/p4_9/meeting/F7_fast_physics.png")


# ------------------------------------------------------------------------------------------- physics
def operator(verts, L, W, h):
    """`scripts/p4_7_m1_modal.operator` with the grid spacing h in place of the FDTD's 0.02 m (same code otherwise)."""
    import scipy.sparse as sp

    import aaf.sim.fdtd_2d as F
    from aaf.sim.polygon_geom import polygon_contains
    L, W, verts = float(L), float(W), [tuple(map(float, v)) for v in verts]
    nx, dxx = F._fit_axis(L, h, "L")
    ny, dxy = F._fit_axis(W, h, "W")
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    inside = polygon_contains(verts, np.stack([ii * dxx, jj * dxy], axis=-1))
    extra = [{"type": "mask", "solid": ~inside, "alpha": ALPHA}] if (~inside).any() else None
    geom = F.build_geometry(L, W, [ALPHA] * 4, dx=h, extra_walls=extra)
    assert (geom.air == inside).all()
    coef = F.build_coefficients(geom, C0 * DT / dxx, C0 * DT / dxy)
    mu = coef["mu"]
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
        rows.append(me[m])
        cols.append(nb)
        vals.append(wd[m] / h2)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(me), len(me)))
    muv = mu[I, J]
    K = -(sp.diags(muv) @ A)
    Ks = 0.5 * (K + K.T)
    adm = np.where(geom.blocked, geom.adm, 0.0)
    D = 2.0 * ((adm[XM] + adm[XP]) / dxx + (adm[YM] + adm[YP]) / dxy)
    perim = sum(np.hypot(verts[(k + 1) % len(verts)][0] - verts[k][0], verts[(k + 1) % len(verts)][1] - verts[k][1])
                for k in range(len(verts)))
    return {"Ks": Ks, "mu": muv, "D": D[I, J], "idx": idx, "dxx": dxx, "dxy": dxy, "dA": dxx * dxy,
            "n_air": int(len(me)), "area_q": float(muv.sum() * dxx * dxy), "perimeter": perim}


def solve(verts, L, W, h, f_max):
    """operator + every eigenpair to f_max (m1's `eigen`) + eta_n. Returns (op, k2, psi, eta)."""
    from scripts.p4_7_m1_modal import eigen
    op = operator(verts, L, W, h)
    k2, psi, _t, _n = eigen(op, f_max)
    eta = ((op["mu"] * op["D"])[:, None] * psi ** 2).sum(0) * op["dA"]
    return op, k2, psi, eta


def interp(op, psi, pts):
    """bilinear interpolation of the node values psi [n_air, M] at points [P, 2], renormalised over air nodes; a point
    with no air node among its four falls back to the nearest air node. Returns (values [P, M], n_fallback)."""
    nxg, nyg = op["idx"].shape
    u, v = pts[:, 0] / op["dxx"], pts[:, 1] / op["dxy"]
    i0 = np.clip(np.floor(u).astype(int), 0, nxg - 2)
    j0 = np.clip(np.floor(v).astype(int), 0, nyg - 2)
    out = np.zeros((len(pts), psi.shape[1]))
    wsum = np.zeros(len(pts))
    for di in (0, 1):
        for dj in (0, 1):
            q = op["idx"][i0 + di, j0 + dj]
            wgt = np.clip(1 - np.abs(u - (i0 + di)), 0, 1) * np.clip(1 - np.abs(v - (j0 + dj)), 0, 1)
            wgt = np.where(q >= 0, wgt, 0.0)
            out += wgt[:, None] * psi[np.maximum(q, 0)]
            wsum += wgt
    bad = wsum < 1e-12
    if bad.any():
        I, J = np.nonzero(op["idx"] >= 0)
        xy = np.stack([I * op["dxx"], J * op["dxy"]], 1)
        for p in np.nonzero(bad)[0]:
            n = int(np.argmin(np.hypot(*(xy - pts[p]).T)))
            out[p] = psi[op["idx"][I[n], J[n]]]
            wsum[p] = 1.0
    return out / wsum[:, None], int(bad.sum())


def zero_record(eta0, g0, s):
    """`aaf.eval.modal_prior.zero_mode_record` as an IIR filter (the same recursion; checked equal in --room)."""
    from scipy.signal import lfilter
    b = C0 * DT * eta0 / 2.0
    c1, c2 = 2.0 / (1.0 + b), (1.0 - b) / (1.0 + b)
    return np.fft.rfft(lfilter([g0], [1.0, -c1, c2], s))[:601]


def field(op, k2, psi, eta, rx, src, f_max, sig, S):
    """m1's transfer function (modal_prior.compute) at receivers rx from the modes with f_n <= f_max."""
    from aaf.eval.modal_prior import C_FIT
    m = C0 * np.sqrt(k2) / (2 * np.pi) <= f_max
    m[0] = True
    pr, nfb = interp(op, psi[:, m], rx)
    ps, _ = interp(op, psi[:, m], np.asarray(src, float)[None])
    ps = ps[0]
    kk, e = k2[m], eta[m]
    f = np.arange(601) * 0.5
    w = 2 * np.pi * f / C0
    den = kk[1:, None] - w[None] ** 2 + 1j * w[None] * e[1:, None]
    den[:, 0] = 1.0
    G = pr[:, 1:] @ (ps[1:, None] / den)
    G = G * (S * np.exp(1j * 2 * np.pi * f * DT))[None]
    G[:, 0] = 0.0
    c_pred = op["dA"] / (C0 * DT) ** 2
    a0 = zero_record(float(e[0]), float(ps[0] * op["dA"]), sig)
    return C_FIT * G + pr[:, :1] * a0[None] * (C_FIT / c_pred), nfb


# ------------------------------------------------------------------------------------------- rooms
def rooms():
    from scripts.p4_9_fdtd_stencils import dense_rooms
    return dense_rooms()


def load(path):
    from scripts.p4_9_grounding import on_lattice
    with h5py.File(path) as f:
        r = {"T": np.asarray(f["ism/H_complex"])[:, :601], "rx": np.asarray(json.loads(f.attrs["receiver_pos"]), float),
             "src": np.asarray(json.loads(f.attrs["source_pos"]), float),
             "verts": [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])],
             "L": float(f.attrs["L"]), "W": float(f.attrs["W"]), "name": Path(path).name}
    r["off"] = ~on_lattice(r["rx"])
    return r


def level_offset(P, T):
    db = lambda x: 20 * np.log10(np.maximum(np.abs(x), 1e-30))  # noqa: E731
    return {"dlev_" + b: float(np.mean(db(P[:, lo:hi]) - db(T[:, lo:hi]))) for b, (lo, hi) in BAND_BINS.items()}


def run_room(ri):
    from aaf.eval.modal_prior import pulse, zero_mode_record
    from scripts.p4_7_m1_modal import scores
    g, fam, path = rooms()[ri]
    r = load(path)
    sig, S = pulse()
    ref = next(x for x in json.loads(Path(NET_REF).read_text())["rows"] if x["room"] == r["name"])["vs_fdtd"]
    row = {"room": r["name"], "group": g, "family": fam, "n_off": int(r["off"].sum()), "configs": {},
           "reference_physics_exact": {b: ref["physics exact"][b] for b in BANDS}}
    for h in H_GRID:
        t0 = time.time()
        op, k2, psi, eta = solve(r["verts"], r["L"], r["W"], h, max(FMAX))
        t_solve = time.time() - t0
        if h == H_GRID[0]:                     # the IIR zero-mode record equals m1's recursion
            g0 = float(interp(op, psi[:, :1], r["src"][None])[0][0, 0] * op["dA"])
            a, b = zero_record(float(eta[0]), g0, sig), zero_mode_record(float(eta[0]), g0, sig)
            row["zero_record_rel_dev"] = float(np.abs(a - b).max() / np.abs(b).max())
        for fm in FMAX:
            P, nfb = field(op, k2, psi, eta, r["rx"], r["src"], fm, sig, S)
            sc = scores(P[r["off"]], r["T"][r["off"]])
            sc.update(level_offset(P[r["off"]], r["T"][r["off"]]))
            sc.update({"n_modes": int((C0 * np.sqrt(k2) / (2 * np.pi) <= fm).sum()), "n_air": op["n_air"],
                       "n_rx_fallback": nfb, "solve_s_to_600": round(t_solve, 2)})
            row["configs"]["h{:.2f}_f{:.0f}".format(h, fm)] = sc
        print(r["name"][:40], "h", h, {fm: round(row["configs"]["h{:.2f}_f{:.0f}".format(h, fm)]["25-100"], 3)
                                      for fm in FMAX}, "solve", round(t_solve, 1), flush=True)
    par = row["configs"]["h0.02_f600"]
    row["parity_max_dev"] = float(max(abs(par[b] - ref["physics exact"][b]) for b in BANDS))
    (OUT / "rooms").mkdir(parents=True, exist_ok=True)
    (OUT / "rooms" / (Path(r["name"]).stem + ".json")).write_text(json.dumps(row, indent=1))
    print("parity", r["name"], row["parity_max_dev"], flush=True)


# ------------------------------------------------------------------------------------------- timing
def timing_rooms():
    out = {}
    best = {"test": (None, -1), "unseen": (None, -1)}
    for g, _fam, p in rooms():
        if p.name == SHAPE1000:
            out["shape1000"] = p
        with h5py.File(p) as f:
            n = len(json.loads(f.attrs["receiver_pos"]))
        if n > best[g][1]:
            best[g] = (p, n)
    out["largest_in_family"] = best["test"][0]
    out["largest_unseen"] = best["unseen"][0]
    return out


def run_timing():
    from aaf.eval.modal_prior import pulse
    from scripts.p4_9_latency import timeit
    cpu = subprocess.run(["bash", "-c", "lscpu | grep 'Model name'"], capture_output=True, text=True).stdout.strip()
    rep = {"node": platform.node(), "cpu": cpu, "threads": "OMP/MKL/OPENBLAS = 4", "rooms": {}}
    sig, S = pulse()
    for tag, path in timing_rooms().items():
        r = load(path)
        res = {"room": r["name"], "n_rx": int(len(r["rx"])), "configs": {}}
        one = r["rx"][len(r["rx"]) // 2:len(r["rx"]) // 2 + 1]
        for h in H_GRID:
            for fm in FMAX:
                hold = {}

                def do_solve():
                    hold["s"] = solve(r["verts"], r["L"], r["W"], h, fm)
                ts = timeit(do_solve)
                op, k2, psi, eta = hold["s"]
                t1 = timeit(lambda: field(op, k2, psi, eta, one, r["src"], fm, sig, S))
                tg = timeit(lambda: field(op, k2, psi, eta, r["rx"], r["src"], fm, sig, S))
                res["configs"]["h{:.2f}_f{:.0f}".format(h, fm)] = {
                    "solve": ts, "field_1rx": t1, "field_grid": tg, "n_modes": int(len(k2)), "n_air": op["n_air"],
                    "single_listener_s": ts["median_s"] + t1["median_s"], "full_grid_s": ts["median_s"] + tg["median_s"]}
                print(tag, h, fm, "solve {:.3f} 1rx {:.4f} grid {:.3f} modes {}".format(
                    ts["median_s"], t1["median_s"], tg["median_s"], len(k2)), flush=True)
        rep["rooms"][tag] = res
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "timing.json").write_text(json.dumps(rep, indent=1))
    print("wrote", OUT / "timing.json")


def _scale_child(path, h, fm, q):
    r = load(path)
    t0 = time.perf_counter()
    op, k2, psi, eta = solve(r["verts"], r["L"], r["W"], h, fm)
    q.put({"solve_s": time.perf_counter() - t0, "n_modes": int(len(k2)), "n_air": op["n_air"]})


def run_scaling():
    import multiprocessing as mp
    path = timing_rooms()["largest_unseen"]
    cpu = subprocess.run(["bash", "-c", "lscpu | grep 'Model name'"], capture_output=True, text=True).stdout.strip()
    rep = {"node": platform.node(), "cpu": cpu, "room": path.name, "cap_s": 1800, "runs": 1, "configs": {}}
    for fm in SCALE_F:
        h = C0 / (10 * fm)
        q = mp.Queue()
        p = mp.Process(target=_scale_child, args=(path, h, fm, q))
        t0 = time.perf_counter()
        p.start()
        p.join(1800)
        if p.is_alive():
            p.terminate()
            p.join()
            res = {"solve_s": "> 30 min", "h": h}
        elif p.exitcode != 0:
            res = {"solve_s": "failed (exit {})".format(p.exitcode), "h": h, "wall_s": time.perf_counter() - t0}
        else:
            res = dict(q.get(), h=h)
        rep["configs"]["f{:.0f}".format(fm)] = res
        print(fm, res, flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "scaling.json").write_text(json.dumps(rep, indent=1))


# ------------------------------------------------------------------------------------------- report
def aggregate():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "rooms").glob("*.json"))]
    assert len(rows) == 31, len(rows)
    cfgs = ["h{:.2f}_f{:.0f}".format(h, fm) for h in H_GRID for fm in FMAX]
    sets = {"unseen16": "unseen", "in_family15": "test"}
    acc = {}
    for s, g in sets.items():
        rr = [x for x in rows if x["group"] == g]
        acc[s] = {c: {k: float(np.mean([x["configs"][c][k] for x in rr]))
                      for k in list(BANDS) + ["lsd_" + b for b in BANDS] + ["dlev_" + b for b in BANDS]}
                  for c in cfgs}
        for c in cfgs:
            acc[s][c]["n_rx_fallback_total"] = int(sum(x["configs"][c]["n_rx_fallback"] for x in rr))
    parity = {"max_dev": float(max(x["parity_max_dev"] for x in rows)), "n_rooms": len(rows),
              "passes": bool(max(x["parity_max_dev"] for x in rows) < 2e-3),
              "zero_record_rel_dev_max": float(max(x.get("zero_record_rel_dev", 0.0) for x in rows))}
    net = json.loads(Path(NET_REF).read_text())["summary"]["vs_fdtd"]
    net_r = {s: {b: net[s]["R0_120K exact"][b] for b in BANDS} for s in sets}
    gpu = json.loads((Path(LATENCY) / "gpu.json").read_text())["rooms"]["shape1000"]["bins_601"]
    lat = json.loads(Path(LATENCY + ".json").read_text())
    net_t = {"single_listener": gpu["geometry_1rx"]["median_s"], "full_grid": gpu["geometry_grid"]["median_s"],
             "full_grid_largest_room_listener": lat["largest_room"]["net_grid_601"]}
    tim = json.loads((OUT / "timing.json").read_text())
    ts = tim["rooms"]["shape1000"]["configs"]
    verdict = {}
    for s in sets:
        verdict[s] = {}
        for use, key in (("single_listener", "single_listener_s"), ("full_grid", "full_grid_s")):
            dom = [c for c in cfgs if ts[c][key] < net_t[use] and all(acc[s][c][b] > net_r[s][b] for b in BANDS)]
            verdict[s][use] = {"network_s": net_t[use], "dominating_configs": dom,
                               "fastest_dominating": (min(dom, key=lambda c: ts[c][key]) if dom else None),
                               "fastest_dominating_s": (min(ts[c][key] for c in dom) if dom else None),
                               "any_dominates": bool(dom)}
    sc = json.loads((OUT / "scaling.json").read_text()) if (OUT / "scaling.json").exists() else None
    rep = {"definition": __doc__.split("\n\n")[0].strip(), "parity": parity, "accuracy": acc, "network_r": net_r,
           "network_times_s": net_t, "timing_node": {"node": tim["node"], "cpu": tim["cpu"]}, "timing": tim["rooms"],
           "verdict": verdict, "scaling": sc}
    (OUT / "fastphys.json").write_text(json.dumps(rep, indent=1))

    def f(x):
        return "{:.0f} ms".format(x * 1e3) if x < 1 else "{:.1f} s".format(x)
    L = ["# P4-9 X2 (D190): fast physics for the geometry edit", "",
         "m1 (the FDTD's own operator in modal form) re-solved on a coarser grid h with modes to f_max; polygon snapped "
         "by the grid; exact beta. Off-lattice r per band (25-100 / 100-200 / 200-300 Hz), frozen scorer. Times on {} "
         "({}), shape 1000, 4 threads, medians (I1 protocol); geometry-edit latency = operator + eigen-solve + field."
         .format(tim["node"], tim["cpu"].replace("Model name:", "").strip()), "",
         "**Parity (h = 2 cm, 600 Hz vs physics_sensitivity's m1 exact, 31 rooms): max |dr| = {:.1e} -> {}.**".format(
             parity["max_dev"], "PASSES" if parity["passes"] else "FAILS"), "",
         "| h | f_max | modes (shape 1000) | single listener | full grid ({} pts) | unseen 16 r | in-family 15 r | "
         "unseen LSD dB | unseen level offset dB |".format(tim["rooms"]["shape1000"]["n_rx"]),
         "|---|---|---|---|---|---|---|---|---|"]
    for c in cfgs:
        h, fm = c[1:5], c.split("_f")[1]
        u, i = acc["unseen16"][c], acc["in_family15"][c]
        L.append("| {} cm | {} Hz | {} | {} | {} | {} | {} | {} | {} |".format(
            int(round(float(h) * 100)), fm, ts[c]["n_modes"], f(ts[c]["single_listener_s"]), f(ts[c]["full_grid_s"]),
            " / ".join("{:.3f}".format(u[b]) for b in BANDS), " / ".join("{:.3f}".format(i[b]) for b in BANDS),
            " / ".join("{:.1f}".format(u["lsd_" + b]) for b in BANDS),
            " / ".join("{:+.1f}".format(u["dlev_" + b]) for b in BANDS)))
    L.append("| network R0 @120K | -- | -- | {} | {} | {} | {} | -- | -- |".format(
        f(net_t["single_listener"]), f(net_t["full_grid"]), " / ".join("{:.3f}".format(net_r["unseen16"][b]) for b in BANDS),
        " / ".join("{:.3f}".format(net_r["in_family15"][b]) for b in BANDS)))
    L += ["", "## D190(c) verdict: does any configuration dominate the network (faster AND higher r in all 3 bands)?", ""]
    for s in sets:
        for use in ("single_listener", "full_grid"):
            v = verdict[s][use]
            L.append("* **{} / {}** (network {}): {}".format(
                s, use.replace("_", " "), f(v["network_s"]),
                "YES -- {} of 15 configs; fastest {} at {}".format(len(v["dominating_configs"]), v["fastest_dominating"],
                                                                   f(v["fastest_dominating_s"]))
                if v["any_dominates"] else "no"))
    for tag, rr in tim["rooms"].items():
        if tag == "shape1000":
            continue
        L += ["", "Timing check on {} ({}, {} pts): ".format(tag, rr["room"], rr["n_rx"]) + "; ".join(
            "{} {} / {}".format(c, f(rr["configs"][c]["single_listener_s"]), f(rr["configs"][c]["full_grid_s"]))
            for c in cfgs)]
    if sc:
        L += ["", "## D190(e) scaling (timing only; {} on {}; h = c / (10 f_max); one run, 30 min cap)".format(
            sc["room"], sc["node"]), "", "| f_max | h | air nodes | modes | eigen-solve |", "|---|---|---|---|---|"]
        for k, v in sc["configs"].items():
            L.append("| {} Hz | {:.2f} cm | {} | {} | {} |".format(
                k[1:], v["h"] * 100, v.get("n_air", "--"), v.get("n_modes", "--"),
                f(v["solve_s"]) if isinstance(v["solve_s"], float) else v["solve_s"]))
    (OUT / "fastphys.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    figure(rep, cfgs)


def figure(rep, cfgs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
                         "xtick.color": "#555555", "ytick.color": "#555555"})
    ts = rep["timing"]["shape1000"]["configs"]
    acc, net_r, net_t = rep["accuracy"]["unseen16"], rep["network_r"]["unseen16"], rep["network_times_s"]
    marks = {"350": "o", "450": "s", "600": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.2), dpi=160, facecolor="white")
    for ax, b in zip(axes, ("25-100", "200-300")):
        ax.set_facecolor("white")
        ax.grid(True, color="#e6e6e6", linewidth=1)
        ax.set_axisbelow(True)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        for fm, mk in marks.items():
            pts = [(ts[c]["full_grid_s"], acc[c][b], c) for c in cfgs if c.endswith("_f" + fm)]
            xs, ys = zip(*[(p[0], p[1]) for p in pts])
            ax.plot(xs, ys, color="#1baf7a", linewidth=1, alpha=0.5, zorder=2)
            for x, y, c in pts:
                col = "#008300" if c.startswith("h0.02") else "#1baf7a"
                ax.scatter(x, y, s=110, marker=mk, color=col, edgecolor="white", linewidth=1.5, zorder=3)
        if b == "200-300":                     # in 25-100 Hz every grid sits at 0.97-0.98 and the labels would collide
            for c in cfgs:
                if c.endswith("_f450"):
                    ax.annotate("h = {} cm".format(int(round(float(c[1:5]) * 100))), (ts[c]["full_grid_s"], acc[c][b]),
                                xytext=(8, -16), textcoords="offset points", fontsize=11, color="#333333")
        else:
            ax.annotate("every grid h = 2-10 cm: r 0.972-0.979", (0.4, 0.975), xytext=(0, -26),
                        textcoords="offset points", fontsize=11, color="#333333")
        for use, mk, lab in (("single_listener", "o", "network, 1 listener"), ("full_grid", "D", "network, full grid")):
            ax.scatter(net_t[use], net_r[b], s=150, marker=mk, color="#2a78d6", edgecolor="white", linewidth=1.5,
                       zorder=4)
            ax.annotate(lab, (net_t[use], net_r[b]), xytext=(8, 8), textcoords="offset points", fontsize=12,
                        color="#333333")
        ax.set_xscale("log")
        ax.set_xlim(5e-3, 300)
        ax.set_ylim(min(0.3, net_r[b] - 0.05), 1.0)
        ax.set_xlabel("geometry-edit latency, s (log)")
        ax.set_ylabel("off-lattice r, mean of 16 unseen rooms")
        ax.set_title("{} Hz".format(b), loc="left", fontsize=16, color="#222222")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = [Line2D([], [], marker=mk, linestyle="", color="#888888", markersize=10, label="modes to {} Hz".format(fm))
          for fm, mk in marks.items()]
    hs += [Patch(color="#1baf7a", label="physics, coarse grid (4-10 cm)"),
           Patch(color="#008300", label="physics, 2 cm grid (m1 re-solve)"),
           Patch(color="#2a78d6", label="network R0 @120K")]
    axes[0].legend(handles=hs, loc="center right", frameon=False, fontsize=12)
    fig.suptitle("F7 -- after a geometry edit: m1 re-solved on a coarser grid vs the network", fontsize=16,
                 color="#222222", x=0.01, ha="left")
    fig.text(0.01, 0.935, "Physics: operator + eigen-solve + field on one CPU node (4 threads, {}), shape 1000, full dense "
             "grid (one listener is within ~0.1 s of it).\nNetwork: A5000 GPU (I1 timings). r: frozen off-lattice "
             "scorer, mean of the 16 unseen rooms; network r is R0 @120K's on the same rooms and receivers."
             .format(rep["timing_node"]["cpu"].replace("Model name:", "").strip().replace("Intel(R) Xeon(R) CPU ", "Xeon ")),
             fontsize=11.5, color="#555555", ha="left", va="top",
             linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, facecolor="white")
    print("wrote", FIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", type=int, default=None)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.room is not None:
        run_room(a.room)
    if a.timing:
        run_timing()
    if a.scaling:
        run_scaling()
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
