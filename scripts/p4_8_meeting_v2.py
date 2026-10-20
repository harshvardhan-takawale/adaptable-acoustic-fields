"""P4-8 meeting deck v2 (manager, 2026-10-19): field maps in P4-5's S1 layout and the 2 cm speckle close-up. Network models
only (no modal-prior or solver results). Outputs go to outputs/p4_8/meeting_v2/.

--fields   Three field-map figures in P4-5's S1 layout (scripts/p4_5_meeting_figs.field_figure): rows 50 / 100 / 200 / 280 Hz
           (single 0.5 Hz bins), columns "Ground truth (FDTD)" | "Start of phase (C250)" | "Last meeting (B5)" | "Now (R0)".
           RAW model output (no render-time projection); shared per-row colour limits = FDTD 2nd-98th percentile; viridis
           (as S1_a); room outline, source star, per-panel corner r and LSD OFF the training lattice with the FROZEN metric
           functions (scripts/p4_2_stage2_eval._db/_pearson/_lsd). 2560 x 1440, white background. Rooms, fixed by rule:
             (a) shape 1000 = 15-shape test index 0, the room of last meeting's S1_a;
             (b) the U room at R0 @120K's LOWER MEDIAN of 25-100 Hz off-lattice r within the frozen U set (8 rooms: the 4th
                 lowest, sorted index (8-1)//2 = 3), from outputs/p4_8/unseen/R0_120K.json;
             (c) the C3 room by the same rule within the C3 set.
--speckle  Speckle close-up on shape 1000's 2 cm FDTD grid (the d6 pipeline, scripts/p4_5_speckle_2cm.py), 280 and 200 Hz:
           columns FDTD | B5 raw | R0 raw | R0 + render-time band-limit (least-squares projection onto |k| <= 1.5 k0 on the
           2 cm grid, as p4_5_speckle_lp_d1.d1_2cm). Top row: full room with the inset window outlined; bottom row: a
           1.5 m x 1.5 m inset. INSET RULE: the 1.5 m window (75 x 75 nodes, fully inside the bounding box) with the largest
           high-pass energy of B5 - FDTD at that bin, sum over the window's air nodes of |D - LP(D)|^2 with D = B5 - FDTD and LP
           the same mask-aware least-squares low-pass at 1.5 k0. Per panel: 2 cm d1 (p4_5_speckle_2cm.pair_ratio, all in-room
           4-neighbour pairs, vs FDTD) and r (dB Pearson over the 2 cm nodes); the inset row reports them over the window.
--hp       Mask-aware fine-scale energy ratio (prediction / FDTD), D152(c)'s primary definition: on the 15 shapes' 64x64
           lattice, OFF-lattice nodes, energy of F - LP(F) with LP the least-squares low-pass at 1.5 k0 (and, as the
           companion, the Gaussian normalised-convolution low-pass), summed over each band's bins, per shape; mean with a
           10k shape bootstrap (seed 20261013). B5 raw and R0 raw are read from their speckle JSONs (same code); R0
           PROJECTED (render-time band-limit applied first) is computed here -- under the least-squares operator it is ~0 BY
           CONSTRUCTION (the projection removes exactly that band).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

OUT = Path("outputs/p4_8/meeting_v2")
FREQS = (50, 100, 200, 280)
DF = 0.5
FIG = dict(figsize=(16, 9), dpi=160)
SRC = (0.5, 1.6)
COLS = [("Start of phase (C250)", "C250_60K"), ("Last meeting (B5)", "B5_60K"), ("Now (R0)", "R0_120K")]
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
UN = "outputs/p4_8/cache/unseen_{}.npz"
BANDS = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}


def onlat(rx):
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


# ------------------------------------------------------------------------------------------------- (4) field maps
def field_figure(path, title, h5path, L, W, verts, T, Ps, rx):
    """P4-5's S1 layout (scripts/p4_5_meeting_figs.field_figure) with the deck's column titles; raw fields."""
    import matplotlib.pyplot as plt
    from scripts.p4_2_stage2_eval import _db, _lsd, _pearson
    from scripts.p4_5_meeting_figs import grid_of, outline
    off = ~onlat(rx)
    fig, axes = plt.subplots(len(FREQS), 1 + len(Ps), **FIG)
    for i, hz in enumerate(FREQS):
        b = int(round(hz / DF))
        gt = _db(T[:, b])
        lo, hi = np.percentile(gt, [2, 98])
        cols = [("Ground truth (FDTD)", T, None)] + [(lab, P, P) for lab, P in Ps]
        for j, (lab, H, P) in enumerate(cols):
            ax = axes[i, j]
            xs, ys, g = grid_of(h5path, L, W, _db(H[:, b]))
            im = ax.pcolormesh(xs, ys, g.T, cmap="viridis", vmin=lo, vmax=hi, shading="nearest")
            outline(ax, [list(p) for p in verts])
            ax.plot(*SRC, marker="*", color="red", ms=11, mec="k")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(lab, fontsize=11)
            if j == 0:
                ax.set_ylabel("{} Hz".format(hz), fontsize=11)
            if P is not None:
                r = _pearson(_db(P[off, b]), _db(T[off, b]))
                l = _lsd(P[off, b:b + 1], T[off, b:b + 1])
                ax.text(0.02, 0.03, "r = {:+.2f}\nLSD = {:.1f} dB".format(r, l), transform=ax.transAxes,
                        fontsize=9, va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.85, lw=0))
        fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.85, pad=0.01, label="dB |H|")
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def run_fields(man_rows):
    from aaf.data.shape_configs import configs_from_rows
    test = configs_from_rows(json.load(open("configs/sweeps_2d_mat/p4_2_shapes_manifest.json"))["configs"], split="test")
    assert test[0].shape_id == 1000
    un = json.load(open("configs/sweeps_2d_mat/p4_8_unseen_manifest.json"))["configs"]
    r0 = json.load(open("outputs/p4_8/unseen/R0_120K.json"))["rooms"]
    assert [r["room"] for r in r0] == [u["label"] for u in un]
    picks = [("a_shape1000", "test", 0,
              "shape id 1000 (15-shape test index 0), the room of last meeting's S1_a ({})".format(test[0].label))]
    for fam in ("U", "C3"):
        idx = [i for i, u in enumerate(un) if u["family"] == fam]
        order = sorted(idx, key=lambda i: (r0[i]["raw"]["25-100"], un[i]["label"]))
        k = order[(len(order) - 1) // 2]
        picks.append(("{}_{}".format("b" if fam == "U" else "c", fam), "unseen", k,
                      "{} room at R0 @120K's lower median of 25-100 Hz off-lattice r within the frozen {} set: rank {} of {} "
                      "ascending (r = {:.3f}); unseen-manifest index {} ({})".format(
                          fam, fam, (len(order) - 1) // 2 + 1, len(order), r0[k]["raw"]["25-100"], k, un[k]["label"])))
    for tag, kind, i, rule in picks:
        Z = {n: np.load((G2 if kind == "test" else UN).format(n)) for _l, n in COLS}
        T, rx = Z["R0_120K"]["T%d" % i][:, :601], Z["R0_120K"]["rx%d" % i]
        for _l, n in COLS:
            assert np.array_equal(Z[n]["rx%d" % i], rx), n
            assert np.array_equal(Z[n]["T%d" % i][:, :601], T), n
        if kind == "test":
            c = test[i]
            h5, L, W, verts = Path("data/track_p4_5_testdense") / c.filename, c.L, c.W, c.verts
            title = "In family: shape id 1000, dense grid, positions off the training lattice"
        else:
            u = un[i]
            h5, L, W, verts = Path("data/track_p4_8_unseen") / u["filename"], u["L"], u["W"], [tuple(v) for v in u["verts"]]
            title = "Unseen family {}: {} (frozen P4-8 set), dense grid, off the training lattice".format(u["family"], u["label"])
        Ps = [(lab, Z[n]["P%d" % i][:, :601]) for lab, n in COLS]
        p = OUT / "fields_{}.png".format(tag)
        field_figure(p, title, h5, L, W, verts, T, Ps, rx)
        man_rows.append((p.name, "S1-layout field maps, rows 50/100/200/280 Hz (single 0.5 Hz bins), columns FDTD / "
                         "C250 @60K / B5 @60K / R0 @120K; RAW (no projection); per-row limits FDTD 2-98th pct; corner r / "
                         "LSD off-lattice, frozen metric functions", rule))
        print("wrote", p, "|", rule, flush=True)


# ------------------------------------------------------------------------------------------------- (5) speckle close-up
def _grid2cm():
    from scripts.p4_5_speckle_2cm import CACHE as C2
    zt = np.load(C2 / "d6_fdtd_shape1000.npz")
    zb = np.load(C2 / "d6_B5_shape1000.npz")
    zr = np.load(C2 / "d6_R0_120K_shape1000.npz")
    i, j = zt["i"], zt["j"]
    for z in (zb, zr):
        assert np.array_equal(z["i"], i) and np.array_equal(z["j"], j)
    keep = np.zeros((i.max() + 1, j.max() + 1), bool); keep[i, j] = True
    order = np.lexsort((j, i))
    assert np.array_equal(np.flatnonzero(keep.ravel()), (i * keep.shape[1] + j)[order])
    n = 4 * max(keep.shape)
    kx = 2 * np.pi * np.fft.fftfreq(n, 0.02)
    K = np.hypot(*np.meshgrid(kx, kx, indexing="ij"))
    return zt, zb, zr, i[order], j[order], order, keep, K


def run_speckle(man_rows):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from aaf.eval.physical_projection import project_lowpass
    from scripts.p4_5_speckle_2cm import pair_ratio, shape_cfg, to_grid
    zt, zb, zr, io, jo, order, keep, K = _grid2cm()
    cfg = shape_cfg()
    W_N = int(round(1.5 / 0.02))                                       # 75 nodes = 1.5 m
    res = {}
    for hz, bn in ((280, 560), (200, 400)):
        k0 = 2 * math.pi * hz / 343.0
        lp = lambda v: project_lowpass(v[None, :], keep, K, np.array([1.5 * k0]), iters=1000)[0]
        T = zt["H"][order, bn]; B = zb["H"][order, bn]; R = zr["H"][order, bn]
        Rp = lp(R)
        D = B - T
        hpE = to_grid(io, jo, np.abs(D - lp(D)) ** 2)
        e = np.nan_to_num(hpE)
        cs = np.pad(e.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        box = cs[W_N:, W_N:] - cs[:-W_N, W_N:] - cs[W_N:, :-W_N] + cs[:-W_N, :-W_N]
        bi, bj = np.unravel_index(np.argmax(box), box.shape)              # window rows [bi, bi+75), cols [bj, bj+75)
        db = lambda v: 20 * np.log10(np.maximum(np.abs(v), 1e-30))
        gT = to_grid(io, jo, db(T))
        panels = [("FDTD", gT), ("B5 raw", to_grid(io, jo, db(B))), ("R0 raw", to_grid(io, jo, db(R))),
                  ("R0 + render-time band-limit", to_grid(io, jo, db(Rp)))]
        lo, hi = np.nanpercentile(gT, [2, 98])
        x0, y0 = bi * 0.02, bj * 0.02
        fig, axes = plt.subplots(2, 4, **FIG)
        stats = {}
        for c, (lab, g) in enumerate(panels):
            win = (slice(bi, bi + W_N), slice(bj, bj + W_N))
            full = {"d1": pair_ratio(g, gT)} if c else {}
            sub = {"d1": pair_ratio(g[win], gT[win])} if c else {}
            if c:
                ok = np.isfinite(g) & np.isfinite(gT)
                full["r"] = float(np.corrcoef(g[ok], gT[ok])[0, 1])
                okw = ok[win]
                sub["r"] = float(np.corrcoef(g[win][okw], gT[win][okw])[0, 1])
            stats[lab] = {"full_room": full, "inset": sub}
            for row, (gg, ext, ttl, st) in enumerate(((g, (0, g.shape[0] * 0.02, 0, g.shape[1] * 0.02), lab, full),
                                                      (g[win], (x0, x0 + 1.5, y0, y0 + 1.5), lab + " (inset)", sub))):
                ax = axes[row, c]
                im = ax.imshow(gg.T, origin="lower", extent=ext, cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
                if row == 0:
                    v = np.asarray([list(p) for p in cfg.verts] + [list(cfg.verts[0])])
                    ax.plot(v[:, 0], v[:, 1], color="k", lw=1.0)
                    ax.plot(*SRC, marker="*", color="red", ms=10, mec="k")
                    ax.add_patch(Rectangle((x0, y0), 1.5, 1.5, fill=False, ec="white", lw=1.5))
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(ttl, fontsize=10)
                if st:
                    ax.text(0.02, 0.03, "2 cm d1 = {:.2f}\nr = {:+.2f}".format(st["d1"], st["r"]), transform=ax.transAxes,
                            fontsize=9, va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.85, lw=0))
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01, label="dB |H|")
        fig.suptitle("Speckle close-up, shape id 1000 on the 2 cm FDTD grid, {} Hz (single 0.5 Hz bin). Inset: the 1.5 m window "
                     "with the largest high-pass energy of B5 - FDTD".format(hz), fontsize=12)
        p = OUT / "speckle_2cm_{}Hz.png".format(hz)
        fig.savefig(p, facecolor="white")
        plt.close(fig)
        res["{} Hz".format(hz)] = {"inset_origin_m": [x0, y0], "inset_size_m": 1.5, "panels": stats,
                                   "inset_hp_energy_B5_minus_FDTD": float(box[bi, bj]),
                                   "hp_energy_frac_R0_projected": float(np.sum(np.abs(Rp - lp(Rp)) ** 2) / np.sum(np.abs(Rp) ** 2))}
        man_rows.append((p.name, "speckle close-up at {} Hz on shape 1000's 2 cm grid (61,874 nodes): FDTD | B5 raw | R0 raw | R0 + "
                         "render-time band-limit; top full room, bottom 1.5 m inset; per-panel 2 cm d1 and dB r (inset row: over "
                         "the window)".format(hz),
                         "inset = argmax over 1.5 m windows inside the bounding box of the summed |hp(B5 - FDTD)|^2, hp = minus the "
                         "least-squares low-pass at 1.5 k0; origin ({:.2f}, {:.2f}) m".format(x0, y0)))
        print("wrote", p, json.dumps(res["{} Hz".format(hz)]["panels"]), flush=True)
    (OUT / "speckle_2cm.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------- fine-scale energy ratio
def boot(v):
    v = np.asarray(v, float)
    idx = np.random.default_rng(20261013).integers(0, len(v), (10000, len(v)))
    m = v[idx].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def run_hp():
    from scripts.p4_5_speckle import lowpass_all, shape_geometry
    tc = np.load("outputs/p4_5/speckle/cache/T_lowpass.npz")
    z = np.load(G2.format("R0_120K"))
    per = {m: {b: [] for b in BANDS} for m in ("ls", "nc")}
    i = 0
    while "P%d" % i in z.files:
        P, T, rx = z["P%d" % i][:, :601], z["T%d" % i][:, :601], z["rx%d" % i]
        keep, K = shape_geometry(i, rx)
        off = ~onlat(rx)
        Pp, _ = lowpass_all(P, keep, K, "ls")                           # the render-time band-limit (bins >= 50)
        for m in ("ls", "nc"):
            LT = tc["T_%s_%d" % (m, i)]
            LPp, _ = lowpass_all(Pp, keep, K, m)
            eP = (np.abs((Pp - LPp)[off]) ** 2).sum(0)
            eT = (np.abs((T - LT)[off]) ** 2).sum(0)
            for b, (lo, hi) in BANDS.items():
                per[m][b].append(float(eP[lo:hi].sum() / eT[lo:hi].sum()))
        print("shape", i, {m: round(per[m]["200-300"][-1], 4) for m in per}, flush=True)
        i += 1
    rep = {"definition": "mask-aware fine-scale energy ratio (prediction / FDTD), D152(c): 64x64 lattice, off-lattice nodes, "
                         "|F - LP(F)|^2 summed over each band's bins, per shape; mean [95% CI], 10k shape bootstrap",
           "operators": {"ls": "least-squares low-pass at 1.5 k0 (D152 primary)", "nc": "Gaussian normalised convolution "
                         "(half power at 1.5 k0; companion)"}, "rows": {}}
    for name, lab in (("B5_60K", "B5 raw"), ("R0_120K", "R0 raw")):
        d = json.load(open("outputs/p4_5/speckle/{}.json".format(name)))
        rep["rows"][lab] = {m: {b: d["summary"]["d2_%s_hp_ratio" % m][b] for b in BANDS} for m in ("ls", "nc")}
        rep["rows"][lab]["source"] = "outputs/p4_5/speckle/{}.json (same code)".format(name)
    rep["rows"]["R0 projected"] = {m: {b: boot(per[m][b]) for b in BANDS} for m in ("ls", "nc")}
    rep["rows"]["R0 projected"]["note"] = "ls is ~0 BY CONSTRUCTION: the render-time band-limit removes exactly |k| > 1.5 k0"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fine_scale_energy.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: {m: v[m]["200-300"] for m in ("ls", "nc")} for k, v in rep["rows"].items()}, indent=1))


def main():
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", action="store_true")
    ap.add_argument("--speckle", action="store_true")
    ap.add_argument("--hp", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    if a.fields:
        run_fields(rows)
    if a.speckle:
        run_speckle(rows)
    if a.hp:
        run_hp()
    if rows:
        p = OUT / "manifest_rows_{}.json".format("fields" if a.fields else "speckle")
        p.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
