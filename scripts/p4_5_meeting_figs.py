"""P4-5 meeting figures (2026-10-14). Existing 60K checkpoints and dense test simulations only.

Every number is OFF the training lattice, on the 15-shape protocol (the frozen 15 test shapes at
dense 64x64 receivers, `data/track_p4_5_testdense`), from the renders already cached by
`p4_5_gate2_dense.py`. Per-panel metrics use the FROZEN metric functions (`p4_2_stage2_eval._db`,
`_pearson`, `_lsd`). S1b needs one render (the held-out-family U room, D111), done by
`--render-s1b` on a GPU. 2560x1440 PNG, white background, into outputs/p4_5/meeting/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path("outputs/p4_5/meeting")
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
MODELS = (("C250 @60K (baseline)", "P4-4_C250", "outputs/p4_4/p4_4_C250/ckpt_iter0060000.pt"),
          ("B5 @60K (DC head)", "B5_60K", "outputs/p4_5/arms/B5/ckpt_iter0060000.pt"))
FREQS = (50, 100, 200, 280)
DF = 0.5
FIG = dict(figsize=(16, 9), dpi=160)


def onlat(rx):
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def render_s1b():
    import torch

    from aaf.eval.p3_2_eval import load_model
    from scripts.build_p4_5_rooms import OUT as BOUT, family_test_rooms
    from scripts.p4_4_family_eval import _render
    rooms = dict(family_test_rooms())
    fam = "U" if "U" in rooms else "T"
    c = rooms[fam]
    with h5py.File(Path(BOUT["famdense"]) / c.filename) as f:
        T = np.asarray(f["ism/H_complex"])[:, :601]
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
    dev = torch.device("cuda")
    blob = {"T": T, "rx": rx, "family": fam, "label": c.label}
    for _lab, key, ck in MODELS:
        model, renderer, _cfg, _m, _it = load_model(Path(ck), dev)
        model.eval(); renderer.eval()
        blob["P_" + key] = _render(model, renderer, c, rx, 601, dev)
        del model, renderer
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "cache_s1b.npz", **blob)
    print("wrote", OUT / "cache_s1b.npz", c.label)


# ----------------------------------------------------------------------------------------------
def grid_of(h5path, L, W, values):
    from scripts.build_p4_3_demo_rooms import GRID_N, WALL_CLEAR
    with h5py.File(h5path) as f:
        keep = np.asarray(json.loads(f.attrs["lattice_keep"]), dtype=bool)
    g = np.full(GRID_N * GRID_N, np.nan)
    g[keep] = values
    xs = WALL_CLEAR + np.arange(GRID_N) * (L - 2 * WALL_CLEAR) / (GRID_N - 1)
    ys = WALL_CLEAR + np.arange(GRID_N) * (W - 2 * WALL_CLEAR) / (GRID_N - 1)
    return xs, ys, g.reshape(GRID_N, GRID_N)            # [ix, iy]


def outline(ax, verts):
    v = np.asarray(verts + [verts[0]])
    ax.plot(v[:, 0], v[:, 1], color="k", lw=1.2)


def field_figure(path, title, h5path, c, T, Ps, rx, src):
    import matplotlib.pyplot as plt
    from scripts.p4_2_stage2_eval import _db, _lsd, _pearson
    off = ~onlat(rx)
    fig, axes = plt.subplots(len(FREQS), 1 + len(Ps), **FIG)
    for i, hz in enumerate(FREQS):
        b = int(round(hz / DF))
        gt = _db(T[:, b])
        lo, hi = np.percentile(gt, [2, 98])
        cols = [("FDTD ground truth", T, None)] + [(lab, P, P) for lab, P in Ps]
        for j, (lab, H, P) in enumerate(cols):
            ax = axes[i, j]
            xs, ys, g = grid_of(h5path, c.L, c.W, _db(H[:, b]))
            im = ax.pcolormesh(xs, ys, g.T, cmap="viridis", vmin=lo, vmax=hi, shading="nearest")
            outline(ax, [list(p) for p in c.verts])
            ax.plot(*src, marker="*", color="red", ms=11, mec="k")
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


def error_figure(path, title, h5path, c, T, Ps, rx, src):
    import matplotlib.pyplot as plt
    from scripts.p4_2_stage2_eval import _db
    fig, axes = plt.subplots(len(FREQS), len(Ps), **FIG)
    for i, hz in enumerate(FREQS):
        b = int(round(hz / DF))
        for j, (lab, P) in enumerate(Ps):
            ax = axes[i, j]
            xs, ys, g = grid_of(h5path, c.L, c.W, np.abs(_db(P[:, b]) - _db(T[:, b])))
            im = ax.pcolormesh(xs, ys, g.T, cmap="Reds", vmin=0, vmax=15, shading="nearest")
            outline(ax, [list(p) for p in c.verts])
            ax.plot(*src, marker="*", color="k", ms=10)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(lab, fontsize=11)
            if j == 0:
                ax.set_ylabel("{} Hz".format(hz), fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="|dB error|")
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def colr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-s1b", action="store_true")
    a = ap.parse_args()
    if a.render_s1b:
        render_s1b()
        return 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aaf.data.shape_configs import SRC, configs_from_rows
    from scripts.p4_2_stage2_eval import _db
    OUT.mkdir(parents=True, exist_ok=True)
    test = configs_from_rows(json.load(open("configs/sweeps_2d_mat/p4_2_shapes_manifest.json"))["configs"], split="test")
    Z = {key: np.load(G2.format(key), allow_pickle=False) for _l, key, _c in MODELS}
    manifest = []

    # per-shape off-lattice band means (D105(d)'s statistic), both models
    def band_means(key):
        z = Z[key]; out = {"100-200": [], "200-300": [], "25-100": []}
        for i in range(15):
            P, T, rx = _db(z["P%d" % i][:, :601]), _db(z["T%d" % i][:, :601]), z["rx%d" % i]
            r = colr(P[~onlat(rx)], T[~onlat(rx)])
            out["25-100"].append(np.nanmean(r[50:200])); out["100-200"].append(np.nanmean(r[200:400]))
            out["200-300"].append(np.nanmean(r[400:601]))
        return {k: np.array(v) for k, v in out.items()}
    bm = {key: band_means(key) for _l, key, _c in MODELS}

    # ---- S1 selection rule, fixed before any field is drawn
    order = np.argsort(bm["B5_60K"]["200-300"])
    ia = int(order[(15 - 1) // 2])
    nseg = [len(c.verts) for c in test]
    cand = sorted(range(15), key=lambda i: (-nseg[i], test[i].shape_id))
    ib = cand[0] if cand[0] != ia else cand[1]
    sel = [("a", ia, "lower-median shape by B5 @60K off-lattice 200-300 Hz band-mean r (rank 8 of 15, r = {:.3f})".format(
        bm["B5_60K"]["200-300"][ia])),
           ("b", ib, "most wall segments ({}), tie-break lowest shape id{}".format(
               nseg[ib], "" if cand[0] != ia else " (the first pick coincided with (a); next-lowest id taken)"))]
    for tag, i, rule in sel:
        c = test[i]; z0 = Z["P4-4_C250"]
        T, rx = z0["T%d" % i][:, :601], z0["rx%d" % i]
        Ps = [(lab, Z[key]["P%d" % i][:, :601]) for lab, key, _c in MODELS]
        assert np.array_equal(Z["B5_60K"]["rx%d" % i], rx)
        h5 = Path("data/track_p4_5_testdense") / c.filename
        p = OUT / "S1_{}_shape{}_fields.png".format(tag, c.shape_id)
        field_figure(p, "Unseen geometry (shape id {}), dense grid, positions off the training lattice".format(c.shape_id),
                     h5, c, T, Ps, rx, SRC)
        manifest.append("| {} | S1 ({}): dB magnitude field maps, FDTD vs C250 @60K vs B5 @60K, rows 50/100/200/280 Hz (single 0.5 Hz bins), shared per-row limits = GT 2-98th pct; corner r/LSD off-lattice, frozen metric functions | {} ({}) |".format(
            p.name, tag, rule, c.label))
        print("wrote", p, flush=True)
        sel_c = (c, T, Ps, rx, h5)
        if tag == "a":
            S1a = sel_c
        else:
            S1b_shape = sel_c

    # ---- S1b: held-out shape family (U room, D111), if rendered
    cache = OUT / "cache_s1b.npz"
    if cache.exists():
        zb = np.load(cache, allow_pickle=False)
        from scripts.build_p4_5_rooms import OUT as BOUT, family_test_rooms
        rooms = dict(family_test_rooms()); fam = str(zb["family"]); c = rooms[fam]
        Ps = [(lab, zb["P_" + key]) for lab, key, _c in MODELS]
        p = OUT / "S1b_{}_family_heldout_fields.png".format(fam)
        field_figure(p, "Held-out shape family ({} room {}), dense grid, positions off the training lattice".format(fam, c.label),
                     Path(BOUT["famdense"]) / c.filename, c, zb["T"], Ps, zb["rx"], SRC)
        manifest.append("| {} | S1b: same layout, held-out shape FAMILY ({}): a {}-room, a family neither model trained on | D111's median-d_hat {} test room (rule fixed in D111) |".format(
            p.name, fam, fam, fam))
        print("wrote", p, flush=True)
    else:
        manifest.append("| (S1b not rendered) | held-out-family U room: GPU render not finished by the deadline | - |")

    # ---- S3: paired per-shape scatter
    fig, axes = plt.subplots(1, 2, **FIG)
    for ax, band in zip(axes, ("100-200", "200-300")):
        x, y = bm["P4-4_C250"][band], bm["B5_60K"][band]
        ax.scatter(x, y, s=90, color="#7A1F5C", edgecolor="k", zorder=3)
        lo, hi = min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], color="0.5", ls="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("C250 @60K (baseline): off-lattice band-mean r", fontsize=12)
        ax.set_ylabel("B5 @60K (DC head): off-lattice band-mean r", fontsize=12)
        ax.set_title("{} Hz: {}/15 shapes above the diagonal".format(band, int((y > x).sum())), fontsize=13)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-shape paired comparison, 15 frozen unseen test shapes, dense receivers off the training lattice", fontsize=13)
    p = OUT / "S3_paired_scatter.png"; fig.savefig(p, facecolor="white"); plt.close(fig)
    manifest.append("| {} | S3: per-shape off-lattice band-mean dB spatial r, C250 (x) vs B5 (y), 100-200 and 200-300 Hz | all 15 frozen test shapes (no selection) |".format(p.name))
    print("wrote", p, flush=True)

    # ---- S2: absolute dB error maps for the S1 shapes
    for tag, (c, T, Ps, rx, h5) in (("a", S1a), ("b", S1b_shape)):
        p = OUT / "S2_{}_shape{}_abs_error.png".format(tag, c.shape_id)
        error_figure(p, "|dB error| (shape id {}), dense grid; 0-15 dB shared scale".format(c.shape_id), h5, c, T, Ps, rx, SRC)
        manifest.append("| {} | S2 ({}): absolute dB error maps, C250 vs B5, shared 0-15 dB, rows 50/100/200/280 Hz | same shape as S1 ({}) |".format(p.name, tag, tag))
        print("wrote", p, flush=True)

    # ---- S4: radial wavenumber spectrum, 15 shapes, 100-300 Hz (D108's field_stats)
    try:
        from scripts.build_p4_3_demo_rooms import GRID_N, WALL_CLEAR
        from scripts.p4_5_kspace_diag import CUT, PAD, RADIAL_EDGES, field_stats
        acc = {"FDTD": [], "C250 @60K": [], "B5 @60K": []}
        for i, c in enumerate(test):
            with h5py.File(Path("data/track_p4_5_testdense") / c.filename) as f:
                keep = np.asarray(json.loads(f.attrs["lattice_keep"]), dtype=bool)
            rx = Z["P4-4_C250"]["rx%d" % i]
            dx = (c.L - 2 * WALL_CLEAR) / (GRID_N - 1); dy = (c.W - 2 * WALL_CLEAR) / (GRID_N - 1)
            n = GRID_N * PAD
            K = np.hypot(*np.meshgrid(2 * np.pi * np.fft.fftfreq(n, dx), 2 * np.pi * np.fft.fftfreq(n, dy), indexing="ij"))
            on = np.zeros(GRID_N * GRID_N, bool); on[np.flatnonzero(keep)] = onlat(rx); on = on.reshape(GRID_N, GRID_N)
            for lab, H in (("FDTD", Z["P4-4_C250"]["T%d" % i]), ("C250 @60K", Z["P4-4_C250"]["P%d" % i]), ("B5 @60K", Z["B5_60K"]["P%d" % i])):
                acc[lab].append(np.mean([field_stats(H[:, :601], keep, on, K, b)[1] for b in range(200, 601, 4)], axis=0))
        ctr = 0.5 * (RADIAL_EDGES[1:] + RADIAL_EDGES[:-1])
        fig, ax = plt.subplots(1, 1, **FIG)
        for lab, col, ls in (("FDTD", "#009E73", "--"), ("C250 @60K", "0.2", "-"), ("B5 @60K", "#7A1F5C", "-")):
            ax.semilogy(ctr, np.mean(acc[lab], axis=0), color=col, ls=ls, lw=3, label=lab)
        ax.axvline(1.0, color="k", lw=1.2); ax.text(1.02, ax.get_ylim()[1] * 0.5, "k0", fontsize=12)
        ax.axvline(CUT, color="#D55E00", ls=":", lw=1.5); ax.text(CUT + 0.03, ax.get_ylim()[1] * 0.2, "1.5 k0", fontsize=11, color="#D55E00")
        ax.set_xlabel("|k| / k0", fontsize=13); ax.set_ylabel("energy fraction per |k| bin (100-300 Hz mean)", fontsize=13)
        ax.set_title("Radial wavenumber spectrum, 100-300 Hz, 15 frozen test shapes: the Helmholtz ring at |k| = k0", fontsize=13)
        ax.legend(fontsize=12); ax.grid(alpha=0.3, which="both")
        p = OUT / "S4_kspace_radial.png"; fig.savefig(p, facecolor="white"); plt.close(fig)
        manifest.append("| {} | S4: radial wavenumber spectrum (D108's field_stats, complex field, 2x zero-pad), FDTD vs C250 vs B5, 100-300 Hz (every 4th bin), k0 and 1.5 k0 marked | all 15 frozen test shapes |".format(p.name))
        print("wrote", p, flush=True)
    except Exception as e:  # noqa: BLE001
        manifest.append("| (S4 failed: {}) | | |".format(e))

    # ---- S5: r against distance to the nearest training-lattice point
    try:
        from scripts.p4_5_lattice_diag import dist_to_lattice
        classes = [(0.0, 1e-6, "0 cm"), (0.019, 0.021, "2 cm"), (0.027, 0.029, "2.8 cm")]
        fig, axes = plt.subplots(1, 2, **FIG)
        for ax, (band, lo_b, hi_b) in zip(axes, (("25-100 Hz", 50, 200), ("200-300 Hz", 400, 601))):
            for lab, key, col in (("C250 @60K", "P4-4_C250", "0.2"), ("B5 @60K", "B5_60K", "#7A1F5C")):
                vals = []
                for lo_d, hi_d, _n in classes:
                    rs = []
                    for i in range(15):
                        z = Z[key]; rx = z["rx%d" % i]
                        d = dist_to_lattice(rx)
                        s = (d >= lo_d) & (d <= hi_d)
                        if s.sum() < 10:
                            continue
                        P, T = _db(z["P%d" % i][s, lo_b:hi_b]), _db(z["T%d" % i][s, lo_b:hi_b])
                        rs.append(np.nanmean(colr(P, T)))
                    vals.append(np.mean(rs))
                ax.plot([0, 2, 2.83], vals, "o-", color=col, lw=3, ms=10, label=lab)
            ax.set_xticks([0, 2, 2.83]); ax.set_xticklabels([c[2] for c in classes])
            ax.set_xlabel("distance to the nearest training-receiver lattice point", fontsize=12)
            ax.set_ylabel("dB spatial r (band mean)", fontsize=12); ax.set_title(band, fontsize=13)
            ax.legend(fontsize=12); ax.grid(alpha=0.3)
        fig.suptitle("Memorisation signature: accuracy AT vs 2 cm FROM the training lattice, 15 frozen test shapes", fontsize=13)
        p = OUT / "S5_lattice_distance.png"; fig.savefig(p, facecolor="white"); plt.close(fig)
        manifest.append("| {} | S5: band-mean dB r by distance class to the nearest 0.12+0.06k lattice point (0 / 2 / 2.8 cm), C250 vs B5, 25-100 and 200-300 Hz | all 15 frozen test shapes |".format(p.name))
        print("wrote", p, flush=True)
    except Exception as e:  # noqa: BLE001
        manifest.append("| (S5 failed: {}) | | |".format(e))

    hdr = ["# P4-5 meeting figures (2026-10-14)", "",
           "All panels: existing 60K checkpoints (C250 = `outputs/p4_4/p4_4_C250`, B5 = `outputs/p4_5/arms/B5`), existing dense FDTD test simulations, "
           "OFF the training lattice, 15-shape protocol unless stated. 2560x1440, white background. Generator: `scripts/p4_5_meeting_figs.py`.", "",
           "| file | what it shows | selection rule |", "|---|---|---|"]
    (OUT / "FIGURE_MANIFEST.md").write_text("\n".join(hdr + manifest) + "\n")
    print("wrote", OUT / "FIGURE_MANIFEST.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
