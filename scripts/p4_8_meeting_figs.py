"""P4-8 meeting figures (D174(h)), generated from caches -- network models only (no modal-prior / solver results).

Models: G1 @60K, GX @60K, R0 @120K, the best wave-1 arm (+ wave 2 if run). "Best" is fixed by rule: among the
wave-1 finals whose pre-registered test HOLDS (outputs/p4_8/primary_table.json), the largest dHF + dUF(25-100) vs
GX @60K; if none holds, the largest over all finals, labelled "(no arm passed)".
  F1  in-family off-lattice r vs frequency, 25-300 Hz (15 frozen shapes; per-bin column Pearson of dB, mean over
      shapes, 11-bin moving average)
  F2  the same on the frozen unseen-family set, U and C3 in separate panels
  F3  S1-layout field maps for three rooms: the lower-median in-family test shape and the lower-median U and C3
      rooms (all by G1 @60K's off-lattice r: 200-300 Hz in family, 25-100 Hz unseen), rows 50 / 100 / 200 / 280 Hz,
      columns FDTD | G1 | R0 | best; model panels carry the render-time physical projection (POST-PROCESSING, labelled)
  F4  per-room paired scatter on the unseen set, best arm vs G1, one panel per band, diagonal and count above it
Colours: categorical slots in fixed order per MODEL (validated palette; adjacent CVD dE 9.1, normal 22.9; two slots
below 3:1 contrast -> every line is direct-labelled); field maps use one sequential blue ramp, light = quiet.
  python scripts/p4_8_meeting_figs.py [--best NAME] [--wave2 NAME]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path("outputs/p4_8/meeting")
DF = 0.5
FREQS = (50, 100, 200, 280)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # blue, orange, aqua, yellow, magenta
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]
FINALS = ["MB1_60K", "MB2_60K", "LS1A_60K", "LS1B_60K", "LS3_60K", "LS2A_30K", "LS2B_30K"]


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def colr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def onlat(rx):
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def pick_best(table):
    v = table.get("verdicts", {})
    pairs = table.get("pairs", {})
    def score(n):
        d = pairs.get(n + " - GX_60K")
        return d["HF"][0] + d["UF_25-100"][0] if d else -9
    passed = [n for n in FINALS if v.get(n, {}).get("holds")]
    pool = passed or [n for n in FINALS if n + " - GX_60K" in pairs]
    best = max(pool, key=score)
    return best, bool(passed)


def curves(cache_fmt, names, idx=None):
    """Per-model per-bin off-lattice r averaged over the rooms in idx (all if None)."""
    out = {}
    for n in names:
        z = np.load(cache_fmt.format(n))
        rs = []
        i = 0
        while "P%d" % i in z.files:
            if idx is None or i in idx:
                P, T, rx = z["P%d" % i][:, :601], z["T%d" % i][:, :601], z["rx%d" % i]
                off = ~onlat(rx)
                rs.append(colr(db(P[off]), db(T[off])))
            i += 1
        r = np.nanmean(np.stack(rs), axis=0)
        out[n] = np.convolve(r, np.ones(11) / 11, mode="same")
    return out


def curve_panel(ax, cv, labels, title):
    f = np.arange(601) * DF
    sel = slice(50, 591)
    for k, (n, r) in enumerate(cv.items()):
        ax.plot(f[sel], r[sel], color=SLOTS[k], lw=2, label=labels[n])
        ax.annotate(labels[n].split(" ")[0], (f[sel][-1], r[sel][-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.axvspan(295, 300, color=GRID, lw=0)
    ax.set_xlim(25, 315); ax.set_ylim(0, 1)
    ax.set_xlabel("frequency (Hz)", color=INK2); ax.set_ylabel("off-lattice spatial r (dB)", color=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    style(ax)


def field_fig(path, title, h5path, verts, L, W, T, Ps, rx, proj_note):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from aaf.eval.physical_projection import project_field
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    with h5py.File(h5path) as f:
        keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool)
    xs = 0.12 + np.arange(64) * (L - 0.24) / 63
    ys = 0.12 + np.arange(64) * (W - 0.24) / 63
    off = ~onlat(rx)
    Pp = [(lab, project_field(P, keep.reshape(64, 64), (L - 0.24) / 63, (W - 0.24) / 63)) for lab, P in Ps]
    fig, axes = plt.subplots(len(FREQS), 1 + len(Pp), figsize=(3.1 * (1 + len(Pp)) + 1.2, 2.9 * len(FREQS)),
                             constrained_layout=True, facecolor="white")
    for i, hz in enumerate(FREQS):
        b = int(round(hz / DF))
        gt = db(T[:, b])
        lo, hi = np.percentile(gt, [2, 98])
        for j, (lab, H) in enumerate([("FDTD ground truth", T)] + Pp):
            ax = axes[i, j]
            g = np.full(4096, np.nan)
            g[keep] = db(H[:, b])
            im = ax.pcolormesh(xs, ys, g.reshape(64, 64).T, cmap=cmap, vmin=lo, vmax=hi, shading="nearest")
            v = np.asarray([list(p) for p in verts] + [list(verts[0])])
            ax.plot(v[:, 0], v[:, 1], color=INK, lw=1.0)
            ax.plot(0.5, 1.6, marker="*", color=INK, ms=9, mec="white", mew=0.8)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(lab, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel("{} Hz".format(hz), fontsize=10, color=INK2)
            if j > 0:
                r = float(colr(db(H[off, b:b + 1]), db(T[off, b:b + 1]))[0])
                ax.text(0.02, 0.03, "r = {:+.2f}".format(r), transform=ax.transAxes, fontsize=8, color=INK,
                        bbox=dict(facecolor="white", alpha=0.85, lw=0))
        fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.85, pad=0.01).set_label("dB |H|", color=INK2)
    fig.suptitle(title + "\n" + proj_note, fontsize=11, color=INK)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default=None)
    ap.add_argument("--wave2", default=None)
    a = ap.parse_args()
    table = json.load(open("outputs/p4_8/primary_table.json"))
    best, passed = (a.best, True) if a.best else pick_best(table)
    models = ["G1_60K", "GX_60K", "R0_120K", best] + ([a.wave2] if a.wave2 else [])
    labels = {"G1_60K": "G1 @60K (500 L shapes)", "GX_60K": "GX @60K (C500 + CX500)", "R0_120K": "R0 @120K (GX, matched exposure)",
              best: "{} (best wave-1{})".format(best, "" if passed else ", no arm passed")}
    if a.wave2:
        labels[a.wave2] = "{} (wave 2)".format(a.wave2)
    OUT.mkdir(parents=True, exist_ok=True)
    man = ["# P4-8 meeting figures (D174(h))", "", "Network models only. Best wave-1 arm by rule: **{}**{}.".format(
        best, "" if passed else " (no arm passed its pre-registered test; largest dHF + dUF)"), "",
        "| file | what | data |", "|---|---|---|"]

    # F1 / F2
    cin = curves("outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz", models)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor="white")
    curve_panel(ax, cin, labels, "In family: 15 frozen test shapes, off-lattice")
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)
    fig.tight_layout(); fig.savefig(OUT / "F1_freq_curve_in_family.png", dpi=160, facecolor="white"); plt.close(fig)
    man.append("| F1_freq_curve_in_family.png | per-bin off-lattice r, mean over 15 shapes, 11-bin smoothing | gate2_dense caches |")
    un = json.load(open("configs/sweeps_2d_mat/p4_8_unseen_manifest.json"))["configs"]
    iu = {i for i, r in enumerate(un) if r["family"] == "U"}
    ic = {i for i, r in enumerate(un) if r["family"] == "C3"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor="white", sharey=True)
    curve_panel(axes[0], curves("outputs/p4_8/cache/unseen_{}.npz", models, iu), labels, "Unseen family U (8 rooms)")
    curve_panel(axes[1], curves("outputs/p4_8/cache/unseen_{}.npz", models, ic), labels, "Unseen family C3: NW+NE+SE (8 rooms)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)
    fig.tight_layout(); fig.savefig(OUT / "F2_freq_curve_unseen.png", dpi=160, facecolor="white"); plt.close(fig)
    man.append("| F2_freq_curve_unseen.png | same, frozen unseen set, U and C3 separately | outputs/p4_8/cache/unseen_* |")

    # F3: selection fixed by G1 @60K, before any field is drawn
    from aaf.data.shape_configs import configs_from_rows
    test = configs_from_rows(json.load(open("configs/sweeps_2d_mat/p4_2_shapes_manifest.json"))["configs"], split="test")
    hf = np.array(json.load(open("outputs/p4_5/speckle/G1_60K.json"))["per_shape"]["r"]["200-300"])
    ia = int(np.argsort(hf)[(15 - 1) // 2])
    g1u = json.load(open("outputs/p4_8/unseen/G1_60K.json"))["rooms"]
    picks = [("in_family", "test", ia, "lower-median in-family shape by G1 @60K 200-300 Hz r ({:.3f})".format(hf[ia]))]
    for fam, idx in (("U", sorted(iu)), ("C3", sorted(ic))):
        rr = np.array([g1u[i]["raw"]["25-100"] for i in idx])
        j = idx[int(np.argsort(rr)[(len(idx) - 1) // 2])]
        picks.append((fam, "unseen", j, "lower-median {} room by G1 @60K 25-100 Hz r ({:.3f})".format(fam, g1u[j]["raw"]["25-100"])))
    cols = ["G1_60K", "R0_120K", best]
    note = "model panels: render-time physical projection applied (post-processing, D160(e)); FDTD unprojected"
    for tag, kind, i, rule in picks:
        if kind == "test":
            Z = {n: np.load("outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz".format(n)) for n in cols}
            c = test[i]
            h5, verts, L, W = Path("data/track_p4_5_testdense") / c.filename, c.verts, c.L, c.W
        else:
            Z = {n: np.load("outputs/p4_8/cache/unseen_{}.npz".format(n)) for n in cols}
            r = un[i]
            h5, verts, L, W = Path("data/track_p4_8_unseen") / r["filename"], [tuple(v) for v in r["verts"]], r["L"], r["W"]
        T, rx = Z["G1_60K"]["T%d" % i][:, :601], Z["G1_60K"]["rx%d" % i]
        for n in cols:
            assert np.array_equal(Z[n]["rx%d" % i], rx)
        Ps = [(labels[n].split(" (")[0], Z[n]["P%d" % i][:, :601]) for n in cols]
        p = OUT / "F3_fields_{}.png".format(tag)
        field_fig(p, "{}: {}".format(tag.replace("_", " "), Path(str(h5)).stem), h5, verts, L, W, T, Ps, rx, note)
        man.append("| {} | S1-layout field maps, rows 50/100/200/280 Hz, columns FDTD / G1 / R0 / best; per-row limits = "
                   "FDTD 2-98th pct; corner r off-lattice | {} |".format(p.name, rule))

    # F4: per-room paired scatter, best vs G1
    bu = json.load(open("outputs/p4_8/unseen/{}.json".format(best)))["rooms"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor="white")
    for ax, band in zip(axes, ("25-100", "100-200", "200-300")):
        for k, fam in enumerate(("U", "C3")):
            xs = [g1u[i]["raw"][band] for i in range(len(un)) if un[i]["family"] == fam]
            ys = [bu[i]["raw"][band] for i in range(len(un)) if un[i]["family"] == fam]
            ax.scatter(xs, ys, s=46, color=SLOTS[k], edgecolor="white", linewidth=1.5, label=fam, zorder=3)
        lim = [0, 1]
        ax.plot(lim, lim, color=AXIS, lw=1, zorder=1)
        above = sum(bu[i]["raw"][band] > g1u[i]["raw"][band] for i in range(len(un)))
        ax.set_title("{} Hz: {} of {} rooms above the diagonal".format(band, above, len(un)), color=INK, fontsize=10, loc="left")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel("G1 @60K r", color=INK2); ax.set_ylabel("{} r".format(best), color=INK2)
        style(ax)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "F4_scatter_unseen.png", dpi=160, facecolor="white"); plt.close(fig)
    man.append("| F4_scatter_unseen.png | per-room off-lattice r on the unseen set, best arm vs G1 @60K, per band | outputs/p4_8/unseen/*.json |")
    (OUT / "FIGURE_MANIFEST.md").write_text("\n".join(man) + "\n")
    print("\n".join(man))


if __name__ == "__main__":
    main()
