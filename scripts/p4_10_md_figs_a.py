"""P4-10 meeting day V1-V5 (D211, D213). Maps: full-grid display (truth FDTD 2 cm nodes, models 4 cm nodes). Every printed number: off-lattice dense receivers, frozen scorer. `--fig V1 V2 ...`"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

import scripts.p4_10_md_style as S
from scripts.p4_10_md import BINS, MED125, POS12, ROOT, SCALES, SHOW, STRIP_P, STRIP_S, UNSEEN, cases
from scripts.p4_10_md_style import COL, NAME, movie, new, outline, plt, save, star
from scripts.p4_10_scorecard import BAND, band_r, colr, db, on_lattice

FB = {50: 100, 100: 200, 125: 250}
BAND_OF = {50: "25-100", 125: "100-200"}
CASE = {c["key"]: c for c in cases()}
CAP = ("Maps are drawn on a full grid (truth: FDTD, 2 cm; model: 4 cm). Every number r is the off-lattice accuracy at that frequency (spatial correlation of level, dense receivers), never read from the map.")


def truth(k):
    return np.load(ROOT / "truth" / (k + ".npz"))


def pred(m, k):
    return np.load(ROOT / "pred" / "{}__{}.npz".format(m, k))


def r_at(P, T, rx, b):
    s = ~on_lattice(rx)
    return float(colr(db(P[s]), db(T[s]))[b])


def r_band(P, T, rx):
    s = ~on_lattice(rx)
    return band_r(P[s], T[s])


def scores(m, k, f):
    t, p = truth(k), pred(m, k)
    return {"r_at_{}Hz".format(f): r_at(p["Pd"], t["Hd"], t["rx"], FB[f]), "r_band": r_band(p["Pd"], t["Hd"], t["rx"])}


def img(gij, v, shape, step):
    a = np.full(shape, np.nan)
    a[gij[:, 0] // step, gij[:, 1] // step] = v
    return a.T


def level_map(ax, c, gij, H, step, ref, vmin=-35, vmax=0):
    nx, ny = int(round(c["L"] / 0.02)) + 1, int(round(c["W"] / 0.02)) + 1
    sx, sy = (nx - 1) // step + 1, (ny - 1) // step + 1
    a = img(gij, 20 * np.log10(np.abs(H) + 1e-12) - ref, (sx, sy), step)
    h = 0.02 * step / 2
    im = ax.imshow(a, origin="lower", extent=[-h, (sx - 1) * 0.02 * step + h, -h, (sy - 1) * 0.02 * step + h], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    outline(ax, c["verts"])
    star(ax, c["src"])
    return im


def panel(ax, k, who, f):
    """who: 'truth' or a model key; colour = level minus the TRUTH's 99th percentile at this frequency (same reference for every row)"""
    c, t, bi = CASE[k], truth(k), BINS.index(FB[f])
    ref = float(np.percentile(20 * np.log10(np.abs(t["Hg"][:, bi]) + 1e-12), 99))
    if who == "truth":
        return level_map(ax, c, t["gij"], t["Hg"][:, bi], 1, ref)
    p = pred(who, k)
    return level_map(ax, c, p["gij"], p["Pg"][:, bi], 2, ref)


def badge(ax, txt):
    ax.text(0.5, -0.035, txt, transform=ax.transAxes, ha="center", va="top", fontsize=15, color="#111111")


def colorbar(fig, im, rect=(0.93, 0.16, 0.012, 0.66), lab="level, dB below the truth's loudest 1%"):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(lab, fontsize=14)
    cb.ax.tick_params(labelsize=14)


def strip(keys, rows, f, title, name, colt, note, meta, lims=None):
    fig = new(16, 9)
    n, m = len(keys), len(rows)
    w0 = 0.035
    L, W = max(CASE[k]["L"] for k in keys), max(CASE[k]["W"] for k in keys)
    gs = fig.add_gridspec(m, n, left=0.075, right=0.915, top=0.865, bottom=0.135, wspace=0.05, hspace=0.20)
    im = None
    meta["panels"] = {}
    for j, k in enumerate(keys):
        for i, (lab, who) in enumerate(rows):
            ax = fig.add_subplot(gs[i, j])
            im = panel(ax, k, who, f)
            ax.set_xlim(-0.1, L + 0.1)
            ax.set_ylim(-0.1, W + 0.1)
            if i == 0:
                ax.set_title(colt[j], fontsize=15)
            if j == 0:
                ax.text(-0.06, 0.5, lab, transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=17, fontweight="bold")
            if who != "truth":
                s = scores(who, k, f)
                badge(ax, "r = {:.2f}".format(s["r_at_{}Hz".format(f)]))
                meta["panels"]["{}|{}".format(k, NAME[who])] = s
    colorbar(fig, im)
    fig.suptitle(title, fontsize=19, x=0.02, ha="left", y=0.965, fontweight="bold")
    fig.text(0.02, 0.012, note, fontsize=11.5, color="#444444", ha="left", va="bottom", wrap=True)
    meta["caption"] = note
    save(fig, name, meta)


def ranks():
    """rank of the showcase among the 15 in-family rooms on each figure's metric (D213), from the posted scorecards"""
    from scripts.p4_10_scorecard import SC, cells, cid
    out = {}

    def vals(name, fam, mags, key):
        d = {}
        for c in cells([fam], False):
            if c["group"] == "test" and (mags is None or c["mag"] in mags):
                f = SC / name / (cid(c) + ".json")
                d.setdefault(c["room"], []).append(json.loads(f.read_text()))
        return {r: [key(x) for x in v] for r, v in d.items()}

    def rank(d):
        s = sorted(d, key=lambda r: -np.mean(d[r]))
        return {"rank": s.index(SHOW) + 1, "of": len(s), "value": float(np.mean(d[SHOW])), "median_room": s[len(s) // 2], "median_value": float(np.mean(d[s[len(s) // 2]]))}
    for b in ("25-100", "100-200"):
        out["V1|R1-S|StretchU r|" + b] = rank(vals("R1S_150K", "StretchU", None, lambda x, b=b: x["edited"]["r"][b]))
        out["V2|R1|third source r|" + b] = rank(vals("R1_150K", "Source", ("third",), lambda x, b=b: x["edited"]["r"][b]))
    out["V4|R1|static r, mean of three bands"] = rank(vals("R1_150K", "Source", ("third",), lambda x: np.mean([x["orig_r"][b] for b in BAND])))
    return out


def rk(R, k):
    return "showcase ranks {} of {}".format(R[k]["rank"], R[k]["of"])


def v1():
    R = ranks()
    keys = ["show_s{:.3f}".format(s) for s in STRIP_S]
    for f in (50, 125):
        note = CAP + "\nStretch factor s applies to both room dimensions, the notch keeps its proportion; source fixed. Known room {}; {} on R1-S stretch accuracy at {} Hz (R1-S).".format(SHOW, rk(R, "V1|R1-S|StretchU r|" + BAND_OF[f]), BAND_OF[f])
        strip(keys, [("Truth", "truth"), ("R1-S", "R1S_150K")], f, "Stretching a room: how the sound field changes, {} Hz".format(f), "V1_stretch_{}Hz".format(f),
              ["s = {:.2f}".format(s) for s in STRIP_S], note, {"room": SHOW, "figure": "V1", "frequency_hz": f, "ranks": R["V1|R1-S|StretchU r|" + BAND_OF[f]], "scales": STRIP_S})
    for f in (50, 125):
        keys = ["unseen_s{:.3f}".format(s) for s in STRIP_S]
        note = CAP + "\nNew room, not in any training family: {} (the unseen room with the median R1 static r; the lower of the two middle rooms). ".format(UNSEEN)
        strip(keys, [("Truth", "truth"), ("R1-S", "R1S_150K")], f, "Backup, a room type never seen in training: stretching, {} Hz".format(f), "V1u_stretch_unseen_{}Hz".format(f),
              ["s = {:.2f}".format(s) for s in STRIP_S], note, {"room": UNSEEN, "figure": "V1 unseen backup", "frequency_hz": f})
    v1_movie()


def frame(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="rgba", dpi=80)
    w, h = fig.canvas.get_width_height()
    a = np.frombuffer(buf.getvalue(), np.uint8).reshape(int(fig.get_figheight() * 80), int(fig.get_figwidth() * 80), 4)[:, :, :3].copy()
    plt.close(fig)
    return a


def v1_movie():
    frames, meta = [], {"figure": "V1 animation", "room": SHOW, "frames": {}}
    Lm, Wm = max(CASE["show_s{:.3f}".format(s)]["L"] for s in SCALES), max(CASE["show_s{:.3f}".format(s)]["W"] for s in SCALES)
    for s in SCALES:
        k = "show_s{:.3f}".format(s)
        fig = new(16, 9)
        gs = fig.add_gridspec(2, 2, left=0.06, right=0.90, top=0.86, bottom=0.06, wspace=0.04, hspace=0.12)
        for i, f in enumerate((50, 125)):
            for j, (lab, who) in enumerate((("Truth", "truth"), ("R1-S", "R1S_150K"))):
                ax = fig.add_subplot(gs[i, j])
                im = panel(ax, k, who, f)
                ax.set_xlim(-0.1, Lm + 0.1)
                ax.set_ylim(-0.1, Wm + 0.1)
                t = lab + ", {} Hz".format(f)
                if who != "truth":
                    sc = scores(who, k, f)
                    t += "   r = {:.2f}".format(sc["r_at_{}Hz".format(f)])
                    meta["frames"]["{}|{}Hz".format(k, f)] = sc
                ax.set_title(t, fontsize=17)
        colorbar(fig, im)
        fig.suptitle("Stretching the room: s = {:.3f}".format(s), fontsize=22, fontweight="bold", x=0.02, ha="left", y=0.96)
        frames.append(frame(fig))
    movie(frames, "V1_stretch_animation", 9)
    (S.OUT / "V1_stretch_animation.json").write_text(json.dumps(meta, indent=1))


def v2():
    R = ranks()
    pos = [("show_p{:02d}".format(i) if i else "show_s1.000") for i in STRIP_P]
    lab = ["start" if i == 0 else "{:.0f}%".format(100 * POS12[i]) if i != 11 else "third source" for i in STRIP_P]
    for f in (50, 125):
        note = CAP + "\nSource positions: six of twelve equally spaced points on the line from the original source to the third source. Known room {}; {} on R1's third-source accuracy at {} Hz.".format(SHOW, rk(R, "V2|R1|third source r|" + BAND_OF[f]), BAND_OF[f])
        strip(pos, [("Truth", "truth"), ("R1", "R1_150K"), ("R0", "R0_120K")], f, "Moving the source: where the sound goes, {} Hz".format(f), "V2_source_{}Hz".format(f), lab, note,
              {"room": SHOW, "figure": "V2", "frequency_hz": f, "ranks": R["V2|R1|third source r|" + BAND_OF[f]], "positions": [CASE[k]["src"] for k in pos]})
    pos = ["med_p{:02d}".format(i) for i in STRIP_P]
    note = CAP + "\nSecond example (the showcase ranks 3 of 15 on this metric, so the median room is shown too): {}, R1's third-source accuracy at 100-200 Hz.".format(MED125)
    strip(pos, [("Truth", "truth"), ("R1", "R1_150K"), ("R0", "R0_120K")], 125, "Second example, the median room: moving the source, 125 Hz", "V2m_source_median_125Hz", lab, note,
          {"room": MED125, "figure": "V2 second example", "frequency_hz": 125, "positions": [CASE[k]["src"] for k in pos]})
    for f in (50, 125):
        pos = [("unseen_p{:02d}".format(i) if i else "unseen_s1.000") for i in STRIP_P]
        note = CAP + "\nNew room, not in any training family: {} (the unseen room with the median R1 static r; the lower of the two middle rooms).".format(UNSEEN)
        strip(pos, [("Truth", "truth"), ("R1", "R1_150K"), ("R0", "R0_120K")], f, "Backup, a room type never seen in training: moving the source, {} Hz".format(f), "V2u_source_unseen_{}Hz".format(f), lab, note,
              {"room": UNSEEN, "figure": "V2 unseen backup", "frequency_hz": f, "positions": [CASE[k]["src"] for k in pos]})
    v2_movie()


def v2_movie():
    frames, meta = [], {"figure": "V2 animation", "room": SHOW, "frames": {}}
    c0 = CASE["show_s1.000"]
    for i in range(12):
        k = "show_s1.000" if i == 0 else "show_p{:02d}".format(i)
        fig = new(16, 9)
        gs = fig.add_gridspec(2, 3, left=0.04, right=0.90, top=0.86, bottom=0.05, wspace=0.04, hspace=0.12)
        for a, f in enumerate((50, 125)):
            for b, (lab, who) in enumerate((("Truth", "truth"), ("R1", "R1_150K"), ("R0", "R0_120K"))):
                ax = fig.add_subplot(gs[a, b])
                im = panel(ax, k, who, f)
                t = "{}, {} Hz".format(lab, f)
                if who != "truth":
                    sc = scores(who, k, f)
                    t += "   r = {:.2f}".format(sc["r_at_{}Hz".format(f)])
                    meta["frames"]["{}|{}|{}Hz".format(k, NAME[who], f)] = sc
                ax.set_title(t, fontsize=16)
        colorbar(fig, im)
        fig.suptitle("Moving the source: {:.0f}% of the way from the original to the third position".format(100 * POS12[i]), fontsize=21, fontweight="bold", x=0.02, ha="left", y=0.96)
        frames.append(frame(fig))
    movie(frames, "V2_source_animation", 12)
    (S.OUT / "V2_source_animation.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="+", required=True)
    a = ap.parse_args()
    for f in a.fig:
        globals()[f.lower()]()
