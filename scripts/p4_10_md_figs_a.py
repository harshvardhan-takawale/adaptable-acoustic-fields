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


def caption(fig, txt, width=143):
    import textwrap
    lines = []
    for para in txt.split("\n"):
        lines += textwrap.wrap(para, width)
    fig.text(0.02, 0.012, "\n".join(lines), fontsize=14, color="#444444", ha="left", va="bottom", linespacing=1.25)


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
    gs = fig.add_gridspec(m, n, left=0.07, right=0.915, top=0.86, bottom=0.185, wspace=0.04, hspace=0.13)
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
    caption(fig, note)
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

# ---------------------------------------------------------------------------------------------------- V3: materials
OCT = ("63", "125", "250")


def _sel(rx, src=(0.5, 1.6)):
    return ~on_lattice(rx) & (np.hypot(rx[:, 0] - src[0], rx[:, 1] - src[1]) >= 1.0)


def decay_rates(P, T0, T, rx, band):
    """mean 1/EDT (1/s) over the receivers valid in BOTH truths (D202's rule), per field -> (model, truth)"""
    from aaf.eval.octave_decay import decay_metrics, valid_mask
    s = _sel(rx)
    m0, m1 = decay_metrics(T0[s], band), decay_metrics(T[s], band)
    ok = valid_mask(m0["t35"]) & valid_mask(m1["t35"])
    f = lambda E: float(np.nanmean(1.0 / np.maximum(E[ok], 1e-6)))  # noqa: E731
    mp = decay_metrics(P[s], band)
    return f(mp["EDT"]), f(m1["EDT"]), int(ok.sum())


def v3_rates():
    out = {"all": {}, "long": {}}
    for kind, alphas in (("all", (0.10, 0.15, 0.20, 0.30, 0.40, 0.50)), ("long", (0.15, 0.30, 0.40, 0.50))):
        for b in OCT:
            e = {"alpha": list(alphas), "truth": [], "R1": [], "R1-M": [], "n_rx": []}
            for a in alphas:
                k = "show_s1.000" if a == 0.15 else "show_{}{:.2f}".format(kind, a)
                t, t0 = truth(k), truth("show_s1.000")
                rx = t["rx"]
                for m, nm in (("R1_150K", "R1"), ("R1M_150K", "R1-M")):
                    rm, rt, n = decay_rates(pred(m, k)["Pd"], t0["Hd"], t["Hd"], rx, b)
                    e[nm].append(rm)
                e["truth"].append(rt)
                e["n_rx"].append(n)
            out[kind][b] = e
    return out


def v3_rank():
    """rank of the showcase among the 15 in-family rooms on |MM1 slope - 1| of R1-M (D213 'V3 rank computed with its figure'); per-room slope over its own Material cells x octaves"""
    from scripts.p4_10_scorecard import SC, cells, cid
    per = {}
    for c in cells(["Material"], False):
        if c["group"] != "test":
            continue
        d = json.loads((SC / "R1M_150K" / (cid(c) + ".json")).read_text())["decay"]
        for b in OCT:
            e = d[b]
            per.setdefault(c["room"], []).append((e["rate_Te"] - e["rate_T0"], e["rate_Pe"] - e["rate_P0"]))
    sl = {r: float(np.polyfit([x for x, _ in v], [y for _, y in v], 1)[0]) for r, v in per.items()}
    order = sorted(sl, key=lambda r: abs(sl[r] - 1))
    return {"rank": order.index(SHOW) + 1, "of": len(order), "value": sl[SHOW], "median_room": order[len(order) // 2], "median_value": sl[order[len(order) // 2]], "all_slopes": sl}


def v3():
    from matplotlib.ticker import MaxNLocator
    rt = v3_rates()
    rk3 = v3_rank()
    fig = new(16, 9)
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.985, top=0.82, bottom=0.175, wspace=0.22, hspace=0.44)
    for i, (kind, lab) in enumerate((("all", "all walls"), ("long", "one wall (the longest)"))):
        for j, b in enumerate(OCT):
            ax = fig.add_subplot(gs[i, j])
            e = rt[kind][b]
            for nm, ls in (("truth", "-"), ("R1", "-"), ("R1-M", "-")):
                ax.plot(e["alpha"], e[nm], ls, color=COL[nm], lw=3 if nm == "truth" else 2.4, marker="o", ms=8, label={"truth": "truth", "R1": "R1", "R1-M": "R1-M"}[nm])
            ax.grid(True, color="#e6e6e6")
            ax.set_title("{} Hz octave, {}".format(b, lab), loc="left", fontsize=15)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if i == 1:
                ax.set_xlabel("wall absorption of the edited walls")
            if j == 0:
                ax.set_ylabel("decay rate 1/EDT (1/s)")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.985, 0.935), fontsize=17)
    fig.suptitle("Changing wall materials: how fast the sound dies away", fontsize=20, x=0.02, ha="left", y=0.97, fontweight="bold")
    fig.text(0.02, 0.89, "Decay rate = 1 / early decay time. R1 was trained at absorption 0.15 only, R1-M on 0.08 to 0.60 per wall.", fontsize=14, color="#444444")
    caption(fig, "Off-lattice dense receivers at least 1 m from the source, valid in both truths; mean 1/EDT. Known room {}; the showcase ranks {} of {} on how close R1-M's response is to the true size (slope {:.2f}; median room {:.2f}).".format(
        SHOW, rk3["rank"], rk3["of"], rk3["value"], rk3["median_value"]))
    save(fig, "V3ab_decay_rate_vs_absorption", {"figure": "V3 (a), (b)", "room": SHOW, "rates": rt, "ranks": rk3})
    v3c(rk3)


def v3c(rk3):
    from aaf.eval.octave_decay import FS, octave_ir
    from scripts.p4_10_meeting_figs import _edc_db
    fig = new(16, 9)
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.80, bottom=0.16, wspace=0.16)
    meta = {"figure": "V3 (c)", "room": SHOW, "curves": {}, "ranks": rk3}
    for j, b in enumerate(("63", "125")):
        ax = fig.add_subplot(gs[0, j])
        t = np.arange(int(1.5 * FS)) / FS
        for k, lab, dash in (("show_all0.10", "absorption 0.10", (0, (5, 2))), ("show_all0.50", "absorption 0.50", "-")):
            tr = truth(k)
            s = np.nonzero(_sel(tr["rx"]))[0][::4]
            for nm, src, col in (("truth", tr["Hd"], COL["truth"]), ("R1-M", pred("R1M_150K", k)["Pd"], COL["R1-M"])):
                y = np.median(_edc_db(octave_ir(src[s], b), FS), 0)
                ax.plot(t, y, color=col, ls=dash, lw=2.8, label="{}, {}".format(nm, lab))
                meta["curves"]["{}|{}|{}".format(b, nm, lab)] = {"t60_like_s_at_-10dB": float(t[np.argmax(y <= -10)]), "db_at_0.1_0.25_0.5_s": [float(y[int(x * FS)]) for x in (0.1, 0.25, 0.5)]}
        ax.set_xlim(0, 0.7)
        ax.set_ylim(-30, 1)
        ax.grid(True, color="#e6e6e6")
        ax.set_xlabel("time after onset (s)")
        ax.set_title("{} Hz octave".format(b), loc="left", fontsize=17)
        if j == 0:
            ax.set_ylabel("energy decay (dB, median over receivers)")
            ax.legend(frameon=False, fontsize=15)
    fig.suptitle("Early energy decay with low and high wall absorption", fontsize=20, x=0.02, ha="left", y=0.97, fontweight="bold")
    fig.text(0.02, 0.87, "All walls changed to 0.10 (dashed) or 0.50 (solid). Black: the simulated truth. Magenta: R1-M, told the new absorption.", fontsize=14, color="#444444")
    caption(fig, "0 to -30 dB of the decay curve; off-lattice receivers at least 1 m from the source; median over receivers. Known room {}.".format(SHOW))
    save(fig, "V3c_early_decay", meta)


# ------------------------------------------------------------------------------------------------- V4: the listener's walk
def v4():
    R = ranks()
    for k, tag, lab in (("show_s1.000", "original", "original room"), ("show_s1.100", "stretched", "room stretched by 1.10")):
        c, t, p = CASE[k], truth(k), pred("R1_150K", k)
        f = np.arange(601) * 0.5
        lo, hi = 50, 600
        Lv = lambda H: 20 * np.log10(np.abs(H) + 1e-12)  # noqa: E731
        ref = np.percentile(Lv(t["Hp"][:, lo:hi]), 99)
        A, B_ = Lv(t["Hp"][:, lo:hi]) - ref, Lv(p["Pp"][:, lo:hi]) - ref
        D = B_ - A
        x = t["path"][:, 0]
        pos = [int(np.argmin(np.abs(x - q))) for q in (0.2 * c["L"], 0.5 * c["L"], 0.8 * c["L"])]
        fig = new(16, 9)
        gs = fig.add_gridspec(2, 3, left=0.06, right=0.94, top=0.80, bottom=0.165, hspace=0.45, wspace=0.22, height_ratios=[1.35, 1])
        ext = [0, len(x), 25, 300]
        for j, (nm, M, cm, vm) in enumerate((("truth", A, "viridis", (-35, 0)), ("R1", B_, "viridis", (-35, 0)), ("R1 minus truth", D, "RdBu_r", (-20, 20)))):
            ax = fig.add_subplot(gs[0, j])
            im = ax.imshow(M.T, origin="lower", aspect="auto", extent=[x[0], x[-1], 25, 300], cmap=cm, vmin=vm[0], vmax=vm[1], interpolation="nearest")
            for q in pos:
                ax.axvline(x[q], color="black" if j == 2 else "white", lw=1.6, ls=(0, (5, 3)))
            ax.set_title(nm, loc="left", fontsize=16)
            ax.set_xlabel("position along the walk, x (m)")
            if j == 0:
                ax.set_ylabel("frequency (Hz)")
            if j >= 1:
                cb = fig.colorbar(im, ax=ax, pad=0.015)
                cb.ax.tick_params(labelsize=14)
                cb.set_label("dB" if j == 1 else "dB difference", fontsize=14)
        for j, q in enumerate(pos):
            ax = fig.add_subplot(gs[1, j])
            ax.plot(f[lo:hi], A[q], color=COL["truth"], lw=2.2, label="truth")
            ax.plot(f[lo:hi], B_[q], color=COL["R1"], lw=2.0, label="R1")
            ax.set_ylim(-45, 8)
            ax.set_xlim(25, 300)
            ax.set_title("walk position x = {:.1f} m".format(x[q]), loc="left", fontsize=15)
            ax.set_xlabel("frequency (Hz)")
            if j == 0:
                ax.set_ylabel("level (dB)")
                ax.legend(frameon=False, loc="upper right", fontsize=14)
            ax.grid(True, color="#e6e6e6")
        rb = r_band(p["Pd"], t["Hd"], t["rx"])
        fig.suptitle("A listener walks across the {}: what they hear at every frequency".format(lab), fontsize=20, x=0.02, ha="left", y=0.97, fontweight="bold")
        ins = fig.add_axes([0.862, 0.825, 0.125, 0.155])
        panel(ins, k, "truth", 50)
        ins.plot(t["path"][:, 0], t["path"][:, 1], color="white", lw=2.2, ls=(0, (4, 2)), zorder=7)
        for q in pos:
            ins.plot(x[q], 1.0, "o", color="white", ms=6, markeredgecolor="black", zorder=8)
        fig.text(0.02, 0.885, "Dashed lines: the three positions below. R1 accuracy in this room: {} (25-100 / 100-200 / 200-300 Hz).".format(" / ".join("{:.2f}".format(rb[b]) for b in BAND)), fontsize=14, color="#444444")
        caption(fig, "Accuracy is the off-lattice dense-receiver score, never read from the pictures. Walk: y = 1.0 m, x from 0.5 m to {:.2f} m (100 points). Known room {}; {} on R1's static accuracy.".format(c["L"] - 0.5, SHOW, rk(R, "V4|R1|static r, mean of three bands")))
        save(fig, "V4_walk_{}".format(tag), {"figure": "V4", "room": SHOW, "case": k, "scale": c["s"], "R1_r_off_lattice_dense": rb, "path_xy": t["path"].tolist(),
                                              "marked_x_m": [float(x[q]) for q in pos], "ranks": R["V4|R1|static r, mean of three bands"]})


# ----------------------------------------------------------------------------------------------- V5: where the energy flows
def v5():
    """active intensity I ~ -Im(conj(p) grad p) / (2 w rho); model gradient = the band-limited (projected) field's lattice gradient (Q-C's proj-FD); truth = FDTD central differences (2 cm, stored stencils)."""
    from aaf.eval.physical_projection import project_field
    from scripts.p4_9_dir_zeroshot import G2, lattice_grad, rho
    R = ranks()
    name = "R1_150K"
    c = CASE["show_s1.000"]
    st = np.load("outputs/p4_9/fdtd_stencils/dense/{}.npz".format(SHOW))
    import h5py
    with h5py.File("data/track_p4_5_testdense/{}.h5".format(SHOW)) as fh:
        keep = np.asarray(json.loads(fh.attrs["lattice_keep"]), bool).reshape(64, 64)
    from scripts.p4_9_dir_zeroshot import dense_rooms
    ci = next(r["cache_i"] for r in dense_rooms() if r["path"].stem == SHOW)
    cache = np.load(G2.format(name))
    P = cache["P%d" % ci][:, :601]
    rx = st["rx"]
    L, W = c["L"], c["W"]
    dx, dy = (L - 0.24) / 63, (W - 0.24) / 63
    Pp = project_field(P, keep, dx, dy)
    lr, LG = lattice_grad(Pp, keep, dx, dy)
    k0 = 2 * np.pi * np.arange(601) * 0.5 / 343.0
    ok = (~on_lattice(rx)) & (np.hypot(*(rx - np.array([0.5, 1.6])).T) >= 0.5)
    T = np.stack([st["g1x"], st["g1y"]], axis=1)
    b = {50: 100, 100: 200}
    meta = {"figure": "V5", "room": SHOW, "numbers": {}, "ranks": v5_rank(name)}
    inten = lambda p, g: -np.imag(np.conj(p)[:, None] * g)  # noqa: E731
    fig = new(16, 9)
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.90, top=0.84, bottom=0.115, wspace=0.04, hspace=0.20)
    tr = truth("show_s1.000")
    from scripts.p4_10_scorecard import stored
    from scripts.p4_10_md import man
    base = stored(man(SHOW, "Source", "third"))
    for row, fr in enumerate((50, 100)):
        i = b[fr]
        pT, pM = base["H"][:, i], Pp[:, i]
        gT = T[:, :, i]
        gM = np.full((len(rx), 2), np.nan + 0j, np.complex128)
        gM[lr] = LG[:, :, i]
        IT, IM = inten(pT, gT), inten(np.where(np.isfinite(pM), pM, 0), gM)
        good = ok & np.isfinite(gM[:, 0].real) & np.all(np.isfinite(IM), 1)
        unit = lambda I: I / np.maximum(np.linalg.norm(I, axis=1, keepdims=True), 1e-30)  # noqa: E731
        cos = (unit(IT[good]) * unit(IM[good])).sum(1)
        sel = good[lr]
        rr = rho(LG[sel][:, :, [i]] / (1j * k0[i]), T[lr][sel][:, :, [i]] / (1j * k0[i]))
        meta["numbers"]["{}Hz".format(fr)] = {"energy_flow_direction_agreement_mean_cosine": float(np.mean(cos)), "median_angle_deg": float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1))))),
                                             "gradient_agreement_Re_rho": float(rr.real[0]), "n_rx": int(good.sum())}
        for col, (lab, I, who) in enumerate((("truth", IT, "truth"), ("R1", IM, "R1"))):
            ax = fig.add_subplot(gs[row, col])
            im = panel(ax, "show_s1.000", "truth" if who == "truth" else "R1_150K", fr)
            s4 = good.copy()
            idx = np.nonzero(s4)[0]
            q = idx[(np.arange(len(idx)) % 5) == 0]
            u = unit(I[q]) * 0.20
            ax.quiver(rx[q, 0], rx[q, 1], u[:, 0], u[:, 1], color="white", scale=1, scale_units="xy", angles="xy", width=0.0032, zorder=7)
            n = meta["numbers"]["{}Hz".format(fr)]
            ax.set_title("{}, {} Hz{}".format("Truth" if who == "truth" else "R1", fr, "" if who == "truth" else "   direction agreement {:.2f}".format(n["energy_flow_direction_agreement_mean_cosine"])), fontsize=16, loc="left")
    colorbar(fig, im)
    fig.suptitle("Where the sound energy flows, original source", fontsize=20, x=0.02, ha="left", y=0.965, fontweight="bold")
    fig.text(0.02, 0.885, "Arrows: direction of the steady energy flow (unit length). Truth: FDTD, 2 cm. Model: gradient of the band-limited field.", fontsize=14, color="#444444")
    caption(fig, "Direction agreement = mean cosine between the two flows over off-lattice receivers at least 0.5 m from the source. Known room {}; rank {} of {} on this number at 50 Hz and {} at 100 Hz (median room {:.2f} and {:.2f}).".format(SHOW, meta["ranks"]["rank_50"], meta["ranks"]["of"], meta["ranks"]["rank_100"], meta["ranks"]["median_value_50"], meta["ranks"]["median_value_100"]))
    save(fig, "V5_energy_flow", meta)


def v5_rank(name):
    """the same direction agreement at 50 Hz (and 100 Hz) in all 15 in-family rooms"""
    from aaf.eval.physical_projection import project_field
    from scripts.p4_9_dir_zeroshot import G2, STEN, dense_rooms, lattice_grad
    import h5py
    vals = {50: {}, 100: {}}
    cache = np.load(G2.format(name))
    for r in dense_rooms():
        if r["group"] != "test":
            continue
        st = np.load(STEN / (r["path"].stem + ".npz"))
        with h5py.File(r["path"]) as f:
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
            keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
            L, W = float(f.attrs["L"]), float(f.attrs["W"])
            T0 = np.asarray(f["ism/H_complex"])[:, :601]
        dx, dy = (L - 0.24) / 63, (W - 0.24) / 63
        Pp = project_field(cache["P%d" % r["cache_i"]][:, :601], keep, dx, dy)
        lr, LG = lattice_grad(Pp, keep, dx, dy)
        ok = (~on_lattice(rx)) & (np.hypot(*(rx - np.array([0.5, 1.6])).T) >= 0.5)
        gT = np.stack([st["g1x"], st["g1y"]], axis=1)
        for fr, i in ((50, 100), (100, 200)):
            gM = np.full((len(rx), 2), np.nan + 0j, np.complex128)
            gM[lr] = LG[:, :, i]
            IT, IM = -np.imag(np.conj(T0[:, i])[:, None] * gT[:, :, i]), -np.imag(np.conj(Pp[:, i])[:, None] * gM)
            good = ok & np.all(np.isfinite(IM), 1)
            unit = lambda I: I / np.maximum(np.linalg.norm(I, axis=1, keepdims=True), 1e-30)  # noqa: E731
            vals[fr][r["path"].stem] = float(np.mean((unit(IT[good]) * unit(IM[good])).sum(1)))
    s50, s100 = sorted(vals[50], key=lambda r: -vals[50][r]), sorted(vals[100], key=lambda r: -vals[100][r])
    return {"rank_50": s50.index(SHOW) + 1, "rank_100": s100.index(SHOW) + 1, "of": len(s50), "value_50": vals[50][SHOW], "value_100": vals[100][SHOW], "median_room_50": s50[len(s50) // 2], "median_room_100": s100[len(s100) // 2],
            "median_value_50": vals[50][s50[len(s50) // 2]], "median_value_100": vals[100][s100[len(s100) // 2]], "all": vals}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="+", required=True)
    a = ap.parse_args()
    for f in a.fig:
        globals()[f.lower()]()
