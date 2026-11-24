"""P4-10 meeting day (D211, D216): V6 (why decay is off), V3(d) (MM1 scatter), V8 (slide scorecards), V7 (late floor under ISO 3382-1 compensation).
  python scripts/p4_10_md_figs_b.py --fig V6 V3d V8      (seconds, CPU)
  python scripts/p4_10_md_figs_b.py --render_r1m          (GPU: R1-M @150K dense renders of the 2 S-1 rooms -> outputs/p4_10/v7/pred/)
  python scripts/p4_10_md_figs_b.py --v7                  (CPU: outputs/p4_10/v7/report.{json,md})
  python scripts/p4_10_md_figs_b.py --fig V7              (after --v7)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.p4_10_md_style import COL, OUT, new, plt, save

E1 = Path("outputs/p4_10/e1")
V7 = Path("outputs/p4_10/v7")
S1 = {"L5.74_W4.78_d0.62_w2.32": "med", "L6.16_W4.56_d0.34_w0.48": "lo"}
R1M_CK = "outputs/p4_10/arms/R1M/ckpt_iter0150000.pt"
FAMS = ("StretchU", "StretchA", "Wall", "Notch", "Source", "Material")
LAB = {"StretchU": "Stretch\n(uniform)", "StretchA": "Stretch\n(one axis)", "Wall": "Wall", "Notch": "Notch", "Source": "Source", "Material": "Material"}
ROUND = {"StretchU": "R1-S", "StretchA": "R1-S", "Wall": "R1", "Notch": "R1", "Source": "R1", "Material": "R1-M"}
KEY = {"R0": "R0_120K", "R1": "R1_150K", "R1-S": "R1S_150K", "R1-M": "R1M_150K"}
TAG = {"R0": "R0_full", "R1": "R1_full", "R1-S": "R1S_full", "R1-M": "R1M_full"}


def edc(H, band, sel):
    from aaf.eval.octave_decay import FS, GATE_S, _edc, _onset, octave_ir
    h = octave_ir(np.asarray(H)[sel], band)
    n = int(GATE_S * FS)
    on = _onset(h)
    seg = np.zeros((len(h), n))
    for r in range(len(h)):
        s = h[r, on[r]:on[r] + n]
        seg[r, :len(s)] = s
    return _edc(seg), seg, FS


def v6():
    from scripts.p4_10_modes import DF, half_width, hungarian, peaks, spec
    from scripts.p4_10_scorecard import on_lattice
    stem = "L5.74_W4.78_d0.62_w2.32"
    t = np.load(E1 / "truth" / (stem + ".npz"))
    rx, src, T = t["rx"], t["src"], t["H"]
    off = ~on_lattice(rx)
    far = off & (np.hypot(rx[:, 0] - src[0], rx[:, 1] - src[1]) >= 1.0)
    F = {"truth": T, "S-1 (one room)": np.load(E1 / "pred" / ("S1med__%s.npz" % stem))["P"], "R1 (1,000 rooms)": np.load(E1 / "pred" / ("R1_150K__%s.npz" % stem))["P"]}
    col = {"truth": COL["truth"], "S-1 (one room)": COL["S-1"], "R1 (1,000 rooms)": COL["R1"]}
    ls = {"truth": "-", "S-1 (one room)": (0, (5, 2)), "R1 (1,000 rooms)": "-"}
    rep = json.loads((E1 / "report.json").read_text())
    r = next(x for x in rep["rooms"] if x["room"] == stem)["models"]
    meta = {"room": stem, "n_rx_spectrum": int(off.sum()), "n_rx_decay": int(far.sum()), "width_ratio": {"S-1": r["S1med"]["D2"]["ratio_median"], "R1": r["R1_150K"]["D2"]["ratio_median"]},
            "peaks_matched": {"S-1": [r["S1med"]["D2"]["matched"], r["S1med"]["D2"]["truth_peaks"]], "R1": [r["R1_150K"]["D2"]["matched"], r["R1_150K"]["D2"]["truth_peaks"]]},
            "EDT_err": {m: {b: r[k]["D1"][b]["EDT_err"] for b in ("63", "125")} for m, k in (("S-1", "S1med"), ("R1", "R1_150K"))}, "source": "outputs/p4_10/e1/report.json"}
    fig = new(16, 9)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], hspace=0.42, wspace=0.2, left=0.07, right=0.985, top=0.86, bottom=0.21)
    ax = fig.add_subplot(gs[:, 0])
    f = np.arange(601) * DF
    lo, hi = 50, 200
    ref = None
    for k, H in F.items():
        S = spec(np.asarray(H)[off])
        ref = S[lo:hi + 1].max() if ref is None else ref
        y = 10 * np.log10(S / ref)
        ax.plot(f[lo:hi + 1], y[lo:hi + 1], color=col[k], lw=3.0 if k == "truth" else 2.4, ls=ls[k], label=k, zorder=3 if k == "truth" else 2)
        meta["spectrum|" + k] = [float(x) for x in y[lo:hi + 1]]
    ax.set_xlim(25, 100)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("level, averaged over the room (dB)")
    ax.set_title("Resonances: truth, one-room model, shared model", loc="left")
    ax.grid(True, color="#e6e6e6")
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.02))
    ax.text(0.24, 0.76, "resonance width relative to the truth:\none-room model {:.2f}, shared model {:.2f}".format(meta["width_ratio"]["S-1"], meta["width_ratio"]["R1"]), transform=ax.transAxes, fontsize=15,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#bbbbbb"))
    for j, b in enumerate(("63", "125")):
        a = fig.add_subplot(gs[j, 1])
        for k, H in F.items():
            L, seg, FS = edc(H, b, far)
            y = np.median(L, 0)
            tt = np.arange(len(y)) / FS
            a.plot(tt, y, color=col[k], lw=3.0 if k == "truth" else 2.4, ls=ls[k], label=k)
            meta["edc|{}|{}".format(b, k)] = {"t_s": [0.0, 0.25, 0.5, 0.75, 1.0], "dB": [float(y[int(x * FS)]) for x in (0.0, 0.25, 0.5, 0.75, 1.0)]}
        a.set_xlim(0, 1.2)
        a.set_ylim(-60, 1)
        a.set_title("{} Hz octave: energy decay".format(b), loc="left")
        a.set_ylabel("energy left (dB)")
        a.grid(True, color="#e6e6e6")
        if j == 1:
            a.set_xlabel("time after the sound arrives (s)")
    fig.suptitle("Why decay is off: a one-room model gets the early decay right, the shared model blurs the resonances", fontsize=19, x=0.01, ha="left", y=0.965)
    fig.text(0.01, 0.02, ("Room L5.74 x W4.78 (one of two). Early decay time error at 63 / 125 Hz: shared model {:.0%} / {:.0%}, one-room model {:.1%} / {:.1%}.\n"
                          "Scored off the training positions: {} receivers for the resonances, {} (at least 1 m from the source) for the decay. Only two one-room\n"
                          "models, each with about 83 times more training per room: sharing one model and training time per room are not separated here.").format(
        meta["EDT_err"]["R1"]["63"], meta["EDT_err"]["R1"]["125"], meta["EDT_err"]["S-1"]["63"], meta["EDT_err"]["S-1"]["125"], int(off.sum()), int(far.sum())), fontsize=14, color="#333333", va="bottom")
    save(fig, "V6_why_decay_is_off", meta)


def v3d():
    from scripts.p4_10_scorecard import SC, _slope, cells, cid, material_mm
    pub = json.loads(Path("outputs/p4_10/verdict_R1M_150K.json").read_text())["MM1"]["in_family_full"]
    fig = new(16, 9)
    ax = fig.add_axes([0.07, 0.22, 0.48, 0.66])
    meta, lim = {"source": "cells JSON behind verdict_R1M_150K.json (R1-M) and material_mm (R1)"}, 0.0
    mk = {"63": "o", "125": "s", "250": "^"}
    side = fig.add_axes([0.575, 0.22, 0.415, 0.64])
    side.axis("off")
    txt = []
    for name, lab in (("R1_150K", "R1"), ("R1M_150K", "R1-M")):
        pts = {b: [] for b in ("63", "125", "250")}
        for c in cells(["Material"], False):
            if c["group"] != "test":
                continue
            d = json.loads((SC / name / (cid(c) + ".json")).read_text())["decay"]
            assert d.get("version") == "d202"
            for b in pts:
                e = d[b]
                pts[b].append((e["rate_Te"] - e["rate_T0"], e["rate_Pe"] - e["rate_P0"]))
        s = _slope([q for v in pts.values() for q in v])
        mm = material_mm(name, False, "test")["EDT"]
        assert abs(s["slope"] - mm["slope"]) < 1e-12 and abs(s["pearson"] - mm["pearson"]) < 1e-12
        if name == "R1M_150K":
            assert abs(s["slope"] - pub["slope"]) < 1e-9 and abs(s["pearson"] - pub["pearson"]) < 1e-9, (s, pub)
        for b, v in pts.items():
            v = np.array(v)
            ax.scatter(v[:, 0], v[:, 1], s=62, marker=mk[b], color=COL[lab], alpha=0.75, edgecolors="white", linewidths=0.6, zorder=3 if lab == "R1-M" else 2)
            lim = max(lim, float(np.nanmax(np.abs(v))))
        meta[lab] = {"slope": s["slope"], "pearson": s["pearson"], "n_cell_bands": s["n"], "points": {b: [[float(x) for x in q] for q in v] for b, v in pts.items()}}
        txt.append((lab, s))
    lo, hi = -0.7, round(lim * 1.06, 2)
    ax.plot([lo, hi], [lo, hi], color="#111111", lw=2.0, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="#cccccc", lw=1)
    ax.axvline(0, color="#cccccc", lw=1)
    ax.set_xlabel("true change in decay rate, 1 / EDT (1/s)")
    ax.set_ylabel("predicted change (1/s)")
    ax.text(0.9, 1.35, "exact (y = x)", fontsize=15, ha="right", color="#111111")
    y = 0.93
    side.text(0, y, "Change the wall absorption: how much\ndoes the decay rate change?", fontsize=18, fontweight="bold", va="top")
    y -= 0.28
    for lab, s in txt:
        side.plot([0.02], [y - 0.02], "o", ms=16, color=COL[lab], clip_on=False)
        side.text(0.10, y, "{}:  slope {:.2f},  Pearson {:.2f}".format(lab, s["slope"], s["pearson"]), fontsize=17, va="center")
        y -= 0.13
    side.text(0, y - 0.02, "Slope 1 = the full true change. Each point is one wall\nedit in one known room at one octave:\ncircle 63 Hz, square 125 Hz, triangle 250 Hz.\n{} points per model.".format(meta["R1-M"]["n_cell_bands"]), fontsize=15, va="top")
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    fig.suptitle("A model that responds to wall materials: right direction, nearly proportional, about half the true size", fontsize=19, x=0.01, ha="left", y=0.965)
    fig.text(0.01, 0.03, "15 known rooms, 4 wall edits each, 3 octaves; scored off the training positions, at least 1 m from the source. The bar set before training was a slope of 0.8 to 1.2\nwith Pearson at least 0.8: R1-M does not reach it (slope). R1 was not trained on materials and hardly responds.", fontsize=14, color="#333333", va="bottom")
    save(fig, "V3d_mm1_scatter", meta)


def v8():
    d = {m: json.loads(Path("outputs/p4_10/scorecard_{}.json".format(TAG[m])).read_text())["families"] for m in ("R0", "R1", "R1-S", "R1-M")}
    fig = new(16, 9)
    meta = {"source": "outputs/p4_10/scorecard_{R0,R1,R1S,R1M}_full.json (edited-field r, mean over rooms)"}
    axs = []
    for i, (g, gl) in enumerate((("test", "Known rooms (15)"), ("unseen", "New rooms (16)"))):
        for j, b in enumerate(("25-100", "100-200")):
            ax = fig.add_axes([0.055 + j * 0.485, 0.585 - i * 0.435, 0.44, 0.31])
            axs.append(ax)
            x = np.arange(len(FAMS))
            v0 = [d["R0"]["{}|{}".format(f, g)][KEY["R0"]]["edited_r"][b][0] for f in FAMS]
            v1 = [d[ROUND[f]]["{}|{}".format(f, g)][KEY[ROUND[f]]]["edited_r"][b][0] for f in FAMS]
            ig = [d["R0"]["{}|{}".format(f, g)][KEY["R0"]]["ignore_r"][b][0] for f in FAMS]
            ax.bar(x - 0.2, v0, 0.38, color=COL["R0"])
            for q, f in enumerate(FAMS):
                ax.bar(q + 0.2, v1[q], 0.38, color=COL[ROUND[f]])
                ax.plot([q - 0.42, q + 0.42], [ig[q]] * 2, color=COL["ignore"], lw=4.2, solid_capstyle="butt", zorder=5)
            for q in range(len(FAMS)):
                for xx, val, bold in ((q - 0.2, v0[q], False), (q + 0.2, v1[q], True)):     # inside the bar at its base; a bar too short for it takes it above
                    inside = val >= 0.16
                    ax.text(xx, 0.02 if inside else val + 0.015, "{:.2f}".format(val).lstrip("0"), ha="center", va="bottom", fontsize=15, color="white" if inside else "#333333", fontweight="bold" if bold else "normal", zorder=6)
            ax.set_ylim(0, 1.16)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_xticks(x)
            ax.set_xticklabels([LAB[f] for f in FAMS] if i == 1 else [""] * len(FAMS), fontsize=14)
            ax.grid(True, axis="y", color="#e6e6e6")
            ax.set_axisbelow(True)
            ax.set_title("{}, {} Hz".format(gl, b), loc="left", fontsize=17, pad=8)
            ax.set_ylabel("accuracy" if j == 0 else "")
            for q, f in enumerate(FAMS):
                meta["{}|{}|{}".format(g, b, f)] = {"R0": v0[q], "this_round_model": ROUND[f], "this_round": v1[q], "ignore_the_edit": ig[q]}
    h = [plt.Rectangle((0, 0), 1, 1, color=COL["R0"]), plt.Rectangle((0, 0), 1, 1, color=COL["R1-S"]), plt.Rectangle((0, 0), 1, 1, color=COL["R1"]), plt.Rectangle((0, 0), 1, 1, color=COL["R1-M"]), plt.Line2D([], [], color=COL["ignore"], lw=4.2)]
    fig.legend(h, ["R0 (start)", "this round: R1-S", "this round: R1", "this round: R1-M", "ignore the edit (the original room's true field)"], loc="lower center", ncol=5, frameon=False, fontsize=15, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("How accurate is each edit? (correlation of the sound-level maps with a simulation of the edited room)", fontsize=19, x=0.01, ha="left", y=0.985)
    save(fig, "V8_scorecards", meta)


# ------------------------------------------------------------------------------------------------------------ V7
def render_r1m():
    from scripts.p4_10_scorecard import Renderer
    (V7 / "pred").mkdir(parents=True, exist_ok=True)
    rd = Renderer("R1M_150K", R1M_CK)
    for stem in S1:
        t = np.load(E1 / "truth" / (stem + ".npz"))
        V = [tuple(map(float, v)) for v in t["verts"]]
        P = rd.render(V, float(t["L"]), float(t["W"]), [0.15] * len(V), tuple(t["src"]), t["rx"])
        np.savez_compressed(V7 / "pred" / "R1M_150K__{}.npz".format(stem), P=P)
        print("rendered", stem, len(P), flush=True)


def load_rooms():
    """(group, room, model label, P, T, rx, src) for every field V7 reads"""
    from scripts.p4_10_d1_decay import SRC, rooms_of
    out = []
    for lab, name in (("R1", "R1_150K"), ("R1-M", "R1M_150K")):
        for j, (P, T, rx) in enumerate(rooms_of(name, "test")):
            out.append(("in-family 15", "room%02d" % j, lab, P, T, rx, SRC))
    for stem, tag in S1.items():
        t = np.load(E1 / "truth" / (stem + ".npz"))
        for lab, f in (("R1", E1 / "pred" / ("R1_150K__%s.npz" % stem)), ("R1-M", V7 / "pred" / ("R1M_150K__%s.npz" % stem)), ("S-1", E1 / "pred" / ("S1%s__%s.npz" % (tag, stem)))):
            if f.exists():
                out.append(("S-1 rooms", stem, lab, np.load(f)["P"], t["H"], t["rx"], t["src"]))
            else:
                print("missing", f)
    return out


def v7():
    from aaf.eval.octave_decay import FS, GATE_S, decay_metrics, valid_mask
    from scripts.p4_10_lundeby import edc_db, fit, lundeby, times  # noqa: F401
    from scripts.p4_10_scorecard import on_lattice
    rows, panel = [], {}
    KEYS = ("EDT", "T20", "T30")
    for g, room, lab, P, T, rx, src in load_rooms():
        sel = np.nonzero(~on_lattice(rx) & (np.hypot(rx[:, 0] - src[0], rx[:, 1] - src[1]) >= 1.0))[0][::4]
        P, T = np.asarray(P)[sel], np.asarray(T)[sel]
        row = {"group": g, "room": room, "model": lab, "bands": {}}
        for b in ("63", "125", "250"):
            t, m = decay_metrics(T, b), decay_metrics(P, b)
            ok = valid_mask(t["t35"])
            _, seg, fs = edc(P, b, slice(None))
            _, segT, _ = edc(T, b, slice(None))
            raw = {k: np.full(len(P), np.nan) for k in KEYS}
            cmp_ = {k: np.full(len(P), np.nan) for k in KEYS}
            fail, below20, below30, curves = 0, 0, 0, {"raw": [], "comp": [], "truth": []}
            can = np.zeros(len(P), bool)
            for i in np.nonzero(ok)[0]:
                Lr, _ = edc_db(seg[i], fs, False)
                Lc, info = edc_db(seg[i], fs, True)
                r = times(Lr, fs)
                for k in KEYS:
                    raw[k][i] = r[k]
                    assert (np.isnan(r[k]) and np.isnan(m[k][i])) or abs(r[k] - m[k][i]) < 1e-9 * max(1.0, abs(m[k][i])), (lab, b, i, k, r[k], m[k][i])
                if Lc is None:
                    fail += 1
                    continue
                can[i] = True
                below20 += int(info["peak_to_noise_db"] < 35.0)
                below30 += int(info["peak_to_noise_db"] < 45.0)
                c = times(Lc, fs)
                for k in KEYS:
                    cmp_[k][i] = c[k]
                if lab == "R1" and g == "in-family 15" and b == "125":
                    curves["raw"].append(Lr)
                    curves["comp"].append(Lc)
                    curves["truth"].append(edc_db(segT[i], fs, False)[0])
            e = {"n_valid": int(ok.sum()), "n_not_compensable": fail, "n_below_ISO_T20_range": below20, "n_below_ISO_T30_range": below30}
            for k in KEYS:
                kp = ok & can & np.isfinite(t[k]) & np.isfinite(raw[k]) & np.isfinite(cmp_[k])
                rel = lambda x: float(np.nanmedian(np.abs(x[kp] - t[k][kp]) / t[k][kp])) if kp.any() else np.nan  # noqa: E731
                e[k] = {"n": int(kp.sum()), "truth": float(np.nanmedian(t[k][kp])) if kp.any() else np.nan, "before": rel(raw[k]), "after": rel(cmp_[k]),
                        "before_bias": float(np.nanmedian((raw[k][kp] - t[k][kp]) / t[k][kp])) if kp.any() else np.nan, "after_bias": float(np.nanmedian((cmp_[k][kp] - t[k][kp]) / t[k][kp])) if kp.any() else np.nan}
            if curves["raw"]:
                panel.setdefault("edcs", []).append({k: np.median(np.array(v), 0).tolist() for k, v in curves.items()})
            row["bands"][b] = e
        rows.append(row)
        print(g, room, lab, {b: (row["bands"][b]["n_not_compensable"], row["bands"][b]["n_valid"]) for b in row["bands"]}, flush=True)
    from scripts.p4_10_d1_decay import boot
    rep = {"definition": "D216", "groups": {}, "rows": rows, "panel_125_R1_in_family": {k: np.mean([e[k] for e in panel["edcs"]], 0).tolist() for k in ("raw", "comp", "truth")}, "fs": float(FS), "gate_s": GATE_S}
    L = ["# P4-10 V7: the late floor under ISO 3382-1 style compensation (D216; descriptive)", "",
         "Per room: receiver median of |model - truth| / truth; table entries are means over rooms (95% CI over rooms when there are at least 3). Same receivers before and after; receivers that cannot be compensated are dropped from both.", "",
         "| rooms | model | octave | receivers (valid in truth / not compensable) | EDT before -> after | T20 before -> after | T30 before -> after | receivers used for EDT / T20 / T30 |", "|---|---|---|---|---|---|---|---|"]
    for g in ("in-family 15", "S-1 rooms"):
        for lab in ("R1", "R1-M", "S-1"):
            sub = [r for r in rows if r["group"] == g and r["model"] == lab]
            if not sub:
                continue
            for b in ("63", "125", "250"):
                e = {"n_rooms": len(sub), "n_valid": sum(r["bands"][b]["n_valid"] for r in sub), "n_not_compensable": sum(r["bands"][b]["n_not_compensable"] for r in sub),
                     "n_below_ISO_T20_range": sum(r["bands"][b]["n_below_ISO_T20_range"] for r in sub), "n_below_ISO_T30_range": sum(r["bands"][b]["n_below_ISO_T30_range"] for r in sub)}
                for k in KEYS:
                    for w in ("before", "after", "before_bias", "after_bias"):
                        v = [r["bands"][b][k][w] for r in sub]
                        e["{}_{}".format(k, w)] = boot(v) if len(sub) >= 3 else [float(np.nanmean(v)), float("nan"), float("nan")]
                    e[k + "_n"] = sum(r["bands"][b][k]["n"] for r in sub)
                rep["groups"]["{}|{}|{}".format(g, lab, b)] = e
                p = lambda k: "{:.0%} -> {:.0%}".format(e[k + "_before"][0], e[k + "_after"][0])  # noqa: E731
                L.append("| {} ({}) | {} | {} Hz | {} / {} | {} | {} | {} | {} / {} / {} |".format(g, len(sub), lab, b, e["n_valid"], e["n_not_compensable"], p("EDT"), p("T20"), p("T30"), e["EDT_n"], e["T20_n"], e["T30_n"]))
    L += ["", "ISO 3382-1 range (model, peak to noise): receivers below 35 dB (T20) and below 45 dB (T30), summed over rooms and octaves, per model: " + "; ".join("{}: {} / {}".format(lab, sum(e["n_below_ISO_T20_range"] for k, e in rep["groups"].items() if k.split("|")[1] == lab), sum(e["n_below_ISO_T30_range"] for k, e in rep["groups"].items() if k.split("|")[1] == lab)) for lab in ("R1", "R1-M", "S-1")),
          "", "The truth is not compensated. At 250 Hz the truth's own record is limited (floor about -40 dB), so T30 there is the weakest reading."]
    (V7 / "report.json").parent.mkdir(parents=True, exist_ok=True)
    (V7 / "report.json").write_text(json.dumps(rep, indent=1))
    (V7 / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def fig_v7():
    rep = json.loads((V7 / "report.json").read_text())
    G = rep["groups"]
    fig = new(16, 9)
    ax = fig.add_axes([0.065, 0.28, 0.4, 0.56])
    fs = rep["fs"]
    p = rep["panel_125_R1_in_family"]
    for k, lab, col, ls, lw in (("truth", "truth", COL["truth"], "-", 3.2), ("raw", "R1, as rendered", COL["R1"], "-", 2.6), ("comp", "R1, floor compensated", COL["R1"], (0, (5, 2)), 2.6)):
        y = np.array(p[k])
        ax.plot(np.arange(len(y)) / fs, y, color=col, ls=ls, lw=lw, label=lab)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-65, 1)
    ax.set_xlabel("time after the sound arrives (s)")
    ax.set_ylabel("energy left (dB)")
    ax.set_title("125 Hz octave, 15 known rooms (median receiver)", loc="left")
    ax.grid(True, color="#e6e6e6")
    ax.legend(loc="upper right", frameon=False)
    a2 = fig.add_axes([0.56, 0.28, 0.425, 0.56])
    meta = {"source": "outputs/p4_10/v7/report.json", "definition": "D216"}
    x, ticks = 0, []
    for b in ("63", "125", "250"):
        for k in ("EDT", "T20", "T30"):
            e = G["in-family 15|R1|{}".format(b)]
            v = (e[k + "_before"][0] * 100, e[k + "_after"][0] * 100)
            a2.bar(x - 0.2, min(v[0], 100), 0.38, color=COL["R1"])
            a2.bar(x + 0.2, min(v[1], 100), 0.38, color="white", edgecolor=COL["R1"], linewidth=2.2, hatch="//")
            for xx, t in ((x - 0.22, v[0]), (x + 0.22, v[1])):
                a2.text(xx, min(t, 100) + 1.5, "{:.0f}".format(t), ha="center", fontsize=14, fontweight="bold" if t > 100 else "normal")
            meta["R1|{}|{}".format(b, k)] = {"before_pct": v[0], "after_pct": v[1], "n_receivers": e[k + "_n"]}
            ticks.append((x, k))
            x += 1
        x += 0.7
    a2.set_xticks([t for t, _ in ticks])
    a2.set_xticklabels([k for _, k in ticks], fontsize=14)
    a2.set_ylabel("error against the truth (%)")
    ylim = 128.0
    a2.set_ylim(0, ylim)
    a2.set_yticks([0, 25, 50, 75, 100])
    a2.text(0.0, 108, "bars above 100% are cut; the number is the value", fontsize=14, color="#444444")
    for xx, b in zip((1.0, 4.7, 8.4), ("63 Hz", "125 Hz", "250 Hz")):
        a2.text(xx, ylim * 0.93, b, ha="center", fontsize=17, fontweight="bold")
    a2.set_title("R1: decay-time error before (solid) and after (hatched)", loc="left", fontsize=15)
    a2.grid(True, axis="y", color="#e6e6e6")
    a2.set_axisbelow(True)
    head = V7_HEAD(rep)
    fig.suptitle(head, fontsize=19, x=0.01, ha="left", y=0.965)
    S, B1 = G["S-1 rooms|S-1|125"], G["in-family 15|R1|125"]
    n30 = {q: G["in-family 15|R1|%s" % q]["n_below_ISO_T30_range"] / G["in-family 15|R1|%s" % q]["n_valid"] for q in ("125", "250")}
    fig.text(0.01, 0.015, ("Floor removed with the standard room-acoustics method (Lundeby truncation plus the energy of the fitted decay after it), models only. Same receivers\n"
                           "before and after, off the training positions, at least 1 m from the source. Check on the one-room models at 125 Hz: T20 {:.0%} to {:.0%}, T30 {:.0%} to {:.0%}:\n"
                           "the method works where the decay is right. EDT reads only the first 10 dB and hardly moves. T30 is indicative: {:.0%} / {:.0%} of R1's receivers at\n"
                           "125 / 250 Hz are below the 45 dB range ISO asks for. The dashed curve continues as the fitted decay after the truncation point.").format(
        S["T20_before"][0], S["T20_after"][0], S["T30_before"][0], S["T30_after"][0], n30["125"], n30["250"]), fontsize=14, color="#333333", va="bottom")
    meta["headline"] = head
    save(fig, "V7_late_floor", meta)


def V7_HEAD(rep):
    G = rep["groups"]
    b = "125"
    e = G["in-family 15|R1|125"]
    return "Removing the late floor does not fix the decay: R1 at 125 Hz, EDT error {:.0%} to {:.0%}, T20 error {:.0%} to {:.0%}".format(e["EDT_before"][0], e["EDT_after"][0], e["T20_before"][0], e["T20_after"][0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="*", default=[])
    ap.add_argument("--render_r1m", action="store_true")
    ap.add_argument("--v7", action="store_true")
    a = ap.parse_args()
    if a.render_r1m:
        render_r1m()
    if a.v7:
        v7()
    for f in a.fig:
        {"V6": v6, "V3d": v3d, "V8": v8, "V7": fig_v7}[f]()
