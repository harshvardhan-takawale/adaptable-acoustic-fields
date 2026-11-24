"""P4-10 meeting figures from posted JSON (D195 reporting list). Every number is read from the scorecard aggregates; the PNG's sidecar JSON holds the plotted values.
  python scripts/p4_10_meeting_figs.py --fig F1 F7
F1: R0's scorecard (edited-field r and r_delta per family and band, in-family and unseen, with ignore-the-edit and m1 marks). F7: the scorecard after: R0 vs R1 vs R1-S vs R1-M, F1's layout.
Colours (entity, fixed): R0 neutral grey (reference), R1 #2a78d6, R1-S #1baf7a, R1-M #c64f97 (validated, light surface); bars carry direct labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = Path("outputs/p4_10")
OUT = Path("outputs/p4_10/meeting")
FAM = ("StretchU", "StretchA", "Wall", "Notch", "Source", "Material")
LAB = {"StretchU": "Stretch\nuniform", "StretchA": "Stretch\none axis", "Wall": "Wall", "Notch": "Notch", "Source": "Source\n(2 pos.)", "Material": "Material"}
B = ("25-100", "100-200", "200-300")
COL = {"R0_120K": "#7a7a7a", "R1_150K": "#2a78d6", "R1S_150K": "#1baf7a", "R1M_150K": "#c64f97"}
NAME = {"R0_120K": "R0", "R1_150K": "R1", "R1S_150K": "R1-S", "R1M_150K": "R1-M"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888888", "figure.facecolor": "white"})


def fam_json(tag):
    return json.loads((SRC / "scorecard_{}.json".format(tag)).read_text())["families"]


def style(ax):
    ax.grid(True, axis="y", color="#e8e8e8", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-0.1, 1.1)


def f1():
    d = fam_json("R0_full")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), dpi=140, sharey="row")
    meta = {}
    for row, met in enumerate(("edited_r", "r_delta")):
        for j, b in enumerate(B):
            a = ax[row, j]
            style(a)
            x = range(len(FAM))
            for k, (g, hatch, lab, dx) in enumerate((("test", None, "known rooms (15)", -0.2), ("unseen", "//", "new rooms (16)", 0.2))):
                v, ig, m1 = [], [], []
                for f in FAM:
                    e = d["{}|{}".format(f, g)]["R0_120K"]
                    v.append(e[met][b][0])
                    ig.append(e["ignore_r"][b][0] if met == "edited_r" else None)
                    m1.append(e["m1_r"][b][0] if met == "edited_r" else None)
                a.bar([i + dx for i in x], v, 0.36, color=COL["R0_120K"] if hatch is None else "#c4c4c4", edgecolor="#7a7a7a", hatch=hatch, label="R0, " + lab)
                for i, t in enumerate(v):
                    a.text(i + dx, max(t, 0) + 0.015, "{:.2f}".format(t), ha="center", fontsize=6.5, color="#555555", rotation=90)
                if met == "edited_r":
                    for i in x:
                        if ig[i] is not None:
                            a.plot([i + dx - 0.16, i + dx + 0.16], [ig[i]] * 2, color="#b00020", lw=2.0, zorder=5, label="ignore the edit" if (i == 0 and j == 0 and row == 0 and k == 0) else None)
                        if m1[i] is not None and m1[i] == m1[i]:
                            a.plot([i + dx], [m1[i]], "D", ms=4.5, color="#111111", zorder=5, label="m1 physics" if (i == 0 and j == 0 and row == 0 and k == 0) else None)
                meta["{}|{}|{}".format(met, b, g)] = dict(zip(FAM, v))
            a.set_xticks(list(x))
            a.set_xticklabels([LAB[f] for f in FAM], fontsize=8)
            a.set_title(("Edited-field r, " if row == 0 else "Pattern of the change, r_delta, ") + b + " Hz", fontsize=11, loc="left")
            if row == 1:
                a.set_ylim(-0.1, 1.02)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("F1  R0: which edits can it make? Grey = known rooms, hatched = new rooms; red tick = field that ignores the edit, diamond = physics at the edited geometry (m1; none for material)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "F1_r0_scorecard.png", facecolor="white")
    (OUT / "F1_r0_scorecard.json").write_text(json.dumps({"source": "scorecard_R0_full.json", "values": meta}, indent=1))
    print("wrote F1")


def f7():
    d = {m: fam_json(t) for m, t in (("R0_120K", "R0_full"), ("R1_150K", "R1_full"), ("R1S_150K", "R1S_full"), ("R1M_150K", "R1M_full"))}
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), dpi=140, sharey="row")
    meta = {}
    for row, met in enumerate(("edited_r", "r_delta")):
        for j, b in enumerate(B):
            a = ax[row, j]
            style(a)
            for k, m in enumerate(d):
                v = [d[m]["{}|{}".format(f, "test")][m][met][b][0] for f in FAM]
                xs = [i + (k - 1.5) * 0.2 for i in range(len(FAM))]
                a.bar(xs, v, 0.18, color=COL[m], label=NAME[m])
                meta["{}|{}|{}|in-family".format(met, b, NAME[m])] = dict(zip(FAM, v))
                for x0, t in zip(xs, v):
                    a.text(x0, max(t, 0) + 0.012, "{:.2f}".format(t), ha="center", fontsize=5.5, color="#555555", rotation=90)
                if met == "edited_r":
                    vu = [d[m]["{}|{}".format(f, "unseen")][m][met][b][0] for f in FAM]
                    a.scatter(xs, vu, marker="_", s=130, color="white", linewidths=4.6, zorder=4)
                    a.scatter(xs, vu, marker="_", s=130, color="#111111", linewidths=1.8, zorder=5, label="new rooms (unseen)" if (k == 0 and j == 0) else None)
                    meta["{}|{}|{}|unseen".format(met, b, NAME[m])] = dict(zip(FAM, vu))
            if met == "edited_r":
                a.plot(range(len(FAM)), [d["R0_120K"]["{}|test".format(f)]["R0_120K"]["ignore_r"][b][0] for f in FAM], "r_", ms=26, mew=2.2, label="ignore the edit (known rooms)" if j == 0 else None)
            a.set_xticks(range(len(FAM)))
            a.set_xticklabels([LAB[f] for f in FAM], fontsize=8)
            a.set_title(("Edited-field r, " if row == 0 else "Pattern of the change, r_delta, ") + b + " Hz", fontsize=11, loc="left")
            if row == 1:
                a.set_ylim(-0.1, 1.02)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("F7  The scorecard after training: bars = known rooms (R0 grey, R1 blue, R1-S green, R1-M magenta); black ticks = new rooms (edited r only); red = ignoring the edit (known rooms)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "F7_scorecard_after.png", facecolor="white")
    (OUT / "F7_scorecard_after.json").write_text(json.dumps({"source": "scorecard_{R0,R1,R1S,R1M}_full.json", "values": meta}, indent=1))
    print("wrote F7")

def _edc_db(h, fs, gate_s=1.5):
    import numpy as np
    n = int(gate_s * fs)
    on = np.argmax(np.abs(h) >= 0.1 * np.abs(h[:, : int(0.4 * fs)]).max(1, keepdims=True), axis=1)
    out = np.zeros((len(h), n))
    for r in range(len(h)):
        seg = h[r, on[r]:on[r] + n]
        out[r, : len(seg)] = seg
    e = np.cumsum(out[:, ::-1] ** 2, axis=1)[:, ::-1]
    return 10 * np.log10(np.maximum(e / np.maximum(e[:, :1], 1e-300), 1e-9))


def f5():
    import numpy as np
    from aaf.eval.octave_decay import FS, octave_ir
    from scripts.p4_10_scorecard import PRED, TRUTH, cells, cid, on_lattice, stored
    MODELS = (("R1_150K", "R1"), ("R1M_150K", "R1-M"))
    best = None
    for c in cells(["Material"], False):
        if c["group"] != "test":
            continue
        f = TRUTH / (cid(c) + ".npz")
        if not all((PRED / m / (cid(c) + ".npz")).exists() for m, _ in MODELS) or not f.exists():
            continue
        z, b = np.load(f), stored(c)
        k = np.where(~on_lattice(b["rx"]))[0][::40]
        r0 = np.median(_edc_db(octave_ir(b["H"][k], "125"), FS)[:, int(0.25 * FS)])
        r1 = np.median(_edc_db(octave_ir(z["Hd"][k], "125"), FS)[:, int(0.25 * FS)])
        if best is None or (r0 - r1) > best[0]:
            best = (r0 - r1, c, z, b)
    diff, c, z, b = best
    rx = z["rx"]
    k = np.where(~on_lattice(rx) & (np.hypot(rx[:, 0] - 0.5, rx[:, 1] - 1.6) >= 1.0))[0][::4]
    op = {m: np.load(PRED / m / "orig" / (c["room"] + ".npz"))["P"] for m, _ in MODELS}
    pe = {m: np.load(PRED / m / (cid(c) + ".npz"))["Pd"] for m, _ in MODELS}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), dpi=140, sharey=True)
    meta = {"cell": c["id"], "rule": "in-family Material cell with the largest truth level drop at 0.25 s in the 125 Hz octave"}
    for j, band in enumerate(("63", "125", "250")):
        a = ax[j]
        t = np.arange(int(1.5 * FS)) / FS
        curves = {"truth, before the edit": (np.median(_edc_db(octave_ir(b["H"][k], band), FS), 0), "#9a9a9a", (0, (5, 3))),
                  "truth, after the edit": (np.median(_edc_db(octave_ir(z["Hd"][k], band), FS), 0), "#111111", "-"),
                  "R1, after the edit": (np.median(_edc_db(octave_ir(pe["R1_150K"][k], band), FS), 0), "#2a78d6", "-"),
                  "R1-M, after the edit": (np.median(_edc_db(octave_ir(pe["R1M_150K"][k], band), FS), 0), "#c64f97", "-")}
        for lab, (y, col, ls) in curves.items():
            a.plot(t, y, color=col, ls=ls, lw=2.0, label=lab if j == 0 else None)
            meta["{}|{}".format(band, lab)] = [float(y[int(x * FS)]) for x in (0.1, 0.25, 0.5, 1.0)]
        a.set_title("{} Hz octave".format(band), loc="left")
        a.set_xlabel("time after onset (s)")
        a.set_ylim(-60, 2)
        a.set_xlim(0, 1.2)
        style(a)
        a.set_ylim(-60, 2)
        a.grid(True, color="#e8e8e8")
    ax[0].set_ylabel("energy decay (dB, median over receivers)")
    ax[1].legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("F5  A wall-absorption edit in a known room ({}; all walls to absorption 0.30): decay of the true field and of the models (median over {} receivers)".format(c["room"], len(k)), fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "F5_material_decay.png", facecolor="white")
    (OUT / "F5_material_decay.json").write_text(json.dumps(meta, indent=1))
    print("wrote F5", c["id"], "drop difference %.1f dB" % diff)


def f3():
    import numpy as np
    from scripts.p4_10_scorecard import PRED, TRUTH, cells, cid
    from matplotlib.colors import Normalize
    ROWS = (("truth", None), ("R0", "R0_120K"), ("RCP", "RCP_150K"), ("R1", "R1_150K"))
    FREQS = (50.0, 200.0)
    cell = {c["mag"]: c for c in cells(["Source"], False) if c["group"] == "test" and c["room"] == "L5.36_W4.34_d1.88_w1.40" and c["mag"] in ("second", "third")}
    fig, ax = plt.subplots(len(ROWS), 4, figsize=(13.5, 11), dpi=130, gridspec_kw={"wspace": 0.06, "hspace": 0.12})
    meta = {"room": "L5.36_W4.34_d1.88_w1.40", "rule": "first in-family room of the manifest; bins at 50 and 200 Hz; colour = level minus the truth's 99th percentile of that column (dB)"}
    for j, (mag, fr) in enumerate([(m, f) for m in ("second", "third") for f in FREQS]):
        c = cell[mag]
        z = np.load(TRUTH / (cid(c) + ".npz"))
        rx, b = z["rx"], int(round(fr / 0.5))
        Ht = z["Hd"][:, b]
        ref = np.percentile(20 * np.log10(np.abs(Ht) + 1e-9), 99)
        for i, (lab, m) in enumerate(ROWS):
            a = ax[i, j]
            h = Ht if m is None else np.load(PRED / m / (cid(c) + ".npz"))["Pd"][:, b]
            v = 20 * np.log10(np.abs(h) + 1e-9) - ref
            s_ = a.scatter(rx[:, 0], rx[:, 1], c=v, s=8, cmap="viridis", norm=Normalize(-40, 0), marker="s", linewidths=0)
            a.plot(c["src"][0], c["src"][1], "*", color="white", ms=13, markeredgecolor="black", markeredgewidth=0.8)
            a.set_aspect("equal")
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title("{} source, {:.0f} Hz".format(mag, fr), fontsize=11, loc="left")
            if j == 0:
                a.set_ylabel(lab, fontsize=12)
            meta["{}|{}|{:.0f}Hz".format(lab, mag, fr)] = {"corr_of_magnitude_with_truth": float(np.corrcoef(np.abs(h), np.abs(Ht))[0, 1])}
    cb = fig.colorbar(s_, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("level relative to the truth's loud receivers in that column (dB)")
    fig.suptitle("F3  Moving the source in a known room: the level map at 50 and 200 Hz (white star = the moved source)", fontsize=12, x=0.01, ha="left", y=0.97)
    fig.savefig(OUT / "F3_moved_source_maps.png", facecolor="white")
    (OUT / "F3_moved_source_maps.json").write_text(json.dumps(meta, indent=1))
    print("wrote F3")


def f4():
    import numpy as np
    from scripts.p4_10_scorecard import PRED, TRUTH, cells, cid, db, stored
    c = next(x for x in cells(["StretchU"], False) if x["group"] == "test" and x["room"] == "L5.36_W4.34_d1.88_w1.40" and x["mag"] == "1.10")
    z = np.load(TRUTH / (cid(c) + ".npz"))
    ci, rx = z["common_idx"], z["common"]
    base = stored(c)
    clip = lambda x: np.clip(x, -20, 20)  # noqa: E731
    fig = plt.figure(figsize=(15, 8), dpi=130)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.1], wspace=0.15, hspace=0.18)
    meta = {"cell": c["id"], "rule": "in-family room L5.36, Stretch-U s=1.10, delta of the dB level at the common receivers (clipped at +/-20 dB, the scorer's r_delta definition)"}
    for r, fr in enumerate((125.0, 200.0)):
        b = int(round(fr / 0.5))
        dT = clip(db(z["Hc"]) - db(base["H"][ci]))[:, b]
        cols = [("truth delta", dT)]
        for m, lab in (("R1_150K", "R1 delta"), ("R1S_150K", "R1-S delta")):
            p_, o_ = np.load(PRED / m / (cid(c) + ".npz")), np.load(PRED / m / "orig" / (c["room"] + ".npz"))
            cols.append((lab, clip(db(p_["Pc"]) - db(o_["P"][ci]))[:, b]))
        for k, (lab, v) in enumerate(cols):
            a = fig.add_subplot(gs[r, k])
            s_ = a.scatter(rx[:, 0], rx[:, 1], c=v, s=9, cmap="RdBu_r", vmin=-15, vmax=15, marker="s", linewidths=0)
            a.set_aspect("equal")
            a.set_xticks([])
            a.set_yticks([])
            a.set_title("{}  {:.0f} Hz".format(lab, fr), fontsize=10, loc="left")
            if k > 0:
                meta["{}|{:.0f}Hz|corr_with_truth_delta".format(lab, fr)] = float(np.corrcoef(v, dT)[0, 1])
        if r == 1:
            continue
        a = fig.add_subplot(gs[:, 3])
        fams = ("StretchU", "StretchA", "Wall")
        js = {m: json.loads((SRC / "modes" / "a8_{}.json".format(m)).read_text()) for m in ("R0_120K", "R1_150K", "R1S_150K")}
        for i, f in enumerate(fams):
            for q, (m, col) in enumerate((("R0_120K", COL["R0_120K"]), ("R1_150K", COL["R1_150K"]), ("R1S_150K", COL["R1S_150K"]))):
                e = js[m]["{}|in-family".format(f)]["25_150"]
                a.plot(i + (q - 1) * 0.22, e["slope"], "o", ms=9, color=col, markeredgecolor="white", label=(NAME[m] if i == 0 and r == 0 else None))
                meta["modeshift|{}|{}".format(f, NAME[m])] = e["slope"]
        a.axhline(1.0, color="#111111", lw=1.2, ls="--")
        a.set_xticks(range(3))
        a.set_xticklabels([LAB[f] for f in fams], fontsize=9)
        a.set_ylim(0, 1.2)
        a.set_title("mode-shift slope, known rooms (1 = right)", fontsize=10, loc="left")
        style(a)
        a.set_ylim(0, 1.2)
        a.legend(frameon=False, fontsize=9, loc="lower right")
    cax = fig.add_axes([0.08, 0.05, 0.4, 0.02])
    fig.colorbar(s_, cax=cax, orientation="horizontal", label="change in level (dB)")
    fig.suptitle("F4  A uniform stretch (s = 1.10) in a known room: change in level, truth against the models; right: how the models shift the resonances (A8, 25-150 Hz)", fontsize=11, x=0.01, ha="left")
    fig.savefig(OUT / "F4_stretch_delta.png", facecolor="white")
    (OUT / "F4_stretch_delta.json").write_text(json.dumps(meta, indent=1))
    print("wrote F4", c["id"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="*", default=["F1", "F7"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for x in a.fig:
        globals()[x.lower()]()