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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", nargs="*", default=["F1", "F7"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for x in a.fig:
        globals()[x.lower()]()
