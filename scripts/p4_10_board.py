"""P4-10 progress board (D195 / D196): outputs/p4_10/progress/board.{json,md} and F2_trajectories.png, rebuilt after every read.

  --arm NAME   add / refresh one arm's reads (NAME_125K, ... as they exist); without it, rebuild everything on disk.
Table A: static accuracy per read (HF, UF) with the delta vs the arm's reference. Table B: the arm's primary metric per
read. Table C: the latest scorecard headline per model and family. Every number is read from JSON, never typed.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.p4_10_scorecard import BAND, SC, boot, cells, cid, material_slope, source_gap
from scripts.p4_8_primary_table import load, paired

ARMS = Path("outputs/p4_10/arms")
OUT = Path("outputs/p4_10/progress")
B = list(BAND)
# arm -> reference. R1 is read against CTL @150K (the same steps from R0); every other arm against R1 at the matched step.
def ref_of(arm, step):
    return "CTL_150K" if arm == "R1" else "R1_{}K".format(step // 1000)


PRIMARY = {"R1": "in-family moved-source gap, band mean, second and third source (bar: each band <= 0.10)",
           "R1S": "r_delta on Stretch (lite) cells, in-family, paired vs R1 at the same step (bar: +0.05, CI > 0, >= 2 of 3 bands)",
           "R1M": "EDT decay-rate edit slope on Material (lite) cells, MM1 (D201/D202; bar: slope in [0.8, 1.2], Pearson >= 0.8)",
           "R1C": "unseen UF 25-100 Hz, paired vs R1 (bar: +0.03, CI > 0)", "R1P": "raw fine-scale energy ratio at 200-300 Hz (<= 1.2x FDTD and <= 0.75x R1)"}


def reads(arm):
    d = ARMS / arm
    out = []
    for p in sorted(d.glob("ckpt_iter*.pt")) if d.exists() else []:
        it = int(p.stem[-7:])
        if it in (125000, 130000, 135000, 140000, 145000, 150000):
            out.append((it, "{}_{}K".format(arm, it // 1000)))
    return out


def arm_rows(arm):
    rows = []
    for it, name in reads(arm):
        row = {"name": name, "step": it, "static": None}
        a = load(name)
        ref = load(ref_of(arm, it))
        if a is not None:
            row["static"] = {"HF": a["test_r"]["200-300"], "UF": {b: a["uf"][b] for b in B}}
            if ref is not None:
                d = paired(a, ref)
                row["static"]["delta_vs_ref"] = {"HF": d["HF"], **{"UF_" + b: d["UF_" + b] for b in B}, "ref": ref_of(arm, it)}
        g = source_gap(name, lite=True)
        if g.get("test"):
            row["gap"] = {grp: {mag: {b: boot([v[b] for v in g[grp][mag].values()]) for b in B} for mag in g[grp]} for grp in g}
        if arm == "R1M":
            row["material"] = {g: material_slope(name, True, g) for g in ("test", "unseen")}
        if arm == "R1S":
            row["stretch"] = stretch_primary(name, ref_of(arm, it))
        rows.append(row)
    return rows


def stretch_primary(name, ref):
    """paired r_delta on the Stretch lite cells, arm - reference, in-family and unseen (the arm's SS1 metric)"""
    out = {}
    for g in ("test", "unseen"):
        per = {}
        for m in (name, ref):
            t = {}
            for fam in ("StretchU", "StretchA"):
                for c in cells([fam], True):
                    if c["group"] != g:
                        continue
                    f = SC / m / (cid(c) + ".json")
                    if f.exists():
                        t.setdefault(c["room"], []).append(json.loads(f.read_text())["r_delta"])
            per[m] = {r: {b: float(np.nanmean([v[b] for v in vs])) for b in B} for r, vs in t.items()}
        common = sorted(set(per[name]) & set(per[ref]))
        if common:
            out[g] = {"n_rooms": len(common), "level": {b: float(np.mean([per[name][r][b] for r in common])) for b in B},
                      "ref_level": {b: float(np.mean([per[ref][r][b] for r in common])) for b in B},
                      "delta": {b: boot([per[name][r][b] - per[ref][r][b] for r in common]) for b in B}}
    return out


def queue():
    try:
        q = subprocess.run(["squeue", "-u", "htakawal", "-h", "-o", "%j|%T|%M|%l|%R"], capture_output=True, text=True, timeout=30).stdout
        return [dict(zip(("job", "state", "elapsed", "limit", "where"), ln.split("|"))) for ln in q.splitlines() if "matlab" not in ln]
    except Exception:
        return []


def eta(arm):
    """next read: from the newest training log of the arm ([step/150000] ... (x s/it)) and the read schedule"""
    import glob
    import re
    fs = sorted(glob.glob("logs/slurm/aaf_{}-*.out".format(arm.lower())), key=lambda f: Path(f).stat().st_mtime)
    if not fs:
        return None
    hits = re.findall(r"\[(\d+)/(\d+)\].*?\(([\d.]+)s/it\)", Path(fs[-1]).read_text()[-20000:])
    if not hits:
        return None
    step, _, sp = int(hits[-1][0]), hits[-1][1], float(hits[-1][2])
    nxt = next((k for k in (125000, 130000, 135000, 140000, 145000, 150000) if k > step), None)
    if nxt is None:
        return None
    t = time.time() + (nxt - step) * sp
    return {"step": step, "s_per_step": sp, "next_read": nxt, "eta": time.strftime("%H:%M", time.localtime(t))}


def _c(out, fam, name, rooms):
    out.setdefault(fam, {})[name] = {"edited_r": {b: float(np.nanmean([rooms[r]["edited_r"][b] for r in rooms])) for b in B},
                                     "r_delta": {b: float(np.nanmean([rooms[r]["r_delta"][b] for r in rooms])) for b in B}, "n_rooms": len(rooms)}


def table_c(rows_by_arm):
    """headline per model and family on the lite cells (in-family): edited r and r_delta, latest read per model"""
    from scripts.p4_10_scorecard import FAMS, room_table
    out = {}
    for fam in FAMS:                                                                   # per family: the latest read of each arm that has it
        for a, rows in rows_by_arm.items():
            for nm in [r["name"] for r in rows][::-1]:
                t = room_table(nm, [fam], True).get(fam, {}).get("test", {})
                if t:
                    _c(out, fam, nm, t)
                    break
        t = room_table("R0_120K", [fam], True).get(fam, {}).get("test", {})
        if t:
            _c(out, fam, "R0_120K", t)
    return out


def f2(rows_by_arm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"R1": "#2a78d6", "R1S": "#1baf7a", "R1M": "#c64f97", "CTL_150K": "#8a8a8a", "RCP_150K": "#eda100", "R0_120K": "#444444"}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), dpi=130, facecolor="white")
    hf, uf = ax[0, 0], ax[0, 1]
    pr, st = ax[1, 0], ax[1, 1]
    for a, rows in rows_by_arm.items():
        xs = [r["step"] / 1000 for r in rows if r["static"]]
        if xs:
            hf.plot(xs, [r["static"]["HF"] for r in rows if r["static"]], "-o", color=col.get(a, "#777777"), label=a, lw=2, ms=6)
            uf.plot(xs, [r["static"]["UF"]["25-100"] for r in rows if r["static"]], "-o", color=col.get(a, "#777777"), label=a, lw=2, ms=6)
        g = [(r["step"] / 1000, [np.mean([r["gap"]["test"][m][b][0] for b in B]) for m in r["gap"]["test"]]) for r in rows if "gap" in r and r["gap"].get("test")]
        if a == "R1" and g:
            pr.plot([x for x, _ in g], [float(np.mean(v)) for _, v in g], "-o", color=col["R1"], lw=2, ms=6, label="R1 (mean of the six gaps)")
        sr = [(r["step"] / 1000, r["stretch"]["test"]["delta"]) for r in rows if r.get("stretch", {}).get("test")]
        if a == "R1S" and sr:
            for j, b in enumerate(B):
                st.plot([x for x, _ in sr], [d[b][0] for _, d in sr], "-o", lw=2, ms=6, label="R1-S - R1, {} Hz".format(b))
    for nm in ("CTL_150K", "RCP_150K"):
        a0 = load(nm)
        if a0 is not None:
            hf.axhline(a0["test_r"]["200-300"], color=col[nm], ls=":", lw=1.8, label=nm)
            uf.axhline(a0["uf"]["25-100"], color=col[nm], ls=":", lw=1.8, label=nm)
    r0 = load("R0_120K")
    if r0 is not None:
        hf.axhline(r0["test_r"]["200-300"], color=col["R0_120K"], ls="--", lw=1.4, label="R0 @120K")
        uf.axhline(r0["uf"]["25-100"], color=col["R0_120K"], ls="--", lw=1.4, label="R0 @120K")
    pr.axhline(0.10, color="#b00020", ls="--", lw=1.4, label="RS1 bar (every band)")
    st.axhline(0.05, color="#b00020", ls="--", lw=1.4, label="SS1 bar")
    for a_, t, yl in ((hf, "In-family accuracy at 200-300 Hz (HF)", "r"), (uf, "Unseen-room accuracy at 25-100 Hz", "r"),
                      (pr, "R1: moved-source gap, mean of the six (lower is better)", "static r - moved r"),
                      (st, "R1-S - R1 on stretch r_delta, per band", "r_delta difference")):
        a_.set_title(t, fontsize=11, loc="left")
        a_.set_xlabel("training step (thousands; warm start 120)", fontsize=9)
        a_.set_ylabel(yl, fontsize=9)
        a_.grid(True, color="#e6e6e6", lw=1)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        h, l = a_.get_legend_handles_labels()
        if l:
            a_.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "F2_trajectories.png", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arms = [d.name for d in sorted(ARMS.glob("*")) if d.is_dir()]
    rep = {"built": time.strftime("%Y-%m-%d %H:%M"), "arms": {a: arm_rows(a) for a in arms}, "queue": queue()}
    L = ["# P4-10 progress board", "", "Built {} (from JSON on disk; every read is a scored checkpoint).".format(rep["built"]), ""]
    run = [q for q in rep["queue"] if q["state"] == "RUNNING"]
    from collections import Counter
    n = Counter(q["job"] for q in run)
    L += ["**Next reads (from each arm's training log):** " + ("; ".join("{}: step {} at {:.2f} s/step, next read {} at about {} ET (posted within 1 h of it)".format(
        a, e["step"], e["s_per_step"], e["next_read"], e["eta"]) for a, e in ((a, eta(a)) for a in arms) if e) or "none running"), ""]
    L += ["**Running jobs:** " + (", ".join("{} x{}".format(k, v) for k, v in n.items()) or "none"), ""]
    for a in arms:
        L += ["## {} -- {}".format(a, PRIMARY.get(a, "primary metric pending")), "",
              "| read | HF (test 200-300) | UF 25-100 / 100-200 / 200-300 | delta HF vs ref [CI] | delta UF 25-100 [CI] |", "|---|---|---|---|---|"]
        for r in rep["arms"][a]:
            s = r["static"]
            if s is None:
                L.append("| {} | pending | | | |".format(r["name"]))
                continue
            d = s.get("delta_vs_ref", {})
            f = lambda v: "{:+.3f} [{:+.3f}, {:+.3f}]".format(*v)  # noqa: E731
            L.append("| {} | {:.3f} | {} | {} | {} |".format(
                r["name"], s["HF"], " / ".join("{:.3f}".format(s["UF"][b]) for b in B),
                f(d["HF"]) if d else "--", f(d["UF_25-100"]) if d else "--"))
        L.append("")
        matr = [r for r in rep["arms"][a] if r.get("material", {}).get("test", {}).get("pooled")]
        if matr:
            L += ["MM1 (re-registered D201 / D202): EDT edit slope on Material lite cells (bar: slope in [0.8, 1.2], Pearson >= 0.8), in-family "
                  "primary; MM2 margin = median |EDT error| edited - unedited, points (bar <= +5):", "",
                  "| read | set | EDT slope | Pearson | cell-bands | T20 slope | MM2 edited / unedited | dropped |", "|---|---|---|---|---|---|---|---|"]
            for r in matr:
                for g in ("test", "unseen"):
                    m = r["material"][g]
                    if not m.get("pooled"):
                        continue
                    mm = m["mm"]
                    t2 = mm["T20"]
                    L.append("| {} | {} | {:.2f} | {:.2f} | {} | {} | {:.0%} / {:.0%} ({:+.1f}) | {} cells-bands; {} of {} rx-bands (T20 floor) |".format(
                        r["name"], "in-family" if g == "test" else g, m["pooled"]["slope"], m["pooled"]["pearson"], m["pooled"]["n"],
                        "{:.2f}".format(t2["slope"]) if t2 else "n/a", mm["edt_err_edit_med"], mm["edt_err_orig_med"], 100 * mm["MM2_margin_pts"],
                        mm["n_dropped_cells_bands"], mm["n_t20_receiver_bands_dropped"], mm["n_receiver_bands_valid"]))
            L.append("")
        strc = [r for r in rep["arms"][a] if r.get("stretch", {}).get("test")]
        if strc:
            L += ["Stretch r_delta (lite cells; paired vs R1 at the same step):", "", "| read | set | level | R1 | delta 25-100 / 100-200 / 200-300 [CI] |", "|---|---|---|---|---|"]
            for r in strc:
                for g in ("test", "unseen"):
                    e = r["stretch"].get(g)
                    if e:
                        L.append("| {} | {} | {} | {} | {} |".format(r["name"], g, " / ".join("{:.3f}".format(e["level"][b]) for b in B),
                                 " / ".join("{:.3f}".format(e["ref_level"][b]) for b in B), " / ".join("{:+.3f} [{:+.3f}, {:+.3f}]".format(*e["delta"][b]) for b in B)))
            L.append("")
        has = [r for r in rep["arms"][a] if "gap" in r]
        if has:
            L += ["Moved-source gap = static r on the original room - r at the moved source (in-family lite rooms; lower is "
                  "better). Bar: <= 0.10 in every band.", "", "| read | source | 25-100 | 100-200 | 200-300 |", "|---|---|---|---|---|"]
            for r in has:
                for mag, bb in r["gap"].get("test", {}).items():
                    L.append("| {} | {} | {} |".format(r["name"], mag, " | ".join("{:.3f} [{:.3f}, {:.3f}]".format(*bb[b]) for b in B)))
            L.append("")
    try:
        rep["table_c"] = table_c(rep["arms"])
        L += ["## Table C -- latest read per model, in-family lite rooms: edited r / r_delta (25-100 / 100-200 / 200-300 Hz)", "",
              "| family | model | n rooms | edited r | r_delta |", "|---|---|---|---|---|"]
        for fam, ms in rep["table_c"].items():
            for m, v in ms.items():
                L.append("| {} | {} | {} | {} | {} |".format(fam, m, v["n_rooms"], " / ".join("{:.3f}".format(v["edited_r"][b]) for b in B), " / ".join("{:.3f}".format(v["r_delta"][b]) for b in B)))
        L.append("")
    except Exception as ex:
        L.append("Table C failed: {!r}".format(ex))
    try:
        f2(rep["arms"])
        rep["f2"] = str(OUT / "F2_trajectories.png")
        L += ["![F2 trajectories](F2_trajectories.png)", ""]
    except Exception as ex:
        L.append("F2 failed: {!r}".format(ex))
    (OUT / "board.json").write_text(json.dumps(rep, indent=1))
    (OUT / "board.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
