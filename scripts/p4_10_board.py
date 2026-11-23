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

from scripts.p4_10_scorecard import BAND, SC, boot, cells, cid, source_gap
from scripts.p4_8_primary_table import load, paired

ARMS = Path("outputs/p4_10/arms")
OUT = Path("outputs/p4_10/progress")
B = list(BAND)
# arm -> reference. R1 is read against CTL @150K (the same steps from R0); every other arm against R1 at the matched step.
def ref_of(arm, step):
    return "CTL_150K" if arm == "R1" else "R1_{}K".format(step // 1000)


REF_NOTE = "R1-M reads its primary metric on the Material lite cells"
PRIMARY = {"R1": "in-family moved-source gap, band mean, second and third source (bar: each band <= 0.10)",
           "R1S": "r_delta on Stretch (lite) cells, in-family, paired vs R1 at the same step (bar: +0.05, CI > 0, >= 2 of 3 bands)",
           "R1M": "decay-rate edit slope on Material (lite) cells (bar: slope in [0.8, 1.2], Pearson >= 0.8)",
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arms = [d.name for d in sorted(ARMS.glob("*")) if d.is_dir()]
    rep = {"built": time.strftime("%Y-%m-%d %H:%M"), "arms": {a: arm_rows(a) for a in arms}, "queue": queue()}
    L = ["# P4-10 progress board", "", "Built {} (from JSON on disk; every read is a scored checkpoint).".format(rep["built"]), ""]
    run = [q for q in rep["queue"] if q["state"] == "RUNNING"]
    from collections import Counter
    n = Counter(q["job"] for q in run)
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
    (OUT / "board.json").write_text(json.dumps(rep, indent=1))
    (OUT / "board.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
