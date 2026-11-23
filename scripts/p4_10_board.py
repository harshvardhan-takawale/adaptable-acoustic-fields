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

from scripts.p4_10_scorecard import BAND, boot, source_gap
from scripts.p4_8_primary_table import load, paired

ARMS = Path("outputs/p4_10/arms")
OUT = Path("outputs/p4_10/progress")
B = list(BAND)
# arm -> (reference name per read step, primary metric name, bar). The reference of R1 is CTL @150K (same steps from R0).
REF = {"R1": "CTL_150K"}
PRIMARY = {"R1": "in-family moved-source gap, band mean, second and third source (bar: each band <= 0.10)"}


def reads(arm):
    d = ARMS / arm
    out = []
    for p in sorted(d.glob("ckpt_iter*.pt")) if d.exists() else []:
        it = int(p.stem[-7:])
        if it in (125000, 130000, 140000, 150000):
            out.append((it, "{}_{}K".format(arm, it // 1000)))
    return out


def arm_rows(arm):
    rows = []
    for it, name in reads(arm):
        row = {"name": name, "step": it, "static": None}
        a = load(name)
        ref = load(REF[arm]) if arm in REF else None
        if a is not None:
            row["static"] = {"HF": a["test_r"]["200-300"], "UF": {b: a["uf"][b] for b in B}}
            if ref is not None:
                d = paired(a, ref)
                row["static"]["delta_vs_ref"] = {"HF": d["HF"], **{"UF_" + b: d["UF_" + b] for b in B}, "ref": REF[arm]}
        g = source_gap(name, lite=True)
        if g.get("test"):
            row["gap"] = {grp: {mag: {b: boot([v[b] for v in g[grp][mag].values()]) for b in B} for mag in g[grp]} for grp in g}
        rows.append(row)
    return rows


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
