"""Build the P2-3.5 comparison table + RESULTS.md from the known-geometry outputs.

Columns per room: P2-3 8-recv search | Exp1 lookup-RBF | Exp1 lookup-linear | Exp2 oracle.
Cells: mag corr full + 0-250 Hz (modal band). Interpolative vs extrapolative never averaged.
Robust to missing oracle rooms (some jobs may still be running).
"""
from __future__ import annotations
import json, glob, os
from pathlib import Path
import numpy as np

ROOT = Path("outputs/known_geometry")

INTERP = ["L4.50_W4.00_H3.25", "L4.40_W4.09_H3.26", "L3.52_W4.31_H3.40", "L4.82_W3.81_H2.92"]
EXTRAP = ["L4.10_W3.01_H3.93", "L5.94_W4.93_H2.51", "L5.92_W3.06_H2.55", "L5.91_W4.17_H3.72",
          "L3.17_W3.00_H3.49", "L5.99_W3.96_H2.54", "L3.14_W3.08_H2.51"]
# box center is the only NATIVE interpolative test room; the other 3 interp are augmented.
NATIVE = {"L4.50_W4.00_H3.25"}


def _g(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


def _pb(metrics_path):
    try:
        m = json.loads(Path(metrics_path).read_text())
        pb = m.get("per_band_mag_corr", {})
        return pb.get("mag_corr_full"), pb.get("mag_corr_0_250"), m.get("lsd_db_full")
    except Exception:
        return None, None, None


def _fmt(x, f=".3f"):
    return format(x, f) if isinstance(x, (int, float)) and np.isfinite(x) else "—"


def collect():
    p23 = json.loads((ROOT / "p2_3_8recv_per_band.json").read_text()) if (ROOT / "p2_3_8recv_per_band.json").exists() else {}
    rows = {}
    for room in INTERP + EXTRAP:
        r = {"room": room}
        # P2-3 8-recv (only the 8 native test rooms have this; augmented interior do not)
        if room in p23:
            r["p23_full"], r["p23_0250"] = p23[room].get("mag_corr_full"), p23[room].get("mag_corr_0_250")
        # Exp1 lookup
        r["rbf_full"], r["rbf_0250"], r["rbf_lsd"] = _pb(ROOT / f"lookup/{room}__rbf/metrics.json")
        r["lin_full"], r["lin_0250"], _ = _pb(ROOT / f"lookup/{room}__linear/metrics.json")
        # Exp2 oracle
        r["orc_full"], r["orc_0250"], r["orc_lsd"] = _pb(ROOT / f"oracle/{room}/metrics.json")
        rows[room] = r
    loo = json.loads((ROOT / "lookup_summary.json").read_text()).get("loo", {}) if (ROOT / "lookup_summary.json").exists() else {}
    return rows, loo


def table_md(rows):
    out = ["| room | type | P2-3 8-recv | lookup-RBF | lookup-lin | oracle |",
           "|---|---|---|---|---|---|"]

    def line(room):
        r = rows[room]
        tag = "interp" + ("·native" if room in NATIVE else "·aug") if room in INTERP else "extrap"
        def cell(full, b0250):
            return f"{_fmt(full)} / {_fmt(b0250)}"
        return (f"| {room} | {tag} | {cell(r.get('p23_full'), r.get('p23_0250'))} | "
                f"{cell(r.get('rbf_full'), r.get('rbf_0250'))} | {cell(r.get('lin_full'), r.get('lin_0250'))} | "
                f"{cell(r.get('orc_full'), r.get('orc_0250'))} |")

    out.append("| **INTERPOLATIVE** | | | | | |")
    for room in INTERP:
        out.append(line(room))
    out.append("| **EXTRAPOLATIVE** | | | | | |")
    for room in EXTRAP:
        out.append(line(room))
    out.append("\n*Cells = mag corr (full / 0-250 Hz modal band). 'aug' = augmented interior room (no 8-recv baseline).*")
    return "\n".join(out)


def _mean(rows, roomset, key):
    vals = [rows[r].get(key) for r in roomset if isinstance(rows[r].get(key), (int, float)) and np.isfinite(rows[r].get(key))]
    return float(np.mean(vals)) if vals else float("nan")


def verdict(rows, loo):
    # interpolative group means (the fair test)
    i_orc = _mean(rows, INTERP, "orc_full"); i_orc_m = _mean(rows, INTERP, "orc_0250")
    i_rbf = _mean(rows, INTERP, "rbf_full"); i_rbf_m = _mean(rows, INTERP, "rbf_0250")
    i_p23 = _mean(rows, [r for r in INTERP if r in NATIVE], "p23_full")
    loo_rbf = _g(loo, "rbf", "mean_mag_corr_full")
    lines = []
    lines.append(f"- **LOO (route on known training geometry)**: RBF mean mag corr "
                 f"{_fmt(loo_rbf)} full, {_fmt(_g(loo,'rbf','mean_mag_corr_0_250'))} (0-250 Hz), "
                 f"LSD {_fmt(_g(loo,'rbf','mean_lsd_db_full'),'.2f')} dB.")
    lines.append(f"- **Interpolative means**: lookup-RBF {_fmt(i_rbf)} full / {_fmt(i_rbf_m)} (0-250); "
                 f"oracle {_fmt(i_orc)} full / {_fmt(i_orc_m)} (0-250); P2-3 8-recv (box center) {_fmt(i_p23)} full.")
    # verdict logic
    hi = 0.7  # "high"
    v = []
    if isinstance(i_orc, float) and np.isfinite(i_orc):
        if i_orc >= hi:
            v.append("**Oracle renders interpolative rooms well** → the decoder CAN render unseen geometry; "
                     "the P2-3 failure was latent-finding from sparse data (positive).")
        else:
            v.append(f"**Oracle is low on interpolative rooms ({_fmt(i_orc)})** → the decoder cannot render "
                     "unseen geometry even with the best latent → the fix is strictly more training rooms (P2-4).")
        if isinstance(i_rbf, float) and np.isfinite(i_rbf):
            if abs(i_rbf - i_orc) < 0.1:
                v.append("lookup ≈ oracle → the (L,W,H)→latent map is about as good as any latent.")
            elif i_rbf < i_orc - 0.1:
                v.append("lookup ≪ oracle → the (L,W,H)→latent map is the weak link (better interpolation would help).")
    else:
        v.append("(oracle interpolative results not yet complete)")
    return "\n".join(lines) + "\n\n**Verdict**: " + " ".join(v)


def main():
    rows, loo = collect()
    md = ["# P2-3.5 — Known-geometry rendering + oracle ceiling: RESULTS\n",
          "Both experiments reuse the converged P3 model (in-dist 2.169 dB); no retraining. "
          "Of the 8 maximin test rooms only the box center is strictly interpolative — 3 augmented "
          "strictly-interior rooms were added so the interpolative headline rests on 4 rooms. "
          "Interpolative and extrapolative rooms are reported separately, never averaged.\n",
          "## Comparison table (the meeting asset)\n", table_md(rows),
          "\n\n## Verdict\n", verdict(rows, loo), "\n"]
    (ROOT / "RESULTS.md").write_text("\n".join(md))
    # CSV
    import csv
    with open(ROOT / "comparison_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["room", "type", "p23_full", "p23_0250", "rbf_full", "rbf_0250",
                    "lin_full", "lin_0250", "orc_full", "orc_0250"])
        for room in INTERP + EXTRAP:
            r = rows[room]
            typ = "interp" if room in INTERP else "extrap"
            w.writerow([room, typ, r.get("p23_full"), r.get("p23_0250"), r.get("rbf_full"),
                        r.get("rbf_0250"), r.get("lin_full"), r.get("lin_0250"),
                        r.get("orc_full"), r.get("orc_0250")])
    print(f"wrote {ROOT}/RESULTS.md + comparison_table.csv")
    print(verdict(rows, loo))


if __name__ == "__main__":
    main()
