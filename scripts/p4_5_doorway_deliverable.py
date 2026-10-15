"""P4-5 D150(3): the doorway edit in deliverable terms, from `p3_3fast_trackB_eval.py`'s EVAL.json.

Per aperture a (6 unseen test geometries each): mean |predicted - true| inter-room level difference
(dB, 20-300 Hz third-octave pooled -- the evaluator's `ld_abs_error_db`); the ratio of the fitted
pooled slope to the true slope; and the residual on the held-out band (a in [0.9, 1.1] m, as the
original evaluation defines it: aaf/data/aperture_configs.py:51) against the line fitted on the
seen apertures (`continuity.seen_line`). CPU only.
Usage: python scripts/p4_5_doorway_deliverable.py --items "Track B @20K=outputs/p3_3fast/trackB/EVAL.json,..."
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

OUT = Path("outputs/p4_5/trackB_dc_eval/DELIVERABLE")


def summarise(path):
    d = json.load(open(path))
    per_a = defaultdict(list)
    for c in d["per_config"]:
        e = c.get("ld_abs_error_db")
        if c["group"] == "sealed" or e is None or not math.isfinite(e):
            continue
        per_a["open" if c["fully_open"] else "{:.2f}".format(c["a"])].append(e)
    pooled = d["continuity"]["pooled"]
    sl = d["continuity"]["seen_line"]
    return {
        "eval": str(path),
        "checkpoint": d["meta"].get("checkpoint"),
        "abs_ld_error_db_per_aperture": {k: [sum(v) / len(v), len(v)] for k, v in per_a.items()},
        "abs_ld_error_db_by_group": {g: d["groups"][g]["level_difference"]["abs_error_db"]["mean"]
                                     for g in ("seen_apertures", "held_out_band")},
        "slope_pred": pooled["pred"]["slope"], "slope_gt": pooled["gt"]["slope"],
        "slope_ratio": pooled["pred"]["slope"] / pooled["gt"]["slope"],
        "r2_pred": pooled["pred"]["r2"], "r2_gt": pooled["gt"]["r2"],
        "heldout_residual_pred": {k: sl["pred"][k] for k in (
            "rms_residual_seen_db", "rms_residual_heldout_db", "mean_residual_heldout_db",
            "heldout_over_seen_rms")},
        "heldout_residual_gt": {k: sl["gt"][k] for k in (
            "rms_residual_seen_db", "rms_residual_heldout_db", "mean_residual_heldout_db",
            "heldout_over_seen_rms")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="label=EVAL.json,...")
    a = ap.parse_args()
    rows = {}
    for it in a.items.split(","):
        lab, p = it.split("=", 1)
        rows[lab] = summarise(p)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    labs = list(rows)
    keys = sorted({k for r in rows.values() for k in r["abs_ld_error_db_per_aperture"]},
                  key=lambda k: (k == "open", float(k) if k != "open" else 0))
    md = ["# Doorway edit in deliverable terms (D150(3))", "",
          "Mean |predicted - true| inter-room level difference (dB, 20-300 Hz), per aperture a (m), "
          "6 unseen test geometries each. Held-out band: a in [0.9, 1.1] m (0.95, 1.00, 1.05).", "",
          "| a (m) | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
    for k in keys:
        md.append("| {} | ".format(k) + " | ".join(
            "{:.2f}".format(rows[l]["abs_ld_error_db_per_aperture"][k][0])
            if k in rows[l]["abs_ld_error_db_per_aperture"] else "--" for l in labs) + " |")
    md += ["", "| | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
    for name, fn in (
            ("pooled slope (GT {:.2f})".format(rows[labs[0]]["slope_gt"]), lambda r: "{:.2f}".format(r["slope_pred"])),
            ("slope ratio pred / GT", lambda r: "{:.2f}".format(r["slope_ratio"])),
            ("pooled r^2 (GT {:.3f})".format(rows[labs[0]]["r2_gt"]), lambda r: "{:.3f}".format(r["r2_pred"])),
            ("held-out RMS residual vs seen line, dB (GT {:.2f})".format(
                rows[labs[0]]["heldout_residual_gt"]["rms_residual_heldout_db"]),
             lambda r: "{:.2f}".format(r["heldout_residual_pred"]["rms_residual_heldout_db"])),
            ("held-out mean residual, dB (GT {:+.2f})".format(
                rows[labs[0]]["heldout_residual_gt"]["mean_residual_heldout_db"]),
             lambda r: "{:+.2f}".format(r["heldout_residual_pred"]["mean_residual_heldout_db"])),
            ("mean |LD error|, seen apertures, dB", lambda r: "{:.2f}".format(r["abs_ld_error_db_by_group"]["seen_apertures"])),
            ("mean |LD error|, held-out band, dB", lambda r: "{:.2f}".format(r["abs_ld_error_db_by_group"]["held_out_band"]))):
        md.append("| {} | ".format(name) + " | ".join(fn(rows[l]) for l in labs) + " |")
    OUT.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
