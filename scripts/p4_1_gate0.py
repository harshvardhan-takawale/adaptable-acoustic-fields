"""P4-1 GATE 0: can boundary tokens carry ALL the geometry?

Arm T-geo drops the global (L, W) prefix entirely and expresses the room as boundary tokens in
absolute world coordinates. Gate 0 asks whether that costs anything measurable against Arm C,
which is identical in every other respect -- same corpus, same manifest sha, same recipe, same
iteration budget -- so a difference is attributable to the conditioning.

GATE 0 (both required):
    spatial Pearson mean  >= 0.85
    demo-protocol band LSD <= Arm C's 2.268 dB + 0.5 = 2.768 dB

The LSD reference is the DEMO-protocol mean, not Arm C's in-distribution val LSD (1.0132 dB).
Gate 0's other half -- spatial Pearson -- exists only in the demo protocol, so that is the
consistent reading; the in-dist number measures held-out receivers of TRAINING configs, which is
a different question. Both are reported, only the demo one gates.

The verdict is computed and printed BEFORE any figure is drawn (D62a).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

#: Arm C, measured. outputs/armC_demo/metrics.json (demo) and its verdict.json (in-dist).
ARMC_DEMO_LSD_MEAN = 2.268
ARMC_DEMO_PEARSON_MEAN = 0.9511
ARMC_DEMO_PEARSON_WORST = 0.9196
ARMC_INDIST_VAL_LSD = 1.0132

GATE_PEARSON_MIN = 0.85
GATE_LSD_MAX = ARMC_DEMO_LSD_MEAN + 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/p4_1/stage0/p4_1_Tgeo")
    ap.add_argument("--out", default="outputs/p4_1/stage0")
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()

    ck = a.checkpoint or str(sorted(Path(a.run_dir).glob("ckpt_iter*.pt"))[-1])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    metrics = out / "metrics.json"

    # Stage 1 of the demo protocol, verbatim -- same script Arm C was measured with, so the
    # two numbers are produced by one code path rather than two that agree by inspection.
    cmd = [sys.executable, "scripts/armC_demo_metrics.py", "--checkpoint", ck,
           "--out", str(metrics), "--npz-dir", str(out / "fields")]
    print("[run] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if not metrics.exists():
        print("demo metrics did not produce {} (exit {})".format(metrics, rc))
        return 3

    m = json.load(open(metrics))
    lsd = [r["band_lsd_db"] for r in m["scenarios"]]
    pear = [r["spatial_pearson_mean"] for r in m["scenarios"]]
    lsd_mean = sum(lsd) / len(lsd)
    p_mean, p_worst = sum(pear) / len(pear), min(pear)

    crit = {
        "spatial_pearson_mean": {"value": p_mean, "op": ">=", "threshold": GATE_PEARSON_MIN,
                                 "pass": bool(p_mean >= GATE_PEARSON_MIN)},
        "band_lsd_mean_db": {"value": lsd_mean, "op": "<=", "threshold": GATE_LSD_MAX,
                             "pass": bool(lsd_mean <= GATE_LSD_MAX)},
    }
    passed = all(c["pass"] for c in crit.values())

    print("\n  {:26s} {:>10s}  {:>10s}".format("metric", "Arm T-geo", "Arm C"))
    print("  {:26s} {:10.3f}  {:10.3f}".format("spatial Pearson (mean)", p_mean,
                                               ARMC_DEMO_PEARSON_MEAN))
    print("  {:26s} {:10.3f}  {:10.3f}".format("spatial Pearson (worst)", p_worst,
                                               ARMC_DEMO_PEARSON_WORST))
    print("  {:26s} {:10.3f}  {:10.3f}".format("band LSD mean (dB)", lsd_mean,
                                               ARMC_DEMO_LSD_MEAN))
    print("\n  GATE 0: {}".format("PASS" if passed else "FAIL"))
    for k, c in crit.items():
        print("    {:22s} {:.3f} {} {:.3f} -> {}".format(
            k, c["value"], c["op"], c["threshold"], "pass" if c["pass"] else "FAIL"))

    json.dump({"gate": "p4_1.stage0/1", "checkpoint": ck, "iter": m.get("iter"),
               "cond_source": m.get("cond_source"),
               "tgeo": {"spatial_pearson_mean": p_mean, "spatial_pearson_worst": p_worst,
                        "band_lsd_mean_db": lsd_mean, "per_scenario": m["scenarios"]},
               "armc_reference": {"spatial_pearson_mean": ARMC_DEMO_PEARSON_MEAN,
                                  "spatial_pearson_worst": ARMC_DEMO_PEARSON_WORST,
                                  "band_lsd_mean_db": ARMC_DEMO_LSD_MEAN,
                                  "in_dist_val_lsd_db": ARMC_INDIST_VAL_LSD},
               "criteria": crit, "passed": passed,
               "lsd_reference_note": ("Gate threshold is the DEMO-protocol mean (2.268 + 0.5). "
                                      "Arm C's in-dist val LSD 1.0132 measures held-out "
                                      "receivers of TRAINING configs -- a different question -- "
                                      "and is reported but does not gate.")},
              open(out / "GATE0.json", "w"), indent=1, default=float)
    print("\n-> {}".format(out / "GATE0.json"))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
