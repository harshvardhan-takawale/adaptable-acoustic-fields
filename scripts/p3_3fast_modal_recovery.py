"""The honest headline: does held-out window recovery survive removing the DC term?

Measured on the FDTD corpus: bin 0 carries 87.3% of in-band power and sits 37.1 dB above the
strongest room mode, because the soft source injects net volume into a nearly-rigid closed room
and the 2 s record integrates it. Consequence for the result I have reported twice:

    GT window energy delta, ALL bins        -4.835 dB
    GT window energy delta, EXCLUDING 0-40  -1.774 dB

So the "-5.28 dB window effect" is ~86% a compliance change, and an all-bin "recovery fraction"
largely measures how well the model reproduces DC. This recomputes recovery over MODAL bins
only. The first room mode of these geometries is 28.6-57 Hz, so cutting below bin 41 (20.5 Hz)
removes no modal content.

Reports both numbers side by side. If held-out recovery collapses without DC, the transfer
claim weakens to "transfers the compliance response" and that is what gets written down.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aaf.data.seg_configs import configs_from_rows
from aaf.eval.p3_2_eval import band_limit, load_gt, load_model
from aaf.eval.p3_2b_eval import render_config_arm

BAND_HI, DF = 300.0, 0.5
LO_BINS = (0, 11, 21, 41)          # 0 Hz, 5 Hz, 10 Hz, 20.5 Hz


def energy_db(H, lo):
    p = (np.abs(np.asarray(H))[:, lo:] ** 2).mean()
    return 10.0 * np.log10(max(p, 1e-300))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="configs/sweeps_2d_mat/p3_3fast_trackA_manifest.json")
    ap.add_argument("--data-dir", default="data/track_p3_3fast_A")
    ap.add_argument("--out", default="outputs/p3_3fast/modal_recovery.json")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer, cfg, meta, it = load_model(Path(a.checkpoint), dev)
    model.eval(); renderer.eval()                      # renderer.eval() required -- D49 C3
    cs = str(cfg["cond_source"])
    hi = int(round(BAND_HI / DF)) + 1

    rows = [r for r in json.load(open(a.manifest))["configs"] if r["split"] == "test"]
    cfgs = {c.label: c for c in configs_from_rows(rows, split="test")}
    by_geom = {}
    for r in rows:
        by_geom.setdefault(r["geom_id"], {})[r["kind"]] = r

    out_rows = []
    for gid, d in sorted(by_geom.items()):
        if "baseline" not in d:
            continue
        base_r = d["baseline"]
        cb = configs_from_rows([base_r])[0]
        H_gt_b, rx, src, _ = load_gt(Path(a.data_dir) / cb.filename)
        gt_b = band_limit(H_gt_b, hi)
        with torch.no_grad():
            pr_b = band_limit(np.asarray(render_config_arm(
                model, renderer, cs, cb.L, cb.W, list(cb.alphas), rx, src, dev))[:, :hi], hi)
        for kind in ("t_window_seen", "t_window_holdout"):
            if kind not in d:
                continue
            c = configs_from_rows([d[kind]])[0]
            H_gt, rx2, src2, _ = load_gt(Path(a.data_dir) / c.filename)
            gt = band_limit(H_gt, hi)
            with torch.no_grad():
                pr = band_limit(np.asarray(render_config_arm(
                    model, renderer, cs, c.L, c.W, list(c.alphas), rx2, src2, dev))[:, :hi], hi)
            rec = {"geom_id": gid, "kind": kind}
            for lo in LO_BINS:
                dg = energy_db(gt, lo) - energy_db(gt_b, lo)
                dp = energy_db(pr, lo) - energy_db(pr_b, lo)
                rec["lo{}".format(lo)] = {
                    "gt_delta_db": float(dg), "pred_delta_db": float(dp),
                    "recovered": float(dp / dg) if abs(dg) > 1e-9 else float("nan"),
                    "from_hz": lo * DF}
            out_rows.append(rec)

    summary = {}
    for kind in ("t_window_seen", "t_window_holdout"):
        sub = [r for r in out_rows if r["kind"] == kind]
        summary[kind] = {}
        for lo in LO_BINS:
            g = np.array([r["lo{}".format(lo)]["gt_delta_db"] for r in sub])
            rr = np.array([r["lo{}".format(lo)]["recovered"] for r in sub])
            rr = rr[np.isfinite(rr)]
            summary[kind]["lo{}".format(lo)] = {
                "from_hz": lo * DF, "n": int(len(sub)),
                "gt_delta_db_mean": float(g.mean()),
                "recovered_mean": float(rr.mean()) if rr.size else float("nan"),
                "recovered_sd": float(rr.std()) if rr.size else float("nan")}

    res = {"checkpoint": a.checkpoint, "iter": int(it), "cond_source": cs,
           "lo_bins": list(LO_BINS),
           "why": ("bin 0 holds 87.3% of in-band power and sits 37.1 dB above the strongest room "
                   "mode; all-bin energy ratios are therefore DC measurements. First room mode "
                   "is 28.6-57 Hz so lo=41 (20.5 Hz) removes no modal content."),
           "summary": summary, "per_config": out_rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)

    print("{:20s} {:>8s} {:>12s} {:>12s}".format("split", "from Hz", "GT delta dB", "recovered"))
    for kind, v in summary.items():
        for k, s in v.items():
            print("{:20s} {:8.1f} {:12.3f} {:12.3f}".format(
                kind if k == "lo0" else "", s["from_hz"], s["gt_delta_db_mean"],
                s["recovered_mean"]))
    print("-> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
