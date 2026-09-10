"""Regression guard for D64: the world-scale change must not touch any pre-P4-1 checkpoint.

`_normalize_unit` went from a staticmethod to an instance method with two regimes. Arm C was
trained under the legacy one, so `world_scale=None` must reproduce `(x + 1) / 2` EXACTLY -- not
approximately, and not "close enough that the metrics still look fine".

The check re-renders Arm C and compares against the field dumps cached by the v1 demo pack
(`outputs/armC_demo/fields/*.npz`), which were produced by the pre-change code. Those are the
actual rendered complex spectra, so this compares the model's output tensor rather than a
summary statistic -- a metric-level check could hide a small systematic shift.

Bit-identity is the pass condition. `renderer.eval()` makes ray azimuths deterministic (D49 C3),
so there is no legitimate source of run-to-run variation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aaf.eval.p3_2_eval import band_limit, load_model
from aaf.eval.p3_2b_eval import render_config_arm
from aaf.walls import WALL_INDEX, WALLS_2D

SCEN = {
    "a_baseline": {},
    "b_east_curtain": {"east": 0.50},
    "c_north_absorber": {"north": 0.70},
    "d_two_wall": {"east": 0.50, "south": 0.70},
}
SRC = (0.5, 0.5)


def alphas_for(edits):
    a = [0.15] * len(WALLS_2D)
    for w, v in edits.items():
        a[WALL_INDEX[w]] = float(v)
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="outputs/p3_2/p3_2b_C_cont_mlinear/ckpt_iter0060000.pt")
    ap.add_argument("--fields", default="outputs/armC_demo/fields")
    ap.add_argument("--out", default="outputs/p4_1/stage0/ARMC_REGRESSION.json")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer, cfg, meta, it = load_model(Path(a.checkpoint), dev)
    model.eval()
    renderer.eval()
    cond_source = str(cfg["cond_source"])
    ws = getattr(model, "world_scale", "MISSING")
    print("[arm] {} iter {} | cond {} | world_scale={!r} (must be None for legacy)".format(
        Path(a.checkpoint).parent.name, it, cond_source, ws), flush=True)
    if ws is not None:
        raise SystemExit("FAIL: a pre-P4-1 checkpoint must load with world_scale=None, got {!r}"
                         .format(ws))

    rows, worst = [], 0.0
    for tag in ("small", "median", "large"):
        for scen, edits in SCEN.items():
            p = Path(a.fields) / "{}_{}.npz".format(tag, scen)
            if not p.exists():
                continue
            z = np.load(p)
            ref, rx = np.asarray(z["pred"]), np.asarray(z["rx"])
            L, W = float(z["L"]), float(z["W"])
            with torch.no_grad():
                got = np.asarray(render_config_arm(model, renderer, cond_source, L, W,
                                                   alphas_for(edits), rx,
                                                   np.asarray(SRC, float), dev))
            got = band_limit(got[:, :ref.shape[1]], ref.shape[1]).astype(ref.dtype)
            exact = bool(np.array_equal(got, ref))
            dmax = float(np.max(np.abs(got - ref))) if not exact else 0.0
            worst = max(worst, dmax)
            rows.append({"geometry": tag, "scenario": scen, "bit_identical": exact,
                         "max_abs_delta": dmax})
            print("  {:7s} {:18s} bit-identical={}  max|delta|={:.3e}".format(
                tag, scen, exact, dmax), flush=True)

    if not rows:
        raise SystemExit("no cached field dumps found under {}".format(a.fields))
    ok = all(r["bit_identical"] for r in rows)
    out = {"checkpoint": a.checkpoint, "iter": int(it), "world_scale": None,
           "n_compared": len(rows), "all_bit_identical": ok,
           "worst_max_abs_delta": worst, "per_scenario": rows,
           "note": ("Re-render of Arm C after the D64 world_scale change, compared against the "
                    "v1 demo pack's cached predictions produced by the pre-change code. "
                    "world_scale=None must reproduce the legacy (x+1)/2 map exactly.")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("\n{}: {} of {} bit-identical, worst |delta| = {:.3e}".format(
        "PASS" if ok else "FAIL", sum(r["bit_identical"] for r in rows), len(rows), worst))
    print("-> {}".format(a.out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
