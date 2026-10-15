"""P4-6 t3 (D160(i)): where in frequency does the narrow-doorway level-difference error sit?

For the a = 0.15 m test configurations (6 unseen geometries) of the Track B corpus, the inter-room
level difference LD = 20 log10( <|H|>_B / <|H|>_A ) (the evaluator's own `band_level_ratio`), per
band 0-20 / 20-100 / 100-200 / 200-300 Hz and pooled 20-300 Hz (the deliverable's definition),
prediction against FDTD. Rendering, ground-truth loading and band-limiting are the Track B
evaluator's own (`p3_3fast_trackB_eval.render_aperture`, `p3_2_eval.load_gt` / `band_limit`).

Gate (D160(i)): the two-region DC head is queued only if 0-20 Hz carries > 50% of the summed per-band
|LD error|. Otherwise the problem is the aperture representation, and the head is not the fix.
  python scripts/p4_6_doorway_bands.py --items "TBdc_30K=outputs/p4_5/arms/TBdc/ckpt_iter0030000.pt,..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.p3_3fast_ftb import band_level_ratio
from scripts.p3_3fast_trackB_eval import (DATA_DIR, DF_HZ, MANIFEST, N_BINS_BAND, render_aperture,
                                          room_masks)

BANDS = {"0-20": (0.0, 20.0), "20-100": (20.0, 100.0), "100-200": (100.0, 200.0),
         "200-300": (200.0, 300.0), "pooled 20-300": (20.0, 300.0)}
A_TARGET = 0.15
OUT = Path("outputs/p4_6/doorway_bands")


def main():
    import torch

    from aaf.data.aperture_configs import configs_from_rows
    from aaf.eval.p3_2_eval import band_limit, load_gt, load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="NAME=CKPT; '+' or ',' separated")
    a = ap.parse_args()
    dev = torch.device("cuda")
    cfgs = [c for c in configs_from_rows(json.loads(Path(MANIFEST).read_text())["configs"], split="test")
            if not c.sealed and not c.fully_open and abs(c.a - A_TARGET) < 1e-6]
    assert len(cfgs) == 6, len(cfgs)
    hi = N_BINS_BAND
    freqs = np.arange(hi) * DF_HZ
    rep = {"a_m": A_TARGET, "n_configs": len(cfgs), "bands_hz": BANDS, "models": {}}
    for item in a.items.replace("+", ",").split(","):
        name, ck = item.split("=", 1)
        model, renderer, cfg, _meta, it = load_model(Path(ck), dev)
        model.eval(); renderer.eval()
        rows = []
        for c in cfgs:
            H_raw, rx, src, _split = load_gt(Path(DATA_DIR) / c.filename)
            H_gt = band_limit(H_raw, hi)[:, :hi]
            H_pr = band_limit(render_aperture(model, renderer, str(cfg["cond_source"]), c.L, c.W, c.x0,
                                              c.a, c.alphas, rx, src, dev), hi)[:, :hi]
            sa, sb, _n = room_masks(rx, c.x0)
            row = {"filename": c.filename, "geom_id": c.geom_id}
            for b, (lo, hh) in BANDS.items():
                g = band_level_ratio(H_gt, freqs, sa, sb, lo, hh)
                p = band_level_ratio(H_pr, freqs, sa, sb, lo, hh)
                row[b] = {"ld_gt": g, "ld_pred": p, "abs_err": abs(p - g)}
            rows.append(row)
            print("  {} {} ".format(name, c.filename) + " ".join(
                "{} {:+.2f}/{:+.2f}".format(b, row[b]["ld_pred"], row[b]["ld_gt"]) for b in BANDS), flush=True)
        mean_err = {b: float(np.mean([r[b]["abs_err"] for r in rows])) for b in BANDS}
        four = ["0-20", "20-100", "100-200", "200-300"]
        tot = sum(mean_err[b] for b in four)
        share = {b: mean_err[b] / tot for b in four}
        rep["models"][name] = {"checkpoint": ck, "iter": int(it), "per_config": rows,
                               "mean_abs_ld_error_db": mean_err, "share_of_band_error": share,
                               "gate_two_region_head": bool(share["0-20"] > 0.5)}
        print("{}: mean |LD err| ".format(name) + ", ".join("{} {:.2f} dB".format(b, mean_err[b]) for b in BANDS)
              + " | shares " + ", ".join("{} {:.2f}".format(b, share[b]) for b in four)
              + " | gate (0-20 > 50%): {}".format(share["0-20"] > 0.5), flush=True)
        del model, renderer
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "doorway_bands_a015.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
