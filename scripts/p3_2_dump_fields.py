"""Dump GT + zero-shot predicted spectra for a handful of P3-2 configs into one NPZ.

Figure E of the P3-2 meeting pack needs spatial |field| maps for
{GT baseline, GT edited, pred baseline, pred edited}. The evaluation JSON stores only
scalar per-mode measurements, so the fields have to come from a render. Rendering needs
tinycudann, which needs a GPU, so this is a separate sbatch-able step whose output the
(CPU-only) figure driver consumes.

The checkpoint is pinned to whatever ``summary.json`` recorded, not to "newest": the
figure overlays these fields on numbers taken from that eval, and a mismatch would put
two different models on one panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from aaf.data.mat_configs import make_config
from aaf.eval.p3_2_eval import load_gt, load_model, render_config


def _parse_spec(spec: str):
    """``"baseline"`` or ``"wall:material"`` -> (wall, material)."""
    if spec == "baseline":
        return None, None
    wall, material = spec.split(":", 1)
    return wall, material


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="outputs/p3_2/eval/summary.json",
                    help="pins the checkpoint so fields and numbers come from one model")
    ap.add_argument("--checkpoint", default=None, help="override the pinned checkpoint")
    ap.add_argument("--data-dir", default="data/track_c_2d")
    ap.add_argument("--geom", default="5.93,3.18", help="L,W of the room to render")
    ap.add_argument("--configs", nargs="+",
                    default=["baseline", "west:M2", "east:M3", "north:M3"])
    ap.add_argument("--out", default="outputs/p3_2/meeting_assets/fields.npz")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    ckpt = Path(args.checkpoint or summary["checkpoint"])
    L, W = (float(v) for v in args.geom.split(","))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer, cfg, meta, it = load_model(ckpt, device)
    print("checkpoint={} iter={} device={}".format(ckpt, it, device))

    store = {}
    labels: List[str] = []
    for spec in args.configs:
        wall, material = _parse_spec(spec)
        mc = make_config(L, W, wall=wall, material=material)
        gt_path = Path(args.data_dir) / mc.filename
        H_gt, rx, src, split = load_gt(gt_path)
        H_pred = render_config(model, renderer, L, W, mc.alphas, rx, src, device)
        key = mc.label
        labels.append(key)
        store["gt/" + key] = H_gt.astype(np.complex64)
        store["pred/" + key] = H_pred.astype(np.complex64)
        store["alphas/" + key] = np.asarray(mc.alphas, dtype=np.float64)
        store["split/" + key] = np.asarray(split)
        print("{:<28} split={:<28} |H_gt|={:.4g} |H_pred|={:.4g}".format(
            key, split, float(np.abs(H_gt).mean()), float(np.abs(H_pred).mean())))

    store["labels"] = np.asarray(labels)
    store["rx"] = rx.astype(np.float64)
    store["src"] = src.astype(np.float64)
    store["L"] = np.asarray(L)
    store["W"] = np.asarray(W)
    store["fs"] = np.asarray(int(cfg["fs"]))
    store["iter"] = np.asarray(int(it))
    store["checkpoint"] = np.asarray(str(ckpt))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **store)
    print("wrote {} ({:.1f} MB)".format(out, out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
