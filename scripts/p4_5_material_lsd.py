"""P4-5 (manager item 3, D127/D128): the material work's OWN band LSD, on its OWN test sets, with
the IDENTICAL definition used for shape -- interval against interval, not point against range.

Definition (`p4_5_risk_metrics.py`, D120): per config, mean |dB(P) - dB(T)| over bins [0, 200)
(0-99.5 Hz at 0.5 Hz/bin) and over all receivers, |.| floored at 1e-30; then mean over configs per
GEOMETRY, then over geometries; 95% CI by a 10k bootstrap over geometries (seed 20261013). Also
100-200, 200-300 and 0-300 Hz for context. Raw H (no DC forcing), as for shape.

Two material corpora, both with per-room receivers on UNSEEN test geometries (D106):
  * `fdtd`: P3-3fast Track A -- material editing on FDTD data, the like-for-like simulator
    (`data/track_p3_3fast_A`, 601 bins to 300 Hz; 120 test configs on 10 geometries x 64 rx).
  * `ism`:  P3-2b -- material editing on ISM data, the source of the old 1.7-2.3 dB reference
    (`data/track_c_2d`, 4097 bins at 0.5 Hz; unseen-geometry splits S1, S2, S4, S5 plus the
    10 baselines; S3 excluded because its geometries are training geometries).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aaf.eval.p3_2_eval import load_gt, load_model
from aaf.eval.p3_2b_eval import render_config_arm

NB, SEED = 10000, 20261013
BANDS = {"0-100": (0, 200), "100-200": (200, 400), "200-300": (400, 601), "0-300": (0, 601)}


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def configs(corpus):
    if corpus == "fdtd":
        from aaf.data.seg_configs import configs_from_rows
        rows = json.load(open("configs/sweeps_2d_mat/p3_3fast_trackA_manifest.json"))["configs"]
        cs = configs_from_rows([r for r in rows if r["split"] == "test"], split="test")
        return [((round(c.L, 2), round(c.W, 2)), c.L, c.W, list(c.alphas), c.filename) for c in cs], "data/track_p3_3fast_A"
    from aaf.eval.p3_2b_splits import build_splits
    splits, ctx = build_splits()
    out = []
    for name, cs in splits.items():
        for c in cs:
            if c.geom_seen:                       # S3: training geometries -- excluded
                continue
            out.append(((c.L, c.W), c.L, c.W, list(c.alphas), c.filename))
    for c in ctx["baselines"].values():
        if not c.geom_seen:
            out.append(((c.L, c.W), c.L, c.W, list(c.alphas), c.filename))
    seen, uniq = set(), []
    for o in out:
        if o[4] not in seen:
            seen.add(o[4]); uniq.append(o)
    return uniq, "data/track_c_2d"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", choices=("fdtd", "ism"), required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    model, renderer, cfg, _m, it = load_model(Path(a.checkpoint), dev)
    model.eval(); renderer.eval()
    cs, ddir = configs(a.corpus)
    per_geom = {}
    for g, L, W, alphas, fn in cs:
        H, rx, src, _ = load_gt(Path(ddir) / fn)
        T = np.asarray(H)[:, :601]
        with torch.no_grad():
            P = np.asarray(render_config_arm(model, renderer, str(cfg["cond_source"]), L, W, alphas, rx, src, dev))[:, :601]
        dP, dT = db(P), db(T)
        per_geom.setdefault(g, []).append({b: float(np.mean(np.abs(dP[:, lo:hi] - dT[:, lo:hi]))) for b, (lo, hi) in BANDS.items()})
    geoms = sorted(per_geom)
    G = {b: np.array([np.mean([r[b] for r in per_geom[g]]) for g in geoms]) for b in BANDS}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(geoms), (NB, len(geoms)))
    rep = {"name": a.name, "checkpoint": a.checkpoint, "iter": int(it), "corpus": a.corpus, "data_dir": ddir,
           "cond_source": str(cfg["cond_source"]), "n_configs": len(cs), "n_geometries": len(geoms),
           "definition": "mean |dB| over bins and receivers per config, mean per geometry, bootstrap over geometries",
           "bands": {}}
    for b in BANDS:
        ci = [float(v) for v in np.percentile(G[b][idx].mean(1), [2.5, 97.5])]
        rep["bands"][b] = {"lsd": float(G[b].mean()), "ci95": ci, "per_geometry": G[b].tolist()}
        print("{:14s} {:4s} {:8s} LSD {:.2f} dB [{:.2f}, {:.2f}]  ({} configs, {} geometries)".format(
            a.name, a.corpus, b, G[b].mean(), *ci, len(cs), len(geoms)), flush=True)
    out = Path("outputs/p4_5/material_lsd"); out.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out / "{}.json".format(a.name), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
