"""P4-5 speckle thread (D150) d3 / d4: re-render the 15 dense test shapes with ONE inference-time change.

  --level-cut N    d3: ``model.level_mask[N:] = 0`` -- hash levels >= N contribute exactly zero (the
                   existing c2f mechanism: point and tx encoders, after FiLM). N = 6 keeps levels 0-5
                   (finest cell ~1.65 cm), the inference analogue of a2's 6-level cap.
  --fan-offset T   d4: the 64-ray fan rotated by T radians (``renderer.theta_grid += T``). Only
                   T in [0, 2 pi / 64) gives distinct fans; larger offsets permute the rays.
  --average A,B..  d4: average the complex predictions of several caches into one.

The render loop is a COPY of ``scripts/p4_5_gate2_dense.render_all`` (frozen; not modified), and the
cache is written in the identical format (``P{i}, T{i}, rx{i}, los{i}``), so ``p4_5_risk_metrics``,
``p4_5_hf_gap``, ``p4_5_seam_check`` and ``p4_5_speckle`` read it by name. The cache ``key`` records the
modification, so it can never be mistaken for an unmodified render.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from aaf.data.shape_configs import SRC, configs_from_rows
from aaf.eval.p3_2_eval import load_model
from aaf.models.conditioning_2d import build_cond_vector_2d

MANIFEST = "configs/sweeps_2d_mat/p4_2_shapes_manifest.json"
DATA_DIR = Path("data/track_p4_5_testdense")
CACHE_DIR = Path("outputs/p4_5/gate2_dense/cache")


def cache_path(name: str) -> Path:
    return CACHE_DIR / "{}__track_p4_5_testdense.npz".format(name)


def load_modified(ck, dev, level_cut=None, fan_offset=0.0):
    model, renderer, _cfg, _meta, _it = load_model(Path(ck), dev)
    model.eval(); renderer.eval()                                # D49 C3
    if level_cut is not None:
        model.level_mask.zero_()
        model.level_mask[:level_cut] = 1.0
    if fan_offset:
        renderer.theta_grid.add_(float(fan_offset))
    return model, renderer


def render_shape(model, renderer, c, dev, rx_limit=None):
    """One shape, verbatim from p4_5_gate2_dense.render_all's loop body."""
    with h5py.File(DATA_DIR / c.filename) as f:
        T = np.asarray(f["ism/H_complex"])
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        los = np.asarray(json.loads(f.attrs["line_of_sight"]), dtype=bool)
    if rx_limit is not None:
        T, rx, los = T[:rx_limit], rx[:rx_limit], los[:rx_limit]
    cond = build_cond_vector_2d("geom_token", c.L, c.W, c.edge_alphas, verts=c.verts, device=dev)
    room_min = torch.zeros(2, device=dev)
    room_max = torch.tensor([c.L, c.W], device=dev, dtype=torch.float32)
    R = torch.tensor(rx, dtype=torch.float32, device=dev)
    S = torch.tensor(np.asarray(SRC, float), dtype=torch.float32, device=dev).unsqueeze(0)
    P = np.zeros_like(T)
    for s0 in range(0, len(rx), 8):
        sl = slice(s0, s0 + 8)
        r = R[sl]
        with torch.no_grad():
            H = renderer(model, r, S.expand(r.shape[0], -1), room_min, room_max,
                         z_s=cond.unsqueeze(0).expand(r.shape[0], -1))
        P[sl] = H[:, :T.shape[1]].cpu().numpy()
    return P, T, rx, los


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--name", required=True)
    ap.add_argument("--level-cut", type=int, default=None)
    ap.add_argument("--fan-offset", type=float, default=0.0)
    ap.add_argument("--average", default="", help="comma-separated cache names to average into --name")
    ap.add_argument("--delete-inputs", action="store_true", help="with --average: delete the inputs")
    a = ap.parse_args()
    out = cache_path(a.name)
    if a.average:
        names = a.average.split(",")
        zs = [np.load(cache_path(n), allow_pickle=False) for n in names]
        blob = {"key": "AVERAGE[" + ";".join(str(z["key"]) for z in zs) + "]"}
        i = 0
        while "P%d" % i in zs[0].files:
            blob["P%d" % i] = np.mean([z["P%d" % i].astype(np.complex128) for z in zs],
                                      axis=0).astype(np.complex64)
            for k in ("T", "rx", "los"):
                ref = zs[0]["%s%d" % (k, i)]
                for z in zs[1:]:
                    assert np.array_equal(ref, z["%s%d" % (k, i)]), (k, i)
                blob["%s%d" % (k, i)] = ref
            i += 1
        np.savez_compressed(out, **blob)
        print("averaged {} caches -> {} ({} shapes)".format(len(names), out, i), flush=True)
        if a.delete_inputs:
            for n in names:
                cache_path(n).unlink()
                print("  deleted", cache_path(n), flush=True)
        return
    assert a.fan_offset == 0.0 or 0.0 <= a.fan_offset < 2 * math.pi, a.fan_offset
    dev = torch.device("cuda")
    model, renderer = load_modified(a.checkpoint, dev, a.level_cut, a.fan_offset)
    test = configs_from_rows(json.load(open(MANIFEST))["configs"], split="test")
    key = "|".join([str(a.checkpoint), str(DATA_DIR),
                    "level_cut={}".format(a.level_cut), "fan_offset={!r}".format(a.fan_offset)])
    blob = {"key": key}
    for i, c in enumerate(test):
        P, T, rx, los = render_shape(model, renderer, c, dev)
        blob["P%d" % i], blob["T%d" % i], blob["rx%d" % i], blob["los%d" % i] = P, T, rx, los
        print("  rendered {:34s} n_rx {:5d}".format(c.label, len(rx)), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **blob)
    print("wrote", out, "key", key, flush=True)


if __name__ == "__main__":
    main()
