"""Build one P3-2 room per SLURM array task.

    python scripts/build_2d_mat_dataset.py --idx 0        # simulate one config
    python scripts/build_2d_mat_dataset.py --manifest     # print the flat config list

All four splits are enumerated into ONE deterministic flat list, so a single array job
covers the whole corpus and `--idx` is stable across resubmits.

Idempotent and crash-safe, following scripts/build_3d_dataset.py: a `.done` sentinel
short-circuits completed work, and the HDF5 is written to a temp file then atomically
renamed, so an interrupted task cannot leave a half-written file that later looks valid.

The `/analytical` group is written ONLY for baseline (uniform-absorption) configs:
`analytical_modal_2d` takes a scalar alpha and its Sabine damping is mode-independent, so
for an edited config it would be physically wrong. Recorded in the attrs, not papered over.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

from aaf.data.mat_configs import HELDOUT_COMBOS, UNSEEN_ALPHA, MatConfig, enumerate_configs
from aaf.sim.analytical_modal_2d import modal_rir_2d
from aaf.sim.ism_2d import simulate_room_2d
from aaf.walls import WALLS_2D

P3_2B_MANIFEST = "configs/sweeps_2d_mat/p3_2b_manifest.json"
TRAIN_YAML = "configs/sweeps_2d_mat/p3_2_train.yaml"
TEST_YAML = "configs/sweeps_2d_mat/p3_2_test_frozen.yaml"
OUT_DIR = "data/track_c_2d"


def make_receiver_grid_2d(L, W, n_per_side=8, margin=0.3):
    """Identical maths to scripts/build_datasets.py:66 -- row-major, outer y, inner x."""
    xs = np.linspace(margin, L - margin, n_per_side)
    ys = np.linspace(margin, W - margin, n_per_side)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float64)


def _geoms(path):
    d = yaml.safe_load(open(path))
    return d, [(g["L"], g["W"]) for g in d["geometries"]]


def build_manifest(train_yaml=TRAIN_YAML, test_yaml=TEST_YAML):
    """The flat, deterministic list of every config to simulate, tagged by split."""
    dtr, gtr = _geoms(train_yaml)
    _, gte = _geoms(test_yaml)
    items = []
    for split, cfgs in (
        ("train", enumerate_configs(gtr, exclude_combos=HELDOUT_COMBOS)),
        ("test_i_iii", enumerate_configs(gte)),
        ("split_ii", enumerate_configs(gtr, only_combos=HELDOUT_COMBOS, include_baseline=False)),
        ("split_iv", enumerate_configs(gte, unseen_alpha=UNSEEN_ALPHA)),
    ):
        for c in cfgs:
            items.append((split, c))
    # A config could only appear twice if it is literally the same room; the filename is
    # the identity, so dedupe on it and keep the first split that claims it.
    seen, out = set(), []
    for split, c in items:
        if c.filename in seen:
            continue
        seen.add(c.filename)
        out.append((split, c))
    return dtr, out


def build_manifest_2b(path=P3_2B_MANIFEST):
    """P3-2b: read the FROZEN manifest rather than re-deriving configs.

    The sampled set is frozen in git so the dataset, the trainer and the eval all agree on
    exactly which rooms exist, independent of the sampler code. Returns the same
    ``(common, [(split, cfg), ...])`` shape as ``build_manifest`` so the task loop is shared.
    """
    from aaf.data.mat_configs_cont import configs_from_rows
    d = json.load(open(path))
    common = yaml.safe_load(open(TRAIN_YAML))
    cfgs = configs_from_rows(d["configs"])
    return common, [(c.split, c) for c in cfgs]


def write_h5(path: Path, cfg: MatConfig, ism: dict, analytical, meta_extra: dict):
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        g = f.create_group("ism")
        g.create_dataset("H_complex", data=ism["H_complex"], compression="gzip")
        g.create_dataset("rir_time", data=ism["rir_time"], compression="gzip")
        for k, v in ism["meta"].items():
            g.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        if analytical is not None:
            ga = f.create_group("analytical")
            ga.create_dataset("H_complex", data=analytical["H_complex"], compression="gzip")
            ga.create_dataset("rir_time", data=analytical["rir_time"], compression="gzip")
            for k, v in analytical["meta"].items():
                ga.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
            ga.attrs["valid_for"] = "baseline (uniform alpha) only"
        for k, v in meta_extra.items():
            f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        f.flush()
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None,
                    help="task index; with --chunk>1 this selects a BLOCK of configs")
    ap.add_argument("--chunk", type=int, default=1,
                    help="configs built per task (keeps the array under the QOS submit cap)")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--train-yaml", default=TRAIN_YAML)
    ap.add_argument("--test-yaml", default=TEST_YAML)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", choices=("p3_2", "p3_2b"), default="p3_2",
                    help="p3_2 = the original preset enumeration; p3_2b = the frozen "
                         "continuous-m manifest")
    ap.add_argument("--manifest-path", default=P3_2B_MANIFEST)
    a = ap.parse_args()

    if a.set == "p3_2b":
        common, manifest = build_manifest_2b(a.manifest_path)
    else:
        common, manifest = build_manifest(a.train_yaml, a.test_yaml)
    if a.manifest or a.idx is None:
        counts = {}
        for split, _ in manifest:
            counts[split] = counts.get(split, 0) + 1
        print(json.dumps({"n_total": len(manifest), "by_split": counts,
                          "array_range": "0-%d" % (len(manifest) - 1)}, indent=2))
        return

    chunk = max(1, int(a.chunk))
    lo = a.idx * chunk
    if not 0 <= lo < len(manifest):
        n_tasks = (len(manifest) + chunk - 1) // chunk
        raise SystemExit("--idx %d out of range 0..%d (chunk=%d)" % (a.idx, n_tasks - 1, chunk))
    block = manifest[lo:lo + chunk]

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, (split, cfg) in enumerate(block):
        _build_one(lo + j, split, cfg, common, out_dir, a.force)


def _build_one(i, split, cfg, common, out_dir, force):
    out_path = out_dir / cfg.filename
    done = out_dir / (cfg.filename + ".done")
    if done.exists() and out_path.exists() and not force:
        print("[%d] SKIP (done): %s" % (i, cfg.filename))
        return

    rx = make_receiver_grid_2d(cfg.L, cfg.W, common["n_rx_per_side"], common["rx_margin"])
    ism = simulate_room_2d(dict(
        L=cfg.L, W=cfg.W,
        source_pos=np.asarray(common["source_pos"], dtype=float),
        receiver_pos=rx, alphas=cfg.alphas,
        fs=common["fs"], n_time_samples=common["n_time_samples"],
        max_order=common["max_order"]))

    analytical = None
    if cfg.is_baseline:
        analytical = modal_rir_2d(dict(
            L=cfg.L, W=cfg.W, source_pos=np.asarray(common["source_pos"], dtype=float),
            receiver_pos=rx, alpha=float(cfg.alphas[0]),
            fs=common["fs"], n_time_samples=common["n_time_samples"]))

    write_h5(out_path, cfg, ism, analytical, {
        "L": cfg.L, "W": cfg.W,
        "alphas": [float(x) for x in cfg.alphas],
        "walls": list(WALLS_2D),
        "wall_edited": cfg.wall if cfg.wall else "none",
        "edited_walls": list(getattr(cfg, "edited", ()) or ([cfg.wall] if cfg.wall else [])),
        "kind": getattr(cfg, "kind", "preset"),
        "material": cfg.material if cfg.material else "M0",
        "is_baseline": bool(cfg.is_baseline),
        "split": split, "label": cfg.label,
        "alpha_eff": float(ism["meta"]["alpha"]),
        "receiver_pos": rx.tolist(),
        "source_pos": list(common["source_pos"]),
        "fs": common["fs"], "n_time_samples": common["n_time_samples"],
        "n_freq_bins": common["n_time_samples"] // 2 + 1,
        "max_order": common["max_order"], "n_rx": int(rx.shape[0]),
        "has_analytical": analytical is not None, "chunk": "P3-2",
    })
    done.write_text("ok\n")
    print("[%d] %-11s %s -> %s" % (i, split, cfg.label, cfg.filename))


if __name__ == "__main__":
    main()
