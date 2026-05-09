"""Generate one HDF5 file per room over the union of L values needed by the
three sweep configs (dense, sparse, extrapolation).

Receiver layout per room: 8×8 uniform grid on [0.3, L-0.3] × [0.3, W-0.3] m.
If a room is too small for that margin, fail loudly.

Source position: fixed at (0.5, 0.5) m (corner offset) per spec.

Skips files that already exist unless --force is set.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aaf.sim.ism_2d import simulate_room_2d
from aaf.sim.analytical_modal_2d import modal_rir_2d
from aaf.data.dataset_builder import write_room_to_h5, room_filename


def union_L_set(configs_dir: Path) -> tuple[list[float], dict]:
    """Return sorted unique L values across the three sweep YAMLs + the common
    cfg fields they all share."""
    paths = {
        "dense": configs_dir / "dense.yaml",
        "sparse": configs_dir / "sparse.yaml",
        "extrapolation": configs_dir / "extrapolation.yaml",
    }
    union: set[float] = set()
    common: dict = {}
    common_keys = ("W", "alpha", "fs", "n_time_samples", "source_pos")
    for name, p in paths.items():
        with open(p) as f:
            cfg = yaml.safe_load(f)
        for L in cfg["train_L"] + cfg["test_L"]:
            union.add(float(L))
        for k in common_keys:
            if k in common and common[k] != cfg[k]:
                raise ValueError(
                    f"sweep {name} disagrees with previous on {k}: {cfg[k]} vs {common[k]}"
                )
            common[k] = cfg[k]
    return sorted(union), common


def make_receiver_grid(L: float, W: float, n_per_side: int = 8, margin: float = 0.3) -> np.ndarray:
    """8×8 uniform grid on [margin, L-margin] × [margin, W-margin]. Fail if too small."""
    if L - 2 * margin <= 0 or W - 2 * margin <= 0:
        raise ValueError(
            f"Room L={L}, W={W} too small for receiver grid with margin={margin} m"
        )
    xs = np.linspace(margin, L - margin, n_per_side)
    ys = np.linspace(margin, W - margin, n_per_side)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    return grid


def build_one(L: float, common: dict, out_dir: Path, force: bool) -> tuple[Path, float]:
    """Build one room. Returns (out_path, elapsed_seconds)."""
    W = float(common["W"])
    alpha = float(common["alpha"])
    fs = float(common["fs"])
    n_time_samples = int(common["n_time_samples"])
    source_pos = tuple(common["source_pos"])

    out_path = out_dir / room_filename(L=L, W=W, alpha=alpha)
    if out_path.exists() and not force:
        return out_path, 0.0

    receiver_pos = make_receiver_grid(L=L, W=W, n_per_side=8, margin=0.3)

    cfg = {
        "L": L,
        "W": W,
        "source_pos": source_pos,
        "receiver_pos": receiver_pos,
        "alpha": alpha,
        "fs": fs,
        "n_time_samples": n_time_samples,
    }
    t0 = time.time()
    ism = simulate_room_2d(cfg)
    t_ism = time.time() - t0

    t1 = time.time()
    ana = modal_rir_2d(cfg)
    t_ana = time.time() - t1

    write_room_to_h5(out_path, ism, ana, sweep_meta={"common": common, "L_set": "union"})
    return out_path, t_ism + t_ana


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default=str(REPO_ROOT / "configs/sweeps"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data/track_a"))
    ap.add_argument("--force", action="store_true", help="rebuild files even if present")
    args = ap.parse_args()

    configs_dir = Path(args.configs_dir)
    out_dir = Path(args.out_dir)

    L_list, common = union_L_set(configs_dir)
    print(f"# union L set ({len(L_list)} rooms): {L_list}")
    print(f"# common cfg: {common}")
    print(f"# out_dir: {out_dir}")

    total = 0.0
    written = []
    for i, L in enumerate(L_list, 1):
        try:
            out_path, secs = build_one(L=L, common=common, out_dir=out_dir, force=args.force)
        except Exception as e:
            print(f"[{i:2d}/{len(L_list)}] L={L:.2f} FAILED: {type(e).__name__}: {e}")
            raise
        total += secs
        written.append(out_path)
        size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0
        status = "skipped" if secs == 0.0 else f"{secs:5.1f}s"
        print(f"[{i:2d}/{len(L_list)}] L={L:.2f}  {status}  -> {out_path.name}  ({size_mb:.2f} MB)")

    print(f"# total wall: {total:.1f}s ({total/60:.1f} min) for {len(written)} files")


if __name__ == "__main__":
    main()
