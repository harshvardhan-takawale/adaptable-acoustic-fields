"""Build one 3D room of the Phase-2 dataset (one SLURM array task per room).

CLI:
    python scripts/build_3d_dataset.py --rooms-yaml configs/sweeps_3d/derisk_rooms.yaml \
        --idx 0

Per-task behavior:
  1. Read room (L, W, H) at index ``idx`` from the rooms YAML.
  2. If ``data/track_a_3d/<filename>.done`` sentinel exists, exit 0 (idempotent).
  3. Build ISM + analytical, write HDF5 to ``data/track_a_3d/<filename>``
     atomically (tmp → fsync → rename), then write the ``.done`` sentinel.
  4. Exit 0.

The sentinel makes each task fully independent + preempt-safe: if SLURM kills
a task partway, re-submitting just that array index re-runs it cleanly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from aaf.data.dataset_builder import room_filename_3d, write_room_3d_to_h5
from aaf.sim.analytical_modal_3d import modal_rir_3d
from aaf.sim.ism_3d import simulate_room_3d


def make_receiver_grid_3d(
    L: float, W: float, H: float, n_per_side: int = 8, margin: float = 0.3
) -> np.ndarray:
    """8×8×8 = 512 receivers row-major over (z, y, x)."""
    if L - 2 * margin <= 0 or W - 2 * margin <= 0 or H - 2 * margin <= 0:
        raise ValueError(
            f"Room L={L}, W={W}, H={H} too small for receiver grid "
            f"with margin={margin} m and n_per_side={n_per_side}"
        )
    xs = np.linspace(margin, L - margin, n_per_side)
    ys = np.linspace(margin, W - margin, n_per_side)
    zs = np.linspace(margin, H - margin, n_per_side)
    return np.array(
        [[x, y, z] for z in zs for y in ys for x in xs],
        dtype=np.float64,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms-yaml", required=True, type=str)
    ap.add_argument("--idx", required=True, type=int,
                    help="0-based index into the YAML's rooms list.")
    ap.add_argument(
        "--data-dir", default=str(REPO_ROOT / "data/track_a_3d"),
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the .done sentinel exists.",
    )
    args = ap.parse_args()

    with open(args.rooms_yaml) as f:
        cfg = yaml.safe_load(f)
    rooms = cfg["rooms"]
    if args.idx < 0 or args.idx >= len(rooms):
        sys.exit(f"--idx {args.idx} out of range [0, {len(rooms)})")
    room = rooms[args.idx]
    L = float(room["L"])
    W = float(room["W"])
    H = float(room["H"])

    alpha = float(cfg["alpha"])
    fs = int(cfg["fs"])
    n_time_samples = int(cfg["n_time_samples"])
    source_offset = tuple(float(x) for x in cfg.get("source_offset", (0.5, 0.5, 0.5)))

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / room_filename_3d(L=L, W=W, H=H)
    sentinel = data_dir / (out_path.name + ".done")

    if sentinel.exists() and not args.force:
        print(f"[skip] sentinel exists: {sentinel}")
        sys.exit(0)

    print(f"# build idx={args.idx}  L={L} W={W} H={H}  set={cfg.get('set_name', '?')}")
    print(f"#   out: {out_path}")
    print(f"#   alpha={alpha} fs={fs} n_time={n_time_samples} source_offset={source_offset}")

    receiver_pos = make_receiver_grid_3d(L=L, W=W, H=H, n_per_side=8, margin=0.3)
    build_cfg = {
        "L": L, "W": W, "H": H,
        "source_pos": source_offset,
        "receiver_pos": receiver_pos,
        "alpha": alpha,
        "fs": fs,
        "n_time_samples": n_time_samples,
    }

    t0 = time.time()
    ism = simulate_room_3d(build_cfg)
    t_ism = time.time() - t0
    print(f"#   ISM done in {t_ism:.1f}s  max_order={ism['meta']['max_order']} "
          f"T60={ism['meta']['T60_sabine_3d']:.2f}s")

    t1 = time.time()
    ana = modal_rir_3d(build_cfg)
    t_ana = time.time() - t1
    print(f"#   analytical done in {t_ana:.1f}s  n_modes={ana['meta']['n_modes']}")

    # Atomic write: tmp → fsync → rename → sentinel
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    write_room_3d_to_h5(
        tmp, ism, ana,
        sweep_meta={
            "set_name": cfg.get("set_name", "?"),
            "rooms_yaml": str(Path(args.rooms_yaml).name),
            "build_wall_clock_s": t_ism + t_ana,
        },
    )
    # h5py closes file in write_room_*; force fsync of the underlying tmp.
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(out_path)
    sentinel.write_text(f"ok\nL={L}\nW={W}\nH={H}\nwall={t_ism + t_ana:.2f}s\n")
    size_mb = out_path.stat().st_size / 1e6
    print(f"#   wrote {out_path}  ({size_mb:.1f} MB)  total wall {t_ism + t_ana:.1f}s")
    sys.exit(0)


if __name__ == "__main__":
    main()
