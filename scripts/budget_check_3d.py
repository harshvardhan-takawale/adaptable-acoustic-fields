"""Budget check before the full 3D dataset generation.

Simulates 2 rooms (smallest + largest of the de-risk set) and reports
wall-clock + HDF5 file size. Hard exit if either:
  - per-room wall-clock > 10 min
  - per-room HDF5 size > 500 MB

Writes ``outputs/budget_check_3d/REPORT.md`` and ``result.json``.

If the check fails, appends to OPEN_QUESTIONS.md with the recommended fix
(reduce receiver count, lower max_order cap, or chunk the analytical sum).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from aaf.data.dataset_builder import room_filename_3d, write_room_3d_to_h5
from aaf.data.sample_rooms_3d import (
    DEFAULT_ALPHA,
    DEFAULT_FS,
    DEFAULT_N_TIME,
    DEFAULT_SOURCE_OFFSET,
)
from aaf.sim.analytical_modal_3d import modal_rir_3d
from aaf.sim.ism_3d import simulate_room_3d


PER_ROOM_WALL_LIMIT_S = 600.0
PER_ROOM_SIZE_LIMIT_MB = 500.0


def make_receiver_grid_3d(
    L: float, W: float, H: float, n_per_side: int = 8, margin: float = 0.3
) -> np.ndarray:
    """8×8×8 = 512 receivers uniform inside the room with ``margin`` from walls.

    Order: row-major over (z, y, x), so receiver i maps to (iz·64 + iy·8 + ix).
    """
    if L - 2 * margin <= 0 or W - 2 * margin <= 0 or H - 2 * margin <= 0:
        raise ValueError(
            f"Room L={L}, W={W}, H={H} too small for receiver grid "
            f"with margin={margin} m and n_per_side={n_per_side}"
        )
    xs = np.linspace(margin, L - margin, n_per_side)
    ys = np.linspace(margin, W - margin, n_per_side)
    zs = np.linspace(margin, H - margin, n_per_side)
    grid = np.array(
        [[x, y, z] for z in zs for y in ys for x in xs],
        dtype=np.float64,
    )
    return grid


def build_one(
    L: float, W: float, H: float,
    out_path: Path,
    alpha: float = DEFAULT_ALPHA,
    fs: int = DEFAULT_FS,
    n_time_samples: int = DEFAULT_N_TIME,
    source_offset: tuple = DEFAULT_SOURCE_OFFSET,
) -> tuple[float, float, dict]:
    """Build one room → HDF5. Returns (wall_clock_s, size_mb, meta)."""
    source_pos = (
        float(source_offset[0]),
        float(source_offset[1]),
        float(source_offset[2]),
    )
    receiver_pos = make_receiver_grid_3d(L=L, W=W, H=H, n_per_side=8, margin=0.3)

    cfg = {
        "L": L, "W": W, "H": H,
        "source_pos": source_pos,
        "receiver_pos": receiver_pos,
        "alpha": alpha,
        "fs": fs,
        "n_time_samples": n_time_samples,
    }
    t0 = time.time()
    ism = simulate_room_3d(cfg)
    t_ism = time.time() - t0
    print(f"#   ISM ({L}x{W}x{H}) done in {t_ism:.1f}s")

    t1 = time.time()
    ana = modal_rir_3d(cfg)
    t_ana = time.time() - t1
    print(f"#   analytical ({L}x{W}x{H}) done in {t_ana:.1f}s "
          f"({ana['meta']['n_modes']} modes, {ana['meta']['n_distinct_freqs']} distinct freqs)")

    write_room_3d_to_h5(out_path, ism, ana, sweep_meta={"budget_check": True})
    size_mb = out_path.stat().st_size / 1e6
    total_s = t_ism + t_ana
    meta = {
        "L": L, "W": W, "H": H,
        "t_ism_s": t_ism,
        "t_analytical_s": t_ana,
        "wall_clock_s": total_s,
        "size_mb": size_mb,
        "ism_max_order": ism["meta"]["max_order"],
        "ism_max_order_was_capped": ism["meta"]["max_order_was_capped"],
        "T60_sabine_3d": ism["meta"]["T60_sabine_3d"],
        "schroeder_freq_hz": ism["meta"]["schroeder_freq_hz"],
        "ir_pra_length_max": ism["meta"]["ir_pra_length_max"],
        "ir_truncated": ism["meta"]["ir_truncated"],
        "n_analytical_modes": ana["meta"]["n_modes"],
        "n_distinct_freqs": ana["meta"]["n_distinct_freqs"],
    }
    return total_s, size_mb, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir", default=str(REPO_ROOT / "outputs/budget_check_3d"),
    )
    ap.add_argument(
        "--data-dir", default=str(REPO_ROOT / "data/track_a_3d"),
    )
    ap.add_argument(
        "--keep-files", action="store_true",
        help="Keep the budget-check HDF5 files (otherwise removed after measuring).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Rooms to test: smallest + largest of the de-risk set.
    rooms_to_test = [
        ("smallest", (3.0, 3.0, 2.5)),
        ("largest", (6.0, 5.0, 4.0)),
    ]

    results = []
    for label, (L, W, H) in rooms_to_test:
        out_path = data_dir / room_filename_3d(L=L, W=W, H=H)
        print(f"# building {label} room L={L} W={W} H={H}  →  {out_path}")
        wall, size, meta = build_one(L=L, W=W, H=H, out_path=out_path)
        meta["label"] = label
        results.append(meta)
        print(
            f"  wall={wall:.1f}s  size={size:.1f} MB  ISM max_order={meta['ism_max_order']} "
            f"(capped={meta['ism_max_order_was_capped']})  T60={meta['T60_sabine_3d']:.2f}s"
        )
        if not args.keep_files:
            out_path.unlink()
            (out_path.parent / (out_path.name + ".done")).unlink(missing_ok=True)

    # Pass/fail logic.
    max_wall = max(r["wall_clock_s"] for r in results)
    max_size = max(r["size_mb"] for r in results)
    passed = max_wall <= PER_ROOM_WALL_LIMIT_S and max_size <= PER_ROOM_SIZE_LIMIT_MB

    md = ["# 3D dataset budget check\n",
          f"\n**Status**: {'PASS' if passed else 'FAIL'}\n",
          f"\n- Per-room wall-clock limit: {PER_ROOM_WALL_LIMIT_S:.0f} s\n",
          f"- Per-room file-size limit: {PER_ROOM_SIZE_LIMIT_MB:.0f} MB\n",
          f"- Worst-case wall observed: {max_wall:.1f} s\n",
          f"- Worst-case size observed: {max_size:.1f} MB\n",
          "\n## Per-room measurements\n",
          "| Label | L | W | H | wall (s) | t_ISM (s) | t_analytical (s) | size (MB) | max_order | T60 (s) | n_modes |\n",
          "|---|---:|---:|---:|---------:|---------:|----------------:|----------:|----------:|--------:|--------:|\n"]
    for r in results:
        md.append(
            f"| {r['label']} | {r['L']:.2f} | {r['W']:.2f} | {r['H']:.2f} | "
            f"{r['wall_clock_s']:.1f} | {r['t_ism_s']:.1f} | {r['t_analytical_s']:.1f} | "
            f"{r['size_mb']:.1f} | "
            f"{r['ism_max_order']}{' (capped)' if r['ism_max_order_was_capped'] else ''} | "
            f"{r['T60_sabine_3d']:.2f} | {r['n_analytical_modes']} |\n"
        )

    if not passed:
        md.append("\n## Recommended fixes (see OPEN_QUESTIONS.md):\n")
        md.append(
            "- Reduce receiver count (e.g., 6×6×6=216 instead of 8×8×8=512).\n"
            "- Lower MAX_ORDER_CAP in `aaf/sim/ism_3d.py` (currently 17).\n"
            "- Chunk the analytical modal sum.\n"
        )
        oq = REPO_ROOT / "OPEN_QUESTIONS.md"
        existing = oq.read_text() if oq.exists() else ""
        oq.write_text(existing + (
            "\n### NEW (P2-1 budget check failed): 3D dataset per-room "
            f"wall-clock {max_wall:.1f}s or size {max_size:.1f}MB exceeded budget.\n"
            "See `outputs/budget_check_3d/REPORT.md` for measurements and "
            "recommended fixes.\n"
        ))

    (out_dir / "REPORT.md").write_text("".join(md))
    (out_dir / "result.json").write_text(json.dumps(
        {"status": "pass" if passed else "fail",
         "per_room_wall_limit_s": PER_ROOM_WALL_LIMIT_S,
         "per_room_size_limit_mb": PER_ROOM_SIZE_LIMIT_MB,
         "results": results}, indent=2))
    print(f"# wrote {out_dir / 'REPORT.md'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
