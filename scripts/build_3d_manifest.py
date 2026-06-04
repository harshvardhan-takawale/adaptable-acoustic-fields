"""Build/refresh ``data/track_a_3d/manifest.json`` from .done sentinels.

Scans the 3D data dir for ``L*.h5.done`` files, parses each, and writes a
single JSON manifest that records which rooms are present + their wall-clock
+ which rooms-YAML they came from. Idempotent.

The manifest is consumed by:
  - ``scripts/run_p2_1_pipeline.sh`` to decide when training can start (gate
    on the 5 de-risk rooms being done).
  - P2-2 (and beyond) to know which training rooms are available.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


DONE_SUFFIX = ".h5.done"
SENTINEL_LWH_RE = re.compile(r"^L([\d.]+)_W([\d.]+)_H([\d.]+)\.h5\.done$")


def parse_sentinel(p: Path) -> dict:
    m = SENTINEL_LWH_RE.match(p.name)
    if not m:
        return {"path": str(p), "ok": False, "error": "bad filename"}
    L, W, H = (float(x) for x in m.groups())
    h5 = p.with_suffix("")  # drop ".done"
    info = {
        "path": str(p.relative_to(p.parent.parent)),
        "h5_path": str(h5.relative_to(p.parent.parent)),
        "h5_exists": h5.exists(),
        "h5_size_mb": (h5.stat().st_size / 1e6) if h5.exists() else None,
        "L": L, "W": W, "H": H,
        "ok": True,
    }
    try:
        body = p.read_text()
        for line in body.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "wall":
                    info["wall_clock"] = v.strip()
    except Exception as e:
        info["sentinel_read_error"] = str(e)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir", default=str(REPO_ROOT / "data/track_a_3d"),
    )
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"data dir does not exist: {data_dir}")

    sentinels = sorted(data_dir.glob(f"*{DONE_SUFFIX}"))
    rooms = [parse_sentinel(p) for p in sentinels]
    rooms_ok = [r for r in rooms if r["ok"] and r["h5_exists"]]
    rooms_bad = [r for r in rooms if not r["ok"] or not r["h5_exists"]]

    manifest = {
        "schema": "aaf-3d-dataset-manifest/v1",
        "data_dir": str(data_dir.relative_to(data_dir.parent.parent)),
        "updated_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_rooms": len(rooms_ok),
        "n_rooms_total_mb": sum(r.get("h5_size_mb", 0.0) or 0.0 for r in rooms_ok),
        "rooms": rooms_ok,
        "incomplete": rooms_bad,
    }
    out_path = data_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"# wrote {out_path}  ({len(rooms_ok)} rooms, "
          f"{manifest['n_rooms_total_mb']:.1f} MB total)")
    if rooms_bad:
        print(f"# WARNING: {len(rooms_bad)} sentinel(s) point to missing or bad HDF5 files")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
