"""P2-3 meeting assets: for the best-K zero-shot test rooms, collect the
signal plots already emitted by the eval and add a receiver-volume-slice plot
(|H| at a low modal frequency, predicted vs ISM).

Reads ``<train_output_dir>/zero_shot/L*/metrics.json`` (+ each room's saved
``H_pred_all.pt``) and the matching ISM ground-truth H5, ranks rooms by
magnitude correlation, and writes ``outputs/meeting_assets_p2_3/<room>/``.

CPU-only (no torch.cuda); safe on a small allocation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from aaf.data.dataset_builder import read_room_h5  # noqa: E402
from aaf.eval.signal_level import make_receiver_slices  # noqa: E402

C_SOUND = 343.0
SIGNAL_PLOTS = (
    "magnitude_overlay.png", "phase_overlay.png", "rir_overlay.png",
    "edc_overlay.png", "signal_metrics_summary.png", "modal_overlay.png",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-output-dir", required=True)
    ap.add_argument("--data-dir", default="data/track_a_3d")
    ap.add_argument("--out-dir", default="outputs/meeting_assets_p2_3")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--fs", type=float, default=4096.0)
    ap.add_argument("--n-time", type=int, default=8192)
    args = ap.parse_args()

    train_dir = Path(args.train_output_dir)
    zs_root = train_dir / "zero_shot"
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rooms = []
    for mp in sorted(zs_root.glob("L*_W*_H*/metrics.json")):
        try:
            ev = json.loads(mp.read_text())
        except Exception as e:
            print(f"# skip {mp}: {e!r}")
            continue
        mag = (ev.get("signal_metrics") or {}).get("mag_corr")
        rooms.append((mag if mag is not None else -1.0, mp.parent, ev))
    if not rooms:
        sys.exit(f"no metrics.json under {zs_root}")
    rooms.sort(key=lambda t: t[0], reverse=True)
    top = rooms[: args.top_k]
    print(f"# ranking {len(rooms)} rooms; top-{len(top)} by mag_corr: "
          + ", ".join(f"{r[1].name}={r[0]:.3f}" for r in top))

    for mag, room_dir, ev in top:
        name = room_dir.name
        dest = out_root / name
        dest.mkdir(parents=True, exist_ok=True)
        # 1. copy the signal plots the eval already wrote
        for fn in SIGNAL_PLOTS:
            src = room_dir / fn
            if src.exists():
                shutil.copy2(src, dest / fn)
        # 2. receiver-volume slices (predicted vs ISM) at a low axial mode
        hpred_p = room_dir / "H_pred_all.pt"
        L, W, H = ev.get("L"), ev.get("W"), ev.get("H")
        h5 = Path(args.data_dir) / f"L{L:.2f}_W{W:.2f}_H{H:.2f}.h5"
        if hpred_p.exists() and h5.exists():
            H_pred = torch.load(hpred_p).numpy()
            rt = read_room_h5(h5)
            H_tgt = rt["ism_H"].astype(np.complex64)
            rx_pos = np.asarray(rt["attrs"]["receiver_pos"], dtype=np.float32)
            f_mode = C_SOUND / (2.0 * float(L))     # (1,0,0) axial mode along L
            p = make_receiver_slices(
                H_pred, H_tgt, rx_pos, fs=args.fs, n_time_samples=args.n_time,
                output_dir=dest, f_target_hz=f_mode,
            )
            print(f"#   {name}: mag_corr={mag:.3f}  receiver slices @ {f_mode:.0f} Hz → {p.name}")
        else:
            print(f"#   {name}: mag_corr={mag:.3f}  (missing H_pred_all.pt or H5 — slices skipped)")
    print(f"# meeting assets in {out_root}")


if __name__ == "__main__":
    main()
