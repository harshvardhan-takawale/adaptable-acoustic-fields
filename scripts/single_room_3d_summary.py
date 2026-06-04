"""Aggregate per-room ``eval.json`` files into a cross-room summary for P2-1.

Reads ``outputs/single_room_3d/<run>/eval.json`` files, produces:
  - ``outputs/single_room_3d/SUMMARY.md``: per-room table (L, W, H, ckpt_iter,
    f_Schroeder, modal MAE, full-band LSD, mag/phase/RIR/EDC/envelope corrs)
    plus a Phase-1 (2D) reference row pulled from ``outputs/single_room/SUMMARY.md``
    if present.
  - ``outputs/single_room_3d/lsd_vs_volume.png``: full-band LSD as a function
    of room volume V = L·W·H.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _safe(v, fmt=".2f"):
    try:
        return format(float(v), fmt)
    except Exception:
        return "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--single-room-dir", default=str(REPO_ROOT / "outputs/single_room_3d"),
    )
    args = ap.parse_args()

    root = Path(args.single_room_dir)
    eval_files = sorted(root.glob("*/eval.json"))
    if not eval_files:
        sys.exit(f"no eval.json under {root}")

    rows = []
    for p in eval_files:
        try:
            ev = json.loads(p.read_text())
        except Exception as e:
            print(f"# skip {p}: {e!r}")
            continue
        rows.append({
            "run": p.parent.name,
            "L": ev.get("L"), "W": ev.get("W"), "H": ev.get("H"),
            "volume": (ev.get("L", 0) or 0) * (ev.get("W", 0) or 0)
                      * (ev.get("H", 0) or 0),
            "ckpt_iter": ev.get("ckpt_iter"),
            "f_schroeder_hz": ev.get("f_schroeder_hz"),
            "modal_mae_hz": ev.get("modal", {}).get("mae_hz"),
            "modal_recall": ev.get("modal", {}).get("recall_at_tol"),
            "n_modes_modal": ev.get("n_modes_modal_band"),
            "lsd_db": ev.get("full_band", {}).get("lsd_db"),
            "complex_l1": ev.get("full_band", {}).get("complex_l1"),
            "phase_l1": ev.get("full_band", {}).get("phase_l1"),
            "mag_corr": ev.get("signal_metrics", {}).get("mag_corr"),
            "phase_corr_mw": ev.get("signal_metrics", {}).get("phase_corr_mw"),
            "rir_pearson": ev.get("signal_metrics", {}).get("rir_pearson"),
            "edc_max_db": ev.get("signal_metrics", {}).get("edc_max_db"),
            "edc_rmse_db": ev.get("signal_metrics", {}).get("edc_rmse_db"),
            "early_corr": ev.get("signal_metrics", {}).get("early_corr"),
            "late_corr": ev.get("signal_metrics", {}).get("late_corr"),
            "envelope_corr": ev.get("signal_metrics", {}).get("envelope_corr"),
        })

    rows.sort(key=lambda r: (r["L"] or 0, r["W"] or 0, r["H"] or 0))

    md = [
        "# Single-room 3D summary (P2-1)\n",
        "\nAggregated from each de-risk room's `eval.json`. Modal MAE is reported\n"
        "in the f<f_Schroeder band only (DECISIONS.md D18 — above f_Schroeder,\n"
        "3D modal density exceeds the RFFT resolution Δf=0.5 Hz).\n",
        "\n## Per-room metrics\n",
        "| Run | L | W | H | V (m³) | ckpt | f_S (Hz) | modal MAE (Hz) | LSD (dB) | "
        "mag corr | phase corr (mw) | RIR Pearson | EDC RMS (dB) | early/late | env corr |\n",
        "|---|---:|---:|---:|------:|------:|---------:|----------------:|---------:|"
        "---------:|----------------:|-----------:|-------------:|-----------|---------:|\n",
    ]
    for r in rows:
        md.append(
            f"| {r['run']} | {_safe(r['L'])} | {_safe(r['W'])} | {_safe(r['H'])} | "
            f"{_safe(r['volume'], '.1f')} | {r['ckpt_iter']} | "
            f"{_safe(r['f_schroeder_hz'], '.0f')} | "
            f"{_safe(r['modal_mae_hz'])} | {_safe(r['lsd_db'])} | "
            f"{_safe(r['mag_corr'], '.3f')} | {_safe(r['phase_corr_mw'], '.3f')} | "
            f"{_safe(r['rir_pearson'], '.3f')} | {_safe(r['edc_rmse_db'])} | "
            f"{_safe(r['early_corr'], '.2f')} / {_safe(r['late_corr'], '.2f')} | "
            f"{_safe(r['envelope_corr'], '.3f')} |\n"
        )

    # 2D Phase-1 reference (if SUMMARY.md is present, pull modal-MAE / LSD ranges).
    ref = REPO_ROOT / "outputs/single_room/SUMMARY.md"
    if ref.exists():
        md.append("\n## Phase-1 (2D) baseline reference\n")
        md.append(f"See [`outputs/single_room/SUMMARY.md`]({ref.relative_to(REPO_ROOT)}). "
                  "Modal MAE 0.34–0.58 Hz on matched peaks; full-band LSD 0.36–0.42 dB.\n")

    # LSD vs volume plot.
    if any(r.get("volume") and r.get("lsd_db") is not None for r in rows):
        vols = [r["volume"] for r in rows]
        lsds = [r["lsd_db"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(vols, lsds, color="steelblue", s=60, alpha=0.85)
        for r, v, l in zip(rows, vols, lsds):
            if v is None or l is None:
                continue
            ax.annotate(
                r["run"], (v, l), fontsize=6, textcoords="offset points",
                xytext=(4, 4), alpha=0.7,
            )
        ax.set_xlabel("Room volume V = L·W·H (m³)")
        ax.set_ylabel("Full-band LSD (dB)")
        ax.set_title("Single-room 3D overfit: LSD vs volume")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = root / "lsd_vs_volume.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        md.append(f"\n## LSD vs room volume\n\n![]({out_png.relative_to(root)})\n")

    summary_path = root / "SUMMARY.md"
    summary_path.write_text("".join(md))
    print(f"# wrote {summary_path}")


if __name__ == "__main__":
    main()
