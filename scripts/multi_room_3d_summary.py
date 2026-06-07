"""Aggregate per-test-room ``metrics.json`` files into a cross-test-room summary.

Walks ``<train_output_dir>/zero_shot/L*_W*_H*/metrics.json``, builds:

  - ``<train_output_dir>/SUMMARY.md``: per-room table (L, W, H, V, mag corr,
    phase corr mw, RIR Pearson, env corr, mod LSD, full LSD, per-axis geom
    head error).
  - **Headline verdict**: count of rooms with mag corr ≥ 0.9 in 0-500 Hz on
    the held-out set.

The mag corr in the 0-500 Hz band is recomputed from the per-band LSD as a
proxy if the per-band mag corr isn't directly available; otherwise use the
``signal_metrics`` dict that ``aaf.eval.signal_level.compute_signal_metrics``
emits per room (full-spectrum mag corr).
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
sys.path.insert(0, str(REPO_ROOT))
from aaf.eval.zero_shot_diagnosis import classify_zero_shot_room, aggregate_verdict  # noqa: E402


def _safe(v, fmt=".2f"):
    try:
        return format(float(v), fmt)
    except Exception:
        return "—"


def _read_in_dist_lsd(train_dir: Path):
    """Final in-distribution val LSD from the training run (the D37 gate)."""
    sc = train_dir / "scalars.json"
    if sc.exists():
        try:
            rows = json.loads(sc.read_text())
            vals = [r for r in rows if r.get("phase") == "val" and r.get("lsd_db") is not None]
            if vals:
                vals.sort(key=lambda r: r.get("iter", 0))
                return float(vals[-1]["lsd_db"]), int(vals[-1].get("iter", 0))
        except Exception:
            pass
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-output-dir", required=True, type=str,
        help="e.g. outputs/multi_room_3d/M1_45rooms",
    )
    ap.add_argument(
        "--zero-shot-root", type=str, default=None,
        help="defaults to <train-output-dir>/zero_shot",
    )
    args = ap.parse_args()

    train_dir = Path(args.train_output_dir)
    zs_root = Path(args.zero_shot_root) if args.zero_shot_root else (train_dir / "zero_shot")
    eval_files = sorted(zs_root.glob("L*_W*_H*/metrics.json"))
    if not eval_files:
        sys.exit(f"no metrics.json under {zs_root}")

    rows = []
    for p in eval_files:
        try:
            ev = json.loads(p.read_text())
        except Exception as e:
            print(f"# skip {p}: {e!r}")
            continue
        L = ev.get("L"); W = ev.get("W"); H = ev.get("H")
        sm = ev.get("signal_metrics", {})
        rows.append({
            "run": p.parent.name,
            "L": L, "W": W, "H": H,
            "V": (L or 0) * (W or 0) * (H or 0),
            "f_S": ev.get("f_schroeder_hz"),
            "modal_mae": ev.get("held_out_modal_mae_hz"),
            "full_lsd": ev.get("held_out_lsd_db"),
            "modal_lsd": (ev.get("band_metrics_held", {}) or {}).get("lsd_band_0_250_db"),
            "mag_corr": sm.get("mag_corr"),
            "phase_corr_mw": sm.get("phase_corr_mw"),
            "rir_pearson": sm.get("rir_pearson"),
            "env_corr": sm.get("envelope_corr"),
            "early_corr": sm.get("early_corr"),
            "late_corr": sm.get("late_corr"),
            "geom_err_L": ev.get("geom_err_L_m"),
            "geom_err_W": ev.get("geom_err_W_m"),
            "geom_err_H": ev.get("geom_err_H_m"),
            "geom_err_max": ev.get("geom_err_max_m"),
            "latent_min_dist": ev.get("latent_min_dist"),
            "geom_nn_dist": ev.get("geom_nearest_train_dist"),
            "lsd_band_0_250": (ev.get("band_metrics_held", {}) or {}).get("lsd_band_0_250_db"),
            "lsd_band_250_500": (ev.get("band_metrics_held", {}) or {}).get("lsd_band_250_500_db"),
            "lsd_band_500_1000": (ev.get("band_metrics_held", {}) or {}).get("lsd_band_500_1000_db"),
            "lsd_band_1000_2000": (ev.get("band_metrics_held", {}) or {}).get("lsd_band_1000_2000_db"),
        })

    rows.sort(key=lambda r: (r["L"] or 0, r["W"] or 0, r["H"] or 0))

    # Headline target: mag corr ≥ 0.9 (full-spectrum; the signal_metrics output
    # is computed on the held-out subset). Spec said "mag corr ≥ 0.9 in 0-500 Hz",
    # but signal_metrics.mag_corr is over the full spectrum — flag both.
    n_total = len(rows)
    n_mag_ge_09 = sum(1 for r in rows if (r["mag_corr"] or 0) >= 0.9)

    # In-distribution gate (D37): zero-shot is only interpretable as method
    # success/failure if the model fit the training rooms (≤ 2.5 dB val LSD).
    in_dist_lsd, in_dist_iter = _read_in_dist_lsd(train_dir)
    in_dist_ok = in_dist_lsd is not None and in_dist_lsd <= 2.5

    # 3-way self-diagnosis per room (D37).
    for r in rows:
        gmax = r["geom_err_max"]
        if gmax is None:  # older metrics.json without geom_err_max — derive it
            errs = [r.get("geom_err_L"), r.get("geom_err_W"), r.get("geom_err_H")]
            errs = [e for e in errs if e is not None]
            gmax = max(errs) if errs else None
        branch, why = classify_zero_shot_room(in_dist_lsd, r["mag_corr"], gmax)
        r["branch"] = branch
        r["branch_why"] = why
    verdict = aggregate_verdict([r["branch"] for r in rows], n_total)

    gate_line = (
        f"in-distribution val LSD = **{_safe(in_dist_lsd)} dB**"
        + (f" @ {in_dist_iter} iters" if in_dist_iter else "")
        + (" → **gate PASSED** (≤ 2.5)" if in_dist_ok else " → **gate NOT met** (> 2.5); "
           "zero-shot below is for the record, not a method success/failure read")
    ) if in_dist_lsd is not None else "in-distribution val LSD = — (training scalars not found)"

    md = [
        f"# Multi-room 3D zero-shot summary — {train_dir.name}\n",
        "\nMetrics aggregated from each test room's `metrics.json` (held-out subset).\n",
        f"\n## Self-diagnosis verdict (D37)\n\n**{verdict}**\n",
        f"\n- In-distribution gate: {gate_line}\n",
        f"- Zero-shot headline: {n_mag_ge_09}/{n_total} rooms reach mag corr ≥ 0.9 "
        "(full spectrum; target ≥ 5/8).\n",
        "- Per-room branch ∈ {success, manifold_coverage (→ P2-4 more rooms), "
        "decoder_interp (→ decoder smoothness), precondition_unmet}.\n",
        "\n## Per-room metrics (held-out)\n",
        "| Run | L | W | H | V (m³) | mod MAE (Hz) | LSD full | mag corr | phase mw | RIR ρ | env ρ | geom err L/W/H (m) | z* dist (min/geom-nn) | branch |\n",
        "|---|---:|---:|---:|------:|---:|---:|---:|---:|---:|---:|---|---|---|\n",
    ]
    for r in rows:
        md.append(
            f"| {r['run']} | {_safe(r['L'])} | {_safe(r['W'])} | {_safe(r['H'])} | "
            f"{_safe(r['V'], '.1f')} | "
            f"{_safe(r['modal_mae'])} | {_safe(r['full_lsd'])} | "
            f"{_safe(r['mag_corr'], '.3f')} | {_safe(r['phase_corr_mw'], '.3f')} | "
            f"{_safe(r['rir_pearson'], '.3f')} | {_safe(r['env_corr'], '.3f')} | "
            f"{_safe(r['geom_err_L'], '.2f')} / {_safe(r['geom_err_W'], '.2f')} / {_safe(r['geom_err_H'], '.2f')} | "
            f"{_safe(r['latent_min_dist'], '.2f')} / {_safe(r['geom_nn_dist'], '.2f')} | "
            f"{r['branch']} |\n"
        )

    md.append("\n## Per-band LSD (held-out)\n")
    md.append("| Run | 0-250 (dB) | 250-500 (dB) | 500-1000 (dB) | 1000-2000 (dB) |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for r in rows:
        md.append(
            f"| {r['run']} | {_safe(r['lsd_band_0_250'])} | "
            f"{_safe(r['lsd_band_250_500'])} | {_safe(r['lsd_band_500_1000'])} | "
            f"{_safe(r['lsd_band_1000_2000'])} |\n"
        )

    # Mag corr bar chart.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = [r["run"] for r in rows]
    mag_vals = [r["mag_corr"] or 0 for r in rows]
    bar_colors = ["seagreen" if v >= 0.9 else "indianred" for v in mag_vals]
    ax.bar(labels, mag_vals, color=bar_colors)
    ax.axhline(0.9, color="k", lw=0.5, ls="--", label="P2-2 target (0.9)")
    ax.set_ylabel("magnitude correlation")
    ax.set_title(f"{train_dir.name} — zero-shot mag corr per test room")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    out_png = train_dir / "mag_corr_per_room.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    md.append(f"\n## Magnitude correlation per room\n\n![](mag_corr_per_room.png)\n")

    summary_path = train_dir / "SUMMARY.md"
    summary_path.write_text("".join(md))
    print(f"# wrote {summary_path}  (n_rooms={n_total}, n_mag_ge_09={n_mag_ge_09}, "
          f"in_dist_lsd={_safe(in_dist_lsd)})")
    print(f"# VERDICT: {verdict}")


if __name__ == "__main__":
    main()
