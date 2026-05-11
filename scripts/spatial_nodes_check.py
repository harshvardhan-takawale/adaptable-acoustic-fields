"""V0 / V1 critical-path: spatial node alignment check (Chunk 3.7).

Loads ``H_pred_all.pt`` (the model's complex predictions on all 64 receivers)
and the ISM ground-truth from the same room HDF5, picks the first 6 distinct
analytical eigenfrequencies below 150 Hz, and quantifies whether the predicted
spatial pressure field matches the ISM pressure field on the receiver grid.

Verdict (written as the first line of ``nodes_check_report.md``):

  GREEN   ≥ 4 of 6 modes show spatial Pearson correlation ≥ 0.7 → exit 0
  YELLOW  2 or 3 modes ≥ 0.7                                     → exit 0
  RED     < 2 modes ≥ 0.7                                        → exit 1

The exit code is consumed by the orchestrator's ``afterok`` dependency to gate
V1-V4 (presentation assembly).
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
import torch

from aaf.data.dataset_builder import read_room_h5, room_filename
from aaf.eval.spatial_modes import (
    analytical_mode_shape,
    extract_pressure_field,
    mode_shape_fit_error,
    node_match_score,
    pick_first_modes,
    receiver_grid_xy,
    spatial_correlation_complex,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_hpred(run_dir: Path, L: float, inner_loop: str) -> np.ndarray:
    """Return [64, n_freq_bins] complex64 H_pred from the saved tensor."""
    if inner_loop == "B1":
        cands = [run_dir / "zero_shot" / f"L{L}" / "H_pred_all.pt"]
        # For Chunk-3.6's Track-C runs the B1 baseline lives under
        # outputs/inner_loop_experiments/B1/<run>/L<L>/.
        inner_alt = (
            REPO_ROOT / "outputs/inner_loop_experiments/B1"
            / run_dir.name / f"L{L}" / "H_pred_all.pt"
        )
        cands.append(inner_alt)
    else:
        cands = [run_dir / f"zero_shot_{inner_loop}" / f"L{L}" / "H_pred_all.pt"]
    for p in cands:
        if p.exists():
            t = torch.load(p, map_location="cpu")
            return t.detach().cpu().numpy().astype(np.complex64)
    raise FileNotFoundError(
        f"no H_pred_all.pt at any of: " + ", ".join(str(c) for c in cands)
    )


def _load_ism(data_dir: Path, L: float, W: float, alpha: float) -> tuple[np.ndarray, dict]:
    """Return (ism_H [64, n_freq] complex, attrs dict).

    Uses ``aaf.data.dataset_builder.read_room_h5`` for compatibility with the
    nested HDF5 layout (the file stores the complex spectrum at
    ``ism/H_complex``, not as a top-level dataset).
    """
    h5_path = data_dir / room_filename(L=L, W=W, alpha=alpha)
    rt = read_room_h5(h5_path)
    return rt["ism_H"].astype(np.complex64), rt["attrs"]


def _train_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "train_meta.json").read_text())


def _plot_mode_triptych(P_pred, P_ism, n_x, n_y, f_hz, corr, L, W, out_path: Path):
    """3-panel: ISM |P|, predicted |P|, ISM phase."""
    X, Y = receiver_grid_xy(L=L, W=W)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    vmin = -60
    vmax = 0

    def db(x):
        return 20.0 * np.log10(np.maximum(np.abs(x), 1e-10) / (np.abs(x).max() + 1e-10))

    im0 = axs[0].imshow(db(P_ism), origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
                        cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axs[0].set_title(f"ISM |P| (normalised, dB)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(db(P_pred), origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
                        cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axs[1].set_title(f"Predicted |P| (normalised, dB)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(np.angle(P_ism), origin="lower",
                        extent=[X.min(), X.max(), Y.min(), Y.max()],
                        cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal")
    axs[2].set_title("ISM phase")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.suptitle(
        f"L={L:.2f} m  mode ({n_x},{n_y})  f={f_hz:.1f} Hz  |  spatial corr = {corr:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(modes_data, L, W, out_path: Path):
    """Single 2-row × n_modes figure: top = ISM |P|, bottom = predicted |P|."""
    n = len(modes_data)
    fig, axs = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    if n == 1:
        axs = axs[:, None]
    for i, m in enumerate(modes_data):
        for row, (label, P) in enumerate([("ISM", m["P_ism"]), ("Pred", m["P_pred"])]):
            ax = axs[row, i]
            db = 20 * np.log10(np.maximum(np.abs(P), 1e-10) / (np.abs(P).max() + 1e-10))
            im = ax.imshow(db, origin="lower", cmap="viridis", vmin=-60, vmax=0,
                           aspect="equal")
            if row == 0:
                ax.set_title(f"({m['n_x']},{m['n_y']})  {m['f']:.1f} Hz",
                             fontsize=9)
            if i == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        # corr annotation in the bottom row
        axs[1, i].text(0.5, -0.15, f"corr={m['corr']:.2f}",
                       transform=axs[1, i].transAxes, ha="center", fontsize=8,
                       color="green" if m["corr"] >= 0.7 else "darkred")
    fig.suptitle(f"L={L:.2f} m — first {n} modes (top = ISM, bottom = predicted)",
                 fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="C2_latent_jitter")
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--inner_loop", type=str, default="B6",
                    choices=["B1", "B6"])
    ap.add_argument("--sweep_root", type=str,
                    default=str(REPO_ROOT / "outputs/multi_room/sweep"))
    ap.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "data/track_a"))
    ap.add_argument("--out_root", type=str,
                    default=str(REPO_ROOT / "outputs/spatial_nodes_check"))
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--n_modes", type=int, default=6)
    ap.add_argument("--f_max", type=float, default=150.0,
                    help="upper bound on modal frequency to include")
    ap.add_argument("--corr_threshold", type=float, default=0.7,
                    help="per-mode correlation threshold for GREEN/YELLOW counting")
    args = ap.parse_args()

    L = float(args.L)
    run_dir = Path(args.sweep_root) / args.run
    out_dir = Path(args.out_root) / f"L{L}"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_meta = _train_meta(run_dir)
    cfg = train_meta["cfg"]
    fs = float(cfg["fs"])
    n_time = int(cfg["n_time_samples"])
    n_freq = n_time // 2 + 1

    # Load predictions + ISM.
    H_pred = _load_hpred(run_dir, L, args.inner_loop)
    H_ism, attrs = _load_ism(Path(args.data_dir), L=L, W=4.0, alpha=args.alpha)
    if H_pred.shape != H_ism.shape:
        raise RuntimeError(
            f"shape mismatch: H_pred {H_pred.shape} vs H_ism {H_ism.shape}"
        )
    W_room = float(attrs.get("W", 4.0))

    # Pick first 6 distinct eigenfrequencies below f_max.
    modes = pick_first_modes(L=L, W=W_room, n_modes=args.n_modes,
                             f_min=1.0, f_max=args.f_max)
    if not modes:
        print(f"[spatial-nodes] no modes found below {args.f_max} Hz at L={L}")
        sys.exit(1)

    # Per-mode analysis.
    per_mode = []
    for n_x, n_y, f_hz in modes:
        P_pred = extract_pressure_field(H_pred, f_hz, fs, n_freq)
        P_ism = extract_pressure_field(H_ism, f_hz, fs, n_freq)
        corr = spatial_correlation_complex(P_pred, P_ism)
        node_score = node_match_score(P_pred, P_ism, threshold_db=-20.0)
        fit_pred = mode_shape_fit_error(P_pred, n_x, n_y, L=L, W=W_room)
        fit_ism = mode_shape_fit_error(P_ism, n_x, n_y, L=L, W=W_room)
        per_mode.append({
            "n_x": n_x, "n_y": n_y, "f": f_hz,
            "corr": corr,
            "node_match": node_score,
            "pred_shape_snr_db": fit_pred["snr_db"],
            "ism_shape_snr_db": fit_ism["snr_db"],
            "P_pred": P_pred, "P_ism": P_ism,
        })
        # Per-mode triptych.
        _plot_mode_triptych(P_pred, P_ism, n_x, n_y, f_hz, corr, L=L, W=W_room,
                            out_path=fig_dir / f"mode_{n_x}{n_y}.png")

    # Overview figure (no P_pred / P_ism in the saved record below).
    _plot_overview(per_mode, L=L, W=W_room,
                   out_path=fig_dir / "all_modes_overview.png")

    # Verdict.
    n_good = sum(1 for m in per_mode if m["corr"] >= args.corr_threshold)
    if n_good >= 4:
        verdict = "GREEN"
        exit_code = 0
    elif n_good >= 2:
        verdict = "YELLOW"
        exit_code = 0
    else:
        verdict = "RED"
        exit_code = 1

    # Report.
    lines = []
    lines.append(
        f"## Verdict: {verdict} — {n_good} of {len(per_mode)} modes have "
        f"spatial correlation ≥ {args.corr_threshold}"
    )
    lines.append("")
    lines.append(f"- run: `{args.run}` (inner loop: `{args.inner_loop}`)")
    lines.append(f"- L = {L:.2f} m, W = {W_room:.2f} m, fs = {int(fs)} Hz, "
                 f"n_freq_bins = {n_freq}")
    lines.append(f"- modes inspected: first {len(per_mode)} distinct eigenfrequencies "
                 f"in (1, {args.f_max:.0f}] Hz")
    lines.append("")
    lines.append("| (n_x, n_y) | f (Hz) | spatial corr | node match | "
                 "pred shape SNR (dB) | ISM shape SNR (dB) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in per_mode:
        nm = (f"{m['node_match']:.2f}" if not (isinstance(m['node_match'], float)
              and (m['node_match'] != m['node_match'])) else "—")
        lines.append(
            f"| ({m['n_x']},{m['n_y']}) | {m['f']:.1f} | {m['corr']:.3f} | "
            f"{nm} | {m['pred_shape_snr_db']:.1f} | {m['ism_shape_snr_db']:.1f} |"
        )
    lines.append("")
    lines.append("Figures:")
    for m in per_mode:
        lines.append(f"  - `figures/mode_{m['n_x']}{m['n_y']}.png`")
    lines.append(f"  - `figures/all_modes_overview.png`")
    (out_dir / "nodes_check_report.md").write_text("\n".join(lines))

    # Also save the raw per-mode JSON (without complex arrays, for downstream tools).
    json_payload = {
        "verdict": verdict,
        "n_good": n_good,
        "n_modes": len(per_mode),
        "corr_threshold": float(args.corr_threshold),
        "L": L, "W": W_room, "fs": int(fs), "n_freq_bins": n_freq,
        "run": args.run, "inner_loop": args.inner_loop,
        "modes": [
            {k: v for k, v in m.items() if k not in ("P_pred", "P_ism")}
            for m in per_mode
        ],
    }
    (out_dir / "nodes_check.json").write_text(json.dumps(json_payload, indent=2,
                                                          default=float))
    print(f"# VERDICT: {verdict} ({n_good}/{len(per_mode)} modes ≥ "
          f"{args.corr_threshold})")
    print(f"# wrote {out_dir/'nodes_check_report.md'}")
    print(f"# wrote {out_dir/'nodes_check.json'}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
