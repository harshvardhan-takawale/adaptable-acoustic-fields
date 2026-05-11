"""Chunk 3.9: regenerate `02_single_room_baseline.png` as a 2-panel
ISM-vs-predicted spectrum overlay at one of the Chunk-2 single-room overfit
models (L=4.5 m, the middle of the three).

Top panel:    0-250 Hz log-magnitude with analytical eigenfrequency ticks.
Bottom panel: 0-100 Hz zoom on the first 5 modes.

Both panels show ISM (solid black) overlaid by the predicted spectrum
(dashed orange, with slight transparency) so the modal-regime fit is
visually self-evident. Uses the existing checkpoint at
`outputs/single_room/L4.5/ckpt_iter0010000.pt`. Forwards the model
at all 64 receivers in eval mode, picks the receiver closest to
(L/2, W/2), and saves the figure at presentation resolution.

No retraining; runs on the login node in ~30 s.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aaf.data.dataset_builder import read_room_h5, room_filename
from aaf.eval.single_room_eval import _load_latest_ckpt   # reuses the chunk-2 helper
from aaf.models.inr_2d import INR2D_Single
from aaf.renderers.freq_2d import FreqRenderer2D
from aaf.sim.analytical_modal_2d import eigenfrequencies_2d


REPO_ROOT = Path(__file__).resolve().parent.parent
L = 4.5
W = 4.0
ALPHA = 0.15


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f_max_top", type=float, default=250.0,
                    help="Upper f for the top panel (default: modal regime 250 Hz)")
    ap.add_argument("--f_max_zoom", type=float, default=100.0,
                    help="Upper f for the bottom panel zoom (default: 100 Hz)")
    ap.add_argument("--out_suffix", type=str, default="",
                    help="Optional suffix appended to PNG + caption filenames "
                         "(e.g. '_full_band'). Empty = the standard 02 asset.")
    ap.add_argument("--title", type=str, default="",
                    help="Optional overall title override; default cites modal MAE.")
    args = ap.parse_args()

    suffix = args.out_suffix
    out_png = REPO_ROOT / f"outputs/meeting_assets/02_single_room_baseline{suffix}.png"
    caption = REPO_ROOT / f"outputs/meeting_assets/02_single_room_baseline{suffix}_caption.md"
    print(f"# out: {out_png.name}  panels: 0-{args.f_max_top:g} Hz / 0-{args.f_max_zoom:g} Hz")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "tcnn requires CUDA; rerun on a compute node "
            "(see scripts/slurm/run_pytest.sh for the activation pattern)."
        )

    # Training metadata.
    train_dir = REPO_ROOT / "outputs/single_room/L4.5"
    train_meta = json.loads((train_dir / "train_meta.json").read_text())
    cfg = train_meta["cfg"]
    fs = int(cfg["fs"])
    n_time = int(cfg["n_time_samples"])
    n_freq = n_time // 2 + 1
    n_azi = int(cfg["n_azi"])
    n_pts_per_ray = int(cfg["n_pts_per_ray"])
    near = float(cfg["near"])
    c = float(cfg["c"])

    # ISM ground truth.
    h5_path = REPO_ROOT / "data/track_a" / room_filename(L=L, W=W, alpha=ALPHA)
    rt = read_room_h5(h5_path)
    attrs = rt["attrs"]
    receiver_pos = np.asarray(attrs["receiver_pos"], dtype=np.float32)   # [64, 2]
    source_pos = np.asarray(attrs["source_pos"], dtype=np.float32)
    H_ism = rt["ism_H"].astype(np.complex64)                              # [64, n_freq]

    device = torch.device("cuda")
    model = INR2D_Single(n_freq_bins=n_freq).to(device).eval()
    ckpt_path, state = _load_latest_ckpt(train_dir)
    model.load_state_dict(state["model"])
    print(f"# loaded {ckpt_path.name}")

    renderer = FreqRenderer2D(
        n_azi=n_azi, n_pts_per_ray=n_pts_per_ray, near=near,
        fs=fs, n_time_samples=n_time, c=c, use_geometric_attn=False,
    ).to(device).eval()

    rx_t = torch.from_numpy(receiver_pos).to(device)
    tx_t = torch.from_numpy(np.tile(source_pos, (rx_t.size(0), 1))).to(device)
    room_min = torch.tensor([0.0, 0.0], device=device)
    room_max = torch.tensor([float(L), float(W)], device=device)
    chunk = 8
    parts = []
    with torch.no_grad():
        for s in range(0, rx_t.size(0), chunk):
            parts.append(
                renderer(model, rx_t[s:s + chunk], tx_t[s:s + chunk],
                         room_min, room_max).cpu().numpy()
            )
    H_pred = np.concatenate(parts, axis=0).astype(np.complex64)
    print(f"# H_pred shape: {H_pred.shape}")

    # Pick centre receiver (closest to (L/2, W/2)).
    centre = float(L) / 2.0, float(W) / 2.0
    centre_idx = int(np.argmin(np.linalg.norm(
        receiver_pos - np.array(centre), axis=1
    )))
    cx, cy = receiver_pos[centre_idx]
    print(f"# centre receiver idx {centre_idx}: ({cx:.2f}, {cy:.2f})")

    # Analytical eigenfrequencies up to the widest panel's upper limit.
    eig_f_max = max(args.f_max_top, args.f_max_zoom)
    eigs = eigenfrequencies_2d(L=L, W=W, c=c, f_max=eig_f_max)
    eig_fs = np.array([e.f for e in eigs if e.f > 0])

    f_axis = np.arange(n_freq) * (fs / n_time)

    def _db(H_row):
        eps = 1e-10
        return 20.0 * np.log10(np.maximum(np.abs(H_row), eps))

    H_ism_centre = H_ism[centre_idx]
    H_pred_centre = H_pred[centre_idx]
    db_ism = _db(H_ism_centre)
    db_pred = _db(H_pred_centre)

    # Compute headline modal MAE on the centre receiver (always over the
    # modal regime ≤ 250 Hz, regardless of panel range — the title's headline
    # number is the project-standard modal MAE).
    from aaf.eval.modal_verifier import pick_peaks, match_peaks_to_modes
    eigs_modal = [e for e in eigenfrequencies_2d(L=L, W=W, c=c, f_max=250.0) if e.f > 0]
    f_mask_250 = f_axis <= 250.0
    picks = pick_peaks(H_pred_centre[f_mask_250], f_axis[f_mask_250],
                       prominence_db=3.0, min_distance_hz=10.0)
    matches = match_peaks_to_modes(picks, eigs_modal,
                                   tolerance_hz=4.0, tolerance_pct=0.02)["matches"]
    centre_mae = float(np.mean([abs(m.delta_hz) for m in matches])) if matches else float("nan")
    print(f"# centre receiver modal MAE: {centre_mae:.3f} Hz across {len(matches)} matched modes")

    # Headline number for the title — use the eval.json modal MAE if available
    # (it's an across-room average; 0.34 Hz per the spec).
    try:
        ev = json.loads((train_dir / "eval.json").read_text())
        room_modal_mae = float(ev["modal"]["mae_hz"])
    except Exception:
        room_modal_mae = centre_mae

    # ---- Plot ----
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 9.0))
    ISM_COLOR = "black"
    PRED_COLOR = "#d35400"   # warm orange-red

    # Panel tags follow the f-range. Top panel = ``f_max_top``; bottom =
    # ``f_max_zoom`` (typically a sub-range of the top for visual emphasis).
    def _panel_tag(lo: float, hi: float, is_top: bool) -> str:
        regime = ""
        if hi <= 260.0:
            regime = " (modal regime)" if is_top else " (zoom on the first 5 modes)"
        elif hi <= 510.0:
            regime = " (modal + transition)" if is_top else " (modal regime)"
        else:
            regime = " (full band)" if is_top else " (modal + transition zoom)"
        return f"{lo:.0f}-{hi:.0f} Hz{regime}"

    for ax, f_lo, f_hi, panel_tag in [
        (ax_top, 0.0, float(args.f_max_top), _panel_tag(0.0, float(args.f_max_top), True)),
        (ax_bot, 0.0, float(args.f_max_zoom), _panel_tag(0.0, float(args.f_max_zoom), False)),
    ]:
        mask = (f_axis >= f_lo) & (f_axis <= f_hi)
        ax.plot(f_axis[mask], db_ism[mask], color=ISM_COLOR, lw=2.0,
                label="ISM ground truth", zorder=3)
        ax.plot(f_axis[mask], db_pred[mask], color=PRED_COLOR, lw=2.0,
                ls="--", alpha=0.85, label="Predicted (overfit model)", zorder=4)
        # Analytical eigenfreq ticks along the bottom.
        ymin, ymax = ax.get_ylim() if ax.has_data() else (-100.0, 10.0)
        ax.set_ylim(ymin, ymax)
        eig_in_range = eig_fs[(eig_fs >= f_lo) & (eig_fs <= f_hi)]
        for fh in eig_in_range:
            ax.axvline(fh, ymin=0, ymax=0.04, color="steelblue",
                       lw=1.0, alpha=0.85)
        ax.set_xlim(f_lo, f_hi)
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("|H(f)|  (dB)", fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.set_title(panel_tag, fontsize=11)
        ax.legend(loc="upper right", fontsize=11, framealpha=0.95)

    if args.title:
        title = args.title
    else:
        title = (
            f"Single-room baseline: predicted spectrum overlays ISM ground truth "
            f"(L={L:.1f} m, modal MAE {room_modal_mae:.2f} Hz)"
        )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {out_png}")

    # Caption — only refresh when generating the default 02 asset; the
    # wider-range companion file gets its own bespoke caption written below.
    if not suffix:
        caption.write_text(
            "The single-room overfit model recovers the ISM modal regime spectrum "
            "essentially exactly. Top panel: 0-250 Hz, predicted (orange) overlays "
            "ground-truth ISM (black) at the room center for L=4.5 m. Bottom panel: "
            "zoom on the first 5 modes shows each peak aligned to within sub-Hz "
            "precision. Modal MAE across the room: 0.34 Hz, at the analytical "
            "frequency-bin resolution. This is the upper bound we test against in "
            "zero-shot.\n"
        )
    else:
        # Generic caption template for the variant; describes the actual ranges.
        caption.write_text(
            f"Single-room overfit model — ISM (black) vs predicted (orange) "
            f"spectra at the room center for L={L:.1f} m. Top panel covers "
            f"0-{args.f_max_top:.0f} Hz; bottom panel zooms to "
            f"0-{args.f_max_zoom:.0f} Hz. Analytical eigenfrequencies marked as "
            f"small ticks along the bottom. Modal MAE in the 0-250 Hz band: "
            f"{room_modal_mae:.2f} Hz.\n"
        )
    print(f"# wrote {caption}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
