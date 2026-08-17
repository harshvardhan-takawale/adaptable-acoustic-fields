"""P3-3-FAST Track 2b demo figure: predicted vs ground-truth spectra across the aperture axis.

Four apertures on one frozen test domain, both sub-rooms:

    a = 0.0   SEALED -- a TOPOLOGICAL reference, not the small-aperture limit. The divider
              disconnects room B exactly, so the ground-truth room-B spectrum is identically
              zero and cannot be drawn on a dB axis at all. The panel says so instead of
              plotting a floor and letting the reader mistake it for a measurement.
    a = 0.3   trained
    a = 1.0   **HELD OUT** -- no training config has a in [0.9, 1.1]. This is the panel the
              chunk is about.
    a = 2.0   trained

Each panel shows the receiver-averaged magnitude ``20 log10(<|H|>)`` over 20-300 Hz for one
sub-room, ground truth against prediction, with that config's inter-room level difference
printed in the corner so the coupling number and the spectrum it came from sit together.

Needs a checkpoint (it renders). Output is 1920x1200 at dpi 120.

Usage
-----
    python scripts/p3_3fast_trackB_demo_fig.py \\
        --checkpoint outputs/p3_3fast/p3_3fast_trackB/ckpt_iter0030000.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from aaf.data.aperture_configs import configs_from_rows, in_holdout   # noqa: E402
from scripts.p3_3fast_ftb import BAND_HI, BAND_LO                     # noqa: E402
from scripts.p3_3fast_trackB_eval import (                            # noqa: E402
    DATA_DIR,
    DF_HZ,
    EPS,
    MANIFEST,
    N_BINS_BAND,
    OUT_DIR,
    TRAIN_DIR,
    level_difference,
    render_aperture,
    room_masks,
)

#: The four demo apertures. All four exist in the test split of every frozen test domain.
DEMO_A = (0.0, 0.3, 1.0, 2.0)
FIG_W_IN, FIG_H_IN, DPI = 16.0, 10.0, 120        # -> 1920 x 1200
GT_COLOR, PRED_COLOR = "#1f3864", "#c0392b"


def mean_db(H: np.ndarray, sel: np.ndarray) -> Optional[np.ndarray]:
    """Receiver-averaged magnitude in dB, or None when the sub-room field is exactly zero.

    Returning None rather than a clamped floor is the point: a sealed divider makes room B
    identically zero, and -inf is a topological fact that must not be drawn as a low curve.
    """
    if not np.any(sel):
        return None
    m = np.mean(np.abs(H[sel]), axis=0)
    if float(np.max(m)) <= EPS:
        return None
    return 20.0 * np.log10(np.maximum(m, EPS))


def main() -> int:
    ap = argparse.ArgumentParser(description="P3-3-FAST Track 2b demo figure")
    ap.add_argument("--train-dir", default=TRAIN_DIR)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--checkpoint", default=None,
                    help="default: newest ckpt_iter*.pt in --train-dir")
    ap.add_argument("--geom-id", type=int, default=2,
                    help="frozen test domain index; 2 is 8.00 x 3.85 with x0 = 3.60, the "
                         "closest of the six to FT-B's 8.0 x 4.0 sweep domain")
    ap.add_argument("--rx-chunk", type=int, default=8)
    # --gt-only / --limit may arrive from the shared sbatch wrapper; they mean nothing here.
    args, extra = ap.parse_known_args()
    if "--gt-only" in extra:
        print("[skip] demo figure renders and cannot run in --gt-only mode")
        return 0

    import torch

    from aaf.eval.p3_2_eval import band_limit, find_checkpoint, load_gt, load_model

    man = json.loads(Path(args.manifest).read_text())
    rows = man["configs"] if isinstance(man, dict) else man
    cfgs = [c for c in configs_from_rows(rows, split="test") if c.geom_id == args.geom_id]
    if not cfgs:
        raise SystemExit("no test configs for geom_id {}".format(args.geom_id))
    picked = []
    for a in DEMO_A:
        match = [c for c in cfgs if abs(c.a - a) < 1e-9]
        if not match:
            raise SystemExit("aperture a = {} is not in the test split of geom {}".format(
                a, args.geom_id))
        picked.append(match[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint) if args.checkpoint else find_checkpoint(args.train_dir)
    model, renderer, cfg, _meta, it = load_model(ckpt, device)
    cond_source = str(cfg["cond_source"])
    if cond_source != "aperture":
        raise SystemExit("expected cond_source 'aperture', got {!r}".format(cond_source))

    freqs = np.arange(N_BINS_BAND) * DF_HZ
    band = (freqs >= BAND_LO) & (freqs <= BAND_HI)

    fig, axes = plt.subplots(len(picked), 2, figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI,
                             sharex=True)
    for r, c in enumerate(picked):
        H_raw, rx, src, _ = load_gt(Path(args.data_dir) / c.filename)
        H_gt = band_limit(H_raw, N_BINS_BAND)[:, :N_BINS_BAND]
        H_pr = band_limit(
            render_aperture(model, renderer, cond_source, c.L, c.W, c.x0, c.a, c.alphas,
                            rx, src, device, args.rx_chunk), N_BINS_BAND)[:, :N_BINS_BAND]
        sel_a, sel_b, _ = room_masks(rx, c.x0)
        ld_gt = level_difference(H_gt, freqs, sel_a, sel_b)["ld_broadband_db"]
        ld_pr = level_difference(H_pr, freqs, sel_a, sel_b)["ld_broadband_db"]

        for k, (room, sel) in enumerate((("room A (source side)", sel_a),
                                         ("room B (across the divider)", sel_b))):
            ax = axes[r, k]
            g, p = mean_db(H_gt, sel), mean_db(H_pr, sel)
            if g is None:
                ax.set_facecolor("#f2f2f2")
                ax.text(0.5, 0.97, "ground truth is IDENTICALLY ZERO -- room B is "
                                   "disconnected (topological, not small)",
                        transform=ax.transAxes, ha="center", va="top", fontsize=10,
                        color="#333333",
                        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999"))
            else:
                ax.plot(freqs[band], g[band], color=GT_COLOR, lw=1.4, label="ground truth")
            if p is not None:
                ax.plot(freqs[band], p[band], color=PRED_COLOR, lw=1.2, ls="--",
                        alpha=0.9, label="predicted")
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_xlim(BAND_LO, BAND_HI)
            if r == 0:
                ax.set_title(room, fontsize=12)
            if k == 0:
                tag = "SEALED (a = 0)\ntopological reference" if c.sealed else \
                    ("a = {:.2f} m  **HELD OUT**".format(c.a) if in_holdout(c.a)
                     else "a = {:.2f} m  (trained)".format(c.a))
                ax.set_ylabel("{}\nmean |H| (dB)".format(tag), fontsize=10)
            if r == len(picked) - 1:
                ax.set_xlabel("frequency (Hz)", fontsize=11)
            if r == 0 and k == 0:
                ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
        axes[r, 1].text(
            0.985, 0.06,
            "level difference  GT {}  |  pred {:+.2f} dB".format(
                "-inf" if not np.isfinite(ld_gt) else "{:+.2f}".format(ld_gt),
                ld_pr if np.isfinite(ld_pr) else float("nan")),
            transform=axes[r, 1].transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9))

    fig.suptitle(
        "P3-3-FAST Track 2b: doorway aperture, predicted vs ground truth\n"
        "domain L = {:.2f} m, W = {:.2f} m, divider x0 = {:.2f} m  |  checkpoint iter {}  |  "
        "training excludes a in [0.9, 1.1] exactly".format(
            picked[0].L, picked[0].W, picked[0].x0, it),
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "demo_aperture_spectra.png"
    fig.savefig(str(path), dpi=DPI)
    plt.close(fig)
    print("[wrote] {} ({:.0f}x{:.0f})".format(path, FIG_W_IN * DPI, FIG_H_IN * DPI))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
