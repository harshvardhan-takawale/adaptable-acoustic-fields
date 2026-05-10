"""Visual sanity pack for the 2D shoebox dataset.

Per room (one PDF per L) — `outputs/visual_sanity/per_room/L_*.pdf`:
  Page 1: 4-panel IR view at the room-centre receiver
          (linear RIR, log-magnitude RIR + EDC, full spectrum, 0–200 Hz zoom).
  Page 2: 8×8 grid of receiver IR sparklines.
  Page 3: spectrum overlay with deduplicated analytical eigenfrequency ticks +
          picked peaks.

Cross-room — `outputs/visual_sanity/cross_room.pdf`:
  Page 1: scatter of (analytical f, picked-ISM f) across all rooms, colored by L.
  Page 2: modal-frequency vs L for the lowest few mode families with analytical curves.
  Page 3: T60 vs L (Sabine theoretical and EDC empirical), with target curve.

Plus: `INDEX.md` listing everything, and `SANITY_NOTES.md` reserved for the agent
to fill in after eyeballing the figures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aaf.data.dataset_builder import read_room_h5
from aaf.eval.modal_verifier import (
    pick_peaks,
    plot_modal_overlay,
)
from aaf.sim.analytical_modal_2d import eigenfrequencies_2d


C_DEFAULT = 343.0


def representative_receiver(receiver_pos: np.ndarray, L: float, W: float) -> int:
    centre = np.array([L / 2.0, W / 2.0])
    return int(np.argmin(np.linalg.norm(receiver_pos - centre[None, :], axis=1)))


def edc_t60(rir: np.ndarray, fs: float) -> tuple[float, np.ndarray]:
    """Schroeder-integrated T60 estimate from a single-channel RIR.

    Returns (T60_seconds, edc_db). T60 is computed by linear regression on the
    -5 dB to -35 dB segment, scaled to 60 dB. NaN if the segment is too short.
    """
    rir = np.asarray(rir, dtype=np.float64)
    edc = np.cumsum(rir[::-1] ** 2)[::-1]
    edc /= max(edc.max(), 1e-30)
    edc_db = 10.0 * np.log10(np.maximum(edc, 1e-12))

    lo, hi = -5.0, -35.0
    mask = (edc_db <= lo) & (edc_db >= hi)
    if mask.sum() < 8:
        return float("nan"), edc_db

    t = np.arange(rir.size) / fs
    t_seg = t[mask]
    db_seg = edc_db[mask]
    slope, intercept = np.polyfit(t_seg, db_seg, 1)
    if slope >= 0:
        return float("nan"), edc_db
    t60 = -60.0 / slope
    return float(t60), edc_db


def per_room_pdf(h5_path: Path, out_pdf: Path):
    rt = read_room_h5(h5_path)
    attrs = rt["attrs"]
    L = float(attrs["L"])
    W = float(attrs["W"])
    fs = float(attrs["fs"])
    n_time = int(attrs["n_time_samples"])
    n_freq = int(attrs["n_freq_bins"])
    receiver_pos = np.asarray(attrs["receiver_pos"], dtype=np.float64)

    f_axis = np.arange(n_freq) * (fs / n_time)
    t_axis = np.arange(n_time) / fs

    rep = representative_receiver(receiver_pos, L, W)
    rir = rt["ism_rir"][rep]
    H = rt["ism_H"][rep]
    rir_db = 20.0 * np.log10(np.maximum(np.abs(rir), 1e-7))
    H_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-7))

    t60_edc, edc_db = edc_t60(rir, fs)
    t60_sabine = float(attrs["T60_sabine_2d"])

    modes = eigenfrequencies_2d(L=L, W=W, c=C_DEFAULT, f_max=2000.0)
    modes_nonzero = [m for m in modes if m.f > 0]
    peaks = pick_peaks(H, f_axis, prominence_db=3.0, min_distance_hz=10.0)

    with PdfPages(out_pdf) as pdf:
        # ----------------------------- Page 1 -------------------------------
        fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
        fig.suptitle(
            f"L={L:.2f} m  W={W:.2f} m  α={float(attrs['alpha']):.2f}  fs={int(fs)} Hz  "
            f"T60_sabine={t60_sabine*1000:.0f} ms  T60_EDC={t60_edc*1000:.0f} ms  "
            f"rep_rx=({receiver_pos[rep, 0]:.2f}, {receiver_pos[rep, 1]:.2f}) m",
            fontsize=10,
        )

        ax = axs[0, 0]
        ax.plot(t_axis, rir, color="steelblue", lw=0.6)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("RIR amplitude")
        ax.set_title("Linear RIR (room-centre receiver)")
        ax.grid(True, alpha=0.3)

        ax = axs[0, 1]
        ax.plot(t_axis, rir_db, color="steelblue", lw=0.6, label="20·log10|h|")
        ax.plot(t_axis, edc_db, color="indianred", lw=1.0, label="EDC (dB)")
        ax.axhline(-5, color="k", lw=0.5, linestyle="--", alpha=0.5)
        ax.axhline(-35, color="k", lw=0.5, linestyle="--", alpha=0.5)
        if not np.isnan(t60_edc):
            ax.axvline(t60_edc, color="indianred", lw=0.7, linestyle=":", label=f"T60_EDC={t60_edc*1000:.0f} ms")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("dB")
        ax.set_ylim(-100, 5)
        ax.set_title("Log-magnitude RIR + Schroeder EDC")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axs[1, 0]
        mask = f_axis <= 2000.0
        ax.plot(f_axis[mask], H_db[mask], color="steelblue", lw=0.6)
        ax.set_xlabel("f (Hz)")
        ax.set_ylabel("|H| (dB)")
        ax.set_xlim(0, 2000)
        ax.set_title("Full spectrum 0–2 kHz")
        ax.grid(True, alpha=0.3)

        ax = axs[1, 1]
        mask200 = f_axis <= 200.0
        ax.plot(f_axis[mask200], H_db[mask200], color="steelblue", lw=0.6)
        for m in modes_nonzero:
            if m.f <= 200.0:
                lw = 0.6 + 0.4 * (m.multiplicity - 1)
                ax.axvline(m.f, color="tab:orange", lw=lw, alpha=0.7)
        ax.set_xlabel("f (Hz)")
        ax.set_ylabel("|H| (dB)")
        ax.set_xlim(0, 200)
        ax.set_title("Modal-regime zoom 0–200 Hz (orange ticks = analytical)")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig, dpi=110, bbox_inches="tight")
        plt.close(fig)

        # ----------------------------- Page 2 -------------------------------
        n_per_side = 8
        n_rx = receiver_pos.shape[0]
        assert n_rx == n_per_side * n_per_side, f"expected 64 rx, got {n_rx}"

        # Receivers are stored row-major: y outer, x inner. Reshape into (Y, X).
        # We laid them out as `for y in ys: for x in xs: ...`, so index = iy*8 + ix.
        fig, axs = plt.subplots(n_per_side, n_per_side, figsize=(11, 8), sharex=True)
        for iy in range(n_per_side):
            for ix in range(n_per_side):
                # Flip y so the first row drawn corresponds to the top of the room.
                idx = (n_per_side - 1 - iy) * n_per_side + ix
                ax = axs[iy, ix]
                ax.plot(rt["ism_rir"][idx], color="steelblue", lw=0.4)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(0, rt["ism_rir"].shape[1])
                ax.set_ylim(-rt["ism_rir"][idx].max() * 1.1, rt["ism_rir"][idx].max() * 1.1)
                if iy == n_per_side - 1:
                    ax.set_xlabel(f"x={receiver_pos[idx, 0]:.1f}", fontsize=6)
                if ix == 0:
                    ax.set_ylabel(f"y={receiver_pos[idx, 1]:.1f}", fontsize=6, rotation=0, labelpad=12)
        fig.suptitle(f"L={L:.2f} m — IR sparklines for the 8×8 receiver grid", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig, dpi=110, bbox_inches="tight")
        plt.close(fig)

        # ----------------------------- Page 3 -------------------------------
        fig, ax = plt.subplots(figsize=(11, 5))
        plot_modal_overlay(
            H, f_axis, modes_nonzero, peaks, ax,
            title=f"L={L:.2f} m — ISM |H(f)| with deduplicated analytical eigenfrequencies and picked peaks",
            f_min=0, f_max=2000, db_floor=-100,
        )
        fig.tight_layout()
        pdf.savefig(fig, dpi=110, bbox_inches="tight")
        plt.close(fig)


def cross_room_pdf(h5_paths: list[Path], out_pdf: Path):
    """Aggregate cross-room visualizations."""
    rooms = []
    for p in h5_paths:
        rt = read_room_h5(p)
        attrs = rt["attrs"]
        L = float(attrs["L"])
        W = float(attrs["W"])
        fs = float(attrs["fs"])
        n_time = int(attrs["n_time_samples"])
        n_freq = int(attrs["n_freq_bins"])
        receiver_pos = np.asarray(attrs["receiver_pos"], dtype=np.float64)
        rep = representative_receiver(receiver_pos, L, W)
        rir = rt["ism_rir"][rep]
        H = rt["ism_H"][rep]
        f_axis = np.arange(n_freq) * (fs / n_time)
        modes = [m for m in eigenfrequencies_2d(L=L, W=W, c=C_DEFAULT, f_max=2000.0) if m.f > 0]
        peaks = pick_peaks(H, f_axis, prominence_db=3.0, min_distance_hz=10.0)
        t60_edc, _ = edc_t60(rir, fs)
        rooms.append({
            "L": L, "W": W, "fs": fs,
            "T60_sabine": float(attrs["T60_sabine_2d"]),
            "T60_edc": t60_edc,
            "modes": modes,
            "peaks": peaks,
            "f_axis": f_axis,
        })
    rooms.sort(key=lambda r: r["L"])
    Ls = [r["L"] for r in rooms]
    cmap = plt.cm.viridis
    L_to_color = {r["L"]: cmap(i / max(len(rooms) - 1, 1)) for i, r in enumerate(rooms)}

    with PdfPages(out_pdf) as pdf:
        # ----------------------------- Page 1 -------------------------------
        fig, ax = plt.subplots(figsize=(8, 8))
        for r in rooms:
            modes_by_f = sorted(r["modes"], key=lambda m: m.f)
            picks_sorted = sorted(r["peaks"], key=lambda p: p.f)
            for p in picks_sorted:
                if p.f > 2000:
                    continue
                # Closest analytical mode within ~2% tolerance.
                best = min(modes_by_f, key=lambda m: abs(m.f - p.f))
                if abs(best.f - p.f) <= max(4.0, 0.02 * best.f):
                    ax.plot(best.f, p.f, "o", color=L_to_color[r["L"]],
                            alpha=0.6, markersize=4)
        ax.plot([0, 2000], [0, 2000], "k--", lw=0.8, label="identity")
        ax.set_xlabel("Analytical eigenfrequency (Hz)")
        ax.set_ylabel("Picked ISM peak (Hz)")
        ax.set_title("ISM peak vs nearest analytical eigenfrequency, all 15 rooms")
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 2000)
        handles = [
            plt.Line2D([], [], marker="o", lw=0, color=L_to_color[L], label=f"L={L:.2f}")
            for L in Ls
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, dpi=110, bbox_inches="tight")
        plt.close(fig)

        # ----------------------------- Page 2 -------------------------------
        # Modal-frequency-vs-L for the lowest mode families.
        # Targets: (1,0), (2,0), (3,0), (0,1), (0,2). For each L the analytical
        # frequency f_{n_x, n_y} = c/2 · sqrt((n_x/L)² + (n_y/W)²); plot curve and
        # overlay the closest picked peak frequency.
        target_pairs = [(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (1, 1)]
        c = C_DEFAULT
        W = rooms[0]["W"]

        fig, ax = plt.subplots(figsize=(8, 6))
        L_grid = np.linspace(min(Ls) - 0.1, max(Ls) + 0.1, 200)
        for (n_x, n_y), color in zip(target_pairs, plt.cm.tab10.colors):
            f_curve = (c / 2.0) * np.sqrt((n_x / L_grid) ** 2 + (n_y / W) ** 2)
            ax.plot(L_grid, f_curve, color=color, lw=1.0,
                    label=f"({n_x},{n_y}) analytical")
            # Overlay closest picked peak per room.
            xs, ys = [], []
            for r in rooms:
                f_target = (c / 2.0) * np.sqrt((n_x / r["L"]) ** 2 + (n_y / W) ** 2)
                if not r["peaks"]:
                    continue
                best = min(r["peaks"], key=lambda p: abs(p.f - f_target))
                if abs(best.f - f_target) <= max(4.0, 0.05 * f_target):
                    xs.append(r["L"])
                    ys.append(best.f)
            ax.scatter(xs, ys, color=color, marker="x", s=30)
        ax.set_xlabel("L (m)")
        ax.set_ylabel("frequency (Hz)")
        ax.set_title("Mode frequency vs L (curves = analytical, x = closest picked ISM peak)")
        ax.set_ylim(0, 250)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, dpi=110, bbox_inches="tight")
        plt.close(fig)

        # ----------------------------- Page 3 -------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        Ls_arr = np.array([r["L"] for r in rooms])
        sabine = np.array([r["T60_sabine"] for r in rooms])
        edc_t = np.array([r["T60_edc"] for r in rooms])
        # Sabine (3D-style applied to 2D with V=A, S=P). pra uses this convention,
        # so the dashed line should overlap pra's blue dots when the convention matches.
        alpha = 0.15  # known constant for our sweep
        L_grid = np.linspace(min(Ls_arr), max(Ls_arr), 200)
        A = L_grid * W
        P = 2 * (L_grid + W)
        T60_sabine_3d_form = 0.161 * A / (alpha * P)

        ax.plot(L_grid, T60_sabine_3d_form * 1000, "k--", lw=0.8,
                label="0.161·A/(α·P) (3D Sabine applied to 2D)")
        ax.plot(Ls_arr, sabine * 1000, "o", color="steelblue", label="pra rt60_theory(sabine)")
        ax.plot(Ls_arr, edc_t * 1000, "x", color="indianred", label="EDC fit (-5..-35 dB)")
        ax.set_xlabel("L (m)")
        ax.set_ylabel("T60 (ms)")
        ax.set_title(
            f"T60 vs L  (W={W:.1f} m, α={alpha:.2f})\n"
            "pra applies the 3D Sabine formula to 2D; the EDC measures the actual decay "
            "and is consistently lower because\nthe diffuse-field assumption breaks down at α=0.15."
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, dpi=110, bbox_inches="tight")
        plt.close(fig)


def write_index(out_dir: Path, h5_paths: list[Path]):
    md = ["# Visual sanity pack — INDEX\n",
          "\nGenerated by `scripts/visual_sanity.py`. Browse PDFs in any reader.\n",
          "\n## Cross-room\n",
          "- [`cross_room.pdf`](cross_room.pdf) — modal-tracking scatter, mode-vs-L lines, T60 vs L.\n",
          "\n## Per-room (one PDF per L)\n"]
    for p in sorted(h5_paths):
        L = float(p.stem.split("_")[1].rstrip("m"))
        rel = f"per_room/L_{L:.2f}m.pdf"
        md.append(f"- [`{rel}`]({rel}) — L={L:.2f} m: 4-panel IR view, 8×8 sparkline grid, spectrum overlay.\n")
    md.append("\n## Notes\n- See [`SANITY_NOTES.md`](SANITY_NOTES.md) for the agent's eyeballing observations.\n")
    (out_dir / "INDEX.md").write_text("".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data/track_a"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/visual_sanity"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    per_room_dir = out_dir / "per_room"
    per_room_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(data_dir.glob("L_*_W_*_alpha_*.h5"))
    if not h5_paths:
        raise RuntimeError(f"no HDF5 files in {data_dir}")

    print(f"# generating per-room PDFs for {len(h5_paths)} rooms")
    for p in h5_paths:
        L = float(p.stem.split("_")[1].rstrip("m"))
        out = per_room_dir / f"L_{L:.2f}m.pdf"
        per_room_pdf(p, out)
        print(f"  wrote {out.relative_to(out_dir)}")

    print("# generating cross-room PDF")
    cross_room_pdf(h5_paths, out_dir / "cross_room.pdf")

    write_index(out_dir, h5_paths)
    # Touch SANITY_NOTES.md placeholder if absent — agent fills this in afterward.
    notes = out_dir / "SANITY_NOTES.md"
    if not notes.exists():
        notes.write_text("# Sanity notes\n\n_(agent fills in observations after viewing the PDFs)_\n")
    print(f"# wrote {out_dir/'INDEX.md'}")


if __name__ == "__main__":
    main()
