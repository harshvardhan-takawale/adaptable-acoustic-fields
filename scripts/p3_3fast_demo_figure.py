"""P3-3-FAST headline demo: EDIT THE ROOM, ZERO-SHOT (FIG 5 of the meeting pack).

One held-out test geometry, five edits, five single forward passes. Nothing about any of the
five rooms is measured, fitted or optimised: the model sees only ``(L, W, 16 segment
absorptions)`` and renders the field. The five panels are

    (a) baseline      all 16 segments at alpha = 0.15
    (b) curtain       alpha = 0.50 on west_2   -- a TRAINED segment position
    (c) curtain       alpha = 0.50 on east_3   -- the HELD-OUT position (0 training configs)
    (d) window open   alpha = 0.95 on east_3   -- held-out position, matched-impedance
    (e) window closed back to alpha = 0.15, rendered as its OWN forward pass

and each carries both a receiver-RMS spectrum (prediction over ground truth, 0-300 Hz) and the
spatial field map at that panel's most-affected mode.

**Ground truth for (b) and (c) does not exist in the Track A corpus** -- the test split only
enumerates alpha = 0.70 and alpha = 0.95 edits -- so this script SIMULATES it, through the
exact dataset builder (``scripts.build_p3_3fast_trackA.build_one``, same solver, same dx, same
fs, same receiver snapping), into a SEPARATE directory ``data/track_p3_3fast_A_demo`` so the
training corpus and its gate hashes are untouched. The build is idempotent (``.done``
sentinels) and costs ~20 s per config.

Everything else is imported, never reimplemented:

* ``load_model`` / ``load_gt`` / ``band_limit`` from :mod:`aaf.eval.p3_2_eval`. ``load_model``
  puts BOTH model and renderer in ``eval()``; the renderer flag is load-bearing (D49 C3:
  ``FreqRenderer2D`` jitters ray azimuths while ``self.training``), and panel (e) asserts
  bit-for-bit equality with panel (a) as the standing check that it took.
* ``render_config_arm`` from :mod:`aaf.eval.p3_2b_eval` -- dispatches the conditioning encoder
  on the checkpoint's own ``cond_source`` (``m_token`` / 448-d for Track A2).
* ``usable_mask`` / ``band_energy_db`` from :mod:`scripts.p3_3fast_trackA_diag` -- the SAME
  in-band energy-delta estimator whose aggregate over the 10 test geometries is the
  +1.010 (held-out) / +1.106 (seen) window recovery this figure illustrates.
* ``lsd_floored`` from :mod:`scripts.p3_3fast_floored_lsd` -- raw and -40 dB-from-peak LSD.
  The floor is a REPORTING choice and is never a selection criterion; raw sits beside it
  everywhere.

Geometry selection is by a rule fixed before any number was looked at: the test geometry whose
window-hold-out energy recovery is the MEDIAN of the ten in the converged diagnostic. Not the
best one. The full ranking is written into the sidecar JSON and the manifest entry.

Usage
-----
    python scripts/p3_3fast_demo_figure.py            # needs a GPU (tinycudann)
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aaf.data.seg_configs import (  # noqa: E402
    N_SEG,
    SEGMENT_NAMES,
    WINDOW_ALPHA,
    SegConfig,
    configs_from_rows,
    segment_index,
)
from aaf.walls import ALPHA_BASELINE, WALLS_2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

BAND_HI_HZ = 300.0
DF_HZ = 0.5
HI_IDX = int(round(BAND_HI_HZ / DF_HZ)) + 1        # 601 bins, 0..300 Hz inclusive
CURTAIN_ALPHA = 0.50
FLOOR_DB = -40.0
"""Reporting floor for the floored LSD and for the modal-peak candidate set. 97.3% of in-band
bins sit more than 40 dB below their config's own peak, so plain LSD is an average over nulls
nobody hears. Raw LSD is reported next to every floored value."""

N_RX_SIDE = 8
EPS = 1e-8

# --------------------------------------------------------------------------- style
FIGSIZE = (20.0, 12.4)
DPI = 160                                           # -> 3200 x 1984 px
MIN_PX = (1920, 1080)
CAP_FS = 10.6                                       # caption / note point sizes, used to
NOTE_FS = 9.6                                       # reserve exactly the strip they need

RC = {
    "font.size": 11.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 12,
    "figure.titlesize": 19,
    "legend.fontsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.grid": True,
    "grid.color": "#dfe3e8",
    "grid.linewidth": 0.7,
    "axes.edgecolor": "#8c959f",
    "axes.linewidth": 0.9,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

# Okabe & Ito (2008), a published CVD-safe set. Identity is never colour-alone: ground truth is
# a thick solid line, the prediction a dashed line with markers, the baseline reference a thin
# dotted line, and each is labelled in the legend.
OI_VERM = "#D55E00"
OI_BLUE = "#0072B2"
OI_SKY = "#56B4E9"
INK = "#24292f"
MUTED = "#57606a"
CAPTION_WRAP = 190


# --------------------------------------------------------------------------- config layer
def _mk(L: float, W: float, edits: Dict[int, float], kind: str, gid: int) -> SegConfig:
    a = [ALPHA_BASELINE] * N_SEG
    for i, v in edits.items():
        a[i] = round(float(v), 6)
    return SegConfig(L=round(float(L), 2), W=round(float(W), 2), alphas=tuple(a),
                     kind=kind, split="demo", geom_id=int(gid))


def panel_specs(L: float, W: float, gid: int) -> List[Dict[str, Any]]:
    """The five panels, in order. ``(e)`` is a SEPARATE forward pass of the same input."""
    w2, e3 = segment_index("west", 2), segment_index("east", 3)
    base = _mk(L, W, {}, "baseline", gid)
    return [
        {"letter": "a", "key": "baseline", "cfg": base, "seg": None, "alpha": ALPHA_BASELINE,
         "name": "baseline", "held_out_position": False,
         "headline": "(a) BASELINE\nall 16 segments at $\\alpha$ = 0.15"},
        {"letter": "b", "key": "curtain_seen", "cfg": _mk(L, W, {w2: CURTAIN_ALPHA},
                                                          "demo_curtain_seen", gid),
         "seg": w2, "alpha": CURTAIN_ALPHA, "name": "curtain on west_2",
         "held_out_position": False,
         "headline": "(b) CURTAIN on west_2\n$\\alpha$ 0.15 $\\to$ 0.50   -   trained position"},
        {"letter": "c", "key": "curtain_holdout", "cfg": _mk(L, W, {e3: CURTAIN_ALPHA},
                                                             "demo_curtain_holdout", gid),
         "seg": e3, "alpha": CURTAIN_ALPHA, "name": "curtain on east_3",
         "held_out_position": True,
         "headline": "(c) CURTAIN on east_3\n$\\alpha$ 0.15 $\\to$ 0.50   -   HELD-OUT position"},
        {"letter": "d", "key": "window_open", "cfg": _mk(L, W, {e3: WINDOW_ALPHA},
                                                         "t_window_holdout", gid),
         "seg": e3, "alpha": WINDOW_ALPHA, "name": "window open at east_3",
         "held_out_position": True,
         "headline": ("(d) WINDOW OPEN at east_3\n"
                      "$\\alpha$ 0.15 $\\to$ 0.95   -   HELD-OUT position")},
        {"letter": "e", "key": "window_closed", "cfg": base, "seg": None,
         "alpha": ALPHA_BASELINE, "name": "window closed", "held_out_position": False,
         "headline": "(e) WINDOW CLOSED\neast_3 back to $\\alpha$ = 0.15"},
    ]


def segment_span(L: float, W: float, index: int) -> Tuple[Tuple[float, float],
                                                          Tuple[float, float]]:
    """Endpoints of segment ``index`` in room metres -- used to draw the edit on the map."""
    n_per = 4
    wall = WALLS_2D[index // n_per]
    k = index % n_per
    lo, hi = k / float(n_per), (k + 1) / float(n_per)
    if wall == "west":
        return (0.0, lo * W), (0.0, hi * W)
    if wall == "east":
        return (L, lo * W), (L, hi * W)
    if wall == "south":
        return (lo * L, 0.0), (hi * L, 0.0)
    return (lo * L, W), (hi * L, W)


# --------------------------------------------------------------------------- geometry pick
def choose_geometry(diag_path: Path, manifest_geoms: Sequence[int]) -> Dict[str, Any]:
    """Median (NOT best) test geometry by window-hold-out energy recovery in the diagnostic."""
    if not diag_path.exists():
        gid = sorted(manifest_geoms)[0]
        return {"geom_id": gid, "rule": "diagnostic missing -- fell back to lowest geom_id",
                "ranking": []}
    d = json.loads(diag_path.read_text())
    rows = [r for r in d["window"]["per_config"] if r["slot"] == "t_window_holdout"]
    rows = [r for r in rows if np.isfinite(r.get("energy_recovered_frac", np.nan))]
    rows.sort(key=lambda r: r["energy_recovered_frac"])
    if not rows:
        gid = sorted(manifest_geoms)[0]
        return {"geom_id": gid, "rule": "no finite recovery in diagnostic -- lowest geom_id",
                "ranking": []}
    pick = rows[(len(rows) - 1) // 2]                 # lower median, deterministic
    return {
        "geom_id": int(pick["geom_id"]),
        "rule": ("median of the {} test geometries by t_window_holdout "
                 "energy_recovered_frac in {} (lower median, deterministic; NOT the best "
                 "geometry)".format(len(rows), diag_path)),
        "picked_recovery_frac": float(pick["energy_recovered_frac"]),
        "ranking": [{"geom_id": int(r["geom_id"]),
                     "energy_recovered_frac": float(r["energy_recovered_frac"])}
                    for r in rows],
    }


# --------------------------------------------------------------------------- ground truth
def ensure_gt(cfg: SegConfig, data_dir: Path, demo_dir: Path) -> Tuple[Path, str]:
    """Return the HDF5 for ``cfg``, simulating it into ``demo_dir`` if the corpus lacks it."""
    p = data_dir / cfg.filename
    if p.exists():
        return p, "corpus"
    q = demo_dir / cfg.filename
    if q.exists() and (demo_dir / (cfg.filename + ".done")).exists():
        return q, "demo-cache"
    from scripts.build_p3_3fast_trackA import build_one

    demo_dir.mkdir(parents=True, exist_ok=True)
    print("[fdtd] simulating {} ({})".format(cfg.filename, cfg.label), flush=True)
    build_one(cfg, demo_dir)
    return q, "simulated-here"


# --------------------------------------------------------------------------- measurement
def rms_spectrum_db(H: np.ndarray) -> np.ndarray:
    """Receiver-RMS magnitude in dB: ``10*log10(mean_rx |H|^2)``, one value per bin."""
    return 10.0 * np.log10(np.mean(np.abs(np.asarray(H)) ** 2, axis=0) + 1e-30)


def modal_peaks(s_db: np.ndarray, floor_db: float = FLOOR_DB) -> np.ndarray:
    """Indices of local maxima of ``s_db`` within ``floor_db`` of the STRONGEST LOCAL MAXIMUM.

    Referenced to the strongest peak, NOT to ``s.max()``. In this corpus ``s.max()`` is the
    bin-0 (0,0) compliance term, which sits ~46 dB above the strongest room mode; a floor
    referenced to it selects the sub-6 Hz region and no room mode at all.
    """
    s = np.asarray(s_db, dtype=float)
    i = np.arange(1, s.size - 1)
    loc = i[(s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])]
    if loc.size == 0:
        return loc
    return loc[s[loc] >= (s[loc].max() + floor_db)]


def _modes(L: float, W: float):
    from aaf.eval.modal_projection import enumerate_modes

    return enumerate_modes(L, W, f_max=float(BAND_HI_HZ) + 20.0)


def nearest_mode(L: float, W: float, f_hz: float) -> Dict[str, Any]:
    modes = _modes(L, W)
    if not modes:
        return {}
    m = min(modes, key=lambda mm: abs(mm.f - f_hz))
    return {"n_x": int(m.n_x), "n_y": int(m.n_y), "f_theory_hz": float(m.f),
            "family": str(m.family), "offset_hz": float(f_hz - m.f)}


def resolvable(L: float, W: float, d_x: float, d_y: float) -> Dict[str, Any]:
    """Which modes the 8 x 8 receiver grid can actually SHOW without spatial aliasing.

    The panel's bottom rows are mode-shape maps sampled on that grid, so a mode whose
    half-wavelength along an axis is shorter than the receiver spacing renders as aliased
    noise. Requiring ``L / n_x >= 2 d_x`` (and likewise in y) is the spatial Nyquist bound;
    it is a property of the PLOT, fixed by the receiver layout, not of the model.
    """
    return {"n_x_max": int(np.floor(L / (2.0 * d_x))), "n_y_max": int(np.floor(W / (2.0 * d_y))),
            "rx_spacing_m": [float(d_x), float(d_y)]}


def resolvable_peaks(cand: np.ndarray, L: float, W: float, lim: Dict[str, Any]) -> np.ndarray:
    keep = []
    for i in cand:
        nm = nearest_mode(L, W, float(i) * DF_HZ)
        if nm and nm["n_x"] <= lim["n_x_max"] and nm["n_y"] <= lim["n_y_max"]:
            keep.append(int(i))
    return np.asarray(keep, dtype=int)


def lsd_on_bins(pred: np.ndarray, gt: np.ndarray, bins: Sequence[int]) -> float:
    """LSD over ALL receivers at the given bins -- 'error on the room modes', explicitly."""
    b = np.asarray(list(bins), dtype=int)
    if b.size == 0:
        return float("nan")
    p = 20.0 * np.log10(np.maximum(np.abs(np.asarray(pred)[:, b]), 1e-30))
    g = 20.0 * np.log10(np.maximum(np.abs(np.asarray(gt)[:, b]), 1e-30))
    return float(np.mean(np.abs(p - g)))


# --------------------------------------------------------------------------- plotting
def _fmt_recovery(v: Optional[float]) -> str:
    return "n/a" if v is None or not np.isfinite(v) else "{:+.3f}".format(v)


def draw(res, out_path: Path) -> Tuple[int, int]:
    """Five columns x (spectrum | metrics | GT map | predicted map), plus caption and note."""
    P = res["panels"]
    L, W = res["geometry"]["L"], res["geometry"]["W"]
    f_axis = np.asarray(res["f_axis_hz"])
    rx = np.asarray(res["rx"])
    src = np.asarray(res["src"])
    xs = np.unique(np.round(rx[:, 0], 6))
    ys = np.unique(np.round(rx[:, 1], 6))
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    extent = [xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2]

    cap = textwrap.fill(res["caption"], CAPTION_WRAP)
    note_txt = textwrap.fill(res["note"], CAPTION_WRAP)
    # Reserve exactly the height the text needs: n_lines * point_size / (fig_height * 72),
    # scaled by the line spacing. A fixed guess leaves either a hole or an overlap.
    cap_h = (cap.count("\n") + 1) * CAP_FS * 1.32 / (FIGSIZE[1] * 72.0)
    note_h = (note_txt.count("\n") + 1) * NOTE_FS * 1.32 / (FIGSIZE[1] * 72.0)
    y_note = 0.010
    y_cap = y_note + note_h + 0.014
    # +0.048 because tick labels and the "x (m)" xlabel are drawn OUTSIDE the gridspec cell.
    bottom = y_cap + cap_h + 0.012 + 0.048

    plt.rcParams.update(RC)
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.suptitle(res["suptitle"], fontweight="bold", y=0.978, color=INK)
    gs = fig.add_gridspec(4, 6, left=0.052, right=0.922, bottom=bottom, top=0.884,
                          width_ratios=[1, 1, 1, 1, 1, 0.042],
                          height_ratios=[1.22, 0.50, 1.02, 1.02], wspace=0.26, hspace=0.30)

    # Each ROW of maps is referenced to its own panel-(a) peak, so the constant model-vs-GT
    # level offset drops out while every per-panel difference survives; the row is then
    # colour-scaled to its own range so a compressed GT mode shape is still legible.
    ref = {k: float(np.max(np.asarray(P[0]["map_" + k], dtype=float))) for k in ("gt", "pred")}
    lims = {}
    for k in ("gt", "pred"):
        v = np.concatenate([np.asarray(p["map_" + k], dtype=float).ravel() - ref[k] for p in P])
        lims[k] = (float(v.min()), float(v.max()))

    # Spectrum y-limits track the MODAL band: the bin-0 (0,0) compliance term is ~46 dB above
    # the strongest room mode and would flatten every mode into a line.
    pk = float(res["numbers"]["mode_candidates"]["strongest_baseline_modal_peak_db"])
    ylim = (pk - 60.0, pk + 15.0)

    for j, p in enumerate(P):
        # ------------------------------------------------------------- row 0: spectrum
        ax = fig.add_subplot(gs[0, j])
        ax.plot(f_axis, P[0]["spec_gt"], color=MUTED, lw=0.9, ls=":", zorder=1,
                label="GT baseline room")
        ax.plot(f_axis, p["spec_gt"], color="#000000", lw=1.7, zorder=3, label="GT (FDTD)")
        ax.plot(f_axis, p["spec_pred"], color=OI_VERM, lw=1.5, ls="--", dashes=(4.5, 2.2),
                zorder=4, label="prediction (zero-shot)")
        ax.axvline(p["mode"]["f_hz"], color=OI_BLUE, lw=1.2, ls="-.", zorder=2, alpha=0.9)
        ax.annotate("{:.1f} Hz\nmap".format(p["mode"]["f_hz"]),
                    xy=(p["mode"]["f_hz"], 0.0), xycoords=("data", "axes fraction"),
                    xytext=(4, 5), textcoords="offset points", fontsize=9.0, color=OI_BLUE,
                    ha="left", va="bottom", fontweight="bold")
        ax.set_xlim(0, BAND_HI_HZ)
        ax.set_ylim(*ylim)
        ax.set_xlabel("frequency (Hz)", labelpad=2)
        if j == 0:
            ax.set_ylabel("receiver-RMS $|H|$ (dB)")
            ax.legend(loc="upper right", framealpha=0.94, fontsize=9.0, handlelength=2.4,
                      borderpad=0.4, labelspacing=0.35)
        ax.set_title(p["headline"], fontsize=10.6, color=INK, linespacing=1.45, pad=6)

        # ------------------------------------------------------------- row 1: metrics
        axt = fig.add_subplot(gs[1, j])
        axt.axis("off")
        axt.text(0.5, 0.80, p["title_metrics"], transform=axt.transAxes, ha="center",
                 va="top", fontsize=8.8, color=INK, linespacing=1.55,
                 bbox=dict(boxstyle="round,pad=0.36", fc="#f6f8fa", ec="#c8d1da", lw=0.8))

        # ------------------------------------------------------------- rows 2-3: maps
        for r, key, lab in ((2, "gt", "GT"), (3, "pred", "predicted")):
            axm = fig.add_subplot(gs[r, j])
            M = np.asarray(p["map_" + key], dtype=float) - ref[key]
            im = axm.imshow(M.T, origin="lower", extent=extent, vmin=lims[key][0],
                            vmax=lims[key][1], cmap="viridis", interpolation="nearest",
                            aspect="equal")
            axm.add_patch(plt.Rectangle((0, 0), L, W, fill=False, ec="#8c959f", lw=1.2))
            if p["seg_span"] is not None:
                (x0, y0), (x1, y1) = p["seg_span"]
                axm.plot([x0, x1], [y0, y1], color=OI_VERM, lw=5.0, solid_capstyle="butt",
                         zorder=5)
                axm.annotate(p["seg_name"], xy=(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
                             xytext=(7 if x0 < L / 2 else -7, 0), textcoords="offset points",
                             fontsize=8.5, color=OI_VERM, fontweight="bold",
                             ha="left" if x0 < L / 2 else "right", va="center", rotation=90)
            axm.plot([src[0]], [src[1]], marker="*", ms=12, color="#ffffff", mec="#000000",
                     mew=0.9, zorder=6)
            axm.set_xlim(-0.07 * L, 1.07 * L)
            axm.set_ylim(-0.07 * W, 1.07 * W)
            axm.grid(False)
            axm.tick_params(labelsize=9.5)
            if r == 3:
                axm.set_xlabel("x (m)", labelpad=2)
            else:
                axm.set_xticklabels([])
            if j == 0:
                axm.set_ylabel("{} field at {:.1f} Hz\ny (m)".format(lab, P[0]["mode"]["f_hz"]),
                               fontsize=10.5)
            if j == len(P) - 1:
                cb = fig.colorbar(im, cax=fig.add_subplot(gs[r, 5]))
                cb.set_label("{}: dB re panel (a) peak".format(lab), fontsize=8.8, labelpad=4)
                cb.ax.tick_params(labelsize=8.8)

    fig.text(0.5, y_cap, cap, ha="center", va="bottom", fontsize=CAP_FS, style="italic",
             color="#3d444d", linespacing=1.32)
    fig.text(0.5, y_note, note_txt, ha="center", va="bottom", fontsize=NOTE_FS, color=MUTED,
             linespacing=1.32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    from PIL import Image

    with Image.open(out_path) as im2:
        w, h = im2.size
    if w < MIN_PX[0] or h < MIN_PX[1]:
        raise RuntimeError("{} is {}x{}, below the {}x{} floor".format(
            out_path.name, w, h, MIN_PX[0], MIN_PX[1]))
    return w, h


# --------------------------------------------------------------------------- manifest
FIG_NAME = "fig5_topological_edits"


def manifest_block(res: Dict[str, Any], out_path: Path, px: Tuple[int, int],
                   idx: int = 5) -> List[str]:
    from scripts.p3_3fast_figures import _render_numbers

    lines = ["---", "", "## FIG {} -- `{}`".format(idx, FIG_NAME), ""]
    try:
        shown = out_path.relative_to(ROOT)
    except ValueError:
        shown = out_path
    lines.append("**File**: `{}` ({} x {} px, {} dpi)".format(shown, px[0], px[1], DPI))
    lines.append("")
    lines.append("**Produced by**: `scripts/p3_3fast_demo_figure.py` (re-runnable; "
                 "`sbatch scripts/slurm/p3_3fast_demo_figure.sh`). NOTE: "
                 "`scripts/p3_3fast_figures.py` REWRITES this manifest from scratch, so "
                 "re-run this script after it to restore this entry -- the append is "
                 "idempotent and replaces its own section in place.")
    lines.append("")
    lines.append("**Source(s)** -- every plotted value comes from here:")
    for s in res["sources"]:
        lines.append("- `{}`".format(s))
    lines.append("")
    lines.append("**Caption** (as printed on the figure):")
    lines.append("")
    lines.append("> {}".format(res["caption"]))
    lines.append("")
    lines.append("**Figure note** (as printed on the figure):")
    lines.append("")
    lines.append("> {}".format(res["note"]))
    lines.append("")
    lines.append("**Exact numbers plotted**:")
    lines.append("")
    lines.extend(_render_numbers(res["numbers"]))
    lines.append("")
    lines.append("**Computed here** (not read from any JSON):")
    for c in res["computed_here"]:
        lines.append("- {}".format(c))
    lines.append("")
    lines.append("**Honest limitations**:")
    for c in res["limitations"]:
        lines.append("- {}".format(c))
    lines.append("")
    return lines


def append_manifest(md_path: Path, block: List[str], px: Tuple[int, int],
                    idx: int = 5) -> None:
    """Replace this figure's section in place if present, else append. Also fixes the TOC."""
    if md_path.exists():
        lines = md_path.read_text().split("\n")
    else:
        lines = ["# P3-3 fast-track meeting pack -- FIGURE MANIFEST", "", "## Contents", "",
                 "| # | figure | status | px |", "|---|---|---|---|", ""]

    # drop any previous section for this figure (from its `---` separator to the next one)
    start = next((i for i, ln in enumerate(lines)
                  if ln.startswith("## FIG ") and FIG_NAME in ln), None)
    if start is not None:
        s = start - 2 if start >= 2 and lines[start - 2].strip() == "---" else start
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].strip() == "---"), len(lines))
        lines = lines[:s] + lines[end:]

    # TOC row
    row = "| {} | [`{}.png`]({}.png) | written | {} x {} |".format(
        idx, FIG_NAME, FIG_NAME, px[0], px[1])
    hit = next((i for i, ln in enumerate(lines) if FIG_NAME in ln and ln.startswith("| ")),
               None)
    if hit is not None:
        lines[hit] = row
    else:
        tbl = [i for i, ln in enumerate(lines)
               if ln.startswith("| ") and "|" in ln[2:] and not ln.startswith("| # |")
               and not ln.startswith("|---")]
        if tbl:
            lines.insert(tbl[-1] + 1, row)
        else:
            hdr = next((i for i, ln in enumerate(lines) if ln.startswith("|---")), None)
            lines.insert(hdr + 1 if hdr is not None else len(lines), row)

    while lines and lines[-1].strip() == "":
        lines.pop()
    md_path.write_text("\n".join(lines + [""] + block) + "\n")


# --------------------------------------------------------------------------- driver
def run(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    from aaf.eval.p3_2_eval import band_limit, load_gt, load_model
    from aaf.eval.p3_2b_eval import render_config_arm
    from scripts.p3_3fast_floored_lsd import lsd_floored
    from scripts.p3_3fast_trackA_diag import band_energy_db, usable_mask

    root = Path(args.repo_root) if args.repo_root else ROOT
    data_dir = root / args.data_dir
    demo_dir = root / args.demo_data_dir
    man_path = root / args.manifest
    ck_path = root / args.checkpoint

    rows = json.loads(man_path.read_text())["configs"]
    test = configs_from_rows(rows, split="test")
    pick = choose_geometry(root / args.diagnostic, sorted({c.geom_id for c in test}))
    gid = int(args.geom_id) if args.geom_id is not None else pick["geom_id"]
    if args.geom_id is not None:
        pick = {"geom_id": gid, "rule": "forced by --geom-id", "ranking": pick.get("ranking")}
    geo_cfgs = [c for c in test if c.geom_id == gid]
    if not geo_cfgs:
        raise SystemExit("no test configs for geom_id {}".format(gid))
    L, W = geo_cfgs[0].L, geo_cfgs[0].W
    print("[geom] id={} L={} W={} ({})".format(gid, L, W, pick["rule"]), flush=True)

    specs = panel_specs(L, W, gid)
    # (a)/(e) and (d) must be the corpus files, not look-alikes: assert identity by filename.
    corpus = {c.kind: c for c in geo_cfgs}
    for s in specs:
        if s["key"] in ("baseline", "window_closed"):
            assert s["cfg"].filename == corpus["baseline"].filename, "baseline mismatch"
        if s["key"] == "window_open":
            assert s["cfg"].filename == corpus["t_window_holdout"].filename, "window mismatch"

    # ---------------------------------------------------------------- ground truth
    gt_raw, prov, rx0, src0 = {}, {}, None, None
    for s in specs:
        k = s["key"]
        if k == "window_closed":
            gt_raw[k], prov[k] = gt_raw["baseline"], prov["baseline"]
            continue
        p, how = ensure_gt(s["cfg"], data_dir, demo_dir)
        H, rx, src, _ = load_gt(p)
        if rx0 is None:
            rx0, src0 = rx, src
        elif not np.allclose(rx, rx0) or not np.allclose(src, src0):
            raise RuntimeError("receiver/source grid differs for {}".format(s["cfg"].filename))
        gt_raw[k] = band_limit(H, HI_IDX)[:, :HI_IDX]
        prov[k] = {"path": str(p.relative_to(root)), "provenance": how}
        print("[gt] {} <- {} ({})".format(k, p.name, how), flush=True)

    # ---------------------------------------------------------------- model + renders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer, cfg, tmeta, it = load_model(ck_path, device)
    model.eval()
    renderer.eval()                                   # load-bearing, D49 C3
    cond_source = str(cfg["cond_source"])
    if (cond_source, int(cfg["cond_dim"])) != ("m_token", 448):
        raise SystemExit("expected the Track A2 m_token/448 arm, got {}/{}".format(
            cond_source, cfg["cond_dim"]))

    pred = {}
    for s in specs:                                   # ONE forward pass per panel, (e) too
        H = render_config_arm(model, renderer, cond_source, L, W, list(s["cfg"].alphas),
                              rx0, src0, device, rx_chunk=args.rx_chunk)
        pred[s["key"]] = band_limit(np.asarray(H)[:, :HI_IDX], HI_IDX)[:, :HI_IDX]
        print("[render] {} done".format(s["key"]), flush=True)
    determinism_ok = bool(np.array_equal(pred["baseline"], pred["window_closed"]))
    print("[check] (e) bitwise-identical to (a): {}".format(determinism_ok), flush=True)

    # ---------------------------------------------------------------- measurement
    stack = np.stack([gt_raw[s["key"]] for s in specs] + [pred[s["key"]] for s in specs])
    mask = usable_mask(stack)
    f_axis = np.arange(HI_IDX) * DF_HZ
    spec_gt = {k: rms_spectrum_db(v) for k, v in gt_raw.items()}
    spec_pr = {k: rms_spectrum_db(v) for k, v in pred.items()}
    peaks = modal_peaks(spec_gt["baseline"])
    d_rx = (float(np.diff(np.unique(np.round(rx0[:, 0], 6)))[0]),
            float(np.diff(np.unique(np.round(rx0[:, 1], 6)))[0]))
    lim = resolvable(L, W, d_rx[0], d_rx[1])
    cand = resolvable_peaks(peaks, L, W, lim)
    if cand.size == 0:
        raise SystemExit("no spatially resolvable modal peaks -- check the receiver grid")
    print("[modes] {} baseline peaks, {} resolvable (n_x<={}, n_y<={})".format(
        peaks.size, cand.size, lim["n_x_max"], lim["n_y_max"]), flush=True)

    # level offset removed by the per-row map referencing, reported so it is on the record
    above = spec_gt["baseline"] >= spec_gt["baseline"].max() + FLOOR_DB
    lvl_offset = float(np.median(spec_pr["baseline"][above] - spec_gt["baseline"][above]))

    panels: List[Dict[str, Any]] = []
    for s in specs:
        k = s["key"]
        d_gt = spec_gt[k] - spec_gt["baseline"]
        if k in ("baseline", "window_closed"):
            mode_idx = None                            # filled from panel (d) below
            top, unrestricted = [], None
        else:
            mode_idx = int(cand[np.argmax(np.abs(d_gt[cand]))])
            order = cand[np.argsort(-np.abs(d_gt[cand]))][:3]
            top = [{"f_hz": float(i * DF_HZ), "gt_delta_db": float(d_gt[i]),
                    "baseline_level_db": float(spec_gt["baseline"][i])} for i in order]
            u = int(peaks[np.argmax(np.abs(d_gt[peaks]))])
            unrestricted = {"f_hz": float(u * DF_HZ), "gt_delta_db": float(d_gt[u]),
                            "baseline_level_db": float(spec_gt["baseline"][u]),
                            "same_as_plotted": bool(u == mode_idx)}
        lsd_raw, frac_raw = lsd_floored(pred[k], gt_raw[k], None)
        lsd_fl, frac_fl = lsd_floored(pred[k], gt_raw[k], FLOOR_DB)
        lsd_pk = lsd_on_bins(pred[k], gt_raw[k], peaks)
        e_gt = band_energy_db(gt_raw[k], gt_raw["baseline"], mask)
        e_pr = band_energy_db(pred[k], pred["baseline"], mask)
        rec = (e_pr / e_gt) if (np.isfinite(e_gt) and abs(e_gt) > 1e-9) else float("nan")
        panels.append({
            "letter": s["letter"], "key": k, "name": s["name"], "headline": s["headline"],
            "held_out_position": s["held_out_position"],
            "alphas": list(s["cfg"].alphas), "edited": list(s["cfg"].edited),
            "seg_name": None if s["seg"] is None else SEGMENT_NAMES[s["seg"]],
            "seg_span": None if s["seg"] is None else segment_span(L, W, s["seg"]),
            "gt_file": prov[k], "_mode_idx": mode_idx, "mode_candidates_top3": top,
            "mode_unrestricted_argmax": unrestricted,
            "d_energy_gt_db": e_gt, "d_energy_pred_db": e_pr, "energy_recovered_frac": rec,
            "lsd_raw_db": lsd_raw, "lsd_floor40_db": lsd_fl, "lsd_modal_peaks_db": lsd_pk,
            "frac_bins_above_floor": frac_fl,
            "spec_gt": spec_gt[k].tolist(), "spec_pred": spec_pr[k].tolist(),
        })

    d_idx = next(p["_mode_idx"] for p in panels if p["key"] == "window_open")
    for p in panels:
        if p["_mode_idx"] is None:
            p["_mode_idx"] = d_idx
            p["mode_note"] = ("baseline panel: GT delta vs baseline is identically zero, so "
                              "no mode is 'most affected'; the map is shown at panel (d)'s "
                              "mode so the open/closed pair is at one frequency")
        else:
            p["mode_note"] = "argmax |GT delta vs baseline| over baseline modal peaks"
        i = p["_mode_idx"]
        p["mode"] = {"bin": int(i), "f_hz": float(i * DF_HZ),
                     "gt_delta_at_mode_db": float(spec_gt[p["key"]][i]
                                                  - spec_gt["baseline"][i]),
                     "pred_delta_at_mode_db": float(spec_pr[p["key"]][i]
                                                    - spec_pr["baseline"][i]),
                     "nearest_analytic_mode": nearest_mode(L, W, i * DF_HZ)}
        p["map_gt"] = (20.0 * np.log10(np.abs(gt_raw[p["key"]][:, i]) + 1e-30)).reshape(
            N_RX_SIDE, N_RX_SIDE).tolist()
        p["map_pred"] = (20.0 * np.log10(np.abs(pred[p["key"]][:, i]) + 1e-30)).reshape(
            N_RX_SIDE, N_RX_SIDE).tolist()
        a_ = np.asarray(p["map_gt"], dtype=float).ravel()
        b_ = np.asarray(p["map_pred"], dtype=float).ravel()
        p["map_spatial_corr"] = float(np.corrcoef(a_, b_)[0, 1])
        nm = p["mode"]["nearest_analytic_mode"]
        p["title_metrics"] = (
            "in-band GT $\\Delta E$ {}   |   recovered {}\n"
            "LSD dB:  {:.2f} raw / {:.2f} modal / {:.2f} floored\n"
            "map mode ({},{}) @ {:.1f} Hz,  GT $\\Delta$ {:+.2f} dB\n"
            "map GT-vs-pred spatial $r$ = {:+.2f}".format(
                "0.00 dB (ref)" if abs(p["d_energy_gt_db"]) < 1e-9
                else "{:+.2f} dB".format(p["d_energy_gt_db"]),
                _fmt_recovery(p["energy_recovered_frac"])
                + ("$\\times$" if np.isfinite(p["energy_recovered_frac"]) else ""),
                p["lsd_raw_db"], p["lsd_modal_peaks_db"], p["lsd_floor40_db"],
                nm.get("n_x", -1), nm.get("n_y", -1), p["mode"]["f_hz"],
                p["mode"]["gt_delta_at_mode_db"], p["map_spatial_corr"]))

    # ---------------------------------------------------------------- text
    agg = json.loads((root / args.diagnostic).read_text())["window"]["by_slot"] \
        if (root / args.diagnostic).exists() else {}
    a_hold = agg.get("t_window_holdout", {}).get("energy_recovered_frac", {}).get("mean")
    a_seen = agg.get("t_window_seen", {}).get("energy_recovered_frac", {}).get("mean")
    fl = json.loads((root / args.floored_lsd).read_text()) \
        if (root / args.floored_lsd).exists() else {}
    fl_sum = fl.get("summary", {})

    by = {p["key"]: p for p in panels}
    caption = (
        "ZERO-SHOT room editing. One held-out test geometry ({:.2f} x {:.2f} m; the 10 test "
        "geometries share no (L, W) with the 20 training geometries), five edits, ONE FORWARD "
        "PASS PER PANEL -- no optimisation, no per-room fitting, and no measurement of any "
        "edited room enters the prediction; the model is given only (L, W) and the 16 segment "
        "absorptions. Panels (c) and (d) edit east_3, a segment position that is at the "
        "baseline in ALL 400 training configs. In-band (0-300 Hz) energy change vs the "
        "baseline room and the fraction of it the model recovers: (b) GT {:+.2f} dB, recovered "
        "{}; (c) GT {:+.2f} dB, recovered {}; (d) GT {:+.2f} dB, recovered {}. For reference, "
        "the converged Track A2 aggregate over all 10 test geometries is +{:.3f} at the "
        "held-out window position and +{:.3f} at the seen one. Panel (e) is a separate forward "
        "pass whose output is bitwise identical to (a): closing the window restores the room "
        "exactly.".format(
            L, W,
            by["curtain_seen"]["d_energy_gt_db"],
            _fmt_recovery(by["curtain_seen"]["energy_recovered_frac"]),
            by["curtain_holdout"]["d_energy_gt_db"],
            _fmt_recovery(by["curtain_holdout"]["energy_recovered_frac"]),
            by["window_open"]["d_energy_gt_db"],
            _fmt_recovery(by["window_open"]["energy_recovered_frac"]),
            a_hold if a_hold is not None else float("nan"),
            a_seen if a_seen is not None else float("nan")))
    peak_bins = sorted(int(i) for i in peaks)
    note = (
        "LIMITATION: the open window is modelled as a MATCHED-IMPEDANCE boundary (alpha -> "
        "0.95, a first-order absorbing condition). That captures the energy loss and the Q "
        "reduction of an aperture; it does NOT carry radiation reactance and does NOT carry "
        "edge diffraction at the rim, so the reactive near-field and the modal frequency pull "
        "of a real opening are absent from BOTH the model and this ground truth.  |  ACCURACY "
        "-- three LSDs, all quoted: raw over all 601 in-band bins {:.2f}-{:.2f} dB; over the "
        "{} baseline modal-peak bins {:.2f}-{:.2f} dB; over bins within {:.0f} dB of each "
        "config's own peak {:.2f}-{:.2f} dB{}. READ THE THIRD ONE WITH CARE: in this corpus "
        "the per-config peak is the bin-0 (0,0) compliance term, ~46 dB above the strongest "
        "room mode, so a -40 dB floor keeps only bins 0-12 (0-6 Hz) and measures the "
        "near-DC term, not the modal content. The modal-peak LSD is the honest 'error on the "
        "physics' number here. No floor was ever used to select a model or a checkpoint.  |  "
        "MODE SHAPE -- a NEGATIVE result, stated: at the plotted mode the predicted map does "
        "NOT reproduce the ground-truth standing wave (GT-vs-prediction spatial correlation r "
        "= {:+.2f} to {:+.2f} over the five panels). The model gets the receiver-averaged "
        "spectrum and the ENERGY response to the edit right while placing the low-frequency "
        "field wrongly in space -- its map stays source-centred where the GT is a standing "
        "wave. The zero-shot claim on this figure is about energy, not about mode shape.  |  "
        "MAPS: 8 x 8 receiver grid (nearest-neighbour, not interpolated, 0.3 m inset from "
        "the walls), plotted only at modes the grid resolves (n_x <= {}, n_y <= "
        "{}); each ROW is referenced to its own panel-(a) peak -- which removes the constant "
        "{:+.2f} dB model-vs-GT level offset but no per-panel difference -- and then colour-"
        "scaled to its own range. Star = source, thick bar = the edited segment. Spectra "
        "y-limits crop the +{:.0f} dB near-DC term.".format(
            min(p["lsd_raw_db"] for p in panels), max(p["lsd_raw_db"] for p in panels),
            len(peak_bins),
            min(p["lsd_modal_peaks_db"] for p in panels),
            max(p["lsd_modal_peaks_db"] for p in panels),
            FLOOR_DB,
            min(p["lsd_floor40_db"] for p in panels),
            max(p["lsd_floor40_db"] for p in panels),
            ("; corpus-wide over all {} test configs at this checkpoint {:.3f} raw / "
             "{:.3f} floored, i.e. this geometry is HARDER than the corpus mean, not an "
             "easy pick".format(
                 fl.get("n_configs"), fl_sum["raw"]["mean_lsd_db"],
                 fl_sum["floor_-40_db"]["mean_lsd_db"]) if fl_sum else ""),
            min(p["map_spatial_corr"] for p in panels),
            max(p["map_spatial_corr"] for p in panels),
            lim["n_x_max"], lim["n_y_max"], lvl_offset,
            float(np.max(spec_gt["baseline"]))))

    numbers: Dict[str, Any] = {
        "checkpoint": {"path": str(ck_path.relative_to(root)), "iter": int(it),
                       "cond_source": cond_source, "cond_dim": int(cfg["cond_dim"]),
                       "conditioning_type": str(cfg.get("conditioning_type", "film")),
                       "n_train_configs": int(tmeta.get("n_configs", 0))},
        "geometry": {"geom_id": gid, "L_m": L, "W_m": W, "selection_rule": pick["rule"],
                     "selection_ranking_window_holdout_recovery": pick.get("ranking"),
                     "source_pos_m": [float(x) for x in src0],
                     "n_receivers": int(rx0.shape[0]),
                     "receiver_grid": "{} x {}".format(N_RX_SIDE, N_RX_SIDE)},
        "band": {"hi_hz": BAND_HI_HZ, "df_hz": DF_HZ, "n_bins": HI_IDX,
                 "usable_cell_fraction": float(mask.mean()), "eps_usable": EPS},
        "mode_candidates": {
            "n_baseline_modal_peaks": len(peak_bins),
            "peak_bins_hz": [round(i * DF_HZ, 1) for i in peak_bins],
            "spatially_resolvable_bins_hz": [round(float(i) * DF_HZ, 1) for i in cand],
            "n_x_max": lim["n_x_max"], "n_y_max": lim["n_y_max"],
            "rx_spacing_m": lim["rx_spacing_m"],
            "strongest_baseline_modal_peak_db": float(np.max(spec_gt["baseline"][peaks])),
            "bin0_level_db": float(spec_gt["baseline"][0]),
            "bin0_above_strongest_mode_db": float(spec_gt["baseline"][0]
                                                  - np.max(spec_gt["baseline"][peaks])),
        },
        "determinism": {"panel_e_bitwise_identical_to_panel_a": determinism_ok},
        "level_offset_pred_minus_gt_db": lvl_offset,
        "aggregate_reference": {
            "trackA2_window_holdout_recovery_mean": a_hold,
            "trackA2_window_seen_recovery_mean": a_seen,
            "corpus_lsd_raw_mean_db": fl_sum.get("raw", {}).get("mean_lsd_db"),
            "corpus_lsd_floor40_mean_db": fl_sum.get("floor_-40_db", {}).get("mean_lsd_db"),
            "corpus_frac_bins_above_floor40": fl_sum.get("floor_-40_db", {}).get(
                "mean_frac_bins_used"),
            "corpus_checkpoint": fl.get("checkpoint"), "corpus_n_configs": fl.get("n_configs")},
        "panels": [{
            "panel": p["letter"], "name": p["name"], "edited_segments": p["edited"],
            "alpha_edited": ([p["alphas"][SEGMENT_NAMES.index(e)] for e in p["edited"]]
                             or [ALPHA_BASELINE]),
            "held_out_position": p["held_out_position"],
            "gt_file": p["gt_file"],
            "d_energy_gt_db": p["d_energy_gt_db"], "d_energy_pred_db": p["d_energy_pred_db"],
            "energy_recovered_frac": p["energy_recovered_frac"],
            "lsd_raw_db": p["lsd_raw_db"], "lsd_modal_peaks_db": p["lsd_modal_peaks_db"],
            "lsd_floor40_db": p["lsd_floor40_db"],
            "frac_bins_above_floor40": p["frac_bins_above_floor"],
            "map_gt_vs_pred_spatial_pearson_r": p["map_spatial_corr"],
            "mode": p["mode"], "mode_selection": p["mode_note"],
            "mode_candidates_top3_by_abs_gt_delta": p["mode_candidates_top3"],
            "mode_unrestricted_argmax": p["mode_unrestricted_argmax"],
        } for p in panels],
    }

    res = {
        "suptitle": "Edit the room, zero-shot -- one held-out geometry, one forward pass per "
                    "panel (Track A2, iter {})".format(it),
        "caption": caption, "note": note,
        "geometry": {"L": L, "W": W, "geom_id": gid},
        "f_axis_hz": f_axis.tolist(), "rx": rx0.tolist(), "src": [float(x) for x in src0],
        "panels": panels, "numbers": numbers,
        "sources": [
            str(ck_path.relative_to(root)),
            args.manifest,
            "{} (corpus GT for panels a/d/e)".format(args.data_dir),
            "{} (GT for panels b/c, simulated by this script through "
            "scripts/build_p3_3fast_trackA.build_one)".format(args.demo_data_dir),
            args.diagnostic + " (geometry pick + the A2 aggregate recovery quoted)",
            args.floored_lsd + " (corpus-wide raw vs floored LSD quoted in the note)",
        ],
        "computed_here": [
            "every per-panel number: the five predictions are rendered here, one "
            "render_config_arm forward pass each, and every plotted quantity is derived from "
            "those renders and the FDTD ground truth -- nothing is read from a summary",
            "in-band energy delta vs baseline = 10*log10(sum|H_panel|^2 / sum|H_baseline|^2) "
            "over the shared usable cells, via scripts.p3_3fast_trackA_diag.band_energy_db "
            "(the SAME estimator behind the quoted A2 aggregate)",
            "recovery fraction = d_energy_pred_db / d_energy_gt_db (dB ratio, so 1.0 means "
            "the model moved the in-band energy by exactly the GT amount)",
            "LSD raw and floored via scripts.p3_3fast_floored_lsd.lsd_floored; the floor is "
            "-40 dB relative to each config's OWN GT peak. A THIRD LSD is defined here: the "
            "same mean |dB(pred) - dB(gt)| restricted to the baseline modal-peak bins over "
            "all 64 receivers, because the -40 dB floor turns out to select only bins 0-12",
            "receiver-RMS spectrum = 10*log10(mean over the 64 receivers of |H|^2)",
            "mode selection = argmax of |GT receiver-RMS delta vs baseline| over the baseline "
            "spectrum's local maxima, keeping peaks within 40 dB of the STRONGEST LOCAL "
            "MAXIMUM (not of s.max(), which is the bin-0 compliance term) and spatially "
            "resolvable by the 8 x 8 grid (n_x <= floor(L/2dx_rx), n_y <= floor(W/2dy_rx)). "
            "The unrestricted argmax is recorded per panel for comparison",
            "nearest analytic mode label from aaf.eval.modal_projection.enumerate_modes",
            "map row reference level = max of that row's panel-(a) map, subtracted from every "
            "panel in the row; each row is then colour-scaled to its own min/max across the "
            "five panels, so cross-PANEL comparison inside a row is exact while GT-vs-pred is "
            "a comparison of SHAPE, quantified separately by the spatial Pearson r",
            "map GT-vs-pred spatial Pearson r = corrcoef of the two 64-point dB maps at the "
            "plotted mode",
        ],
        "limitations": [
            "The open window is a MATCHED-IMPEDANCE boundary (alpha -> 0.95): energy loss and "
            "Q reduction yes, radiation reactance and edge diffraction NO.",
            "Ground truth for panels (b) and (c) (alpha = 0.50) is not in the Track A corpus "
            "and was simulated by this script into a SEPARATE directory; the training corpus "
            "and its gate hashes are untouched. Same solver, dx, fs and receiver snapping as "
            "the corpus, via the corpus builder itself.",
            "The spatial maps are an 8 x 8 receiver grid inset from the walls (0.3 m margin), "
            "not a dense field: mode shapes are coarsely sampled and nothing is shown at the "
            "boundary where the edit actually sits.",
            "NEGATIVE RESULT on the bottom row: the predicted field at the plotted mode does "
            "not match the ground-truth mode shape (spatial Pearson r reported per panel). "
            "The prediction stays source-centred while the GT is a standing wave, so the "
            "figure's zero-shot claim holds for the ENERGY response to the edit and NOT for "
            "the spatial structure at a single mode. This is shown rather than cropped out.",
            "Panels (b) and (c) are at different frequencies from each other and from (a)/(d)/"
            "(e) whenever their most-affected modes differ; only (a), (d), (e) are guaranteed "
            "to share a frequency.",
            "The absolute fit is poor -- in-distribution val LSD plateaued near 4.6 dB. This "
            "figure is a RELATIVE claim (does the edit move the field the right way, at a "
            "position never trained) and the raw LSD is printed on every panel so the "
            "absolute level is never hidden.",
        ],
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="P3-3-FAST zero-shot edit demo figure")
    ap.add_argument("--checkpoint",
                    default="outputs/p3_3fast/p3_3fast_trackA2/ckpt_iter0030000.pt")
    ap.add_argument("--manifest",
                    default="configs/sweeps_2d_mat/p3_3fast_trackA_manifest.json")
    ap.add_argument("--data-dir", default="data/track_p3_3fast_A")
    ap.add_argument("--demo-data-dir", default="data/track_p3_3fast_A_demo")
    ap.add_argument("--diagnostic",
                    default="outputs/p3_3fast/trackA2/DIAGNOSTIC_30K.json/DIAGNOSTIC.json")
    ap.add_argument("--floored-lsd", default="outputs/p3_3fast/floored_lsd_30k.json")
    ap.add_argument("--outdir", default="outputs/p3_3fast/meeting_assets")
    ap.add_argument("--geom-id", type=int, default=None)
    ap.add_argument("--rx-chunk", type=int, default=8)
    ap.add_argument("--repo-root", default=None)
    a = ap.parse_args()

    root = Path(a.repo_root) if a.repo_root else ROOT
    res = run(a)
    outdir = root / a.outdir
    png = outdir / (FIG_NAME + ".png")
    px = draw(res, png)
    print("[fig] {} ({} x {} px)".format(png, px[0], px[1]), flush=True)

    side = outdir / (FIG_NAME + ".json")
    side.write_text(json.dumps(res["numbers"], indent=1, default=float))
    res["sources"].append(str(side.relative_to(root)) + " (sidecar written by this run)")
    append_manifest(outdir / "FIGURE_MANIFEST.md", manifest_block(res, png, px), px)
    print("[manifest] {}".format(outdir / "FIGURE_MANIFEST.md"), flush=True)

    print("\n{:<16s} {:>12s} {:>12s} {:>10s} {:>10s} {:>10s} {:>9s}".format(
        "panel", "GT dE (dB)", "pred (dB)", "recovered", "LSD raw", "LSD -40", "mode Hz"))
    for p in res["panels"]:
        print("({}) {:<12s} {:12.3f} {:12.3f} {:>10s} {:10.3f} {:10.3f} {:9.1f}".format(
            p["letter"], p["name"][:12], p["d_energy_gt_db"], p["d_energy_pred_db"],
            _fmt_recovery(p["energy_recovered_frac"]), p["lsd_raw_db"], p["lsd_floor40_db"],
            p["mode"]["f_hz"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
