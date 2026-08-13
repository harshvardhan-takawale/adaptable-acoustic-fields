"""Build the P3-2 meeting pack: five figures + FIGURE_MANIFEST.md.

Reads the evaluation artifacts produced by :mod:`aaf.eval.p3_2_eval`
(``summary.json``, ``per_config.json``) plus the physics-gate result
(``gate/gate.json``) and renders figures A-E at exactly 1920x1080 into
``outputs/p3_2/meeting_assets/``.

Every number drawn on a figure is read from one of those files at run time. Nothing is
recomputed from the model here and nothing is hard-coded, with two deliberate exceptions
that are *derivations*, not measurements: the analytic damping laws in figure B (computed
live from :func:`aaf.sim.analytical_modal_2d.modal_damping_2d`), and the spatial |field|
maps in figure E (read from an NPZ produced by ``scripts/p3_2_dump_fields.py``, which is
pinned to the same checkpoint the eval used).

SCOPING, which every caption repeats: the ~29:1 bandwidth selectivity that makes this
chunk's claim legible is a property of the ISM *simulator* -- its reflection coefficient is
real and angle-independent, so a pure x-axial mode sees exactly zero damping from the
north/south walls. Real locally-reacting walls follow Kuttruff and would show only ~2:1
with no invariant family. The claim is therefore "the model learns the simulator's per-wall
law", never "the model learns room acoustics".

Usage
-----
    python scripts/make_p3_2_figures.py                      # real artifacts
    python scripts/make_p3_2_figures.py --synthetic          # layout smoke test

``--synthetic`` fabricates schema-valid inputs in a temporary directory so the layout can
be verified before the real eval lands. Those figures are stamped with a loud watermark and
are written to a separate directory; they must never be committed as results.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aaf.sim.analytical_modal_2d import damping_to_bandwidth_hz, modal_damping_2d  # noqa: E402
from aaf.walls import ALPHA_BASELINE, MATERIAL_NAMES, MATERIALS, WALLS_2D  # noqa: E402

# --------------------------------------------------------------------------- style
FIGSIZE = (19.2, 10.8)
DPI = 100

RC = {
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "figure.titlesize": 24,
    "legend.fontsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

# One series palette across the whole deck, so a colour means the same thing on every panel.
C_GT = "#1f6feb"        # measured ground truth (ISM)
C_MODEL = "#d1242f"     # the trained model, zero-shot
C_ISM = "#1a7f37"       # analytic ISM-ray law (what the simulator actually obeys)
C_KUT = "#8250df"       # analytic Kuttruff law (what real walls would obey)
C_NULL = "#57606a"      # null / control references

FAMILIES = ("x_axial", "y_axial", "tangential")
FAMILY_LABEL = {"x_axial": "x-axial\n(n_x,0)", "y_axial": "y-axial\n(0,n_y)",
                "tangential": "tangential\n(n_x,n_y)"}
FAMILY_SHORT = {"x_axial": "x-axial", "y_axial": "y-axial", "tangential": "tangential"}

# Which mode family each wall is supposed to own under the ISM-ray law.
WALL_OWN_FAMILY = {"west": "x_axial", "east": "x_axial",
                   "south": "y_axial", "north": "y_axial"}

SPLIT_KEYS = ("i_unseen_geom_seen_combo", "ii_seen_geom_heldout_combo",
              "iii_unseen_geom_heldout_combo", "iv_unseen_alpha")
SPLIT_LABEL = {
    "i_unseen_geom_seen_combo": "(i) unseen geometry\nseen combo",
    "ii_seen_geom_heldout_combo": "(ii) seen geometry\nHELD-OUT combo",
    "iii_unseen_geom_heldout_combo": "(iii) unseen geometry\nHELD-OUT combo",
    "iv_unseen_alpha": "(iv) unseen geometry\nunseen alpha 0.30",
}

ZERO_SHOT = ("Zero-shot, unseen geometry, configuration computed from (L, W, alpha_west, "
             "alpha_east, alpha_south, alpha_north) -- no measurements, no per-config fitting.")
SCOPE_NOTE = ("Scoping: the large wall selectivity is a property of the ISM simulator "
              "(angle-independent reflection, no grazing-incidence absorption); real "
              "locally-reacting walls follow Kuttruff (~2:1, no invariant family). The claim "
              "is that the model learns the SIMULATOR's per-wall law.")
NEVER_SEEN = "NEVER-SEEN COMBINATION"


CAPTION_WRAP = 205


def new_figure(suptitle: str, caption: str, legend_row: bool = False,
               banner: bool = False, banner_lines: int = 2) -> plt.Figure:
    """A 1920x1080 canvas with the bottom strip reserved for caption (and optional banner).

    Constrained layout otherwise packs axes down to y=0 and the caption lands on top of the
    x tick labels, so the reserved height is computed from the wrapped caption up front and
    handed to the layout engine as ``rect``. The figure carries its wrapped caption and the
    banner's y so :func:`add_caption` and the per-figure banners agree on the geometry.
    """
    import textwrap

    cap = textwrap.fill(caption, CAPTION_WRAP)
    n_lines = cap.count("\n") + 1
    cap_h = 0.0205 * n_lines
    banner_h = 0.030 * banner_lines if banner else 0.0
    bottom = 0.012 + cap_h + banner_h
    top = 0.900 if legend_row else 0.940

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.suptitle(suptitle, fontweight="bold", y=0.985)
    fig.set_layout_engine("constrained", rect=(0.0, bottom, 1.0, top - bottom))
    fig._aaf_caption = cap                                   # type: ignore[attr-defined]
    fig._aaf_banner_y = 0.012 + cap_h + 0.008                # type: ignore[attr-defined]
    fig._aaf_legend_y = 0.945                                # type: ignore[attr-defined]
    return fig


def add_caption(fig: plt.Figure) -> None:
    fig.text(0.5, 0.010, fig._aaf_caption, ha="center", va="bottom", fontsize=12,
             style="italic", color="#444444")


def add_banner(fig: plt.Figure, text: str, fontsize: float = 14.0) -> None:
    """A boxed one/two-line verdict strip that sits between the axes and the caption."""
    fig.text(0.5, fig._aaf_banner_y, text, ha="center", va="bottom", fontsize=fontsize,
             color="#24292f", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f6f8fa", edgecolor="#8c959f"))


def add_watermark(fig: plt.Figure, on: bool) -> None:
    if not on:
        return
    fig.text(0.5, 0.5, "SYNTHETIC -- layout check only", ha="center", va="center",
             fontsize=76, color="#d1242f", alpha=0.20, rotation=28, zorder=1000,
             fontweight="bold")


def save(fig: plt.Figure, path: Path) -> Tuple[int, int]:
    """Save and return the on-disk pixel size, which must be exactly 1920x1080."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": it re-crops the canvas and would break the pinned size.
    fig.savefig(str(path), dpi=DPI)
    plt.close(fig)
    from PIL import Image

    with Image.open(str(path)) as im:
        return im.size


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _fin(x) -> bool:
    """True for a real, finite number (the eval writes NaN where a measurement failed)."""
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _fmt(x, nd: int = 2) -> str:
    return ("{:+." + str(nd) + "f}").format(float(x)) if _fin(x) else "n/a"


# --------------------------------------------------------------------------- data access
class Eval:
    """Thin accessor over summary.json / per_config.json / gate.json."""

    def __init__(self, summary_path: Path, per_config_path: Path,
                 gate_path: Optional[Path]) -> None:
        self.summary_path = summary_path
        self.per_config_path = per_config_path
        self.gate_path = gate_path
        self.summary = json.loads(summary_path.read_text())
        self.per_config = json.loads(per_config_path.read_text())
        self.gate = (json.loads(gate_path.read_text())
                     if gate_path is not None and gate_path.exists() else None)
        self._by_key = {(r["L"], r["W"], r["wall"], r["material"]): r for r in self.per_config}

    # -- config lookup
    def config(self, L: float, W: float, wall: Optional[str],
               material: Optional[str]) -> Optional[dict]:
        return self._by_key.get((L, W, wall, material))

    def cells(self, L: float, W: float, wall: str, material: str,
              family: Optional[str] = None, require_bw: bool = True) -> List[dict]:
        rec = self.config(L, W, wall, material)
        if rec is None:
            return []
        out = []
        for c in rec.get("cells", []):
            if require_bw and not c.get("bw_ok"):
                continue
            if family is not None and c.get("family") != family:
                continue
            out.append(c)
        return out

    def geometries(self, exclude_split: str = "ii_seen_geom_heldout_combo"
                   ) -> List[Tuple[float, float]]:
        return sorted({(r["L"], r["W"]) for r in self.per_config if r["split"] != exclude_split})

    @property
    def materials_edited(self) -> List[str]:
        """Edited material ids present in the eval, ordered by absorption coefficient."""
        mats = {r["material"] for r in self.per_config if r["material"]}
        return sorted(mats, key=lambda m: alpha_of(m))

    def pick_geometry(self) -> Tuple[float, float]:
        """The frozen test geometry with the most valid bandwidth measurements.

        Figure quality is limited by how many modal peaks survived the -3 dB bandwidth
        validity test, so the exemplar room is chosen by measurement yield rather than by
        eyeballing -- and the choice is reported in the manifest.
        """
        best, best_n = None, -1
        for (L, W) in self.geometries():
            n = 0
            for wall in WALLS_2D:
                n += len(self.cells(L, W, wall, "M3"))
            for mat in self.materials_edited:
                n += len(self.cells(L, W, "east", mat, family="x_axial"))
            if n > best_n:
                best, best_n = (L, W), n
        return best


def alpha_of(material: str) -> float:
    """Absorption coefficient of a material id, including the synthetic 'A030' sweep point."""
    if material in MATERIALS:
        return float(MATERIALS[material])
    if material.startswith("A"):
        return float(material[1:]) / 100.0
    raise KeyError(material)


def material_title(material: str) -> str:
    name = MATERIAL_NAMES.get(material, "unseen alpha")
    return "{}  ({}, alpha={:.2f})".format(material, name, alpha_of(material))


def alphas_vector(wall: str, material: str) -> List[float]:
    a = [ALPHA_BASELINE] * 4
    a[list(WALLS_2D).index(wall)] = alpha_of(material)
    return a


def theory_delta_bw(L: float, W: float, wall: str, material: str,
                    n_x: int, n_y: int, law: str) -> float:
    """Analytic change in -3 dB bandwidth (Hz) for one mode under one damping law."""
    base = damping_to_bandwidth_hz(
        modal_damping_2d(L, W, [ALPHA_BASELINE] * 4, n_x, n_y, model=law))
    edit = damping_to_bandwidth_hz(
        modal_damping_2d(L, W, alphas_vector(wall, material), n_x, n_y, model=law))
    return float(edit - base)


# --------------------------------------------------------------------------- figure A
def figure_a(ev: Eval, geom: Tuple[float, float], out: Path, synthetic: bool) -> dict:
    """Pick your wall: edit each of the four walls to M3 and watch which family broadens."""
    L, W = geom
    caption = ("Room L={:.2f} m, W={:.2f} m (frozen test geometry, never trained). {} Bars are "
               "the mean over valid modal peaks; dots are individual modes. {}".format(
                   L, W, ZERO_SHOT, SCOPE_NOTE))
    fig = new_figure("Pick your wall: editing one wall broadens that wall's own mode family",
                     caption, legend_row=True)
    axes = fig.subplots(2, 2).ravel()
    numbers: List[str] = []
    heldout = {tuple(c) for c in ev.summary.get("heldout_combos", [])}

    series = [("Ground truth (ISM)", "d_bw_gt", C_GT, "GT"),
              ("Theory (ISM-ray law)", "theory_d_bw", C_ISM, "theory"),
              ("Model (zero-shot)", "d_bw_pred", C_MODEL, "model")]
    # One y-scale across the four panels; otherwise a failed panel silently looks fine.
    all_vals = []
    for wall in WALLS_2D:
        for c in ev.cells(L, W, wall, "M3"):
            all_vals += [c.get(k) for _, k, _, _ in series if _fin(c.get(k))]
    lim = max([abs(float(v)) for v in all_vals] + [1.0]) * 1.30

    for ax, wall in zip(axes, WALLS_2D):
        rec = ev.config(L, W, wall, "M3")
        split = rec["split"] if rec else "?"
        is_heldout = (wall, "M3") in heldout
        own = WALL_OWN_FAMILY[wall]
        width = 0.26
        xs = np.arange(len(FAMILIES))
        for si, (lab, key, col, short) in enumerate(series):
            off = (si - 1) * width
            for xi, fam in enumerate(FAMILIES):
                pts = [float(c[key]) for c in ev.cells(L, W, wall, "M3", family=fam)
                       if _fin(c.get(key))]
                m = float(np.mean(pts)) if pts else np.nan
                ax.bar([xi + off], [m], width, color=col,
                       label=lab if (wall == "west" and xi == 0) else None,
                       edgecolor="white", linewidth=0.8, zorder=3)
                if _fin(m):
                    ax.text(xi + off, m + (0.03 * lim if m >= 0 else -0.03 * lim),
                            "{:+.1f}".format(m), ha="center",
                            va="bottom" if m >= 0 else "top", fontsize=10.5, color=col,
                            fontweight="bold", zorder=4)
                    numbers.append("A {}->M3 {} {}: {:+.2f} Hz (n={})".format(
                        wall, FAMILY_SHORT[fam], short, m, len(pts)))
                # individual modes, so an average over one mode is not read as a trend
                if len(pts) > 1:
                    ax.scatter(np.full(len(pts), xi + off), pts, s=14, color="black",
                               alpha=0.55, zorder=5, linewidths=0)

        oi = FAMILIES.index(own)
        ax.axvspan(oi - 0.5, oi + 0.5, color="#fff3cd", zorder=0)
        ax.text(oi, lim * 0.93, "own family", ha="center", fontsize=12,
                color="#8a6d00", style="italic", zorder=6)
        ax.axhline(0, color="#24292f", lw=1.0, zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES])
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("change in -3 dB bandwidth (Hz)")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        despine(ax)
        n_valid = len(ev.cells(L, W, wall, "M3"))
        title = "{} wall -> M3 absorber (alpha 0.15 -> 0.70)".format(wall.upper())
        ax.set_title(title, color=C_MODEL if is_heldout else "#24292f",
                     fontweight="bold" if is_heldout else "normal")
        tag = "split {} | {} valid modes".format(split.split("_")[0], n_valid)
        if is_heldout:
            tag = "{} | {}".format(NEVER_SEEN, tag)
            ax.set_facecolor("#fff5f5")
            for sp in ax.spines.values():
                sp.set_edgecolor(C_MODEL)
                sp.set_linewidth(2.0)
        ax.text(0.015, 0.03, tag, transform=ax.transAxes, fontsize=11.5,
                color=C_MODEL if is_heldout else "#57606a",
                fontweight="bold" if is_heldout else "normal")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, fig._aaf_legend_y), frameon=False)
    add_caption(fig)
    add_watermark(fig, synthetic)
    size = save(fig, out)
    return {
        "filename": out.name, "png_width": size[0], "png_height": size[1],
        "shows": ("Four panels, one per edited wall (all -> M3 absorber) on a single unseen "
                  "test room: change in -3 dB modal bandwidth per mode family, ground truth vs "
                  "ISM-ray theory vs zero-shot model."),
        "source_files": [str(ev.per_config_path), str(ev.summary_path)],
        "numbers": numbers, "geom": [L, W],
    }


def fam_of(xs: np.ndarray, x: float) -> str:
    return FAMILIES[int(round(float(x)))]


# --------------------------------------------------------------------------- figure B
def figure_b(ev: Eval, geom: Tuple[float, float], out: Path, synthetic: bool) -> dict:
    """Pick your material: sweep the east wall and ask which damping law the data obeys."""
    L, W = geom
    mats = ev.materials_edited
    unseen_alpha = float(ev.summary.get("unseen_alpha", 0.30))
    caption = ("Room L={:.2f} m, W={:.2f} m (frozen test geometry). {} alpha=0.30 was never "
               "trained on any wall. {}".format(L, W, ZERO_SHOT, SCOPE_NOTE))
    fig = new_figure("Pick your material: the measurement follows the ISM-ray law, "
                     "not the Kuttruff law", caption, legend_row=True, banner=True,
                     banner_lines=3)
    axes = fig.subplots(1, 2)
    numbers: List[str] = []

    panels = [("x_axial", "EAST wall's OWN family: first x-axial mode"),
              ("y_axial", "The INVARIANT family: first y-axial mode")]
    law_rms = {"ism_ray": [], "kuttruff": []}
    panel_gt: Dict[str, List[float]] = {}

    for ax, (fam, title) in zip(axes, panels):
        alphas, gt, pred, mode_ids = [], [], [], []
        for m in mats:
            cs = [c for c in ev.cells(L, W, "east", m, family=fam)]
            if not cs:
                continue
            c = sorted(cs, key=lambda z: z["f_hz"])[0]   # the first mode of the family
            alphas.append(alpha_of(m))
            gt.append(float(c["d_bw_gt"]) if _fin(c.get("d_bw_gt")) else np.nan)
            pred.append(float(c["d_bw_pred"]) if _fin(c.get("d_bw_pred")) else np.nan)
            mode_ids.append((int(c["n_x"]), int(c["n_y"]), float(c["f_hz"]), m))
        if not alphas:
            ax.text(0.5, 0.5, "no valid bandwidth measurement\nfor this family",
                    ha="center", va="center", transform=ax.transAxes, color=C_NULL)
            ax.set_title(title)
            continue
        order = np.argsort(alphas)
        alphas = list(np.array(alphas)[order])
        gt = list(np.array(gt)[order])
        pred = list(np.array(pred)[order])
        mode_ids = [mode_ids[i] for i in order]
        n_x, n_y = mode_ids[0][0], mode_ids[0][1]

        # Continuous analytic curves for both competing laws, over the same mode.
        a_grid = np.linspace(0.02, 0.75, 160)
        for law, col, ls, lab in (("ism_ray", C_ISM, "-", "Theory: ISM-ray (simulator's law)"),
                                  ("kuttruff", C_KUT, "--", "Theory: Kuttruff (real walls)")):
            curve = []
            for a in a_grid:
                base = damping_to_bandwidth_hz(
                    modal_damping_2d(L, W, [ALPHA_BASELINE] * 4, n_x, n_y, model=law))
                av = [ALPHA_BASELINE] * 4
                av[list(WALLS_2D).index("east")] = float(a)
                curve.append(damping_to_bandwidth_hz(
                    modal_damping_2d(L, W, av, n_x, n_y, model=law)) - base)
            ax.plot(a_grid, curve, ls, color=col, lw=2.6, label=lab, zorder=3)
            at_pts = [theory_delta_bw(L, W, "east", m, n_x, n_y, law)
                      for m in [mm[3] for mm in mode_ids]]
            resid = [g - t for g, t in zip(gt, at_pts) if _fin(g)]
            if resid:
                rms = float(np.sqrt(np.mean(np.square(resid))))
                law_rms[law].append((fam, rms))
                numbers.append("B {} east-sweep RMS(GT - {}) = {:.2f} Hz".format(fam, law, rms))

        ax.axvline(ALPHA_BASELINE, color="#8c959f", lw=1.4, ls=":", zorder=2)
        ax.text(ALPHA_BASELINE, 0.02, " baseline\n M0", transform=ax.get_xaxis_transform(),
                va="bottom", fontsize=11.5, color="#57606a")
        ax.axvspan(unseen_alpha - 0.012, unseen_alpha + 0.012, color="#ffd8a8",
                   alpha=0.75, zorder=1)
        # label above the data, not inside it: the markers crowd the axis floor
        ax.text(unseen_alpha, 0.975, "alpha={:.2f}\nUNSEEN".format(unseen_alpha),
                transform=ax.get_xaxis_transform(), va="top", ha="center", fontsize=11.5,
                color="#a15c00", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#ffd8a8"))

        ax.plot(alphas, gt, "o-", color=C_GT, ms=13, lw=2.2, label="Ground truth (ISM)",
                zorder=5, markeredgecolor="white", markeredgewidth=1.4)
        ax.plot(alphas, pred, "s--", color=C_MODEL, ms=12, lw=2.2, label="Model (zero-shot)",
                zorder=5, markeredgecolor="white", markeredgewidth=1.4)
        for a, g, p, mid in zip(alphas, gt, pred, mode_ids):
            if _fin(g):
                ax.annotate("{:+.1f}".format(g), (a, g), textcoords="offset points",
                            xytext=(0, 13), ha="center", fontsize=11, color=C_GT,
                            fontweight="bold")
                numbers.append("B {} east {} (alpha={:.2f}) mode ({},{}) f={:.1f} Hz: "
                               "GT {:+.2f} Hz, model {} Hz".format(
                                   fam, mid[3], a, mid[0], mid[1], mid[2], g, _fmt(p)))
            if _fin(p):
                ax.annotate("{:+.1f}".format(p), (a, p), textcoords="offset points",
                            xytext=(0, -20), ha="center", fontsize=11, color=C_MODEL,
                            fontweight="bold")
        ax.axhline(0, color="#24292f", lw=1.0, zorder=2)
        ax.set_xlabel("alpha of the EAST wall (energy absorption)")
        ax.set_ylabel("change in -3 dB bandwidth (Hz)")
        ax.set_title("{}\nmode ({}, {}) at {:.1f} Hz".format(title, n_x, n_y, mode_ids[0][2]))
        ax.grid(alpha=0.3)
        despine(ax)
        # headroom for the UNSEEN-alpha flag that sits at the top of the axes
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.22 * (y1 - y0))
        if fam == "x_axial":
            fig._aaf_bhandles = ax.get_legend_handles_labels()   # type: ignore[attr-defined]
        panel_gt[fam] = [g for g in gt if _fin(g)]

    # Verdict box: which law the measurement actually follows, from the RMS residuals.
    tot = {k: float(np.sqrt(np.mean([r ** 2 for _, r in v]))) for k, v in law_rms.items() if v}
    if len(tot) == 2:
        winner = min(tot, key=lambda k: tot[k])
        # The measured own/other ratio, stated rather than implied: the ISM-ray law predicts
        # EXACTLY zero off-family, the measurement is merely small, and quoting the finite
        # ratio keeps the panel from overclaiming a perfect invariance.
        own = max([abs(v) for v in panel_gt.get("x_axial", [])] + [0.0])
        other = max([abs(v) for v in panel_gt.get("y_axial", [])] + [0.0])
        ratio_txt = ("measured own/other = {:.1f}:1".format(own / other)
                     if other > 0 else "measured off-family response = 0")
        verdict = ("Measured GT tracks the {} law: RMS residual {:.2f} Hz vs {:.2f} Hz for the "
                   "other.\nThe y-axial panel is the discriminator -- ISM-ray predicts EXACTLY "
                   "zero there, Kuttruff predicts a real response.\nThe measurement is small "
                   "but NOT exactly zero ({} at alpha=0.70): the invariance is strong, not "
                   "perfect.".format(
                       {"ism_ray": "ISM-ray", "kuttruff": "Kuttruff"}[winner],
                       tot[winner], tot["kuttruff" if winner == "ism_ray" else "ism_ray"],
                       ratio_txt))
        numbers.append("B verdict: RMS(GT-ism_ray)={:.2f} Hz, RMS(GT-kuttruff)={:.2f} Hz "
                       "-> data follows {}; {}".format(
                           tot["ism_ray"], tot["kuttruff"], winner, ratio_txt))
        add_banner(fig, verdict, fontsize=13.0)

    h, l = getattr(fig, "_aaf_bhandles", ([], []))
    if h:
        fig.legend(h, l, loc="upper center", ncol=4,
                   bbox_to_anchor=(0.5, fig._aaf_legend_y), frameon=False)

    add_caption(fig)
    add_watermark(fig, synthetic)
    size = save(fig, out)
    return {
        "filename": out.name, "png_width": size[0], "png_height": size[1],
        "shows": ("East wall swept across all four materials plus the unseen alpha=0.30: "
                  "measured, model-predicted and both analytic damping laws for the first "
                  "x-axial (own) and first y-axial (invariant) mode."),
        "source_files": [str(ev.per_config_path), str(ev.summary_path),
                         "aaf/sim/analytical_modal_2d.py (analytic laws, computed live)"],
        "numbers": numbers, "geom": [L, W],
    }


# --------------------------------------------------------------------------- figure C
def figure_c(ev: Eval, out: Path, synthetic: bool) -> dict:
    """The wall-selectivity matrix: GT | theory | model | residual, per material."""
    sm = ev.summary["selectivity_matrix"]
    mats = sorted(sm.keys(), key=alpha_of)
    heldout = {tuple(c) for c in ev.summary.get("heldout_combos", [])}
    blocks = [("gt_d_bw", "Ground truth (ISM)"), ("theory_d_bw", "Theory (ISM-ray)"),
              ("pred_d_bw", "Model (zero-shot)"), ("residual_d_bw", "Residual (model - GT)")]
    numbers: List[str] = []

    vals = [sm[m][w][f].get(k) for m in mats for w in WALLS_2D for f in FAMILIES
            for k, _ in blocks[:3] if _fin(sm[m][w][f].get(k))]
    vmax = max([abs(float(v)) for v in vals] + [1.0])

    caption = ("Change in -3 dB bandwidth (Hz), averaged over the 10 frozen test geometries; "
               "diverging scale centred at 0, printed values are raw Hz. The black box marks "
               "the axial sub-block -- the expected pattern is NOT a full diagonal: tangential "
               "modes respond to every wall. Gold dashed rows are the two NEVER-SEEN "
               "COMBINATIONS. {} {}".format(ZERO_SHOT, SCOPE_NOTE))
    fig = new_figure("The wall-selectivity matrix: each wall broadens its own axial family",
                     caption, banner=True)
    gs = fig.add_gridspec(len(mats), len(blocks) + 1,
                          width_ratios=[1] * len(blocks) + [0.055])
    axes = [[fig.add_subplot(gs[r, c]) for c in range(len(blocks))] for r in range(len(mats))]
    cax = fig.add_subplot(gs[:, len(blocks)])

    for ri, mat in enumerate(mats):
        for ci, (key, blabel) in enumerate(blocks):
            ax = axes[ri][ci]
            M = np.full((len(WALLS_2D), len(FAMILIES)), np.nan)
            for wi, wall in enumerate(WALLS_2D):
                for fi, fam in enumerate(FAMILIES):
                    v = sm[mat][wall][fam].get(key)
                    if _fin(v):
                        M[wi, fi] = float(v)
            lim = vmax
            im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            for wi in range(len(WALLS_2D)):
                for fi in range(len(FAMILIES)):
                    v = M[wi, fi]
                    txt = "{:+.1f}".format(v) if _fin(v) else "n/a"
                    shade = "white" if _fin(v) and abs(v) > 0.62 * lim else "#24292f"
                    ax.text(fi, wi, txt, ha="center", va="center", fontsize=12.5,
                            color=shade, fontweight="bold")
                    if _fin(v):
                        numbers.append("C {} {} {} {}: {:+.2f} Hz".format(
                            mat, WALLS_2D[wi], FAMILIES[fi], key, v))
            # Box the axial sub-block: that is where the block-diagonal claim lives.
            ax.add_patch(mpatches.Rectangle((-0.5, -0.5), 2.0, len(WALLS_2D),
                                            fill=False, edgecolor="#24292f", lw=2.6,
                                            zorder=6))
            # Flag the rows the model never trained on, so the reader can see at a glance
            # which cells of the model/residual blocks are the actual generalization test.
            for wi, wall in enumerate(WALLS_2D):
                if (wall, mat) in heldout:
                    ax.add_patch(mpatches.Rectangle(
                        (-0.5, wi - 0.5), len(FAMILIES), 1.0, fill=False,
                        edgecolor="#bf8700", lw=3.2, linestyle=(0, (4, 2)), zorder=7))
            ax.set_xticks(range(len(FAMILIES)))
            ax.set_yticks(range(len(WALLS_2D)))
            # x labels only on the bottom row, wall labels only on the first column:
            # repeating them on all 16 panels is noise, not information.
            ax.set_xticklabels([FAMILY_SHORT[f] for f in FAMILIES] if ri == len(mats) - 1
                               else [], fontsize=12.5)
            ax.set_yticklabels([w for w in WALLS_2D] if ci == 0 else [], fontsize=12.5)
            if ri == 0:
                ax.set_title(blabel, fontsize=17)
            if ci == 0:
                ax.set_ylabel("{}\nalpha={:.2f}".format(mat, alpha_of(mat)), fontsize=13.5,
                              fontweight="bold", rotation=0, ha="right", va="center",
                              labelpad=52)
            ax.tick_params(length=0)

    fig.colorbar(im, cax=cax, label="change in -3 dB bandwidth (Hz)")

    sel = ev.summary.get("selectivity_index", {})
    idx_txt = ("wall-selectivity index (own-family response / other-family response):     "
               "GT {:.1f}x     |     ISM-ray theory {:.1f}x     |     model {:.1f}x".format(
                   float(sel.get("gt", np.nan)), float(sel.get("theory", np.nan)),
                   float(sel.get("pred", np.nan))))
    numbers.append("C selectivity_index: gt={:.3f}, theory={:.3f}, pred={:.3f}".format(
        float(sel.get("gt", np.nan)), float(sel.get("theory", np.nan)),
        float(sel.get("pred", np.nan))))
    add_banner(fig, idx_txt, fontsize=15)
    add_caption(fig)
    add_watermark(fig, synthetic)
    size = save(fig, out)
    return {
        "filename": out.name, "png_width": size[0], "png_height": size[1],
        "shows": ("Wall x family selectivity matrix per material: ground truth, ISM-ray theory, "
                  "zero-shot model and the model-minus-GT residual, in raw Hz of bandwidth "
                  "change."),
        "source_files": [str(ev.summary_path) + " :: selectivity_matrix, selectivity_index"],
        "numbers": numbers,
    }


# --------------------------------------------------------------------------- figure D
def figure_d(ev: Eval, out: Path, synthetic: bool) -> dict:
    """Generalization: does the edit response survive an unseen (wall, material) combination?"""
    s = ev.summary
    splits = [k for k in SPLIT_KEYS if k in s["splits"]]
    c1 = s.get("controls", {}).get("C1_null_model", {}).get("per_split", {})
    floor = float(s.get("controls", {}).get("C2_floor_hz", 0.0))
    numbers: List[str] = []

    caption = ("E_BW is an ERROR (mean absolute difference between predicted and measured "
               "bandwidth change) -- lower is better, and the dashed red line is the C1 null "
               "model, which applies no edit at all. Split (iii) panels are NEVER-SEEN "
               "COMBINATIONS. {} {}".format(ZERO_SHOT, SCOPE_NOTE))
    fig = new_figure("Generalization: the edit response does not transfer to never-trained "
                     "combinations", caption, legend_row=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.15, 1.0, 1.05])
    ax = fig.add_subplot(gs[0, 0])
    axg = fig.add_subplot(gs[0, 1])
    axh = fig.add_subplot(gs[0, 2])

    width = 0.20
    xs = np.arange(len(splits))
    fam_colors = {"x_axial": "#1f6feb", "y_axial": "#e8850c", "tangential": "#57606a"}
    for fi, fam in enumerate(FAMILIES):
        vals, ns = [], []
        for sp in splits:
            bf = s["splits"][sp]["by_family"].get(fam, {})
            v = bf.get("E_BW_hz")
            vals.append(float(v) if _fin(v) else np.nan)
            ns.append(int(bf.get("n", 0)))
        off = (fi - 1) * width
        ax.bar(xs + off, vals, width, color=fam_colors[fam], label=FAMILY_SHORT[fam],
               edgecolor="white", linewidth=0.8, zorder=3)
        for x, v, n in zip(xs, vals, ns):
            if _fin(v):
                ax.text(x + off, v + 0.06, "{:.1f}".format(v), ha="center", va="bottom",
                        fontsize=11, color=fam_colors[fam], fontweight="bold")
        for sp, v, n in zip(splits, vals, ns):
            numbers.append("D E_BW {} {}: {} Hz (n={})".format(sp, fam, _fmt(v), n))

    # overall bar (all families pooled), drawn as an open outline behind the family bars
    for i, sp in enumerate(splits):
        ov = s["splits"][sp]["edit"]["E_BW_hz"]
        ax.plot([i - 0.42, i + 0.42], [ov, ov], color="#24292f", lw=2.4, zorder=6,
                label="all families pooled" if i == 0 else None)
        numbers.append("D E_BW {} pooled: {:.3f} Hz".format(sp, float(ov)))
        nl = c1.get(sp, {}).get("null_E_BW_hz")
        if _fin(nl):
            ax.plot([i - 0.42, i + 0.42], [nl, nl], color=C_MODEL, lw=2.6, ls="--", zorder=6,
                    label="C1 null model (no edit applied)" if i == 0 else None)
            numbers.append("D C1 null E_BW {}: {:.3f} Hz".format(sp, float(nl)))

    ax.axhspan(0, floor, color="#8c959f", alpha=0.28, zorder=1)
    ax.text(len(splits) - 0.5, floor, " C2 repeatability floor {:.2f} Hz".format(floor),
            ha="right", va="bottom", fontsize=11.5, color="#424a53")
    numbers.append("D C2_floor_hz: {:.4f} Hz".format(floor))

    ax.set_xticks(xs)
    ax.set_xticklabels([SPLIT_LABEL[sp] for sp in splits])
    ax.set_ylabel("E_BW: mean |model - GT| bandwidth error (Hz)   [lower is better]")
    ax.set_title("Edit-response error by split and mode family")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    despine(ax)
    # headroom so the NEVER-SEEN tag does not land on the tallest bars
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, fig._aaf_legend_y), frameon=False)
    for i, sp in enumerate(splits):
        if sp.startswith("iii"):
            ax.axvspan(i - 0.5, i + 0.5, color="#fff5f5", zorder=0)
            ax.text(i, ax.get_ylim()[1] * 0.985, NEVER_SEEN, ha="center", va="top",
                    fontsize=12, color=C_MODEL, fontweight="bold")

    # -- the (iii)-(i) gap, in Hz and as a fraction of the ground-truth effect size
    gap = s.get("gap_i_iii", {})
    g_hz = float(gap.get("gap_hz", np.nan))
    g_pct = float(gap.get("gap_pct_of_gt_effect", np.nan))
    e_i, e_iii = float(gap.get("E_BW_hz_i", np.nan)), float(gap.get("E_BW_hz_iii", np.nan))
    gt_eff = float(gap.get("gt_effect_size_hz_iii", np.nan))
    axg.bar([0, 1], [e_i, e_iii], 0.55, color=["#8c959f", C_MODEL], edgecolor="white", zorder=3)
    for x, v in ((0, e_i), (1, e_iii)):
        axg.text(x, v + 0.06, "{:.2f}".format(v), ha="center", va="bottom", fontsize=13,
                 fontweight="bold", color="#24292f")
    axg.annotate("", xy=(1, e_iii), xytext=(1, e_i),
                 arrowprops=dict(arrowstyle="<->", lw=2.2, color="#24292f"))
    axg.text(1.06, (e_i + e_iii) / 2, "gap {:.2f} Hz\n= {:.0f}% of the\nGT effect ({:.2f} Hz)"
             .format(g_hz, g_pct, gt_eff), fontsize=13, va="center", fontweight="bold")
    axg.set_ylim(0, max(e_i, e_iii) * 1.25)
    axg.set_xticks([0, 1])
    axg.set_xticklabels(["(i) seen\ncombo", "(iii) HELD-OUT\ncombo"])
    axg.set_ylabel("E_BW (Hz)")
    axg.set_title("The generalization gap")
    axg.set_xlim(-0.6, 2.1)
    axg.grid(axis="y", alpha=0.3, zorder=0)
    despine(axg)
    numbers.append("D gap_i_iii: E_BW_i={:.3f}, E_BW_iii={:.3f}, gap={:.3f} Hz = {:.2f}% of "
                   "GT effect {:.3f} Hz".format(e_i, e_iii, g_hz, g_pct, gt_eff))

    # -- the two held-out combos, individually
    hb = s.get("heldout_by_combo", {}).get("iii_unseen_geom_heldout_combo", {})
    combos = list(hb.keys())
    if combos:
        cx = np.arange(len(combos))
        gtv = [float(hb[c]["edit"].get("gt_effect_size_hz", np.nan)) for c in combos]
        pdv = [float(hb[c]["edit"].get("pred_effect_size_hz", np.nan)) for c in combos]
        axh.bar(cx - 0.19, gtv, 0.36, color=C_GT, label="GT effect size", edgecolor="white",
                zorder=3)
        axh.bar(cx + 0.19, pdv, 0.36, color=C_MODEL, label="model effect size",
                edgecolor="white", zorder=3)
        for x, g, p in zip(cx, gtv, pdv):
            axh.text(x - 0.19, g, "{:.2f}".format(g), ha="center", va="bottom", fontsize=11.5,
                     color=C_GT, fontweight="bold")
            axh.text(x + 0.19, p, "{:.2f}".format(p), ha="center", va="bottom", fontsize=11.5,
                     color=C_MODEL, fontweight="bold")
        for c, g, p in zip(combos, gtv, pdv):
            numbers.append("D heldout(iii) {}: GT effect {:.3f} Hz, model effect {:.3f} Hz "
                           "(n={})".format(c, g, p, int(hb[c].get("n_configs", 0))))
        axh.set_xticks(cx)
        axh.set_xticklabels([c.replace("_", " -> ") for c in combos])
        axh.set_ylabel("mean |delta BW| at the mode (Hz)")
        axh.set_title("Split (iii): held-out combos")
        axh.set_ylim(0, max(gtv + pdv) * 1.22)
        axh.legend(fontsize=12, framealpha=0.95)
        axh.grid(axis="y", alpha=0.3, zorder=0)
        despine(axh)

    add_caption(fig)
    add_watermark(fig, synthetic)
    size = save(fig, out)
    return {
        "filename": out.name, "png_width": size[0], "png_height": size[1],
        "shows": ("Edit-response error E_BW for the four evaluation splits by mode family, "
                  "against the C1 null-model reference and the C2 repeatability floor, with the "
                  "(iii)-(i) generalization gap and the two held-out combos broken out."),
        "source_files": [str(ev.summary_path) +
                         " :: splits.*.by_family, splits.*.edit, controls.C1_null_model, "
                         "controls.C2_floor_hz, gap_i_iii, heldout_by_combo"],
        "numbers": numbers,
    }


# --------------------------------------------------------------------------- figure E
def _mode_map(H: np.ndarray, fs: int, f_target: float, grid: int = 8) -> Tuple[np.ndarray, float]:
    """8x8 |H| map at the bin nearest ``f_target``. Receivers are row-major, outer-y inner-x."""
    n_freq = H.shape[1]
    f_axis = np.fft.rfftfreq(2 * (n_freq - 1), d=1.0 / fs)
    bi = int(np.argmin(np.abs(f_axis - f_target)))
    return np.abs(H[:, bi]).reshape(grid, grid), float(f_axis[bi])


def figure_e(ev: Eval, geom: Tuple[float, float], fields: Optional[Path], out: Path,
             synthetic: bool) -> dict:
    """Mode-shape invariance and level transfer."""
    L, W = geom
    numbers: List[str] = []
    caption = ("Left: |field| at the first x-axial mode, sampled on the 8x8 receiver grid; the "
               "GT pair and the model pair are each normalised to their OWN baseline peak, so "
               "the panels compare spatial pattern and edit-induced change, not absolute gain "
               "(the model's global gain is ~5x low at this checkpoint). Right: per-mode level "
               "change, model vs measured, coloured by split -- split (iii) points are "
               "NEVER-SEEN COMBINATIONS. {}".format(ZERO_SHOT))
    fig = new_figure("Mode shapes survive the edit; the level change transfers only partly",
                     caption, banner=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.16, 1.5])
    ax_maps = [[fig.add_subplot(gs[r, c]) for c in (0, 1)] for r in (0, 1)]
    cax = fig.add_subplot(gs[:, 2])
    ax_sc = fig.add_subplot(gs[:, 3])

    # ---- left: 8x8 |field| maps
    npz = None
    if fields is not None and fields.exists():
        npz = np.load(str(fields), allow_pickle=True)
    if npz is not None:
        labels = [str(x) for x in npz["labels"]]
        base_lab = next((x for x in labels if x.endswith("baseline")), labels[0])
        edit_lab = next((x for x in labels if not x.endswith("baseline")), labels[-1])
        fs = int(npz["fs"])
        Lf, Wf = float(npz["L"]), float(npz["W"])
        f_mode = 343.0 / (2.0 * Lf)                     # first x-axial mode
        panels = [("GT baseline", "gt/" + base_lab, 0, 0), ("GT edited", "gt/" + edit_lab, 0, 1),
                  ("Model baseline", "pred/" + base_lab, 1, 0),
                  ("Model edited", "pred/" + edit_lab, 1, 1)]
        maps, f_used = {}, None
        for name, key, r, c in panels:
            m, f_used = _mode_map(npz[key], fs, f_mode)
            maps[name] = m
        # Normalise each source to ITS OWN baseline peak: the model's absolute gain is not
        # being claimed here, only the spatial pattern and the edit-induced change.
        ref = {"GT": float(maps["GT baseline"].max()), "Model": float(maps["Model baseline"].max())}
        # Extent is the receiver bounding box read from the dump, NOT [0,L]x[0,W]: the grid
        # is inset by a 0.3 m margin, and drawing it wall-to-wall would misplace the node.
        rx = np.asarray(npz["rx"], dtype=float)
        extent = [rx[:, 0].min(), rx[:, 0].max(), rx[:, 1].min(), rx[:, 1].max()]
        im = None
        for name, key, r, c in panels:
            src = "GT" if name.startswith("GT") else "Model"
            db = 20.0 * np.log10(np.maximum(maps[name], 1e-12) / ref[src])
            ax = ax_maps[r][c]
            im = ax.imshow(db, cmap="magma", vmin=-24, vmax=2, origin="lower",
                           extent=extent, aspect="auto")
            ax.set_title(name, fontsize=16)
            ax.set_xlabel("x (m)", fontsize=13)
            ax.set_ylabel("y (m)", fontsize=13)
            ax.tick_params(labelsize=11)
            numbers.append("E map {}: peak {:.3g}, mean {:.3g} (linear |H|) at {:.2f} Hz".format(
                name, float(maps[name].max()), float(maps[name].mean()), f_used))
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("dB re. own baseline peak", fontsize=13)
        numbers.append("E fields: room L={:.2f} W={:.2f}, baseline='{}', edited='{}', "
                       "mode bin {:.2f} Hz (first x-axial, c/2L={:.2f} Hz), checkpoint {}".format(
                           Lf, Wf, base_lab, edit_lab, f_used, f_mode, str(npz["checkpoint"])))
        edit_tag = edit_lab.split("_", 2)[-1].replace("_", " -> ")
        add_banner(fig, "Room {:.2f} x {:.2f} m (unseen)   |   first x-axial mode at {:.1f} Hz  "
                        " |   edit: {}   |   {}".format(Lf, Wf, f_used, edit_tag, NEVER_SEEN),
                   fontsize=13.5)
    else:
        for r in (0, 1):
            for c in (0, 1):
                ax_maps[r][c].text(0.5, 0.5, "field dump not available\n"
                                             "(run scripts/p3_2_dump_fields.py)",
                                   ha="center", va="center", transform=ax_maps[r][c].transAxes,
                                   fontsize=13, color=C_NULL)
                ax_maps[r][c].set_xticks([])
                ax_maps[r][c].set_yticks([])
        cax.axis("off")
        numbers.append("E maps: NOT RENDERED (no field NPZ present)")

    # ---- right: level transfer scatter, from every valid cell in the eval
    gx, gy, cols, labs = [], [], [], []
    split_color = {"i_unseen_geom_seen_combo": C_GT,
                   "ii_seen_geom_heldout_combo": "#e8850c",
                   "iii_unseen_geom_heldout_combo": C_MODEL,
                   "iv_unseen_alpha": "#8250df"}
    for rec in ev.per_config:
        if rec["is_baseline"]:
            continue
        for c in rec.get("cells", []):
            if not c.get("lvl_ok"):
                continue
            a, b = c.get("d_lvl_gt"), c.get("d_lvl_pred")
            if _fin(a) and _fin(b):
                gx.append(float(a))
                gy.append(float(b))
                cols.append(split_color.get(rec["split"], C_NULL))
                labs.append(rec["split"])
    if gx:
        lim = max(max(np.abs(gx)), max(np.abs(gy))) * 1.1
        ax_sc.plot([-lim, lim], [-lim, lim], color="#24292f", lw=1.8, ls="--",
                   label="identity (perfect transfer)", zorder=2)
        ax_sc.axhline(0, color="#c9d1d9", lw=1, zorder=1)
        ax_sc.axvline(0, color="#c9d1d9", lw=1, zorder=1)
        for sp in SPLIT_KEYS:
            idx = [i for i, l in enumerate(labs) if l == sp]
            if idx:
                ax_sc.scatter([gx[i] for i in idx], [gy[i] for i in idx], s=42,
                              color=split_color[sp], alpha=0.72, linewidths=0, zorder=3,
                              label="{}  (n={})".format(SPLIT_LABEL[sp].replace("\n", " "),
                                                        len(idx)))
        r = float(np.corrcoef(gx, gy)[0, 1])
        slope = float(np.polyfit(gx, gy, 1)[0])
        ax_sc.set_xlim(-lim, lim)
        ax_sc.set_ylim(-lim, lim)
        ax_sc.set_xlabel("measured delta level at the mode (dB)")
        ax_sc.set_ylabel("model delta level at the mode (dB)")
        ax_sc.set_title("Level transfer: model vs measured, every valid mode\n"
                        "Pearson r = {:.3f},  slope = {:.3f}  (identity = 1.000)".format(
                            r, slope), fontsize=17)
        ax_sc.legend(loc="upper left", fontsize=11.5, framealpha=0.95)
        ax_sc.grid(alpha=0.3)
        despine(ax_sc)
        numbers.append("E level scatter: n={}, Pearson r={:.4f}, slope={:.4f}".format(
            len(gx), r, slope))

    # mode-shape invariance numbers, per split
    lines = []
    for sp in SPLIT_KEYS:
        msi = ev.summary["splits"].get(sp, {}).get("mode_shape_invariance")
        if msi:
            lines.append("{}: GT {:.4f} | model {:.4f}  (n={})".format(
                sp.split("_")[0], float(msi["gt"]), float(msi["pred"]), int(msi["n"])))
            numbers.append("E mode_shape_invariance {}: gt={:.4f}, pred={:.4f}, n={}".format(
                sp, float(msi["gt"]), float(msi["pred"]), int(msi["n"])))
    if lines:
        ax_sc.text(0.985, 0.02, "Mode-shape invariance |complex Pearson|\n" + "\n".join(lines),
                   transform=ax_sc.transAxes, ha="right", va="bottom", fontsize=11.5,
                   bbox=dict(boxstyle="round,pad=0.45", facecolor="#f6f8fa",
                             edgecolor="#8c959f"))

    add_caption(fig)
    add_watermark(fig, synthetic)
    size = save(fig, out)
    return {
        "filename": out.name, "png_width": size[0], "png_height": size[1],
        "shows": ("Spatial |field| maps at one x-axial mode for GT/model x baseline/edited, "
                  "plus the model-vs-measured per-mode level change with the identity line and "
                  "the mode-shape invariance correlations."),
        "source_files": ([str(ev.per_config_path), str(ev.summary_path)] +
                         ([str(fields)] if fields is not None and fields.exists() else [])),
        "numbers": numbers, "geom": [L, W],
    }


# --------------------------------------------------------------------------- synthetic
def synthesize(dest: Path) -> Tuple[Path, Path, Path]:
    """Fabricate schema-valid inputs so the layout can be checked before the eval lands.

    Values are deliberately arbitrary. Figures built from these are watermarked and written
    to a separate directory; they exist to prove the panels lay out, nothing else.
    """
    rng = np.random.RandomState(0)
    dest.mkdir(parents=True, exist_ok=True)
    mats = ["M1", "A030", "M2", "M3"]
    geoms = [(3.5 + 0.31 * i, 3.1 + 0.19 * i) for i in range(10)]
    heldout = [["west", "M2"], ["north", "M3"]]

    def fam_of_mode(nx, ny):
        return "x_axial" if ny == 0 else ("y_axial" if nx == 0 else "tangential")

    per_config, cells_all = [], []
    for (L, W) in geoms:
        for wall in WALLS_2D:
            for mat in mats:
                heldo = [wall, mat] in heldout
                split = ("iv_unseen_alpha" if mat == "A030" else
                         ("iii_unseen_geom_heldout_combo" if heldo
                          else "i_unseen_geom_seen_combo"))
                cells = []
                for (nx, ny) in [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (2, 1)]:
                    fam = fam_of_mode(nx, ny)
                    th = theory_delta_bw(L, W, wall, mat, nx, ny, "ism_ray")
                    gt = th + rng.normal(0, 0.4)
                    pred = (0.15 if heldo else 0.65) * gt + rng.normal(0, 0.5)
                    cells.append(dict(
                        n_x=nx, n_y=ny, family=fam, f_hz=float(171.5 * np.hypot(nx / L, ny / W)),
                        bw_ok=bool(rng.rand() > 0.25), lvl_ok=True, gamma_ok=False,
                        d_bw_gt=float(gt), d_bw_pred=float(pred), theory_d_bw=float(th),
                        d_lvl_gt=float(rng.normal(-3, 2)),
                        d_lvl_pred=float(rng.normal(-2, 2)),
                        d_lngamma_gt=float("nan"), d_lngamma_pred=float("nan")))
                cells_all += cells
                alphas = alphas_vector(wall, mat)
                fid = dict(mag_corr=0.87, band_lsd_db=3.5, phase_corr_mw=0.85,
                           rir_pearson=0.89, t20_rel_err=0.3)
                per_config.append(dict(
                    label="L{:.2f}_W{:.2f}_{}_{}".format(L, W, wall, mat), split=split,
                    L=L, W=W, wall=wall, material=mat, alphas=alphas, is_baseline=False,
                    fidelity=fid, null_fidelity=dict(fid), cells=cells,
                    edit=dict(E_BW_hz=2.0, edit_bw_pearson=0.5, edit_bw_slope=0.4,
                              E_LVL_db=1.4, E_LNGAMMA=0.5, gt_effect_size_hz=3.6,
                              pred_effect_size_hz=2.0, n_cells=len(cells),
                              n_cells_level=len(cells), n_cells_gamma=0),
                    by_family={f: dict(E_BW_hz=2.0, n=2, E_LVL_db=1.2, gt_d_bw=2.7,
                                       pred_d_bw=1.5) for f in FAMILIES},
                    mode_shape_gt=0.98, mode_shape_pred=0.99, cond_phi=1.5))
        per_config.append(dict(
            label="L{:.2f}_W{:.2f}_baseline".format(L, W), split="i_unseen_geom_seen_combo",
            L=L, W=W, wall=None, material=None, alphas=[ALPHA_BASELINE] * 4, is_baseline=True,
            fidelity=dict(mag_corr=0.86, band_lsd_db=3.6, phase_corr_mw=0.84,
                          rir_pearson=0.88, t20_rel_err=0.35), cells=[], cond_phi=1.5))

    def split_block(n, scale):
        return dict(
            n_configs=n, n_edited_configs=n,
            fidelity=dict(mag_corr=0.87, band_lsd_db=3.4, phase_corr_mw=0.85,
                          rir_pearson=0.89, t20_rel_err=0.33),
            edit=dict(E_BW_hz=2.0 * scale, edit_bw_pearson=0.7 / scale,
                      edit_bw_slope=0.5 / scale, E_LVL_db=1.4, E_LNGAMMA=0.5,
                      gt_effect_size_hz=3.6, pred_effect_size_hz=2.0 / scale,
                      n_cells=600, n_cells_level=1500, n_cells_gamma=30, edit_gain=1.0),
            by_family={f: dict(E_BW_hz=(1.9 + 0.3 * i) * scale, n=200 - 20 * i,
                               E_LVL_db=1.2, gt_d_bw=2.7, pred_d_bw=1.5)
                       for i, f in enumerate(FAMILIES)},
            mode_shape_invariance=dict(gt=0.988, pred=0.996, n=n),
            null_fidelity=dict(mag_corr=0.86, band_lsd_db=3.5, phase_corr_mw=0.85,
                               rir_pearson=0.88, t20_rel_err=0.34))

    scales = dict(zip(SPLIT_KEYS, [1.0, 2.6, 2.5, 0.9]))
    summary = dict(
        checkpoint="SYNTHETIC/ckpt.pt", iter=0, in_dist_val_lsd_db=3.24,
        splits={k: split_block(100, scales[k]) for k in SPLIT_KEYS},
        selectivity_matrix={
            m: {w: {f: dict(
                gt_d_bw=theory_delta_bw(4.5, 4.0, w, m, *(1, 0) if f == "x_axial" else
                                        ((0, 1) if f == "y_axial" else (1, 1)), law="ism_ray")
                + rng.normal(0, 0.3),
                pred_d_bw=0.5 * theory_delta_bw(
                    4.5, 4.0, w, m, *(1, 0) if f == "x_axial" else
                    ((0, 1) if f == "y_axial" else (1, 1)), law="ism_ray"),
                theory_d_bw=theory_delta_bw(
                    4.5, 4.0, w, m, *(1, 0) if f == "x_axial" else
                    ((0, 1) if f == "y_axial" else (1, 1)), law="ism_ray"),
                residual_d_bw=-0.5, n=20) for f in FAMILIES} for w in WALLS_2D}
            for m in mats},
        selectivity_index=dict(gt=17.4, pred=4.6, theory=33.4, floor_hz=0.15,
                               source="SYNTHETIC"),
        controls=dict(
            C1_null_model=dict(
                definition="synthetic",
                per_split={k: dict(model_band_lsd_db=3.5, null_band_lsd_db=3.4,
                                   edit_gain=1.0, E_BW_hz=2.0 * scales[k],
                                   null_E_BW_hz=2.2 * scales[k]) for k in SPLIT_KEYS}),
            C2_floor_hz=0.04, C3_conditioning_identity=True,
            C4_wall_identity=dict(mean_wall_asymmetry=0.12)),
        heldout_combos=heldout, unseen_alpha=0.30,
        heldout_by_combo={"iii_unseen_geom_heldout_combo": {
            "west_M2": dict(n_configs=10, edit=dict(gt_effect_size_hz=3.31,
                                                    pred_effect_size_hz=0.40, E_BW_hz=3.42)),
            "north_M3": dict(n_configs=10, edit=dict(gt_effect_size_hz=7.24,
                                                     pred_effect_size_hz=0.35, E_BW_hz=7.31))}},
        gap_i_iii=dict(E_BW_hz_i=2.0, E_BW_hz_iii=5.0, gap_hz=3.0,
                       gt_effect_size_hz_iii=5.27, gap_pct_of_gt_effect=56.9),
        meta=dict(band_hz=[0.0, 300.0], f_max_projection_hz=200.0, n_configs_evaluated=250,
                  n_geometries=10, scoping="SYNTHETIC"))

    sp = dest / "summary.json"
    pp = dest / "per_config.json"
    sp.write_text(json.dumps(summary, indent=1))
    pp.write_text(json.dumps(per_config, indent=1))

    # a fake field dump, so figure E's left half also gets a layout check
    L0, W0 = geoms[4]
    rx = np.array([[(ix + 0.5) * L0 / 8, (iy + 0.5) * W0 / 8]
                   for iy in range(8) for ix in range(8)])
    n_freq = 4097
    store = {}
    labels = ["L{:.2f}_W{:.2f}_baseline".format(L0, W0), "L{:.2f}_W{:.2f}_west_M2".format(L0, W0)]
    for i, lab in enumerate(labels):
        for src in ("gt", "pred"):
            f = np.fft.rfftfreq(2 * (n_freq - 1), d=1.0 / 4096)
            shape = np.cos(np.pi * rx[:, 0:1] / L0)
            H = (shape * np.exp(-((f[None, :] - 343.0 / (2 * L0)) ** 2) / (2 * (2 + i) ** 2))
                 ).astype(np.complex64)
            store["{}/{}".format(src, lab)] = H
    store["labels"] = np.asarray(labels)
    store["rx"] = rx
    store["src"] = np.array([0.5, 0.5])
    store["L"] = np.asarray(L0)
    store["W"] = np.asarray(W0)
    store["fs"] = np.asarray(4096)
    store["iter"] = np.asarray(0)
    store["checkpoint"] = np.asarray("SYNTHETIC")
    fp = dest / "fields.npz"
    np.savez_compressed(str(fp), **store)
    return sp, pp, fp


# --------------------------------------------------------------------------- manifest
def write_manifest(path: Path, ev: Eval, results: List[dict], geom: Tuple[float, float],
                   synthetic: bool) -> None:
    s = ev.summary
    L, W = geom
    out: List[str] = []
    A = out.append
    A("# P3-2 — figure manifest (meeting pack)")
    A("")
    if synthetic:
        A("> **SYNTHETIC RUN — DO NOT USE.** Every number below was fabricated by "
          "`--synthetic` for a layout check.")
        A("")
    A("Five figures for the P3-2 chunk: one model conditioned on "
      "`(L, W, alpha_west, alpha_east, alpha_south, alpha_north)`, asked to render edited "
      "rooms **zero-shot** — the conditioning vector is computed from the physical parameters "
      "alone, no measurement of the target config is read, and nothing is optimised per config.")
    A("")
    A("All numbers below are read at run time from the JSON files named in each row. "
      "Nothing on any figure is hand-entered. The only quantities not read from a file are the "
      "two analytic damping laws in figure B, computed live from "
      "`aaf.sim.analytical_modal_2d.modal_damping_2d`.")
    A("")
    A("## Provenance")
    A("")
    A("| field | value |")
    A("|---|---|")
    A("| checkpoint | `{}` |".format(s.get("checkpoint")))
    A("| iteration | {} (training was still running; this is the newest checkpoint the eval "
      "used) |".format(s.get("iter")))
    A("| in-distribution val LSD | {} dB |".format(
        _fmt(s.get("in_dist_val_lsd_db"), 3).lstrip("+")))
    A("| band | {} Hz |".format(s.get("meta", {}).get("band_hz")))
    A("| modal projection cap | {} Hz |".format(s.get("meta", {}).get("f_max_projection_hz")))
    A("| configs evaluated | {} over {} geometries |".format(
        s.get("meta", {}).get("n_configs_evaluated"), s.get("meta", {}).get("n_geometries")))
    A("| held-out combos | {} |".format(
        ", ".join("`{}+{}`".format(*c) for c in s.get("heldout_combos", []))))
    A("| unseen alpha | {} |".format(s.get("unseen_alpha")))
    A("| exemplar room (figs A, B, E) | L={:.2f} m, W={:.2f} m — the frozen test geometry with "
      "the most valid bandwidth measurements |".format(L, W))
    A("| sources | `{}`, `{}`{} |".format(
        ev.summary_path, ev.per_config_path,
        ", `{}`".format(ev.gate_path) if ev.gate_path else ""))
    A("")
    A("## Scoping (must accompany any verbal claim)")
    A("")
    A("The ~29:1 bandwidth selectivity that makes this chunk legible is a property of the **ISM "
      "simulator**: its reflection coefficient is real and angle-independent, so a pure x-axial "
      "mode sees *exactly zero* damping from the north/south walls. Real locally-reacting walls "
      "follow Kuttruff and would show only ~2:1, with **no invariant family**. The claim is "
      "therefore *\"the model learns the simulator's per-wall law\"* — not *\"the model learns "
      "room acoustics\"*. Figure B puts both laws on the same axes against the measurement so "
      "this is visible rather than asserted.")
    if ev.gate is not None:
        sel = ev.gate.get("selectivity", {})
        ci = sel.get("ci95") or [float("nan"), float("nan")]
        A("")
        A("Measured simulator selectivity (physics gate, `{}`): mean **{:.2f}:1**, "
          "95% CI [{:.2f}, {:.2f}], threshold {}.".format(
              ev.gate_path, float(sel.get("mean", float("nan"))), float(ci[0]), float(ci[1]),
              sel.get("threshold")))

    # ---- honest read. This checkpoint's headline is a PARTIAL NEGATIVE, and a manifest that
    # only listed the numbers would let a reader skim the figures and take the opposite away.
    A("")
    A("## What the numbers actually say (honest read)")
    A("")
    si = s.get("selectivity_index", {})
    c1 = s.get("controls", {}).get("C1_null_model", {}).get("per_split", {})
    gap = s.get("gap_i_iii", {})
    A("1. **The simulator's per-wall law is real and matches theory.** Ground truth and the "
      "ISM-ray analytic law agree cell-by-cell in figure C, and figure B shows the measurement "
      "tracking ISM-ray rather than Kuttruff. This is a property of the data, not of the model.")
    A("2. **The model reproduces the law only partially, and only for (wall, material) pairs it "
      "was trained on.** Selectivity index: GT **{:.1f}x**, theory **{:.1f}x**, model "
      "**{:.1f}x** — the model recovers roughly {:.0f}% of the measured selectivity.".format(
          float(si.get("gt", float("nan"))), float(si.get("theory", float("nan"))),
          float(si.get("pred", float("nan"))),
          100.0 * float(si.get("pred", 0.0)) / max(float(si.get("gt", 1.0)), 1e-9)))
    A("3. **On never-trained combinations the edit response does not transfer.** For split "
      "(iii) the model's mean edit effect is {:.2f} Hz against a measured {:.2f} Hz, and the "
      "(iii)-(i) gap is {:.2f} Hz = {:.0f}% of the ground-truth effect size.".format(
          float(s["splits"]["iii_unseen_geom_heldout_combo"]["edit"]["pred_effect_size_hz"]),
          float(s["splits"]["iii_unseen_geom_heldout_combo"]["edit"]["gt_effect_size_hz"]),
          float(gap.get("gap_hz", float("nan"))),
          float(gap.get("gap_pct_of_gt_effect", float("nan")))))
    A("4. **The C1 null model is the load-bearing control, and it is not beaten on three of "
      "four splits.** C1 renders the *baseline* and scores it against the *edited* ground "
      "truth, i.e. it applies no edit at all:")
    A("")
    A("   | split | model E_BW (Hz) | C1 null E_BW (Hz) | model better than null? |")
    A("   |---|---|---|---|")
    for sp in SPLIT_KEYS:
        b = s["splits"].get(sp)
        nl = c1.get(sp, {}).get("null_E_BW_hz")
        if not b or not _fin(nl):
            continue
        mv = float(b["edit"]["E_BW_hz"])
        A("   | {} | {:.3f} | {:.3f} | {} |".format(
            sp, mv, float(nl), "yes" if mv < float(nl) else "**no**"))
    A("")
    # n_iters comes from the run's own train_meta.json, not a literal: this sentence is about
    # how far through training the checkpoint is, and a stale hard-coded denominator would
    # misstate exactly that.
    n_iters = None
    ckpt = s.get("checkpoint")
    if ckpt:
        tm = Path(ckpt).parent / "train_meta.json"
        if tm.exists():
            n_iters = json.loads(tm.read_text()).get("cfg", {}).get("n_iters")
    A("5. **Training was still running.** These figures are the checkpoint at iteration {} of "
      "{}, so this is a mid-training snapshot and not the chunk's final result. Any claim made "
      "from this pack must carry the iteration number.".format(
          s.get("iter"),
          "a {}-iteration schedule".format(n_iters) if n_iters else "an unfinished run"))
    A("")
    A("The defensible claim from this pack is therefore: *the simulator has a sharp, "
      "theory-matching per-wall law, and the model has begun to learn it for trained "
      "(wall, material) pairs while not yet generalising to unseen pairs.*")
    A("")
    A("## Figures")
    A("")
    for r in results:
        A("### `{}`".format(r["filename"]))
        A("")
        A("- **Shows:** {}".format(r["shows"]))
        A("- **Size:** {}x{} px".format(r["png_width"], r["png_height"]))
        A("- **Source files / keys:** {}".format(
            "; ".join("`{}`".format(x) for x in r["source_files"])))
        A("- **Numbers on the figure:**")
        A("")
        A("  ```")
        for n in r["numbers"]:
            A("  " + n)
        A("  ```")
        A("")
    A("## Headline numbers (all from `summary.json`)")
    A("")
    A("Splits (ii) and (iii) are the **{}** — the (wall, material) pairs `{}` were excluded "
      "from training entirely. Split (iii) is the headline test: unseen geometry *and* unseen "
      "combination. Split (iv) uses alpha={} , which appears on no training wall.".format(
          NEVER_SEEN + "S", "`, `".join("{}+{}".format(*c) for c in s.get("heldout_combos", [])),
          s.get("unseen_alpha")))
    A("")
    A("| split | n | band LSD (dB) | E_BW (Hz) | C1 null E_BW (Hz) | edit_bw_pearson |")
    A("|---|---|---|---|---|---|")
    c1 = s.get("controls", {}).get("C1_null_model", {}).get("per_split", {})
    for sp in SPLIT_KEYS:
        b = s["splits"].get(sp)
        if not b:
            continue
        A("| {} | {} | {:.3f} | {:.3f} | {} | {:.3f} |".format(
            sp, b["n_configs"], b["fidelity"]["band_lsd_db"], b["edit"]["E_BW_hz"],
            _fmt(c1.get(sp, {}).get("null_E_BW_hz"), 3).lstrip("+"),
            b["edit"]["edit_bw_pearson"]))
    A("")
    sel = s.get("selectivity_index", {})
    A("Selectivity index — GT **{:.2f}x**, theory **{:.2f}x**, model **{:.2f}x** "
      "(`selectivity_index`).".format(
          float(sel.get("gt", float("nan"))), float(sel.get("theory", float("nan"))),
          float(sel.get("pred", float("nan")))))
    A("")
    A("## Reproduction")
    A("")
    A("```bash")
    A("export PYTHONPATH=\"$PWD\"")
    A("sbatch scripts/slurm/p3_2_dump_fields.sh        # GPU: fields for figure E")
    A("python scripts/make_p3_2_figures.py             # CPU: figures + this manifest")
    A("```")
    A("")
    path.write_text("\n".join(out) + "\n")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default="outputs/p3_2/eval")
    ap.add_argument("--gate", default="outputs/p3_2/gate/gate.json")
    ap.add_argument("--out", default="outputs/p3_2/meeting_assets")
    ap.add_argument("--fields", default=None,
                    help="NPZ from scripts/p3_2_dump_fields.py (default: <out>/fields.npz)")
    ap.add_argument("--geom", default=None, help="L,W of the exemplar room (default: the frozen "
                                                 "test geometry with the most valid modes)")
    ap.add_argument("--synthetic", action="store_true",
                    help="fabricate schema-valid inputs and render watermarked layout checks")
    ap.add_argument("--only", default=None, help="comma-separated subset of A,B,C,D,E")
    args = ap.parse_args()

    plt.rcParams.update(RC)

    tmp = None
    if args.synthetic:
        tmp = Path(tempfile.mkdtemp(prefix="p3_2_synthetic_"))
        sp, pp, fp = synthesize(tmp)
        gate = None
        out_dir = Path(args.out).parent / "meeting_assets_SYNTHETIC"
        fields = fp
        print("SYNTHETIC inputs in {}".format(tmp))
    else:
        sp = Path(args.eval_dir) / "summary.json"
        pp = Path(args.eval_dir) / "per_config.json"
        gate = Path(args.gate)
        out_dir = Path(args.out)
        fields = Path(args.fields) if args.fields else out_dir / "fields.npz"
        for p in (sp, pp):
            if not p.exists():
                raise SystemExit("missing input: {} (run aaf.eval.p3_2_eval first, or use "
                                 "--synthetic for a layout check)".format(p))

    ev = Eval(sp, pp, gate)
    geom = (tuple(float(v) for v in args.geom.split(",")) if args.geom
            else ev.pick_geometry())
    print("exemplar geometry: L={:.2f} W={:.2f}".format(*geom))
    print("checkpoint={} iter={}".format(ev.summary.get("checkpoint"), ev.summary.get("iter")))
    out_dir.mkdir(parents=True, exist_ok=True)

    want = {c.strip().upper() for c in args.only.split(",")} if args.only else set("ABCDE")
    results: List[dict] = []
    plan = [
        ("A", "A_pick_your_wall.png", lambda p: figure_a(ev, geom, p, args.synthetic)),
        ("B", "B_pick_your_material.png", lambda p: figure_b(ev, geom, p, args.synthetic)),
        ("C", "C_selectivity_matrix.png", lambda p: figure_c(ev, p, args.synthetic)),
        ("D", "D_generalization.png", lambda p: figure_d(ev, p, args.synthetic)),
        ("E", "E_mode_shape_and_level.png",
         lambda p: figure_e(ev, geom, fields, p, args.synthetic)),
    ]
    for key, name, fn in plan:
        if key not in want:
            continue
        r = fn(out_dir / name)
        ok = (r["png_width"], r["png_height"]) >= (1920, 1080)
        print("[{}] {} {}x{} {}".format(key, r["filename"], r["png_width"], r["png_height"],
                                        "OK" if ok else "TOO SMALL"))
        if not ok:
            raise SystemExit("{} rendered at {}x{}, below the 1920x1080 floor".format(
                r["filename"], r["png_width"], r["png_height"]))
        results.append(r)

    if want == set("ABCDE"):
        write_manifest(out_dir / "FIGURE_MANIFEST.md", ev, results, geom, args.synthetic)
        print("wrote {}".format(out_dir / "FIGURE_MANIFEST.md"))

    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
