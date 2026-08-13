"""Build the P3-2b meeting pack: five figures + FIGURE_MANIFEST.md.

Reads the P3-2b evaluation artifacts -- ``outputs/p3_2b/eval/<arm>/summary.json`` and
``m_response.json`` -- plus the cross-arm table ``outputs/p3_2b/ablation.json`` written by
``scripts/p3_2b_ablation.py``, and renders figures A-E at exactly 1920x1080 into
``outputs/p3_2b/meeting_assets/``.

Every number drawn on a figure is read from one of those files at run time. The only
quantities computed here rather than read are (a) the standard error on a fitted slope,
re-derived from the published sweep points because the m-response schema carries ``a`` and
``r2`` but no SE, and (b) the alpha <-> m conversion, taken from the frozen
:mod:`aaf.data.mat_configs_cont` helpers.

THE kappa CORRECTION, which every rho on every panel depends on: the bandwidth estimator
measures a *calibrated* -3 dB width, not the raw Lorentzian width. The physics gate's T5 fit
gives ``BW = 0.302 + 1.6608 * (gamma/pi)``. The intercept cancels in a paired delta; the
slope does not. So the theoretical slope of a MEASURED delta-bandwidth against delta-m is
``a_theory = kappa * c / (4 pi D)`` with ``kappa = 1.6607564051417665`` and ``D = L`` for
west/east on the x-axial family, ``D = W`` for south/north on the y-axial family. Scoring
against the raw Lorentzian slope would hand a perfect model rho ~ 0.60 and fail it. Every
figure that shows a rho shows the kappa-scaled one and prints the raw comparison alongside.

SCOPING, which every caption repeats: the wall selectivity that makes this chunk legible is
a property of the ISM *simulator* -- its reflection coefficient is real and
angle-independent, so a pure x-axial mode sees exactly zero damping from the north/south
walls. Real locally-reacting walls follow Kuttruff and would show only ~2:1 with no
invariant family. The claim is "the model learns the simulator's per-wall law", never "the
model learns room acoustics".

Usage
-----
    python scripts/make_p3_2b_figures.py                     # real artifacts
    python scripts/make_p3_2b_figures.py --synthetic         # layout smoke test

``--synthetic`` fabricates schema-valid inputs in a temporary directory so the layout can be
verified before the real evals land. Those figures are stamped with a loud watermark, are
written to ``outputs/p3_2b/meeting_assets_SYNTHETIC/``, and are gitignored; they must never
be committed as results.
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

from aaf.data.mat_configs_cont import HOLDOUT_SLABS, alpha_of_m, in_slab, m_of_alpha  # noqa: E402
from aaf.walls import ALPHA_BASELINE, MATERIALS, WALLS_2D  # noqa: E402

# --------------------------------------------------------------------------- style
# Forked verbatim from scripts/make_p3_2_figures.py: the two packs are shown side by side
# in the same meeting, and a colour or a caption position that means something different
# between them is a reading error waiting to happen.
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

C_GT = "#1f6feb"        # measured ground truth (ISM)
C_MODEL = "#d1242f"     # the trained model, zero-shot
C_ISM = "#1a7f37"       # analytic ISM-ray law, kappa-scaled (what the simulator obeys)
C_KUT = "#8250df"       # Kuttruff law; carried from the P3-2 palette, not plotted here
C_NULL = "#57606a"      # null / control references
C_SLAB = "#ffd8a8"      # the held-out m-slab
C_ORTH = "#9aa4ae"      # the orthogonal (invariant) family

FAMILIES = ("x_axial", "y_axial", "tangential")
FAMILY_LABEL = {"x_axial": "x-axial\n(n_x,0)", "y_axial": "y-axial\n(0,n_y)",
                "tangential": "tangential\n(n_x,n_y)"}
FAMILY_SHORT = {"x_axial": "x-axial", "y_axial": "y-axial", "tangential": "tangential"}

WALL_OWN_FAMILY = {"west": "x_axial", "east": "x_axial",
                   "south": "y_axial", "north": "y_axial"}

SPLIT_KEYS = ("S1_unseen_geom_nonslab_1wall", "S2_unseen_geom_slab", "S3_seen_geom_slab",
              "S4_unseen_geom_alpha030", "S5_unseen_geom_2wall")
GATE_SPLIT = "S2_unseen_geom_slab"

ZERO_SHOT = ("Zero-shot, unseen geometry, configuration computed from (L, W, alpha_west, "
             "alpha_east, alpha_south, alpha_north) -- no measurements, no per-config "
             "fitting.")
SCOPE_NOTE = ("Scoping: the large wall selectivity is a property of the ISM simulator "
              "(angle-independent reflection, no grazing-incidence absorption); real "
              "locally-reacting walls follow Kuttruff (~2:1, no invariant family). The claim "
              "is that the model learns the SIMULATOR's per-wall law.")
KAPPA_NOTE = ("Theory is kappa-scaled: the estimator returns a calibrated -3 dB width, so "
              "a_theory = kappa*c/(4*pi*D) with kappa = {:.4f} from the frozen P3-2 gate. The "
              "raw Lorentzian slope is quoted alongside for transparency; scoring against it "
              "would give a perfect model rho ~ 0.60.")
NEVER_SEEN = "NEVER-SEEN COMBINATION"
PRESET_ALPHAS = (0.05, 0.15, 0.30, 0.50, 0.70)

CAPTION_WRAP = 205


def new_figure(suptitle: str, caption: str, legend_row: bool = False,
               banner: bool = False, banner_lines: int = 2) -> plt.Figure:
    """A 1920x1080 canvas with the bottom strip reserved for caption (and optional banner).

    Constrained layout otherwise packs axes down to y=0 and the caption lands on top of the
    x tick labels, so the reserved height is computed from the wrapped caption up front and
    handed to the layout engine as ``rect``.
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


def add_banner(fig: plt.Figure, text: str, fontsize: float = 14.0,
               facecolor: str = "#f6f8fa", edgecolor: str = "#8c959f") -> None:
    """A boxed one/two-line verdict strip that sits between the axes and the caption."""
    fig.text(0.5, fig._aaf_banner_y, text, ha="center", va="bottom", fontsize=fontsize,
             color="#24292f", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.45", facecolor=facecolor, edgecolor=edgecolor))


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


def _u(x, nd: int = 2) -> str:
    """Unsigned format -- for quantities like rho where a leading '+' reads as noise."""
    return ("{:." + str(nd) + "f}").format(float(x)) if _fin(x) else "n/a"


def alpha_of(material: str) -> float:
    """Absorption coefficient of a material id, including the 'A030' sweep point."""
    if material in MATERIALS:
        return float(MATERIALS[material])
    if material.startswith("A"):
        return float(material[1:]) / 100.0
    raise KeyError(material)


def se_through_origin(x: Sequence[float], y: Sequence[float]) -> float:
    """Standard error of the slope in ``y = a x``, no intercept.

    The m-response schema publishes ``a`` and ``r2`` per cell but no SE, and a slope on 20
    swept points with no uncertainty invites over-reading a rho that is two points from
    changing. Re-derived here from the same published points the fit used, so the number on
    the panel and the number in the JSON cannot drift.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    sxx = float(np.sum(xa * xa))
    if xa.size < 3 or sxx <= 0.0:
        return float("nan")
    a = float(np.sum(xa * ya) / sxx)
    ss_res = float(np.sum((ya - a * xa) ** 2))
    return float(np.sqrt(ss_res / ((xa.size - 1) * sxx)))


# --------------------------------------------------------------------------- data access
class Arm:
    """Thin accessor over one arm's summary.json / m_response.json (+ the ablation table)."""

    def __init__(self, summary_path: Path, mresp_path: Optional[Path],
                 ablation_path: Optional[Path]) -> None:
        self.summary_path = summary_path
        self.mresp_path = mresp_path
        self.ablation_path = ablation_path
        self.summary = json.loads(summary_path.read_text())
        self.mresp = (json.loads(mresp_path.read_text())
                      if mresp_path is not None and mresp_path.exists() else None)
        self.ablation = (json.loads(ablation_path.read_text())
                         if ablation_path is not None and ablation_path.exists() else None)

    @property
    def arm(self) -> str:
        return str(self.summary.get("arm", self.summary_path.parent.name))

    @property
    def meta(self) -> dict:
        return self.summary.get("meta") or {}

    @property
    def kappa(self) -> float:
        sf = self.summary.get("slope_fit") or {}
        for v in (sf.get("kappa"), self.meta.get("kappa"),
                  (self.mresp or {}).get("kappa")):
            if _fin(v):
                return float(v)
        return float("nan")

    @property
    def verdict(self) -> dict:
        """Verdict with ``criteria`` normalized to a list and ``spec_sha`` resolved.

        The gate module keys criteria by name and calls the flag ``pass`` /
        ``thresholds_sha256``; the chunk spec describes a list with ``passed`` / ``spec_sha``.
        Both are accepted so the panel does not depend on which writer produced the file.
        """
        v = dict(self.summary.get("verdict") or {})
        crit = v.get("criteria", [])
        items = list(crit.values()) if isinstance(crit, dict) else list(crit)
        v["criteria"] = [
            {"name": c.get("name"), "value": c.get("value"), "op": c.get("op"),
             "threshold": c.get("threshold"),
             "passed": bool(c.get("passed", c.get("pass", False))), "note": c.get("note")}
            for c in items]
        v["spec_sha"] = v.get("spec_sha") or v.get("thresholds_sha256")
        return v

    def slope_group(self, group: str = "slab_local") -> dict:
        return (((self.summary.get("slope_fit") or {}).get("aggregate", {})
                 .get("own_family", {}).get(group, {})) or {})

    @property
    def slabs(self) -> Dict[str, Sequence[float]]:
        """The held-out slabs, from the eval file when present, else the frozen helper."""
        s = self.summary.get("slabs_m") or {}
        hs = s.get("holdout_slabs_m") if isinstance(s, dict) else None
        if isinstance(hs, dict) and hs:
            return {k: list(v) for k, v in hs.items()}
        if isinstance(s, dict) and s and all(isinstance(v, (list, tuple)) for v in s.values()):
            return {k: list(v) for k, v in s.items()}
        return {k: list(v) for k, v in HOLDOUT_SLABS.items()}

    def slab_of(self, wall: str) -> Optional[Sequence[float]]:
        return self.slabs.get(wall)

    def is_slab_combo(self, wall: str, material: str) -> bool:
        try:
            return bool(in_slab(wall, alpha_of(material)))
        except KeyError:
            return False

    @property
    def materials(self) -> List[str]:
        sm = self.summary.get("selectivity_matrix") or {}
        return sorted(sm.keys(), key=alpha_of)


# --------------------------------------------------------------------------- figure A
def figure_a(arm: Arm, material: str, out: Path, synthetic: bool) -> dict:
    """Pick your wall: edit each wall in turn and watch which family broadens."""
    sm = arm.summary.get("selectivity_matrix") or {}
    if material not in sm:
        material = arm.materials[-1]
    alpha = alpha_of(material)
    numbers: List[str] = []

    caption = ("Change in -3 dB modal bandwidth, averaged over the frozen test geometries, for "
               "a single edit alpha {:.2f} -> {:.2f} applied to one wall at a time. {} {} {}"
               .format(ALPHA_BASELINE, alpha, ZERO_SHOT,
                       KAPPA_NOTE.format(arm.kappa), SCOPE_NOTE))
    fig = new_figure("Pick your wall: editing one wall broadens that wall's own mode family "
                     "({}, alpha {:.2f})".format(material, alpha), caption, legend_row=True)
    axes = fig.subplots(2, 2).ravel()

    series = [("Ground truth (ISM)", "gt_d_bw", C_GT, "GT"),
              ("Theory (ISM-ray, kappa-scaled)", "theory_d_bw", C_ISM, "theory"),
              ("Model (zero-shot)", "pred_d_bw", C_MODEL, "model")]
    # One y-scale across the four panels; otherwise a dead panel silently looks fine.
    all_vals = [sm[material][w][f].get(k) for w in WALLS_2D for f in FAMILIES
                for _, k, _, _ in series if _fin(sm[material][w][f].get(k))]
    lim = max([abs(float(v)) for v in all_vals] + [1.0]) * 1.35

    for ax, wall in zip(axes, WALLS_2D):
        held = arm.is_slab_combo(wall, material)
        own = WALL_OWN_FAMILY[wall]
        width = 0.26
        for si, (lab, key, col, short) in enumerate(series):
            off = (si - 1) * width
            for xi, fam in enumerate(FAMILIES):
                cell = sm[material][wall][fam]
                v = cell.get(key)
                vv = float(v) if _fin(v) else np.nan
                ax.bar([xi + off], [vv], width, color=col,
                       label=lab if (wall == "west" and xi == 0) else None,
                       edgecolor="white", linewidth=0.8, zorder=3)
                if _fin(vv):
                    ax.text(xi + off, vv + (0.03 * lim if vv >= 0 else -0.03 * lim),
                            "{:+.1f}".format(vv), ha="center",
                            va="bottom" if vv >= 0 else "top", fontsize=10.5, color=col,
                            fontweight="bold", zorder=4)
                    numbers.append("A {}->{} {} {}: {:+.3f} Hz (n={})".format(
                        wall, material, FAMILY_SHORT[fam], short, vv, cell.get("n")))

        oi = FAMILIES.index(own)
        ax.axvspan(oi - 0.5, oi + 0.5, color="#fff3cd", zorder=0)
        ax.text(oi, lim * 0.93, "own family", ha="center", fontsize=12,
                color="#8a6d00", style="italic", zorder=6)
        ax.axhline(0, color="#24292f", lw=1.0, zorder=2)
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES])
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("change in -3 dB bandwidth (Hz)")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        despine(ax)
        ax.set_title("{} wall -> alpha {:.2f}  (m = {:.2f})".format(
            wall.upper(), alpha, m_of_alpha(alpha)),
            color=C_MODEL if held else "#24292f", fontweight="bold" if held else "normal")
        n_cells = sm[material][wall][own].get("n")
        tag = "own-family n={} modes".format(n_cells)
        if held:
            slab = arm.slab_of(wall)
            tag = "{} | held-out slab m in [{:.2f}, {:.2f}] | {}".format(
                NEVER_SEEN, slab[0], slab[1], tag)
            ax.set_facecolor("#fff5f5")
            for sp in ax.spines.values():
                sp.set_edgecolor(C_MODEL)
                sp.set_linewidth(2.0)
        ax.text(0.015, 0.03, tag, transform=ax.transAxes, fontsize=11.5,
                color=C_MODEL if held else "#57606a",
                fontweight="bold" if held else "normal")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, fig._aaf_legend_y), frameon=False)
    add_caption(fig)
    add_watermark(fig, synthetic)
    size = save(fig, out)
    return {
        "filename": out.name, "png_width": size[0], "png_height": size[1],
        "shows": ("Four panels, one per edited wall, all edited to alpha={:.2f} ({}): change in "
                  "-3 dB modal bandwidth per mode family -- ground truth, kappa-scaled ISM-ray "
                  "theory, and the zero-shot model. Panels whose (wall, alpha) lands in a "
                  "held-out m-slab are outlined in red as NEVER-SEEN COMBINATIONS."
                  .format(alpha, material)),
        "source_files": ["{} :: selectivity_matrix.{}.<wall>.<family>."
                         "{{gt_d_bw, pred_d_bw, theory_d_bw, n}}".format(
                             arm.summary_path, material)],
        "numbers": numbers, "material": material,
    }


# --------------------------------------------------------------------------- figure B
def _wall_curves(w: dict) -> dict:
    """Unpack one (geometry, wall) cell of m_response into plottable arrays."""
    pts = w.get("points", [])
    g = {k: [] for k in ("m", "d_m", "alpha", "in_slab", "above_a_max",
                         "pred_own", "pred_own_se", "gt_own", "gt_own_se",
                         "pred_orth", "gt_orth", "theory_own", "theory_orth")}
    for p in pts:
        g["m"].append(float(p.get("m", np.nan)))
        g["d_m"].append(float(p.get("d_m", np.nan)))
        g["alpha"].append(float(p.get("alpha", np.nan)))
        g["in_slab"].append(bool(p.get("in_slab")))
        g["above_a_max"].append(bool(p.get("alpha_above_arm_A_max")))
        for side, fam, key in (("pred", "own", "pred_own"), ("gt", "own", "gt_own"),
                               ("pred", "orth", "pred_orth"), ("gt", "orth", "gt_orth")):
            v = ((p.get(side) or {}).get(fam) or {}).get("d_bw_mean")
            g[key].append(float(v) if _fin(v) else np.nan)
        for side, key in (("pred", "pred_own_se"), ("gt", "gt_own_se")):
            v = ((p.get(side) or {}).get("own") or {}).get("d_bw_sem")
            g[key].append(float(v) if _fin(v) else np.nan)
        g["theory_own"].append(float(p.get("theory_d_bw", np.nan)))
        g["theory_orth"].append(float(p.get("theory_d_bw_orth", np.nan)))
    return {k: np.asarray(v) for k, v in g.items()}


def figure_b(arm: Arm, out: Path, synthetic: bool) -> dict:
    """THE MONEY FIGURE: the m-response, geometry x wall, against kappa-scaled theory."""
    mr = arm.mresp
    geoms = (mr or {}).get("geometries", [])
    numbers: List[str] = []
    kappa = float(mr.get("kappa", arm.kappa)) if mr else arm.kappa
    a_max = float((mr or {}).get("arm_A_alpha_max", 0.70))
    m_a_max = m_of_alpha(a_max)

    caption = (
        "One dense alpha sweep per (geometry, wall). x is m = -ln(1-alpha), the coordinate the "
        "target law is EXACTLY linear in; the top axis marks the preset alphas and the dotted "
        "vertical line is the alpha=0.15 baseline. y is the change in own-family -3 dB "
        "bandwidth against that baseline, so both axes are zero there and the theory line "
        "passes through the origin by construction. The orthogonal family (thin grey) should "
        "sit at zero: that is the selectivity claim. {} {} {}".format(
            ZERO_SHOT, KAPPA_NOTE.format(kappa), SCOPE_NOTE))
    fig = new_figure("The m-response: does the model follow the theory line through the "
                     "held-out slab?", caption, legend_row=True, banner=True, banner_lines=2)
    if not geoms:
        ax = fig.subplots(1, 1)
        ax.text(0.5, 0.5, "m_response.json has no geometries", ha="center", va="center",
                transform=ax.transAxes, color=C_NULL)
        add_caption(fig)
        add_watermark(fig, synthetic)
        size = save(fig, out)
        return {"filename": out.name, "png_width": size[0], "png_height": size[1],
                "shows": "empty m-response", "source_files": [str(arm.mresp_path)],
                "numbers": []}

    axes = fig.subplots(len(geoms), len(WALLS_2D), squeeze=False)
    rho_slab_pred: List[float] = []

    for ri, g in enumerate(geoms):
        for ci, wall in enumerate(WALLS_2D):
            ax = axes[ri][ci]
            w = (g.get("walls") or {}).get(wall)
            if not w:
                ax.set_axis_off()
                continue
            d = _wall_curves(w)
            fit = w.get("fit", {}) or {}
            a_th = float(w.get("a_theory_hz_per_m", np.nan))
            a_th_raw = float(w.get("a_theory_raw_hz_per_m", np.nan))
            a_pred = float((fit.get("pred_own") or {}).get("a", np.nan))
            a_gt = float((fit.get("gt_own") or {}).get("a", np.nan))
            rho_pred = float(fit.get("rho_pred", np.nan))
            rho_gt = float(fit.get("rho_gt", np.nan))
            rho_pred_raw = float(fit.get("rho_pred_raw_theory", np.nan))
            se_pred = se_through_origin(d["d_m"], d["pred_own"])
            se_gt = se_through_origin(d["d_m"], d["gt_own"])
            # The pivot of both axes: m at the unedited baseline, recovered from the sweep's
            # own (m, d_m) pair rather than assumed, so a change of baseline alpha upstream
            # moves the line here too.
            m_base = float(np.nanmedian(d["m"] - d["d_m"])) if d["m"].size \
                else m_of_alpha(ALPHA_BASELINE)

            # -- the held-out slab, shaded. This is the whole point of the panel: correct
            #    behaviour is the predicted curve crossing the band ON the theory line.
            slab = w.get("slab_m")
            if slab:
                ax.axvspan(float(slab[0]), float(slab[1]), color=C_SLAB, alpha=0.85, zorder=0)
                # Rotated inside the band rather than above it: the corners of these panels
                # already carry the fit box and the GT rho, and a horizontal label there
                # collides with both at 12-panels-per-canvas.
                ax.text(float(np.mean(slab)), 0.55, "HELD-OUT SLAB", transform=(
                    ax.get_xaxis_transform()), ha="center", va="center", rotation=90,
                    fontsize=9.5, color="#a15c00", fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="square,pad=0.12", facecolor="white", alpha=0.72,
                              edgecolor="none"))
                if _fin(rho_pred):
                    rho_slab_pred.append(rho_pred)

            # -- alpha > arm-A max: extrapolation for arm A only, marked but not scored
            m_hi = float(np.nanmax(d["m"])) if d["m"].size else m_a_max
            if m_hi > m_a_max:
                ax.axvspan(m_a_max, m_hi * 1.02, facecolor="#f0f0f0", zorder=0, hatch="//",
                           edgecolor="#d0d7de", linewidth=0.0)
            ax.axvline(m_base, color="#8c959f", lw=1.0, ls=":", zorder=1)

            # -- kappa-scaled theory through the origin (in d_m; the baseline m is the pivot)
            grid = np.linspace(float(np.nanmin(d["m"])), float(np.nanmax(d["m"])), 64)
            ax.plot(grid, a_th * (grid - m_base), "-", color=C_ISM, lw=2.4, zorder=3,
                    label="Theory (kappa-scaled)" if (ri == 0 and ci == 0) else None)
            if _fin(a_th_raw):
                ax.plot(grid, a_th_raw * (grid - m_base), ":", color=C_ISM, lw=1.5, alpha=0.75,
                        zorder=3,
                        label="RAW Lorentzian (not scored)"
                        if (ri == 0 and ci == 0) else None)

            # -- orthogonal family, thin grey, expected ~0
            ax.plot(d["m"], d["gt_orth"], "-", color=C_ORTH, lw=1.0, alpha=0.9, zorder=2,
                    label="orthogonal family (expect 0)"
                    if (ri == 0 and ci == 0) else None)
            ax.plot(d["m"], d["pred_orth"], "--", color=C_ORTH, lw=1.0, alpha=0.9, zorder=2)

            # -- ground truth points and the model curve
            ax.plot(d["m"], d["gt_own"], "o", color=C_GT, ms=5.0, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label="Ground truth (ISM)" if (ri == 0 and ci == 0) else None)
            ax.plot(d["m"], d["pred_own"], "-", color=C_MODEL, lw=2.2, zorder=6,
                    label="Model (zero-shot)" if (ri == 0 and ci == 0) else None)
            fin = np.isfinite(d["pred_own"]) & np.isfinite(d["pred_own_se"])
            if fin.any():
                ax.fill_between(d["m"][fin], (d["pred_own"] - d["pred_own_se"])[fin],
                                (d["pred_own"] + d["pred_own_se"])[fin], color=C_MODEL,
                                alpha=0.18, linewidth=0, zorder=4)

            ax.axhline(0, color="#24292f", lw=0.8, zorder=1)
            ax.grid(alpha=0.25, zorder=0)
            despine(ax)
            ax.tick_params(labelsize=10.5)
            if ri == len(geoms) - 1:
                ax.set_xlabel("m = -ln(1 - alpha)", fontsize=12.5)
            if ci == 0:
                ax.set_ylabel("{}  L={:.2f} W={:.2f}\nd(own BW)  Hz".format(
                    g.get("role", "geom {}".format(g.get("geom_id"))), g["L"], g["W"]),
                    fontsize=10.5)
            ax.set_title("{} wall  (D={:.2f} m, own={})".format(
                wall.upper(), float(w.get("D_m", np.nan)),
                FAMILY_SHORT[w.get("own_family", WALL_OWN_FAMILY[wall])]), fontsize=13.5,
                color=C_MODEL if slab else "#24292f",
                fontweight="bold" if slab else "normal")

            # -- the per-panel number the whole figure exists to show
            txt = "a_fit {:.2f} +/- {:.2f}\na_theory {:.2f}\nrho = {}".format(
                a_pred, se_pred, a_th, _u(rho_pred, 2))
            ax.text(0.025, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                    fontsize=10.5, color=C_MODEL, fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86,
                              edgecolor="#d0d7de"))
            ax.text(0.975, 0.03, "GT rho {}".format(_u(rho_gt, 2)), transform=ax.transAxes,
                    va="bottom", ha="right", fontsize=10, color=C_GT, zorder=9)

            numbers.append(
                "B geom{} ({}) L={:.2f} W={:.2f} {} wall: a_fit_pred={:.3f}+/-{:.3f}, "
                "a_fit_gt={:.3f}+/-{:.3f}, a_theory={:.3f} (raw {:.3f}) Hz/m, rho_pred={:.3f}, "
                "rho_gt={:.3f}, rho_pred_vs_raw={:.3f}, slab={}, n_pts={}".format(
                    g.get("geom_id"), g.get("role"), g["L"], g["W"], wall, a_pred, se_pred,
                    a_gt, se_gt, a_th, a_th_raw, rho_pred, rho_gt, rho_pred_raw,
                    slab if slab else "none", int(d["m"].size)))

            # -- secondary alpha axis at the presets
            axt = ax.twiny()
            axt.set_xlim(ax.get_xlim())
            # alpha=0.05 is dropped from the top axis: at m=0.051 its label overprints the
            # m=0.163 baseline tick at this panel width, and the baseline is the one that has
            # to stay readable (it is the pivot of both axes).
            ticks = [(m_of_alpha(a), a) for a in PRESET_ALPHAS if a > 0.10]
            keep = [(t, a) for t, a in ticks
                    if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
            axt.set_xticks([t for t, _ in keep])
            axt.set_xticklabels(["{:.2f}".format(a) for _, a in keep], fontsize=9.5)
            axt.tick_params(length=2.5, pad=1.5, colors="#57606a")
            for sp in ("left", "right", "top"):
                axt.spines[sp].set_visible(False)
            if ri == 0:
                axt.set_xlabel("alpha", fontsize=10.5, color="#57606a", labelpad=1.0)

    sg = arm.slope_group("slab_local")
    banner = ("SLAB-LOCAL slope over the full eval:  rho = a_fit / a_theory = {}  "
              "(95% CI {}),  a_fit {} vs a_theory {} Hz per unit m,  n_cells {}\n"
              "kappa = {:.4f}     |     the same rho against the RAW Lorentzian slope would "
              "be {} -- shown only to make the correction visible, never to score".format(
                  _u(sg.get("rho_median"), 3),
                  "[{}, {}]".format(*[_u(v, 2) for v in (sg.get("rho_ci95") or [None, None])]),
                  _u(sg.get("a_fit_median"), 2), _u(sg.get("a_theory_median"), 2),
                  sg.get("n_cells"), kappa,
                  _u((arm.summary.get("slope_fit") or {}).get("rho_vs_raw_theory_median"), 3)))
    numbers.append("B aggregate slab_local: rho_median={}, ci95={}, a_fit={}, a_theory={}, "
                   "n_cells={}".format(_u(sg.get("rho_median"), 4), sg.get("rho_ci95"),
                                       _u(sg.get("a_fit_median"), 4),
                                       _u(sg.get("a_theory_median"), 4), sg.get("n_cells")))
    # Cross-check: the slab panels ON THIS FIGURE against the aggregate quoted in the banner.
    # They are different estimators over different sets of geometries, so they should agree
    # in magnitude but need not match; a large divergence means the sweep and the eval
    # disagree about the same model and is worth catching before the meeting, not during it.
    if rho_slab_pred:
        numbers.append("B slab panels on this figure: median rho_pred = {} over {} panels "
                       "(cross-check against the aggregate above)".format(
                           _u(float(np.median(rho_slab_pred)), 4), len(rho_slab_pred)))
    add_banner(fig, banner, fontsize=12.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(mpatches.Patch(facecolor=C_SLAB, edgecolor="#bf8700",
                                  label="held-out m-slab"))
    handles.append(mpatches.Patch(facecolor="#f0f0f0", hatch="//", edgecolor="#d0d7de",
                                  label="alpha>{:.2f} (extrapolation, arm A)".format(a_max)))
    labels += [h.get_label() for h in handles[-2:]]
    # One row: a second legend row would land on the top row of panel titles, and the
    # reserved header height is fixed by new_figure(legend_row=True).
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), fontsize=12.0,
               bbox_to_anchor=(0.5, fig._aaf_legend_y), frameon=False,
               handlelength=1.6, columnspacing=1.1, handletextpad=0.5)
    add_caption(fig)
    add_watermark(fig, synthetic)
    size = save(fig, out)
    return {
        "filename": out.name, "png_width": size[0], "png_height": size[1],
        "shows": ("The m-response, {} geometries x 4 walls. Model curve, GT points and the "
                  "kappa-scaled ISM-ray theory line through the origin, with the held-out "
                  "m-slab shaded on the west and north panels and the orthogonal family drawn "
                  "as a thin grey ~0 series. Correct behaviour is the predicted curve passing "
                  "straight through the shaded band along the theory line. The hatched region "
                  "is alpha > {:.2f}, an EXTRAPOLATION for arm A only.".format(
                      len(geoms), a_max)),
        "source_files": ["{} :: geometries[].walls.<wall>.{{points[], fit, a_theory_hz_per_m, "
                         "a_theory_raw_hz_per_m, slab_m, D_m}}".format(arm.mresp_path),
                         "{} :: slope_fit.aggregate.own_family.slab_local, slope_fit.kappa, "
                         "slope_fit.rho_vs_raw_theory_median".format(arm.summary_path)],
        "numbers": numbers,
        "note_se": ("a_fit SE is not in the schema; it is re-derived here from the same "
                    "published sweep points as sqrt(SSres/((n-1)*Sxx)) for the through-origin "
                    "fit."),
    }


# --------------------------------------------------------------------------- figure C
def figure_c(arm: Arm, out: Path, synthetic: bool) -> dict:
    """The wall-selectivity matrix: model vs ground truth vs residual."""
    sm = arm.summary.get("selectivity_matrix") or {}
    mats = sorted(sm.keys(), key=alpha_of)
    blocks = [("gt_d_bw", "Ground truth (ISM)"), ("pred_d_bw", "Model (zero-shot)"),
              ("residual_d_bw", "Residual (model - GT)")]
    numbers: List[str] = []

    vals = [sm[m][w][f].get(k) for m in mats for w in WALLS_2D for f in FAMILIES
            for k, _ in blocks[:2] if _fin(sm[m][w][f].get(k))]
    vmax = max([abs(float(v)) for v in vals] + [1.0])

    caption = ("Change in -3 dB bandwidth (Hz), averaged over the frozen test geometries; "
               "diverging scale centred at 0, printed values are raw Hz. The black box marks "
               "the axial sub-block -- the expected pattern is NOT a full diagonal: tangential "
               "modes respond to every wall. Gold dashed rows are the NEVER-SEEN COMBINATIONS "
               "(the (wall, alpha) pairs inside a held-out m-slab). {} {}".format(
                   ZERO_SHOT, SCOPE_NOTE))
    fig = new_figure("The wall-selectivity matrix: each wall broadens its own axial family",
                     caption, banner=True)
    gs = fig.add_gridspec(len(mats), len(blocks) + 1,
                          width_ratios=[1] * len(blocks) + [0.045])
    axes = [[fig.add_subplot(gs[r, c]) for c in range(len(blocks))] for r in range(len(mats))]
    cax = fig.add_subplot(gs[:, len(blocks)])

    im = None
    for ri, mat in enumerate(mats):
        for ci, (key, blabel) in enumerate(blocks):
            ax = axes[ri][ci]
            M = np.full((len(WALLS_2D), len(FAMILIES)), np.nan)
            for wi, wall in enumerate(WALLS_2D):
                for fi, fam in enumerate(FAMILIES):
                    v = sm[mat][wall][fam].get(key)
                    if _fin(v):
                        M[wi, fi] = float(v)
            im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            for wi in range(len(WALLS_2D)):
                for fi in range(len(FAMILIES)):
                    v = M[wi, fi]
                    txt = "{:+.1f}".format(v) if _fin(v) else "n/a"
                    shade = "white" if _fin(v) and abs(v) > 0.62 * vmax else "#24292f"
                    ax.text(fi, wi, txt, ha="center", va="center", fontsize=12.5,
                            color=shade, fontweight="bold")
                    if _fin(v):
                        numbers.append("C {} {} {} {}: {:+.3f} Hz".format(
                            mat, WALLS_2D[wi], FAMILIES[fi], key, v))
            ax.add_patch(mpatches.Rectangle((-0.5, -0.5), 2.0, len(WALLS_2D), fill=False,
                                            edgecolor="#24292f", lw=2.6, zorder=6))
            for wi, wall in enumerate(WALLS_2D):
                if arm.is_slab_combo(wall, mat):
                    ax.add_patch(mpatches.Rectangle(
                        (-0.5, wi - 0.5), len(FAMILIES), 1.0, fill=False,
                        edgecolor="#bf8700", lw=3.2, linestyle=(0, (4, 2)), zorder=7))
            ax.set_xticks(range(len(FAMILIES)))
            ax.set_yticks(range(len(WALLS_2D)))
            ax.set_xticklabels([FAMILY_SHORT[f] for f in FAMILIES] if ri == len(mats) - 1
                               else [], fontsize=12.5)
            ax.set_yticklabels(list(WALLS_2D) if ci == 0 else [], fontsize=12.5)
            if ri == 0:
                ax.set_title(blabel, fontsize=17)
            if ci == 0:
                ax.set_ylabel("{}\nalpha={:.2f}\nm={:.2f}".format(
                    mat, alpha_of(mat), m_of_alpha(alpha_of(mat))), fontsize=13.0,
                    fontweight="bold", rotation=0, ha="right", va="center", labelpad=58)
            ax.tick_params(length=0)

    if im is not None:
        fig.colorbar(im, cax=cax, label="change in -3 dB bandwidth (Hz)")

    sel = arm.summary.get("selectivity_index", {}) or {}
    add_banner(fig, "wall-selectivity index (own-family response / other-family response):     "
                    "GT {}x     |     kappa-scaled ISM-ray theory {}x     |     model {}x"
                    .format(_u(sel.get("gt"), 1), _u(sel.get("theory"), 1),
                            _u(sel.get("pred"), 1)), fontsize=15)
    numbers.append("C selectivity_index: gt={}, theory={}, pred={}".format(
        _u(sel.get("gt"), 3), _u(sel.get("theory"), 3), _u(sel.get("pred"), 3)))
    add_caption(fig)
    add_watermark(fig, synthetic)
    size = save(fig, out)
    return {
        "filename": out.name, "png_width": size[0], "png_height": size[1],
        "shows": ("Wall x family selectivity matrix per material: ground truth, the zero-shot "
                  "model, and the model-minus-GT residual, in raw Hz of bandwidth change. Rows "
                  "in a held-out m-slab are outlined gold."),
        "source_files": ["{} :: selectivity_matrix.<material>.<wall>.<family>."
                         "{{gt_d_bw, pred_d_bw, residual_d_bw}}, selectivity_index".format(
                             arm.summary_path)],
        "numbers": numbers,
    }


# --------------------------------------------------------------------------- figure D
def figure_d(arm: Arm, out: Path, synthetic: bool) -> dict:
    """The S2 headline: the pre-registered verdict, its criteria, and its spec hash."""
    s = arm.summary
    s2 = (s.get("splits") or {}).get(GATE_SPLIT, {}) or {}
    v = arm.verdict
    numbers: List[str] = []
    passed = bool(v.get("passed"))

    caption = ("S2 is the chunk's question: an unseen geometry edited to a (wall, alpha) pair "
               "whose m lies inside a HELD-OUT slab that was never trained on any wall. The "
               "thresholds were frozen and hashed before any arm was evaluated -- the hash is "
               "printed below so the table cannot be re-scored after the fact. {} {}".format(
                   ZERO_SHOT, SCOPE_NOTE))
    fig = new_figure("S2 headline: unseen geometry x HELD-OUT m-slab  ({})".format(arm.arm),
                     caption, banner=True, banner_lines=3)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 0.95], height_ratios=[1.0, 0.80])
    axc = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1])
    axf = fig.add_subplot(gs[0, 2])
    axb = fig.add_subplot(gs[1, :])

    # -- criteria table
    axc.set_axis_off()
    axc.set_title("Acceptance criteria (frozen, pre-registered)", fontsize=16, loc="left")
    rows = [["criterion", "value", "", "threshold", ""]]
    colors = [["#eaeef2"] * 5]
    for c in v.get("criteria", []):
        ok = bool(c.get("passed"))
        rows.append([str(c.get("name")), _u(c.get("value"), 3), str(c.get("op") or ""),
                     _u(c.get("threshold"), 2), "PASS" if ok else "FAIL"])
        colors.append(["white", "white", "white", "white",
                       "#d2f2d9" if ok else "#ffd7d5"])
        numbers.append("D criterion {}: value={} {} {} -> {}".format(
            c.get("name"), _u(c.get("value"), 4), c.get("op"), _u(c.get("threshold"), 3),
            "PASS" if ok else "FAIL"))
    # bbox rather than loc: with loc the table is auto-sized to its text and the long
    # criterion names get clipped instead of the column widening.
    tb = axc.table(cellText=rows[1:], colLabels=rows[0], bbox=[0.0, 0.24, 1.0, 0.68],
                   cellColours=colors[1:], colColours=colors[0], cellLoc="center",
                   colWidths=[0.36, 0.16, 0.10, 0.20, 0.18])
    tb.auto_set_font_size(False)
    tb.set_fontsize(12.5)
    for (r, _c), cell in tb.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if r == 0:
            cell.set_text_props(fontweight="bold")

    blk = v.get("blockers") or []
    axc.text(0.02, 0.16, "blockers: {}".format(
        "; ".join("{} = {}".format(b.get("name"), _u(b.get("value"), 3)) for b in blk)
        if blk else "none"), transform=axc.transAxes, fontsize=12.5,
        color=C_MODEL if blk else "#1a7f37", fontweight="bold")
    axc.text(0.02, 0.05, "spec {}\nsha256 {}".format(
        (v.get("thresholds") or {}).get("spec", "n/a"), v.get("spec_sha") or "n/a"),
        transform=axc.transAxes, fontsize=10.0, family="monospace", color="#57606a",
        va="bottom")
    numbers.append("D spec_sha: {}".format(v.get("spec_sha")))

    # -- S2 measurements
    axm.set_axis_off()
    axm.set_title("S2 measurements", fontsize=16, loc="left")
    fid = s2.get("fidelity", {}) or {}
    edit = s2.get("edit", {}) or {}
    sg = arm.slope_group("slab_local")
    mrows = [
        ("configs / mode cells", "{} / {}".format(s2.get("n_configs"), s2.get("n_cells"))),
        ("frac modes dropped", _u(s2.get("frac_modes_dropped"), 3)),
        ("band LSD (dB)", _u(fid.get("band_lsd_db"), 3)),
        ("mag corr", _u(fid.get("mag_corr"), 3)),
        ("phase corr (mag-weighted)", _u(fid.get("phase_corr_mw"), 3)),
        ("RIR pearson", _u(fid.get("rir_pearson"), 3)),
        ("E_BW (Hz, lower better)", _u(edit.get("E_BW_hz"), 3)),
        ("E_LVL (dB, lower better)", _u(edit.get("E_LVL_db"), 3)),
        ("GT effect size (Hz)", _u((s2.get("edit_detail") or {}).get("gt_effect_size_hz"), 3)),
        ("model effect size (Hz)",
         _u((s2.get("edit_detail") or {}).get("pred_effect_size_hz"), 3)),
        ("rho slab-local (kappa)", _u(sg.get("rho_median"), 3)),
        ("rho 95% CI", "[{}, {}]".format(
            *[_u(x, 2) for x in (sg.get("rho_ci95") or [None, None])])),
        ("rho vs RAW theory", _u((s.get("slope_fit") or {}).get("rho_vs_raw_theory_median"), 3)),
        ("in-dist val LSD (dB)", _u(s.get("in_dist_val_lsd_db"), 3)),
    ]
    for k, val in mrows:
        numbers.append("D S2 {}: {}".format(k, val))
    tb2 = axm.table(cellText=[[k, val] for k, val in mrows], bbox=[0.0, 0.02, 1.0, 0.94],
                    cellLoc="left", colWidths=[0.62, 0.38],
                    cellColours=[["white", "#f6f8fa"]] * len(mrows))
    tb2.auto_set_font_size(False)
    tb2.set_fontsize(11.5)
    for _k, cell in tb2.get_celld().items():
        cell.set_edgecolor("#e1e4e8")

    # -- fidelity vs the null render, so "it renders the room" is not read as "it edits"
    nullf = s2.get("null_fidelity", {}) or {}
    labels = ["model\n(edited)", "null\n(baseline)"]
    vals = [float(fid.get("band_lsd_db", np.nan)), float(nullf.get("band_lsd_db", np.nan))]
    axf.bar([0, 1], vals, 0.55, color=[C_MODEL, C_NULL], edgecolor="white", zorder=3)
    for x, val in zip([0, 1], vals):
        if _fin(val):
            axf.text(x, val, "{:.3f}".format(val), ha="center", va="bottom", fontsize=13,
                     fontweight="bold")
    gain = edit.get("edit_gain")
    axf.set_xticks([0, 1])
    axf.set_xticklabels(labels)
    axf.set_ylabel("band LSD vs edited GT (dB)")
    axf.set_title("edit_gain = {}\n(null / model, must exceed 1)".format(_u(gain, 3)),
                  fontsize=14,
                  color=C_MODEL if not _fin(gain) or float(gain) <= 1 else "#1a7f37")
    axf.set_ylim(0, max([v for v in vals if _fin(v)] + [1.0]) * 1.25)
    axf.grid(axis="y", alpha=0.3, zorder=0)
    despine(axf)
    numbers.append("D S2 edit_gain: {} (model band LSD {} dB, null band LSD {} dB)".format(
        _u(gain, 4), _u(fid.get("band_lsd_db"), 4), _u(nullf.get("band_lsd_db"), 4)))

    # -- per-combo breakout: the two held-out slab combos, individually
    pc = s2.get("per_combo", {}) or {}
    combos = sorted(pc.keys())
    if combos:
        cx = np.arange(len(combos))
        gtv, pdv = [], []
        for c in combos:
            e = pc[c].get("edit", {}) or {}
            gtv.append(float(e.get("gt_effect_size_hz", np.nan)))
            pdv.append(float(e.get("pred_effect_size_hz", np.nan)))
            numbers.append("D S2 combo {}: GT effect {} Hz, model effect {} Hz, slope {}, "
                           "gain {} (n_configs={})".format(
                               c, _u(gtv[-1], 3), _u(pdv[-1], 3),
                               _u(e.get("edit_bw_slope"), 3), _u(e.get("edit_gain"), 3),
                               pc[c].get("n_configs")))
        axb.bar(cx - 0.19, gtv, 0.36, color=C_GT, label="GT effect size", edgecolor="white",
                zorder=3)
        axb.bar(cx + 0.19, pdv, 0.36, color=C_MODEL, label="model effect size",
                edgecolor="white", zorder=3)
        for x, gv, pv in zip(cx, gtv, pdv):
            for xo, val, col in ((x - 0.19, gv, C_GT), (x + 0.19, pv, C_MODEL)):
                if _fin(val):
                    axb.text(xo, val, "{:.2f}".format(val), ha="center", va="bottom",
                             fontsize=12, color=col, fontweight="bold")
            if _fin(gv) and _fin(pv) and abs(gv) > 1e-9:
                axb.text(x, max(gv, pv) * 1.10, "model recovers {:.0f}% of the edit".format(
                    100.0 * pv / gv), ha="center", fontsize=12.5, fontweight="bold",
                    color="#24292f")
        axb.set_xticks(cx)
        axb.set_xticklabels(["{}   [{}]".format(c, NEVER_SEEN) for c in combos], fontsize=13)
        axb.set_ylabel("mean |delta BW| (Hz)")
        axb.set_title("The held-out slab combos, individually", fontsize=16)
        axb.set_ylim(0, max([v for v in gtv + pdv if _fin(v)] + [1.0]) * 1.32)
        axb.legend(fontsize=12.5, framealpha=0.95, ncol=2)
        axb.grid(axis="y", alpha=0.3, zorder=0)
        despine(axb)
    else:
        axb.set_axis_off()

    import textwrap

    mid = " (MID-TRAINING: iter {} of {})".format(s.get("iter"),
                                                  arm.meta.get("total_iters")) \
        if arm.meta.get("mid_training") else ""
    one_line = textwrap.fill(v.get("one_line") or "", 165)
    add_banner(fig, "S2 VERDICT: {}{}\n{}".format("PASS" if passed else "FAIL", mid, one_line),
               fontsize=12.5, facecolor="#d2f2d9" if passed else "#ffd7d5",
               edgecolor="#1a7f37" if passed else C_MODEL)
    numbers.append("D verdict: passed={} | {}".format(passed, v.get("one_line")))
    add_caption(fig)
    add_watermark(fig, synthetic)
    size = save(fig, out)
    return {
        "filename": out.name, "png_width": size[0], "png_height": size[1],
        "shows": ("The S2 headline table: the four frozen acceptance criteria with their "
                  "pass/fail state and the thresholds' sha256, the S2 fidelity and edit "
                  "measurements, the edit_gain against the null (baseline) render, and the two "
                  "held-out slab combos broken out individually."),
        "source_files": ["{} :: splits.{}.{{n_configs, n_cells, frac_modes_dropped, fidelity, "
                         "null_fidelity, edit, edit_detail, per_combo}}, verdict, "
                         "slope_fit.aggregate.own_family.slab_local".format(
                             arm.summary_path, GATE_SPLIT)],
        "numbers": numbers,
    }


# --------------------------------------------------------------------------- figure E
def figure_e(ablation: dict, out: Path, synthetic: bool) -> dict:
    """The ablation table as a figure: which change, if any, moved S2."""
    rows = ablation.get("rows", [])
    steps = ablation.get("attribution", [])
    first = (ablation.get("s2") or {}).get("first_clearing_arm")
    numbers: List[str] = []

    caption = ("Rows are the four arms plus the P3-2 baseline for context. B-A isolates "
               "continuous alpha sampling, C-B isolates the m = -ln(1-alpha) coordinate, C-D "
               "asks whether multi-wall training was necessary. The P3-2 row is matched by "
               "experimental role, not by construction, and arm A inherits the P3-2 dataset in "
               "which one holdout was an EXTRAPOLATION -- so a null on arm A cannot separate "
               "'the renderer did not help' from 'that holdout was unfair'. {}".format(
                   SCOPE_NOTE))
    fig = new_figure("The ablation: which fix moved the held-out slab?", caption, banner=True,
                     banner_lines=2)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0])
    axt = fig.add_subplot(gs[0, 0])
    axd = fig.add_subplot(gs[1, 0])

    # -- the table
    axt.set_axis_off()
    header = ["arm", "conditioning", "val LSD\n(dB)"]
    for sp in SPLIT_KEYS:
        header.append("{}\nslope".format(sp.split("_")[0]))
        header.append("{}\ngain".format(sp.split("_")[0]))
    header += ["rho slab\n(kappa)", "rho vs\nRAW", "S2\nverdict"]
    # Abbreviated so the column can stay narrow enough for 16 columns on one canvas; the
    # full cond_source string is in ablation.json and in EVAL.md's provenance table.
    short_cond = {"geom_alpha_fourier": "fourier", "m_linear": "m_linear",
                  "geom_alpha": "geom_alpha"}
    cells, colours = [], []
    for r in rows:
        p = (r.get("verdict") or {}).get("passed")
        cs = r.get("cond_source") or "n/a"
        line = [r.get("letter", "?"),
                "{}{}".format(short_cond.get(cs, cs),
                              "/{}".format(r.get("cond_dim")) if r.get("cond_dim") else ""),
                _u(r.get("in_dist_val_lsd_db"), 3)]
        for sp in SPLIT_KEYS:
            b = (r.get("splits") or {}).get(sp, {}) or {}
            line += [_u(b.get("edit_bw_slope"), 3), _u(b.get("edit_gain"), 3)]
        line += [_u(r.get("rho_slab_local"), 3), _u(r.get("rho_vs_raw_theory_median"), 3),
                 "PASS" if p else ("FAIL" if p is not None else "not eval'd")]
        cells.append(line)
        base = "#f6f8fa" if r.get("is_baseline") else "white"
        row_c = [base] * len(line)
        row_c[-1] = "#d2f2d9" if p else ("#ffd7d5" if p is not None else "#eaeef2")
        # the S2 columns are the gate; tint them so the eye lands there first
        i_s2 = 3 + 2 * SPLIT_KEYS.index(GATE_SPLIT)
        row_c[i_s2] = row_c[i_s2 + 1] = "#fff3cd" if not r.get("is_baseline") else "#f0e6c8"
        if r.get("arm") == first:
            row_c[0] = "#d2f2d9"
        colours.append(row_c)
        numbers.append("E row {}: cond={}, val_lsd={}, S2 slope={}, S2 gain={}, rho={}, "
                       "verdict={}".format(
                           r.get("letter"), r.get("cond_source") or "n/a",
                           _u(r.get("in_dist_val_lsd_db"), 4),
                           _u(r.get("s2_edit_bw_slope"), 4), _u(r.get("s2_edit_gain"), 4),
                           _u(r.get("rho_slab_local"), 4),
                           "PASS" if p else ("FAIL" if p is not None else "not evaluated")))

    widths = [0.045, 0.105] + [0.055] * (len(header) - 5) + [0.075, 0.055, 0.075]
    tb = axt.table(cellText=cells, colLabels=header, bbox=[0.0, 0.02, 1.0, 0.88],
                   cellLoc="center", cellColours=colours, colWidths=widths,
                   colColours=["#eaeef2"] * len(header))
    tb.auto_set_font_size(False)
    tb.set_fontsize(11.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if r == 0:
            cell.set_text_props(fontweight="bold")
        elif c == 0:
            cell.set_text_props(fontweight="bold")
    axt.set_title("Per-split edit slope and edit_gain, with the S2 gate columns highlighted",
                  fontsize=16, loc="left")

    # -- the attribution deltas
    labels = ["{}\n{}".format(st["step"], st["comparison"]) for st in steps]
    keys = [("s2_edit_bw_slope", "S2 edit slope", C_MODEL),
            ("s2_edit_bw_pearson", "S2 pearson", C_GT),
            ("rho_slab_local", "rho (slab)", C_ISM)]
    x = np.arange(len(steps))
    width = 0.26
    any_delta = False
    for i, (k, lab, col) in enumerate(keys):
        vals = [float((st.get("delta") or {}).get(k, np.nan)) for st in steps]
        any_delta = any_delta or any(_fin(v) for v in vals)
        axd.bar(x + (i - 1) * width, vals, width, color=col, label=lab, edgecolor="white",
                zorder=3)
        for xx, val in zip(x, vals):
            if _fin(val):
                axd.text(xx + (i - 1) * width, val, "{:+.2f}".format(val), ha="center",
                         va="bottom" if val >= 0 else "top", fontsize=11, color=col,
                         fontweight="bold")
        for st, val in zip(steps, vals):
            numbers.append("E delta {} [{}] {}: {}".format(
                st["step"], st["comparison"], k, _fmt(val, 4)))
    if any_delta:
        axd.axhline(0, color="#24292f", lw=1.0, zorder=2)
    axd.set_xticks(x)
    axd.set_xticklabels(labels, fontsize=12.5)
    # Margins on the category axis: the outermost tick labels are two lines wide and run off
    # the canvas at the default 0.05 margin.
    axd.set_xlim(-0.62, len(steps) - 0.38)
    axd.set_ylabel("change in the S2 metric (to - from)")
    axd.set_title("Attribution: what each fix bought on the held-out slab", fontsize=16,
                  loc="left")
    axd.grid(axis="y", alpha=0.3, zorder=0)
    despine(axd)
    if any_delta:
        axd.legend(fontsize=12.5, ncol=3, framealpha=0.95)
    else:
        # An empty axes with an auto y-scale reads as "the deltas are zero". They are not
        # measured at all, and the difference matters.
        axd.set_yticks([])
        axd.text(0.5, 0.55, "No ladder delta is computable yet — {} of {} arms evaluated"
                            "\n(missing: {})\nre-run scripts/p3_2b_ablation.py when the "
                            "remaining evals land".format(
                                len(ablation.get("arms_evaluated", [])),
                                len(ablation.get("arms_expected", [])),
                                ", ".join(ablation.get("arms_missing", [])) or "none"),
                 transform=axd.transAxes, ha="center", va="center", fontsize=15,
                 color=C_NULL, style="italic")

    if first:
        letter = next((r.get("letter") for r in rows if r.get("arm") == first), first)
        step = next((st for st in steps if st.get("to") == first), None)
        msg = "S2 PASSED -- first cleared by arm {}{}".format(
            letter, "; the change that produced it: {} ({})".format(
                step["step"], step["comparison"]) if step else "")
        add_banner(fig, msg, fontsize=15, facecolor="#d2f2d9", edgecolor="#1a7f37")
    else:
        n_eval = len(ablation.get("arms_evaluated", []))
        add_banner(fig, "S2 NOT PASSED by any evaluated arm ({} of {} arms evaluated) -- "
                        "no fix on this ladder clears the pre-registered thresholds yet".format(
                            n_eval, len(ablation.get("arms_expected", []))),
                   fontsize=15, facecolor="#ffd7d5", edgecolor=C_MODEL)
    numbers.append("E s2.first_clearing_arm: {}".format(first or "none"))

    add_caption(fig)
    add_watermark(fig, synthetic)
    size = save(fig, out)
    return {
        "filename": out.name, "png_width": size[0], "png_height": size[1],
        "shows": ("The cross-arm ablation table -- per-split edit slope and edit_gain, "
                  "in-distribution val LSD and kappa-scaled rho for each arm plus the P3-2 "
                  "baseline -- with the attribution deltas for each rung of the ladder and a "
                  "banner naming the first arm (if any) to clear the S2 gate."),
        "source_files": ["{} :: rows[], attribution[], s2.first_clearing_arm".format(
            ablation.get("_path", "outputs/p3_2b/ablation.json"))],
        "numbers": numbers,
    }


# --------------------------------------------------------------------------- synthetic
def synthesize(dest: Path) -> Tuple[Path, Path, Path]:
    """Fabricate schema-valid summary.json / m_response.json / ablation.json.

    Values are deliberately arbitrary; the point is only that every panel has something of
    the right shape to lay out. Figures built from these are watermarked and written to a
    separate, gitignored directory.
    """
    import subprocess
    import sys

    rng = np.random.RandomState(11)
    dest.mkdir(parents=True, exist_ok=True)
    kappa = 1.6607564051417665
    c_sound = 343.0
    arm = "p3_2b_C_cont_mlinear"
    mats = ["M1", "A030", "M2", "M3"]
    m_base = m_of_alpha(ALPHA_BASELINE)
    gain_model = 0.62                       # the fabricated model recovers ~62% of the law

    # ---- selectivity matrix
    sm: Dict[str, dict] = {}
    L0, W0 = 4.50, 3.80
    for mat in mats:
        a = alpha_of(mat)
        sm[mat] = {}
        for wall in WALLS_2D:
            own = WALL_OWN_FAMILY[wall]
            D = L0 if own == "x_axial" else W0
            th_own = kappa * c_sound / (4.0 * np.pi * D) * (m_of_alpha(a) - m_base)
            sm[mat][wall] = {}
            for fam in FAMILIES:
                if fam == own:
                    th = th_own
                elif fam == "tangential":
                    th = 0.74 * th_own
                else:
                    th = 0.0
                gt = th + rng.normal(0, 0.12)
                pred = gain_model * gt + rng.normal(0, 0.20)
                sm[mat][wall][fam] = {"gt_d_bw": float(gt), "pred_d_bw": float(pred),
                                      "theory_d_bw": float(th),
                                      "residual_d_bw": float(pred - gt),
                                      "n": int(20 + rng.randint(0, 14))}

    # ---- splits
    def split_block(n_cfg, n_cells, slope, gain):
        return {
            "n_configs": n_cfg, "n_cells": n_cells, "n_modes_candidate": int(n_cells * 2.0),
            "frac_modes_dropped": float(0.30 + 0.10 * rng.rand()),
            "fidelity": {"mag_corr": 0.889, "band_lsd_db": float(2.8 + 0.4 * rng.rand()),
                         "phase_corr_mw": 0.888, "rir_pearson": 0.904, "t20_rel_err": 0.72},
            "null_fidelity": {"mag_corr": 0.882, "band_lsd_db": float(3.1 + 0.3 * rng.rand()),
                              "phase_corr_mw": 0.882, "rir_pearson": 0.899,
                              "t20_rel_err": 0.66},
            "edit": {"E_BW_hz": float(1.9 + rng.rand()), "edit_bw_slope": float(slope),
                     "edit_bw_pearson": float(min(0.99, slope + 0.16)),
                     "edit_gain": float(gain), "E_LVL_db": 1.44},
            "edit_detail": {"gt_effect_size_hz": 5.0, "pred_effect_size_hz": float(5.0 * slope)},
            "by_family": {f: {"E_BW_hz": float(1.6 + rng.rand()), "n": 200, "gt_d_bw": 2.6,
                              "pred_d_bw": float(2.6 * slope)} for f in FAMILIES},
            "per_combo": {},
        }

    splits = {
        "S1_unseen_geom_nonslab_1wall": split_block(100, 756, 0.67, 1.04),
        "S2_unseen_geom_slab": split_block(20, 162, 0.57, 1.07),
        "S3_seen_geom_slab": split_block(80, 1055, 0.38, 1.36),
        "S4_unseen_geom_alpha030": split_block(40, 330, 0.32, 1.00),
        "S5_unseen_geom_2wall": split_block(40, 317, 0.81, 1.15),
    }
    for combo, wall, a in (("west0.50", "west", 0.50), ("north0.70", "north", 0.70)):
        splits["S2_unseen_geom_slab"]["per_combo"][combo] = {
            "n_configs": 10, "n_cells": 81,
            "edit": {"gt_effect_size_hz": float(4.0 + 2.0 * rng.rand()),
                     "pred_effect_size_hz": float(gain_model * (4.0 + 2.0 * rng.rand())),
                     "edit_bw_slope": 0.57, "edit_gain": 1.07}}

    rho = gain_model
    def sg(r):                                                            # noqa: E306
        return {"rho_median": float(r), "rho_ci95": [float(r - 0.10), float(r + 0.14)],
                "a_fit_median": float(r * 11.22), "a_theory_median": 11.22,
                "a_fit_gt_median": 11.46, "rho_gt_median": 1.006, "n_cells": 16,
                "frac_modes_dropped": 0.28}

    summary = {
        "arm": arm, "checkpoint": "outputs/p3_2/{}/ckpt_iter0010000.pt".format(arm),
        "iter": 10000, "in_dist_val_lsd_db": 1.742, "cond_source": "m_linear", "cond_dim": 60,
        "splits": splits,
        "slope_fit": {"aggregate": {"own_family": {"all": sg(rho), "non_slab": sg(rho * 0.97),
                                                   "slab_local": sg(rho)},
                                    "orthogonal_family": {"a_fit_median": 0.50,
                                                          "n_cells": 40}},
                      "per_cell": [], "kappa": kappa,
                      "rho_vs_raw_theory_median": float(rho * kappa)},
        "controls": {"C2_floor_hz": 0.040, "C3_conditioning_identity": True},
        "selectivity_matrix": sm,
        "selectivity_index": {"gt": 18.07, "pred": 9.14, "theory": 32.92, "floor_hz": 0.15},
        "slabs_m": {"holdout_slabs_m": {k: list(v) for k, v in HOLDOUT_SLABS.items()},
                    "slab_combos": [["west", 0.5], ["north", 0.7]]},
        "meta": {"band_hz": [0.0, 300.0], "kappa": kappa, "total_iters": 60000,
                 "mid_training": True, "n_train_configs": 960, "manifest_sha": "0" * 64,
                 "n_geometries": 50},
    }
    from aaf.eval import p3_2b_accept

    summary["verdict"] = p3_2b_accept.verdict(
        arm, splits[GATE_SPLIT], summary["slope_fit"], iter_=10000, mid_training=True)

    # ---- m_response: 3 geometries x 4 walls x 20 swept alphas
    m_pts = np.linspace(0.05, 1.55, 20)
    geoms = []
    for gid, (role, L, W) in enumerate((("min_aspect", 3.27, 4.98),
                                        ("median_aspect", 5.56, 4.90),
                                        ("max_aspect", 5.93, 3.18)), start=1):
        walls = {}
        for wall in WALLS_2D:
            own = WALL_OWN_FAMILY[wall]
            D = L if own == "x_axial" else W
            a_th = kappa * c_sound / (4.0 * np.pi * D)
            a_raw = c_sound / (4.0 * np.pi * D)
            pts = []
            for m in m_pts:
                dm = float(m - m_base)
                gt_own = a_th * dm + rng.normal(0, 0.05)
                pr_own = gain_model * a_th * dm + rng.normal(0, 0.10)
                pts.append({
                    "alpha": float(alpha_of_m(m)), "m": float(m), "d_m": dm,
                    "in_slab": bool(in_slab(wall, alpha_of_m(m))),
                    "alpha_above_arm_A_max": bool(alpha_of_m(m) > 0.70),
                    "gt_source": "simulated",
                    "pred": {"own": {"d_bw_mean": float(pr_own), "d_bw_sem": 0.09,
                                     "n_modes": 3, "n_modes_total": 3},
                             "orth": {"d_bw_mean": float(rng.normal(0, 0.05)),
                                      "d_bw_sem": 0.02, "n_modes": 4, "n_modes_total": 4}},
                    "gt": {"own": {"d_bw_mean": float(gt_own), "d_bw_sem": 0.01,
                                   "n_modes": 3, "n_modes_total": 3},
                           "orth": {"d_bw_mean": float(rng.normal(0, 0.02)),
                                    "d_bw_sem": 0.01, "n_modes": 4, "n_modes_total": 4}},
                    "theory_d_bw": float(a_th * dm), "theory_d_bw_orth": 0.0})
            fit_gt = float(np.sum([p["d_m"] * p["gt"]["own"]["d_bw_mean"] for p in pts])
                           / np.sum([p["d_m"] ** 2 for p in pts]))
            fit_pr = float(np.sum([p["d_m"] * p["pred"]["own"]["d_bw_mean"] for p in pts])
                           / np.sum([p["d_m"] ** 2 for p in pts]))
            walls[wall] = {
                "axis": "x" if own == "x_axial" else "y", "own_family": own,
                "orth_family": "y_axial" if own == "x_axial" else "x_axial",
                "D_m": D, "a_theory_hz_per_m": a_th, "a_theory_raw_hz_per_m": a_raw,
                "slab_m": list(HOLDOUT_SLABS[wall]) if wall in HOLDOUT_SLABS else None,
                "n_modes_own": 3, "n_modes_orth": 4,
                "fit": {"a_theory_hz_per_m": a_th, "a_theory_raw_hz_per_m": a_raw,
                        "gt_own": {"a": fit_gt, "r2": 0.999, "n": 20},
                        "rho_gt": fit_gt / a_th, "rho_gt_raw_theory": fit_gt / a_raw,
                        "gt_orth": {"a": 0.45, "r2": 0.99, "n": 20},
                        "pred_own": {"a": fit_pr, "r2": 0.97, "n": 20},
                        "rho_pred": fit_pr / a_th, "rho_pred_raw_theory": fit_pr / a_raw,
                        "pred_orth": {"a": 0.05, "r2": 0.22, "n": 20}},
                "points": pts}
        geoms.append({"role": role, "geom_id": gid, "L": L, "W": W,
                      "aspect": round(L / W, 4), "modes": [], "walls": walls})
    mresp = {"schema": "p3_2b.m_response/1", "arm": arm,
             "checkpoint": summary["checkpoint"], "iter": 10000, "cond_source": "m_linear",
             "kappa": kappa, "gt_source": "outputs/p3_2b/mresponse_gt.json",
             "arm_A_alpha_max": 0.70, "geometries": geoms,
             "meta": {"band_hz": [0.0, 300.0], "n_points": 20}}

    ed = dest / "eval" / arm
    ed.mkdir(parents=True, exist_ok=True)
    (ed / "summary.json").write_text(json.dumps(summary, indent=1))
    (ed / "m_response.json").write_text(json.dumps(mresp, indent=1))

    # ---- ablation.json, produced by the real script over fabricated arm summaries so the
    #      two scripts cannot drift in what figure E expects.
    synth_dir = dest / "ablation_out"
    subprocess.check_call(
        [sys.executable, str(Path(__file__).resolve().parent / "p3_2b_ablation.py"),
         "--synthetic", "--out-dir", str(synth_dir)])
    return ed / "summary.json", ed / "m_response.json", synth_dir / "synthetic" / "ablation.json"


# --------------------------------------------------------------------------- manifest
def write_manifest(path: Path, arm: Arm, ablation: Optional[dict], results: List[dict],
                   synthetic: bool) -> None:
    s = arm.summary
    v = arm.verdict
    out: List[str] = []
    A = out.append
    A("# P3-2b — figure manifest (meeting pack)")
    A("")
    if synthetic:
        A("> **SYNTHETIC RUN — DO NOT USE.** Every number below was fabricated by "
          "`--synthetic` for a layout check.")
        A("")
    A("Five figures for the P3-2b chunk: one model conditioned on the room's geometry and its "
      "four wall absorptions, asked to render edited rooms **zero-shot** — the conditioning "
      "vector is computed from the physical parameters alone, no measurement of the target "
      "config is read, and nothing is optimised per config.")
    A("")
    A("All numbers below are read at run time from the JSON files named in each row. Nothing "
      "on any figure is hand-entered. The only quantities not read from a file are the slope "
      "standard errors on figure B (re-derived from the same published sweep points, because "
      "the m-response schema carries `a` and `r2` but no SE) and the alpha <-> m conversion, "
      "taken from the frozen `aaf.data.mat_configs_cont` helpers.")
    A("")
    A("## Provenance")
    A("")
    A("| field | value |")
    A("|---|---|")
    A("| arm | `{}` |".format(arm.arm))
    A("| conditioning | `{}`, cond_dim {} |".format(s.get("cond_source"), s.get("cond_dim")))
    A("| checkpoint | `{}` |".format(s.get("checkpoint")))
    A("| iteration | {}{} |".format(
        s.get("iter"),
        " of {} — TRAINING WAS STILL RUNNING; this is a mid-training snapshot".format(
            arm.meta.get("total_iters")) if arm.meta.get("mid_training") else ""))
    A("| in-distribution val LSD | {} dB |".format(_u(s.get("in_dist_val_lsd_db"), 4)))
    A("| band | {} Hz |".format(arm.meta.get("band_hz")))
    A("| configs evaluated | {} over {} geometries |".format(
        arm.meta.get("n_configs_evaluated"), arm.meta.get("n_geometries")))
    A("| training configs | {} |".format(arm.meta.get("n_train_configs")))
    A("| manifest sha256 | `{}` |".format(arm.meta.get("manifest_sha")))
    A("| held-out slabs (m) | {} |".format(
        ", ".join("{} m in [{:.2f}, {:.2f}] (brackets alpha={:.2f})".format(
            w, sl[0], sl[1], alpha_of_m(float(np.mean(sl)))) for w, sl in arm.slabs.items())))
    A("| kappa (frozen, P3-2 gate) | {:.10f} |".format(arm.kappa))
    A("| acceptance spec / sha256 | `{}` / `{}` |".format(
        (v.get("thresholds") or {}).get("spec", "n/a"), v.get("spec_sha")))
    A("| sources | `{}`{}{} |".format(
        arm.summary_path,
        ", `{}`".format(arm.mresp_path) if arm.mresp_path else "",
        ", `{}`".format(arm.ablation_path) if arm.ablation_path else ""))
    A("")
    A("## Scoping (must accompany any verbal claim)")
    A("")
    A("The wall selectivity that makes this chunk legible is a property of the **ISM "
      "simulator**: its reflection coefficient is real and angle-independent, so a pure "
      "x-axial mode sees *exactly zero* damping from the north/south walls. Real "
      "locally-reacting walls follow Kuttruff and would show only ~2:1, with **no invariant "
      "family**. The claim is therefore *\"the model learns the simulator's per-wall law\"* — "
      "not *\"the model learns room acoustics\"*.")
    A("")
    A("## The kappa correction (every rho on every figure depends on it)")
    A("")
    A("The bandwidth estimator returns a **calibrated** -3 dB width, not the raw Lorentzian "
      "width. The P3-2 physics gate's T5 fit gives `BW = 0.302 + {:.4f} * (gamma/pi)`. The "
      "intercept cancels in a paired delta; **the slope does not**. So the theoretical slope "
      "of a measured delta-bandwidth against delta-m is".format(arm.kappa))
    A("")
    A("```")
    A("a_theory = kappa * c / (4*pi*D),   kappa = {:.16f}".format(arm.kappa))
    A("D = L for west/east on the x-axial family; D = W for south/north on the y-axial family")
    A("a_theory = 0 for the orthogonal family")
    A("```")
    A("")
    A("Every rho on every figure is `a_fit / a_theory` against that **kappa-scaled** line, and "
      "the raw-Lorentzian comparison is printed alongside (figure B banner, figure E column "
      "`rho vs RAW`) so the correction is visible rather than assumed. Scoring against the raw "
      "value would hand a perfect model rho ~ 0.60 and fail it.")
    A("")
    A("## S2 verdict")
    A("")
    A("```")
    A(str(v.get("one_line")))
    A("```")
    A("")
    A("| criterion | value | op | threshold | result |")
    A("|---|---|---|---|---|")
    for c in v.get("criteria", []):
        A("| `{}` | {} | {} | {} | {} |".format(
            c.get("name"), _u(c.get("value"), 4), c.get("op"), _u(c.get("threshold"), 3),
            "PASS" if c.get("passed") else "**FAIL**"))
    A("")
    if ablation:
        first = (ablation.get("s2") or {}).get("first_clearing_arm")
        A("Cross-arm: {}".format(
            "arm `{}` is the first in ladder order to clear the S2 gate.".format(first)
            if first else "**no evaluated arm clears the S2 gate.** Arms evaluated: {}; "
                          "missing: {}.".format(ablation.get("arms_evaluated"),
                                                ablation.get("arms_missing"))))
        A("")
        A("Honesty note carried from `EVAL.md`: arm A inherits the P3-2 dataset, in which one "
          "holdout was an **EXTRAPOLATION**, so arm A cannot separate *the renderer did not "
          "help* from *that holdout was unfair*. Arms B/C/D use the P3-2b manifest, whose "
          "held-out slabs are strictly interior to the sampled m range.")
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
        if r.get("note_se"):
            A("- **Derived on the figure:** {}".format(r["note_se"]))
        A("- **Numbers on the figure:**")
        A("")
        A("  ```")
        for n in r["numbers"]:
            A("  " + n)
        A("  ```")
        A("")
    A("## Headline numbers (all from `summary.json`)")
    A("")
    A("| split | n cfg | n cells | band LSD (dB) | E_BW (Hz) | edit slope | edit pearson | "
      "edit gain |")
    A("|---|---|---|---|---|---|---|---|")
    for sp in SPLIT_KEYS:
        b = (s.get("splits") or {}).get(sp)
        if not b:
            continue
        e = b.get("edit", {})
        A("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            sp, b.get("n_configs"), b.get("n_cells"),
            _u((b.get("fidelity") or {}).get("band_lsd_db"), 3), _u(e.get("E_BW_hz"), 3),
            _u(e.get("edit_bw_slope"), 3), _u(e.get("edit_bw_pearson"), 3),
            _u(e.get("edit_gain"), 3)))
    A("")
    A("**S2 is the chunk's question.** A strong S1 or S3 with a dead S2 is exactly the P3-2 "
      "result being re-tested and must not be reported as progress.")
    A("")
    A("## Reproduction")
    A("")
    A("```bash")
    A("export PYTHONPATH=\"$PWD\"")
    A("python scripts/p3_2b_ablation.py                       # CPU: ablation.json + EVAL.md")
    A("python scripts/make_p3_2b_figures.py --arm {}   # CPU: figures + this manifest".format(
        arm.arm))
    A("```")
    A("")
    path.write_text("\n".join(out) + "\n")


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", default="outputs/p3_2b/eval")
    ap.add_argument("--arm", default="p3_2b_C_cont_mlinear",
                    help="arm whose eval drives figures A-D")
    ap.add_argument("--ablation", default="outputs/p3_2b/ablation.json")
    ap.add_argument("--out", default="outputs/p3_2b/meeting_assets")
    ap.add_argument("--material", default="M3",
                    help="material shown on figure A (default M3, alpha=0.70)")
    ap.add_argument("--synthetic", action="store_true",
                    help="fabricate schema-valid inputs and render watermarked layout checks")
    ap.add_argument("--only", default=None, help="comma-separated subset of A,B,C,D,E")
    args = ap.parse_args()

    plt.rcParams.update(RC)

    tmp = None
    if args.synthetic:
        tmp = Path(tempfile.mkdtemp(prefix="p3_2b_synthetic_"))
        sp, mp, abl = synthesize(tmp)
        out_dir = Path(args.out).parent / "meeting_assets_SYNTHETIC"
        print("SYNTHETIC inputs in {}".format(tmp))
    else:
        sp = Path(args.eval_root) / args.arm / "summary.json"
        mp = Path(args.eval_root) / args.arm / "m_response.json"
        abl = Path(args.ablation)
        out_dir = Path(args.out)
        if not sp.exists():
            raise SystemExit(
                "missing input: {}\nrun the P3-2b eval for this arm first, or use --synthetic "
                "for a layout check".format(sp))
        if not mp.exists():
            print("WARNING: {} missing -- figure B will be empty".format(mp))
        if not abl.exists():
            raise SystemExit(
                "missing input: {}\nbuild it first:  python scripts/p3_2b_ablation.py".format(
                    abl))

    arm = Arm(sp, mp, abl)
    ablation = arm.ablation
    if ablation is not None:
        ablation["_path"] = str(abl)
    print("arm={} checkpoint={} iter={}".format(arm.arm, arm.summary.get("checkpoint"),
                                                arm.summary.get("iter")))
    out_dir.mkdir(parents=True, exist_ok=True)

    want = {c.strip().upper() for c in args.only.split(",")} if args.only else set("ABCDE")
    results: List[dict] = []
    plan = [
        ("A", "A_pick_your_wall.png", lambda p: figure_a(arm, args.material, p, args.synthetic)),
        ("B", "B_m_response.png", lambda p: figure_b(arm, p, args.synthetic)),
        ("C", "C_selectivity_matrix.png", lambda p: figure_c(arm, p, args.synthetic)),
        ("D", "D_s2_headline.png", lambda p: figure_d(arm, p, args.synthetic)),
        ("E", "E_ablation.png", lambda p: figure_e(ablation or {}, p, args.synthetic)),
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
        write_manifest(out_dir / "FIGURE_MANIFEST.md", arm, ablation, results, args.synthetic)
        print("wrote {}".format(out_dir / "FIGURE_MANIFEST.md"))

    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
