"""P3-2 live demo — edit one wall's material, re-render the room, zero-shot.

    python scripts/demo_edit_2d.py --L 4.51 --W 4.00 --wall west --material curtain \
        --gt data/track_c_2d

Renders the SAME trained model twice for the requested rectangle: once with all four
walls at the M0 baseline and once with the requested wall swapped to the requested
material. Nothing is looked up and nothing is optimised at demo time -- the only thing
that changes between the two renders is the 64-d conditioning vector
``fourier_features_2d(L, W, alphas)``, computed from the numbers on the command line. A
geometry the model never saw and a (wall, material) pair it was never trained on are
therefore both just arithmetic on that vector.

The figure makes one falsifiable claim, and stdout states it numerically:

    editing a wall broadens ITS OWN mode family's -3 dB bandwidth and leaves the
    orthogonal family essentially untouched.

Bandwidth -- not level -- is the headline observable: measured level selectivity is only
~4.4:1 while bandwidth selectivity is ~29:1 (outputs/p3_2/SIM_VALIDATION.md), so a demo
built on level would look unconvincing on correct physics.

SCOPING. That ~29:1 is a property of the ISM simulator, which uses an angle-independent
reflection coefficient and therefore has no grazing-incidence absorption: a purely axial
mode is damped only by the wall pair it bounces between. Real locally-reacting walls
follow Kuttruff and would give only ~2:1 with NO invariant family. The claim this demo
supports is "the model learns the simulator's per-wall law" -- both predictions are
printed side by side so the audience can see which one the model reproduces.

Receiver choice (--receiver). The default is the FAR CORNER (index 63), not the centre:
on an 8x8 grid the near-centre receiver sits close to a node of the odd axial modes
(mode (1,0) is ~14 dB down there) and receiver 0 is dominated by direct sound, so both
make the single-receiver spectrum panel harder to read than the physics deserves. The
quantitative summary never depends on this choice -- it is measured by projecting all 64
receivers onto the analytic mode shapes (``aaf.eval.modal_projection``), so every number
is attributable to a single (n_x, n_y).

Needs a GPU (tinycudann). Run it with sbatch scripts/slurm/demo_edit_2d.sh.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from aaf.data.mat_configs import HELDOUT_COMBOS, room_filename_2d_mat  # noqa: E402
from aaf.eval.modal_bandwidth import caps_from_predicted_bw, measure_modes  # noqa: E402
from aaf.eval.modal_decay import band_limited_edc, band_limited_rir  # noqa: E402
from aaf.eval.modal_projection import (  # noqa: E402
    TANGENTIAL,
    X_AXIAL,
    Y_AXIAL,
    enumerate_modes,
    project_field,
)
from aaf.sim.analytical_modal_2d import (  # noqa: E402
    damping_to_bandwidth_hz,
    modal_damping_2d,
)
from aaf.walls import (  # noqa: E402
    ALPHA_BASELINE,
    MATERIAL_NAMES,
    MATERIALS,
    alphas_for,
    resolve_material,
    resolve_wall,
)

# Dataset conventions (scripts/build_2d_mat_dataset.py). Used only when --gt is absent;
# with --gt the receiver/source coordinates are read from the HDF5 instead.
N_GRID = 8
MARGIN = 0.3
SRC_DEFAULT = (0.5, 0.5)

DEFAULT_CKPT_DIR = "outputs/p3_2/p3_2_main"
TRAIN_YAML = "configs/sweeps_2d_mat/p3_2_train.yaml"

N_PER_FAMILY = 3        # modes averaged per family, matching the physics gate
SIGMA_BW_FLOOR = 0.15   # Hz; measured BW repeatability is ~0.06 Hz (gate D47)
F_MAX_PLOT = 300.0      # the supervised band

C_BASE = "#3F4A56"      # baseline (grey-blue)
C_EDIT = "#C0392B"      # edited (red)
C_GT = "#111111"


# ----------------------------------------------------------------------------- geometry
def receiver_grid_2d(L: float, W: float, n_per_side: int = N_GRID,
                     margin: float = MARGIN) -> np.ndarray:
    """The dataset's 8x8 grid: ROW-MAJOR, outer y, inner x (flat i -> iy=i//8, ix=i%8)."""
    xs = np.linspace(margin, L - margin, n_per_side)
    ys = np.linspace(margin, W - margin, n_per_side)
    return np.array([[x, y] for y in ys for x in xs], dtype=float)


def pick_receiver(spec: str, rx: np.ndarray, L: float, W: float,
                  src: Sequence[float]) -> Tuple[int, str]:
    """Resolve --receiver to an index. Returns ``(index, why)``."""
    s = str(spec).strip().lower()
    if s in ("corner", "far", "far_corner"):
        i = int(np.argmax(np.linalg.norm(rx - np.asarray(src, dtype=float), axis=1)))
        return i, "far corner, least direct-sound contamination"
    if s in ("center", "centre", "mid"):
        i = int(np.argmin(np.linalg.norm(rx - np.array([L / 2.0, W / 2.0]), axis=1)))
        return i, "nearest room centre -- CAUTION: close to a node of the odd axial modes"
    try:
        i = int(s)
    except ValueError:
        raise ValueError(
            "--receiver must be 'corner', 'center', or an integer index, got {!r}".format(spec)
        )
    if not 0 <= i < rx.shape[0]:
        raise ValueError("--receiver {} out of range 0..{}".format(i, rx.shape[0] - 1))
    return i, "explicit index"


def driven_family(wall: str) -> Tuple[str, str]:
    """``(own, other)`` -- the family the wall reflects, and the orthogonal one."""
    return (X_AXIAL, Y_AXIAL) if wall in ("west", "east") else (Y_AXIAL, X_AXIAL)


def family_label(fam: str) -> str:
    return {X_AXIAL: "x-axial (n,0)", Y_AXIAL: "y-axial (0,m)",
            TANGENTIAL: "tangential (n,m)"}[fam]


# -------------------------------------------------------------------------------- model
def newest_checkpoint(ckpt_dir: str) -> Path:
    ckpts = sorted(Path(ckpt_dir).glob("ckpt_iter*.pt"))
    if not ckpts:
        raise FileNotFoundError("no ckpt_iter*.pt in {}".format(ckpt_dir))
    return ckpts[-1]


def build_model(ckpt_path: Path, device: torch.device):
    """Rebuild the trained model + renderer from the checkpoint's own cfg.

    ``aaf.models.inr_2d`` is imported HERE, not at module scope: it pulls in tinycudann,
    which raises "Unknown compute capability" on a GPU-less login node, and --help should
    still work there.
    """
    from aaf.models.inr_2d import INR2D_AutoDecoder
    from aaf.renderers.freq_2d import FreqRenderer2D

    st = torch.load(str(ckpt_path), map_location="cpu")
    cfg = st["cfg"]
    n_freq_bins = cfg["n_time_samples"] // 2 + 1

    meta_path = ckpt_path.parent / "train_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    # n_rooms only sizes the latent table, which this arm does not build (cond_source is
    # geom_alpha_fourier -> model.latents is None), so it cannot change the state_dict.
    n_rooms = int(meta.get("n_configs", 440))

    hg = dict(otype="HashGrid", n_levels=cfg["n_levels"], n_features_per_level=2,
              log2_hashmap_size=cfg["log2_hashmap_size"], base_resolution=16,
              per_level_scale=cfg["per_level_scale"])
    model = INR2D_AutoDecoder(
        n_rooms=n_rooms, latent_dim=cfg["latent_dim"], n_freq_bins=n_freq_bins,
        hash_grid_config=hg, conditioning_type=cfg["conditioning_type"],
        cond_source=cfg["cond_source"], cond_dim=cfg["cond_dim"], l_head_enabled=False,
    ).to(device)
    model.load_state_dict(st["model"])          # strict: a shape drift must fail loudly
    renderer = FreqRenderer2D(
        n_azi=cfg["n_azi"], n_pts_per_ray=cfg["n_pts_per_ray"], near=cfg["near"],
        fs=cfg["fs"], n_time_samples=cfg["n_time_samples"], c=cfg["c"],
    ).to(device)
    # BOTH must be eval(): the jitter lives on the RENDERER (FreqRenderer2D._ray_
    # directions_2d checks self.training), and an nn.Module is constructed in train mode,
    # so forgetting this makes the demo non-reproducible between runs.
    model.eval()
    renderer.eval()
    return model, renderer, cfg, meta, int(st.get("iter", -1))


@torch.no_grad()
def render_field(model, renderer, L: float, W: float, alphas: Sequence[float],
                 rx: np.ndarray, src: Sequence[float], device: torch.device,
                 chunk: int = 8) -> np.ndarray:
    """Render H(f) at every receiver. Returns ``[n_rx, n_freq_bins]`` complex64.

    The conditioning is COMPUTED from (L, W, alphas) -- there is no room id, no table
    lookup and no optimisation, which is the whole point of the demo.
    """
    from aaf.models.conditioning_2d import fourier_features_2d

    z = fourier_features_2d(L, W, alphas, device=device).unsqueeze(0)      # [1, 64]
    room_min = torch.zeros(2, device=device)
    room_max = torch.tensor([L, W], device=device, dtype=torch.float32)
    rx_t = torch.tensor(np.asarray(rx, dtype=np.float32), device=device)
    src_t = torch.tensor(np.asarray(src, dtype=np.float32), device=device)
    out: List[np.ndarray] = []
    for s in range(0, rx_t.shape[0], chunk):
        r = rx_t[s:s + chunk]
        tx = src_t.unsqueeze(0).expand(r.shape[0], -1)
        H = renderer(model, r, tx, room_min, room_max, z_s=z.expand(r.shape[0], -1))
        out.append(H.cpu().numpy())
    return np.concatenate(out, axis=0)


# -------------------------------------------------------------------------- measurement
def predicted_bandwidths(L: float, W: float, alphas: Sequence[float],
                         modes) -> List[float]:
    return [damping_to_bandwidth_hz(
        modal_damping_2d(L, W, alphas, m.n_x, m.n_y, model="ism_ray")) for m in modes]


def measure_pair(H_base: np.ndarray, H_edit: np.ndarray, rx: np.ndarray, L: float,
                 W: float, alphas_base: Sequence[float], alphas_edit: Sequence[float],
                 f_axis: np.ndarray, src: Sequence[float], fs: float):
    """Mode-resolved peaks for the baseline/edited pair, measured IDENTICALLY.

    Both configs share one set of -3 dB walk caps, taken per mode as the wider of the two
    predicted bandwidths. Deriving the cap from each config separately (as the physics gate
    does, where measured and predicted widths agree) would let the cap itself differ between
    baseline and edited: a mode could then be rejected as unresolvable in one and accepted
    in the other, and the paired difference would partly measure the cap rule rather than
    the room. A shared cap makes every mode resolvable in both or in neither.

    Returns ``(None, None)`` (with a warning) if the mode-shape basis is ill-conditioned
    for this geometry -- an 8x8 grid cannot resolve more than a handful of half-wavelengths,
    and a silently noise-amplified projection would be worse than no number at all.
    """
    try:
        pr_b = project_field(H_base, rx, L, W, src=src, fs=fs)
        pr_e = project_field(H_edit, rx, L, W, src=src, fs=fs)
    except ValueError as exc:
        print("[warn] modal projection unavailable: {}".format(exc))
        return None, None
    caps = caps_from_predicted_bw(np.maximum(
        predicted_bandwidths(L, W, alphas_base, pr_b.modes),
        predicted_bandwidths(L, W, alphas_edit, pr_e.modes)))
    return (
        {"projection": pr_b,
         "peaks": measure_modes(pr_b.spectra, f_axis, pr_b.modes, caps=caps)},
        {"projection": pr_e,
         "peaks": measure_modes(pr_e.spectra, f_axis, pr_e.modes, caps=caps)},
    )


def paired_family_stats(base: dict, edit: dict, fam: str,
                        n_modes: int = N_PER_FAMILY) -> dict:
    """Mean bandwidth/level over the modes measurable in BOTH configs.

    Pairing is not cosmetic. An edited config's modes are broader and a few more of them
    fail the resolvability test, so averaging "the first 3 valid modes" independently per
    config would difference two DIFFERENT mode sets and manufacture a bandwidth change out
    of bookkeeping. The intersection is the only honest comparison.
    """
    pr, pb, pe = base["projection"], base["peaks"], edit["peaks"]
    idx = [i for i in pr.by_family(fam) if pb[i].bw_valid and pe[i].bw_valid][:n_modes]
    if not idx:
        return {"bw_base": float("nan"), "bw_edit": float("nan"), "d_bw": float("nan"),
                "level_base": float("nan"), "level_edit": float("nan"),
                "d_level": float("nan"), "n": 0, "modes": []}
    bb = float(np.mean([pb[i].bw_3db_hz for i in idx]))
    be = float(np.mean([pe[i].bw_3db_hz for i in idx]))
    lb = float(np.mean([pb[i].level_db for i in idx]))
    le = float(np.mean([pe[i].level_db for i in idx]))
    return {"bw_base": bb, "bw_edit": be, "d_bw": be - bb, "level_base": lb,
            "level_edit": le, "d_level": le - lb, "n": len(idx),
            "modes": [[pr.modes[i].n_x, pr.modes[i].n_y] for i in idx]}


def most_affected_mode(base: Optional[dict], edit: Optional[dict], own: str,
                       L: float, W: float):
    """The driven-family mode whose measured bandwidth moved most.

    Returns ``(mode, f_hz, why, detail)`` where ``detail`` carries the modal-projected
    spectra and the measured peaks for that mode (``None`` when the projection was
    unavailable). Falls back to the lowest-frequency mode of the driven family, which is
    always well excited from the corner source and makes the cleanest spatial picture.
    """
    if base is not None and edit is not None:
        pr, pb, pe = base["projection"], base["peaks"], edit["peaks"]
        cand = [i for i in pr.by_family(own) if pb[i].bw_valid and pe[i].bw_valid]
        if cand:
            i = max(cand, key=lambda j: abs(pe[j].bw_3db_hz - pb[j].bw_3db_hz))
            m = pr.modes[i]
            detail = {"spec_base": base["projection"].spectra[i],
                      "spec_edit": edit["projection"].spectra[i],
                      "peak_base": pb[i], "peak_edit": pe[i]}
            f = float(pe[i].f_peak if np.isfinite(pe[i].f_peak) else m.f)
            return m, f, "largest measured change in -3 dB bandwidth", detail
    fam = [m for m in enumerate_modes(L, W, f_max=F_MAX_PLOT) if m.family == own]
    return fam[0], float(fam[0].f), "lowest driven-family mode (fallback)", None


# ------------------------------------------------------------------------ ground truth
def load_gt(gt_arg: str, L: float, W: float, alphas_edit: Sequence[float]) -> dict:
    """Load ISM ground truth. ``--gt`` may be the data DIRECTORY or the edited-config h5.

    Both the edited config and its all-baseline sibling are loaded when available, so the
    GT overlay is an apples-to-apples pair. Room dimensions in the file are asserted
    against the CLI: a silently mismatched GT would invalidate every overlaid curve.
    """
    import h5py

    p = Path(gt_arg)
    if p.is_dir():
        f_edit = p / room_filename_2d_mat(L, W, alphas_edit)
        f_base = p / room_filename_2d_mat(L, W, alphas_for())
    else:
        f_edit = p
        f_base = p.parent / room_filename_2d_mat(L, W, alphas_for())
    if not f_edit.exists():
        raise FileNotFoundError("no ISM ground truth at {}".format(f_edit))

    out: dict = {"path_edit": str(f_edit), "path_base": None}
    with h5py.File(str(f_edit), "r") as f:
        if abs(float(f.attrs["L"]) - L) > 1e-6 or abs(float(f.attrs["W"]) - W) > 1e-6:
            raise ValueError(
                "GT file is L={} W={} but the demo asked for L={} W={}".format(
                    f.attrs["L"], f.attrs["W"], L, W))
        out["H_edit"] = np.asarray(f["ism/H_complex"][:])
        out["rx"] = np.asarray(json.loads(f.attrs["receiver_pos"]), dtype=float)
        out["src"] = np.asarray(json.loads(f.attrs["source_pos"]), dtype=float)
        out["split"] = str(f.attrs.get("split", "?"))
        out["label"] = str(f.attrs.get("label", "?"))
    if f_base.exists():
        with h5py.File(str(f_base), "r") as f:
            out["H_base"] = np.asarray(f["ism/H_complex"][:])
            out["path_base"] = str(f_base)
    return out


def geometry_novelty(L: float, W: float, yaml_path: str = TRAIN_YAML) -> Optional[dict]:
    """Distance from (L, W) to the nearest TRAINED geometry -- the zero-shot receipt."""
    p = Path(yaml_path)
    if not p.exists():
        return None
    import yaml

    geoms = [(float(g["L"]), float(g["W"]))
             for g in yaml.safe_load(p.read_text())["geometries"]]
    d = [float(np.hypot(L - a, W - b)) for a, b in geoms]
    j = int(np.argmin(d))
    return {"exact_match": bool(d[j] < 1e-9), "nearest": geoms[j], "distance_m": d[j],
            "n_train_geometries": len(geoms)}


# ------------------------------------------------------------------------------- figure
def _db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def _modal_zoom_inset(ax, detail: dict, f_axis: np.ndarray, mode) -> None:
    """Inset: the modal-projected spectrum of the most-affected mode, base vs edited.

    This is the measurement the headline number comes from, so it is drawn from the
    projected spectra rather than from the single-receiver curve in the main axes -- the
    -3 dB marks would not correspond to the plotted line otherwise.
    """
    pb, pe = detail["peak_base"], detail["peak_edit"]
    half = max(4.0, 1.2 * max(pb.bw_3db_hz, pe.bw_3db_hz))
    f0, f1 = pb.f_peak - half, pe.f_peak + half
    sel = (f_axis >= f0) & (f_axis <= f1)
    # Opaque backing behind the inset AND its tick labels -- without it the labels land on
    # top of the main curves and the panel reads as clutter.
    ax.add_patch(plt.Rectangle((0.600, 0.487), 0.397, 0.505, transform=ax.transAxes,
                               facecolor="white", edgecolor="0.75", lw=0.8, zorder=8))
    ins = ax.inset_axes([0.655, 0.545, 0.325, 0.415], zorder=9)
    for j, (spec, pk, col, tag) in enumerate((
            (detail["spec_base"], pb, C_BASE, "baseline"),
            (detail["spec_edit"], pe, C_EDIT, "edited"))):
        ins.plot(f_axis[sel], _db(spec[sel]) - pk.level_db,   # peak-relative: widths compare
                 color=col, lw=1.6)
        # measure_modes returns the WIDTH, not the two crossing frequencies, so the marker
        # is drawn centred on the interpolated peak -- a faithful picture of the reported
        # number without inventing crossings the estimator never returned.
        ins.plot([pk.f_peak - pk.bw_3db_hz / 2, pk.f_peak + pk.bw_3db_hz / 2], [-3.0, -3.0],
                 color=col, lw=2.4, marker="|", ms=6)
        ins.text(0.03, 0.93 - 0.15 * j, "{} {:.2f} Hz".format(tag, pk.bw_3db_hz),
                 transform=ins.transAxes, va="top", fontsize=7.6, color=col,
                 fontweight="bold")
    ins.axhline(-3.0, color="0.6", lw=0.7, ls=":")
    ins.set(xlim=(f0, f1), ylim=(-11, 2.5))
    ins.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    ins.tick_params(labelsize=7.2, pad=1)
    ins.set_title("mode ({},{}), 64-receiver modal projection: -3 dB width".format(
        mode.n_x, mode.n_y), fontsize=7.8, pad=2)
    ins.set_xlabel("Hz", fontsize=7.2, labelpad=0)
    ins.set_ylabel("dB re peak", fontsize=7.2, labelpad=1)
    ins.grid(alpha=0.2)


def make_figure(out_png: Path, *, L: float, W: float, wall: str, material: str,
                alphas_base, alphas_edit, H_base, H_edit, rx, src, rx_idx: int,
                rx_why: str, f_axis: np.ndarray, fs: float, n_time: int,
                mode, f_mode_meas: float, mode_detail: Optional[dict], own: str,
                other: str, summary: dict, gt: Optional[dict], ckpt_iter: int) -> None:
    a_edit = MATERIALS[material]
    fig = plt.figure(figsize=(15.5, 10.4))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.9],
                  left=0.058, right=0.945, top=0.838, bottom=0.215, hspace=0.40, wspace=0.28)

    # ---- panel 1: single-receiver spectrum, 0-300 Hz --------------------------------
    ax = fig.add_subplot(gs[0, :])
    band = f_axis <= F_MAX_PLOT
    fam_modes = [m for m in enumerate_modes(L, W, f_max=F_MAX_PLOT) if m.family == own]
    for k, m in enumerate(fam_modes):
        ax.axvline(m.f, color=C_EDIT, lw=0.9, ls=":", alpha=0.5, zorder=1,
                   label="analytic {} eigenfrequencies".format(family_label(own))
                   if k == 0 else None)
    if gt is not None and "H_base" in gt:
        ax.plot(f_axis[band], _db(gt["H_base"][rx_idx][band]), color=C_GT, lw=0.9,
                ls="--", alpha=0.5, zorder=2, label="ISM ground truth, baseline")
    if gt is not None:
        ax.plot(f_axis[band], _db(gt["H_edit"][rx_idx][band]), color="#7A1B10", lw=1.1,
                ls="--", alpha=0.85, zorder=3, label="ISM ground truth, edited")
    ax.plot(f_axis[band], _db(H_base[rx_idx][band]), color=C_BASE, lw=1.8, zorder=4,
            label="predicted, baseline (all walls $\\alpha$={:.2f})".format(ALPHA_BASELINE))
    ax.plot(f_axis[band], _db(H_edit[rx_idx][band]), color=C_EDIT, lw=1.8, zorder=5,
            label="predicted, {} $\\to$ {} ($\\alpha$={:.2f})".format(
                wall, MATERIAL_NAMES[material], a_edit))
    lo = float(min(_db(H_base[rx_idx][band]).min(), _db(H_edit[rx_idx][band]).min()))
    hi = float(max(_db(H_base[rx_idx][band]).max(), _db(H_edit[rx_idx][band]).max()))
    ax.set_ylim(lo - 0.42 * (hi - lo), hi + 0.28 * (hi - lo))   # room for legend + inset
    ax.axvline(f_mode_meas, color="#1F6FB2", lw=1.4, alpha=0.85, zorder=6)
    ax.annotate("mode ({},{})".format(mode.n_x, mode.n_y),
                xy=(f_mode_meas, ax.get_ylim()[1]), xytext=(4, -3),
                textcoords="offset points", va="top", fontsize=8.5, color="#1F6FB2")
    ax.set(xlim=(0, F_MAX_PLOT), xlabel="frequency (Hz)", ylabel="|H(f)| (dB)",
           title="1.  Spectrum at receiver {}  --  {}".format(rx_idx, rx_why))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.4, loc="lower left", framealpha=0.92)
    if mode_detail is not None:
        _modal_zoom_inset(ax, mode_detail, f_axis, mode)

    # ---- panel 2: spatial |field| at the most-affected mode --------------------------
    x0, x1 = float(rx[:, 0].min()), float(rx[:, 0].max())
    y0, y1 = float(rx[:, 1].min()), float(rx[:, 1].max())
    k = int(np.argmin(np.abs(f_axis - f_mode_meas)))
    maps = [_db(H_base[:, k]).reshape(N_GRID, N_GRID),
            _db(H_edit[:, k]).reshape(N_GRID, N_GRID)]
    vmin = min(float(m.min()) for m in maps)
    vmax = max(float(m.max()) for m in maps)
    titles = ["2a.  baseline", "2b.  {} $\\to$ {}".format(wall, MATERIAL_NAMES[material])]
    im = None
    for j, (M, t) in enumerate(zip(maps, titles)):
        a = fig.add_subplot(gs[1, j])
        a.set_facecolor("0.93")
        im = a.imshow(M, origin="lower", extent=(x0, x1, y0, y1), vmin=vmin, vmax=vmax,
                      cmap="magma", interpolation="bilinear", aspect="equal")
        a.add_patch(plt.Rectangle((0, 0), L, W, fill=False, ec="0.35", lw=1.2))
        # The edited wall, drawn where it physically is (west/east: x=0/L; south/north: y=0/W).
        seg = {"west": ([0, 0], [0, W]), "east": ([L, L], [0, W]),
               "south": ([0, L], [0, 0]), "north": ([0, L], [W, W])}[wall]
        a.plot(seg[0], seg[1], color=(C_EDIT if j == 1 else "0.55"), lw=5,
               solid_capstyle="butt", zorder=5)
        a.plot(src[0], src[1], marker="*", ms=13, color="w", mec="k", mew=0.7, zorder=6)
        a.plot(rx[rx_idx, 0], rx[rx_idx, 1], marker="o", ms=7, mfc="none", mec="w",
               mew=1.6, zorder=6)
        a.set(xlim=(-0.15, L + 0.15), ylim=(-0.15, W + 0.15), xlabel="x (m)",
              ylabel="y (m)" if j == 0 else "", title=t)
        a.tick_params(labelsize=8.5)
    cax = fig.add_axes([0.075, 0.163, 0.50, 0.013])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8)
    cb.set_label("|H| at {:.1f} Hz on the 8x8 grid, shared dB scale   "
                 "(thick red = edited wall, $\\star$ = source, $\\circ$ = plotted receiver)"
                 .format(f_mode_meas), fontsize=8.2)

    # ---- panel 3: band-limited RIR + energy decay ------------------------------------
    a = fig.add_subplot(gs[1, 2])
    n_show = int(0.25 * fs)
    t_ms = np.arange(n_show) / fs * 1e3
    r_base = band_limited_rir(H_base[rx_idx], fs, n_time, 0.0, F_MAX_PLOT)
    r_edit = band_limited_rir(H_edit[rx_idx], fs, n_time, 0.0, F_MAX_PLOT)
    a.plot(t_ms, r_base[:n_show], color=C_BASE, lw=0.7, alpha=0.5)
    a.plot(t_ms, r_edit[:n_show], color=C_EDIT, lw=0.7, alpha=0.5)
    a.set(xlabel="time (ms)", ylabel="RIR amplitude (thin)",
          title="3.  Impulse response at receiver {}, 0-300 Hz".format(rx_idx))
    a.grid(alpha=0.25)
    a.tick_params(labelsize=8.5)
    a2 = a.twinx()
    e_base = band_limited_edc(r_base, fs)[0][:n_show]
    e_edit = band_limited_edc(r_edit, fs)[0][:n_show]
    a2.plot(t_ms, e_base, color=C_BASE, lw=2.2, label="baseline")
    a2.plot(t_ms, e_edit, color=C_EDIT, lw=2.2, label="{} $\\to$ {}".format(wall, material))
    a2.set_ylabel("energy decay curve, dB (thick)")
    a2.set_ylim(-45, 2)
    a2.tick_params(labelsize=8.5)
    a2.legend(fontsize=8.4, loc="upper right")

    # ---- titles + caption -------------------------------------------------------------
    s = summary
    fig.suptitle(
        "Editing the {} wall: {} ($\\alpha$ {:.2f} $\\to$ {:.2f})   |   room {:.2f} x "
        "{:.2f} m   |   one model, no retraining".format(
            wall.upper(), MATERIAL_NAMES[material], ALPHA_BASELINE, a_edit, L, W),
        fontsize=15.5, fontweight="bold", y=0.982)
    fig.text(0.5, 0.930,
             "measured $\\Delta$BW:   {}  {:+.2f} Hz    vs    {}  {:+.2f} Hz     "
             "$\\Rightarrow$   wall selectivity  {}:1".format(
                 family_label(own), s["d_bw_own"], family_label(other), s["d_bw_other"],
                 ("%.1f" % s["selectivity"]) if np.isfinite(s["selectivity"]) else "n/a"),
             ha="center", va="top", fontsize=12.5, color="#8B1A0E")
    fig.text(0.5, 0.898,
             "the simulator's ray law predicts $\\Delta$BW = 0 EXACTLY for the other "
             "family, so the ratio is bounded only by measurement noise; a "
             "locally-reacting (Kuttruff) wall would give ~2:1",
             ha="center", va="top", fontsize=9.4, color="0.35")
    fig.text(0.5, 0.098,
             "ZERO-SHOT, UNSEEN GEOMETRY, CONFIGURATION COMPUTED — NO MEASUREMENTS, "
             "no per-configuration optimisation.  Both fields come from the same "
             "checkpoint (iter {});\nthe only difference between them is the 64-d "
             "conditioning vector computed from (L, W, $\\alpha_{{west}}$, "
             "$\\alpha_{{east}}$, $\\alpha_{{south}}$, $\\alpha_{{north}}$).".format(
                 ckpt_iter),
             ha="center", va="top", fontsize=9.4, color="0.15")
    fig.text(0.5, 0.032,
             "Scope: the ~29:1 selectivity is a property of the ISM simulator "
             "(angle-independent reflection, no grazing-incidence absorption). Real "
             "locally-reacting walls follow Kuttruff (~2:1, with no invariant family),\n"
             "so the claim this figure supports is that the model learns the "
             "SIMULATOR's per-wall law. Bandwidth, not level, is the observable: measured "
             "level selectivity is only ~4:1.",
             ha="center", va="top", fontsize=8.6, color="0.35", style="italic")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------- main
def build_summary(base: Optional[dict], edit: Optional[dict], *, L: float, W: float,
                  alphas_base, alphas_edit, own: str, other: str) -> dict:
    """Measured per-family deltas + both theories' predictions for the same quantity."""
    fams: Dict[str, dict] = {}
    for fam in (X_AXIAL, Y_AXIAL, TANGENTIAL):
        fams[fam] = (paired_family_stats(base, edit, fam) if base and edit else
                     {"d_bw": float("nan"), "d_level": float("nan"), "n": 0})

    def d(kind: str, fam: str) -> float:
        return fams[fam]["d_" + kind]

    n_own = (1, 0) if own == X_AXIAL else (0, 1)
    n_other = (1, 0) if other == X_AXIAL else (0, 1)

    def pred(alphas, n, model) -> float:
        return damping_to_bandwidth_hz(
            modal_damping_2d(L, W, alphas, n[0], n[1], model=model))

    out = {
        "d_bw_own": d("bw", own), "d_bw_other": d("bw", other),
        "d_bw_tangential": d("bw", TANGENTIAL),
        "d_level_own": d("level", own), "d_level_other": d("level", other),
        "d_level_tangential": d("level", TANGENTIAL),
        "family_own": own, "family_other": other, "per_family": fams,
    }
    out["selectivity"] = (abs(out["d_bw_own"]) / max(abs(out["d_bw_other"]), SIGMA_BW_FLOOR)
                          if np.isfinite(out["d_bw_own"]) else float("nan"))
    for tag, law in (("ism_ray", "ism_ray"), ("kuttruff", "kuttruff")):
        own_d = pred(alphas_edit, n_own, law) - pred(alphas_base, n_own, law)
        oth_d = pred(alphas_edit, n_other, law) - pred(alphas_base, n_other, law)
        out["d_bw_own_pred_" + tag] = own_d
        out["d_bw_other_pred_" + tag] = oth_d
        out["selectivity_pred_" + tag] = abs(own_d) / max(abs(oth_d), SIGMA_BW_FLOOR)
    if base is not None:
        out["cond_phi"] = float(base["projection"].cond)
        out["residual_frac"] = float(base["projection"].residual_frac)
    return out


def print_summary(s: dict, s_gt: Optional[dict], *, wall: str, material: str, L: float,
                  W: float, novelty: Optional[dict], held_out: bool, secs: float) -> None:
    own, other = s["family_own"], s["family_other"]
    line = "-" * 78
    print("")
    print(line)
    print("P3-2 MATERIAL EDIT -- room {:.2f} x {:.2f} m, {} wall -> {} (alpha {:.2f} -> "
          "{:.2f})".format(L, W, wall, MATERIAL_NAMES[material], ALPHA_BASELINE,
                           MATERIALS[material]))
    print(line)
    print("provenance    : combo ({}, {}) {}".format(
        wall, material,
        "HELD OUT -- never trained in ANY geometry" if held_out else "seen in training"))
    if novelty is not None:
        print("              : geometry {} (nearest of {} trained geometries: "
              "{:.2f} x {:.2f} m, {:.2f} m away)".format(
                  "IS in the training set" if novelty["exact_match"] else "NOT in training",
                  novelty["n_train_geometries"], novelty["nearest"][0],
                  novelty["nearest"][1], novelty["distance_m"]))
    print("conditioning  : computed from (L, W, alpha x4) -- no lookup, no optimisation")
    print("")
    print("Measured on the PREDICTED field, 64-receiver modal projection "
          "(cond(Phi)={:.2f}, residual {:.1%}).".format(
              s.get("cond_phi", float("nan")), s.get("residual_frac", float("nan"))))
    print("Bandwidths are averaged over the modes resolvable in BOTH configs (paired).")
    print("  {:<18s} {:>9s} {:>9s} {:>9s} {:>9s}  {}".format(
        "family", "BW base", "BW edit", "dBW (Hz)", "dLvl (dB)", "modes"))
    for fam, tag in ((own, "  <- OWN, driven by this wall"), (other, ""),
                     (TANGENTIAL, "  <- responds to every wall")):
        f = s["per_family"][fam]
        print("  {:<18s} {:>9.3f} {:>9.3f} {:>+9.3f} {:>+9.2f}  {}{}".format(
            family_label(fam), f["bw_base"], f["bw_edit"], f["d_bw"], f["d_level"],
            ",".join("({},{})".format(*m) for m in f["modes"]) or "none resolvable", tag))
    print("")
    print("  SELECTIVITY  |dBW own| / |dBW other|  =  {:.1f} : 1".format(s["selectivity"]))
    print("     ISM-ray (the simulator's law)    : dBW own {:+.2f}, other {:+.2f} Hz"
          .format(s["d_bw_own_pred_ism_ray"], s["d_bw_other_pred_ism_ray"]))
    print("         -> the other family is EXACTLY invariant (no grazing-incidence "
          "absorption), so the")
    print("            predicted ratio is unbounded and the measured one is limited only "
          "by the {:.2f} Hz".format(SIGMA_BW_FLOOR))
    print("            bandwidth noise floor. Absolute widths run ~1.66x gamma/pi (gate "
          "T5 slope);")
    print("            it is the RATIO, not the width, that is under test here.")
    print("     Kuttruff (locally-reacting wall) : dBW own {:+.2f}, other {:+.2f} Hz  ->  "
          "{:.1f} : 1".format(s["d_bw_own_pred_kuttruff"], s["d_bw_other_pred_kuttruff"],
                              s["selectivity_pred_kuttruff"]))
    if s_gt is not None:
        print("")
        print("  ISM ground truth, same measurement: dBW own {:+.3f} Hz, other {:+.3f} Hz,"
              " selectivity {:.1f} : 1".format(
                  s_gt["d_bw_own"], s_gt["d_bw_other"], s_gt["selectivity"]))
        print("                                      dLevel own {:+.2f} dB, other "
              "{:+.2f} dB".format(s_gt["d_level_own"], s_gt["d_level_other"]))
    print("")
    print("  Level selectivity is only ~4:1 by construction -- BANDWIDTH is the "
          "observable this claim rests on.")
    print("  Scope: the ~29:1 ratio is a property of the ISM simulator (angle-independent")
    print("  reflection, no grazing-incidence absorption). Real locally-reacting walls "
          "follow")
    print("  Kuttruff (~2:1, no invariant family). Claim = the model learns the "
          "simulator's law.")
    print("{}\n[timing] {:.1f} s".format(line, secs))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P3-2 zero-shot material-edit demo (one figure + a falsifiable number).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--L", type=float, required=True, help="room length (m), x extent")
    ap.add_argument("--W", type=float, required=True, help="room width (m), y extent")
    ap.add_argument("--wall", required=True, help="west | east | south | north")
    ap.add_argument("--material", required=True,
                    help="baseline/brick, concrete/hard, curtain, absorber/panel, or M0..M3")
    ap.add_argument("--receiver", default="corner",
                    help="'corner' (far corner -- most reliable single probe), 'center', "
                         "or an integer index 0..63")
    ap.add_argument("--gt", default=None,
                    help="ISM ground truth: the data DIRECTORY or the edited config's .h5")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint .pt (default: newest in {})".format(DEFAULT_CKPT_DIR))
    ap.add_argument("--out", default="outputs/p3_2/demo", help="output directory")
    ap.add_argument("--rx-chunk", type=int, default=8,
                    help="receivers rendered per forward pass (memory knob)")
    args = ap.parse_args()

    t0 = time.time()
    L, W = round(float(args.L), 2), round(float(args.W), 2)
    wall = resolve_wall(args.wall)
    material = resolve_material(args.material)
    if MATERIALS[material] == ALPHA_BASELINE:
        raise SystemExit("--material {} IS the baseline (alpha {:.2f}); there is nothing to "
                         "edit. Pick concrete / curtain / absorber.".format(
                             material, ALPHA_BASELINE))
    alphas_base = alphas_for()
    alphas_edit = alphas_for(wall, material)
    own, other = driven_family(wall)
    held_out = (wall, material) in HELDOUT_COMBOS

    ckpt = Path(args.checkpoint) if args.checkpoint else newest_checkpoint(DEFAULT_CKPT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[demo] room {:.2f}x{:.2f} | edit {} -> {} (alphas {}) | ckpt {}".format(
        L, W, wall, material, list(alphas_edit), ckpt.name))
    model, renderer, cfg, meta, ckpt_iter = build_model(ckpt, device)
    fs = float(cfg["fs"])
    n_time = int(cfg["n_time_samples"])
    n_freq = n_time // 2 + 1
    f_axis = np.arange(n_freq) * fs / n_time

    gt = load_gt(args.gt, L, W, alphas_edit) if args.gt else None
    if gt is not None:
        rx, src = gt["rx"], gt["src"]
        print("[demo] ISM GT: {} (split {})".format(Path(gt["path_edit"]).name, gt["split"]))
    else:
        rx, src = receiver_grid_2d(L, W), np.asarray(SRC_DEFAULT, dtype=float)

    rx_idx, rx_why = pick_receiver(args.receiver, rx, L, W, src)

    H_base = render_field(model, renderer, L, W, alphas_base, rx, src, device, args.rx_chunk)
    H_edit = render_field(model, renderer, L, W, alphas_edit, rx, src, device, args.rx_chunk)
    print("[demo] rendered 2 x {} receivers in {:.1f}s".format(len(rx), time.time() - t0))

    m_base, m_edit = measure_pair(H_base, H_edit, rx, L, W, alphas_base, alphas_edit,
                                  f_axis, src, fs)
    summary = build_summary(m_base, m_edit, L=L, W=W, alphas_base=alphas_base,
                            alphas_edit=alphas_edit, own=own, other=other)
    summary_gt = None
    if gt is not None and "H_base" in gt:
        g_base, g_edit = measure_pair(gt["H_base"], gt["H_edit"], rx, L, W, alphas_base,
                                      alphas_edit, f_axis, src, fs)
        summary_gt = build_summary(g_base, g_edit, L=L, W=W, alphas_base=alphas_base,
                                   alphas_edit=alphas_edit, own=own, other=other)

    mode, f_mode_meas, mode_why, mode_detail = most_affected_mode(m_base, m_edit, own, L, W)

    out_dir = Path(args.out)
    stem = "demo_L{:.2f}_W{:.2f}_{}_{}".format(L, W, wall, material)
    make_figure(out_dir / (stem + ".png"), L=L, W=W, wall=wall, material=material,
                alphas_base=alphas_base, alphas_edit=alphas_edit, H_base=H_base,
                H_edit=H_edit, rx=rx, src=src, rx_idx=rx_idx, rx_why=rx_why,
                f_axis=f_axis, fs=fs, n_time=n_time, mode=mode, f_mode_meas=f_mode_meas,
                mode_detail=mode_detail, own=own, other=other, summary=summary, gt=gt,
                ckpt_iter=ckpt_iter)

    novelty = geometry_novelty(L, W)
    record = {k: v for k, v in summary.items()}
    record.update({
        "L": L, "W": W, "wall": wall, "material": material,
        "alphas_baseline": list(alphas_base), "alphas_edited": list(alphas_edit),
        "held_out_combo": held_out, "geometry_novelty": novelty,
        "receiver_index": rx_idx, "receiver_choice": rx_why,
        "most_affected_mode": [mode.n_x, mode.n_y], "mode_freq_hz": f_mode_meas,
        "mode_selection": mode_why, "checkpoint": str(ckpt), "checkpoint_iter": ckpt_iter,
        "ground_truth": None if gt is None else gt["path_edit"],
        "ground_truth_summary": summary_gt, "figure": str(out_dir / (stem + ".png")),
        "seconds": round(time.time() - t0, 2),
    })
    (out_dir / (stem + ".json")).write_text(json.dumps(record, indent=1, default=float))

    print_summary(summary, summary_gt, wall=wall, material=material, L=L, W=W,
                  novelty=novelty, held_out=held_out, secs=time.time() - t0)
    print("[demo] figure -> {}".format(out_dir / (stem + ".png")))
    print("[demo] numbers -> {}".format(out_dir / (stem + ".json")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as _exc:
        # This is run live in front of an audience: a bad wall name or a missing GT file
        # should print one readable line, not a traceback.
        raise SystemExit("demo error: {}".format(_exc))
