"""P3-2 zero-shot evaluation driver: does the model reproduce a *per-wall* material edit?

Zero-shot here means exactly one thing: the conditioning vector is built analytically from
(L, W, alpha_west..alpha_north) with :func:`aaf.models.conditioning_2d.fourier_features_2d`
and pushed through the network. No measurements are consulted, no latent is optimised, no
lookup table is touched. Rendering an unseen (wall, material) combination on an unseen
geometry costs one forward pass.

Everything about an *edit* is measured as a PAIRED difference::

    delta_pred = measure(pred_edited) - measure(pred_baseline_of_the_same_geometry)
    delta_gt   = measure( gt_edited ) - measure( gt_baseline_of_the_same_geometry )

and scored as ``|delta_pred - delta_gt|``. The pairing is not cosmetic: the -3 dB
bandwidth estimator has a bias that depends on the geometry and on the mode's own
frequency, and that bias cancels to first order in the difference. It also moves the
target from "reconstruct the room" to "reproduce the edit", which is the actual claim.

Measurement chain (per config, per mode -- never per receiver):

    64-receiver field --project_field--> mode-resolved spectra --measure_modes--> -3 dB BW
                                                              --modal_decay_rate--> gamma

with the -3 dB walk cap set from the ISM-ray predicted width
(``modal_damping_2d(model="ism_ray") -> damping_to_bandwidth_hz``) so that genuinely broad
absorber modes are not rejected by a spacing-based cap. Modes are capped at 200 Hz because
an 8x8 receiver grid cannot condition the mode-shape basis above that (see
``aaf.eval.modal_projection``).

**Scoping.** The ~29:1 bandwidth selectivity this eval scores against is a property of the
ISM simulator: pyroomacoustics applies an angle-independent reflection coefficient, so an
axial mode is damped only by the wall pair it bounces between and grazing incidence costs
nothing. Real locally-reacting walls follow Kuttruff and would show only ~2:1 with NO
invariant mode family. Every claim supported by this file is therefore "the model learns
the simulator's per-wall law", not "the model learns room acoustics".

Controls (a bare edit number is not evidence; each of these can fail independently):

C1 NULL MODEL   score the model's BASELINE render against the EDITED ground truth. The M1
                edit is only ~0.3 dB of LSD, well under the in-distribution val LSD, so a
                model that ignored the material channel entirely could still post a
                respectable LSD. ``edit_gain = LSD(null) / LSD(model)`` must exceed 1.
C2 FLOOR        within-family mode-to-mode std of the measured baseline bandwidth. Under
                the ISM-ray law every x-axial mode has *identical* damping, so this spread
                is pure estimator noise and sets the resolution of every delta reported.
C3 IDENTITY     "wall k set to M0" must be bit-identical to the baseline, both in the
                conditioning vector and in the render. Requires ``renderer.eval()`` --
                ``FreqRenderer2D`` jitters ray azimuths while ``self.training``.
C4 WALL IDENTITY each held-out combo is compared with its trained opposite-wall TWIN
                (``aaf.walls.WALL_TWIN``), which carries identical mean absorption and
                identical T60 and differs only in WHERE the absorber sits. A model that
                collapsed the 4 absorptions to a scalar alpha_eff renders the twin's field
                for both, so its predicted per-receiver dB map correlates with the twin's
                ground truth as well as with its own.

Usage::

    python -m aaf.eval.p3_2_eval --train-dir outputs/p3_2/p3_2_main \\
        --out outputs/p3_2/eval [--checkpoint PATH] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import yaml

from aaf.data.mat_configs import HELDOUT_COMBOS, UNSEEN_ALPHA, MatConfig, enumerate_configs
from aaf.eval.band_limited import band_indices, compute_band_limited_metrics
from aaf.eval.modal_bandwidth import caps_from_predicted_bw, measure_modes
from aaf.eval.modal_decay import band_limited_rir, modal_decay_rate, t20_band
from aaf.eval.modal_projection import (
    F_MAX_PROJECTION_HZ,
    TANGENTIAL,
    X_AXIAL,
    Y_AXIAL,
    project_field,
)
from aaf.eval.signal_level import (
    magnitude_correlation,
    phase_correlation_mag_weighted,
    rir_pearson,
)
from aaf.eval.spatial_modes import bin_index_for_freq, spatial_correlation_complex
from aaf.models.conditioning_2d import fourier_features_2d
from aaf.renderers.freq_2d import FreqRenderer2D
from aaf.sim.analytical_modal_2d import damping_to_bandwidth_hz, modal_damping_2d
from aaf.walls import WALL_AXIS, WALL_TWIN, WALLS_2D, alphas_for

FAMILIES = (X_AXIAL, Y_AXIAL, TANGENTIAL)
AXIAL_FAMILIES = (X_AXIAL, Y_AXIAL)
BAND_HI_HZ = 300.0

# Split names are contractual -- the figure script keys off them.
SPLIT_I = "i_unseen_geom_seen_combo"
SPLIT_II = "ii_seen_geom_heldout_combo"
SPLIT_III = "iii_unseen_geom_heldout_combo"
SPLIT_IV = "iv_unseen_alpha"
SPLIT_ORDER = (SPLIT_I, SPLIT_II, SPLIT_III, SPLIT_IV)

# Measured-vs-ISM-ray-theory calibration from the physics gate (T5):
# BW_measured = 0.302 + 1.661 * gamma/pi. The intercept is an estimator offset and cancels
# in a delta, so only the slope is carried into `theory_d_bw`.
THEORY_SLOPE_FALLBACK = 1.6607564051417665
GATE_JSON = "outputs/p3_2/gate/gate.json"

# Denominator floor for the axial selectivity index -- 0.3 bins, ~2.5x the measured
# mode-to-mode repeatability, so a null "other-axis" delta cannot manufacture selectivity.
SELECTIVITY_FLOOR_HZ = 0.15


# --------------------------------------------------------------------------- small utils
def _mean(vals: Sequence[float]) -> float:
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def _std(vals: Sequence[float]) -> float:
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    return float(np.std(v)) if len(v) > 1 else float("nan")


def _nan() -> float:
    return float("nan")


def theory_slope(gate_json: str = GATE_JSON) -> float:
    """Slope of measured -3 dB BW vs ISM-ray gamma/pi, from the physics gate."""
    try:
        d = json.loads(Path(gate_json).read_text())
        return float(d["T5_calibration"]["ism_ray"]["slope"])
    except Exception:
        return THEORY_SLOPE_FALLBACK


# --------------------------------------------------------------------------- model / render
def find_checkpoint(train_dir: str) -> Path:
    """Newest ``ckpt_iter*.pt`` in ``train_dir`` (training may still be running)."""
    ck = sorted(Path(train_dir).glob("ckpt_iter*.pt"))
    if not ck:
        raise FileNotFoundError(f"no ckpt_iter*.pt under {train_dir}")
    return ck[-1]


def load_model(ckpt_path: Path, device: torch.device):
    """Rebuild the trained model + renderer from the checkpoint's own cfg.

    Both are put in ``eval()`` mode. The renderer flag is load-bearing, not hygiene:
    ``FreqRenderer2D._ray_directions_2d`` jitters the azimuth grid whenever
    ``self.training``, which makes two renders of the same config differ and breaks C3.

    ``inr_2d`` is imported lazily: it pulls in tinycudann, which raises "Unknown compute
    capability" on a GPU-less login node. Keeping it out of module scope lets the whole
    measurement half of this file be imported and unit-tested on CPU.
    """
    from aaf.models.inr_2d import INR2D_AutoDecoder

    st = torch.load(str(ckpt_path), map_location=device)
    cfg = dict(st["cfg"])
    meta_path = ckpt_path.parent / "train_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    n_freq_bins = int(cfg["n_time_samples"]) // 2 + 1
    hg = dict(
        otype="HashGrid",
        n_levels=int(cfg["n_levels"]),
        n_features_per_level=2,
        log2_hashmap_size=int(cfg["log2_hashmap_size"]),
        base_resolution=16,
        per_level_scale=float(cfg["per_level_scale"]),
    )
    model = INR2D_AutoDecoder(
        n_rooms=int(meta.get("n_configs", 1)),
        latent_dim=int(cfg["latent_dim"]),
        n_freq_bins=n_freq_bins,
        hash_grid_config=hg,
        conditioning_type=str(cfg.get("conditioning_type", "film")),
        cond_source=str(cfg["cond_source"]),
        cond_dim=int(cfg["cond_dim"]),
        l_head_enabled=False,
    ).to(device)
    model.load_state_dict(st["model"])
    model.eval()
    renderer = FreqRenderer2D(
        n_azi=int(cfg["n_azi"]), n_pts_per_ray=int(cfg["n_pts_per_ray"]),
        near=float(cfg["near"]), fs=int(cfg["fs"]),
        n_time_samples=int(cfg["n_time_samples"]), c=float(cfg["c"]),
    ).to(device)
    renderer.eval()
    return model, renderer, cfg, meta, int(st["iter"])


def in_dist_val_lsd(train_dir: Path, it: int) -> Optional[float]:
    """Latest in-distribution (held-out receiver) val LSD at or before iteration ``it``."""
    p = train_dir / "scalars.json"
    if not p.exists():
        return None
    recs = [r for r in json.loads(p.read_text())
            if r.get("phase") == "val" and int(r.get("iter", 0)) <= it]
    return float(recs[-1]["lsd_db"]) if recs else None


@torch.no_grad()
def render_config(model, renderer, L: float, W: float, alphas: Sequence[float],
                  rx: np.ndarray, src: np.ndarray, device: torch.device,
                  rx_chunk: int = 8) -> np.ndarray:
    """ZERO-SHOT render of one config at every receiver -> ``[n_rx, n_freq]`` complex64.

    The conditioning vector is computed from the physical parameters alone; nothing about
    this config's ground truth is read.
    """
    cond = fourier_features_2d(L, W, alphas, device=device)          # [64]
    room_min = torch.zeros(2, device=device)
    room_max = torch.tensor([float(L), float(W)], dtype=torch.float32, device=device)
    rx_t = torch.as_tensor(np.asarray(rx, dtype=np.float32), device=device)
    src_t = torch.as_tensor(np.asarray(src, dtype=np.float32), device=device)
    out = []
    for s in range(0, rx_t.shape[0], rx_chunk):
        r = rx_t[s:s + rx_chunk]
        tx = src_t.unsqueeze(0).expand(r.shape[0], -1)
        z = cond.unsqueeze(0).expand(r.shape[0], -1)
        out.append(renderer(model, r, tx, room_min, room_max, z_s=z).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.complex64)


# --------------------------------------------------------------------------- data
def load_gt(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with h5py.File(str(path), "r") as f:
        H = np.asarray(f["ism/H_complex"][...])
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]), dtype=float)
        src = np.asarray(json.loads(f.attrs["source_pos"]), dtype=float)
        split = str(f.attrs["split"])
    return H, rx, src, split


def band_limit(H: np.ndarray, hi_idx: int) -> np.ndarray:
    """Zero every bin at/above ``hi_idx`` and force DC real.

    Applied identically to prediction and ground truth: the model was supervised only on
    0-300 Hz, so bins above that are unconstrained and must not enter any metric.
    """
    out = np.zeros_like(np.asarray(H, dtype=np.complex64))
    out[..., :hi_idx] = np.nan_to_num(np.asarray(H)[..., :hi_idx])
    out[..., 0] = out[..., 0].real
    return out


# --------------------------------------------------------------------------- measurement
@dataclass
class Stream:
    """One 64-receiver field plus its per-mode measurements."""

    H: np.ndarray                 # [n_rx, n_freq] complex, band-limited
    bw: np.ndarray                # [n_modes] -3 dB bandwidth (Hz), nan if not resolvable
    level: np.ndarray             # [n_modes] peak level (dB)
    gamma: np.ndarray             # [n_modes] modal decay rate (1/s), nan if unfit
    residual_frac: float
    cond_phi: float


@dataclass
class GeomCtx:
    """Everything that depends only on the geometry, computed once."""

    L: float
    W: float
    rx: np.ndarray
    src: np.ndarray
    modes: list
    used: np.ndarray
    f_axis: np.ndarray
    bw_theory_base: np.ndarray
    gt_base: Optional[Stream] = None
    pred_base: Optional[Stream] = None
    key: Tuple[float, float] = (0.0, 0.0)


def theory_bw(L: float, W: float, alphas: Sequence[float], modes) -> np.ndarray:
    """ISM-ray predicted -3 dB width per mode (Hz), used for the walk caps and for theory."""
    return np.array(
        [damping_to_bandwidth_hz(modal_damping_2d(L, W, alphas, m.n_x, m.n_y, model="ism_ray"))
         for m in modes],
        dtype=float,
    )


def analyse(H: np.ndarray, geom: GeomCtx, alphas: Sequence[float], fs: float, n_time: int,
            with_decay: bool = True) -> Stream:
    """Project onto the analytic mode shapes, then measure BW / level / decay per mode."""
    pr = project_field(H, geom.rx, geom.L, geom.W, src=geom.src, fs=fs,
                       f_max=F_MAX_PROJECTION_HZ)
    bw_th = theory_bw(geom.L, geom.W, alphas, pr.modes)
    peaks = measure_modes(pr.spectra, geom.f_axis, pr.modes, caps=caps_from_predicted_bw(bw_th))
    bw = np.array([p.bw_3db_hz if p.bw_valid else np.nan for p in peaks], dtype=float)
    level = np.array([p.level_db for p in peaks], dtype=float)
    gamma = np.full(len(pr.modes), np.nan)
    if with_decay:
        for i, m in enumerate(pr.modes):
            if not pr.used[i]:
                continue
            g, _ = modal_decay_rate(pr.spectra[i], m.f, fs, n_time,
                                    gamma_prior=float(bw_th[i] * np.pi), spread_comp=True)
            gamma[i] = g
    return Stream(H=H, bw=bw, level=level, gamma=gamma,
                  residual_frac=float(pr.residual_frac), cond_phi=float(pr.cond))


def make_geom_ctx(L: float, W: float, rx: np.ndarray, src: np.ndarray, fs: float,
                  n_freq: int) -> GeomCtx:
    f_axis = np.arange(n_freq) * (fs / (2.0 * (n_freq - 1)))
    # `project_field` on a zero field still enumerates modes and the excitation mask, both
    # of which depend only on (L, W, src) -- so grab them from a throwaway projection.
    pr = project_field(np.zeros((rx.shape[0], n_freq), dtype=np.complex64), rx, L, W,
                       src=src, fs=fs, f_max=F_MAX_PROJECTION_HZ)
    return GeomCtx(
        L=L, W=W, rx=rx, src=src, modes=pr.modes, used=np.asarray(pr.used, dtype=bool),
        f_axis=f_axis, bw_theory_base=theory_bw(L, W, alphas_for(), pr.modes),
        key=(round(L, 2), round(W, 2)),
    )


# --------------------------------------------------------------------------- metrics
def fidelity(H_pred: np.ndarray, H_gt: np.ndarray, fs: float, n_time: int,
             hi_idx: int) -> dict:
    """In-band (0-300 Hz) reconstruction fidelity, all metrics on identical masks."""
    hp, ht = H_pred[:, :hi_idx], H_gt[:, :hi_idx]
    band = compute_band_limited_metrics(H_pred, H_gt, fs, H_pred.shape[-1],
                                        bands=((0.0, BAND_HI_HZ),))
    rp = band_limited_rir(H_pred, fs, n_time, 0.0, BAND_HI_HZ)
    rt = band_limited_rir(H_gt, fs, n_time, 0.0, BAND_HI_HZ)
    tp = t20_band(H_pred, fs, n_time=n_time, f_hi=BAND_HI_HZ)
    tg = t20_band(H_gt, fs, n_time=n_time, f_hi=BAND_HI_HZ)
    t20_p, t20_g = _mean(tp["t20"]), _mean(tg["t20"])
    rel = abs(t20_p - t20_g) / t20_g if np.isfinite(t20_p) and np.isfinite(t20_g) \
        and t20_g > 0 else _nan()
    return {
        "mag_corr": float(magnitude_correlation(hp, ht)),
        "band_lsd_db": float(band["lsd_band_0_300_db"]),
        "phase_corr_mw": float(phase_correlation_mag_weighted(hp, ht)),
        "rir_pearson": float(rir_pearson(rp, rt)),
        "t20_rel_err": float(rel),
        "t20_pred_s": float(t20_p),
        "t20_gt_s": float(t20_g),
        "t20_frac_valid_pred": float(tp["frac_valid"]),
        "t20_frac_valid_gt": float(tg["frac_valid"]),
    }


def db_map(H_edit: np.ndarray, H_base: np.ndarray, hi_idx: int) -> np.ndarray:
    """Per-receiver in-band level change vs the baseline, dB -> ``[n_rx]``.

    The C4 observable: it is a *spatial* signature of where the absorber sits, which a
    scalar effective absorption cannot express.
    """
    eps = 1e-12
    a = 20.0 * np.log10(np.abs(H_edit[:, :hi_idx]) + eps)
    b = 20.0 * np.log10(np.abs(H_base[:, :hi_idx]) + eps)
    return np.asarray((a - b).mean(axis=1), dtype=float)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, with a flat vector scored 0.0 rather than nan.

    A model that ignores the material channel emits an identically-zero edit map. That is
    a RESULT (no spatial association whatsoever), not missing data, and reporting it as nan
    would let a completely dead model show up as "not measured". nan is reserved for too
    few finite samples.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return _nan()
    if np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def mode_shape_invariance(H_edit: np.ndarray, H_base: np.ndarray, geom: GeomCtx,
                          fs: float) -> float:
    """Mean |complex Pearson| between edited and baseline fields at each mode's bin.

    Ground truth sits at 0.955-1.000: a wall edit changes a mode's WIDTH and LEVEL, not its
    SHAPE. A prediction that scores much lower is redrawing the field, not editing it.
    """
    n_freq = H_edit.shape[-1]
    vals = []
    for i, m in enumerate(geom.modes):
        if not geom.used[i]:
            continue
        b = bin_index_for_freq(m.f, fs, n_freq)
        vals.append(spatial_correlation_complex(H_edit[:, b], H_base[:, b]))
    return _mean(vals)


def paired_cells(geom: GeomCtx, gt_e: Stream, pr_e: Stream, alphas: Sequence[float],
                 slope: float) -> List[dict]:
    """One record per well-excited mode: the paired GT and predicted edit deltas."""
    bw_th_edit = theory_bw(geom.L, geom.W, alphas, geom.modes)
    gt_b, pr_b = geom.gt_base, geom.pred_base
    cells: List[dict] = []
    for i, m in enumerate(geom.modes):
        if not geom.used[i]:
            continue
        bw_ok = bool(np.isfinite(gt_e.bw[i]) and np.isfinite(gt_b.bw[i])
                     and np.isfinite(pr_e.bw[i]) and np.isfinite(pr_b.bw[i]))
        lvl_ok = bool(np.isfinite(gt_e.level[i]) and np.isfinite(gt_b.level[i])
                      and np.isfinite(pr_e.level[i]) and np.isfinite(pr_b.level[i]))
        g_ok = bool(np.all(np.isfinite([gt_e.gamma[i], gt_b.gamma[i],
                                        pr_e.gamma[i], pr_b.gamma[i]]))
                    and min(gt_e.gamma[i], gt_b.gamma[i], pr_e.gamma[i], pr_b.gamma[i]) > 0)
        cells.append({
            "n_x": int(m.n_x), "n_y": int(m.n_y), "family": m.family, "f_hz": float(m.f),
            "bw_ok": bw_ok, "lvl_ok": lvl_ok, "gamma_ok": g_ok,
            "d_bw_gt": float(gt_e.bw[i] - gt_b.bw[i]) if bw_ok else _nan(),
            "d_bw_pred": float(pr_e.bw[i] - pr_b.bw[i]) if bw_ok else _nan(),
            "theory_d_bw": float(slope * (bw_th_edit[i] - geom.bw_theory_base[i])),
            "d_lvl_gt": float(gt_e.level[i] - gt_b.level[i]) if lvl_ok else _nan(),
            "d_lvl_pred": float(pr_e.level[i] - pr_b.level[i]) if lvl_ok else _nan(),
            "d_lngamma_gt": float(np.log(gt_e.gamma[i]) - np.log(gt_b.gamma[i]))
            if g_ok else _nan(),
            "d_lngamma_pred": float(np.log(pr_e.gamma[i]) - np.log(pr_b.gamma[i]))
            if g_ok else _nan(),
        })
    return cells


def edit_stats(cells: Sequence[dict]) -> dict:
    """Headline edit-transfer statistics over a pool of (mode, wall, material) cells."""
    g = np.array([c["d_bw_gt"] for c in cells if c["bw_ok"]], dtype=float)
    p = np.array([c["d_bw_pred"] for c in cells if c["bw_ok"]], dtype=float)
    lg = np.array([c["d_lvl_gt"] for c in cells if c["lvl_ok"]], dtype=float)
    lp = np.array([c["d_lvl_pred"] for c in cells if c["lvl_ok"]], dtype=float)
    gg = np.array([c["d_lngamma_gt"] for c in cells if c["gamma_ok"]], dtype=float)
    gp = np.array([c["d_lngamma_pred"] for c in cells if c["gamma_ok"]], dtype=float)
    slope = _nan()
    if g.size >= 2 and np.std(g) > 1e-12:
        slope = float(np.polyfit(g, p, 1)[0])
    return {
        "E_BW_hz": float(np.mean(np.abs(p - g))) if g.size else _nan(),
        "edit_bw_pearson": _pearson(g, p),
        "edit_bw_slope": slope,
        "E_LVL_db": float(np.mean(np.abs(lp - lg))) if lg.size else _nan(),
        "E_LNGAMMA": float(np.mean(np.abs(gp - gg))) if gg.size else _nan(),
        # The null model predicts delta = 0 everywhere, so mean|delta_gt| is both the GT
        # effect size and the E_BW a "did nothing" model would post.
        "gt_effect_size_hz": float(np.mean(np.abs(g))) if g.size else _nan(),
        "pred_effect_size_hz": float(np.mean(np.abs(p))) if p.size else _nan(),
        "n_cells": int(g.size),
        "n_cells_level": int(lg.size),
        "n_cells_gamma": int(gg.size),
    }


def by_family_stats(cells: Sequence[dict]) -> dict:
    out = {}
    for fam in FAMILIES:
        sub = [c for c in cells if c["family"] == fam]
        g = np.array([c["d_bw_gt"] for c in sub if c["bw_ok"]], dtype=float)
        p = np.array([c["d_bw_pred"] for c in sub if c["bw_ok"]], dtype=float)
        lg = np.array([c["d_lvl_gt"] for c in sub if c["lvl_ok"]], dtype=float)
        lp = np.array([c["d_lvl_pred"] for c in sub if c["lvl_ok"]], dtype=float)
        out[fam] = {
            "E_BW_hz": float(np.mean(np.abs(p - g))) if g.size else _nan(),
            "n": int(g.size),
            "E_LVL_db": float(np.mean(np.abs(lp - lg))) if lg.size else _nan(),
            "gt_d_bw": float(np.mean(g)) if g.size else _nan(),
            "pred_d_bw": float(np.mean(p)) if p.size else _nan(),
        }
    return out


# --------------------------------------------------------------------------- controls
@torch.no_grad()
def control_c3(model, renderer, geom: GeomCtx, device: torch.device, rx_chunk: int) -> dict:
    """C3: setting a wall to M0 must be a no-op, in the conditioning AND in the render."""
    v_base = fourier_features_2d(geom.L, geom.W, alphas_for(), device=device)
    vec_ok = all(
        torch.equal(v_base, fourier_features_2d(geom.L, geom.W, alphas_for(w, "M0"),
                                                device=device))
        for w in WALLS_2D
    )
    args = (model, renderer, geom.L, geom.W)
    kw = dict(rx=geom.rx, src=geom.src, device=device, rx_chunk=rx_chunk)
    H_base = render_config(*args, alphas=alphas_for(), **kw)
    H_m0 = render_config(*args, alphas=alphas_for("west", "M0"), **kw)
    H_again = render_config(*args, alphas=alphas_for(), **kw)
    render_ok = bool(np.array_equal(H_base, H_m0))
    determinism_ok = bool(np.array_equal(H_base, H_again))
    return {
        "pass": bool(vec_ok and render_ok and determinism_ok),
        "cond_vector_identical": bool(vec_ok),
        "render_bitwise_identical": render_ok,
        "render_deterministic": determinism_ok,
        "geometry": [geom.L, geom.W],
    }


def control_c4(maps: Dict[Tuple, Dict[str, np.ndarray]]) -> dict:
    """C4: does the model put the absorber on the right WALL, or only the right amount?

    ``maps[(L, W, wall, material)] = {"gt": [n_rx], "pred": [n_rx]}`` of per-receiver dB
    change vs that geometry's baseline. For each held-out combo we ask whether the
    prediction matches its OWN ground truth better than it matches its TWIN's ground truth
    (same mean absorption, same T60, absorber on the opposite wall).
    """
    out: Dict[str, dict] = {}
    for wall, mat in HELDOUT_COMBOS:
        twin = WALL_TWIN[wall]
        same, cross, gt_self_twin, pred_self_twin = [], [], [], []
        for key, m in maps.items():
            L, W, w, mm = key
            if w != wall or mm != mat:
                continue
            tk = (L, W, twin, mat)
            if tk not in maps:
                continue
            same.append(_pearson(m["pred"], m["gt"]))
            cross.append(_pearson(m["pred"], maps[tk]["gt"]))
            gt_self_twin.append(_pearson(m["gt"], maps[tk]["gt"]))
            pred_self_twin.append(_pearson(m["pred"], maps[tk]["pred"]))
        out["{}_{}".format(wall, mat)] = {
            "twin": "{}_{}".format(twin, mat),
            "n_geoms": len(same),
            "r_pred_vs_own_gt": _mean(same),
            "r_pred_vs_twin_gt": _mean(cross),
            # > 0 means the prediction is closer to its own wall than to the twin's.
            "wall_asymmetry": _mean(same) - _mean(cross),
            "r_gt_own_vs_twin_gt": _mean(gt_self_twin),
            "r_pred_own_vs_twin_pred": _mean(pred_self_twin),
        }
    finite = [v["wall_asymmetry"] for v in out.values() if np.isfinite(v["wall_asymmetry"])]
    out["mean_wall_asymmetry"] = float(np.mean(finite)) if finite else _nan()
    return out


# --------------------------------------------------------------------------- driver
def build_splits(train_yaml: str, test_yaml: str
                 ) -> Tuple[Dict[str, List[MatConfig]], set]:
    """The four evaluation splits plus the set of frozen-test geometry keys."""
    train_geoms = [(g["L"], g["W"]) for g in yaml.safe_load(open(train_yaml))["geometries"]]
    test_geoms = [(g["L"], g["W"]) for g in yaml.safe_load(open(test_yaml))["geometries"]]
    return {
        SPLIT_I: enumerate_configs(test_geoms, exclude_combos=HELDOUT_COMBOS,
                                   include_baseline=True),
        SPLIT_II: enumerate_configs(train_geoms, only_combos=HELDOUT_COMBOS,
                                    include_baseline=False),
        SPLIT_III: enumerate_configs(test_geoms, only_combos=HELDOUT_COMBOS,
                                     include_baseline=False),
        SPLIT_IV: enumerate_configs(test_geoms, unseen_alpha=UNSEEN_ALPHA),
    }, set((round(g[0], 2), round(g[1], 2)) for g in test_geoms)


def run(train_dir: str, out_dir: str, checkpoint: Optional[str], data_dir: str,
        train_yaml: str, test_yaml: str, limit: Optional[int], rx_chunk: int,
        with_decay: bool, device_str: str) -> dict:
    t_start = time.time()
    train_path, out_path = Path(train_dir), Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt = Path(checkpoint) if checkpoint else find_checkpoint(train_dir)
    model, renderer, cfg, meta, it = load_model(ckpt, device)
    fs, n_time = float(cfg["fs"]), int(cfg["n_time_samples"])
    n_freq = n_time // 2 + 1
    _, hi_idx = band_indices(fs, n_freq, 0.0, BAND_HI_HZ)
    slope = theory_slope()
    print("[ckpt] {} (iter {}) | band 0:{} | device {}".format(ckpt, it, hi_idx, device),
          flush=True)

    splits, test_keys = build_splits(train_yaml, test_yaml)
    if limit:
        splits = {k: v[:limit] for k, v in splits.items()}

    geoms: Dict[Tuple[float, float], GeomCtx] = {}
    records: List[dict] = []
    cells_by_split: Dict[str, List[dict]] = {k: [] for k in SPLIT_ORDER}
    maps: Dict[Tuple, Dict[str, np.ndarray]] = {}
    # selectivity[material][wall][family] -> lists of per-mode deltas (test geometries only)
    selectivity: Dict[str, Dict[str, Dict[str, Dict[str, list]]]] = {}
    floor_bw: Dict[str, List[float]] = {f: [] for f in FAMILIES}
    floor_bw_pred: Dict[str, List[float]] = {f: [] for f in FAMILIES}

    def geom_ctx(c: MatConfig) -> GeomCtx:
        """Geometry context + its baseline streams (GT and predicted), cached."""
        key = (round(c.L, 2), round(c.W, 2))
        if key in geoms:
            return geoms[key]
        base_file = Path(data_dir) / "L{:.2f}_W{:.2f}_aW0.15_aE0.15_aS0.15_aN0.15.h5".format(
            c.L, c.W)
        H_gt, rx, src, _ = load_gt(base_file)
        g = make_geom_ctx(c.L, c.W, rx, src, fs, n_freq)
        g.gt_base = analyse(band_limit(H_gt, hi_idx), g, alphas_for(), fs, n_time,
                            with_decay=with_decay)
        H_pred = render_config(model, renderer, c.L, c.W, alphas_for(), rx, src, device,
                               rx_chunk)
        g.pred_base = analyse(band_limit(H_pred, hi_idx), g, alphas_for(), fs, n_time,
                              with_decay=with_decay)
        # C2 estimator floor: under the ISM-ray law every mode of an axial family shares
        # one damping rate, so the within-family spread is pure estimator noise.
        for fam in FAMILIES:
            idx = [i for i, m in enumerate(g.modes) if m.family == fam and g.used[i]]
            floor_bw[fam].append(_std([g.gt_base.bw[i] for i in idx]))
            floor_bw_pred[fam].append(_std([g.pred_base.bw[i] for i in idx]))
        geoms[key] = g
        return g

    n_total = sum(len(v) for v in splits.values())
    done = 0
    for split_name in SPLIT_ORDER:
        for c in splits[split_name]:
            done += 1
            g = geom_ctx(c)
            if c.is_baseline:
                # The baseline has no edit; it still reports fidelity (and is the anchor
                # every delta in this geometry is taken against).
                rec = {"label": c.label, "split": split_name, "L": c.L, "W": c.W,
                       "wall": None, "material": None, "alphas": list(c.alphas),
                       "is_baseline": True,
                       "fidelity": fidelity(g.pred_base.H, g.gt_base.H, fs, n_time, hi_idx),
                       "cond_phi": g.gt_base.cond_phi,
                       "residual_frac_gt": g.gt_base.residual_frac,
                       "residual_frac_pred": g.pred_base.residual_frac}
                records.append(rec)
                continue

            H_gt = band_limit(load_gt(Path(data_dir) / c.filename)[0], hi_idx)
            H_pred = band_limit(
                render_config(model, renderer, c.L, c.W, c.alphas, g.rx, g.src, device,
                              rx_chunk), hi_idx)
            gt_e = analyse(H_gt, g, c.alphas, fs, n_time, with_decay=with_decay)
            pr_e = analyse(H_pred, g, c.alphas, fs, n_time, with_decay=with_decay)
            cells = paired_cells(g, gt_e, pr_e, c.alphas, slope)
            cells_by_split[split_name].extend(cells)

            fid = fidelity(H_pred, H_gt, fs, n_time, hi_idx)
            # C1: the model's own BASELINE render scored against the EDITED ground truth.
            fid_null = fidelity(g.pred_base.H, H_gt, fs, n_time, hi_idx)
            rec = {
                "label": c.label, "split": split_name, "L": c.L, "W": c.W,
                "wall": c.wall, "material": c.material, "alphas": list(c.alphas),
                "is_baseline": False,
                "fidelity": fid,
                "null_fidelity": fid_null,
                "edit": edit_stats(cells),
                "by_family": by_family_stats(cells),
                "mode_shape_gt": mode_shape_invariance(H_gt, g.gt_base.H, g, fs),
                "mode_shape_pred": mode_shape_invariance(H_pred, g.pred_base.H, g, fs),
                "cond_phi": g.gt_base.cond_phi,
                "cells": cells,
            }
            records.append(rec)

            key = (round(c.L, 2), round(c.W, 2), c.wall, c.material)
            maps[key] = {"gt": db_map(H_gt, g.gt_base.H, hi_idx),
                         "pred": db_map(H_pred, g.pred_base.H, hi_idx)}

            if (round(c.L, 2), round(c.W, 2)) in test_keys:
                mat = selectivity.setdefault(c.material, {})
                wal = mat.setdefault(c.wall, {})
                for fam in FAMILIES:
                    d = wal.setdefault(fam, {"gt": [], "pred": [], "theory": []})
                    for cell in cells:
                        if cell["family"] != fam or not cell["bw_ok"]:
                            continue
                        d["gt"].append(cell["d_bw_gt"])
                        d["pred"].append(cell["d_bw_pred"])
                        d["theory"].append(cell["theory_d_bw"])
            if done % 10 == 0 or done == n_total:
                print("  [{}/{}] {} ({:.1f}s)".format(done, n_total, c.label,
                                                      time.time() - t_start), flush=True)

    # ------------------------------------------------------------------ aggregation
    def agg_split(name: str) -> dict:
        recs = [r for r in records if r["split"] == name]
        edited = [r for r in recs if not r["is_baseline"]]
        cells = cells_by_split[name]
        fid_keys = ("mag_corr", "band_lsd_db", "phase_corr_mw", "rir_pearson", "t20_rel_err")
        e = edit_stats(cells)
        lsd_model = _mean([r["fidelity"]["band_lsd_db"] for r in edited])
        lsd_null = _mean([r["null_fidelity"]["band_lsd_db"] for r in edited])
        e["edit_gain"] = float(lsd_null / lsd_model) if np.isfinite(lsd_null) \
            and np.isfinite(lsd_model) and lsd_model > 0 else _nan()
        return {
            "n_configs": len(recs),
            "n_edited_configs": len(edited),
            "fidelity": {k: _mean([r["fidelity"][k] for r in recs]) for k in fid_keys},
            "edit": e,
            "by_family": by_family_stats(cells),
            "mode_shape_invariance": {
                "gt": _mean([r["mode_shape_gt"] for r in edited]),
                "pred": _mean([r["mode_shape_pred"] for r in edited]),
                "n": len(edited),
            },
            "null_fidelity": {k: _mean([r["null_fidelity"][k] for r in edited])
                              for k in fid_keys},
        }

    splits_out = {name: agg_split(name) for name in SPLIT_ORDER}

    # Held-out combos kept separate: (west,M2) tests material-value transfer onto a seen
    # wall, (north,M3) tests wall transfer of a seen material. Pooling hides which works.
    heldout_by_combo: Dict[str, dict] = {}
    for name in (SPLIT_II, SPLIT_III):
        per: Dict[str, dict] = {}
        for wall, mat in HELDOUT_COMBOS:
            recs = [r for r in records if r["split"] == name and r["wall"] == wall
                    and r["material"] == mat]
            cs = [c for r in recs for c in r["cells"]]
            st = edit_stats(cs)
            lsd_model = _mean([r["fidelity"]["band_lsd_db"] for r in recs])
            lsd_null = _mean([r["null_fidelity"]["band_lsd_db"] for r in recs])
            st["edit_gain"] = float(lsd_null / lsd_model) if np.isfinite(lsd_null) \
                and np.isfinite(lsd_model) and lsd_model > 0 else _nan()
            per["{}_{}".format(wall, mat)] = {
                "n_configs": len(recs),
                "edit": st,
                "by_family": by_family_stats(cs),
                "fidelity": {k: _mean([r["fidelity"][k] for r in recs])
                             for k in ("mag_corr", "band_lsd_db", "phase_corr_mw",
                                       "rir_pearson", "t20_rel_err")},
                "mode_shape_invariance": {
                    "gt": _mean([r["mode_shape_gt"] for r in recs]),
                    "pred": _mean([r["mode_shape_pred"] for r in recs]),
                },
            }
        heldout_by_combo[name] = per

    sel_out: Dict[str, dict] = {}
    for mat in sorted(selectivity):
        sel_out[mat] = {}
        for wall in WALLS_2D:
            if wall not in selectivity[mat]:
                continue
            sel_out[mat][wall] = {
                fam: {"gt_d_bw": _mean(selectivity[mat][wall][fam]["gt"]),
                      "pred_d_bw": _mean(selectivity[mat][wall][fam]["pred"]),
                      "theory_d_bw": _mean(selectivity[mat][wall][fam]["theory"]),
                      "residual_d_bw": _mean(selectivity[mat][wall][fam]["pred"])
                      - _mean(selectivity[mat][wall][fam]["gt"]),
                      "n": len(selectivity[mat][wall][fam]["gt"])}
                for fam in FAMILIES if fam in selectivity[mat][wall]
            }

    def sel_index(which: str) -> Tuple[float, dict]:
        """A = |delta-BW on the wall's own axis| / |delta-BW on the other axis|."""
        per_mat: Dict[str, float] = {}
        allv: List[float] = []
        for mat, walls in sel_out.items():
            vals = []
            for wall, fams in walls.items():
                own = X_AXIAL if WALL_AXIS[wall] == "x" else Y_AXIAL
                other = Y_AXIAL if own == X_AXIAL else X_AXIAL
                if own not in fams or other not in fams:
                    continue
                a, b = fams[own][which], fams[other][which]
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                vals.append(abs(a) / max(abs(b), SELECTIVITY_FLOOR_HZ))
            per_mat[mat] = _mean(vals)
            allv.extend(vals)
        return (_mean(allv), per_mat)

    a_gt, a_gt_mat = sel_index("gt_d_bw")
    a_pr, a_pr_mat = sel_index("pred_d_bw")
    a_th, a_th_mat = sel_index("theory_d_bw")

    c2_axial = [v for f in AXIAL_FAMILIES for v in floor_bw[f] if np.isfinite(v)]
    controls = {
        "C1_null_model": {
            "definition": "model's BASELINE render scored against the EDITED ground truth",
            "per_split": {n: {"model_band_lsd_db": splits_out[n]["fidelity"]["band_lsd_db"],
                              "null_band_lsd_db": splits_out[n]["null_fidelity"]["band_lsd_db"],
                              "edit_gain": splits_out[n]["edit"]["edit_gain"],
                              "E_BW_hz": splits_out[n]["edit"]["E_BW_hz"],
                              "null_E_BW_hz": splits_out[n]["edit"]["gt_effect_size_hz"]}
                          for n in SPLIT_ORDER},
        },
        "C2_floor_hz": float(np.mean(c2_axial)) if c2_axial else _nan(),
        "C2_detail": {
            "gt_within_family_bw_std_hz": {f: _mean(floor_bw[f]) for f in FAMILIES},
            "pred_within_family_bw_std_hz": {f: _mean(floor_bw_pred[f]) for f in FAMILIES},
            "note": ("axial families only for the headline floor: ISM-ray damping is "
                     "mode-independent within an axial family, so the spread is estimator "
                     "noise; tangential modes genuinely differ and are reported separately"),
            "n_geometries": len(geoms),
        },
        "C3_conditioning_identity": None,   # filled below
        "C3_detail": None,
        "C4_wall_identity": control_c4(maps),
    }
    c3 = control_c3(model, renderer, geoms[sorted(geoms)[0]], device, rx_chunk)
    controls["C3_conditioning_identity"] = bool(c3["pass"])
    controls["C3_detail"] = c3

    gap_hz = splits_out[SPLIT_III]["edit"]["E_BW_hz"] - splits_out[SPLIT_I]["edit"]["E_BW_hz"]
    eff_iii = splits_out[SPLIT_III]["edit"]["gt_effect_size_hz"]
    summary = {
        "checkpoint": str(ckpt),
        "iter": it,
        "in_dist_val_lsd_db": in_dist_val_lsd(train_path, it),
        "splits": splits_out,
        "selectivity_matrix": sel_out,
        "selectivity_index": {"gt": a_gt, "pred": a_pr, "theory": a_th,
                              "gt_per_material": a_gt_mat, "pred_per_material": a_pr_mat,
                              "theory_per_material": a_th_mat,
                              "floor_hz": SELECTIVITY_FLOOR_HZ,
                              "source": "frozen test geometries only"},
        "controls": controls,
        "heldout_combos": [list(x) for x in HELDOUT_COMBOS],
        "unseen_alpha": UNSEEN_ALPHA,
        "heldout_by_combo": heldout_by_combo,
        "gap_i_iii": {
            "E_BW_hz_i": splits_out[SPLIT_I]["edit"]["E_BW_hz"],
            "E_BW_hz_iii": splits_out[SPLIT_III]["edit"]["E_BW_hz"],
            "gap_hz": float(gap_hz),
            "gt_effect_size_hz_iii": eff_iii,
            "gap_pct_of_gt_effect": float(100.0 * gap_hz / eff_iii)
            if np.isfinite(eff_iii) and eff_iii > 0 else _nan(),
        },
        "meta": {
            "band_hz": [0.0, BAND_HI_HZ],
            "f_max_projection_hz": F_MAX_PROJECTION_HZ,
            "theory_slope_ism_ray": slope,
            "with_decay": bool(with_decay),
            "limit": limit,
            "n_configs_evaluated": len(records),
            "n_geometries": len(geoms),
            "runtime_s": round(time.time() - t_start, 1),
            "scoping": ("the ~29:1 bandwidth selectivity is a property of the ISM "
                        "simulator (angle-independent reflection, no grazing-incidence "
                        "absorption); real locally-reacting walls follow Kuttruff (~2:1, "
                        "no invariant family), so the claim is that the model learns the "
                        "SIMULATOR's per-wall law"),
        },
    }

    (out_path / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out_path / "per_config.json").write_text(json.dumps(records, indent=1, default=float))
    print("[done] wrote {} and {} ({:.1f}s)".format(
        out_path / "summary.json", out_path / "per_config.json", time.time() - t_start))
    return summary


def main():
    ap = argparse.ArgumentParser(description="P3-2 zero-shot material-edit evaluation")
    ap.add_argument("--train-dir", default="outputs/p3_2/p3_2_main")
    ap.add_argument("--checkpoint", default=None, help="default: newest ckpt in --train-dir")
    ap.add_argument("--out", default="outputs/p3_2/eval")
    ap.add_argument("--data-dir", default="data/track_c_2d")
    ap.add_argument("--train-yaml", default="configs/sweeps_2d_mat/p3_2_train.yaml")
    ap.add_argument("--test-yaml", default="configs/sweeps_2d_mat/p3_2_test_frozen.yaml")
    ap.add_argument("--limit", type=int, default=None, help="configs per split (fast iter)")
    ap.add_argument("--rx-chunk", type=int, default=8, help="receivers per renderer call")
    ap.add_argument("--no-decay", action="store_true", help="skip per-mode gamma fits")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    s = run(a.train_dir, a.out, a.checkpoint, a.data_dir, a.train_yaml, a.test_yaml,
            a.limit, a.rx_chunk, not a.no_decay, a.device)
    print(json.dumps({k: s[k] for k in ("checkpoint", "iter", "in_dist_val_lsd_db")},
                     indent=1))
    print(json.dumps(s["splits"][SPLIT_III], indent=1))


if __name__ == "__main__":
    main()
