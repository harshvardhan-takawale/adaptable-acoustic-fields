"""P3-2b m-response: the per-wall dose-response curve, ground truth and model side by side.

P3-2 scored a single held-out (wall, alpha) pair and got ~13% of the edit magnitude back.
One number on one point cannot say *why*. This module measures the whole curve instead:
sweep one wall's absorption over 20 values of the linearizing coordinate
``m = -ln(1 - alpha)`` and plot the resulting change in the AFFECTED family's modal
bandwidth against ``m``. Under the ISM-ray law that curve is a straight line through the
origin whose slope is fixed by the geometry alone, so the figure exposes three failure
modes a scalar score cannot separate:

* **wrong slope** -- the model under- or over-responds to material;
* **wrong shape** -- curvature, saturation or a sign flip inside the held-out slab;
* **wrong selectivity** -- the ORTHOGONAL family (measured here as an in-figure control)
  moves when it should stay at zero.

Three geometries are swept, chosen to span the aspect ratio L/W of the frozen test set.
That is not decoration: the theoretical slope goes as 1/L for west/east and 1/W for
south/north, so it varies ~1.8x across the three rooms. A model that learned one global
"more alpha -> more broadening" map produces the SAME slope everywhere and is falsified by
the figure without any further statistics.

The kappa correction
--------------------
``aaf.eval.modal_bandwidth`` measures a CALIBRATED -3 dB width, not the raw Lorentzian
width: the P3-2 physics gate (T5) fits ``BW_measured = 0.302 + 1.6608 * (gamma/pi)``. The
intercept cancels in a paired delta; the SLOPE DOES NOT. So the theoretical slope of a
MEASURED delta-bandwidth against delta-m is

    a_theory = kappa * c / (4 pi D),     kappa = 1.6607564051417665   (frozen, T5)

with ``D = L`` for west/east (x-axial family) and ``D = W`` for south/north (y-axial).
Using the raw ``c / (4 pi D)`` instead would score a perfect model at rho = 0.602 and fail
it. Both are reported -- ``a_theory_hz_per_m`` (kappa-scaled, the one rho is taken against)
and ``a_theory_raw_hz_per_m`` (transparency only).

Estimator reuse
---------------
Every bandwidth here comes from the frozen P3-2 chain -- ``modal_projection.project_field``
-> ``modal_bandwidth.measure_modes`` with ``caps_from_predicted_bw`` on the ISM-ray
predicted widths -- via :func:`aaf.eval.p3_2_eval.analyse`. Nothing is reimplemented, so
every number is comparable with the published P3-2 numbers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import yaml

from aaf.data.mat_configs import ALPHA_QUANT_DP, room_filename_2d_mat_v2
from aaf.data.mat_configs_cont import HOLDOUT_SLABS, alpha_of_m, in_slab, m_of_alpha
from aaf.eval.band_limited import band_indices
from aaf.eval.modal_projection import F_MAX_PROJECTION_HZ, X_AXIAL, Y_AXIAL
from aaf.eval.p3_2_eval import (
    BAND_HI_HZ,
    analyse,
    band_limit,
    find_checkpoint,
    load_gt,
    load_model,
    make_geom_ctx,
)
from aaf.eval.p3_2b_slopefit import KAPPA, a_theory_hz_per_m
from aaf.models.conditioning_2d import build_cond_vector_2d
from aaf.walls import ALPHA_BASELINE, WALL_AXIS, WALL_INDEX, WALLS_2D

# The kappa correction has exactly one home -- ``aaf.eval.p3_2b_slopefit`` -- and this
# module imports it rather than restating the formula. Two copies of the highest-risk
# constant in the chunk is how a published rho quietly moves by 1.66x.
if abs(KAPPA - 1.6607564051417665) > 1e-12:
    raise ImportError(
        "aaf.eval.p3_2b_slopefit.KAPPA is {} but the P3-2 gate (T5 ism_ray slope) froze it "
        "at 1.6607564051417665; every rho in P3-2b is taken against that value".format(KAPPA))

M_SWEEP: Tuple[float, float] = (0.05, 1.55)
N_SWEEP = 20
N_GEOMS = 3

# Arm A trained on the P3-2 preset set, whose largest absorption is M3 = 0.70; arms B/C/D
# sample continuously up to alpha = 0.80. Points above this are range EXTRAPOLATION for A
# and interpolation for the rest -- the distinction has to survive into the figure.
ARM_A_ALPHA_MAX = 0.70

GT_SCHEMA = "p3_2b.m_response_gt/1"
SCHEMA = "p3_2b.m_response/1"

MANIFEST = "configs/sweeps_2d_mat/p3_2b_manifest.json"
SIM_YAML = "configs/sweeps_2d_mat/p3_2_train.yaml"
DATA_DIR = "data/track_c_2d"
GT_JSON = "outputs/p3_2b/mresponse_gt.json"
EVAL_ROOT = "outputs/p3_2b/eval"
TRAIN_ROOT = "outputs/p3_2"

ROLES = ("min_aspect", "median_aspect", "max_aspect")


# --------------------------------------------------------------------------- sweep design
def sweep_points(m_lo: float = M_SWEEP[0], m_hi: float = M_SWEEP[1],
                 n: int = N_SWEEP) -> List[Tuple[float, float]]:
    """The 20 sweep points as ``(alpha, m)``, uniform in m then quantized in ALPHA.

    Quantizing alpha (not m) is what makes the sweep point addressable: the v2 filename
    encodes alpha to 6 dp, so an alpha that survives ``round(..., 6)`` round-trips exactly
    between the manifest, the HDF5 name and the conditioning vector. ``m`` is then
    recomputed FROM the quantized alpha so the theory line is evaluated at the value
    actually simulated.
    """
    out: List[Tuple[float, float]] = []
    for m_target in np.linspace(float(m_lo), float(m_hi), int(n)):
        a = round(alpha_of_m(float(m_target)), ALPHA_QUANT_DP)
        out.append((a, m_of_alpha(a)))
    return out


def select_geometries(manifest_path: str = MANIFEST, n: int = N_GEOMS) -> List[dict]:
    """The ``n`` frozen-test geometries spanning aspect ratio L/W: min, median, max.

    Deterministic and manifest-driven -- no hand-picked room list to drift out of sync.
    The point of spanning L/W is that ``a_theory`` scales as 1/L (west/east) or 1/W
    (south/north), so these three rooms have visibly different slopes and a model with one
    global material response cannot fit all three.
    """
    rows = json.load(open(manifest_path))["configs"]
    geoms = sorted({(int(r["geom_id"]), round(float(r["L"]), 2), round(float(r["W"]), 2))
                    for r in rows if r["split"] == "test"})
    if len(geoms) < n:
        raise ValueError(f"only {len(geoms)} test geometries in {manifest_path}")
    by_aspect = sorted(geoms, key=lambda g: (g[1] / g[2], g[0]))
    idx = [0, len(by_aspect) // 2, len(by_aspect) - 1] if n == 3 else \
        list(np.linspace(0, len(by_aspect) - 1, n).round().astype(int))
    roles = ROLES if n == 3 else tuple("q%d" % i for i in range(n))
    out = []
    for role, i in zip(roles, idx):
        gid, L, W = by_aspect[i]
        out.append({"role": role, "geom_id": gid, "L": L, "W": W,
                    "aspect": round(L / W, 4)})
    return out


def families_for(wall: str) -> Tuple[str, str]:
    """``(own, orthogonal)`` mode family for a wall. west/east damp x-axial modes only."""
    own = X_AXIAL if WALL_AXIS[wall] == "x" else Y_AXIAL
    return own, (Y_AXIAL if own == X_AXIAL else X_AXIAL)


def theory_slopes(wall: str, L: float, W: float, kappa: float = KAPPA) -> dict:
    """Slope of MEASURED delta-BW against delta-m for this (wall, geometry), Hz per unit m.

    ISM-ray damping for the wall's own axial family is ``gamma = c (m_a + m_b) / (4 D)``,
    hence a raw Lorentzian ``d_BW = c d_m / (4 pi D)``; the estimator multiplies that by
    ``kappa``. The orthogonal family has cos(theta) = 0 exactly and gets nothing.

    Delegates to :func:`aaf.eval.p3_2b_slopefit.a_theory_hz_per_m` -- the sweep and the
    per-config eval must not be able to disagree about the theory line they score against.
    """
    own, orth = families_for(wall)
    return {
        "D_m": float(L) if WALL_AXIS[wall] == "x" else float(W),
        "a_theory_hz_per_m": float(a_theory_hz_per_m(L, W, wall, own, kappa=kappa)),
        "a_theory_raw_hz_per_m": float(a_theory_hz_per_m(L, W, wall, own, kappa=1.0)),
        "a_theory_orth_hz_per_m": float(a_theory_hz_per_m(L, W, wall, orth, kappa=kappa)),
    }


def alphas_with(wall: str, alpha: float, baseline: float = ALPHA_BASELINE) -> Tuple[float, ...]:
    """Baseline 4-vector with one wall overridden (WALLS_2D order)."""
    a = [float(baseline)] * len(WALLS_2D)
    a[WALL_INDEX[wall]] = float(alpha)
    return tuple(a)


def sim_common(path: str = SIM_YAML) -> dict:
    """The simulator settings the P3-2/P3-2b corpus was built with (single source)."""
    d = yaml.safe_load(open(path))
    return {k: d[k] for k in ("fs", "n_time_samples", "max_order", "source_pos",
                              "n_rx_per_side", "rx_margin")}


# --------------------------------------------------------------------------- measurement
def modes_to_dicts(geom) -> List[dict]:
    """Serializable mode table for a :class:`aaf.eval.p3_2_eval.GeomCtx`."""
    return [{"n_x": int(m.n_x), "n_y": int(m.n_y), "family": str(m.family),
             "f_hz": round(float(m.f), 4), "used": bool(geom.used[i])}
            for i, m in enumerate(geom.modes)]


def family_indices(modes: Sequence[dict], family: str, only_used: bool = True) -> List[int]:
    return [i for i, m in enumerate(modes)
            if m["family"] == family and (not only_used or m["used"])]


def delta_stats(bw_edit: Sequence[float], bw_base: Sequence[float],
                idx: Sequence[int]) -> dict:
    """Paired mean delta-bandwidth over one family.

    Paired per mode, then averaged: the estimator's bias depends on the mode's own
    frequency, and pairing cancels it to first order (the same argument P3-2's
    ``paired_cells`` rests on). Modes whose width is unresolvable at either end are
    dropped, and the drop is COUNTED -- ``n_modes < n_modes_total`` is the visible symptom
    of estimator breakdown at large m, not something to average away silently.
    """
    e = np.asarray(bw_edit, dtype=float)
    b = np.asarray(bw_base, dtype=float)
    d = np.array([e[i] - b[i] for i in idx
                  if np.isfinite(e[i]) and np.isfinite(b[i])], dtype=float)
    n = int(d.size)
    return {
        "d_bw_mean": float(np.mean(d)) if n else float("nan"),
        "d_bw_sem": float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "n_modes": n,
        "n_modes_total": int(len(idx)),
    }


def measure_bw(H: np.ndarray, geom, alphas: Sequence[float], fs: float, n_time: int,
               hi_idx: int) -> np.ndarray:
    """Per-mode -3 dB bandwidth of one 64-receiver field, via the FROZEN P3-2 chain."""
    st = analyse(band_limit(H, hi_idx), geom, alphas, fs, n_time, with_decay=False)
    return np.asarray(st.bw, dtype=float)


def geometry_context(L: float, W: float, data_dir: str = DATA_DIR):
    """``(GeomCtx, H_baseline_gt, fs, n_time, hi_idx)`` for one geometry.

    Receiver coordinates come from the stored baseline HDF5, never rebuilt from a margin
    default (see ``aaf.eval.modal_projection``).
    """
    base_file = Path(data_dir) / "L{:.2f}_W{:.2f}_aW0.15_aE0.15_aS0.15_aN0.15.h5".format(L, W)
    H_gt, rx, src, _ = load_gt(base_file)
    with h5py.File(str(base_file), "r") as f:
        fs = float(f.attrs["fs"])
        n_time = int(f.attrs["n_time_samples"])
    n_freq = int(H_gt.shape[1])
    if n_freq != n_time // 2 + 1:
        raise ValueError(f"{base_file}: n_freq {n_freq} inconsistent with n_time {n_time}")
    _, hi_idx = band_indices(fs, n_freq, 0.0, BAND_HI_HZ)
    geom = make_geom_ctx(float(L), float(W), rx, src, fs, n_freq)
    return geom, H_gt, fs, n_time, hi_idx, str(base_file)


# --------------------------------------------------------------------------- ground truth
def simulate_or_load(L: float, W: float, alphas: Sequence[float], rx: np.ndarray,
                     common: dict, data_dir: str = DATA_DIR) -> Tuple[np.ndarray, str, str]:
    """``(H, source, filename)``. Reuses a stored HDF5 when the exact config already exists.

    Sweep points are quantized to the same 6-dp alpha grid the corpus uses, so a point that
    coincides with a config already simulated for P3-2/P3-2b is byte-addressable by name --
    reusing it saves ~4.3 s and guarantees the reused point is the identical field, not a
    re-run with different pyroomacoustics internals.
    """
    fname = room_filename_2d_mat_v2(L, W, alphas)
    p = Path(data_dir) / fname
    if p.exists():
        return load_gt(p)[0], "reused", fname
    from aaf.sim.ism_2d import simulate_room_2d
    ism = simulate_room_2d(dict(
        L=float(L), W=float(W),
        source_pos=np.asarray(common["source_pos"], dtype=float),
        receiver_pos=np.asarray(rx, dtype=float), alphas=tuple(float(a) for a in alphas),
        fs=common["fs"], n_time_samples=common["n_time_samples"],
        max_order=common["max_order"]))
    return np.asarray(ism["H_complex"]), "simulated", fname


def gt_geometry(g: dict, walls: Sequence[str], points: Sequence[Tuple[float, float]],
                common: dict, data_dir: str = DATA_DIR, verbose: bool = True) -> dict:
    """Ground-truth sweep for one geometry over ``walls``. Per-mode bandwidths only.

    The 64x4097 complex fields are NOT persisted: 240 of them would be ~4 GB of HDF5 that
    nothing downstream reads, while the per-mode widths -- the only quantity the figure and
    the verdict use -- are a few kB per cell.
    """
    geom, H_base, fs, n_time, hi_idx, base_file = geometry_context(g["L"], g["W"], data_dir)
    base_alphas = alphas_with(WALLS_2D[0], ALPHA_BASELINE)
    bw_base = measure_bw(H_base, geom, base_alphas, fs, n_time, hi_idx)
    out = {
        "role": g["role"], "geom_id": g["geom_id"], "L": g["L"], "W": g["W"],
        "aspect": g["aspect"], "baseline_file": base_file,
        "cond_phi": None, "modes": modes_to_dicts(geom),
        "baseline_bw_hz": [float(x) for x in bw_base],
        "walls": {},
    }
    for wall in walls:
        t0 = time.time()
        pts = []
        for (alpha, m) in points:
            alphas = alphas_with(wall, alpha)
            H, source, fname = simulate_or_load(g["L"], g["W"], alphas, geom.rx, common,
                                                data_dir)
            bw = measure_bw(H, geom, alphas, fs, n_time, hi_idx)
            pts.append({"alpha": float(alpha), "m": float(m), "source": source,
                        "file": fname, "bw_hz": [float(x) for x in bw]})
        out["walls"][wall] = {"points": pts}
        if verbose:
            print("  [gt] L{:.2f} W{:.2f} {:<5s} {} points ({:.1f}s)".format(
                g["L"], g["W"], wall, len(pts), time.time() - t0), flush=True)
    return out


def gt_document(geometries: Sequence[dict], points: Sequence[Tuple[float, float]],
                common: dict) -> dict:
    """Envelope for the ground-truth file (shards fill in ``geometries``)."""
    return {
        "schema": GT_SCHEMA,
        "kappa": KAPPA,
        "sweep": {"m_lo": M_SWEEP[0], "m_hi": M_SWEEP[1], "n": len(points),
                  "alphas": [p[0] for p in points], "m": [p[1] for p in points],
                  "baseline_alpha": ALPHA_BASELINE,
                  "baseline_m": m_of_alpha(ALPHA_BASELINE)},
        "sim": dict(common),
        "measure": {
            "band_hz": [0.0, BAND_HI_HZ],
            "f_max_projection_hz": F_MAX_PROJECTION_HZ,
            "chain": ("modal_projection.project_field -> modal_bandwidth.measure_modes "
                      "with caps_from_predicted_bw(ism_ray predicted BW)"),
        },
        "holdout_slabs": {k: list(v) for k, v in HOLDOUT_SLABS.items()},
        "geometries": list(geometries),
    }


def merge_gt(parts: Sequence[dict]) -> dict:
    """Merge per-shard ground-truth documents into one, asserting they agree.

    Shards recompute each geometry's baseline independently; the measurement is
    deterministic, so disagreement means the shards did not see the same corpus and must
    fail loudly rather than average.
    """
    if not parts:
        raise ValueError("no parts to merge")
    doc = {k: v for k, v in parts[0].items() if k != "geometries"}
    by_gid: Dict[int, dict] = {}
    for part in parts:
        for g in part["geometries"]:
            gid = int(g["geom_id"])
            if gid not in by_gid:
                by_gid[gid] = {k: v for k, v in g.items() if k != "walls"}
                by_gid[gid]["walls"] = {}
            prev = np.asarray(by_gid[gid]["baseline_bw_hz"], dtype=float)
            cur = np.asarray(g["baseline_bw_hz"], dtype=float)
            if not np.array_equal(np.nan_to_num(prev, nan=-1.0),
                                  np.nan_to_num(cur, nan=-1.0)):
                raise AssertionError(f"baseline bandwidths disagree across shards for {gid}")
            for wall, w in g["walls"].items():
                if wall in by_gid[gid]["walls"]:
                    raise AssertionError(f"wall {wall} of geom {gid} appears in two shards")
                by_gid[gid]["walls"][wall] = w
    order = {int(g["geom_id"]): i for i, g in enumerate(parts[0]["geometries"])}
    doc["geometries"] = sorted(by_gid.values(),
                               key=lambda g: (order.get(int(g["geom_id"]), 99),
                                              int(g["geom_id"])))
    return doc


# --------------------------------------------------------------------------- model side
@torch.no_grad()
def render_cond(model, renderer, cond_source: str, L: float, W: float,
                alphas: Sequence[float], rx: np.ndarray, src: np.ndarray,
                device: torch.device, rx_chunk: int = 4) -> np.ndarray:
    """Zero-shot render of one config -> ``[n_rx, n_freq]`` complex64.

    Same contract as ``p3_2_eval.render_config`` but dispatched on ``cond_source``: the
    P3-2b arms differ in their conditioning encoder (``geom_alpha_fourier`` vs
    ``m_linear``), and hard-coding one of them would silently render arm C with arm A's
    features. ``rx_chunk`` defaults to 4 because these arms use ``n_pts_per_ray=64``,
    twice P3-2's, and the per-chunk activation is [rx * n_azi * n_pts, n_freq].
    """
    cond = build_cond_vector_2d(cond_source, float(L), float(W), tuple(alphas),
                                device=device)
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


def pred_mresponse(model, renderer, cond_source: str, geom, wall: str,
                   points: Sequence[Tuple[float, float]], fs: float, n_time: int,
                   hi_idx: int, device: torch.device, rx_chunk: int = 4) -> List[np.ndarray]:
    """Per-mode bandwidths of the MODEL's render at every sweep point of one wall.

    Identical measurement to the ground-truth side -- same projection, same caps, same
    estimator -- so the only difference between the two curves is the field itself.
    """
    out: List[np.ndarray] = []
    for (alpha, _m) in points:
        alphas = alphas_with(wall, alpha)
        H = render_cond(model, renderer, cond_source, geom.L, geom.W, alphas, geom.rx,
                        geom.src, device, rx_chunk)
        out.append(measure_bw(H, geom, alphas, fs, n_time, hi_idx))
    return out


def build_mresponse(arm: str, train_root: str = TRAIN_ROOT, gt_json: str = GT_JSON,
                    out_root: str = EVAL_ROOT, checkpoint: Optional[str] = None,
                    data_dir: str = DATA_DIR, device_str: str = "cuda",
                    rx_chunk: int = 4, walls: Sequence[str] = WALLS_2D,
                    limit_points: Optional[int] = None, kappa: float = KAPPA) -> dict:
    """Full m-response for one arm -> ``outputs/p3_2b/eval/<arm>/m_response.json``.

    The ground truth is read from ``gt_json`` (per-mode bandwidths), never re-simulated, so
    every arm is scored against literally the same numbers.
    """
    t0 = time.time()
    gt = json.load(open(gt_json))
    if gt.get("schema") != GT_SCHEMA:
        raise ValueError(f"{gt_json} has schema {gt.get('schema')!r}, expected {GT_SCHEMA!r}")
    points = list(zip(gt["sweep"]["alphas"], gt["sweep"]["m"]))
    if limit_points:
        points = points[:limit_points]
    m_base = float(gt["sweep"]["baseline_m"])

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    train_dir = Path(train_root) / arm
    ckpt = Path(checkpoint) if checkpoint else find_checkpoint(str(train_dir))
    model, renderer, cfg, meta, it = load_model(ckpt, device)
    cond_source = str(cfg["cond_source"])
    print("[{}] {} (iter {}) cond_source={} device={}".format(
        arm, ckpt, it, cond_source, device), flush=True)

    geoms_out: List[dict] = []
    for g in gt["geometries"]:
        geom, H_base_gt, fs, n_time, hi_idx, base_file = geometry_context(
            g["L"], g["W"], data_dir)
        modes = modes_to_dicts(geom)
        if modes != g["modes"]:
            raise AssertionError(
                "mode table for L{:.2f} W{:.2f} does not match the ground-truth file; the "
                "projection basis changed and the two sides are not comparable".format(
                    g["L"], g["W"]))
        bw_base_gt = np.asarray(g["baseline_bw_hz"], dtype=float)
        base_alphas = alphas_with(WALLS_2D[0], ALPHA_BASELINE)
        H_base_pred = render_cond(model, renderer, cond_source, geom.L, geom.W, base_alphas,
                                  geom.rx, geom.src, device, rx_chunk)
        bw_base_pred = measure_bw(H_base_pred, geom, base_alphas, fs, n_time, hi_idx)

        walls_out: Dict[str, dict] = {}
        for wall in walls:
            if wall not in g["walls"]:
                continue
            own, orth = families_for(wall)
            i_own, i_orth = family_indices(modes, own), family_indices(modes, orth)
            th = theory_slopes(wall, g["L"], g["W"], kappa=kappa)
            slab = HOLDOUT_SLABS.get(wall)
            gt_pts = {round(float(p["alpha"]), ALPHA_QUANT_DP): p
                      for p in g["walls"][wall]["points"]}
            bw_pred = pred_mresponse(model, renderer, cond_source, geom, wall, points, fs,
                                     n_time, hi_idx, device, rx_chunk)
            pts_out = []
            for (alpha, m), bwp in zip(points, bw_pred):
                gp = gt_pts[round(float(alpha), ALPHA_QUANT_DP)]
                bwg = np.asarray(gp["bw_hz"], dtype=float)
                d_m = float(m) - m_base
                pts_out.append({
                    "alpha": float(alpha), "m": float(m), "d_m": d_m,
                    "in_slab": bool(in_slab(wall, float(alpha))),
                    "alpha_above_arm_A_max": bool(float(alpha) > ARM_A_ALPHA_MAX),
                    "gt_source": gp["source"],
                    "pred": {"own": delta_stats(bwp, bw_base_pred, i_own),
                             "orth": delta_stats(bwp, bw_base_pred, i_orth)},
                    "gt": {"own": delta_stats(bwg, bw_base_gt, i_own),
                           "orth": delta_stats(bwg, bw_base_gt, i_orth)},
                    "theory_d_bw": float(th["a_theory_hz_per_m"] * d_m),
                    "theory_d_bw_orth": 0.0,
                })
            walls_out[wall] = {
                "axis": WALL_AXIS[wall], "own_family": own, "orth_family": orth,
                "D_m": th["D_m"], "a_theory_hz_per_m": th["a_theory_hz_per_m"],
                "a_theory_raw_hz_per_m": th["a_theory_raw_hz_per_m"],
                "slab_m": list(slab) if slab else None,
                "n_modes_own": len(i_own), "n_modes_orth": len(i_orth),
                "fit": fit_wall_cell(pts_out, th["a_theory_hz_per_m"], th["a_theory_raw_hz_per_m"]),
                # How often a mode fell out of the paired mean. A paired delta needs BOTH
                # ends finite, so a single unresolvable BASELINE mode silently removes that
                # mode from all 20 points -- the counts make that visible instead.
                "n_points_with_dropped_modes": {
                    "{}_{}".format(side, fam): sum(
                        1 for p in pts_out
                        if p[side][fam]["n_modes"] < p[side][fam]["n_modes_total"])
                    for side in ("gt", "pred") for fam in ("own", "orth")},
                "points": pts_out,
            }
        geoms_out.append({
            "role": g["role"], "geom_id": g["geom_id"], "L": g["L"], "W": g["W"],
            "aspect": g["aspect"], "baseline_file": base_file,
            # Kept so a dropped mode can be traced to its origin without re-rendering:
            # an unresolvable PREDICTED baseline width is a property of the model's field,
            # not of the edit, and it is the usual reason a cell loses a mode everywhere.
            "modes": modes,
            "baseline_bw_gt_hz": [float(x) for x in bw_base_gt],
            "baseline_bw_pred_hz": [float(x) for x in bw_base_pred],
            "walls": walls_out,
        })
        print("  [{}] L{:.2f} W{:.2f} done ({:.1f}s)".format(
            arm, g["L"], g["W"], time.time() - t0), flush=True)

    doc = {
        "schema": SCHEMA,
        "arm": arm,
        "checkpoint": str(ckpt),
        "iter": int(it),
        "cond_source": cond_source,
        "kappa": float(kappa),
        "gt_source": str(gt_json),
        "arm_A_alpha_max": ARM_A_ALPHA_MAX,
        "geometries": geoms_out,
        "meta": {
            "band_hz": [0.0, BAND_HI_HZ],
            "f_max_projection_hz": F_MAX_PROJECTION_HZ,
            "n_points": len(points),
            "rx_chunk": rx_chunk,
            "runtime_s": round(time.time() - t0, 1),
            "a_theory_note": ("kappa-scaled: a = kappa*c/(4 pi D) with D = L for west/east "
                              "and D = W for south/north; the raw Lorentzian slope is "
                              "reported alongside for transparency only"),
        },
    }
    out_dir = Path(out_root) / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "m_response.json").write_text(json.dumps(doc, indent=1, default=float))
    print("[done] wrote {} ({:.1f}s)".format(out_dir / "m_response.json",
                                             time.time() - t0), flush=True)
    return doc


# --------------------------------------------------------------------------- calibration
def fit_through_origin(x: Sequence[float], y: Sequence[float]) -> dict:
    """Least-squares ``y = a x`` (no intercept) with an R^2 taken about zero.

    No intercept because the paired delta is zero by construction at ``d_m = 0``; fitting
    one would absorb exactly the estimator offset the pairing already removed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or float(np.sum(x * x)) <= 0.0:
        return {"a": float("nan"), "r2": float("nan"), "n": int(x.size)}
    a = float(np.sum(x * y) / np.sum(x * x))
    ss_res = float(np.sum((y - a * x) ** 2))
    ss_tot = float(np.sum(y * y))
    return {"a": a, "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n": int(x.size)}


def fit_wall_cell(points: Sequence[dict], a_theory: float, a_theory_raw: float) -> dict:
    """Fitted slope of the GT and predicted curves for one (geometry, wall) cell.

    Named apart from ``p3_2b_slopefit.fit_cell``: that one fits a single family from the
    per-config eval with bootstrap CIs, this one fits both families of a dense sweep.
    """
    d_m = [p["d_m"] for p in points]
    out = {"a_theory_hz_per_m": float(a_theory),
           "a_theory_raw_hz_per_m": float(a_theory_raw)}
    for side in ("gt", "pred"):
        for fam in ("own", "orth"):
            f = fit_through_origin(d_m, [p[side][fam]["d_bw_mean"] for p in points])
            out["{}_{}".format(side, fam)] = f
            if fam == "own":
                out["rho_{}".format(side)] = (
                    float(f["a"] / a_theory) if np.isfinite(f["a"]) and a_theory else
                    float("nan"))
                out["rho_{}_raw_theory".format(side)] = (
                    float(f["a"] / a_theory_raw) if np.isfinite(f["a"]) and a_theory_raw
                    else float("nan"))
    return out


def gt_calibration(gt: dict, kappa: float = KAPPA) -> List[dict]:
    """GT-vs-theory slope table: one row per (geometry, wall) cell.

    This is the calibration evidence the whole verdict rests on. If the ground truth itself
    does not reproduce ``kappa*c/(4 pi D)``, then no rho computed against that line means
    anything -- so this table is checked BEFORE any model number is looked at.
    """
    m_base = float(gt["sweep"]["baseline_m"])
    rows: List[dict] = []
    for g in gt["geometries"]:
        modes = g["modes"]
        bw_base = np.asarray(g["baseline_bw_hz"], dtype=float)
        for wall, w in g["walls"].items():
            own, orth = families_for(wall)
            i_own, i_orth = family_indices(modes, own), family_indices(modes, orth)
            th = theory_slopes(wall, g["L"], g["W"], kappa=kappa)
            d_m, d_own, d_orth, n_drop, m_bad = [], [], [], 0, []
            for p in w["points"]:
                bw = np.asarray(p["bw_hz"], dtype=float)
                so = delta_stats(bw, bw_base, i_own)
                sx = delta_stats(bw, bw_base, i_orth)
                d_m.append(float(p["m"]) - m_base)
                d_own.append(so["d_bw_mean"])
                d_orth.append(sx["d_bw_mean"])
                if so["n_modes"] < so["n_modes_total"]:
                    n_drop += 1
                    m_bad.append({"m": round(float(p["m"]), 4),
                                  "alpha": float(p["alpha"]),
                                  "n_modes": so["n_modes"],
                                  "n_modes_total": so["n_modes_total"]})
            f_own = fit_through_origin(d_m, d_own)
            f_orth = fit_through_origin(d_m, d_orth)
            rows.append({
                "role": g["role"], "L": g["L"], "W": g["W"], "wall": wall,
                "axis": WALL_AXIS[wall], "own_family": own, "D_m": th["D_m"],
                "n_modes_own": len(i_own), "n_modes_orth": len(i_orth),
                "a_theory_hz_per_m": th["a_theory_hz_per_m"],
                "a_theory_raw_hz_per_m": th["a_theory_raw_hz_per_m"],
                "a_fit_hz_per_m": f_own["a"], "r2": f_own["r2"], "n_points": f_own["n"],
                "rho_vs_kappa_theory": (float(f_own["a"] / th["a_theory_hz_per_m"])
                                        if np.isfinite(f_own["a"]) else float("nan")),
                "rho_vs_raw_theory": (float(f_own["a"] / th["a_theory_raw_hz_per_m"])
                                      if np.isfinite(f_own["a"]) else float("nan")),
                "pct_err_vs_kappa_theory": (
                    float(100.0 * (f_own["a"] / th["a_theory_hz_per_m"] - 1.0))
                    if np.isfinite(f_own["a"]) else float("nan")),
                "a_fit_orth_hz_per_m": f_orth["a"],
                "n_points_with_dropped_modes": n_drop,
                "dropped": m_bad,
            })
    return rows


def arm_summary(doc: dict) -> List[dict]:
    """Per-cell rho for one arm, the compact form the verdict table is built from."""
    rows = []
    for g in doc["geometries"]:
        for wall, w in g["walls"].items():
            f = w["fit"]
            rows.append({
                "role": g["role"], "L": g["L"], "W": g["W"], "wall": wall,
                "D_m": w["D_m"], "a_theory_hz_per_m": w["a_theory_hz_per_m"],
                "a_gt_hz_per_m": f["gt_own"]["a"], "a_pred_hz_per_m": f["pred_own"]["a"],
                "rho_gt": f["rho_gt"], "rho_pred": f["rho_pred"],
                "r2_pred": f["pred_own"]["r2"],
                "a_pred_orth_hz_per_m": f["pred_orth"]["a"],
                "a_gt_orth_hz_per_m": f["gt_orth"]["a"],
            })
    return rows


def format_arm_summary(rows: Sequence[dict]) -> str:
    hdr = ("role            L     W     wall   D     a_theory  a_gt     a_pred   "
           "rho_gt rho_pred  orth_pred")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            "{:<14s} {:<5.2f} {:<5.2f} {:<6s} {:<5.2f} {:>8.3f} {:>8.3f} {:>8.3f} "
            "{:>6.3f} {:>8.3f} {:>10.3f}".format(
                r["role"], r["L"], r["W"], r["wall"], r["D_m"], r["a_theory_hz_per_m"],
                r["a_gt_hz_per_m"], r["a_pred_hz_per_m"], r["rho_gt"], r["rho_pred"],
                r["a_pred_orth_hz_per_m"]))
    return "\n".join(lines)


def format_calibration(rows: Sequence[dict]) -> str:
    """Fixed-width table for the chunk writeup."""
    hdr = ("role            L     W     wall   fam        D     a_theory  a_fit    "
           "rho    err%    r2      n  drop")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            "{:<14s} {:<5.2f} {:<5.2f} {:<6s} {:<9s} {:<5.2f} {:>8.3f} {:>8.3f} "
            "{:>6.3f} {:>6.2f} {:>6.4f} {:>3d} {:>4d}".format(
                r["role"], r["L"], r["W"], r["wall"], r["own_family"], r["D_m"],
                r["a_theory_hz_per_m"], r["a_fit_hz_per_m"], r["rho_vs_kappa_theory"],
                r["pct_err_vs_kappa_theory"], r["r2"], r["n_points"],
                r["n_points_with_dropped_modes"]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI
def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="P3-2b model-side m-response")
    ap.add_argument("--arm", required=True, help="run_id, e.g. p3_2b_C_cont_mlinear")
    ap.add_argument("--train-root", default=TRAIN_ROOT)
    ap.add_argument("--checkpoint", default=None, help="default: newest in the arm's dir")
    ap.add_argument("--gt", default=GT_JSON)
    ap.add_argument("--out-root", default=EVAL_ROOT)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rx-chunk", type=int, default=4)
    ap.add_argument("--walls", default=",".join(WALLS_2D))
    ap.add_argument("--limit-points", type=int, default=0,
                    help="truncate the sweep (smoke tests only; 0 = all)")
    a = ap.parse_args()
    doc = build_mresponse(
        a.arm, train_root=a.train_root, gt_json=a.gt, out_root=a.out_root,
        checkpoint=a.checkpoint, data_dir=a.data_dir, device_str=a.device,
        rx_chunk=a.rx_chunk, walls=tuple(w.strip() for w in a.walls.split(",") if w.strip()),
        limit_points=a.limit_points or None)
    print(format_arm_summary(arm_summary(doc)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
