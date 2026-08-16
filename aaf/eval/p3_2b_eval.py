"""P3-2b zero-shot evaluation driver: the 4-arm ablation on compositional material transfer.

P3-2 established the measurement chain and then failed its headline test: a never-trained
(wall, alpha) pair recovered ~13% of the edit magnitude. Four causes were verified --
degenerate alpha sampling, one holdout that was an extrapolation rather than an
interpolation, a Fourier bandwidth far above what the alpha sampling could constrain, and
a target law that is exactly linear in ``m = -ln(1 - alpha)`` while the model was
conditioned on alpha. P3-2b fixes all four and runs arms A-D. This file scores them.

**Every estimator is imported from the frozen P3-2 modules, never reimplemented.** The
paired-delta machinery (``analyse``, ``paired_cells``, ``edit_stats``, ``by_family_stats``,
``fidelity``, ``db_map``, ``mode_shape_invariance``, ``theory_bw``, ``band_limit``) and the
C4 wall-identity control come in verbatim from ``aaf.eval.p3_2_eval``;
``tests/test_p3_2b_eval.py`` asserts object identity for each so a copy-paste divergence
fails CI. Comparability with every published P3-2 number depends on this: a
re-derived -3 dB walk, a different cap policy or a different projection cutoff would change
the numbers by more than the effect being measured.

Two things genuinely cannot be inherited verbatim, and both are about the arms differing in
their conditioning encoder:

* ``render_config`` hard-codes ``fourier_features_2d``. Arms C and D condition on the
  m-coordinate (60-d), so rendering them through the P3-2 function would silently feed a
  64-d alpha-Fourier vector into a network expecting 60 m-linear features -- a shape error
  at best and a wrong-arm evaluation at worst. :func:`render_config_arm` dispatches on the
  checkpoint's own ``cond_source`` via ``build_cond_vector_2d``, and is asserted to agree
  bit-for-bit with the P3-2 function on the ``geom_alpha_fourier`` arms.
* ``control_c3`` builds its identity-check vectors with the same hard-coded encoder, for the
  same reason.

Split definitions, the manifest-only assignment rule and the count assertions live in
``aaf.eval.p3_2b_splits``; the physics slope fit and the kappa correction live in
``aaf.eval.p3_2b_slopefit``; the frozen acceptance thresholds live in
``aaf.eval.p3_2b_accept``. The verdict is computed and printed before anything else is
written, so a figure can never be produced for a run whose verdict was not emitted.

Usage::

    python -m aaf.eval.p3_2b_eval --arm-dir outputs/p3_2/p3_2b_C_cont_mlinear \\
        [--checkpoint PATH] [--limit N] [--out outputs/p3_2b/eval/<arm>]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from aaf.eval.band_limited import band_indices
from aaf.eval.modal_projection import X_AXIAL, Y_AXIAL
from aaf.eval.p3_2_eval import (
    BAND_HI_HZ,
    FAMILIES,
    SELECTIVITY_FLOOR_HZ,
    GeomCtx,
    _mean,
    _nan,
    _pearson,
    _std,
    analyse,
    band_limit,
    by_family_stats,
    control_c4,
    db_map,
    edit_stats,
    fidelity,
    find_checkpoint,
    in_dist_val_lsd,
    load_gt,
    load_model,
    make_geom_ctx,
    mode_shape_invariance,
    paired_cells,
    theory_slope,
)
from aaf.eval.p3_2b_accept import verdict
from aaf.eval.p3_2b_slopefit import KAPPA, MIN_ALPHA_POINTS, fit_cell, slope_fit
from aaf.eval.p3_2b_splits import (
    S2,
    S5,
    SPLIT_ORDER,
    EvalConfig,
    assert_split_counts,
    build_splits,
    s5_slab_subset,
    slab_summary,
)
from aaf.models.conditioning_2d import build_cond_vector_2d
from aaf.walls import WALL_AXIS, WALLS_2D, alphas_for

AXIAL_FAMILIES = (X_AXIAL, Y_AXIAL)


# --------------------------------------------------------------------------- arm rendering
@torch.no_grad()
def render_config_arm(model, renderer, cond_source: str, L: float, W: float,
                      alphas: Sequence[float], rx: np.ndarray, src: np.ndarray,
                      device: torch.device, rx_chunk: int = 8) -> np.ndarray:
    """ZERO-SHOT render of one config at every receiver -> ``[n_rx, n_freq]`` complex64.

    Identical to ``p3_2_eval.render_config`` except that the conditioning vector is built
    through ``build_cond_vector_2d(cond_source, ...)`` instead of the hard-coded 64-d
    alpha-Fourier encoder, so the m-linear arms are rendered with the encoder they were
    trained with. Nothing about this config's ground truth is read.
    """
    cond = build_cond_vector_2d(cond_source, L, W, alphas, device=device)
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


@torch.no_grad()
def control_c3_arm(model, renderer, cond_source: str, geom: GeomCtx, device: torch.device,
                   rx_chunk: int) -> dict:
    """C3: setting a wall to the baseline material must be a no-op, in the conditioning
    vector AND in the render, for THIS arm's encoder.

    Structurally the P3-2 control; it cannot be imported verbatim because that version
    builds its vectors with ``fourier_features_2d``. Determinism is part of the check:
    ``FreqRenderer2D`` jitters ray azimuths while ``self.training``, so a renderer left in
    train mode fails here rather than quietly adding noise to every delta.
    """
    v_base = build_cond_vector_2d(cond_source, geom.L, geom.W, alphas_for(), device=device)
    vec_ok = all(
        torch.equal(v_base, build_cond_vector_2d(cond_source, geom.L, geom.W,
                                                 alphas_for(w, "M0"), device=device))
        for w in WALLS_2D
    )
    kw = dict(rx=geom.rx, src=geom.src, device=device, rx_chunk=rx_chunk)
    H_base = render_config_arm(model, renderer, cond_source, geom.L, geom.W,
                               alphas_for(), **kw)
    H_m0 = render_config_arm(model, renderer, cond_source, geom.L, geom.W,
                             alphas_for("west", "M0"), **kw)
    H_again = render_config_arm(model, renderer, cond_source, geom.L, geom.W,
                                alphas_for(), **kw)
    return {
        "pass": bool(vec_ok and np.array_equal(H_base, H_m0)
                     and np.array_equal(H_base, H_again)),
        "cond_vector_identical": bool(vec_ok),
        "render_bitwise_identical": bool(np.array_equal(H_base, H_m0)),
        "render_deterministic": bool(np.array_equal(H_base, H_again)),
        "cond_source": cond_source,
        "cond_dim": int(v_base.numel()),
        "geometry": [geom.L, geom.W],
    }


# --------------------------------------------------------------------------- aggregation
def _frac_dropped(cells: Sequence[dict]) -> float:
    """Fraction of well-excited modes that failed PAIRED validity (any of pred/GT x
    edit/baseline non-finite). Reported everywhere a delta is reported: a small E_BW over
    a handful of survivors is not the same result as a small E_BW over everything."""
    if not cells:
        return _nan()
    return float(1.0 - sum(1 for c in cells if c["bw_ok"]) / len(cells))


def _fid_mean(recs: Sequence[dict], key: str) -> dict:
    keys = ("mag_corr", "band_lsd_db", "phase_corr_mw", "rir_pearson", "t20_rel_err")
    return {k: _mean([r[key][k] for r in recs if key in r]) for k in keys}


def _edit_block(recs: Sequence[dict], cells: Sequence[dict]) -> dict:
    e = edit_stats(cells)
    lsd_model = _mean([r["fidelity"]["band_lsd_db"] for r in recs])
    lsd_null = _mean([r["null_fidelity"]["band_lsd_db"] for r in recs])
    e["edit_gain"] = float(lsd_null / lsd_model) if np.isfinite(lsd_null) \
        and np.isfinite(lsd_model) and lsd_model > 0 else _nan()
    e["model_band_lsd_db"] = lsd_model
    e["null_band_lsd_db"] = lsd_null
    return e


def _split_block(recs: Sequence[dict], cells: Sequence[dict]) -> dict:
    e = _edit_block(recs, cells)
    per_combo: Dict[str, dict] = {}
    for combo in sorted(set(r["combo_key"] for r in recs)):
        sub = [r for r in recs if r["combo_key"] == combo]
        sub_cells = [c for r in sub for c in r["cells"]]
        per_combo[combo] = {
            "n_configs": len(sub),
            "n_cells": int(sum(1 for c in sub_cells if c["bw_ok"])),
            "frac_modes_dropped": _frac_dropped(sub_cells),
            "edit": _edit_block(sub, sub_cells),
            "by_family": by_family_stats(sub_cells),
            "fidelity": _fid_mean(sub, "fidelity"),
            "mode_shape_invariance": {"gt": _mean([r["mode_shape_gt"] for r in sub]),
                                      "pred": _mean([r["mode_shape_pred"] for r in sub])},
        }
    return {
        "n_configs": len(recs),
        "n_cells": int(e["n_cells"]),
        "n_modes_candidate": len(cells),
        "frac_modes_dropped": _frac_dropped(cells),
        "fidelity": _fid_mean(recs, "fidelity"),
        "null_fidelity": _fid_mean(recs, "null_fidelity"),
        "edit": {k: e[k] for k in ("E_BW_hz", "edit_bw_slope", "edit_bw_pearson",
                                   "edit_gain", "E_LVL_db")},
        "edit_detail": e,
        "by_family": by_family_stats(cells),
        "mode_shape_invariance": {
            "gt": _mean([r["mode_shape_gt"] for r in recs]),
            "pred": _mean([r["mode_shape_pred"] for r in recs]),
            "n": len(recs),
        },
        "per_combo": per_combo,
    }


# --------------------------------------------------------------------------- driver
def run(arm_dir: str, out_dir: Optional[str], checkpoint: Optional[str], data_dir: str,
        limit: Optional[int], rx_chunk: int, with_decay: bool, device_str: str,
        n_boot: int, arm_spec: Optional[str] = None) -> dict:
    t0 = time.time()
    arm_path = Path(arm_dir)
    arm = arm_path.name
    out_path = Path(out_dir) if out_dir else Path("outputs/p3_2b/eval") / arm
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    ckpt = Path(checkpoint) if checkpoint else find_checkpoint(arm_dir)
    model, renderer, cfg, meta, it = load_model(ckpt, device)
    cond_source = str(cfg["cond_source"])
    cond_dim = int(cfg["cond_dim"])
    fs, n_time = float(cfg["fs"]), int(cfg["n_time_samples"])
    n_freq = n_time // 2 + 1
    _, hi_idx = band_indices(fs, n_freq, 0.0, BAND_HI_HZ)
    slope = theory_slope()
    if abs(slope - KAPPA) > 1e-9:
        raise RuntimeError(
            "gate slope {} disagrees with the frozen kappa {}; every rho in this chunk "
            "would be off by that ratio".format(slope, KAPPA))
    print("[arm] {} | ckpt {} (iter {}) | cond {} ({}d) | band 0:{} | {}".format(
        arm, ckpt.name, it, cond_source, cond_dim, hi_idx, device), flush=True)

    # ``--arm-spec p3_2c:<ARM>`` swaps in that arm's manifest and its per-arm expected
    # counts. Split ASSIGNMENT is unchanged -- P3-2c designates S2 by the frozen W015
    # predicate precisely so the density curve compares the same 20 rooms in every arm (see
    # aaf.eval.p3_2c_splits). Only XTRAP's split set differs, by separating its
    # beyond-edge west configs out of S1.
    if arm_spec and arm_spec.startswith("p3_2d:"):
        run_name = arm_spec.split(":", 1)[1]
        from aaf.eval.p3_2d_splits import (assert_split_counts_p3_2d,
                                           build_splits_p3_2d, curve_point)
        from aaf.eval.p3_2d_splits import split_order as _p3_2d_order
        splits, ctx = build_splits_p3_2d(run_name)
        assert_split_counts_p3_2d(splits, run_name)
        split_names = _p3_2d_order(run_name)
        ctx["curve_point"] = curve_point(run_name)
    elif arm_spec:
        family, _, arm_name = arm_spec.partition(":")
        if family != "p3_2c" or not arm_name:
            raise ValueError(
                f"--arm-spec must look like 'p3_2c:<ARM>' or 'p3_2d:<RUN>', got {arm_spec!r}")
        from aaf.eval.p3_2c_splits import (assert_split_counts_p3_2c,
                                           build_splits_p3_2c, curve_point)
        from aaf.eval.p3_2c_splits import split_order as _split_order
        splits, ctx = build_splits_p3_2c(arm_name)
        assert_split_counts_p3_2c(splits, arm_name)
        split_names = _split_order(arm_name)
        ctx["curve_point"] = curve_point(arm_name)
    else:
        splits, ctx = build_splits()
        assert_split_counts(splits)
        split_names = SPLIT_ORDER
    if limit:
        splits = {k: v[:limit] for k, v in splits.items()}
        print("[limit] {} configs per split (counts already asserted on the full set)"
              .format(limit), flush=True)

    test_keys = set(ctx["test_geoms"])
    baselines = ctx["baselines"]
    geoms: Dict[Tuple[float, float], GeomCtx] = {}
    records: List[dict] = []
    cells_by_split: Dict[str, List[dict]] = {k: [] for k in split_names}
    maps: Dict[Tuple, Dict[str, np.ndarray]] = {}
    selectivity: Dict[str, Dict[str, Dict[str, dict]]] = {}
    floor_bw: Dict[str, List[float]] = {f: [] for f in FAMILIES}
    floor_bw_pred: Dict[str, List[float]] = {f: [] for f in FAMILIES}
    # slope points: (geom_key, wall, family) -> {alpha: point}
    sf_points: Dict[Tuple, Dict[float, dict]] = {}

    def add_points(key_geom: Tuple[float, float], wall: str, alpha: float, d_m: float,
                   cells: Sequence[dict], in_slab_pt: bool) -> None:
        for fam in FAMILIES:
            fam_cells = [c for c in cells if c["family"] == fam]
            ok = [c for c in fam_cells if c["bw_ok"]]
            sf_points.setdefault((key_geom, wall, fam), {})[round(float(alpha), 6)] = {
                "alpha": float(alpha), "d_m": float(d_m), "in_slab": bool(in_slab_pt),
                "n_modes": len(ok), "n_modes_candidate": len(fam_cells),
                "d_bw_pred": float(np.mean([c["d_bw_pred"] for c in ok])) if ok else _nan(),
                "d_bw_gt": float(np.mean([c["d_bw_gt"] for c in ok])) if ok else _nan(),
            }

    def geom_ctx(L: float, W: float) -> GeomCtx:
        """Geometry context + its GT and predicted baseline streams, cached per geometry."""
        key = (round(L, 2), round(W, 2))
        if key in geoms:
            return geoms[key]
        base = baselines[key]
        H_gt, rx, src, _ = load_gt(Path(data_dir) / base.filename)   # h5 "split" ignored
        g = make_geom_ctx(L, W, rx, src, fs, n_freq)
        g.gt_base = analyse(band_limit(H_gt, hi_idx), g, alphas_for(), fs, n_time,
                            with_decay=with_decay)
        H_pred = render_config_arm(model, renderer, cond_source, L, W, alphas_for(),
                                   rx, src, device, rx_chunk)
        g.pred_base = analyse(band_limit(H_pred, hi_idx), g, alphas_for(), fs, n_time,
                              with_decay=with_decay)
        for fam in FAMILIES:
            idx = [i for i, m in enumerate(g.modes) if m.family == fam and g.used[i]]
            floor_bw[fam].append(_std([g.gt_base.bw[i] for i in idx]))
            floor_bw_pred[fam].append(_std([g.pred_base.bw[i] for i in idx]))
        # The d_m = 0 anchor of every slope fit on this geometry: the baseline against
        # itself, which is exactly (0, 0) and carries the baseline's own validity mask.
        anchor = paired_cells(g, g.gt_base, g.pred_base, alphas_for(), slope)
        for w in WALLS_2D:
            add_points(key, w, 0.15, 0.0, anchor, False)
        geoms[key] = g
        return g

    n_total = sum(len(v) for v in splits.values())
    done = 0
    for split_name in split_names:
        for c in splits[split_name]:
            assert isinstance(c, EvalConfig)
            done += 1
            g = geom_ctx(c.L, c.W)
            H_gt = band_limit(load_gt(Path(data_dir) / c.filename)[0], hi_idx)
            H_pred = band_limit(
                render_config_arm(model, renderer, cond_source, c.L, c.W, c.alphas,
                                  g.rx, g.src, device, rx_chunk), hi_idx)
            gt_e = analyse(H_gt, g, c.alphas, fs, n_time, with_decay=with_decay)
            pr_e = analyse(H_pred, g, c.alphas, fs, n_time, with_decay=with_decay)
            cells = paired_cells(g, gt_e, pr_e, c.alphas, slope)
            cells_by_split[split_name].extend(cells)

            records.append({
                "label": c.label, "split": split_name, "L": c.L, "W": c.W,
                "wall": c.wall, "material": c.material, "combo_key": c.combo_key,
                "edited": list(c.edited), "alphas": list(c.alphas),
                "d_m": c.d_m, "geom_seen": bool(c.geom_seen), "source": c.source,
                "touches_slab": bool(c.touches_slab),
                "fidelity": fidelity(H_pred, H_gt, fs, n_time, hi_idx),
                # C1 null model: the model's own BASELINE render vs the EDITED ground truth.
                "null_fidelity": fidelity(g.pred_base.H, H_gt, fs, n_time, hi_idx),
                "edit": edit_stats(cells),
                "by_family": by_family_stats(cells),
                "frac_modes_dropped": _frac_dropped(cells),
                "mode_shape_gt": mode_shape_invariance(H_gt, g.gt_base.H, g, fs),
                "mode_shape_pred": mode_shape_invariance(H_pred, g.pred_base.H, g, fs),
                "cond_phi": g.gt_base.cond_phi,
                "cells": cells,
            })

            if c.wall is not None:
                # Single-wall edits only: a two-wall delta is not attributable to one wall,
                # so it can enter neither the per-wall slope fit nor C4/selectivity.
                add_points(c.geom_key, c.wall, c.alphas[WALLS_2D.index(c.wall)],
                           c.d_m[c.wall], cells, c.touches_slab)
                maps[(c.L, c.W, c.wall, c.material)] = {
                    "gt": db_map(H_gt, g.gt_base.H, hi_idx),
                    "pred": db_map(H_pred, g.pred_base.H, hi_idx)}
                if c.geom_key in test_keys:
                    d = selectivity.setdefault(c.material, {}).setdefault(c.wall, {})
                    for fam in FAMILIES:
                        acc = d.setdefault(fam, {"gt": [], "pred": [], "theory": []})
                        for cell in cells:
                            if cell["family"] != fam or not cell["bw_ok"]:
                                continue
                            acc["gt"].append(cell["d_bw_gt"])
                            acc["pred"].append(cell["d_bw_pred"])
                            acc["theory"].append(cell["theory_d_bw"])
            if done % 20 == 0 or done == n_total:
                print("  [{}/{}] {} ({:.0f}s)".format(done, n_total, c.label,
                                                      time.time() - t0), flush=True)

    # ------------------------------------------------------------------ splits
    splits_out = {}
    for name in split_names:
        recs = [r for r in records if r["split"] == name]
        splits_out[name] = _split_block(recs, cells_by_split[name])
    s5_slab_labels = set(c.label for c in s5_slab_subset(splits))
    s5_slab = [r for r in records if r["split"] == S5 and r["label"] in s5_slab_labels]
    if s5_slab:
        splits_out[S5]["slab_subset"] = _split_block(
            s5_slab, [c for r in s5_slab for c in r["cells"]])
        splits_out[S5]["slab_subset"]["note"] = (
            "two-wall configs containing a slab value; branch order places them in S5, "
            "they are surfaced here so nothing is hidden by that ordering")

    # ------------------------------------------------------------------ slope fit
    sf_cells: List[dict] = []
    for (key_geom, wall, fam), pts in sorted(sf_points.items(), key=lambda kv: str(kv[0])):
        non_anchor = [p for a, p in pts.items() if abs(p["d_m"]) > 1e-12]
        if len(non_anchor) < MIN_ALPHA_POINTS - 1:
            # No alpha ladder on this geometry (the frozen test geometries carry the four
            # presets; a training geometry here carries at most the single slab value).
            continue
        sf_cells.append(fit_cell(sorted(pts.values(), key=lambda p: p["d_m"]),
                                 key_geom[0], key_geom[1], wall, fam))
    sf = slope_fit(sf_cells, n_boot=n_boot)

    # ------------------------------------------------------------------ selectivity
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

    # ------------------------------------------------------------------ controls
    c3 = control_c3_arm(model, renderer, cond_source, geoms[sorted(geoms)[0]], device,
                        rx_chunk)
    c2_axial = [v for f in AXIAL_FAMILIES for v in floor_bw[f] if np.isfinite(v)]
    controls = {
        "C1_edit_gain_per_split": {
            n: {"model_band_lsd_db": splits_out[n]["edit_detail"]["model_band_lsd_db"],
                "null_band_lsd_db": splits_out[n]["edit_detail"]["null_band_lsd_db"],
                "edit_gain": splits_out[n]["edit"]["edit_gain"],
                "E_BW_hz": splits_out[n]["edit"]["E_BW_hz"],
                "null_E_BW_hz": splits_out[n]["edit_detail"]["gt_effect_size_hz"]}
            for n in split_names},
        "C2_floor_hz": float(np.mean(c2_axial)) if c2_axial else _nan(),
        "C2_detail": {
            "gt_within_family_bw_std_hz": {f: _mean(floor_bw[f]) for f in FAMILIES},
            "pred_within_family_bw_std_hz": {f: _mean(floor_bw_pred[f]) for f in FAMILIES},
            "n_geometries": len(geoms),
            "note": ("axial families only for the headline floor: under the ISM-ray law "
                     "damping is mode-independent within an axial family, so the spread "
                     "is estimator noise and sets the resolution of every delta"),
        },
        "C3_conditioning_identity": bool(c3["pass"]),
        "C3_detail": c3,
        "C4_wall_identity": control_c4(maps),
    }

    # ------------------------------------------------------------------ verdict FIRST
    s2 = splits_out[S2]
    total_iters = int(meta.get("cfg", {}).get("n_iters", 0) or cfg.get("n_iters", 0))
    mid = bool(total_iters and it < total_iters)
    vd = verdict(arm, s2, sf, iter_=it, mid_training=mid)
    vd["total_iters"] = total_iters
    print("\n[VERDICT] " + vd["one_line"] + "\n", flush=True)

    summary = {
        "arm": arm,
        "checkpoint": str(ckpt),
        "iter": it,
        "in_dist_val_lsd_db": in_dist_val_lsd(arm_path, it),
        "cond_source": cond_source,
        "cond_dim": cond_dim,
        "splits": splits_out,
        "slope_fit": sf,
        "controls": controls,
        "selectivity_matrix": sel_out,
        "selectivity_index": {"gt": a_gt, "pred": a_pr, "theory": a_th,
                              "gt_per_material": a_gt_mat, "pred_per_material": a_pr_mat,
                              "theory_per_material": a_th_mat,
                              "floor_hz": SELECTIVITY_FLOOR_HZ,
                              "source": "frozen test geometries, single-wall edits only"},
        "verdict": vd,
        "slabs_m": slab_summary(),
        "meta": {
            "band_hz": [0.0, BAND_HI_HZ],
            "theory_slope_ism_ray": slope,
            "kappa": KAPPA,
            "manifest_sha": ctx["manifest_sha"],
            "arm_spec": arm_spec,
            "p3_2d": {k: ctx[k] for k in
                      ("run", "realized_delta_m", "nominal_delta_m", "grid_m", "midpoints",
                       "near_preset_grid_values", "midpoint_policy", "curve_point",
                       "delta_axis_note")
                      if k in ctx} or None,
            "p3_2c": {k: ctx[k] for k in
                      ("arm", "arm_spec", "manifest_path", "s2_designation",
                       "s2x_extra_curve_points", "training_support_m",
                       "curve_point", "annotations")
                      if k in ctx} or None,
            "train_manifest_sha": meta.get("manifest_sha"),
            "n_train_configs": meta.get("n_configs"),
            "total_iters": total_iters,
            "mid_training": mid,
            "with_decay": bool(with_decay),
            "limit": limit,
            "n_configs_evaluated": len(records),
            "n_geometries": len(geoms),
            "n_s5_slab_subset": len(s5_slab),
            "rx_chunk": rx_chunk,
            "runtime_s": round(time.time() - t0, 1),
            "scoping": ("the per-wall selectivity scored here is a property of the ISM "
                        "simulator (angle-independent reflection, no grazing-incidence "
                        "absorption); the claim is that the model learns the SIMULATOR's "
                        "per-wall law, not room acoustics in general"),
        },
    }

    (out_path / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out_path / "per_config.json").write_text(json.dumps(records, indent=1, default=float))
    (out_path / "verdict.json").write_text(json.dumps(vd, indent=1, default=float))
    print("[done] {} ({:.0f}s)".format(out_path, time.time() - t0), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="P3-2b zero-shot material-edit evaluation")
    ap.add_argument("--arm-dir", default="outputs/p3_2/p3_2b_C_cont_mlinear")
    ap.add_argument("--checkpoint", default=None, help="default: newest ckpt in --arm-dir")
    ap.add_argument("--out", default=None, help="default: outputs/p3_2b/eval/<arm>")
    ap.add_argument("--data-dir", default="data/track_c_2d")
    ap.add_argument("--limit", type=int, default=None, help="configs per split (fast iter)")
    ap.add_argument("--rx-chunk", type=int, default=8, help="receivers per renderer call")
    ap.add_argument("--no-decay", action="store_true", help="skip per-mode gamma fits")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm-spec", default=None,
                    help="p3_2c:<ARM> -- use that arm's manifest and counts")
    a = ap.parse_args()
    s = run(a.arm_dir, a.out, a.checkpoint, a.data_dir, a.limit, a.rx_chunk,
            not a.no_decay, a.device, a.n_boot, a.arm_spec)
    print(json.dumps({"arm": s["arm"], "iter": s["iter"],
                      "in_dist_val_lsd_db": s["in_dist_val_lsd_db"],
                      "S2": s["splits"][S2]["edit"],
                      "slab_local": s["slope_fit"]["aggregate"]["own_family"]["slab_local"],
                      "verdict": s["verdict"]["one_line"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
