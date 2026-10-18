"""P3-2 trainer: one 2D model conditioned on (L, W, alpha_west..alpha_north).

Follows the P3-1 Arm-G recipe -- physical parameters -> Fourier features -> FiLM, with no
latent table -- under the band-limited (0-300 Hz) protocol.

Written as a focused 2D trainer rather than a clone of ``multi_room_3d`` (which carries
DDP, the eigen/resonance arm, an elevation ray axis and a 3-wide geometry head, none of
which apply) and NOT as an extension of ``multi_room`` (which has no band mask, no arm
dispatch, builds two optimizer groups unconditionally and dereferences
``model.latents.weight`` in validate() -- all of which break with ``latents=None``).
Phase-1/2 trainers are untouched.

Three things here differ from the 3D trainer for reasons specific to this chunk:

* **Targets are stored band-limited.** The loss and the val metric are both masked to
  bins 0..600, so only those bins are preloaded: 440 x 64 x 601 complex64 = 135 MB
  instead of 923 MB, which fits comfortably on-GPU and removes the per-iteration
  host->device copy entirely.
* **Validation is on HELD-OUT RECEIVERS** of the training configs (8 of the 64 per
  config, deterministic stride), not a subsample of receivers the model trained on. That
  makes the reported val LSD a genuine in-distribution generalization number, which is
  what every zero-shot claim is qualified by.
* **Renders are grouped by GEOMETRY, not by config.** The renderer's AABB depends only on
  (L, W) while ``z_s`` is per-row, so all configs sharing a geometry render in one call --
  strictly fewer launches than grouping per config, which with 440 configs would make
  almost every sub-batch size 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from aaf.data.mat_configs import HELDOUT_COMBOS, enumerate_configs
from aaf.eval.band_limited import band_indices
from aaf.models.conditioning_2d import (
    COND_SOURCE,
    FOURIER_DIM_2D,
    build_cond_vector_2d,
    cond_dim_for,
    fourier_features_2d,
)
from aaf.models.inr_2d import INR2D_AutoDecoder
from aaf.renderers.freq_2d import FreqRenderer2D

# Receivers held out of training for the in-distribution val metric (8 of 64).
VAL_RX = tuple(range(3, 64, 8))
"""Held-out receivers for the frozen 64-receiver corpora. Kept as a literal so every existing
run's ``train_meta.json`` and any resumed run stay byte-comparable."""


def signal_kwargs(cfg, n_band: int) -> dict:
    """D144: the model kwargs for the PB3 / W3 signal pathway, from a train cfg (dataclass or
    dict). The SAME function builds the model in training and in `load_model`, so an evaluation
    cannot rebuild a different pathway from the same checkpoint (D73)."""
    g = (lambda k, d: getattr(cfg, k, d)) if not isinstance(cfg, dict) else (lambda k, d: cfg.get(k, d))
    heads = int(g("signal_heads", 1) or 1)
    band_only = bool(g("signal_band_only", False))
    width = int(g("signal_n_neurons", 0) or 0)
    if heads <= 1 and not band_only and width <= 0:
        return {}
    df = float(g("fs", 4096)) / float(g("n_time_samples", 8192))
    edges_hz = [float(x) for x in str(g("signal_band_edges_hz", "") or "").split(",") if x.strip()]
    return {"signal_heads": heads, "signal_band_bins": int(n_band), "signal_n_neurons": width,
            "signal_band_edges": [int(round(e / df)) for e in edges_hz],
            "signal_xfade_bins": int(round(float(g("signal_xfade_hz", 10.0)) / df))}


P48_KEYS = ("mb_grids", "mb_f_top_hz", "mb_base_cell_m", "mb_n_levels", "mb_cells_per_wavelength", "ls_mode",
            "ls1_weight", "ls1_radius_m", "ls_band_hz", "ls2_weight", "ls2_rx_per_step", "ls2_floor_db")


def multiband_kwargs(cfg) -> dict:
    """P4-8 (D174(e)): the model kwargs for the per-band models, from a train cfg (dataclass or dict) -- the SAME
    function builds the model in training and in `load_model` (D73). MB1 ("matched"): per band a 6-level grid
    whose coarsest cell is mb_base_cell_m and finest is c / (mb_cells_per_wavelength f_top), on the positions /
    world_scale domain (tcnn: cell = world_scale / (base_resolution * s^l - 1)). MB2 ("default"): the cfg grid."""
    g = (lambda k, d: getattr(cfg, k, d)) if not isinstance(cfg, dict) else (lambda k, d: cfg.get(k, d))
    mode = str(g("mb_grids", "") or "")
    if not mode:
        return {}
    base = dict(otype="HashGrid", n_levels=int(g("n_levels", 20)), n_features_per_level=2,
                log2_hashmap_size=int(g("log2_hashmap_size", 18)), base_resolution=int(g("base_resolution", 16)),
                per_level_scale=float(g("per_level_scale", 1.5)))
    if mode == "default":
        return {"mb_grid_cfgs": [dict(base) for _ in range(3)]}
    if mode != "matched":
        raise ValueError("mb_grids must be '', 'matched' or 'default', got {!r}".format(mode))
    ws = g("world_scale", None)
    if ws is None:
        raise ValueError("MB1 grids are defined on the positions / world_scale domain; world_scale is unset")
    ws = float(ws)
    br = ws / float(g("mb_base_cell_m", 1.0)) + 1.0
    if abs(br - round(br)) > 1e-9:
        raise ValueError("world_scale / mb_base_cell_m + 1 = {} must be an integer base resolution".format(br))
    br = int(round(br))
    nl = int(g("mb_n_levels", 6))
    c = float(g("c", 343.0))
    out = []
    for f in [float(x) for x in str(g("mb_f_top_hz", "173,245,300")).split(",")]:
        cell = c / (float(g("mb_cells_per_wavelength", 16.0)) * f)
        s = ((ws / cell + 1.0) / br) ** (1.0 / (nl - 1))
        out.append(dict(base, n_levels=nl, base_resolution=br, per_level_scale=float(s)))
    if len(out) != 3:
        raise ValueError("mb_f_top_hz needs three band tops, got {}".format(len(out)))
    return {"mb_grid_cfgs": out}


def mb_grid_table(grid_cfgs, world_scale: float) -> list:
    """Per band: each level's resolution, cell size (m) and whether tcnn stores it densely (res^2 <= 2^log2)."""
    import numpy as _np
    rows = []
    for g in grid_cfgs:
        lv = []
        for l in range(int(g["n_levels"])):
            scale = _np.float32(_np.exp2(_np.float32(l) * _np.log2(_np.float32(g["per_level_scale"])))
                                * _np.float32(g["base_resolution"]) - 1.0)
            res = int(_np.ceil(scale)) + 1
            lv.append({"level": l, "res": res, "cell_m": float(world_scale) / float(scale),
                       "dense": bool(res * res <= 2 ** int(g["log2_hashmap_size"]))})
        rows.append({"grid": dict(g), "levels": lv})
    return rows


def ls1_loss(P: torch.Tensor, T: torch.Tensor, lo: int, hi: int, m_b: torch.Tensor, scale: float) -> torch.Tensor:
    """P4-8 LS1: rows come as interleaved (anchor, neighbour) pairs [a0, n0, a1, n1, ...]. Per pair and bin in
    [lo, hi): |(P_a - P_n) - (T_a - T_n)| (complex modulus) / m_b, averaged, times `scale` (S_init)."""
    dP = P[0::2, lo:hi] - P[1::2, lo:hi]
    dT = T[0::2, lo:hi] - T[1::2, lo:hi]
    return scale * ((dP - dT).abs() / m_b).mean()


def ls2_loss(P: torch.Tensor, T: torch.Tensor, bands, floor_db: float = -60.0, eps: float = 1e-12) -> torch.Tensor:
    """P4-8 LS2 for ONE room: P, T [n_rx, n_bins]. Per bin: 1 - Pearson over receivers of dB|.|, with dB floored
    at floor_db relative to that bin's RMS over the same receivers (each array by its own RMS; the floor is
    detached). Averaged over each band's bins, the bands weighted equally. bands: [(lo, hi), ...] bin ranges."""
    def db(X):
        pw = X.real ** 2 + X.imag ** 2
        fl = (10.0 ** (floor_db / 10.0)) * pw.mean(0, keepdim=True).detach()
        return 10.0 * torch.log10(pw + fl + eps)
    out = 0.0
    for lo, hi in bands:
        a, b = db(P[:, lo:hi]), db(T[:, lo:hi])
        a = a - a.mean(0, keepdim=True)
        b = b - b.mean(0, keepdim=True)
        r = (a * b).sum(0) / torch.sqrt((a * a).sum(0) * (b * b).sum(0) + eps)
        out = out + (1.0 - r).mean() / len(bands)
    return out


def val_rx_indices(n_rx: int):
    """Held-out receiver indices for a grid of ``n_rx`` receivers.

    The 64-receiver corpora keep the historical set EXACTLY. A per-scene grid of another size
    (P4-1 Stage 1 fits one L-room with ~1900 receivers) gets the same 1-in-8 stride, which
    matters because the hard-coded ``range(3, 64, 8)`` would otherwise hold out 8 receivers out
    of 1903 -- all of them inside the first two columns of the grid -- and the val metric that
    drives early stopping would be measuring one corner of the room."""
    if n_rx == 64:
        return VAL_RX
    if n_rx < 16:
        raise ValueError("need >= 16 receivers to hold any out, got {}".format(n_rx))
    return tuple(range(3, n_rx, 8))


@dataclass
class P32TrainCfg:
    run_id: str = "p3_2_main"
    rooms_yaml: str = "configs/sweeps_2d_mat/p3_2_train.yaml"
    data_dir: str = "data/track_c_2d"
    configs_manifest: str = ""            # "" -> legacy preset enumeration (arm A / P3-2)
    config_kinds: Tuple[str, ...] = ()    # () = all kinds; ("baseline","single") = arm D
    n_iters: int = 60_000
    batch_size: int = 16
    rx_per_config: int = 4
    grad_accum_steps: int = 1
    lr_network: float = 2.0e-4
    eta_min: float = 5.0e-5
    grad_clip_max_norm: float = 1.0
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.1)
    n_azi: int = 64
    n_pts_per_ray: int = 64
    max_rows_per_render: int = 8          # caps rows per renderer call in the no_grad
                                          # validation path; see the note in _render
    near: float = 1e-3
    fs: int = 4096
    n_time_samples: int = 8192
    c: float = 343.0
    band_max_hz: float = 300.0
    latent_dim: int = 16                 # unused by the geom arm; >0 satisfies the model
    cond_source: str = COND_SOURCE
    cond_dim: int = FOURIER_DIM_2D
    conditioning_type: str = "film"
    log2_hashmap_size: int = 18
    n_levels: int = 20
    per_level_scale: float = 1.5
    base_resolution: int = 16
    #: P4-1 (D64). None = legacy (x+1)/2 position map; a float switches to absolute
    #: metres / world_scale. Changing it WITHOUT raising base_resolution costs real spatial
    #: resolution -- at world_scale 10 a 6 m room spans 0.6 hash units instead of 3.0, i.e.
    #: 5x fewer cells across the room -- so the two keys are meant to move together.
    world_scale: Optional[float] = None
    #: P4-2. Pooling ARM for the polygon token families: "masked_mean" (P4-1 default) or
    #: "extent_sum" (sum_i extent_i * phi(token_i)). Empty string = the arm's own default.
    #: Token counts vary across shapes, so this is a real experimental axis, not a detail.
    token_pool: str = ""
    #: P4-1. Low edge of the LOSS band in Hz (the eval band is unchanged). Set to 20.0 to mask
    #: the FDTD corpus's DC/compliance term out of the gradient; 0.0 keeps every bin.
    loss_band_lo_hz: float = 0.0
    #: P4-3 Arm S. Weight on the sigma-occupancy auxiliary loss. 0.0 disables it entirely, so
    #: every pre-P4-3 arm is bit-identical by default. See `_sigma_aux` for what it penalizes.
    sigma_aux_weight: float = 0.0
    #: Points per L-room config in the sigma auxiliary term, split 50/50 solid/air.
    sigma_aux_n_points: int = 512
    #: Target sigma ratio (solid / air) the auxiliary hinge pushes toward. The loss is zero once
    #: the measured ratio exceeds this, so a model that discovers occlusion on its own is not
    #: dragged further by a term it no longer needs.
    sigma_aux_ratio_target: float = 3.0
    #: P4-5 B1/B1b. Path to a .npy of one weight per supervised bin, applied to L_spec_real and
    #: L_spec_imag only (L_amp and L_phase are already scale-free). "" = off, and the loss then
    #: runs the ORIGINAL code path, so every earlier arm is bit-identical. See D92 for why this
    #: re-weights rather than equalises: the spec terms are L1.
    loss_spec_bin_weights: str = ""
    #: P4-5 a3 (D150). Floored log-amplitude loss: L_amp on log10(|H| + eps_b) with
    #: eps_b = loss_amp_floor_frac x the per-bin target RMS read from loss_amp_floor_ref (a JSON
    #: with "rms_per_bin", e.g. outputs/p4_5/arms/weights/B1_weights.json). 0.0 = off, and the
    #: loss runs the ORIGINAL code path (scalar eps 1e-6), so every earlier arm is bit-identical.
    loss_amp_floor_frac: float = 0.0
    loss_amp_floor_ref: str = ""
    #: P4-5 B3. Frequency curriculum "it0:hz0,it1:hz1,...": from iteration it_k the loss covers
    #: bins below hz_k. Masked bins carry exactly zero gradient, and the mean stays over the FULL
    #: band so an active bin weighs what it does in the baseline. "" = off. Validation is always
    #: full-band and unweighted, so val curves stay comparable across arms.
    loss_curriculum: str = ""
    #: P4-5 B2. Coarse-to-fine hash levels "start:every": `start` levels active at iteration 0
    #: and one more every `every` iterations. "" = off (all levels, bit-identical).
    c2f_levels: str = ""
    #: P4-5. Keep every checkpoint but strip optimizer/scheduler state from all but the newest,
    #: which is all a resume needs. ~2/3 of a checkpoint is Adam state, so retaining every 2K
    #: checkpoint for five arms costs ~38 GB instead of ~102 GB. Off by default.
    ckpt_thin_old: bool = False
    #: P4-5 B5. Number of low bins (from 0) covered by the near-DC compliance head, which predicts
    #: one spatially uniform complex value per room per bin from the conditioning and is added
    #: inside the renderer. 0 = no head (bit-identical to every earlier arm). 41 = 0-20 Hz.
    dc_head_bins: int = 0
    #: P4-5 PB3 / W3 (D144). Default = the single full-output signal MLP, bit-identical.
    #: signal_heads > 1: per-band signal heads with their own FiLM, split at signal_band_edges_hz
    #: (comma list), raised-cosine crossfade +-signal_xfade_hz. signal_band_only: heads produce
    #: only the supervised band. signal_n_neurons > 0: hidden-width override (W3 control).
    signal_heads: int = 1
    signal_band_only: bool = False
    signal_n_neurons: int = 0
    signal_band_edges_hz: str = ""
    signal_xfade_hz: float = 10.0
    val_every: int = 2_000
    val_max_configs: int = 88
    ckpt_every: int = 2_000
    ckpt_keep_last: int = 10
    #: P4-6 (D160). Also keep every checkpoint whose iteration is a multiple of this, beyond the
    #: `ckpt_keep_last` newest. 0 = off (the original rule). With keep_last 1 and keep_every
    #: 10000 a run retains every 10K plus its newest -- the final one when it ends.
    ckpt_keep_every: int = 0
    #: P4-6 (D160). Comma-separated iterations also kept, e.g. "4000" for D135's 4K kill-signal
    #: read, which keep_every would otherwise delete two thousand iterations later. "" = none.
    ckpt_keep_iters: str = ""
    log_every: int = 200
    early_stop_warmup: int = 10_000
    early_stop_patience: int = 10_000
    early_stop_min_rel_improvement: float = 0.003
    seed: int = 0
    #: P4-7 prep p4 (D164). Geometric augmentation, ALL OFF by default: with every probability at
    #: 0.0 nothing is imported, drawn or allocated, and `_step` runs its original lines.
    #: aug_d4_p: probability that a drawn config is replaced by a non-identity D4 image (receivers,
    #: source, polygon and wall alphas transformed together; the field is unchanged). aug_d4_elems:
    #: the allowed images; default = the flips, because the transposing ones carry rooms into the
    #: held-out d_hat slab (D164; refused when they would). aug_recip_p: probability that a row swaps
    #: source and receiver (target unchanged). aug_seed: -1 derives it from `seed`. The draws use
    #: their own CPU generator and never touch the global RNG streams.
    aug_d4_p: float = 0.0
    aug_d4_elems: str = "fx,fy,r180"
    aug_recip_p: float = 0.0
    aug_seed: int = -1
    #: P4-7 m4 (D169). Render every row with ITS OWN room's `source_pos` (the CX500 corpus draws one
    #: source per room). Off (default): the single global source, exactly as before -- and a corpus
    #: whose files disagree on the source is now REFUSED rather than silently trained against the
    #: last file's source.
    per_room_source: bool = False
    #: P4-7 m3 (D170(g)). modal_prior_dir: a directory of per-room `aaf.eval.modal_prior` files
    #: (`scripts/p4_7_m3_precompute.py`); when set, every render is H_modal + H_net and the loss is
    #: applied to the sum. modal_k: the number of mode clusters whose kernels (and k / k_600) are fed
    #: to the field network (0 = none; needs modal_prior_dir). Both default off; off, nothing is
    #: loaded and every render is the pre-change one.
    modal_prior_dir: str = ""
    modal_k: int = 0
    #: P4-8 (D174). mb_grids: "" (off) | "matched" (MB1: per-band wavelength-matched position grids) | "default"
    #: (MB2: the cfg grid in every band) -- separate per-band models (`multiband_kwargs`). mb_f_top_hz: each
    #: band's top frequency for MB1's finest cell = c / (mb_cells_per_wavelength f_top); mb_base_cell_m: the
    #: coarsest cell; mb_n_levels: levels per band grid.
    mb_grids: str = ""
    mb_f_top_hz: str = "173,245,300"
    mb_base_cell_m: float = 1.0
    mb_n_levels: int = 6
    mb_cells_per_wavelength: float = 16.0
    #: P4-8 spatial losses. ls_mode: "" (off: the original `_step`) | "ls1" (neighbour-difference loss with the
    #: pair sampler) | "ls2" (per-room correlation loss, one room x ls2_rx_per_step receivers per step, exact
    #: two-pass step). ls1_weight / ls2_weight: lambda (LS1 is scaled so its value at init equals the spec terms'
    #: value at init). ls_band_hz: the bins the losses see. ls2_floor_db: the dB floor relative to the bin's RMS.
    #: ls2_det_check_steps: steps (after any start) on which pass 2 must reproduce pass 1 bit-for-bit.
    ls_mode: str = ""
    ls1_weight: float = 0.0
    ls1_radius_m: float = 0.30
    ls_band_hz: str = "20,300"
    ls2_weight: float = 0.0
    ls2_rx_per_step: int = 32
    ls2_floor_db: float = -60.0
    ls2_det_check_steps: int = 20


def sample_solid_air(d: float, w: float, L: float, W: float, half: int,
                     device=None, generator=None) -> torch.Tensor:
    """``[2*half, 2]`` points in a top-left-notched room: first ``half`` INSIDE the removed
    corner, second ``half`` inside the air region. Module-level so it is directly testable.

    Stratified 50/50 rather than uniform over the bounding box, because at shallow ``d_hat``
    the notch is a few percent of the box and a uniform draw would put almost no points in
    solid -- exactly the regime P4-2's U-shaped accuracy curve says is weakest.

    The air half is drawn over the FULL box and any point that lands in the notch is redrawn
    along x only, to the right of the notch face. That keeps it allocation-free and branchless
    while guaranteeing the two halves are disjoint by construction rather than by rejection.
    """
    if half < 1:
        raise ValueError("half must be >= 1, got {}".format(half))
    if not (d > 0.0 and w > 0.0):
        raise ValueError("sample_solid_air needs a real notch (d>0, w>0), got d={}, w={}"
                         .format(d, w))
    kw = {"device": device, "generator": generator}
    u = torch.rand(half, 2, **kw)
    solid = torch.stack([u[:, 0] * w, (W - d) + u[:, 1] * d], dim=-1)
    v = torch.rand(half, 2, **kw)
    air = torch.stack([v[:, 0] * L, v[:, 1] * W], dim=-1)
    bad = (air[:, 0] < w) & (air[:, 1] > (W - d))
    n_bad = int(bad.sum())
    if n_bad:
        air[bad, 0] = w + (L - w) * torch.rand(n_bad, **kw)
    return torch.cat([solid, air], dim=0)


def _losses(H_pred, H_target, lo_bin: int = 0, w_spec=None, hi_bin=None, amp_eps=None):
    """The P3-1 4-term frequency loss. Both tensors are ALREADY band-limited.

    ``lo_bin`` drops bins ``[0, lo_bin)`` from every term by SLICING, so the excluded bins leave
    the autograd graph entirely and their gradient is exactly zero -- not merely down-weighted.
    Same mechanism as ``multi_room_3d._losses(..., band=)``.

    This exists for the FDTD corpus, whose bin-0 compliance term carries 87.3% of in-band power
    and sits 37.1 dB above the strongest room mode. Note the four terms are NOT equally exposed
    to it: ``L_spec_real``/``L_spec_imag`` are linear-domain L1 and so are dominated by it,
    while ``L_amp`` is log10 and ``L_phase`` is a cosine and neither is. The ISM corpus does not
    have the problem at all (bin 0 is 18.3 dB BELOW the strongest mode), so ``lo_bin`` stays 0
    there.

    P4-5 adds two optional arguments. With both at their default (None) the ORIGINAL code path
    below runs untouched, so bit-identity with every earlier arm is guaranteed by branching
    rather than argued from arithmetic.

    ``w_spec`` (B1/B1b): one weight per bin, multiplied into the two linear-domain spec terms.
    ``hi_bin`` (B3): bins ``[hi_bin, n)`` are dropped by SLICING -- the same mechanism as
    ``lo_bin``, so their gradient is exactly zero -- but the mean is still taken over the FULL
    band width. Plain slicing would divide by the shorter width and silently weight each active
    bin up by n/hi_bin (3x at a 0-100 Hz stage); dividing by the full width keeps every active
    bin at exactly the weight it has in the baseline, so the stage changes WHICH bins are fitted
    and nothing else.

    ``amp_eps`` (a3, D150): one floor per bin replacing L_amp's scalar eps, so the log-amplitude
    term stops amplifying differences between two near-null magnitudes (eps_b = 0.05 x the bin's
    target RMS, about -26 dB). Only L_amp changes; the other three terms are untouched.
    """
    if lo_bin:
        H_pred = H_pred[..., lo_bin:]
        H_target = H_target[..., lo_bin:]
        if w_spec is not None:
            w_spec = w_spec[lo_bin:]
        if amp_eps is not None:
            amp_eps = amp_eps[lo_bin:]
    eps = 1e-6
    if w_spec is not None or hi_bin is not None or amp_eps is not None:
        n_full = H_pred.shape[-1]
        if hi_bin is not None:
            H_pred, H_target = H_pred[..., :hi_bin], H_target[..., :hi_bin]
            if w_spec is not None:
                w_spec = w_spec[:hi_bin]
            if amp_eps is not None:
                amp_eps = amp_eps[:hi_bin]
        e_amp = eps if amp_eps is None else amp_eps
        denom = float(H_pred[..., 0].numel() * n_full)               # rows x FULL band
        e_r = (H_pred.real - H_target.real).abs()
        e_i = (H_pred.imag - H_target.imag).abs()
        if w_spec is not None:
            e_r, e_i = e_r * w_spec, e_i * w_spec
        return {
            "L_spec_real": e_r.sum() / denom,
            "L_spec_imag": e_i.sum() / denom,
            "L_amp": (torch.log10(H_pred.abs() + e_amp)
                      - torch.log10(H_target.abs() + e_amp)).abs().sum() / denom,
            "L_phase": (1.0 - torch.cos(H_pred.angle() - H_target.angle())).sum() / denom,
        }
    return {
        "L_spec_real": F.l1_loss(H_pred.real, H_target.real),
        "L_spec_imag": F.l1_loss(H_pred.imag, H_target.imag),
        "L_amp": F.l1_loss(torch.log10(H_pred.abs() + eps),
                           torch.log10(H_target.abs() + eps)),
        "L_phase": (1.0 - torch.cos(H_pred.angle() - H_target.angle())).mean(),
    }


def _lsd_db(H_pred, H_target, eps=1e-8):
    return float((20.0 * (torch.log10(H_pred.abs() + eps)
                          - torch.log10(H_target.abs() + eps))).abs().mean())


def _aug_requested(cfg) -> bool:
    """P4-7 prep p4: augmentation is on iff some probability is > 0."""
    return float(cfg.aug_d4_p) > 0.0 or float(cfg.aug_recip_p) > 0.0


def _prune_ckpts(out_dir: Path, keep_last: int, keep_every: int = 0, keep_iters=()) -> list:
    """Delete all but the `keep_last` newest `ckpt_iter*.pt`, sparing iteration multiples of
    `keep_every` when it is > 0. Returns the deleted paths. With keep_every = 0 this is exactly
    the rule `save_ckpt` always used."""
    ckpts = sorted(Path(out_dir).glob("ckpt_iter*.pt"))
    gone = []
    for old in ckpts[:-max(1, keep_last)]:
        it = int(old.stem.replace("ckpt_iter", ""))
        if (keep_every > 0 and it % keep_every == 0) or it in keep_iters:
            continue
        old.unlink()
        gone.append(old)
    return gone


class P32Trainer:
    def __init__(self, cfg: P32TrainCfg, output_dir: str):
        self.cfg = cfg
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        common = yaml.safe_load(open(cfg.rooms_yaml))
        geoms = [(g["L"], g["W"]) for g in common["geometries"]]
        self.manifest_sha = ""
        if cfg.configs_manifest:
            # P3-2b: the sampled set is FROZEN in git, so the dataset, the trainer and the
            # eval cannot disagree about which rooms exist.
            man = json.load(open(cfg.configs_manifest))
            # Dispatch on the manifest schema: P3-3-FAST rows carry 16 SEGMENT absorptions,
            # not 4 wall absorptions, so they need their own config class. Everything after
            # this point is schema-agnostic -- it only touches .alphas, .filename and .strata.
            if str(man.get("schema", "")).startswith("p3_3fast.trackA"):
                from aaf.data.seg_configs import configs_from_rows
            elif str(man.get("schema", "")).startswith("p4_1.poly"):
                # P4-1: polygonal rooms. Rows carry a VERTEX LIST and one absorption per edge
                # rather than (L, W) + 4 wall alphas -- an L-room has six edges and (L, W)
                # describes only its bounding box. The config exposes .verts, which
                # build_cond_vector_2d forwards to the geom_token featurizer.
                from aaf.data.poly_configs import configs_from_rows
            elif str(man.get("schema", "")).startswith("p4_2.shape"):
                # P4-2: the notch family. Rows carry (L, W, d, w) + a vertex list, and the
                # config exposes .verts / .edge_alphas which build_cond_vector_2d forwards to
                # the polygon token featurizer. WITHOUT this branch the schema falls through to
                # the mat_configs_cont default, which parses the rows happily -- they do have
                # L, W, alphas and filename -- and produces configs with NO verts, so every
                # L-room would be conditioned as a RECTANGLE with no error anywhere.
                from aaf.data.shape_configs import configs_from_rows
            elif str(man.get("schema", "")).startswith("p4_4.family"):
                # P4-4 Task 3: rect / L / U / T / DN in one corpus. Rows carry a NOTCH LIST
                # rather than (d, w), and the config derives .verts from it -- so the same
                # silent-rectangle failure described above applies with more force here. These
                # rows have no `d`/`w` keys at all, so the notch-family reader would raise a bare
                # KeyError rather than mis-parse; either way the branch is mandatory.
                from aaf.data.multi_notch import configs_from_rows
            elif str(man.get("schema", "")).startswith("p3_3fast.trackB"):
                # Track 2b rows carry a divider (x0) and a doorway width (a); their four wall
                # alphas are all baseline, so the aperture lives on the config object, not in
                # the alpha vector. Everything after this point stays schema-agnostic.
                from aaf.data.aperture_configs import configs_from_rows
            else:
                from aaf.data.mat_configs_cont import configs_from_rows
            self.manifest_sha = str(man.get("rows_sha256", ""))
            self.configs = configs_from_rows(man["configs"], split="train",
                                             kinds=tuple(cfg.config_kinds))
        else:
            self.configs = enumerate_configs(geoms, exclude_combos=HELDOUT_COMBOS)
        # A cond_dim that contradicts cond_source trains happily and silently produces the
        # WRONG ARM, which no metric would reveal -- so it is a hard check.
        expect = cond_dim_for(cfg.cond_source)
        if int(cfg.cond_dim) != expect:
            raise ValueError(
                f"cond_dim={cfg.cond_dim} contradicts cond_source={cfg.cond_source!r} "
                f"(expected {expect})")
        self.n_freq_bins = cfg.n_time_samples // 2 + 1
        self.band = band_indices(cfg.fs, self.n_freq_bins, 0.0, cfg.band_max_hz)
        # Loss-only low cut, expressed RELATIVE to the already-sliced band (which starts at
        # bin 0 = DC). Eval and the reported LSD keep the full band -- this masks the gradient,
        # not the metric.
        _df = float(cfg.fs) / float(cfg.n_time_samples)
        self.loss_lo_bin = int(round(float(cfg.loss_band_lo_hz) / _df))
        if self.loss_lo_bin >= (self.band[1] - self.band[0]):
            raise ValueError(
                "loss_band_lo_hz={} removes the whole band".format(cfg.loss_band_lo_hz))
        if self.loss_lo_bin:
            print("[loss] masking bins 0..{} (<{} Hz) out of the gradient".format(
                self.loss_lo_bin - 1, cfg.loss_band_lo_hz), flush=True)
        lo, hi = self.band
        self.n_band = hi - lo
        print(f"[data] {len(self.configs)} configs | band bins {lo}:{hi} ({self.n_band})")

        # ---- preload: only the supervised bins ------------------------------------
        data_dir = Path(cfg.data_dir)
        H_list, rx_list, geom_id, cfg_id, conds, srcs = [], [], [], [], [], []
        geom_index: Dict[Tuple[float, float], int] = {}
        self.geom_dims: List[Tuple[float, float]] = []
        t0 = time.time()
        for ci, c in enumerate(self.configs):
            with h5py.File(data_dir / c.filename, "r") as f:
                H = np.asarray(f["ism/H_complex"][:, lo:hi], dtype=np.complex64)
                rx = np.asarray(json.loads(f.attrs["receiver_pos"]), dtype=np.float32)
                src = np.asarray(json.loads(f.attrs["source_pos"]), dtype=np.float32)
                a_disk = json.loads(f.attrs["alphas"]) if isinstance(
                    f.attrs["alphas"], str) else list(f.attrs["alphas"])
            if not np.allclose(a_disk, c.alphas, atol=1e-9):
                raise ValueError(
                    f"manifest/data drift for {c.filename}: manifest {c.alphas} vs "
                    f"file {a_disk}")
            key = (c.L, c.W)
            if key not in geom_index:
                geom_index[key] = len(self.geom_dims)
                self.geom_dims.append(key)
            H_list.append(H)
            rx_list.append(rx)
            srcs.append(src)
            geom_id.append(geom_index[key])
            cfg_id.append(ci)
            # x0 / a exist only on Track 2b's ApertureConfig; every other arm ignores them.
            # A polygonal room carries ONE absorption PER EDGE, and there are more edges than
            # the shoebox 4 -- so the per-edge vector is passed when it exists. c.alphas is a
            # lossy bounding-box view that would be the wrong length for a 6-edge L-room.
            _verts = getattr(c, "verts", None)
            _al = getattr(c, "edge_alphas", None) if _verts is not None else None
            conds.append(
                build_cond_vector_2d(cfg.cond_source, c.L, c.W,
                                     c.alphas if _al is None else _al,
                                     x0=getattr(c, "x0", None),
                                     a=getattr(c, "a", None),
                                     verts=_verts).numpy())
        self.src = torch.tensor(src, device=self.device)
        # P4-7 m4 (D169): one source per config when asked; otherwise every file must agree.
        src_all = np.stack(srcs)
        if cfg.per_room_source:
            if cfg.sigma_aux_weight > 0.0:
                raise ValueError("per_room_source with sigma_aux_weight > 0 is not supported: the "
                                 "sigma term renders with the single global source")
            self.src_per = torch.tensor(src_all, device=self.device)               # [C,2]
        else:
            spread = float(np.abs(src_all - src_all[-1]).max())
            if spread > 1e-6:
                raise ValueError(
                    "the corpus has {} distinct source positions (max deviation {:.3f} m) but "
                    "per_room_source is off: the trainer would render every room with the LAST "
                    "file's source".format(len({tuple(x) for x in src_all.tolist()}), spread))
            self.src_per = None
        self.H = torch.tensor(np.stack(H_list), device=self.device)
        # P4-7 m3 (D170(g)): the physics prior and the modal-kernel features, per (config, receiver).
        self.H_prior = self.modal_feat = None
        if cfg.modal_k > 0 and not cfg.modal_prior_dir:
            raise ValueError("modal_k > 0 needs modal_prior_dir (the features come with the prior)")
        if cfg.modal_prior_dir and cfg.sigma_aux_weight > 0.0:
            raise ValueError("a modal prior with sigma_aux_weight > 0 is not supported (the sigma term "
                             "queries the field network without the modal features)")
        if cfg.modal_prior_dir:
            from aaf.eval.modal_prior import feat_rows, load_room
            if lo != 0 or hi > 601:
                raise ValueError("the modal prior is stored for bins 0-600; band {}:{} unsupported".format(lo, hi))
            hp, mf = [], []
            for ci, c in enumerate(self.configs):
                d = load_room(Path(cfg.modal_prior_dir) / (c.filename + ".npz"))
                if d["rx"].shape != rx_list[ci].shape or np.abs(d["rx"] - rx_list[ci]).max() > 1e-4:
                    raise ValueError("{}: the modal prior's receivers are not the corpus file's".format(
                        c.filename))
                if np.abs(d["src"] - srcs[ci]).max() > 1e-4:
                    raise ValueError("{}: the modal prior's source is not the corpus file's".format(c.filename))
                hp.append(d["H"][:, :self.n_band])
                if cfg.modal_k > 0:
                    mf.append(feat_rows(d["s"], d["kc"], int(cfg.modal_k)))
            self.H_prior = torch.tensor(np.stack(hp), device=self.device)           # [C, n_rx, B]
            if cfg.modal_k > 0:
                self.modal_feat = torch.tensor(np.stack(mf), device=self.device)    # [C, n_rx, 2k]
            print("[modal] prior loaded for {} configs | features {}".format(
                len(hp), 0 if self.modal_feat is None else int(self.modal_feat.shape[-1])), flush=True)          # [C,64,B]
        self.rx = torch.tensor(np.stack(rx_list), device=self.device)        # [C,64,2]
        self.geom_id = torch.tensor(geom_id, dtype=torch.long, device=self.device)
        self.cond = torch.tensor(np.stack(conds), dtype=torch.float32, device=self.device)
        print(f"[data] preloaded {self.H.numel()*8/1e6:.0f} MB in {time.time()-t0:.1f}s "
              f"| {len(self.geom_dims)} geometries")

        n_rx = int(self.rx.shape[1])
        self.val_rx = val_rx_indices(n_rx)
        val_mask = torch.zeros(n_rx, dtype=torch.bool)
        val_mask[list(self.val_rx)] = True
        self.train_rx_idx = torch.nonzero(~val_mask).squeeze(1).to(self.device)
        self.val_rx_idx = torch.nonzero(val_mask).squeeze(1).to(self.device)

        # Validation configs: a deterministic stride, stratified so every (wall, material)
        # appears -- otherwise the val curve is not comparable across iterations.
        # Stratify on a COARSE key and select by stride. Keying on the exact (wall, alpha)
        # combo -- as the P3-2 trainer did -- degenerates to one singleton group per config
        # once alpha is continuous, and the head-slice then draws the whole val set from the
        # first few geometries. Early stopping reads this metric, so that silently makes the
        # val curve non-comparable across arms.
        by_strata: Dict[str, List[int]] = {}
        for i, c in enumerate(self.configs):
            key = getattr(c, "strata", None) or str(getattr(c, "combo", i))
            by_strata.setdefault(key, []).append(i)
        val_ids: List[int] = []
        per = max(1, cfg.val_max_configs // max(1, len(by_strata)))
        for k in sorted(by_strata):
            grp = by_strata[k]
            stride = max(1, len(grp) // per)
            val_ids.extend(grp[::stride][:per])
        self.val_cfg_ids = torch.tensor(sorted(val_ids)[:cfg.val_max_configs],
                                        dtype=torch.long, device=self.device)
        print(f"[val] {len(self.val_cfg_ids)} configs x {len(self.val_rx_idx)} held-out receivers")

        # ---- model / renderer ------------------------------------------------------
        hg = dict(otype="HashGrid", n_levels=cfg.n_levels, n_features_per_level=2,
                  log2_hashmap_size=cfg.log2_hashmap_size,
                  base_resolution=cfg.base_resolution,
                  per_level_scale=cfg.per_level_scale)
        self.model = INR2D_AutoDecoder(
            n_rooms=len(self.configs), latent_dim=cfg.latent_dim,
            n_freq_bins=self.n_freq_bins, hash_grid_config=hg,
            conditioning_type=cfg.conditioning_type,
            cond_source=cfg.cond_source, cond_dim=cfg.cond_dim,
            l_head_enabled=False, world_scale=cfg.world_scale,
            token_pool=(cfg.token_pool or None),
            dc_head_bins=int(cfg.dc_head_bins),
            **signal_kwargs(cfg, self.n_band),
            **({"modal_feat_dim": 2 * int(cfg.modal_k), "modal_prior": True} if cfg.modal_prior_dir else {}),
            **multiband_kwargs(cfg),
        ).to(self.device)
        if self.model._model_signal is None:
            print("[signal] {}".format(self.model.signal_param_counts()), flush=True)
        if cfg.dc_head_bins > 0:
            # The head's output scale: corpus RMS of the SPATIAL-MEAN target per bin, training
            # receivers only. Set once before training; it is a buffer, so it is saved in every
            # checkpoint and a resume or an evaluation reproduces it without the corpus.
            nb = int(cfg.dc_head_bins)
            # P4-7 m3: with a modal prior the head models the RESIDUAL (the prior carries the zero mode)
            tgt = self.H if self.H_prior is None else self.H - self.H_prior
            m = tgt[:, self.train_rx_idx, :nb].mean(dim=1)                 # [C, nb] complex
            sc = m.abs().pow(2).mean(dim=0).sqrt().clamp(min=1e-6).float()
            with torch.no_grad():
                self.model.dc_scale.copy_(sc)
            print(f"[dc_head] {nb} bins (0-{(nb - 1) * cfg.fs / cfg.n_time_samples:.1f} Hz) | "
                  f"scale bin0 {float(sc[0]):.1f}  bin{nb - 1} {float(sc[-1]):.3g}")
        self.renderer = FreqRenderer2D(
            n_azi=cfg.n_azi, n_pts_per_ray=cfg.n_pts_per_ray, near=cfg.near,
            fs=cfg.fs, n_time_samples=cfg.n_time_samples, c=cfg.c,
        ).to(self.device)
        # Single param group: the geom arm has no latent table (no lr_latent).
        self.opt = torch.optim.Adam(
            [{"params": list(self.model.parameters()), "lr": cfg.lr_network,
              "name": "network"}])
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.n_iters, eta_min=cfg.eta_min)

        # ---- P4-5 loss/encoder options, all off by default ------------------------
        self.spec_w, self.spec_w_info = None, None
        if cfg.loss_spec_bin_weights:
            p = Path(cfg.loss_spec_bin_weights)
            w = np.load(p).astype(np.float32)
            if w.shape != (self.n_band,):
                raise ValueError(f"{p}: {w.shape} weights for a {self.n_band}-bin band")
            if not np.all(np.isfinite(w)) or (w <= 0).any():
                raise ValueError(f"{p}: weights must be finite and strictly positive")
            self.spec_w = torch.tensor(w, device=self.device)
            self.spec_w_info = {"path": str(p),
                                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                "min": float(w.min()), "max": float(w.max()),
                                "max_over_min": float(w.max() / w.min())}
            print(f"[loss] spec bin weights {p}: min {w.min():.4g} max {w.max():.4g} "
                  f"(ratio {w.max() / w.min():.1f})")
        self.amp_eps, self.amp_floor_info = None, None
        if cfg.loss_amp_floor_frac > 0.0:
            p = Path(cfg.loss_amp_floor_ref)
            rms = np.asarray(json.loads(p.read_text())["rms_per_bin"], dtype=np.float64)
            if rms.shape != (self.n_band,) or not np.all(np.isfinite(rms)) or (rms <= 0).any():
                raise ValueError(f"{p}: rms_per_bin must be {self.n_band} finite positive values")
            e = (cfg.loss_amp_floor_frac * rms).astype(np.float32)
            self.amp_eps = torch.tensor(e, device=self.device)
            self.amp_floor_info = {"path": str(p), "frac": cfg.loss_amp_floor_frac,
                                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                   "eps_min": float(e.min()), "eps_max": float(e.max())}
            print(f"[loss] L_amp floor {cfg.loss_amp_floor_frac} x rms_per_bin ({p}): "
                  f"eps min {e.min():.4g} max {e.max():.4g}")
        self.curriculum = self._parse_curriculum(cfg.loss_curriculum)
        self.c2f = None
        if cfg.c2f_levels:
            start, every = (int(x) for x in cfg.c2f_levels.split(":"))
            if not (1 <= start <= cfg.n_levels and every >= 1):
                raise ValueError(f"c2f_levels {cfg.c2f_levels!r}: need 1<=start<=n_levels, "
                                 f"every>=1")
            self.c2f = (start, every)
            print(f"[c2f] {start} levels at it 0, +1 every {every} -> all {cfg.n_levels} at "
                  f"it {(cfg.n_levels - start) * every}")

        self.scalars: List[dict] = []
        self.start_iter = self._maybe_resume()
        self._aug = self._init_augment() if _aug_requested(cfg) else None
        self._p48_refusals()
        self._ls = self._init_ls() if cfg.ls_mode else None
        self._step_fn = {"": self._step, "ls1": self._step_ls1, "ls2": self._step_ls2}[cfg.ls_mode]

    def _parse_curriculum(self, spec: str):
        """'0:100,15000:200,35000:300' -> [(0, hi_bin), (15000, hi_bin), ...], hi_bin in band
        bins. A stage whose edge reaches the band top maps to None: the full band, which runs
        the ORIGINAL loss path rather than an equivalent-but-different one."""
        if not spec:
            return None
        df = float(self.cfg.fs) / float(self.cfg.n_time_samples)
        out = []
        for part in spec.split(","):
            it_s, hz_s = part.split(":")
            hb = int(round(float(hz_s) / df)) + 1                    # bins 0..hz INCLUSIVE
            out.append((int(it_s), None if hb >= self.n_band else hb))
        out.sort()
        if out[0][0] != 0:
            raise ValueError(f"loss_curriculum {spec!r} must start at iteration 0")
        print(f"[curriculum] {out}")
        return out

    def _stage(self, it: int):
        """(hi_bin or None, active hash levels or None) for this iteration."""
        hi = None
        if self.curriculum:
            for start, hb in self.curriculum:
                if it >= start:
                    hi = hb
        lv = None
        if self.c2f:
            lv = min(self.cfg.n_levels, self.c2f[0] + it // self.c2f[1])
        return hi, lv

    # ---------------------------------------------------------------- rendering
    def _render(self, cfg_ids: torch.Tensor, rx_ids: torch.Tensor) -> torch.Tensor:
        """Render the given (config, receiver) pairs, grouped by geometry for the AABB."""
        out = torch.zeros(cfg_ids.numel(), self.n_band,
                          dtype=torch.complex64, device=self.device)
        g = self.geom_id[cfg_ids]
        for gid in torch.unique(g):
            sel = torch.nonzero(g == gid).squeeze(1)
            L, W = self.geom_dims[int(gid)]
            room_min = torch.zeros(2, device=self.device)
            room_max = torch.tensor([L, W], device=self.device, dtype=torch.float32)
            # Chunk the rows per renderer call. NOTE this bounds peak memory only under
            # no_grad (validation): during training every sub-batch feeds the SAME backward
            # graph, so their activations all stay alive until .backward() and chunking the
            # forward pass buys nothing. Training memory is controlled by rows-per-backward,
            # i.e. batch_size / grad_accum_steps -- at n_pts_per_ray=64 a row's renderer
            # tensor is ~134 MB, so 16 rows per backward OOMs a 24 GB card (measured) while
            # 8 rows reproduces P3-2's proven per-tensor footprint.
            step = max(1, int(self.cfg.max_rows_per_render))
            for s0 in range(0, sel.numel(), step):
                sub = sel[s0:s0 + step]
                rx = self.rx[cfg_ids[sub], rx_ids[sub]]
                src_per = getattr(self, "src_per", None)
                if src_per is None:
                    tx = self.src.unsqueeze(0).expand(sub.numel(), -1)
                else:
                    tx = src_per[cfg_ids[sub]]
                z = self.cond[cfg_ids[sub]]
                h_prior = getattr(self, "H_prior", None)
                if h_prior is None:
                    H = self.renderer(self.model, rx, tx, room_min, room_max, z_s=z)
                else:
                    mf = self.modal_feat
                    H = self.renderer(self.model, rx, tx, room_min, room_max, z_s=z,
                                      h_prior=h_prior[cfg_ids[sub], rx_ids[sub]],
                                      modal_feat=None if mf is None else mf[cfg_ids[sub], rx_ids[sub]])
                out[sub] = H[:, self.band[0]:self.band[1]]
        return out

    # ---------------------------------------------------------------- augmentation (P4-7 prep p4)
    def _init_augment(self) -> dict:
        """D164: precompute the D4 tables (index 0 = identity) and refuse unsupported set-ups."""
        from aaf.data.augment_2d import (D4_ELEMS, TRANSPOSING, canonical_ring, is_interior_node,
                                         precompute_d4_tables, transpose_holdout_leaks)
        cfg = self.cfg
        if cfg.sigma_aux_weight > 0.0:
            raise ValueError("augmentation with sigma_aux_weight > 0 is not supported: the sigma "
                             "term hard-codes the NW notch and the global source")
        if cfg.per_room_source:
            raise ValueError("augmentation with per_room_source is not implemented yet: the D4 "
                             "tables transform one global source (P4-7 m4, D169)")
        if cfg.modal_prior_dir:
            raise ValueError("augmentation with a modal prior is not implemented (P4-7 m3, D170)")
        elems = ["e"] + [g.strip() for g in str(cfg.aug_d4_elems).split(",")
                         if g.strip() and g.strip() != "e"]
        for g in elems:
            if g not in D4_ELEMS:
                raise ValueError("unknown D4 element {!r} in aug_d4_elems".format(g))
        if cfg.aug_d4_p > 0.0:
            if cfg.cond_source not in ("geom_token", "geom_token_m"):
                raise ValueError("D4 augmentation needs a geometry-token cond_source, got {!r}".format(
                    cfg.cond_source))
            if len(elems) < 2:
                raise ValueError("aug_d4_p > 0 but aug_d4_elems names no non-identity element")
        for c in self.configs:
            if getattr(c, "x0", None) is not None:
                raise ValueError("augmentation does not support aperture configs ({})".format(c.label))
            v = getattr(c, "verts", None)
            if v is None:
                raise ValueError("augmentation needs configs with verts ({})".format(c.label))
            if canonical_ring(v, c.edge_alphas)[0] != [tuple(map(float, q)) for q in v]:
                raise ValueError("native ring of {} is not canonical".format(c.label))
        leaks = []
        if cfg.aug_d4_p > 0.0 and any(g in TRANSPOSING for g in elems) and cfg.configs_manifest:
            slab = json.loads(Path(cfg.configs_manifest).read_text()).get("d_hat_holdout")
            if slab:
                leaks = transpose_holdout_leaks(self.configs, tuple(slab))
                if leaks:
                    raise ValueError("{} training rooms would be carried into the held-out d_hat slab "
                                     "{} by a transposing element; use the flips fx,fy,r180 (D164)".format(
                                         len(leaks), slab))
        src = self.src.detach().cpu().numpy().astype(np.float64)
        rx = self.rx.detach().cpu().numpy().astype(np.float64)
        n_wall = 0
        for i, c in enumerate(self.configs):
            with h5py.File(Path(cfg.data_dir) / c.filename) as f:
                s_i = np.asarray(json.loads(f.attrs["source_pos"]), dtype=np.float64)
                dxx = float(f.attrs.get("dx_x", f.attrs.get("dx", 0.02)))
                dxy = float(f.attrs.get("dx_y", f.attrs.get("dx", 0.02)))
            if np.abs(s_i - src).max() > 1e-6:
                raise ValueError("{}: source {} differs from the global source {}; the trainer keeps "
                                 "one".format(c.label, s_i.tolist(), src.tolist()))
            if cfg.aug_recip_p > 0.0:
                pts = np.concatenate([src[None], rx[i]], 0)
                n_wall += int((~is_interior_node(c.verts, pts, dxx, dxy)).sum())
        if n_wall:
            raise ValueError("reciprocity swaps need interior (non-wall) source and receivers; {} wall "
                             "nodes found".format(n_wall))
        T = precompute_d4_tables(self.configs, rx, src, cfg.cond_source, elems)
        dev = self.device
        seed_eff = int(cfg.aug_seed) if int(cfg.aug_seed) >= 0 else 1_000_003 * int(cfg.seed) + 17
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed_eff + int(self.start_iter))
        info = {"elems": elems, "aug_d4_p": float(cfg.aug_d4_p), "aug_recip_p": float(cfg.aug_recip_p),
                "seed_eff": seed_eff, "transpose_leaks": len(leaks)}
        print("[augment] {}".format(info), flush=True)
        return {"rx": torch.as_tensor(T["rx"], device=dev), "src": torch.as_tensor(T["src"], device=dev),
                "dims": torch.as_tensor(T["dims"], device=dev), "cond": torch.as_tensor(T["cond"], device=dev),
                "elems": elems, "gen": gen, "info": info}

    def _aug_draw(self, n_cfg: int, rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-config element index (0 = identity) and per-row swap flags, from the private generator
        with a FIXED consumption per call; the global CPU / CUDA RNG streams are never touched."""
        A, cfg = self._aug, self.cfg
        G = len(A["elems"])
        coin = torch.rand(n_cfg, generator=A["gen"]) < float(cfg.aug_d4_p)
        el = torch.randint(1, G, (n_cfg,), generator=A["gen"]) if G > 1 else torch.zeros(n_cfg, dtype=torch.long)
        g = torch.where(coin, el, torch.zeros_like(el))
        swap = torch.rand(rows, generator=A["gen"]) < float(cfg.aug_recip_p)
        return g, swap

    def _aug_forward(self, ci: torch.Tensor, ri: torch.Tensor, agg: dict):
        """Transformed rows -> (H_pred, H_tgt). D4 and the swap leave the target unchanged."""
        cfg, A = self.cfg, self._aug
        g, swap = self._aug_draw(ci.numel() // cfg.rx_per_config, ci.numel())
        g = g.repeat_interleave(cfg.rx_per_config).to(self.device)
        sw = swap.to(self.device)[:, None]
        rx = A["rx"][ci, g, ri]
        tx = A["src"][ci, g]
        rx, tx = torch.where(sw, tx, rx), torch.where(sw, rx, tx)
        k = float(cfg.grad_accum_steps)
        agg["aug_d4_frac"] = agg.get("aug_d4_frac", 0.0) + float((g > 0).float().mean()) / k
        agg["aug_recip_frac"] = agg.get("aug_recip_frac", 0.0) + float(sw.float().mean()) / k
        return self._render_rows(rx, tx, A["cond"][ci, g], A["dims"][ci, g]), self.H[ci, ri]

    def _render_rows(self, rx: torch.Tensor, tx: torch.Tensor, z: torch.Tensor,
                     dims: torch.Tensor) -> torch.Tensor:
        """`_render` for per-row receivers / sources / cond / room dims, grouped by room dims (the
        renderer takes one AABB per call)."""
        out = torch.zeros(rx.shape[0], self.n_band, dtype=torch.complex64, device=self.device)
        uniq, inv = torch.unique(dims, dim=0, return_inverse=True)
        step = max(1, int(self.cfg.max_rows_per_render))
        for gi in range(uniq.shape[0]):
            sel = torch.nonzero(inv == gi).squeeze(1)
            room_min = torch.zeros(2, device=self.device)
            room_max = uniq[gi].to(torch.float32)
            for s0 in range(0, sel.numel(), step):
                sub = sel[s0:s0 + step]
                H = self.renderer(self.model, rx[sub], tx[sub], room_min, room_max, z_s=z[sub])
                out[sub] = H[:, self.band[0]:self.band[1]]
        return out

    # ---------------------------------------------------------------- train step
    # ------------------------------------------------------------- sigma supervision (Arm S)
    def _sigma_aux(self, cfg_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Push the learned sigma field HIGH inside the removed corner (P4-3 Arm S).

        WHY THIS CANNOT READ THE RENDERER. `FreqRenderer2D.forward` returns only the integrated
        `H`; sigma exists inside it and is never exposed. Rather than change the renderer -- a
        wide-blast-radius edit this chunk is explicitly told not to make -- this calls the model
        DIRECTLY on its own point batch, exactly as `aaf.eval.sigma_probe.sigma_along_ray` does,
        and uses the SAME reduction (`attn.real.clamp(min=0).mean over frequency`) so the
        quantity being trained is the quantity the diagnostic reports. If those two diverged,
        Arm S could move its own loss without moving the number it is judged by.

        WHY THE MASK IS RECOMPUTED. P4-2's writeup claimed the dataset stores solid masks; it
        does not -- `build_p4_2_shapes.py` stores `notch_solid_nodes`, a scalar COUNT. What is
        stored is the geometry, and for this axis-aligned top-left notch solidity is a closed
        form: `x < w and y > W - d`. That is exact, needs no HDF5 read, and stays on the GPU
        (`polygon_contains` is numpy-only and would force a round-trip every step).

        WHY THE SAMPLING IS STRATIFIED. At shallow d_hat the notch is a few percent of the
        bounding box, so uniform sampling would put almost no points in solid -- precisely the
        regime the U-shaped accuracy curve says is weakest. Half the points are drawn inside the
        notch and half outside, so the signal does not thin out exactly where it is needed.

        Returns None when the batch contains no L-room (a pure rectangle has no solid region).
        """
        cfg_s = self.cfg
        n_pts = int(cfg_s.sigma_aux_n_points)
        ids = sorted({int(i) for i in cfg_ids.tolist()})
        ids = [i for i in ids if getattr(self.configs[i], "d", 0.0) > 0.0
               and getattr(self.configs[i], "w", 0.0) > 0.0]
        if not ids or n_pts < 2:
            return None
        half = n_pts // 2
        dev = self.device
        P, Z = [], []
        for i in ids:
            c = self.configs[i]
            P.append(sample_solid_air(float(c.d), float(c.w), float(c.L), float(c.W),
                                      half, device=dev))
            Z.append(self.cond[i])
        pts = torch.stack(P, dim=0)                                    # [G, n_pts, 2]
        # ONE vector per config, NOT one per point: `_expand_z_s` broadcasts over N itself and
        # rejects a pre-expanded [G, n_pts, D] tensor outright. Every point of a config shares
        # its conditioning here, which is exactly the property Arm X exists to change.
        z = torch.stack(Z, dim=0)                                      # [G, cond_dim]
        G, Np = pts.shape[0], pts.shape[1]
        view = torch.zeros_like(pts)
        view[..., 0] = 1.0                                             # sigma is view-free here
        tx = self.src.to(pts.dtype).view(1, 1, 2).expand(G, Np, 2)
        attn, _ = self.model(pts, view, tx, tx_view=None, z_s=z)
        sigma = attn.real.clamp(min=0).mean(dim=-1)                    # [G, n_pts]
        s_solid = sigma[:, :half].clamp(min=1e-8)
        s_air = sigma[:, half:].clamp(min=1e-8)
        # A MARGIN ON THE LOG-RATIO, not an absolute target: the probe reports solid/air, the
        # field's overall scale is free, and an absolute floor would fight the reconstruction
        # loss for no reason. relu() makes the term vanish once the ratio clears the target.
        gap = torch.log(s_solid).mean() - torch.log(s_air).mean()
        return torch.relu(math.log(float(cfg_s.sigma_aux_ratio_target)) - gap)

    def _step(self, it: int) -> dict:
        cfg = self.cfg
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        n_cfg = max(1, cfg.batch_size // cfg.rx_per_config)
        agg = {}
        hi_bin, levels = self._stage(it)
        if levels is not None:
            self.model.set_active_levels(levels)
            agg["c2f_active_levels"] = float(levels)
        if hi_bin is not None:
            agg["curriculum_hi_bin"] = float(hi_bin)
        for _ in range(cfg.grad_accum_steps):
            ci = torch.randint(0, len(self.configs), (n_cfg,), device=self.device)
            ci = ci.repeat_interleave(cfg.rx_per_config)
            ri = self.train_rx_idx[
                torch.randint(0, self.train_rx_idx.numel(), (ci.numel(),), device=self.device)]
            aug = getattr(self, "_aug", None)          # getattr: __new__ test stubs
            if aug is None:
                H_pred = self._render(ci, ri)
                H_tgt = self.H[ci, ri]
            else:
                H_pred, H_tgt = self._aug_forward(ci, ri, agg)
            terms = _losses(H_pred, H_tgt, self.loss_lo_bin, w_spec=self.spec_w, hi_bin=hi_bin,
                            amp_eps=self.amp_eps)
            w_r, w_i, w_a, w_p = cfg.weights
            loss = (w_r * terms["L_spec_real"] + w_i * terms["L_spec_imag"]
                    + w_a * terms["L_amp"] + w_p * terms["L_phase"])
            if cfg.sigma_aux_weight > 0.0:
                l_sig = self._sigma_aux(ci)
                if l_sig is not None:
                    contrib = cfg.sigma_aux_weight * l_sig
                    loss = loss + contrib
                    # The spec asks for the auxiliary term to sit at ~5-10% of the total. That
                    # is a property of the RUN, not of the weight, so it is measured rather
                    # than assumed -- the weight that produces 5% at iteration 0 need not
                    # produce 5% at 40K once the reconstruction loss has fallen.
                    terms["L_sigma_aux"] = l_sig.detach()
                    terms["sigma_aux_frac"] = (contrib / loss.clamp(min=1e-12)).detach()
            (loss / cfg.grad_accum_steps).backward()
            for k, v in terms.items():
                agg[k] = agg.get(k, 0.0) + float(v) / cfg.grad_accum_steps
            agg["loss"] = agg.get("loss", 0.0) + float(loss) / cfg.grad_accum_steps
        for p in self.model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_max_norm)
        self.opt.step()
        self.sched.step()
        return agg

    # ---------------------------------------------------------------- P4-8 (D174): MB / LS1 / LS2
    def _p48_refusals(self) -> None:
        cfg = self.cfg
        if cfg.ls_mode not in ("", "ls1", "ls2"):
            raise ValueError("ls_mode must be '', 'ls1' or 'ls2', got {!r}".format(cfg.ls_mode))
        if not (cfg.mb_grids or cfg.ls_mode):
            return
        bad = [k for k, on in (("modal_prior_dir", bool(cfg.modal_prior_dir)), ("augmentation", _aug_requested(cfg)),
                               ("sigma_aux_weight", cfg.sigma_aux_weight > 0.0),
                               ("loss_curriculum", bool(cfg.loss_curriculum)), ("c2f_levels", bool(cfg.c2f_levels)))
               if on]
        if bad:
            raise ValueError("P4-8 options (mb_grids / ls_mode) do not support: {}".format(bad))
        if cfg.ls_mode == "ls1" and (cfg.rx_per_config != 2 or cfg.batch_size % 2 or cfg.max_rows_per_render % 2):
            raise ValueError("LS1 rows come as (anchor, neighbour) pairs: need rx_per_config 2 and an even batch_size "
                             "and max_rows_per_render, so a pair never splits across renderer calls")
        if cfg.ls_mode == "ls2" and cfg.ls2_rx_per_step < 3:
            raise ValueError("LS2 needs >= 3 receivers per room for a Pearson correlation")

    def _init_ls(self) -> dict:
        """Precompute what the spatial losses need. No random number is drawn here."""
        cfg = self.cfg
        df = float(cfg.fs) / float(cfg.n_time_samples)
        lo_hz, hi_hz = (float(x) for x in str(cfg.ls_band_hz).split(","))
        lo, hi = int(round(lo_hz / df)), int(round(hi_hz / df)) + 1
        if self.band[0] != 0 or hi > self.n_band:
            raise ValueError("ls_band_hz {} does not fit the supervised band {}".format(cfg.ls_band_hz, self.band))
        L = {"lo": lo, "hi": hi, "info": {"mode": cfg.ls_mode, "band_bins": [lo, hi]}}
        tr = self.train_rx_idx
        if cfg.ls_mode == "ls1":
            r2 = float(cfg.ls1_radius_m) ** 2 + 1e-9
            C, n = len(self.configs), int(tr.numel())
            masks = []
            for c in range(C):
                P = self.rx[c, tr].double()
                d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
                m = d2 <= r2
                m.fill_diagonal_(False)
                masks.append(m)
            kmax = max(int(m.sum(1).max()) for m in masks)
            nbr = torch.zeros(C, n, kmax, dtype=torch.long, device=self.device)
            cnt = torch.zeros(C, n, dtype=torch.long, device=self.device)
            valid = torch.zeros(C, n, dtype=torch.long, device=self.device)
            n_valid = torch.zeros(C, dtype=torch.long, device=self.device)
            for c, m in enumerate(masks):
                cnt[c] = m.sum(1)
                nbr[c] = torch.sort(m.to(torch.uint8), dim=1, descending=True, stable=True).indices[:, :kmax]
                va = torch.nonzero(cnt[c] > 0).squeeze(1)
                valid[c, :va.numel()] = va
                n_valid[c] = va.numel()
            # m_b: the sampler's own expectation of |T_a - T_n| per bin (config uniform, anchor uniform among valid
            # anchors, neighbour uniform among its neighbours), so E[mean_b |dT_b| / m_b] = 1 exactly.
            m_b = torch.zeros(hi - lo, device=self.device)
            S = 0.0
            for c in range(C):
                T = self.H[c, tr]                                              # [n, B]
                S += float(T.real.abs().mean() + T.imag.abs().mean()) / C
                Tb = T[:, lo:hi]
                k_ok = torch.arange(kmax, device=self.device)[None] < cnt[c][:, None]       # [n, kmax]
                d = (Tb[:, None, :] - Tb[nbr[c]]).abs() * k_ok[..., None]                   # [n, kmax, B']
                per_anchor = d.sum(1) / cnt[c].clamp(min=1)[:, None].float()
                m_b += per_anchor[cnt[c] > 0].mean(0) / C
            d_all = [float(((self.rx[c, tr][:, None] - self.rx[c, tr][nbr[c]]).norm(dim=-1) *
                            (torch.arange(kmax, device=self.device)[None] < cnt[c][:, None])).sum()
                           / cnt[c].sum().clamp(min=1)) for c in range(C)]
            L.update(nbr=nbr, cnt=cnt, valid=valid, n_valid=n_valid, m_b=m_b, S=float(S))
            L["info"].update({"radius_m": float(cfg.ls1_radius_m), "weight": float(cfg.ls1_weight),
                              "anchors_with_neighbour": int((cnt > 0).sum()), "anchors_without": int((cnt == 0).sum()),
                              "neighbours_min_median_max": [int(cnt.min()), float(cnt.float().median()), int(cnt.max())],
                              "mean_pair_distance_m": float(np.mean(d_all)), "S_init_spec_terms": float(S),
                              "m_b_min_max": [float(m_b.min()), float(m_b.max())],
                              "m_b_sha256": hashlib.sha256(m_b.cpu().numpy().tobytes()).hexdigest()})
            print("[ls1] {}".format(L["info"]), flush=True)
        else:
            edges = [int(round(float(e) / df)) for e in str(cfg.signal_band_edges_hz or "173,245").split(",") if e.strip()]
            b = [lo] + [e for e in edges if lo < e < hi] + [hi]
            L["bands"] = list(zip(b[:-1], b[1:]))
            L["info"].update({"weight": float(cfg.ls2_weight), "rx_per_step": int(cfg.ls2_rx_per_step),
                              "bands_bins": L["bands"], "floor_db": float(cfg.ls2_floor_db), "pass1": "grad-enabled, detached"})
            self._ls2_steps = 0
            print("[ls2] {}".format(L["info"]), flush=True)
        return L

    def _apply_grads(self) -> None:
        cfg = self.cfg
        for p in self.model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_max_norm)
        self.opt.step()
        self.sched.step()

    def _ls1_draw(self):
        """n_pairs configs (uniform); per config one anchor uniform among its training receivers that HAVE a
        neighbour within the radius, and one neighbour uniform among them. Rows interleaved [a0, n0, a1, n1, ...]."""
        L = self._ls
        n_pairs = self.cfg.batch_size // 2
        ci = torch.randint(0, len(self.configs), (n_pairs,), device=self.device)
        a_pos = (torch.rand(n_pairs, device=self.device) * L["n_valid"][ci]).long()
        a = L["valid"][ci, a_pos]
        n_pos = (torch.rand(n_pairs, device=self.device) * L["cnt"][ci, a]).long()
        nb = L["nbr"][ci, a, n_pos]
        tr = self.train_rx_idx
        ri = torch.stack([tr[a], tr[nb]], dim=1).reshape(-1)
        rows = ci.repeat_interleave(2)
        dist = (self.rx[ci, tr[a]] - self.rx[ci, tr[nb]]).norm(dim=-1)
        return rows, ri, dist

    def _step_ls1(self, it: int) -> dict:
        cfg = self.cfg
        L = self._ls
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        agg = {}
        w_r, w_i, w_a, w_p = cfg.weights
        for _ in range(cfg.grad_accum_steps):
            ci, ri, dist = self._ls1_draw()
            H_pred = self._render(ci, ri)
            H_tgt = self.H[ci, ri]
            terms = _losses(H_pred, H_tgt, self.loss_lo_bin, w_spec=self.spec_w, amp_eps=self.amp_eps)
            loss = (w_r * terms["L_spec_real"] + w_i * terms["L_spec_imag"]
                    + w_a * terms["L_amp"] + w_p * terms["L_phase"])
            l1 = ls1_loss(H_pred, H_tgt, L["lo"], L["hi"], L["m_b"], L["S"])
            contrib = cfg.ls1_weight * l1
            loss = loss + contrib
            terms["L_ls1"] = l1.detach()
            terms["ls1_frac"] = (contrib / loss.clamp(min=1e-12)).detach()
            terms["ls1_pair_dist_m"] = dist.mean()
            (loss / cfg.grad_accum_steps).backward()
            for k, v in terms.items():
                agg[k] = agg.get(k, 0.0) + float(v) / cfg.grad_accum_steps
            agg["loss"] = agg.get("loss", 0.0) + float(loss) / cfg.grad_accum_steps
        agg["ls1_pairs_per_step"] = float(cfg.batch_size // 2 * cfg.grad_accum_steps)
        self._apply_grads()
        return agg

    def _render_room(self, c: int, ri: torch.Tensor, jitter: Optional[torch.Tensor]) -> torch.Tensor:
        """One room's rows in one renderer call with an explicit jitter (LS2's two passes)."""
        L_, W_ = self.geom_dims[int(self.geom_id[c])]
        room_min = torch.zeros(2, device=self.device)
        room_max = torch.tensor([L_, W_], device=self.device, dtype=torch.float32)
        src_per = getattr(self, "src_per", None)
        tx = (self.src if src_per is None else src_per[c]).unsqueeze(0).expand(ri.numel(), -1)
        z = self.cond[c].unsqueeze(0).expand(ri.numel(), -1)
        H = self.renderer(self.model, self.rx[c, ri], tx, room_min, room_max, z_s=z, jitter=jitter)
        return H[:, self.band[0]:self.band[1]]

    def _step_ls2(self, it: int) -> dict:
        """One room x ls2_rx_per_step receivers, EXACT two-pass: pass 1 renders every chunk with one fixed jitter
        (grad-enabled, detached -- tcnn's no-grad inference kernels are not guaranteed bit-identical), giving LS2
        and dLS2/dP; pass 2 re-renders each chunk with the same jitter and back-propagates the PB3 loss (as a mean
        over all the room's rows) plus the injected lambda * dLS2/dP. Pass 2 must reproduce pass 1."""
        cfg = self.cfg
        L = self._ls
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        c = int(torch.randint(0, len(self.configs), (1,), device=self.device))
        ri = self.train_rx_idx[torch.randperm(self.train_rx_idx.numel(), device=self.device)[:cfg.ls2_rx_per_step]]
        jitter = torch.empty(1, device=self.device).uniform_(0.0, 2.0 * math.pi / self.renderer.n_azi)
        ch = max(1, int(cfg.max_rows_per_render))
        chunks = [ri[i:i + ch] for i in range(0, ri.numel(), ch)]
        P1 = []
        for r in chunks:
            P1.append(self._render_room(c, r, jitter).detach())
        P1 = torch.cat(P1)
        T = self.H[c, ri]
        Pl = P1.clone().requires_grad_(True)
        l2 = ls2_loss(Pl, T, L["bands"], cfg.ls2_floor_db)
        G, = torch.autograd.grad(l2, Pl)
        if not bool(torch.isfinite(G.real).all() and torch.isfinite(G.imag).all()):
            raise FloatingPointError("LS2 gradient is not finite at iteration {}".format(it))
        self._ls2_steps += 1
        n = ri.numel()
        w_r, w_i, w_a, w_p = cfg.weights
        agg = {}
        off = 0
        for r in chunks:
            sl = slice(off, off + r.numel())
            off += r.numel()
            H2 = self._render_room(c, r, jitter)
            if self._ls2_steps <= cfg.ls2_det_check_steps:
                if not torch.equal(H2.detach(), P1[sl]):
                    raise RuntimeError("LS2 pass 2 does not reproduce pass 1 (max |dH| {:.3e}) at iteration {}".format(
                        float((H2.detach() - P1[sl]).abs().max()), it))
            else:
                rel = float((H2.detach() - P1[sl]).abs().max() / P1[sl].abs().max().clamp(min=1e-12))
                if rel > 1e-4:
                    raise RuntimeError("LS2 pass 2 drifted from pass 1 by {:.3e} at iteration {}".format(rel, it))
            terms = _losses(H2, T[sl], self.loss_lo_bin, w_spec=self.spec_w, amp_eps=self.amp_eps)
            loss = (w_r * terms["L_spec_real"] + w_i * terms["L_spec_imag"]
                    + w_a * terms["L_amp"] + w_p * terms["L_phase"])
            frac = r.numel() / n
            torch.autograd.backward([loss * frac, H2], [torch.ones_like(loss), cfg.ls2_weight * G[sl]])
            for k, v in terms.items():
                agg[k] = agg.get(k, 0.0) + float(v) * frac
            agg["loss"] = agg.get("loss", 0.0) + float(loss) * frac
        agg["L_ls2"] = float(l2)
        agg["loss"] = agg["loss"] + cfg.ls2_weight * float(l2)
        agg["ls2_rx_per_room"] = float(n)
        self._apply_grads()
        return agg

    # ---------------------------------------------------------------- validation
    @torch.no_grad()
    def validate(self, it: int) -> dict:
        self.model.eval()
        # D160: the renderer too. In training mode it jitters the ray fan on every call, so the
        # val LSD early stopping reads moved with no change in the weights (D150(d)), and it did
        # not match `load_model`-based evaluation, which forces eval mode. Restored after, so
        # training keeps its jitter.
        was_training = self.renderer.training
        self.renderer.eval()
        try:
            return self._validate(it)
        finally:
            self.renderer.train(was_training)

    def _validate(self, it: int) -> dict:
        preds, tgts = [], []
        chunk = 64
        for cid in self.val_cfg_ids:
            ci = cid.repeat(self.val_rx_idx.numel())
            for s in range(0, ci.numel(), chunk):
                sl = slice(s, s + chunk)
                preds.append(self._render(ci[sl], self.val_rx_idx[sl]))
                tgts.append(self.H[ci[sl], self.val_rx_idx[sl]])
        P, T = torch.cat(preds), torch.cat(tgts)
        terms = _losses(P, T, self.loss_lo_bin)
        # `lsd_db` is the MODEL-SELECTION metric: early stopping reads it and checkpoints are
        # judged by it, so it must be measured on the band the model is actually trained on.
        # With a low cut active, the full-band value is dominated by bins the run deliberately
        # never fits -- and in this corpus those bins sit ~37 dB above the strongest room mode,
        # so the full-band number moves by >1.5 dB for reasons that have nothing to do with
        # learning. Measured on the FDTD corpus: masking made full-band lsd_db 1.65 dB WORSE
        # while L_amp and L_phase both improved. Selecting on that would have early-stopped a
        # run that was still improving everywhere it was being asked to.
        #
        # The full-band value is still recorded, just not used to steer anything.
        lo = self.loss_lo_bin
        rec = {"phase": "val", "iter": it,
               "lsd_db": _lsd_db(P[..., lo:], T[..., lo:]) if lo else _lsd_db(P, T),
               **{k: float(v) for k, v in terms.items()}}
        if lo:
            rec["lsd_db_full_band"] = _lsd_db(P, T)
            rec["loss_lo_bin"] = int(lo)
        self.scalars.append(rec)
        return rec

    # ---------------------------------------------------------------- ckpt / resume
    def save_ckpt(self, it: int):
        path = self.out / f"ckpt_iter{it:07d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        meta_cfg = asdict(self.cfg)
        meta_cfg["_manifest_sha"] = self.manifest_sha
        torch.save({"iter": it, "model": self.model.state_dict(),
                    "optimizer": self.opt.state_dict(),
                    "scheduler": self.sched.state_dict(),
                    "cfg": meta_cfg}, tmp)
        tmp.replace(path)
        (self.out / "scalars.json").write_text(json.dumps(self.scalars, indent=1))
        keep_iters = {int(x) for x in str(self.cfg.ckpt_keep_iters).split(",") if x.strip()}
        _prune_ckpts(self.out, self.cfg.ckpt_keep_last, self.cfg.ckpt_keep_every, keep_iters)
        if self.cfg.ckpt_thin_old:
            self._thin_older_than(path)

    def _thin_older_than(self, newest: Path):
        """Strip optimizer/scheduler state from every checkpoint but `newest`.

        Model weights, cfg and iteration are kept, so every retained checkpoint stays fully
        evaluable through `load_model`; only the newest -- the one a resume reads -- keeps the
        Adam moments. Written to a temp file and renamed, so a crash mid-rewrite leaves the
        original intact rather than a truncated file.

        `_thinned` remembers what is already done, so after the first call -- which scans
        everything, covering a resume -- each save touches only the one checkpoint it just
        superseded, instead of re-reading up to ~29 files (~6.6 GB) every 2K iterations."""
        if not hasattr(self, "_thinned"):
            self._thinned = set()
        for c in sorted(self.out.glob("ckpt_iter*.pt")):
            if c == newest or c in self._thinned:
                continue
            try:
                st = torch.load(c, map_location="cpu")
            except Exception:
                continue
            if st.get("thinned"):
                self._thinned.add(c)
                continue
            st.pop("optimizer", None)
            st.pop("scheduler", None)
            st["thinned"] = True
            tmp = c.with_suffix(".pt.thin")
            torch.save(st, tmp)
            tmp.replace(c)
            self._thinned.add(c)

    def _maybe_resume(self) -> int:
        ckpts = sorted(self.out.glob("ckpt_iter*.pt"), reverse=True)
        for c in ckpts:
            try:
                # P4-8: load to CPU. load_state_dict copies the values onto the model's / optimiser's device, so
                # the result is identical; loading to the GPU left every inspected checkpoint (thinned ones too)
                # in torch's cache, which tinycudann's own allocator cannot use -- requeued jobs (M3-fam, LS2A,
                # LS2B) died with cuMemCreate OUT_OF_MEMORY in their first step after a resume.
                st = torch.load(c, map_location="cpu")
            except Exception:
                continue
            if st.get("thinned") or "optimizer" not in st:
                # A thinned checkpoint has no Adam moments; resuming from it would restart the
                # optimiser cold mid-run. Only the newest is ever full, so skip to it.
                continue
            prev = st.get("cfg", {})
            # world_scale and token_pool change what the SAME-SHAPED weights MEAN, so a
            # cross-load between two arms would load silently and train on the wrong
            # semantics. base_resolution at least fails loudly with a tensor-size mismatch;
            # these two do not.
            # sigma_aux_weight joins the list for the same reason: it changes what the weights
            # MEAN (a model trained under the auxiliary term has a deliberately reshaped sigma
            # field), and resuming across it would silently mix two objectives.
            # P4-5: the bin weights, the curriculum and the c2f schedule each define a different
            # objective or encoder; a cross-resume would silently splice two arms together.
            for k in ("cond_source", "cond_dim", "n_pts_per_ray", "n_azi", "n_iters",
                      "world_scale", "token_pool", "base_resolution", "sigma_aux_weight",
                      "loss_spec_bin_weights", "loss_curriculum", "c2f_levels",
                      "dc_head_bins", "signal_heads", "signal_band_only", "signal_n_neurons",
                      "signal_band_edges_hz", "signal_xfade_hz", "loss_amp_floor_frac",
                      "loss_amp_floor_ref", "aug_d4_p", "aug_d4_elems", "aug_recip_p",
                      "per_room_source", "modal_prior_dir", "modal_k"):
                if k in prev and prev[k] != getattr(self.cfg, k):
                    raise RuntimeError(
                        f"refusing to resume {c.name}: {k}={prev[k]!r} in the checkpoint but "
                        f"{getattr(self.cfg, k)!r} in this config. Point --output_dir at a "
                        f"fresh directory for a different arm.")
            # P4-8: a key absent from an older checkpoint means its DEFAULT, so a GX checkpoint cannot be
            # resumed silently under ls_mode / mb_grids.
            for k in P48_KEYS:
                was = prev.get(k, P32TrainCfg.__dataclass_fields__[k].default)
                if was != getattr(self.cfg, k):
                    raise RuntimeError(f"refusing to resume {c.name}: {k}={was!r} in the checkpoint but "
                                       f"{getattr(self.cfg, k)!r} in this config.")
            if prev.get("configs_manifest") and self.manifest_sha and \
                    prev.get("_manifest_sha", self.manifest_sha) != self.manifest_sha:
                raise RuntimeError(f"refusing to resume {c.name}: manifest sha mismatch")
            self.model.load_state_dict(st["model"])
            self.opt.load_state_dict(st["optimizer"])
            self.sched.load_state_dict(st["scheduler"])
            sp = self.out / "scalars.json"
            if sp.exists():
                self.scalars = [r for r in json.loads(sp.read_text())
                                if r.get("iter", 0) <= st["iter"]]
            print(f"[resume] from {c.name} at iter {st['iter']}")
            it = int(st["iter"])
            del st
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return it
        return 0

    def _early_stop(self, it: int) -> bool:
        cfg = self.cfg
        if it < cfg.early_stop_warmup:
            return False
        boundary = it - cfg.early_stop_patience
        if boundary <= 0:
            return False
        vals = [(r["iter"], r["lsd_db"]) for r in self.scalars if r.get("phase") == "val"]
        before = [v for i, v in vals if i <= boundary]
        window = [v for i, v in vals if i > boundary]
        if not before or not window:
            return False
        imp = (min(before) - min(window)) / max(min(before), 1e-12)
        return imp < cfg.early_stop_min_rel_improvement

    # ---------------------------------------------------------------- loop
    def train(self):
        cfg = self.cfg
        (self.out / "train_meta.json").write_text(json.dumps({
            "cfg": asdict(cfg), "n_configs": len(self.configs),
            "manifest_sha": self.manifest_sha,
            "val_config_labels": [self.configs[int(i)].label for i in self.val_cfg_ids],
            "geometries": self.geom_dims, "band": list(self.band),
            "val_rx": list(self.val_rx),
            "config_labels": [c.label for c in self.configs],
            # P4-5: the weight FILE can change under the same path; its sha pins what ran.
            "spec_bin_weights": self.spec_w_info,
            "amp_floor": self.amp_floor_info,
            "curriculum_bins": self.curriculum, "c2f": self.c2f,
            **({"augment": self._aug["info"]} if getattr(self, "_aug", None) is not None else {}),
            **({"modal_prior": {"dir": cfg.modal_prior_dir, "modal_k": int(cfg.modal_k)}}
               if cfg.modal_prior_dir else {}),
            **({"per_room_source": {"n_distinct": int(len({tuple(x) for x in
                                                           self.src_per.tolist()}))}}
               if getattr(self, "src_per", None) is not None else {}),
            **({"multiband": {"grids": mb_grid_table(multiband_kwargs(cfg)["mb_grid_cfgs"], cfg.world_scale or 2.0),
                              "params": self.model.signal_param_counts()}} if cfg.mb_grids else {}),
            **({"spatial_loss": self._ls["info"]} if getattr(self, "_ls", None) is not None else {}),
        }, indent=1, default=str))
        t0 = time.time()
        for it in range(self.start_iter + 1, cfg.n_iters + 1):
            m = self._step_fn(it)
            if it % cfg.log_every == 0:
                self.scalars.append({"phase": "train", "iter": it, **m})
                print(f"[{it:6d}/{cfg.n_iters}] loss={m['loss']:.4f} "
                      f"amp={m['L_amp']:.4f} phase={m['L_phase']:.4f} "
                      f"lr={self.sched.get_last_lr()[0]:.2e} "
                      f"({(time.time()-t0)/max(it-self.start_iter,1):.3f}s/it)", flush=True)
            if it % cfg.val_every == 0:
                v = self.validate(it)
                print(f"  [val {it}] band-limited LSD = {v['lsd_db']:.4f} dB", flush=True)
                if self._early_stop(it):
                    print(f"[early-stop] at iter {it}")
                    self.save_ckpt(it)
                    break
            if it % cfg.ckpt_every == 0:
                self.save_ckpt(it)
        else:
            self.save_ckpt(cfg.n_iters)
        (self.out / "scalars.json").write_text(json.dumps(self.scalars, indent=1))
        vals = [r for r in self.scalars if r.get("phase") == "val"]
        if vals:
            print(f"[done] final band-limited val LSD = {vals[-1]['lsd_db']:.4f} dB "
                  f"(best {min(v['lsd_db'] for v in vals):.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--allow_unknown_keys", action="store_true",
                    help="P4-8: unknown config keys are FATAL by default (a misspelt option would silently train "
                         "the baseline); this restores the old warning")
    a = ap.parse_args()
    d = yaml.safe_load(open(a.config))
    known = {f for f in P32TrainCfg.__dataclass_fields__}
    cfg = P32TrainCfg(**{k: v for k, v in d.items() if k in known})
    unknown = [k for k in d if k not in known]
    if unknown:
        if not a.allow_unknown_keys:
            raise SystemExit(f"unknown config keys {unknown} (pass --allow_unknown_keys to ignore them)")
        print(f"[warn] ignoring unknown config keys: {unknown}")
    out = a.output_dir or f"outputs/p3_2/{cfg.run_id}"
    print(json.dumps(asdict(cfg), indent=1, default=str))
    P32Trainer(cfg, out).train()


if __name__ == "__main__":
    main()
