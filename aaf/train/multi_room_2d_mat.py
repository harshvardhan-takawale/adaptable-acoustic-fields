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
    val_every: int = 2_000
    val_max_configs: int = 88
    ckpt_every: int = 2_000
    ckpt_keep_last: int = 10
    log_every: int = 200
    early_stop_warmup: int = 10_000
    early_stop_patience: int = 10_000
    early_stop_min_rel_improvement: float = 0.003
    seed: int = 0


def _losses(H_pred, H_target, lo_bin: int = 0):
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
    """
    if lo_bin:
        H_pred = H_pred[..., lo_bin:]
        H_target = H_target[..., lo_bin:]
    eps = 1e-6
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
        H_list, rx_list, geom_id, cfg_id, conds = [], [], [], [], []
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
        self.H = torch.tensor(np.stack(H_list), device=self.device)          # [C,64,B]
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
        ).to(self.device)
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
        self.scalars: List[dict] = []
        self.start_iter = self._maybe_resume()

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
                tx = self.src.unsqueeze(0).expand(sub.numel(), -1)
                z = self.cond[cfg_ids[sub]]
                H = self.renderer(self.model, rx, tx, room_min, room_max, z_s=z)
                out[sub] = H[:, self.band[0]:self.band[1]]
        return out

    # ---------------------------------------------------------------- train step
    def _step(self, it: int) -> dict:
        cfg = self.cfg
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        n_cfg = max(1, cfg.batch_size // cfg.rx_per_config)
        agg = {}
        for _ in range(cfg.grad_accum_steps):
            ci = torch.randint(0, len(self.configs), (n_cfg,), device=self.device)
            ci = ci.repeat_interleave(cfg.rx_per_config)
            ri = self.train_rx_idx[
                torch.randint(0, self.train_rx_idx.numel(), (ci.numel(),), device=self.device)]
            H_pred = self._render(ci, ri)
            H_tgt = self.H[ci, ri]
            terms = _losses(H_pred, H_tgt, self.loss_lo_bin)
            w_r, w_i, w_a, w_p = cfg.weights
            loss = (w_r * terms["L_spec_real"] + w_i * terms["L_spec_imag"]
                    + w_a * terms["L_amp"] + w_p * terms["L_phase"])
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

    # ---------------------------------------------------------------- validation
    @torch.no_grad()
    def validate(self, it: int) -> dict:
        self.model.eval()
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
        ckpts = sorted(self.out.glob("ckpt_iter*.pt"))
        for old in ckpts[:-max(1, self.cfg.ckpt_keep_last)]:
            old.unlink()

    def _maybe_resume(self) -> int:
        ckpts = sorted(self.out.glob("ckpt_iter*.pt"), reverse=True)
        for c in ckpts:
            try:
                st = torch.load(c, map_location=self.device)
            except Exception:
                continue
            prev = st.get("cfg", {})
            # world_scale and token_pool change what the SAME-SHAPED weights MEAN, so a
            # cross-load between two arms would load silently and train on the wrong
            # semantics. base_resolution at least fails loudly with a tensor-size mismatch;
            # these two do not.
            for k in ("cond_source", "cond_dim", "n_pts_per_ray", "n_azi", "n_iters",
                      "world_scale", "token_pool", "base_resolution"):
                if k in prev and prev[k] != getattr(self.cfg, k):
                    raise RuntimeError(
                        f"refusing to resume {c.name}: {k}={prev[k]!r} in the checkpoint but "
                        f"{getattr(self.cfg, k)!r} in this config. Point --output_dir at a "
                        f"fresh directory for a different arm.")
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
            return int(st["iter"])
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
        }, indent=1, default=str))
        t0 = time.time()
        for it in range(self.start_iter + 1, cfg.n_iters + 1):
            m = self._step(it)
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
    a = ap.parse_args()
    d = yaml.safe_load(open(a.config))
    known = {f for f in P32TrainCfg.__dataclass_fields__}
    cfg = P32TrainCfg(**{k: v for k, v in d.items() if k in known})
    unknown = [k for k in d if k not in known]
    if unknown:
        print(f"[warn] ignoring unknown config keys: {unknown}")
    out = a.output_dir or f"outputs/p3_2/{cfg.run_id}"
    print(json.dumps(asdict(cfg), indent=1, default=str))
    P32Trainer(cfg, out).train()


if __name__ == "__main__":
    main()
