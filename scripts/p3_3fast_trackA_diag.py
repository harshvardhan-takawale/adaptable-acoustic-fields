"""P3-3-FAST Track A diagnostic: does the per-segment conditioning localize the edit?

The Track A run (``outputs/p3_3fast/p3_3fast_trackA``) plateaued near 4.5-4.7 dB
in-distribution val LSD where P3-2b reached ~1.0 dB, so the ABSOLUTE fit is poor. The
chunk's question is nonetheless RELATIVE and is answerable on a poorly-fit model: given
16 per-segment absorptions, does the field the model renders depend on WHICH segment was
edited, or only on how much absorption was added in total?

Three diagnostics on the test split (10 geometries, 12 configs each):

1. ``segment_discrimination`` -- the decisive one. Per geometry, the four
   ``t_single_segment`` configs put alpha = 0.70 on segment 3 of west / east / south /
   north and are otherwise identical, so the four fields differ ONLY in edit position.
   Take the spread across those four (per receiver + bin std, then averaged) for the
   predictions and for the ground truth separately. ``pred_spread / gt_spread`` near zero
   with a clearly non-zero ``gt_spread`` means the conditioning cannot localize -- the
   FiLM generator modulates the whole field globally, so a fixed total absorption
   produces a fixed response wherever it is placed. Near one means it localizes despite
   the poor absolute fit.

2. ``holdout_position`` -- segment ``east_3`` (flat index 6) is in NO training config.
   Compare configs that touch it against matched configs on seen segments, paired within
   geometry against that geometry's own baseline so the per-geometry fit level divides
   out. Note ``t_uniform_wall`` on east ALSO covers east_3 and is grouped as held-out.

3. ``window`` -- the alpha = 0.95 configs. GT shows a large in-band energy drop; the
   question is whether the model reproduces its magnitude, and whether it does so equally
   at the seen (``west_2``) and held-out (``east_3``) position.

Every estimator that already exists is imported, never reimplemented: ``load_model`` /
``load_gt`` / ``band_limit`` / ``find_checkpoint`` from :mod:`aaf.eval.p3_2_eval`,
``render_config_arm`` from :mod:`aaf.eval.p3_2b_eval` (it dispatches the conditioning
encoder on ``cond_source``, which must be ``m_segment`` / 144-d here), and ``_lsd_db``
from :mod:`aaf.eval.band_limited`. ``load_model`` puts BOTH model and renderer in
``eval()``; the renderer flag is load-bearing, not hygiene (D49 C3: ``FreqRenderer2D``
jitters ray azimuths while ``self.training``, which would inject noise into exactly the
across-config spread this script measures).

Track A ground truth is stored already truncated to the 601 supervised bins (0-300 Hz)
while the renderer emits 4097, so both sides go through ``band_limit`` and are then cut
to ``hi_idx``; the truncation is applied identically to prediction and GT.

Usage
-----
    python scripts/p3_3fast_trackA_diag.py                     # needs a GPU (tinycudann)
    python scripts/p3_3fast_trackA_diag.py --gt-only           # GT denominators, CPU-only
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aaf.data.seg_configs import SEGMENT_NAMES, SegConfig, configs_from_rows, segment_index
from aaf.eval.band_limited import _lsd_db, band_indices

BAND_HI_HZ = 300.0
EPS = 1e-8
"""Usability floor on |H|. Matches the eps inside ``_lsd_db`` so a cell that would be
clamped by the log is excluded from every statistic rather than silently floored."""

HOLDOUT_INDEX = segment_index("east", 3)
WALLS = ("west", "east", "south", "north")


# --------------------------------------------------------------------------- small stats
def _f(x) -> float:
    x = float(x)
    return x if np.isfinite(x) else float("nan")


def _mean_sd(xs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "sem": float("nan"), "n": 0}
    sd = float(a.std(ddof=1)) if a.size > 1 else float("nan")
    return {"mean": float(a.mean()), "sd": sd,
            "sem": (sd / float(np.sqrt(a.size))) if a.size > 1 else float("nan"),
            "n": int(a.size)}


def usable_mask(stack: np.ndarray) -> np.ndarray:
    """``[n_rx, n_bin]`` bool: cells finite AND above the log floor in EVERY member.

    ``stack`` is ``[K, n_rx, n_bin]``. Everything downstream is conditioned on this mask,
    and ``frac`` is reported next to every number it gates.
    """
    a = np.asarray(stack)
    return np.all(np.isfinite(a) & (np.abs(a) > EPS), axis=0)


def spread_db(stack: np.ndarray, mask: np.ndarray) -> float:
    """Mean over usable cells of the across-config std of the magnitude in dB.

    dB is the primary unit because it is what LSD measures and because it is invariant to
    the model's overall amplitude error, which is large here and would otherwise dominate
    a linear-magnitude ratio.
    """
    d = 20.0 * np.log10(np.abs(np.asarray(stack)) + EPS)
    return _f(np.mean(d.std(axis=0, ddof=1)[mask]))


def spread_mag_rel(stack: np.ndarray, mask: np.ndarray) -> float:
    """Scale-free cross-check: mean across-config std of |H| over mean |H|."""
    a = np.abs(np.asarray(stack))
    num = float(np.mean(a.std(axis=0, ddof=1)[mask]))
    den = float(np.mean(a.mean(axis=0)[mask]))
    return _f(num / den) if den > 0 else float("nan")


def pairwise_lsd(stack: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Mean / min / max band-limited LSD over the unordered pairs of ``stack``."""
    vals = [_lsd_db(stack[i][mask], stack[j][mask])
            for i, j in combinations(range(stack.shape[0]), 2)]
    return {"mean": _f(np.mean(vals)), "min": _f(np.min(vals)), "max": _f(np.max(vals)),
            "n_pairs": len(vals)}


def band_energy_db(H: np.ndarray, H_ref: np.ndarray, mask: np.ndarray) -> float:
    """In-band total energy of ``H`` relative to ``H_ref``, dB, on shared usable cells."""
    e = float(np.sum(np.abs(H[mask]) ** 2))
    e0 = float(np.sum(np.abs(H_ref[mask]) ** 2))
    if e <= 0.0 or e0 <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(e / e0))


# --------------------------------------------------------------------------- config layer
def _wall_of(c: SegConfig) -> Optional[str]:
    walls = sorted({e.split("_")[0] for e in c.edited})
    return walls[0] if len(walls) == 1 else None


def touches_holdout(c: SegConfig) -> bool:
    return SEGMENT_NAMES[HOLDOUT_INDEX] in c.edited


def group_by_geometry(cfgs: Sequence[SegConfig]) -> Dict[int, Dict[str, SegConfig]]:
    """``geom_id -> {slot: SegConfig}``; slots name the role, not the file."""
    out = {}  # type: Dict[int, Dict[str, SegConfig]]
    for c in cfgs:
        slot = c.kind
        if c.kind in ("t_single_segment", "t_uniform_wall"):
            w = _wall_of(c)
            if w is None:
                raise ValueError("multi-wall config in {}: {}".format(c.kind, c.edited))
            slot = "{}:{}".format(c.kind, w)
        d = out.setdefault(c.geom_id, {})
        if slot in d:
            raise ValueError("duplicate slot {} for geom {}".format(slot, c.geom_id))
        d[slot] = c
    return out


# --------------------------------------------------------------------------- field access
class Fields:
    """Lazy per-config band-limited fields: GT always, prediction when a model is loaded."""

    def __init__(self, data_dir: Path, hi_idx: int, model=None, renderer=None,
                 cond_source: str = "m_segment", device=None, rx_chunk: int = 8):
        from aaf.eval.p3_2_eval import band_limit, load_gt

        self._band_limit = band_limit
        self._load_gt = load_gt
        self.data_dir = Path(data_dir)
        self.hi = int(hi_idx)
        self.model, self.renderer = model, renderer
        self.cond_source, self.device, self.rx_chunk = cond_source, device, rx_chunk
        self._gt = {}    # type: Dict[str, np.ndarray]
        self._pred = {}  # type: Dict[str, np.ndarray]
        self._geo = {}   # type: Dict[str, Tuple[np.ndarray, np.ndarray]]
        self.render_seconds = 0.0
        self.n_rendered = 0

    def gt(self, c: SegConfig) -> np.ndarray:
        key = c.filename
        if key not in self._gt:
            H, rx, src, _ = self._load_gt(self.data_dir / key)
            if H.shape[-1] < self.hi:
                raise ValueError("GT {} has {} bins, need {}".format(
                    key, H.shape[-1], self.hi))
            self._gt[key] = self._band_limit(H, self.hi)[:, :self.hi]
            self._geo[key] = (rx, src)
        return self._gt[key]

    def pred(self, c: SegConfig) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("no model loaded; --gt-only mode cannot render")
        key = c.filename
        if key not in self._pred:
            from aaf.eval.p3_2b_eval import render_config_arm

            self.gt(c)                                    # populates rx / src
            rx, src = self._geo[key]
            t0 = time.time()
            H = render_config_arm(self.model, self.renderer, self.cond_source, c.L, c.W,
                                  c.alphas, rx, src, self.device, rx_chunk=self.rx_chunk)
            self.render_seconds += time.time() - t0
            self.n_rendered += 1
            self._pred[key] = self._band_limit(H, self.hi)[:, :self.hi]
        return self._pred[key]

    def drop_geometry(self) -> None:
        self._gt.clear()
        self._pred.clear()
        self._geo.clear()


# --------------------------------------------------------------------------- diagnostic 1
def diag_segment_discrimination(fields: Fields, geoms: Dict[int, Dict[str, SegConfig]],
                                with_pred: bool) -> dict:
    """Spread across the four single-segment edits, prediction vs ground truth."""
    per_geom = []
    for gid in sorted(geoms):
        slots = geoms[gid]
        keys = ["t_single_segment:{}".format(w) for w in WALLS]
        if any(k not in slots for k in keys):
            continue
        cfgs = [slots[k] for k in keys]
        gt = np.stack([fields.gt(c) for c in cfgs], axis=0)
        m_gt = usable_mask(gt)
        rec = {
            "geom_id": gid, "L": cfgs[0].L, "W": cfgs[0].W,
            "edited": [c.edited[0] for c in cfgs],
            "alpha": float(max(cfgs[0].alphas)),
            "gt_spread_db": spread_db(gt, m_gt),
            "gt_spread_mag_rel": spread_mag_rel(gt, m_gt),
            "gt_pairwise_lsd_db": pairwise_lsd(gt, m_gt),
            "frac_gt": _f(m_gt.mean()),
        }
        # How big is the EDIT itself, in the same units? The positional signal (the spread
        # above) is only a fraction of it, and the model's own fit error has to be below
        # the positional signal for the ratio to mean anything. Reported so the resolvability
        # of the whole test is on the record next to the answer.
        if "baseline" in slots:
            g_b = fields.gt(slots["baseline"])
            mb = m_gt & usable_mask(np.stack([g_b]))
            rec["gt_edit_lsd_vs_baseline_db"] = _f(np.mean(
                [_lsd_db(gt[i][mb], g_b[mb]) for i in range(gt.shape[0])]))
            rec["gt_positional_fraction"] = _f(
                rec["gt_spread_db"] / rec["gt_edit_lsd_vs_baseline_db"]) \
                if rec["gt_edit_lsd_vs_baseline_db"] > 0 else float("nan")
        if with_pred:
            pr = np.stack([fields.pred(c) for c in cfgs], axis=0)
            m = m_gt & usable_mask(pr)
            rec.update({
                "pred_spread_db": spread_db(pr, m),
                "pred_spread_mag_rel": spread_mag_rel(pr, m),
                "pred_pairwise_lsd_db": pairwise_lsd(pr, m),
                "gt_spread_db_shared_mask": spread_db(gt, m),
                "gt_spread_mag_rel_shared_mask": spread_mag_rel(gt, m),
                "gt_pairwise_lsd_db_shared_mask": pairwise_lsd(gt, m),
                "frac": _f(m.mean()),
                "pred_vs_gt_lsd_db": _f(np.mean(
                    [_lsd_db(pr[i][m], gt[i][m]) for i in range(pr.shape[0])])),
            })
            rec["spread_ratio_db"] = _f(
                rec["pred_spread_db"] / rec["gt_spread_db_shared_mask"])
            rec["spread_ratio_mag_rel"] = _f(
                rec["pred_spread_mag_rel"] / rec["gt_spread_mag_rel_shared_mask"])
            rec["pairwise_lsd_ratio"] = _f(
                rec["pred_pairwise_lsd_db"]["mean"]
                / rec["gt_pairwise_lsd_db_shared_mask"]["mean"])
        per_geom.append(rec)
        fields.drop_geometry()

    agg = {"n_geometries": len(per_geom)}
    for k in ("gt_spread_db", "gt_spread_mag_rel", "frac_gt",
              "gt_edit_lsd_vs_baseline_db", "gt_positional_fraction"):
        agg[k] = _mean_sd([r.get(k, float("nan")) for r in per_geom])
    agg["gt_pairwise_lsd_db"] = _mean_sd([r["gt_pairwise_lsd_db"]["mean"]
                                          for r in per_geom])
    if with_pred:
        for k in ("pred_spread_db", "pred_spread_mag_rel", "gt_spread_db_shared_mask",
                  "gt_spread_mag_rel_shared_mask", "spread_ratio_db",
                  "spread_ratio_mag_rel", "pairwise_lsd_ratio", "frac",
                  "pred_vs_gt_lsd_db"):
            agg[k] = _mean_sd([r[k] for r in per_geom])
        agg["pred_pairwise_lsd_db"] = _mean_sd(
            [r["pred_pairwise_lsd_db"]["mean"] for r in per_geom])
        agg["gt_pairwise_lsd_db_shared_mask"] = _mean_sd(
            [r["gt_pairwise_lsd_db_shared_mask"]["mean"] for r in per_geom])
        # Pooled ratio: sum of numerators over sum of denominators, so one geometry with a
        # tiny denominator cannot dominate a mean of per-geometry ratios.
        num = float(np.nansum([r["pred_spread_db"] for r in per_geom]))
        den = float(np.nansum([r["gt_spread_db_shared_mask"] for r in per_geom]))
        agg["spread_ratio_db_pooled"] = _f(num / den) if den > 0 else float("nan")
        num = float(np.nansum([r["pred_pairwise_lsd_db"]["mean"] for r in per_geom]))
        den = float(np.nansum([r["gt_pairwise_lsd_db_shared_mask"]["mean"]
                               for r in per_geom]))
        agg["pairwise_lsd_ratio_pooled"] = _f(num / den) if den > 0 else float("nan")
        agg["verdict"] = _discrimination_verdict(agg)
    return {"per_geometry": per_geom, "aggregate": agg}


def _discrimination_verdict(agg: dict) -> dict:
    """Threshold fixed BEFORE any number existed: ratio < 0.15 = cannot localize."""
    r = agg["spread_ratio_db_pooled"]
    gt = agg["gt_spread_db"]["mean"]
    if not np.isfinite(r) or not np.isfinite(gt):
        return {"label": "undetermined", "reason": "non-finite spread"}
    if gt < 0.5:
        return {"label": "undetermined",
                "reason": "gt_spread {:.3f} dB too small to resolve a ratio".format(gt)}
    if r < 0.15:
        label = "does_not_localize"
    elif r < 0.6:
        label = "partial"
    else:
        label = "localizes"
    return {"label": label, "spread_ratio_db_pooled": r, "gt_spread_db": gt,
            "threshold_lt": 0.15}


# --------------------------------------------------------------------------- diagnostic 2
GROUPS = {
    "single_seen": ["t_single_segment:{}".format(w) for w in ("west", "south", "north")],
    "single_holdout": ["t_single_segment:east"],
    "uniform_seen": ["t_uniform_wall:{}".format(w) for w in ("west", "south", "north")],
    "uniform_holdout": ["t_uniform_wall:east"],
    "window_seen": ["t_window_seen"],
    "window_holdout": ["t_window_holdout"],
    "patch3_holdout": ["t_patch3_holdout"],
}

PAIRS = (("single", "single_holdout", "single_seen"),
         ("uniform", "uniform_holdout", "uniform_seen"),
         ("window", "window_holdout", "window_seen"))


def diag_holdout_position(fields: Fields, geoms: Dict[int, Dict[str, SegConfig]],
                          with_pred: bool) -> dict:
    """Held-out segment position (east_3) vs matched seen positions, paired by geometry."""
    if not with_pred:
        return {"skipped": "requires model predictions (no GT-only form of this test)"}
    rows = []       # type: List[dict]
    base_lsd = {}   # type: Dict[int, float]
    for gid in sorted(geoms):
        slots = geoms[gid]
        if "baseline" not in slots:
            continue
        base = slots["baseline"]
        g_b, p_b = fields.gt(base), fields.pred(base)
        m_b = usable_mask(np.stack([g_b, p_b]))
        base_lsd[gid] = _f(_lsd_db(p_b[m_b], g_b[m_b]))
        for group, keys in GROUPS.items():
            for k in keys:
                if k not in slots:
                    continue
                c = slots[k]
                g, p = fields.gt(c), fields.pred(c)
                m = usable_mask(np.stack([g, p]))
                lsd = _f(_lsd_db(p[m], g[m]))
                rows.append({
                    "geom_id": gid, "slot": k, "group": group, "kind": c.kind,
                    "edited": list(c.edited), "touches_holdout": touches_holdout(c),
                    "lsd_db": lsd, "lsd_baseline_db": base_lsd[gid],
                    "delta_vs_baseline_db": _f(lsd - base_lsd[gid]),
                    "frac": _f(m.mean()),
                })
        fields.drop_geometry()

    by_group = {}
    for group in GROUPS:
        sel = [r for r in rows if r["group"] == group]
        by_group[group] = {
            "n": len(sel),
            "touches_holdout": bool(sel) and all(r["touches_holdout"] for r in sel),
            "lsd_db": _mean_sd([r["lsd_db"] for r in sel]),
            "delta_vs_baseline_db": _mean_sd([r["delta_vs_baseline_db"] for r in sel]),
            "frac": _mean_sd([r["frac"] for r in sel]),
        }
    by_group["baseline"] = {
        "n": len(base_lsd), "touches_holdout": False,
        "lsd_db": _mean_sd([base_lsd[g] for g in sorted(base_lsd)]),
        "delta_vs_baseline_db": {"mean": 0.0, "sd": 0.0, "sem": 0.0, "n": len(base_lsd)},
        "frac": {"mean": float("nan"), "sd": float("nan"), "sem": float("nan"), "n": 0},
    }

    # Paired within geometry: held-out position minus the mean of its seen counterparts.
    paired = {}
    for name, hold_g, seen_g in PAIRS:
        diffs = []
        for gid in sorted(geoms):
            h = [r["delta_vs_baseline_db"] for r in rows
                 if r["geom_id"] == gid and r["group"] == hold_g]
            s = [r["delta_vs_baseline_db"] for r in rows
                 if r["geom_id"] == gid and r["group"] == seen_g]
            if h and s:
                diffs.append(float(np.mean(h) - np.mean(s)))
        st = _mean_sd(diffs)
        st["distinguishable"] = bool(np.isfinite(st["sem"]) and st["sem"] > 0.0
                                     and abs(st["mean"]) > 2.0 * st["sem"])
        paired[name] = st
    return {"per_config": rows, "by_group": by_group, "paired_holdout_minus_seen": paired,
            "note": ("t_patch3_holdout has no position-matched seen counterpart in the "
                     "test split and is reported ungrouped; t_uniform_wall:east also "
                     "covers east_3 and is counted as held-out. 'distinguishable' is "
                     "|mean| > 2*sem over the 10 paired geometries.")}


# --------------------------------------------------------------------------- diagnostic 3
def diag_window(fields: Fields, geoms: Dict[int, Dict[str, SegConfig]],
                with_pred: bool) -> dict:
    """alpha = 0.95 configs: does the model reproduce GT's in-band energy drop?"""
    rows = []
    for gid in sorted(geoms):
        slots = geoms[gid]
        if "baseline" not in slots:
            continue
        base = slots["baseline"]
        g_b = fields.gt(base)
        p_b = fields.pred(base) if with_pred else None
        for slot in ("t_window_seen", "t_window_holdout"):
            if slot not in slots:
                continue
            c = slots[slot]
            g = fields.gt(c)
            rec = {"geom_id": gid, "slot": slot, "edited": list(c.edited),
                   "alpha": float(max(c.alphas)),
                   "touches_holdout": touches_holdout(c)}
            if with_pred:
                p = fields.pred(c)
                m = usable_mask(np.stack([g, g_b, p, p_b]))
                rec["lsd_db"] = _f(_lsd_db(p[m], g[m]))
                rec["d_energy_gt_db"] = band_energy_db(g, g_b, m)
                rec["d_energy_pred_db"] = band_energy_db(p, p_b, m)
                rec["frac"] = _f(m.mean())
                rec["energy_recovered_frac"] = (
                    _f(rec["d_energy_pred_db"] / rec["d_energy_gt_db"])
                    if rec["d_energy_gt_db"] not in (0.0,)
                    and np.isfinite(rec["d_energy_gt_db"]) else float("nan"))
            else:
                m = usable_mask(np.stack([g, g_b]))
                rec["d_energy_gt_db"] = band_energy_db(g, g_b, m)
                rec["frac_gt"] = _f(m.mean())
            rows.append(rec)
        fields.drop_geometry()

    by_slot = {}
    for slot in ("t_window_seen", "t_window_holdout"):
        sel = [r for r in rows if r["slot"] == slot]
        blk = {"n": len(sel),
               "d_energy_gt_db": _mean_sd([r["d_energy_gt_db"] for r in sel])}
        if with_pred:
            blk["lsd_db"] = _mean_sd([r["lsd_db"] for r in sel])
            blk["d_energy_pred_db"] = _mean_sd([r["d_energy_pred_db"] for r in sel])
            blk["energy_recovered_frac"] = _mean_sd([r["energy_recovered_frac"]
                                                     for r in sel])
            blk["frac"] = _mean_sd([r["frac"] for r in sel])
        else:
            blk["frac_gt"] = _mean_sd([r["frac_gt"] for r in sel])
        by_slot[slot] = blk
    return {"per_config": rows, "by_slot": by_slot}


# --------------------------------------------------------------------------- driver
def run(train_dir: str, manifest: str, data_dir: str, out_dir: str,
        checkpoint: Optional[str], gt_only: bool, rx_chunk: int,
        limit: Optional[int]) -> dict:
    man = json.loads(Path(manifest).read_text())
    rows = man["configs"] if isinstance(man, dict) else man
    cfgs = configs_from_rows(rows, split="test")
    geoms = group_by_geometry(cfgs)
    if limit:
        geoms = dict((k, geoms[k]) for k in sorted(geoms)[:limit])

    meta = {
        "manifest": manifest, "data_dir": data_dir, "train_dir": train_dir,
        "n_test_configs": len(cfgs), "n_geometries": len(geoms),
        "holdout_segment": SEGMENT_NAMES[HOLDOUT_INDEX], "holdout_index": HOLDOUT_INDEX,
        "band_hi_hz": BAND_HI_HZ, "eps_usable": EPS, "gt_only": bool(gt_only),
    }

    model = renderer = device = None
    cond_source = "m_segment"
    if gt_only:
        import h5py

        with h5py.File(str(Path(data_dir) / cfgs[0].filename), "r") as f:
            hi_idx = int(f["ism/H_complex"].shape[-1])
            meta["gt_n_bins"] = hi_idx
            meta["gt_band_hi_hz"] = float(f.attrs["band_hi_hz"])
    else:
        import torch

        from aaf.eval.p3_2_eval import find_checkpoint, load_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = Path(checkpoint) if checkpoint else find_checkpoint(train_dir)
        model, renderer, cfg, tmeta, it = load_model(ckpt, device)
        cond_source = str(cfg["cond_source"])
        if cond_source != "m_segment" or int(cfg["cond_dim"]) != 144:
            raise ValueError("expected m_segment/144 conditioning, got {}/{}".format(
                cond_source, cfg["cond_dim"]))
        n_freq = int(cfg["n_time_samples"]) // 2 + 1
        _, hi_idx = band_indices(float(cfg["fs"]), n_freq, 0.0, BAND_HI_HZ)
        sp = Path(train_dir) / "scalars.json"
        val = [r for r in json.loads(sp.read_text())
               if r.get("phase") == "val" and int(r.get("iter", 0)) <= it] \
            if sp.exists() else []
        meta.update({
            "checkpoint": str(ckpt), "iter": int(it), "cond_source": cond_source,
            "cond_dim": int(cfg["cond_dim"]),
            "conditioning_type": str(cfg.get("conditioning_type", "film")),
            "device": str(device), "n_freq_bins_pred": n_freq,
            "in_dist_val_lsd_db": float(val[-1]["lsd_db"]) if val else None,
            "n_train_configs": int(tmeta.get("n_configs", 0)),
        })
    meta["hi_idx"] = int(hi_idx)

    fields = Fields(Path(data_dir), hi_idx, model, renderer, cond_source, device, rx_chunk)
    with_pred = not gt_only
    t0 = time.time()
    res = {
        "meta": meta,
        "segment_discrimination": diag_segment_discrimination(fields, geoms, with_pred),
        "holdout_position": diag_holdout_position(fields, geoms, with_pred),
        "window": diag_window(fields, geoms, with_pred),
    }
    res["meta"]["wall_seconds"] = round(time.time() - t0, 1)
    res["meta"]["render_seconds"] = round(fields.render_seconds, 1)
    res["meta"]["n_renders"] = fields.n_rendered

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = "DIAGNOSTIC_GT_ONLY.json" if gt_only else "DIAGNOSTIC.json"
    (out / name).write_text(json.dumps(res, indent=2))
    print("[wrote] {}".format(out / name), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="P3-3-FAST Track A localization diagnostic")
    ap.add_argument("--train-dir", default="outputs/p3_3fast/p3_3fast_trackA")
    ap.add_argument("--manifest",
                    default="configs/sweeps_2d_mat/p3_3fast_trackA_manifest.json")
    ap.add_argument("--data-dir", default="data/track_p3_3fast_A")
    ap.add_argument("--out", default="outputs/p3_3fast/trackA")
    ap.add_argument("--checkpoint", default=None,
                    help="default: newest ckpt_iter*.pt in --train-dir")
    ap.add_argument("--gt-only", action="store_true",
                    help="ground-truth denominators only; runs without a GPU")
    ap.add_argument("--rx-chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N geometries")
    a = ap.parse_args()
    res = run(a.train_dir, a.manifest, a.data_dir, a.out, a.checkpoint, a.gt_only,
              a.rx_chunk, a.limit)
    d1 = res["segment_discrimination"]["aggregate"]
    print("[D1] gt_spread {:.3f} dB | pred_spread {} | ratio {}".format(
        d1["gt_spread_db"]["mean"],
        "n/a" if "pred_spread_db" not in d1
        else "{:.3f} dB".format(d1["pred_spread_db"]["mean"]),
        "n/a" if "spread_ratio_db_pooled" not in d1
        else "{:.3f}".format(d1["spread_ratio_db_pooled"])), flush=True)


if __name__ == "__main__":
    main()
