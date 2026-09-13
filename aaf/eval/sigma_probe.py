"""Does the learned sigma field actually encode the occluder? A reusable mechanism probe.

P4-1 asked this once, by hand, for one L-room. The answer was the most important finding of that
chunk -- sigma was statistically elevated inside the notch (p = 2.5e-05, Cohen's d = 0.736) but
PHYSICALLY NEGLIGIBLE: transmittance across the 3.68 m crossing was 0.068 against 0.082 for the
same path length in air, a 17% relative difference on a near-uniform field. So the architecture
could FIT a non-convex room without representing occlusion geometrically, and Gate 1's +0.974
measured capacity, not mechanism.

That analysis had NO producer script. It was done ad hoc in-session and only its JSON survived,
which means the one number that would tell us whether shape transfer can work was not
reproducible. This module makes it a standing diagnostic callable on any (model, polygon,
source), so mechanism is MEASURED in every future shape chunk rather than inferred from
in-distribution accuracy.

WHY THE PROBE RAY POINTS WHERE IT DOES
--------------------------------------
`FreqRenderer2D` fans rays OUTWARD FROM THE RECEIVER and feeds the source in as a network input
(D66); transmittance accumulates only along the receiver->point leg. So sigma is structurally
capable of representing a wall between the RECEIVER and a sample point, and structurally
INCAPABLE of representing one between the SOURCE and that point -- the latter can only be
absorbed into `signal(pts, tx)`. The probe therefore fires from an NLOS receiver toward the
source: the one direction where sigma could express the occluder. If it does not rise there, the
representation has not found the wall even where it was able to.

Interpretation guide for the returned `solid_over_air`:
    ~1.0                -> sigma is doing generic distance attenuation; the shape lives in
                           `signal`, i.e. the model memorizes each room. Predicts that transfer
                           to an unseen shape family will FAIL however good in-distribution
                           numbers look.
    >1 but transmittance
    ratio near 1        -> a statistically real but physically inert bump (P4-1's case).
    >>1 with a large
    transmittance gap   -> a genuine learned occluder.
The statistical test and the transmittance comparison are both returned because the first alone
is misleading: with 400 samples a 7% mean difference is overwhelmingly "significant" and still
means nothing physically.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from aaf.models.conditioning_2d import build_cond_vector_2d
from aaf.sim.polygon_geom import line_of_sight, polygon_contains
from aaf.walls import ALPHA_BASELINE, WORLD_SCALE


def sigma_along_ray(model, cond: torch.Tensor, rx0: Sequence[float],
                    direction: Sequence[float], src: Sequence[float], device,
                    n: int = 400, reach: Optional[float] = None,
                    verts: Optional[Sequence[Sequence[float]]] = None):
    """Band-mean sigma sampled along one ray from ``rx0``.

    Queries the model DIRECTLY rather than through the renderer: the renderer returns only the
    integrated H, and the question here is what the field looks like along the path.
    """
    d = np.asarray(direction, dtype=float)
    nrm = float(np.linalg.norm(d))
    if nrm <= 0:
        raise ValueError("direction must be non-zero")
    d = d / nrm
    if reach is None:
        if verts is None:
            raise ValueError("give either reach or verts (reach is derived from the bbox)")
        xs = [p[0] for p in verts]
        ys = [p[1] for p in verts]
        reach = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    t = np.linspace(1e-3, float(reach), int(n))
    pts = np.asarray(rx0, dtype=float)[None, :] + t[:, None] * d[None, :]
    P = torch.tensor(pts, dtype=torch.float32, device=device).unsqueeze(0)
    V = torch.tensor(np.repeat(d[None, :], n, axis=0), dtype=torch.float32,
                     device=device).unsqueeze(0)
    TX = torch.tensor(np.repeat(np.asarray(src, dtype=float)[None, :], n, axis=0),
                      dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        attn, _ = model(P, V, TX, tx_view=None, z_s=cond.unsqueeze(0).expand(1, -1))
        sig = attn.real.clamp(min=0)[0].mean(dim=-1).cpu().numpy()
    return t, pts, sig


def pick_nlos_probe_receiver(verts, src, rx: np.ndarray) -> Optional[np.ndarray]:
    """The NLOS receiver whose straight line to the source spends the most of its length inside
    solid -- i.e. the one with the most occluder to find. Returns None if nothing is NLOS."""
    rx = np.asarray(rx, dtype=float).reshape(-1, 2)
    los = line_of_sight(verts, src, rx)
    cand = rx[~los]
    if not len(cand):
        return None
    s = np.asarray(src, dtype=float)
    t = np.linspace(0.0, 1.0, 200)[None, :, None]
    seg = cand[:, None, :] + t * (s[None, None, :] - cand[:, None, :])
    frac_solid = (~polygon_contains(verts, seg)).mean(axis=1)
    return cand[int(np.argmax(frac_solid))]


def probe_sigma_occlusion(model, verts, src, device, rx: Optional[np.ndarray] = None,
                          edge_alphas: Optional[Sequence[float]] = None,
                          cond_source: str = "geom_token", n: int = 400,
                          world_scale: Optional[float] = None) -> Dict:
    """Full mechanism probe on one polygon. Returns the same keys P4-1's GATE1.json carried,
    plus the statistics that analysis produced with no code behind it.

    ``rx`` is optional; without it a coarse interior grid is generated to find a probe receiver.
    """
    verts = [(float(a), float(b)) for a, b in verts]
    xs = [p[0] for p in verts]
    ys = [p[1] for p in verts]
    L, W = max(xs) - min(xs), max(ys) - min(ys)
    if edge_alphas is None:
        edge_alphas = [ALPHA_BASELINE] * len(verts)

    # The tokens divide positions by WORLD_SCALE; the model applies its own world_scale to
    # sample points. If those disagree the probe silently reads the field at the wrong place.
    if world_scale is not None and abs(float(world_scale) - WORLD_SCALE) > 1e-12:
        raise ValueError(
            "checkpoint world_scale {} != conditioning WORLD_SCALE {}; the probe would sample "
            "the field at the wrong coordinates".format(world_scale, WORLD_SCALE))

    if rx is None:
        step = 0.15
        g = np.array([[x, y] for x in np.arange(step, L, step)
                      for y in np.arange(step, W, step)], dtype=float)
        rx = g[polygon_contains(verts, g)]
    rx0 = pick_nlos_probe_receiver(verts, src, rx)
    if rx0 is None:
        return {"probed": False,
                "reason": "no NLOS receiver exists for this polygon and source -- sigma has no "
                          "occluder to find along any receiver->source ray, so the probe is "
                          "undefined rather than negative"}

    cond = build_cond_vector_2d(cond_source, L, W, edge_alphas, verts=verts, device=device)
    t, pts, sig = sigma_along_ray(model, cond, rx0, np.asarray(src, float) - rx0, src,
                                  device, n=n, verts=verts)
    # THE OCCLUDER IS THE REMOVED CORNER, NOT EVERYTHING OUTSIDE THE POLYGON.
    #
    # The ray runs to `reach` = the bbox diagonal, so it passes the source and exits through the
    # far wall; every sample beyond that wall is also "outside the polygon". Counting those as
    # solid measured 82% out-of-room samples on P4-2's test shapes (99% on one), which is
    # harmless for a near-uniform sigma field -- P4-1 and P4-2 both read ~1.0 either way -- but
    # it DILUTES a genuinely localized notch contrast toward 1.0 and would hide exactly the
    # mechanism this probe exists to detect. Arm S reads 1.31 unclipped and 8.08 clipped.
    outside_poly = ~polygon_contains(verts, pts)
    in_bbox = ((pts[:, 0] >= min(xs)) & (pts[:, 0] <= max(xs))
               & (pts[:, 1] >= min(ys)) & (pts[:, 1] <= max(ys)))
    solid = outside_poly & in_bbox
    n_beyond_room = int((outside_poly & ~in_bbox).sum())
    if not solid.any():
        return {"probed": False,
                "reason": "the probe ray does not cross the removed corner; nothing to measure",
                "n_beyond_room": n_beyond_room}

    a, b = sig[solid], sig[~solid]
    dx = float(t[1] - t[0])
    cross_m = float(solid.sum() * dx)
    # Transmittance across the solid crossing vs the SAME path length in air. This is the
    # physical question -- a sigma bump that does not attenuate is not an occluder.
    trans_solid = float(np.exp(-a.mean() * cross_m))
    trans_air = float(np.exp(-b.mean() * cross_m))
    try:
        from scipy import stats
        pval = float(stats.mannwhitneyu(a, b, alternative="greater").pvalue)
    except Exception:
        pval = float("nan")
    pooled = math.sqrt((a.var() + b.var()) / 2.0) if len(a) and len(b) else 0.0
    d_eff = float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")

    return {
        "probed": True,
        "rx": [float(x) for x in rx0],
        "n_samples": int(len(sig)), "n_samples_in_solid": int(solid.sum()),
        # How much of the ray left the room entirely. Kept so the clipping above is auditable
        # and so a future reader can reproduce the unclipped (diluted) number if they need it.
        "n_samples_beyond_room": n_beyond_room,
        "mean_sigma_solid": float(a.mean()), "mean_sigma_air": float(b.mean()),
        "sd_sigma_solid": float(a.std()), "sd_sigma_air": float(b.std()),
        "solid_over_air": float(a.mean() / b.mean()) if b.mean() > 0 else float("nan"),
        "mannwhitney_p_solid_gt_air": pval,
        "cohens_d": d_eff,
        "sigma_min": float(sig.min()), "sigma_max": float(sig.max()),
        "sigma_spread": float(sig.max() / max(sig.min(), 1e-12)),
        "crossing_m": cross_m,
        "transmittance_across_solid": trans_solid,
        "transmittance_same_path_in_air": trans_air,
        "transmittance_ratio": float(trans_solid / trans_air) if trans_air > 0 else float("nan"),
        "t": t.tolist(), "pts": pts.tolist(), "sigma": sig.tolist(),
        "interpretation": _interpret(float(a.mean() / b.mean()) if b.mean() > 0 else float("nan"),
                                     trans_solid, trans_air),
    }


def _interpret(ratio: float, trans_solid: float, trans_air: float) -> str:
    if not np.isfinite(ratio):
        return "undefined"
    tr = trans_solid / trans_air if trans_air > 0 else float("nan")
    if ratio < 1.02:
        return ("sigma is NOT encoding the occluder -- generic distance attenuation only. The "
                "shape is carried by `signal`, i.e. the model memorizes each room, which "
                "predicts that transfer to an unseen shape family will fail regardless of "
                "in-distribution accuracy.")
    if tr > 0.5:
        return ("sigma is statistically elevated inside the solid but PHYSICALLY NEGLIGIBLE "
                "(transmittance ratio {:.2f}); this is P4-1's regime -- capacity without "
                "mechanism.".format(tr))
    return ("sigma attenuates materially across the solid (transmittance ratio {:.2f}) -- a "
            "genuine learned occluder.".format(tr))
