"""Zero-shot self-diagnosis primitives (P2-3, DECISIONS.md D37).

Two pure functions, kept free of torch so they unit-test fast:

  compute_manifold_distances — where does the optimized z* sit relative to the
    trained latent manifold? Returns nearest-latent distance, the geometrically
    nearest training room's latent distance, etc.

  classify_zero_shot_room — the 3-way verdict for one test room, given the model's
    in-distribution fit, the room's zero-shot mag-corr, and its geometry-placement.

The 3-way rule (D37), assuming in-distribution fit cleared its bar:
  1. mag-corr ≥ mag_thresh                      → "success" (method works)
  2. mag-corr < mag_thresh AND geometry MISplaced → "manifold_coverage"
        (z* couldn't reach the right region → fix = more training rooms, P2-4)
  3. mag-corr < mag_thresh AND geometry placed   → "decoder_interp"
        (z* is right but the decoder renders interpolated latents poorly →
         investigate decoder smoothness)
If in-distribution did NOT clear its bar, the room is "precondition_unmet" — the
zero-shot number is not interpretable as method success/failure.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def compute_manifold_distances(
    z_star: np.ndarray,
    z_train: np.ndarray,
    train_LWH: Optional[Sequence[Sequence[float]]] = None,
    test_LWH: Optional[Sequence[float]] = None,
) -> dict:
    """Distances from z* to the trained latent manifold.

    Parameters
    ----------
    z_star : [d] optimized test-time latent.
    z_train : [n_rooms, d] trained per-room latents.
    train_LWH : optional [n_rooms, 3] true dims of each training room.
    test_LWH : optional [3] true dims of this test room.

    Returns a dict with at least ``latent_min_dist``, ``latent_mean_dist``,
    ``latent_nearest_room_idx``; and if dims are given,
    ``geom_nearest_train_idx`` / ``geom_nearest_train_dist`` (the L2 latent
    distance to the training room whose TRUE geometry is closest to this test
    room — i.e. is z* near where its geometric neighbours sit?).
    """
    z_star = np.asarray(z_star, dtype=np.float64).reshape(-1)
    z_train = np.asarray(z_train, dtype=np.float64)
    if z_train.ndim != 2 or z_train.shape[1] != z_star.shape[0]:
        raise ValueError(f"z_train {z_train.shape} vs z_star {z_star.shape}")
    d_all = np.linalg.norm(z_train - z_star[None, :], axis=1)
    nn_idx = int(np.argmin(d_all))
    out = {
        "latent_min_dist": float(d_all[nn_idx]),
        "latent_mean_dist": float(d_all.mean()),
        "latent_nearest_room_idx": nn_idx,
    }
    if train_LWH is not None and test_LWH is not None:
        dims = np.asarray(train_LWH, dtype=np.float64)
        if dims.shape == (z_train.shape[0], 3):
            geom_d = np.linalg.norm(dims - np.asarray(test_LWH, dtype=np.float64)[None, :], axis=1)
            geom_nn = int(np.argmin(geom_d))
            out["geom_nearest_train_idx"] = geom_nn
            out["geom_nearest_train_dist"] = float(d_all[geom_nn])
            out["latent_nearest_room_LWH"] = [float(x) for x in dims[nn_idx]]
            out["geom_nearest_train_room_LWH"] = [float(x) for x in dims[geom_nn]]
    return out


def classify_zero_shot_room(
    in_dist_lsd: Optional[float],
    mag_corr: Optional[float],
    geom_err_max_m: Optional[float],
    in_dist_thresh: float = 2.5,
    mag_thresh: float = 0.9,
    geom_thresh: float = 0.3,
) -> tuple[str, str]:
    """The D37 3-way verdict for one room. Returns (branch_id, human_label).

    branch_id ∈ {"precondition_unmet", "success", "manifold_coverage",
    "decoder_interp", "unknown"}.
    """
    if in_dist_lsd is None or not np.isfinite(in_dist_lsd) or in_dist_lsd > in_dist_thresh:
        return ("precondition_unmet",
                f"in-distribution fit {in_dist_lsd} dB did not clear "
                f"≤{in_dist_thresh} dB — zero-shot not interpretable as success/failure")
    if mag_corr is None or not np.isfinite(mag_corr):
        return ("unknown", "missing mag_corr")
    if mag_corr >= mag_thresh:
        return ("success", f"mag corr {mag_corr:.3f} ≥ {mag_thresh} — method works")
    # zero-shot poor → split on geometry placement
    if geom_err_max_m is None or not np.isfinite(geom_err_max_m):
        return ("unknown", "missing geometry-placement to split poor zero-shot")
    if geom_err_max_m > geom_thresh:
        return ("manifold_coverage",
                f"mag corr {mag_corr:.3f} < {mag_thresh} AND geometry misplaced "
                f"(max axis err {geom_err_max_m:.2f} m > {geom_thresh}) → manifold-coverage "
                f"problem → more training rooms (P2-4)")
    return ("decoder_interp",
            f"mag corr {mag_corr:.3f} < {mag_thresh} but geometry well-placed "
            f"(max axis err {geom_err_max_m:.2f} m ≤ {geom_thresh}) → decoder-at-interpolated-"
            f"latent problem → investigate decoder smoothness")


def aggregate_verdict(branch_ids: Sequence[str], n_total: int) -> str:
    """One-line headline from the per-room branch ids."""
    if not branch_ids:
        return "no rooms evaluated"
    n = {b: branch_ids.count(b) for b in set(branch_ids)}
    n_success = n.get("success", 0)
    if n.get("precondition_unmet", 0) == len(branch_ids):
        return ("PRECONDITION UNMET — the model did not fit in-distribution (≤2.5 dB); "
                "zero-shot numbers below are recorded but not interpretable as method "
                "success/failure.")
    if n_success >= 5:
        return (f"SUCCESS — the method generalizes to 3D zero-shot: {n_success}/{n_total} "
                f"rooms reach mag corr ≥ 0.9.")
    dominant = max(("manifold_coverage", "decoder_interp"), key=lambda b: n.get(b, 0))
    if n.get(dominant, 0) == 0:
        return (f"MIXED — {n_success}/{n_total} rooms succeed; remainder unclassified "
                f"(missing diagnostics).")
    if dominant == "manifold_coverage":
        return (f"BELOW TARGET — {n_success}/{n_total} succeed; the dominant failure is "
                f"MANIFOLD-COVERAGE ({n.get('manifold_coverage', 0)} rooms: z* geometrically "
                f"misplaced) → fix is more training rooms (P2-4), not more iters/capacity.")
    return (f"BELOW TARGET — {n_success}/{n_total} succeed; the dominant failure is "
            f"DECODER-AT-INTERPOLATED-LATENT ({n.get('decoder_interp', 0)} rooms: z* well-placed "
            f"but spectrum off) → investigate decoder smoothness at unseen latents.")
