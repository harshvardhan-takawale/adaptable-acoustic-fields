"""2D (L, W) geometry sampling for P3-2.

A parallel of ``aaf.data.sample_rooms_3d`` rather than a generalization of it: that module
is ``d=3``-typed throughout (``Room3D``, ``as_tuple()`` -> 3-tuple, an explicit
``if len(ranges) != 3: raise``) and it produced the FROZEN 45/90/150/250 room sets that
Phase-2 results depend on. Touching it for zero upside is the wrong trade.

Two deliberate differences from the 3D sampler:

* **Dimensions are rounded to 2 dp at generation time** and uniqueness is asserted on the
  rounded set. Filenames encode ``{:.2f}``, so two geometries agreeing to 2 dp would
  silently share an HDF5 file. That bug is live in the 3D path
  (``configs/sweeps_3d/train_rooms_45.yaml`` stores ``L: 3.620434065857401`` while
  ``room_filename_3d`` formats ``L3.62``); it is not inherited here.
* **Near-square rooms are rejected** (``|L - W| < 0.05``). In 2D, L = W makes (n,m) and
  (m,n) exactly degenerate, which would collapse the x- and y-axial families the whole
  chunk is built on distinguishing.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import qmc

from aaf.data.mat_configs import round_dim

DEFAULT_RANGES_2D = ((3.0, 6.0), (3.0, 5.0))     # (L, W), matches conditioning_2d's box
DEFAULT_LHS_SEED = 42
DEFAULT_TEST_SEED = 2024
NEAR_SQUARE_TOL = 0.05
MIN_TEST_TRAIN_SEP_M = 0.05                      # Chebyshev, metres


def _normalize(pts: np.ndarray, ranges) -> np.ndarray:
    lo = np.array([r[0] for r in ranges], dtype=float)
    hi = np.array([r[1] for r in ranges], dtype=float)
    return (np.asarray(pts, dtype=float) - lo) / (hi - lo)


def sample_train_geometries(
    n: int = 40,
    ranges: Sequence[Sequence[float]] = DEFAULT_RANGES_2D,
    seed: int = DEFAULT_LHS_SEED,
    near_square_tol: float = NEAR_SQUARE_TOL,
) -> List[Tuple[float, float]]:
    """``n`` training geometries by Latin hypercube, rounded to 2 dp and unique."""
    lo = np.array([r[0] for r in ranges], dtype=float)
    hi = np.array([r[1] for r in ranges], dtype=float)
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    out: List[Tuple[float, float]] = []
    seen = set()
    for _ in range(64):                              # oversample; rejection is cheap
        pts = qmc.scale(sampler.random(n=4 * n), lo, hi)
        for L, W in pts:
            L, W = round_dim(L), round_dim(W)
            if abs(L - W) < near_square_tol:         # (n,m)/(m,n) degeneracy
                continue
            if (L, W) in seen:
                continue
            seen.add((L, W))
            out.append((L, W))
            if len(out) == n:
                return out
    raise RuntimeError(f"could not draw {n} distinct 2D geometries (got {len(out)})")


def sample_test_geometries(
    train: Sequence[Tuple[float, float]],
    n: int = 10,
    ranges: Sequence[Sequence[float]] = DEFAULT_RANGES_2D,
    seed: int = DEFAULT_TEST_SEED,
    candidate_pool: int = 8192,
    min_sep_m: float = MIN_TEST_TRAIN_SEP_M,
    near_square_tol: float = NEAR_SQUARE_TOL,
) -> List[Tuple[float, float]]:
    """``n`` frozen test geometries: strictly inside the training hull, maximin-spread.

    Interior (not extrapolative) so the comparison is genuinely interpolative, and at least
    ``min_sep_m`` (Chebyshev) from every training geometry so they are new rooms and cannot
    collide with a training filename after 2 dp rounding.
    """
    lo = np.array([r[0] for r in ranges], dtype=float)
    hi = np.array([r[1] for r in ranges], dtype=float)
    train_arr = np.asarray(train, dtype=float)
    hull = Delaunay(train_arr)

    cand = qmc.scale(qmc.Sobol(d=2, scramble=True, seed=seed).random(candidate_pool), lo, hi)
    keep = []
    for L, W in cand:
        L, W = round_dim(L), round_dim(W)
        if abs(L - W) < near_square_tol:
            continue
        if hull.find_simplex(np.array([L, W])) < 0:                 # outside the hull
            continue
        if np.min(np.max(np.abs(train_arr - np.array([L, W])), axis=1)) < min_sep_m:
            continue
        keep.append((L, W))
    keep = sorted(set(keep))
    if len(keep) < n:
        raise RuntimeError(f"only {len(keep)} interior candidates; need {n}")

    # Greedy maximin in normalized coordinates, seeded at the candidate nearest the centre.
    cand_n = _normalize(np.array(keep), ranges)
    centre = _normalize(np.array([[(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2]]), ranges)
    chosen = [int(np.argmin(np.linalg.norm(cand_n - centre, axis=1)))]
    for _ in range(n - 1):
        d = np.min(np.linalg.norm(cand_n[:, None, :] - cand_n[None, chosen, :], axis=2), axis=1)
        d[chosen] = -1.0
        chosen.append(int(np.argmax(d)))
    return [keep[i] for i in chosen]


def nn_distance_report(test: Sequence[Tuple[float, float]],
                       train: Sequence[Tuple[float, float]],
                       ranges: Sequence[Sequence[float]] = DEFAULT_RANGES_2D) -> dict:
    """Test->train nearest-neighbour distances (raw m and normalized) — the coverage axis."""
    te, tr = np.asarray(test, dtype=float), np.asarray(train, dtype=float)
    raw = np.min(np.linalg.norm(te[:, None, :] - tr[None, :, :], axis=2), axis=1)
    ten, trn = _normalize(te, ranges), _normalize(tr, ranges)
    nrm = np.min(np.linalg.norm(ten[:, None, :] - trn[None, :, :], axis=2), axis=1)
    cheb = np.min(np.max(np.abs(te[:, None, :] - tr[None, :, :]), axis=2), axis=1)
    return {
        "nn_raw_m": raw.tolist(), "nn_raw_mean": float(raw.mean()), "nn_raw_min": float(raw.min()),
        "nn_norm_mean": float(nrm.mean()), "nn_norm_min": float(nrm.min()),
        "nn_chebyshev_min_m": float(cheb.min()),
    }
