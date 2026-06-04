"""3D room sampling: LHS training set, structured test + de-risk sets.

Used by Chunk P2-1 to generate the room configs that the dataset builder
consumes. Three sets:

  - **Train (45 rooms)**: Latin hypercube sample over ``[3, 6] × [3, 5] ×
    [2.5, 4]`` m. Fixed seed=42 for reproducibility. Samples where
    ``|L - W| < reject_cubic_tol`` are rejected and redrawn (avoids 2-axis
    modal degeneracy).
  - **De-risk (5 rooms)**: spec-prescribed coordinates spanning the box;
    used for this chunk's single-room overfit experiments. Box center +
    4 near-corner rooms.
  - **Test (8 rooms)**: structured interpolative interior rooms. First is
    the box center; the remaining 7 are picked by maximizing minimum
    distance to the LHS training samples in normalized [0, 1]³ space.
    NOT simulated in P2-1 (spec defers full sim to P2-2).

Persistence (called by ``scripts/build_3d_dataset.py``):
    configs/sweeps_3d/derisk_rooms.yaml
    configs/sweeps_3d/train_rooms.yaml
    configs/sweeps_3d/test_rooms.yaml
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml
from scipy.stats import qmc


# Default ranges, spec-prescribed (DECISIONS.md D1).
DEFAULT_RANGES: tuple[tuple[float, float], ...] = (
    (3.0, 6.0),    # L
    (3.0, 5.0),    # W
    (2.5, 4.0),    # H
)
DEFAULT_LHS_SEED = 42
DEFAULT_REJECT_CUBIC_TOL = 0.05    # |L - W| < this triggers redraw

# Spec-prescribed shared cfg (DECISIONS.md D3, D4, D5).
DEFAULT_ALPHA = 0.15
DEFAULT_FS = 4096
DEFAULT_N_TIME = 8192
DEFAULT_SOURCE_OFFSET: tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class Room3D:
    L: float
    W: float
    H: float

    def as_dict(self) -> dict:
        return {"L": float(self.L), "W": float(self.W), "H": float(self.H)}

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.L), float(self.W), float(self.H))


# ----------------------------------------------------------------------
# LHS training rooms
# ----------------------------------------------------------------------


def sample_train_rooms_lhs(
    n: int = 45,
    ranges: Sequence[tuple[float, float]] = DEFAULT_RANGES,
    seed: int = DEFAULT_LHS_SEED,
    reject_cubic_tol: float = DEFAULT_REJECT_CUBIC_TOL,
) -> list[Room3D]:
    """Latin hypercube sampler over (L, W, H), with reject-and-redraw for
    near-cubic L≈W draws.

    Implementation: draw ``2·n`` samples up front (oversampled), filter out
    ``|L - W| < tol`` violations, take the first ``n`` survivors. If fewer
    than ``n`` survive after oversampling (extremely unlikely at tol=0.05
    over a 3 m × 2 m × 1.5 m parameter box), raises ``RuntimeError`` so the
    caller can bump the oversample factor.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if len(ranges) != 3:
        raise ValueError(f"ranges must specify 3 axes, got {len(ranges)}")
    for (lo, hi) in ranges:
        if not lo < hi:
            raise ValueError(f"range {lo, hi} must satisfy lo < hi")

    oversample = 4
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    raw = sampler.random(n=oversample * n)   # shape (oversample*n, 3) in [0, 1)
    lows = np.array([r[0] for r in ranges], dtype=np.float64)
    highs = np.array([r[1] for r in ranges], dtype=np.float64)
    scaled = lows + raw * (highs - lows)     # (oversample*n, 3)

    rooms: list[Room3D] = []
    for L, W, H in scaled:
        if abs(L - W) < reject_cubic_tol:
            continue
        rooms.append(Room3D(L=float(L), W=float(W), H=float(H)))
        if len(rooms) >= n:
            break
    if len(rooms) < n:
        raise RuntimeError(
            f"LHS oversample factor {oversample} insufficient: only "
            f"{len(rooms)} / {n} survived the |L - W| >= {reject_cubic_tol} "
            "filter. Increase oversample or relax tol."
        )
    return rooms


# ----------------------------------------------------------------------
# De-risk rooms (spec-prescribed)
# ----------------------------------------------------------------------


def derisk_rooms() -> list[Room3D]:
    """5 spec-prescribed rooms: box center + 4 extreme-corner-ish rooms."""
    return [
        Room3D(L=4.5, W=4.0, H=3.25),   # box center
        Room3D(L=3.0, W=3.0, H=2.5),
        Room3D(L=6.0, W=5.0, H=4.0),
        Room3D(L=3.0, W=5.0, H=2.5),
        Room3D(L=6.0, W=3.0, H=4.0),
    ]


# ----------------------------------------------------------------------
# Test rooms (structured, interpolative, maximin-vs-LHS)
# ----------------------------------------------------------------------


def _normalize_room(
    room: Room3D, ranges: Sequence[tuple[float, float]]
) -> np.ndarray:
    """Map (L, W, H) → [0, 1]³ given the (lo, hi) per-axis ranges."""
    lows = np.array([r[0] for r in ranges], dtype=np.float64)
    highs = np.array([r[1] for r in ranges], dtype=np.float64)
    return (np.array(room.as_tuple(), dtype=np.float64) - lows) / (highs - lows)


def sample_test_rooms(
    n: int = 8,
    lhs_rooms: Iterable[Room3D] = (),
    ranges: Sequence[tuple[float, float]] = DEFAULT_RANGES,
    seed: int = 1729,
    candidate_pool: int = 4096,
    box_center_first: bool = True,
) -> list[Room3D]:
    """Pick ``n`` test rooms inside the box, maximizing min-distance to the
    LHS training set.

    Algorithm:
      1. Optionally seed the first room at the box center.
      2. Pre-generate ``candidate_pool`` quasi-random candidates inside the
         box (Sobol).
      3. Greedy maximin: for each remaining slot, pick the candidate whose
         minimum distance (in normalized [0, 1]³ space) to the union of
         already-picked-and-training rooms is largest.

    This avoids the test rooms overlapping the LHS draws, so P2-2's zero-shot
    eval at these rooms genuinely probes the interpolative interior.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    lhs_norm = (
        [_normalize_room(r, ranges) for r in lhs_rooms] if lhs_rooms else []
    )

    picked: list[Room3D] = []
    picked_norm: list[np.ndarray] = []

    if box_center_first:
        lows = np.array([r[0] for r in ranges], dtype=np.float64)
        highs = np.array([r[1] for r in ranges], dtype=np.float64)
        c = 0.5 * (lows + highs)
        picked.append(Room3D(L=float(c[0]), W=float(c[1]), H=float(c[2])))
        picked_norm.append(np.full(3, 0.5))

    sampler = qmc.Sobol(d=3, seed=seed, scramble=True)
    cand_norm = sampler.random(n=candidate_pool)        # (candidate_pool, 3)
    lows = np.array([r[0] for r in ranges], dtype=np.float64)
    highs = np.array([r[1] for r in ranges], dtype=np.float64)
    cand_real = lows + cand_norm * (highs - lows)        # (candidate_pool, 3)

    # Mask out candidates too close to any picked or LHS room.
    while len(picked) < n:
        # Reference points to maximize distance from.
        ref_pts = lhs_norm + picked_norm
        if not ref_pts:
            # No references — pick a candidate near 0.5, 0.5, 0.5 corner-of-corner.
            idx = int(np.argmin(np.sum((cand_norm - 0.5) ** 2, axis=1)))
        else:
            ref = np.stack(ref_pts, axis=0)              # (R, 3)
            # Pairwise dist: cand × ref → (candidate_pool, R)
            diff = cand_norm[:, None, :] - ref[None, :, :]
            d2 = np.sum(diff * diff, axis=2)            # (candidate_pool, R)
            min_d2 = np.min(d2, axis=1)                  # (candidate_pool,)
            idx = int(np.argmax(min_d2))
        picked.append(
            Room3D(L=float(cand_real[idx, 0]),
                   W=float(cand_real[idx, 1]),
                   H=float(cand_real[idx, 2]))
        )
        picked_norm.append(cand_norm[idx].copy())
        # Drop this candidate from future considerations.
        cand_norm = np.delete(cand_norm, idx, axis=0)
        cand_real = np.delete(cand_real, idx, axis=0)

    return picked


# ----------------------------------------------------------------------
# Diagnostic subset (P2-2.5): greedy maximin over an existing finite room set
# ----------------------------------------------------------------------


def select_diag_subset_maximin(
    rooms: Sequence[Room3D],
    n: int = 10,
    ranges: Sequence[tuple[float, float]] = DEFAULT_RANGES,
    box_center_first: bool = True,
) -> list[Room3D]:
    """Greedy maximin pick of ``n`` rooms from ``rooms`` (a finite candidate
    set, typically the 45 LHS training rooms).

    Algorithm:
      1. Optionally seed the first pick as the room nearest the box center
         (deterministic; no randomness needed for a finite-set maximin).
      2. For each remaining slot, pick the unselected room whose minimum
         distance to the already-picked set (in normalized [0, 1]³ space) is
         largest.

    Used by Chunk P2-2.5 to construct a 10-room subset of the 45-room LHS
    training set for diagnostic runs A and C. The same input + parameters
    always produce the same output (deterministic over a fixed input list).
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    rooms_list = list(rooms)
    if n > len(rooms_list):
        raise ValueError(
            f"n={n} exceeds number of candidate rooms ({len(rooms_list)})"
        )

    norms = [_normalize_room(r, ranges) for r in rooms_list]
    remaining = set(range(len(rooms_list)))
    picked_idx: list[int] = []

    if box_center_first:
        center = np.full(3, 0.5)
        d2_to_center = np.array(
            [float(np.sum((norms[i] - center) ** 2)) for i in remaining]
        )
        seed_idx = list(remaining)[int(np.argmin(d2_to_center))]
        picked_idx.append(seed_idx)
        remaining.remove(seed_idx)

    while len(picked_idx) < n:
        picked_arr = np.stack([norms[i] for i in picked_idx], axis=0)        # (k, 3)
        remaining_list = sorted(remaining)
        cand_arr = np.stack(
            [norms[i] for i in remaining_list], axis=0
        )                                                                    # (R, 3)
        # Pairwise dist²: (R, k)
        diff = cand_arr[:, None, :] - picked_arr[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        min_d2 = np.min(d2, axis=1)
        # Pick the candidate whose min-distance to the picked set is largest.
        # Ties are broken by candidate list order (deterministic).
        next_local_idx = int(np.argmax(min_d2))
        next_idx = remaining_list[next_local_idx]
        picked_idx.append(next_idx)
        remaining.remove(next_idx)

    return [rooms_list[i] for i in picked_idx]


# ----------------------------------------------------------------------
# YAML I/O
# ----------------------------------------------------------------------


def _rooms_to_payload(
    rooms: Sequence[Room3D],
    *,
    set_name: str,
    extra_meta: dict | None = None,
) -> dict:
    return {
        "set_name": set_name,
        "alpha": DEFAULT_ALPHA,
        "fs": DEFAULT_FS,
        "n_time_samples": DEFAULT_N_TIME,
        "source_offset": list(DEFAULT_SOURCE_OFFSET),
        "rooms": [r.as_dict() for r in rooms],
        **(extra_meta or {}),
    }


def write_rooms_yaml(
    path: str | Path,
    rooms: Sequence[Room3D],
    *,
    set_name: str,
    extra_meta: dict | None = None,
) -> Path:
    """Persist a rooms list to YAML using the canonical schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _rooms_to_payload(rooms, set_name=set_name, extra_meta=extra_meta)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def read_rooms_yaml(path: str | Path) -> dict:
    """Load a rooms YAML; returns a dict with `set_name`, common params,
    and ``rooms = [{L, W, H}, …]``."""
    with open(path) as f:
        payload = yaml.safe_load(f)
    if "rooms" not in payload:
        raise ValueError(f"{path}: missing 'rooms' key")
    return payload
