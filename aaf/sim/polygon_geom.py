"""General polygon helpers for the FDTD shape family. No solver dependency.

Three things P4-2 needs that did not exist:

1. **A general point-in-polygon.** `build_p4_1_lroom.inside_polygon` is a hand-written boolean
   expression for one specific notch and cannot be generalized. Every shape in the notch family
   has different vertices, so membership must be computed from the vertex list.

2. **A general line-of-sight test**, for the LOS/NLOS evaluation split.

3. **The D65 extent assertion the chunk spec requires.** `wall_segments` and `patch` are
   hard-coded to the bounding rectangle's four outer edges and never consult `solid`: applied to
   a notched room they raise nothing, report `tiles_exactly: True`, and write extents summing to
   the FULL bounding-box wall length for a wall that physically exists for a fraction of it --
   and that `True` goes straight into the HDF5 attrs. `assert_extents_match_polygon` compares
   every reported extent against the polygon's actual edge lengths and raises on disagreement,
   so a silent `True` can never reach an artifact again.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def polygon_contains(verts: Sequence[Sequence[float]], pts: np.ndarray,
                     eps: float = 1e-12) -> np.ndarray:
    """Even-odd ray crossing. ``pts`` is ``[..., 2]``; returns a matching boolean array.

    Points exactly on an edge are treated as INSIDE: a receiver on a wall is a legitimate
    probe point, and excluding it would silently thin the receiver set asymmetrically (only on
    the axis-aligned walls, which is all of them here).
    """
    p = np.asarray(pts, dtype=float)
    x, y = p[..., 0], p[..., 1]
    inside = np.zeros(x.shape, dtype=bool)
    on_edge = np.zeros(x.shape, dtype=bool)
    n = len(verts)
    for i in range(n):
        x0, y0 = float(verts[i][0]), float(verts[i][1])
        x1, y1 = float(verts[(i + 1) % n][0]), float(verts[(i + 1) % n][1])
        # on this edge?
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        within = (np.minimum(x0, x1) - eps <= x) & (x <= np.maximum(x0, x1) + eps) & \
                 (np.minimum(y0, y1) - eps <= y) & (y <= np.maximum(y0, y1) + eps)
        on_edge |= (np.abs(cross) <= eps * max(1.0, abs(x1 - x0) + abs(y1 - y0))) & within
        # ray crossing to +x
        straddles = ((y0 > y) != (y1 > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x0 + (y - y0) * (x1 - x0) / np.where(y1 == y0, np.nan, y1 - y0)
        inside ^= straddles & (x < xint)
    return inside | on_edge


def line_of_sight(verts: Sequence[Sequence[float]], src: Sequence[float],
                  rx: np.ndarray, n_samples: int = 400) -> np.ndarray:
    """True where the straight ``src -> rx`` segment stays inside the polygon.

    A property of the GEOMETRY, not of the renderer -- `FreqRenderer2D` never traces a
    source->receiver ray (D66: rays fan outward from the receiver and the source is a network
    input). The split is still the right one: it separates receivers the source reaches
    directly from those it can only reach by diffracting around the reflex corner, which is
    what the architecture is being asked to represent.
    """
    s = np.asarray(src, dtype=float)
    r = np.asarray(rx, dtype=float).reshape(-1, 2)
    t = np.linspace(0.0, 1.0, n_samples)[None, :, None]
    seg = s[None, None, :] + t * (r[:, None, :] - s[None, None, :])
    return polygon_contains(verts, seg).all(axis=1)


def edge_lengths(verts: Sequence[Sequence[float]]) -> List[float]:
    n = len(verts)
    return [math.hypot(float(verts[(i + 1) % n][0]) - float(verts[i][0]),
                       float(verts[(i + 1) % n][1]) - float(verts[i][1])) for i in range(n)]


def wall_extent(verts: Sequence[Sequence[float]], wall: str, tol: float = 1e-9) -> float:
    """Total length of polygon edges lying ON the named bounding-box wall.

    This is the quantity `wall_segments` silently gets wrong. For a top-left notch the EAST and
    SOUTH walls still span the full bounding box, but WEST is truncated to ``W - d`` and NORTH
    to ``L - w`` -- so a guard that compares against "any edge length" passes the very case it
    exists to catch (5.0 m is a real edge here, just not the west one). The check has to be
    per-wall.
    """
    v = [(float(a), float(b)) for a, b in verts]
    xs = [p[0] for p in v]
    ys = [p[1] for p in v]
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    line = {"west": ("x", lo_x), "east": ("x", hi_x),
            "south": ("y", lo_y), "north": ("y", hi_y)}.get(str(wall))
    if line is None:
        raise ValueError("unknown wall {!r}".format(wall))
    axis, val = line
    total, n = 0.0, len(v)
    for i in range(n):
        x0, y0 = v[i]
        x1, y1 = v[(i + 1) % n]
        if axis == "x" and abs(x0 - val) <= tol and abs(x1 - val) <= tol:
            total += abs(y1 - y0)
        elif axis == "y" and abs(y0 - val) <= tol and abs(y1 - val) <= tol:
            total += abs(x1 - x0)
    return total


def assert_extents_match_polygon(specs: Iterable[Dict], verts: Sequence[Sequence[float]],
                                 tol: float = 1e-3) -> None:
    """Raise if any solver-reported wall extent disagrees with the polygon's real wall (D65).

    The guard the chunk spec asks for by name. `_apply_wall_segments` / `_apply_patch` index the
    FULL node count of the bounding rectangle's edge and never consult `solid`, so on a notched
    room they report a tidy `tiles_exactly: True` over a length the wall does not have. Nothing
    downstream checks it, and the value lands in the HDF5 attrs where it reads as a passing
    check.

    Checked PER WALL against `wall_extent`, not against the set of edge lengths: for a top-left
    notch the east wall really is the full bounding-box height, so an "any edge length" test
    passes the exact failure it is meant to catch.
    """
    for spec in specs or ():
        if not isinstance(spec, dict):
            continue
        wall = spec.get("wall")
        ext = spec.get("extents_m") or spec.get("extents") or []
        if wall is None or not ext:
            continue
        actual = wall_extent(verts, wall)
        total = float(sum(float(e) for e in ext))
        if abs(total - actual) > tol:
            raise AssertionError(
                "spec for the {!r} wall reports extents summing to {:.4f} m, but that wall of "
                "this polygon is {:.4f} m long (D65). {} describes the BOUNDING RECTANGLE, not "
                "this room -- the difference would have been written into the HDF5 attrs as a "
                "passing check. spec={}".format(wall, total, actual, spec.get("type"), spec))
        if any(float(e) > actual + tol for e in ext):
            raise AssertionError(
                "spec for the {!r} wall reports a single extent longer than the whole wall "
                "({:.4f} m of {:.4f} m) (D65). spec={}".format(
                    wall, max(float(e) for e in ext), actual, spec))


def notch_solid_mask(nx: int, ny: int, dxx: float, dxy: float,
                     W: float, d: float, w: float) -> np.ndarray:
    """Solid nodes for a TOP-LEFT notch of depth ``d`` and width ``w``.

    The reflecting plane sits at the first AIR node on each side, one dx outside the solid
    block, so the node at ``x = w`` and the node at ``y = W - d`` must both stay AIR -- they ARE
    the notch's two faces. Off by one and the walls land a cell away from where the polygon says
    they are; wrong corner and you simulate a different room that still looks plausible. Both
    were live mistakes in P4-1, and the second one actually happened, which is why every caller
    cross-checks the result against the analytic polygon node-for-node.
    """
    m = np.zeros((nx, ny), dtype=bool)
    if d <= 0.0 or w <= 0.0:
        return m
    i_w = int(round(w / dxx))
    j_d = int(round((W - d) / dxy))
    m[:i_w, j_d + 1:] = True
    return m
