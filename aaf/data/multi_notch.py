"""P4-4 Task 3: shapes with TWO reflex corners -- U, T and diagonal double-notch rooms.

WHY A NEW MODULE RATHER THAN AN EDIT TO `shape_configs.py`. That module's draws are frozen into
three manifests (P4-2's 75 rooms, P4-3's 107, P4-4's 515) and `build_p4_4_manifest.py` asserts
they reproduce row-for-row. Generalizing its `(d, w)` scalars in place would put every one of
those assertions at risk for no benefit, so the single-notch family stays exactly as it is and
the multi-notch family lives beside it.

THE STEP THIS TAKES, AND WHY IT STOPS HERE. P4-1 fit one L-room; P4-2/3/4 generalize across
one-corner rooms. The Z-corridor needs three reflex corners. Jumping straight there would
confound "more corners" with "a shape family the model has never seen", so this adds exactly ONE
corner and asks whether corner count extrapolates a step at a time. Rectangle 4 boundary tokens,
L 6, U/T/DN 8 -- and `MAX_SEG_POLY = 12` already covers all of them plus a 10-vertex Z, so
nothing in the model, the conditioner or the renderer changes.

WHERE THE NOTCHES MAY SIT, AND WHY. The corpus source is frozen at `SRC = (0.5, 1.6)`, chosen to
clear a single top-left notch. A notch on the west or south-west would swallow it, and the
trainer would then train every room against one room's source with no error anywhere (it keeps
only the LAST config's `source_pos`). So notches are restricted to placements that provably
leave `(0.5, 1.6)` in air for every legal draw:

    NW   [0, w] x [W-d, W]          W - d >= 0.55 W >= 2.2 > 1.6        safe
    NE   [L-w, L] x [W-d, W]        L - w >= 0.55 L >= 2.75 > 0.5       safe
    midN [x0, x0+w] x [W-d, W]      same y argument as NW               safe
    SE   [L-w, L] x [0, d]          x argument as NE                    safe

    rect = {}            L = {NW}            U = {NW, NE}
    T    = {midN}        DN = {NW, SE}
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aaf.data.shape_configs import (
    ALPHA, D_FRAC_MAX, GRID_DX, L_RANGE, MIN_PASSAGE_M, SRC, W_FRAC_MAX, W_RANGE, _q,
    d_max_for, w_max_for,
)

SIDES = ("NW", "NE", "midN", "SE")
FAMILIES = {"rect": (), "L": ("NW",), "U": ("NW", "NE"), "T": ("midN",), "DN": ("NW", "SE")}
N_TOKENS = {"rect": 4, "L": 6, "U": 8, "T": 8, "DN": 8}
FAMILY_SEED = 20260915
MIN_STEM_M = 0.8           # the north stem of a U, and each shoulder of a T
EPS = 1e-9


@dataclass(frozen=True)
class Notch:
    side: str
    d: float
    w: float
    x0: float = 0.0        # midN only: left edge of the notch

    def __post_init__(self):
        if self.side not in SIDES:
            raise ValueError("unknown side {!r}, expected one of {}".format(self.side, SIDES))


def _merge(pts: Sequence[Tuple[float, float]], tol: float = 1e-9) -> List[Tuple[float, float]]:
    """Drop coincident and collinear vertices.

    `polygon_edge_tokens` raises on a zero-length edge, and a collinear triple would emit two
    tokens describing one physical wall -- which would silently change the token COUNT without
    changing the room, exactly the confound Task 3 exists to measure.
    """
    out: List[Tuple[float, float]] = []
    for p in pts:
        if out and abs(p[0] - out[-1][0]) < tol and abs(p[1] - out[-1][1]) < tol:
            continue
        out.append((float(p[0]), float(p[1])))
    while len(out) > 1 and abs(out[0][0] - out[-1][0]) < tol and abs(out[0][1] - out[-1][1]) < tol:
        out.pop()
    merged, n = [], len(out)
    for i in range(n):
        a, b, c = out[(i - 1) % n], out[i], out[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > tol:
            merged.append(b)
    return merged


def verts_for_notches(L: float, W: float, notches: Sequence[Notch]) -> List[Tuple[float, float]]:
    """CCW, origin-anchored ring for a rectangle with the given notches removed.

    Built by WALKING the boundary counter-clockwise and inserting each notch's detour in
    traversal order, rather than by writing out a vertex list per family. A hand-written 8-vertex
    ring is the single easiest thing to get wrong here: `polygon_edge_tokens` asserts the shoelace
    sign and a clockwise list would invert every inward normal, training the model on a room
    turned inside out.
    """
    by = {n.side: n for n in notches}
    if len(by) != len(list(notches)):
        raise ValueError("at most one notch per side")
    p: List[Tuple[float, float]] = [(0.0, 0.0)]

    se = by.get("SE")                                   # south edge, x: 0 -> L
    if se:
        p += [(L - se.w, 0.0), (L - se.w, se.d), (L, se.d)]
    else:
        p += [(L, 0.0)]

    ne = by.get("NE")                                   # east edge, y: -> W
    if ne:
        p += [(L, W - ne.d), (L - ne.w, W - ne.d), (L - ne.w, W)]
    else:
        p += [(L, W)]

    mn = by.get("midN")                                 # north edge, x: L -> 0
    if mn:
        p += [(mn.x0 + mn.w, W), (mn.x0 + mn.w, W - mn.d), (mn.x0, W - mn.d), (mn.x0, W)]

    nw = by.get("NW")
    if nw:
        p += [(nw.w, W), (nw.w, W - nw.d), (0.0, W - nw.d)]
    else:
        p += [(0.0, W)]

    v = _merge(p)                                       # west edge closes implicitly
    area2 = sum(v[i][0] * v[(i + 1) % len(v)][1] - v[(i + 1) % len(v)][0] * v[i][1]
                for i in range(len(v)))
    if area2 <= 0.0:
        raise ValueError("ring came out clockwise (2A = {:.6g}) for L={} W={} notches={}".format(
            area2, L, W, notches))
    return v


def solid_blocks(nx: int, ny: int, dxx: float, dxy: float, L: float, W: float,
                 notches: Sequence[Notch]) -> np.ndarray:
    """Solid-node mask for any legal notch set, OR-ed over notches.

    THE INDEX CONVENTION IS THE WHOLE GAME, and it is asymmetric on purpose. The reflecting plane
    sits at the first AIR node, one dx outside the solid block, so a face that is INTERIOR to the
    room must leave its node air (hence the `+1` / the exclusive upper bound) while a face that
    lies ON an outer wall must not, because that wall already carries its own face. A corner notch
    has one interior face per axis; a mid-wall notch (T) has TWO interior x-faces and needs the
    exclusion on both sides. Getting this wrong puts the wall one cell from where the polygon says
    it is and simulates a different room that still looks entirely plausible -- the P4-1 failure.

    Every caller cross-checks the result node-for-node against `polygon_contains`, which is what
    makes this safe rather than merely careful.
    """
    m = np.zeros((nx, ny), dtype=bool)
    for n in notches:
        if n.d <= 0.0 or n.w <= 0.0:
            continue
        if n.side == "NW":                              # x: outer west -> interior face at w
            m[:int(round(n.w / dxx)), int(round((W - n.d) / dxy)) + 1:] = True
        elif n.side == "NE":                            # x: interior face at L-w -> outer east
            m[int(round((L - n.w) / dxx)) + 1:, int(round((W - n.d) / dxy)) + 1:] = True
        elif n.side == "midN":                          # BOTH x faces interior
            m[int(round(n.x0 / dxx)) + 1:int(round((n.x0 + n.w) / dxx)),
              int(round((W - n.d) / dxy)) + 1:] = True
        elif n.side == "SE":                            # y: outer south -> interior face at d
            m[int(round((L - n.w) / dxx)) + 1:, :int(round(n.d / dxy))] = True
    return m


def reflex_corners(L: float, W: float, notches: Sequence[Notch]) -> List[Tuple[float, float]]:
    """Every reflex vertex, so receiver sampling can clear the r^(2/3) singularity at ALL of them.

    The single-notch builder hard-codes one corner; with two notches the receivers nearest the
    second one would sit in the unresolved singular field and be scored as if they were fine.
    """
    out = []
    for n in notches:
        if n.d <= 0.0 or n.w <= 0.0:
            continue
        if n.side == "NW":
            out.append((n.w, W - n.d))
        elif n.side == "NE":
            out.append((L - n.w, W - n.d))
        elif n.side == "midN":
            out += [(n.x0, W - n.d), (n.x0 + n.w, W - n.d)]
        elif n.side == "SE":
            out.append((L - n.w, n.d))
    return out


def geometry_ok(L: float, W: float, notches: Sequence[Notch]) -> Tuple[bool, str]:
    """Reject draws that are degenerate, disconnected, or collapse the bounding box.

    `passage_ok` in the single-notch family is vacuous by its own admission -- minimum passage is
    0.55 * W_min = 2.20 m, so it never fires. With two notches it becomes live, and three of these
    failures are SILENT rather than loud:

    * **A collapsed bbox.** `wall_extent` derives the four walls from the polygon's own bounding
      box, so if the notches consume an entire bbox edge it measures the notch floor instead of
      the wall and `assert_extents_match_polygon` validates against the wrong line. Nothing else
      checks that `max(x) == L` and `max(y) == W`.
    * **A disconnected domain.** There is no connectivity check anywhere in the repo, and the
      solver would accept a split domain with the dead component silently identically zero.
      Measured here, though: this parameterization CANNOT express one. `verts_for_notches`
      emits a single simple ring and the interior of a simple polygon is connected by
      definition, so the extreme draws that would separate two corners come out as a ring whose
      edges cross and are rejected by the shoelace assert first. `_connected` is kept as
      belt-and-braces for a future parameterization -- disjoint blocks, interior pillars --
      that could express a split.
    * **A self-intersecting ring.** `polygon_contains` on one is even-odd garbage, and the
      builder's node-for-node cross-check would NOT flag it, because both sides use the same bad
      polygon.

    The FDTD's own zero-width-channel raise is a backstop, but it fires after the manifest is
    frozen and the array has launched, so the constraint belongs here.
    """
    for n in notches:
        if n.d <= 0.0 or n.w <= 0.0:
            return False, "degenerate notch {}".format(n)
        if n.d > D_FRAC_MAX * W + EPS:
            return False, "depth {:.3f} exceeds {:.3f}".format(n.d, D_FRAC_MAX * W)
        if n.w > W_FRAC_MAX * L + EPS:
            return False, "width {:.3f} exceeds {:.3f}".format(n.w, W_FRAC_MAX * L)
    by = {n.side: n for n in notches}
    if "NW" in by and "NE" in by:                       # U: a north stem must survive
        stem = L - by["NW"].w - by["NE"].w
        if stem < MIN_STEM_M:
            return False, "U stem {:.3f} m < {:.3f}".format(stem, MIN_STEM_M)
    if "midN" in by:                                    # T: both shoulders must survive
        n = by["midN"]
        if n.x0 < MIN_STEM_M or (L - n.x0 - n.w) < MIN_STEM_M:
            return False, "T shoulders {:.3f}/{:.3f} m too thin".format(
                n.x0, L - n.x0 - n.w)
    if "NW" in by and "SE" in by:                       # DN: opposite corners must not meet
        if by["NW"].w + by["SE"].w > L - MIN_PASSAGE_M:
            return False, "DN corners pinch in x"
    v = verts_for_notches(L, W, notches)
    xs = [p[0] for p in v]
    ys = [p[1] for p in v]
    if abs(max(xs) - L) > EPS or abs(max(ys) - W) > EPS:
        return False, "bounding box collapsed to {:.3f}x{:.3f}, expected {:.3f}x{:.3f}".format(
            max(xs), max(ys), L, W)
    if abs(min(xs)) > EPS or abs(min(ys)) > EPS:
        return False, "not origin-anchored"
    if not _connected(L, W, notches):
        return False, "air region is disconnected"
    if _self_intersects(v):
        return False, "ring self-intersects"
    from aaf.sim.polygon_geom import polygon_contains
    if not bool(polygon_contains(v, np.asarray([SRC], dtype=float))[0]):
        return False, "source {} is inside solid".format(SRC)
    return True, "ok"


def _connected(L: float, W: float, notches: Sequence[Notch], step: float = 0.05) -> bool:
    """Flood-fill the air region on a coarse grid. Cheap, and there is no other such check."""
    from aaf.sim.polygon_geom import polygon_contains
    v = verts_for_notches(L, W, notches)
    xs = np.arange(step / 2, L, step)
    ys = np.arange(step / 2, W, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    air = polygon_contains(v, np.stack([X, Y], axis=-1))
    if not air.any():
        return False
    seen = np.zeros_like(air)
    start = tuple(np.argwhere(air)[0])
    stack = [start]
    seen[start] = True
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < air.shape[0] and 0 <= b < air.shape[1] and air[a, b] and not seen[a, b]:
                seen[a, b] = True
                stack.append((a, b))
    return bool(seen.sum() == air.sum())


def _self_intersects(v: Sequence[Tuple[float, float]]) -> bool:
    """Non-adjacent edge pairs must not cross. O(n^2) over <= 10 vertices."""
    n = len(v)

    def _seg(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    for i in range(n):
        a1, a2 = v[i], v[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b1, b2 = v[j], v[(j + 1) % n]
            d1, d2 = _seg(b1, b2, a1), _seg(b1, b2, a2)
            d3, d4 = _seg(a1, a2, b1), _seg(a1, a2, b2)
            if ((d1 > EPS) != (d2 > EPS)) and ((d3 > EPS) != (d4 > EPS)):
                return True
    return False
