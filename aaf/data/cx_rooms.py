"""P4-7 m4 (D169): the configuration-diverse corpus extension CX500.

Every corpus so far fixes the source at (0.5, 1.6) and puts notches only at NW (the L corpora) or at
NW / NE / midN / SE (the family corpus; D165 p1). CX rooms break both:

  * bbox L in [5, 7], W in [4, 5.5], on the 0.02 m grid (the corpora's ranges, D67c quantisation);
  * 1 or 2 rectangular notches (p = 1/2 each) at any of the 4 corners or the 4 side midpoints;
  * the source uniform in air, >= 0.5 m from every wall and reflex corner (point-to-edge distance).

A notch at position p has depth d (normal to its wall) and width w (along it):
    corner (NW / NE / SW / SE): d along y, d <= 0.45 W; w along x, w <= 0.45 L  (the family's convention)
    midN / midS: d along y (<= 0.45 W), w along x (<= 0.45 L), x0 with >= 0.8 m shoulders
    midW / midE: d along x (<= 0.45 L), w along y (<= 0.45 W), y0 with >= 0.8 m shoulders
with d = u * d_max, u ~ U(0, 1), and w = v * w_max, v ~ U(0.15, 1), as `multi_notch._draw_family`.
d_hat = d / d_max.

Excluded by redraw (D169):
  * the U topology -- two corner notches on the same wall, in any orientation -- i.e. everything in the
    held-out U family's configuration and parameter ranges;
  * any notch with d_hat in the held-out slab [0.45, 0.60], and any CORNER notch whose transposed
    hat w / (0.45 L) is in it, so no room (nor any D4 image of one) enters either frozen test set's slab;
  * notch rectangles closer than MIN_PASSAGE_M (0.8 m) to each other;
  * a source that cannot be placed with the clearance (redraw the room).
The boundary polygon is traced from the cell complex of the notch edges (no geometry library in the env),
counter-clockwise, collinear vertices merged, in `augment_2d.canonical_ring` form.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from aaf.data.augment_2d import canonical_ring

L_RANGE = (5.0, 7.0)
W_RANGE = (4.0, 5.5)
FRAC_MAX = 0.45
W_HAT_LO = 0.15
MIN_STEM_M = 0.8
MIN_PASSAGE_M = 0.8
SRC_CLEAR_M = 0.5
SLAB = (0.45, 0.60)
ALPHA = 0.15
GRID = 0.02
CX_SEED = 20261016
CORNERS = ("NW", "NE", "SW", "SE")
MIDS = ("midN", "midS", "midW", "midE")
POSITIONS = CORNERS + MIDS
U_PAIRS = (frozenset({"NW", "NE"}), frozenset({"SW", "SE"}), frozenset({"NW", "SW"}), frozenset({"NE", "SE"}))


def q(x: float) -> float:
    """Quantise to the 0.02 m FDTD grid (D67c)."""
    return round(round(float(x) / GRID) * GRID, 6)


@dataclass(frozen=True)
class CxNotch:
    pos: str
    d: float
    w: float
    off: float = 0.0          # mids only: x0 (midN / midS) or y0 (midW / midE)

    def rect(self, L: float, W: float) -> Tuple[float, float, float, float]:
        """(x0, x1, y0, y1) of the solid rectangle."""
        p, d, w, o = self.pos, self.d, self.w, self.off
        return {"NW": (0.0, w, W - d, W), "NE": (L - w, L, W - d, W),
                "SW": (0.0, w, 0.0, d), "SE": (L - w, L, 0.0, d),
                "midN": (o, o + w, W - d, W), "midS": (o, o + w, 0.0, d),
                "midW": (0.0, d, o, o + w), "midE": (L - d, L, o, o + w)}[p]

    def d_max(self, L: float, W: float) -> float:
        return FRAC_MAX * (L if self.pos in ("midW", "midE") else W)

    def d_hat(self, L: float, W: float) -> float:
        return self.d / self.d_max(L, W)


def rect_gap(a, b) -> float:
    """Euclidean distance between two axis-aligned rectangles (0 if they touch or overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[1], b[1]))
    dy = max(0.0, max(a[2], b[2]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def trace_polygon(L: float, W: float, rects: Sequence[Tuple[float, float, float, float]]):
    """CCW boundary of [0, L] x [0, W] minus the rectangles, from the cell complex of their edges."""
    xs = sorted({0.0, L, *[r[0] for r in rects], *[r[1] for r in rects]})
    ys = sorted({0.0, W, *[r[2] for r in rects], *[r[3] for r in rects]})
    nx, ny = len(xs) - 1, len(ys) - 1

    def solid(i, j):
        if i < 0 or j < 0 or i >= nx or j >= ny:
            return True
        cx, cy = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])
        return any(r[0] < cx < r[1] and r[2] < cy < r[3] for r in rects)

    nxt = {}
    for i in range(nx):
        for j in range(ny):
            if solid(i, j):
                continue
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            for nb, a, b in (((i, j - 1), (x0, y0), (x1, y0)), ((i + 1, j), (x1, y0), (x1, y1)),
                             ((i, j + 1), (x1, y1), (x0, y1)), ((i - 1, j), (x0, y1), (x0, y0))):
                if solid(*nb):
                    if a in nxt:
                        raise ValueError("boundary vertex {} visited twice (pinch point)".format(a))
                    nxt[a] = b
    start = min(nxt, key=lambda p: (p[1], p[0]))
    ring, p = [start], nxt[start]
    while p != start:
        ring.append(p)
        p = nxt[p]
        if len(ring) > len(nxt):
            raise ValueError("boundary does not close")
    if len(ring) != len(nxt):
        raise ValueError("boundary has {} edges but the ring visits {}: disconnected air".format(
            len(nxt), len(ring)))
    out = []                                   # merge collinear vertices
    n = len(ring)
    for k in range(n):
        a, b, c = ring[k - 1], ring[k], ring[(k + 1) % n]
        if not ((a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1])):
            out.append(b)
    verts, _ = canonical_ring(out, [ALPHA] * len(out))
    return verts


def reflex_vertices(verts) -> List[Tuple[float, float]]:
    n = len(verts)
    out = []
    for k in range(n):
        a, b, c = verts[k - 1], verts[k], verts[(k + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross < 0:                          # a right turn on a CCW ring
            out.append(tuple(b))
    return out


def edge_distance(verts, pts) -> np.ndarray:
    """Distance from each point to the nearest polygon edge (walls, notch faces and their corners)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    best = np.full(len(p), np.inf)
    n = len(verts)
    for k in range(n):
        a = np.asarray(verts[k], float)
        b = np.asarray(verts[(k + 1) % n], float)
        ab = b - a
        t = np.clip(((p - a) @ ab) / float(ab @ ab), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return best


def topology(notches: Sequence[CxNotch]) -> str:
    pos = sorted(n.pos for n in notches)
    if len(pos) == 1:
        return "L" if pos[0] in CORNERS else "T"
    a, b = pos
    if a in CORNERS and b in CORNERS:
        return "U" if frozenset(pos) in U_PAIRS else "DN"
    if a in CORNERS or b in CORNERS:
        return "LT"
    return "TT"


@dataclass(frozen=True)
class CxRoom:
    L: float
    W: float
    notches: Tuple[CxNotch, ...]
    src: Tuple[float, float]
    shape_id: int
    split: str = "train"

    @property
    def rects(self):
        return [n.rect(self.L, self.W) for n in self.notches]

    @property
    def verts(self):
        return trace_polygon(self.L, self.W, self.rects)

    @property
    def kind(self) -> str:
        return "cx_" + topology(self.notches)

    @property
    def strata(self) -> str:
        return "cx{}".format(len(self.notches))

    @property
    def filename(self) -> str:
        return "CX{:04d}_L{:.2f}_W{:.2f}_{}.h5".format(
            self.shape_id, self.L, self.W, "+".join(sorted(n.pos for n in self.notches)))

    @property
    def label(self) -> str:
        return self.filename[:-3]

    def row(self, i: int) -> dict:
        v = self.verts
        return {"i": i, "split": self.split, "kind": self.kind, "shape": self.kind, "strata": self.strata,
                "shape_id": self.shape_id, "L": self.L, "W": self.W,
                "notches": [{"pos": n.pos, "d": n.d, "w": n.w, "off": n.off,
                             "d_hat": round(n.d_hat(self.L, self.W), 6)} for n in self.notches],
                "src": list(self.src), "verts": [list(p) for p in v],
                "edge_alphas": [ALPHA] * len(v), "alphas": [ALPHA] * 4,
                "filename": self.filename, "label": self.label}


def _draw_notch(rng, pos: str, L: float, W: float) -> Optional[CxNotch]:
    along_x = pos not in ("midW", "midE")
    d_max = FRAC_MAX * (W if along_x else L)
    w_max = FRAC_MAX * (L if along_x else W)
    d = q(rng.uniform(0.0, 1.0) * d_max)
    w = q(rng.uniform(W_HAT_LO, 1.0) * w_max)
    if d <= 0.0 or w <= 0.0:
        return None
    off = 0.0
    if pos in MIDS:
        span = (L if along_x else W) - w
        if span - MIN_STEM_M < MIN_STEM_M:
            return None
        off = q(rng.uniform(MIN_STEM_M, span - MIN_STEM_M))
        if off < MIN_STEM_M - 1e-9 or span - off < MIN_STEM_M - 1e-9:
            return None
    return CxNotch(pos, d, w, off)


def room_ok(L, W, notches: Sequence[CxNotch]) -> Tuple[bool, str]:
    if topology(notches) == "U":
        return False, "U topology"
    for n in notches:
        if SLAB[0] <= n.d_hat(L, W) <= SLAB[1]:
            return False, "d_hat in slab"
        if n.pos in CORNERS and SLAB[0] <= n.w / (FRAC_MAX * L) <= SLAB[1]:
            return False, "transposed hat in slab"
    rs = [n.rect(L, W) for n in notches]
    if len(rs) == 2 and rect_gap(rs[0], rs[1]) < MIN_PASSAGE_M - 1e-9:
        return False, "notches closer than {} m".format(MIN_PASSAGE_M)
    try:
        v = trace_polygon(L, W, rs)
    except ValueError as e:
        return False, str(e)
    xs, ys = [p[0] for p in v], [p[1] for p in v]
    if min(xs) != 0.0 or min(ys) != 0.0 or max(xs) != L or max(ys) != W:
        return False, "bbox collapsed"
    return True, "ok"


def draw_source(rng, verts, L, W, tries: int = 2000) -> Optional[Tuple[float, float]]:
    from aaf.sim.polygon_geom import polygon_contains
    for _ in range(tries):
        p = (q(rng.uniform(0.0, L)), q(rng.uniform(0.0, W)))
        if not bool(polygon_contains(verts, np.asarray([p]))[0]):
            continue
        if edge_distance(verts, [p])[0] >= SRC_CLEAR_M - 1e-9:
            return p
    return None


def sample_cx_rooms(n: int = 500, seed: int = CX_SEED, id0: int = 9000) -> List[CxRoom]:
    """n rooms, each from its own generator default_rng([seed, k]) so the list is prefix-stable."""
    out = []
    k = 0
    while len(out) < n:
        rng = np.random.default_rng([seed, k])
        k += 1
        L, W = q(rng.uniform(*L_RANGE)), q(rng.uniform(*W_RANGE))
        if rng.uniform() < 0.5:
            pos = [POSITIONS[int(rng.integers(len(POSITIONS)))]]
        else:
            pairs = [(a, b) for i, a in enumerate(POSITIONS) for b in POSITIONS[i + 1:]
                     if frozenset({a, b}) not in U_PAIRS]
            pos = list(pairs[int(rng.integers(len(pairs)))])
        ns = [_draw_notch(rng, p, L, W) for p in pos]
        if any(x is None for x in ns):
            continue
        ok, _why = room_ok(L, W, ns)
        if not ok:
            continue
        v = trace_polygon(L, W, [x.rect(L, W) for x in ns])
        src = draw_source(rng, v, L, W)
        if src is None:
            continue
        out.append(CxRoom(L, W, tuple(ns), src, shape_id=id0 + len(out)))
    return out
