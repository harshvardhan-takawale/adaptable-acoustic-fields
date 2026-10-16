"""P4-7 prep p4 (D164): geometric augmentations for the 2D room-acoustics corpora, as PURE functions.

The dihedral group D4 of the room's bounding box acts on everything at once -- receivers, source,
boundary polygon, wall alphas -- and leaves the scalar pressure field unchanged:
    H_g(g x_r; g x_s, g room) = H(x_r; x_s, room).
Every element is implemented as select / negate / offset on [0, L] x [0, W], re-anchored at the origin, so
the identity is an exact copy and the bit-exact FDTD check of `tests/test_augment_2d.py` is meaningful.

    id  name   (x', y')          (L', W')   det   new walls (W', E', S', N') <- old
    0   e      (x, y)            (L, W)     +1    (W, E, S, N)
    1   r90    (W - y, x)        (W, L)     +1    (N, S, W, E)
    2   r180   (L - x, W - y)    (L, W)     +1    (E, W, N, S)
    3   r270   (y, L - x)        (W, L)     +1    (S, N, E, W)
    4   fx     (L - x, y)        (L, W)     -1    (E, W, S, N)
    5   fy     (x, W - y)        (L, W)     -1    (W, E, N, S)
    6   tr     (y, x)            (W, L)     -1    (S, N, W, E)
    7   atr    (W - y, L - x)    (W, L)     -1    (N, S, E, W)

Polygons stay counter-clockwise (reversed after a det = -1 map, with edge alphas carried by their
edges) and are put in CANONICAL form: the ring starts at the lexicographic minimum of (y, x). That rule is
the identity on every existing room (all start at (0, 0), the SW corner, which no family notches), so
native conditioning vectors are unchanged and every image gets the same slot semantics for the DC head.

The TRANSPOSING elements (r90, r270, tr, atr) turn a notch's width into its depth, so they can carry
rooms into a corpus's HELD-OUT d_hat slab (`transpose_holdout_leaks`); the flips fx, fy, r180 cannot.
Reciprocity (swap source and receiver) needs no geometry here; `is_interior_node` flags wall nodes,
where the FDTD's soft source and receiver are not symmetric (factor 2, D65).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

D4_ELEMS = ("e", "r90", "r180", "r270", "fx", "fy", "tr", "atr")
TRANSPOSING = frozenset({"r90", "r270", "tr", "atr"})
_DET = {"e": 1, "r90": 1, "r180": 1, "r270": 1, "fx": -1, "fy": -1, "tr": -1, "atr": -1}
_INV = {"e": "e", "r90": "r270", "r180": "r180", "r270": "r90", "fx": "fx", "fy": "fy", "tr": "tr", "atr": "atr"}
# new (W, E, S, N) wall <- index of the old wall in (W, E, S, N)
_WALLS = {"e": (0, 1, 2, 3), "r90": (3, 2, 0, 1), "r180": (1, 0, 3, 2), "r270": (2, 3, 1, 0),
          "fx": (1, 0, 2, 3), "fy": (0, 1, 3, 2), "tr": (2, 3, 0, 1), "atr": (3, 2, 1, 0)}
# linear parts (for direction vectors such as inward normals)
_LIN = {"e": ((1, 0), (0, 1)), "r90": ((0, -1), (1, 0)), "r180": ((-1, 0), (0, -1)),
        "r270": ((0, 1), (-1, 0)), "fx": ((-1, 0), (0, 1)), "fy": ((1, 0), (0, -1)),
        "tr": ((0, 1), (1, 0)), "atr": ((0, -1), (-1, 0))}


def _check(g: str) -> None:
    if g not in _DET:
        raise ValueError("unknown D4 element {!r}; expected one of {}".format(g, D4_ELEMS))


def d4_det(g: str) -> int:
    _check(g)
    return _DET[g]


def d4_inverse(g: str) -> str:
    _check(g)
    return _INV[g]


def d4_dims(g: str, L: float, W: float) -> Tuple[float, float]:
    _check(g)
    return (W, L) if g in TRANSPOSING else (L, W)


def d4_points(g: str, pts, L: float, W: float) -> np.ndarray:
    """Map points (..., 2) of [0, L] x [0, W] into the re-anchored image frame [0, L'] x [0, W']."""
    _check(g)
    p = np.asarray(pts, dtype=np.float64)
    if g == "e":
        return p.copy()
    x, y = p[..., 0], p[..., 1]
    q = {"r90": (W - y, x), "r180": (L - x, W - y), "r270": (y, L - x), "fx": (L - x, y),
         "fy": (x, W - y), "tr": (y, x), "atr": (W - y, L - x)}[g]
    return np.stack(q, axis=-1)


def d4_vectors(g: str, v) -> np.ndarray:
    """The linear part of the map applied to direction vectors (..., 2)."""
    _check(g)
    A = np.asarray(_LIN[g], dtype=np.float64)
    return np.asarray(v, dtype=np.float64) @ A.T


def d4_wall_alphas(g: str, alphas_wesn: Sequence[float]) -> List[float]:
    """Wall alphas in WALLS_2D order (west, east, south, north) for the image room."""
    _check(g)
    a = list(alphas_wesn)
    if len(a) != 4:
        raise ValueError("expected 4 wall alphas (W, E, S, N), got {}".format(len(a)))
    return [a[i] for i in _WALLS[g]]


def canonical_ring(verts, alphas) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Rotate a CCW ring to start at the lexicographic minimum of (y, x); edge alphas move with
    their edges (edge k runs verts[k] -> verts[k+1])."""
    v = [(float(p[0]), float(p[1])) for p in verts]
    n = len(v)
    area2 = sum(v[i][0] * v[(i + 1) % n][1] - v[(i + 1) % n][0] * v[i][1] for i in range(n))
    if area2 <= 0.0:
        raise ValueError("canonical_ring needs a counter-clockwise ring (2A = {:.6g})".format(area2))
    s = min(range(n), key=lambda i: (v[i][1], v[i][0]))
    return [v[(k + s) % n] for k in range(n)], [float(alphas[(k + s) % n]) for k in range(n)]


def d4_polygon(g: str, verts, edge_alphas, L: float, W: float):
    """Image of a CCW origin-anchored polygon: (verts', alphas', perm) with alphas'[k] = alphas[perm[k]].

    `g == "e"` returns the INPUT objects unchanged. Otherwise every vertex is mapped; for det = -1
    the order is reversed (u_k = T(v_{-k}), alpha'_k = alpha_{(-k-1) mod n}); then `canonical_ring`.
    Coordinates are rounded to 1e-12 m to remove float noise from L - x."""
    _check(g)
    n = len(verts)
    if g == "e":
        return verts, edge_alphas, list(range(n))
    T = np.round(d4_points(g, np.asarray(verts, dtype=np.float64), L, W), 12)
    if _DET[g] > 0:
        u = [tuple(T[i]) for i in range(n)]
        perm = list(range(n))
    else:
        u = [tuple(T[(-k) % n]) for k in range(n)]
        perm = [(-k - 1) % n for k in range(n)]
    a = [float(edge_alphas[perm[k]]) for k in range(n)]
    v2, _ = canonical_ring(u, a)
    s = [tuple(x) for x in u].index(v2[0])
    perm = [perm[(k + s) % n] for k in range(n)]
    return v2, [float(edge_alphas[perm[k]]) for k in range(n)], perm


def is_interior_node(verts, pts, dx_x: float, dx_y: float) -> np.ndarray:
    """True where all four grid neighbours of a node are in the room (polygon_contains, edges count
    as inside), i.e. the node is not an FDTD wall node; reciprocity is exact only between such nodes."""
    from aaf.sim.polygon_geom import polygon_contains
    p = np.asarray(pts, dtype=np.float64)
    ok = np.ones(len(p), dtype=bool)
    for d in ((dx_x, 0.0), (-dx_x, 0.0), (0.0, dx_y), (0.0, -dx_y)):
        ok &= polygon_contains(verts, p + np.asarray(d))
    return ok


def transpose_holdout_leaks(configs, slab: Tuple[float, float]) -> List[int]:
    """Indices of configs that a TRANSPOSING element would carry into the held-out d_hat slab: the
    image of a notch of width w in a room of length L has depth w in a room of height L, so its
    d_hat is w / (0.45 L) (ShapeConfig: (d, w); FamilyConfig: any notch)."""
    lo, hi = float(slab[0]), float(slab[1])
    out = []
    for i, c in enumerate(configs):
        if hasattr(c, "notches"):
            ws = [n.w for n in c.notches]
        elif getattr(c, "d", 0.0) > 0 and getattr(c, "w", 0.0) > 0:
            ws = [c.w]
        else:
            ws = []
        if any(lo <= w / (0.45 * c.L) <= hi for w in ws):
            out.append(i)
    return out


def precompute_d4_tables(configs, rx: np.ndarray, src: Sequence[float], cond_source: str,
                         elems: Sequence[str]) -> Dict[str, np.ndarray]:
    """Per (config, element): receivers [C, G, n_rx, 2], source [C, G, 2], dims [C, G, 2] and the
    conditioning vector [C, G, D], the cond built with the trainer's own preload expression on the
    transformed geometry. Element index 0 must be the identity."""
    from aaf.models.conditioning_2d import build_cond_vector_2d
    if not elems or elems[0] != "e":
        raise ValueError("elems[0] must be 'e' (the identity), got {}".format(list(elems)))
    C, G = len(configs), len(elems)
    rx = np.asarray(rx, dtype=np.float64)
    out_rx = np.zeros((C, G) + rx.shape[1:], dtype=np.float32)
    out_src = np.zeros((C, G, 2), dtype=np.float32)
    out_dims = np.zeros((C, G, 2), dtype=np.float32)
    conds = []
    for i, c in enumerate(configs):
        row = []
        verts = getattr(c, "verts", None)
        for j, g in enumerate(elems):
            Lg, Wg = d4_dims(g, c.L, c.W)
            out_rx[i, j] = d4_points(g, rx[i], c.L, c.W).astype(np.float32)
            out_src[i, j] = d4_points(g, np.asarray(src, dtype=np.float64), c.L, c.W).astype(np.float32)
            out_dims[i, j] = (Lg, Wg)
            if verts is not None:
                al = getattr(c, "edge_alphas", None)
                vg, ag, _ = d4_polygon(g, verts, al, c.L, c.W)
                cond = build_cond_vector_2d(cond_source, Lg, Wg, ag, x0=getattr(c, "x0", None),
                                            a=getattr(c, "a", None), verts=vg)
            else:
                cond = build_cond_vector_2d(cond_source, Lg, Wg, d4_wall_alphas(g, c.alphas),
                                            x0=getattr(c, "x0", None), a=getattr(c, "a", None), verts=None)
            row.append(np.asarray(cond.detach().cpu() if hasattr(cond, "detach") else cond, dtype=np.float32))
        conds.append(np.stack(row))
    return {"rx": out_rx, "src": out_src, "dims": out_dims, "cond": np.stack(conds), "elems": np.asarray(elems)}
