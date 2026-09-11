"""P4-2 Stage 2: the NOTCH FAMILY -- one continuous family spanning rectangles and L-rooms.

A room is a base rectangle L x W with a rectangular notch of depth d and width w removed from
the TOP-LEFT corner (the same corner P4-1's single L-room used). d = w = 0 gives a rectangle;
d, w > 0 gives an L. That is why this is the *minimal honest* shape-edit claim: the degenerate
end of the family is the shape the model was already known to handle, so a failure cannot be
blamed on the family being exotic.

    (0,W)         (L,W)
      +-----+  . . . +          notch = [0,w] x [W-d, W], SOLID
      |notch|        |
   (0,W-d) +---------+ ...  reflex corner at (w, W-d)
      |    (w,W-d)   |
      +--------------+
    (0,0)         (L,0)

Vertices are emitted COUNTER-CLOCKWISE and ORIGIN-ANCHORED. Both matter:
  * CCW because `polygon_edge_tokens` computes the inward normal as (-dy, dx) and raises on a
    clockwise list -- a reversed list would invert every normal and the model would learn a room
    turned inside out.
  * origin-anchored because the trainer sets `room_min = zeros(2)` and `room_max = [L, W]`
    (multi_room_2d_mat), so a polygon whose bbox does not start at the origin gets an AABB
    shifted by its own offset and every ray terminates at the wrong distance, silently.

THE NORMALIZED DEPTH, AND WHY IT IS d / (0.45 W)
------------------------------------------------
The spec gives d in [0, 0.45 W] and a held-out slab at d_hat in [0.45, 0.60]. Under the other
obvious reading, d_hat = d / W, the sampled range is [0, 0.45] and the slab lies ENTIRELY
outside it -- no training shape would be excluded and no test shape would be inside, so the
experiment would not exist. d_hat = d / d_max is the only reading under which it does, and it
puts the slab interior with width 0.15 of the range. Frozen here as a constant (the `A_NORM`
precedent from the aperture axis) so the coordinate cannot move when the geometry does.

SAMPLING INTERVAL
-----------------
The spec cites P3-2d for "interval <= 0.2 of each parameter's normalized range". That citation
does not survive checking: Delta* ~ 0.275 is unratified (no DECISIONS entry;
`sampling_law.json` records `delta_star.point_estimate = null`), it rests on a rho definition
P3-2d explicitly escalated rather than decided, and it DOES NOT REPLICATE at a second seed --
G030 went rho 1.3083 (FAIL) at seed 1 to 1.1753 (PASS) at seed 2, and G030 seed 1 was the sole
evidence for the crossing. Worse, 0.2 x 1.59 = 0.318 in m is *exactly* the G030 arm. The arm
that passes under both rho definitions at both seeds is G020 = 0.125 of the normalized range,
so that is the target used here. P3-2d also measured ONE axis (absorption) in its own
linearizing coordinate; nothing establishes transfer to geometry parameters.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------- frozen family spec
L_RANGE = (5.0, 7.0)
W_RANGE = (4.0, 5.5)
D_FRAC_MAX = 0.45           # d_max = D_FRAC_MAX * W
W_FRAC_MAX = 0.45           # w_max = W_FRAC_MAX * L
ALPHA = 0.15                # fixed everywhere: this chunk isolates SHAPE

#: Held-out band in normalized notch depth. Interior and non-extrapolative (the P3-2 lesson:
#: P3-2's north@0.70 hold-out was an extrapolation on its own axis and was never comparable to
#: the west@0.50 interpolation).
D_HAT_HOLDOUT = (0.45, 0.60)

#: Minimum clear passage. VACUOUS at these ranges -- min passage is 0.55 * W_min = 2.20 m, so
#: this never rejects. Implemented anyway because it becomes live the moment the ranges move,
#: and reported as never-firing rather than silently assumed.
MIN_PASSAGE_M = 0.8

N_TRAIN = 60
N_TEST = 15
N_TEST_IN_SLAB = 6          # placed deliberately: uniform sampling gives only ~2 of 15
MIN_RECTANGLES = 8          # anchor the degenerate end of the family
SEED = 20260911
TEST_SEED = 20260912
GEOM_QUANT_DP = 2           # dims quantized to the dp the filename encodes, at generation time
MAX_GAP_FRAC = 0.125        # see SAMPLING INTERVAL above

#: Fixed source, per spec. Must satisfy sy < min(W - d) = 0.55 * W_min = 2.20 over the whole
#: family or it would fall inside the notch of the tightest shapes; 1.6 leaves 0.6 m clearance.
#: Swept during planning: this placement gives ~3% NLOS at the slab centre and ~25% at maximum
#: notch. The slab-centre figure is thin, and that is a property of the family (shallow notches
#: cast little shadow), not of the placement -- reported with its receiver count rather than
#: quietly averaged in.
SRC = (0.5, 1.6)


def d_max_for(W: float) -> float:
    return D_FRAC_MAX * float(W)


def w_max_for(L: float) -> float:
    return W_FRAC_MAX * float(L)


def d_hat(d: float, W: float) -> float:
    return float(d) / d_max_for(W)


def w_hat(w: float, L: float) -> float:
    return float(w) / w_max_for(L)


def in_holdout(dh: float) -> bool:
    """Closed on both ends with an epsilon, matching the aperture axis's `in_holdout`."""
    return D_HAT_HOLDOUT[0] - 1e-12 <= float(dh) <= D_HAT_HOLDOUT[1] + 1e-12


def passage_ok(L: float, W: float, d: float, w: float) -> bool:
    if d <= 0.0 or w <= 0.0:
        return True
    return (W - d) >= MIN_PASSAGE_M and (L - w) >= MIN_PASSAGE_M


def verts_for(L: float, W: float, d: float, w: float) -> List[Tuple[float, float]]:
    """CCW, origin-anchored, no repeated first vertex. 4 vertices if there is no notch."""
    L, W, d, w = float(L), float(W), float(d), float(w)
    if d <= 0.0 or w <= 0.0:
        return [(0.0, 0.0), (L, 0.0), (L, W), (0.0, W)]
    return [(0.0, 0.0), (L, 0.0), (L, W), (w, W), (w, W - d), (0.0, W - d)]


def shape_filename(L: float, W: float, d: float, w: float) -> str:
    """DERIVED from the geometry, never stored loose.

    `PolyConfig.filename` is a plain stored string, so nothing structural stops two shapes
    aliasing onto one .h5 -- and the trainer would then train one room on another's field with
    no error. Uniqueness is asserted at manifest-freeze time on top of this.
    """
    return "L{:.2f}_W{:.2f}_d{:.2f}_w{:.2f}.h5".format(L, W, d, w)


def shape_label(L: float, W: float, d: float, w: float) -> str:
    """UNIQUE per shape. `PolyConfig.label` is "L[6v]" for EVERY 6-edge room, which makes
    `train_meta.json`'s config_labels non-identifying and silently collapses any downstream
    tool that keys on label."""
    kind = "rect" if (d <= 0.0 or w <= 0.0) else "L"
    return "{}_L{:.2f}_W{:.2f}_d{:.2f}_w{:.2f}".format(kind, L, W, d, w)


def strata_for(L: float, W: float, d: float, w: float) -> str:
    """Coarse key for the trainer's validation stratification.

    Deliberately coarse, for the reason `mat_configs_cont` records: keying on the exact
    parameters gives one singleton stratum per config, and the val subsample then degenerates
    onto the first few rooms in manifest order.
    """
    if d <= 0.0 or w <= 0.0:
        return "rect"
    dh = d_hat(d, W)
    return "L_d{}".format(min(3, int(dh * 4)))      # 4 depth buckets


# ------------------------------------------------------------------------------ the config
@dataclass(frozen=True)
class ShapeConfig:
    L: float
    W: float
    d: float
    w: float
    split: str = "train"
    shape_id: int = 0

    @property
    def is_rect(self) -> bool:
        return self.d <= 0.0 or self.w <= 0.0

    @property
    def verts(self) -> List[Tuple[float, float]]:
        return verts_for(self.L, self.W, self.d, self.w)

    @property
    def edge_alphas(self) -> Tuple[float, ...]:
        return tuple([ALPHA] * len(self.verts))

    @property
    def alphas(self) -> List[float]:
        """4-vector view for the trainer's manifest/data drift check. Uniform alpha here, so it
        is exact rather than lossy -- but it verifies nothing about the notch, which is why the
        dataset gate carries a simulator-level item instead."""
        return [ALPHA] * 4

    @property
    def d_hat(self) -> float:
        return d_hat(self.d, self.W)

    @property
    def w_hat(self) -> float:
        return w_hat(self.w, self.L)

    @property
    def filename(self) -> str:
        return shape_filename(self.L, self.W, self.d, self.w)

    @property
    def label(self) -> str:
        return shape_label(self.L, self.W, self.d, self.w)

    @property
    def name(self) -> str:
        return self.label

    @property
    def strata(self) -> str:
        return strata_for(self.L, self.W, self.d, self.w)

    @property
    def kind(self) -> str:
        return "rect" if self.is_rect else "L"

    def row(self, i: int) -> dict:
        return {"i": i, "split": self.split, "kind": self.kind, "shape_id": self.shape_id,
                "L": self.L, "W": self.W, "d": self.d, "w": self.w,
                "d_hat": round(self.d_hat, 6), "w_hat": round(self.w_hat, 6),
                "verts": [list(v) for v in self.verts],
                "edge_alphas": list(self.edge_alphas), "alphas": list(self.alphas),
                "filename": self.filename, "label": self.label, "strata": self.strata,
                "in_holdout": bool(in_holdout(self.d_hat))}


def configs_from_rows(rows: Sequence[dict], split: Optional[str] = None,
                      kinds: Sequence[str] = ()) -> List[ShapeConfig]:
    """Signature matches every sibling `configs_from_rows` verbatim so the trainer's schema
    dispatch can swap it in with no other change."""
    out = []
    for r in rows:
        if split is not None and r.get("split") != split:
            continue
        if kinds and r.get("kind") not in kinds:
            continue
        out.append(ShapeConfig(L=float(r["L"]), W=float(r["W"]), d=float(r["d"]),
                               w=float(r["w"]), split=str(r.get("split", "train")),
                               shape_id=int(r.get("shape_id", 0))))
    return out


# ----------------------------------------------------------------------------- the sampler
def _q(x: float) -> float:
    return round(float(x), GEOM_QUANT_DP)


def _draw(rng, want_rect: bool, force_dhat: Optional[Tuple[float, float]] = None):
    """One shape. Redraws until the hold-out band and the passage rule are both satisfied."""
    for _ in range(10000):
        L = _q(rng.uniform(*L_RANGE))
        W = _q(rng.uniform(*W_RANGE))
        if want_rect:
            return L, W, 0.0, 0.0
        lo, hi = force_dhat if force_dhat else (0.0, 1.0)
        dh = float(rng.uniform(lo, hi))
        wh = float(rng.uniform(0.15, 1.0))          # a sliver-thin notch is not a shape edit
        d, w = _q(dh * d_max_for(W)), _q(wh * w_max_for(L))
        if d <= 0.0 or w <= 0.0:
            continue
        if force_dhat is None and in_holdout(d_hat(d, W)):
            continue                                 # the hold-out band, enforced by redraw
        if force_dhat is not None and not in_holdout(d_hat(d, W)):
            continue                                 # quantization can push it out of the band
        if not passage_ok(L, W, d, w):
            continue
        return L, W, d, w
    raise RuntimeError("rejection sampling failed (want_rect={}, force_dhat={})".format(
        want_rect, force_dhat))


def sample_train_shapes(n: int = N_TRAIN, n_rect: int = MIN_RECTANGLES,
                        seed: int = SEED) -> List[ShapeConfig]:
    """RNG keyed on (seed, shape_id) rather than a sequential stream, so every SLURM array task
    rebuilds an identical manifest and adding a shape cannot perturb the existing draws."""
    out, used = [], set()
    for i in range(n):
        for attempt in range(64):
            rng = np.random.default_rng([seed, i, attempt])
            L, W, d, w = _draw(rng, want_rect=(i < n_rect))
            fn = shape_filename(L, W, d, w)
            if fn not in used:
                used.add(fn)
                out.append(ShapeConfig(L, W, d, w, split="train", shape_id=i))
                break
        else:
            raise RuntimeError("could not find a unique shape for slot {}".format(i))
    return out


def sample_test_shapes(train: Sequence[ShapeConfig], n: int = N_TEST,
                       n_in_slab: int = N_TEST_IN_SLAB,
                       seed: int = TEST_SEED) -> List[ShapeConfig]:
    """Maximin-spread test shapes, strictly interior to the training hull in (L, W, w_hat).

    d_hat is handled separately: `n_in_slab` shapes are placed INSIDE the held-out band (the
    headline generalization case) and the rest outside it. They must be placed rather than
    sampled -- the band is 15% of the range, so uniform sampling would put only ~2 of 15 inside
    and the in-slab mean would be too thin to read.

    Interiority is a per-axis margin rather than a Delaunay hull test: in 4-D with 60 points a
    hull test is expensive and degenerates easily, and the margin form is what actually matters
    here (no test shape sits at the edge of the sampled box).
    """
    tr = np.array([[c.L, c.W, c.w_hat] for c in train], dtype=float)
    lo, hi = tr.min(axis=0), tr.max(axis=0)
    pad = 0.08 * (hi - lo)                               # strictly interior
    ilo, ihi = lo + pad, hi - pad

    out, used = [], {c.filename for c in train}
    for k in range(n):
        want_slab = k < n_in_slab
        for attempt in range(4000):
            rng = np.random.default_rng([seed, k, attempt])
            L = _q(rng.uniform(ilo[0], ihi[0]))
            W = _q(rng.uniform(ilo[1], ihi[1]))
            wh = float(rng.uniform(max(0.15, ilo[2]), ihi[2]))
            dh = (float(rng.uniform(*D_HAT_HOLDOUT)) if want_slab
                  else float(rng.uniform(0.05, D_HAT_HOLDOUT[0] - 0.05))
                  if k % 2 else float(rng.uniform(D_HAT_HOLDOUT[1] + 0.05, 0.98)))
            d, w = _q(dh * d_max_for(W)), _q(wh * w_max_for(L))
            if d <= 0 or w <= 0 or not passage_ok(L, W, d, w):
                continue
            if in_holdout(d_hat(d, W)) != want_slab:
                continue                                  # quantization moved it across
            fn = shape_filename(L, W, d, w)
            if fn in used:
                continue
            # maximin: keep it away from every training shape and every test shape so far
            pt = np.array([L, W, w_hat(w, L)])
            span = np.maximum(hi - lo, 1e-9)
            dist = np.min(np.max(np.abs((tr - pt) / span), axis=1))
            if dist < 0.04:
                continue
            used.add(fn)
            out.append(ShapeConfig(L, W, d, w, split="test", shape_id=1000 + k))
            break
        else:
            raise RuntimeError("could not place test shape {} (want_slab={})".format(k, want_slab))
    return out


# ------------------------------------------------------------------------------- invariants
def max_normalized_gap(vals: Sequence[float], lo: float, hi: float,
                       exclude: Sequence[Tuple[float, float]] = ()) -> float:
    """Largest gap between consecutive sampled values, as a fraction of ``[lo, hi]``.

    ``exclude`` lists intervals the design DELIBERATELY does not sample -- the held-out slab in
    d_hat, and the sliver region below w_hat = 0.15. Without it this metric reports a design
    choice as under-sampling: measured on the real draws, d_hat's largest raw gap is 0.166,
    which is exactly the 0.15-wide hold-out band plus quantization, and w_hat's is 0.175, which
    is exactly the excluded sliver region. Neither is a coverage defect, and reporting them as
    one would either fail a healthy manifest or invite quietly loosening the threshold.

    End gaps are included (a range sampled only in its middle IS under-sampled), except where
    an exclusion covers them.
    """
    if not len(vals):
        return 1.0
    span = max(hi - lo, 1e-12)
    u = np.sort((np.asarray(vals, float) - lo) / span)
    ex = [((a - lo) / span, (b - lo) / span) for a, b in exclude]
    edges = np.concatenate([[0.0], u, [1.0]])
    worst = 0.0
    for g0, g1 in zip(edges[:-1], edges[1:]):
        gap = g1 - g0
        # subtract any deliberately-unsampled region lying inside this gap
        for a, b in ex:
            gap -= max(0.0, min(g1, b) - max(g0, a))
        worst = max(worst, gap)
    return float(worst)


def assert_invariants(train: Sequence[ShapeConfig], test: Sequence[ShapeConfig]) -> Dict:
    """Freeze-time gate. Raises rather than warns -- a broken manifest must never reach the
    cluster, and every one of these failures is silent downstream."""
    bad = [c.label for c in train if in_holdout(c.d_hat)]
    assert not bad, "training shapes inside the held-out band: {}".format(bad[:5])
    n_slab = sum(1 for c in test if in_holdout(c.d_hat))
    assert n_slab >= N_TEST_IN_SLAB, "expected >= {} in-slab test shapes, got {}".format(
        N_TEST_IN_SLAB, n_slab)
    names = [c.filename for c in list(train) + list(test)]
    assert len(set(names)) == len(names), "duplicate filenames in the manifest"
    labels = [c.label for c in list(train) + list(test)]
    assert len(set(labels)) == len(labels), "duplicate labels in the manifest"
    n_rect = sum(1 for c in train if c.is_rect)
    assert n_rect >= MIN_RECTANGLES, "only {} rectangles, need >= {}".format(
        n_rect, MIN_RECTANGLES)
    for c in list(train) + list(test):
        assert passage_ok(c.L, c.W, c.d, c.w), "passage rule violated by {}".format(c.label)
        v = c.verts
        assert len(v) in (4, 6), "{} has {} vertices".format(c.label, len(v))
        assert min(p[0] for p in v) == 0.0 and min(p[1] for p in v) == 0.0, \
            "{} is not origin-anchored".format(c.label)
    notched = [c for c in train if not c.is_rect]
    gaps = {
        "L": max_normalized_gap([c.L for c in train], *L_RANGE),
        "W": max_normalized_gap([c.W for c in train], *W_RANGE),
        # the hold-out band is a DELIBERATE hole, not a coverage defect
        "d_hat": max_normalized_gap([c.d_hat for c in notched], 0.0, 1.0,
                                    exclude=[D_HAT_HOLDOUT]),
        # w_hat below 0.15 is deliberately unsampled (a sliver is not a shape edit)
        "w_hat": max_normalized_gap([c.w_hat for c in notched], 0.0, 1.0,
                                    exclude=[(0.0, 0.15)]),
    }
    too_coarse = {k: v for k, v in gaps.items() if v > MAX_GAP_FRAC}
    assert not too_coarse, "sampling too coarse (target {}): {}".format(MAX_GAP_FRAC, too_coarse)
    return {"n_train": len(train), "n_test": len(test), "n_rect": n_rect,
            "n_test_in_holdout": n_slab, "n_train_in_holdout": 0,
            "max_normalized_gap": gaps, "max_gap_target": MAX_GAP_FRAC,
            "gap_exclusions": {"d_hat": [list(D_HAT_HOLDOUT)], "w_hat": [[0.0, 0.15]]},
            "gap_note": ("Gaps are measured EXCLUDING deliberately-unsampled regions. Raw "
                         "(unexcluded) d_hat and w_hat gaps are ~0.166 and ~0.175, which are "
                         "exactly the hold-out band and the sliver cutoff -- design choices, "
                         "not coverage defects."),
            "gap_ok": {k: bool(v <= MAX_GAP_FRAC) for k, v in gaps.items()},
            "passage_rule_ever_fired": False}


def manifest_rows(train: Sequence[ShapeConfig], test: Sequence[ShapeConfig]) -> List[dict]:
    return [c.row(i) for i, c in enumerate(list(train) + list(test))]


def rows_sha256(rows: Sequence[dict]) -> str:
    return hashlib.sha256(json.dumps(list(rows), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
