"""P3-3-FAST Track 1: per-segment absorption configs (16 segments) + the window value.

Each wall is divided into 4 equal segments, so a room carries **16** absorptions rather than
P3-2b's 4. Two things about that are load-bearing:

**Canonical segment order.** ``(west_1..4, east_1..4, south_1..4, north_1..4)``, each indexed
along INCREASING coordinate (y for west/east, x for south/north). This is the same class of
convention as P3-2's wall order, with four times the surface area to get wrong, and a wrong
index here trains happily and encodes the wrong physics. ``segment_index`` /
``index_to_segment`` are the single source of truth and are asserted end-to-end in
``tests/test_seg_configs.py`` from manifest row through to ``geom.face_alpha``.

**The absorption range is extended to alpha = 0.95, and that is the window.** m = -ln(1-alpha)
is drawn over [0.02, 3.0] rather than P3-2b's [0.02, 1.61]. A segment at alpha = 0.95 is very
nearly matched-impedance, which is the classical open-window model and is equivalent to a
first-order absorbing boundary. Its limitation is real and is stated wherever the window result
appears: it carries no radiation reactance and no edge diffraction, so it models the energy
loss of an opening but not the reactive near-field or the diffraction at its rim.

Because m_max = 3.0 here and 1.6094 in P3-2b, the normalized conditioning coordinate differs
between the two chunks, and a model trained here is NOT numerically comparable to a P3-2b
model. Comparisons are qualitative only.

Filenames hash the 16-vector: sixteen 6-dp tokens would produce a ~200-character name. The
manifest carries the full vector and the dataset gate asserts hash <-> vector injectivity, so
the P3-2d collision hazard (a discrete alphabet silently sharing one HDF5 file between two
configs) cannot recur unnoticed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aaf.walls import ALPHA_BASELINE, WALLS_2D

N_SEG_PER_WALL = 4
N_SEG = len(WALLS_2D) * N_SEG_PER_WALL          # 16
M_RANGE_SEG: Tuple[float, float] = (0.02, 3.0)  # alpha in [0.0198, 0.9502]
M_NORM_SEG = 3.0
ALPHA_QUANT_DP = 6

WINDOW_ALPHA_RANGE: Tuple[float, float] = (0.93, 0.95)
WINDOW_ALPHA = 0.95
TEST_PATCH_ALPHA = 0.70

HOLDOUT_SEGMENT = ("east", 3)   # 1-based within the wall; excluded from ALL training configs


def m_of_alpha(alpha: float) -> float:
    return float(-np.log1p(-float(alpha)))


def alpha_of_m(m: float) -> float:
    return float(1.0 - np.exp(-float(m)))


def segment_index(wall: str, k: int) -> int:
    """Canonical flat index for segment ``k`` (1-based) of ``wall``."""
    if wall not in WALLS_2D:
        raise ValueError("unknown wall {!r}; expected one of {}".format(wall, list(WALLS_2D)))
    if not 1 <= k <= N_SEG_PER_WALL:
        raise ValueError("segment must be 1..{}, got {}".format(N_SEG_PER_WALL, k))
    return WALLS_2D.index(wall) * N_SEG_PER_WALL + (k - 1)


def index_to_segment(i: int) -> Tuple[str, int]:
    if not 0 <= i < N_SEG:
        raise ValueError("segment index must be 0..{}, got {}".format(N_SEG - 1, i))
    return WALLS_2D[i // N_SEG_PER_WALL], (i % N_SEG_PER_WALL) + 1


SEGMENT_NAMES: Tuple[str, ...] = tuple(
    "{}_{}".format(*index_to_segment(i)) for i in range(N_SEG))


def seg_alphas_to_wall_specs(alphas: Sequence[float]) -> List[dict]:
    """16 segment absorptions -> the solver's ``wall_segments`` specs, one per wall.

    Uses the PARTITION primitive, not four independent patches: patches optimize each span's
    realized extent on its own and therefore do not tile (measured: 10 of 16 segments off by up
    to dx/2, with boundary nodes left at the baseline).
    """
    a = [float(x) for x in alphas]
    if len(a) != N_SEG:
        raise ValueError("expected {} segment alphas, got {}".format(N_SEG, len(a)))
    out = []
    for w in WALLS_2D:
        i0 = WALLS_2D.index(w) * N_SEG_PER_WALL
        out.append({"type": "wall_segments", "wall": w,
                    "alphas": a[i0:i0 + N_SEG_PER_WALL]})
    return out


def seg_filename(L: float, W: float, alphas: Sequence[float]) -> str:
    a = [round(float(x), ALPHA_QUANT_DP) for x in alphas]
    h = hashlib.sha1(",".join("{:.6f}".format(x) for x in a).encode()).hexdigest()[:12]
    return "L{:.2f}_W{:.2f}_seg{}.h5".format(round(float(L), 2), round(float(W), 2), h)


@dataclass(frozen=True)
class SegConfig:
    L: float
    W: float
    alphas: Tuple[float, ...]      # 16, canonical order
    kind: str
    split: str
    geom_id: int

    @property
    def is_baseline(self) -> bool:
        return all(abs(x - ALPHA_BASELINE) <= 1e-12 for x in self.alphas)

    @property
    def edited(self) -> Tuple[str, ...]:
        return tuple(SEGMENT_NAMES[i] for i, x in enumerate(self.alphas)
                     if abs(x - ALPHA_BASELINE) > 1e-9)

    @property
    def filename(self) -> str:
        return seg_filename(self.L, self.W, self.alphas)

    @property
    def m(self) -> Tuple[float, ...]:
        return tuple(m_of_alpha(x) for x in self.alphas)

    @property
    def label(self) -> str:
        if self.is_baseline:
            return "L{:.2f}_W{:.2f}_baseline".format(self.L, self.W)
        return "L{:.2f}_W{:.2f}_{}".format(self.L, self.W, "+".join(self.edited[:3]))

    @property
    def strata(self) -> str:
        """Coarse validation stratum -- the P3-2b lesson: keying on the exact edited set would
        make every continuous config its own singleton group and collapse the val subsample."""
        return "baseline" if self.is_baseline else self.kind

    @property
    def wall(self):
        e = self.edited
        return e[0].split("_")[0] if len(e) == 1 else None

    @property
    def material(self):
        return None if self.is_baseline else "+".join(self.edited)

    @property
    def touches_window(self) -> bool:
        return any(x >= WINDOW_ALPHA_RANGE[0] - 1e-9 for x in self.alphas)


def _mk(L, W, edits: Dict[int, float], kind: str, split: str, gid: int) -> SegConfig:
    a = [ALPHA_BASELINE] * N_SEG
    for i, v in edits.items():
        a[i] = round(float(v), ALPHA_QUANT_DP)
    return SegConfig(L=round(float(L), 2), W=round(float(W), 2), alphas=tuple(a),
                     kind=kind, split=split, geom_id=int(gid))


def _draw_m(rng) -> float:
    return round(alpha_of_m(float(rng.uniform(*M_RANGE_SEG))), ALPHA_QUANT_DP)


def _holdout_hit(edits: Dict[int, float]) -> bool:
    """Does this config touch the held-out segment? Those are excluded from TRAINING."""
    return segment_index(*HOLDOUT_SEGMENT) in edits


def sample_train_configs(geoms, seed: int = 20260816) -> List[SegConfig]:
    """20 configs per geometry. Any config touching east_3 is redrawn onto another segment,
    so the hold-out is exact rather than approximate."""
    out: List[SegConfig] = []
    hold = segment_index(*HOLDOUT_SEGMENT)
    other = [i for i in range(N_SEG) if i != hold]
    for gid, (L, W) in enumerate(geoms):
        out.append(_mk(L, W, {}, "baseline", "train", gid))
        for j in range(5):                                   # uniform-wall
            rng = np.random.default_rng([seed, gid, 100 + j])
            w = WALLS_2D[(gid + j) % len(WALLS_2D)]
            v = _draw_m(rng)
            ed = {segment_index(w, k): v for k in range(1, N_SEG_PER_WALL + 1)}
            if _holdout_hit(ed):                             # skip the wall carrying east_3
                w = WALLS_2D[(WALLS_2D.index(w) + 1) % len(WALLS_2D)]
                ed = {segment_index(w, k): v for k in range(1, N_SEG_PER_WALL + 1)}
            out.append(_mk(L, W, ed, "uniform_wall", "train", gid))
        for j in range(6):                                   # single segment
            rng = np.random.default_rng([seed, gid, 200 + j])
            i = int(other[rng.integers(len(other))])
            out.append(_mk(L, W, {i: _draw_m(rng)}, "single_segment", "train", gid))
        for j in range(4):                                   # contiguous patch (2-3 segments)
            rng = np.random.default_rng([seed, gid, 300 + j])
            for _ in range(50):
                w = WALLS_2D[int(rng.integers(len(WALLS_2D)))]
                n = int(rng.integers(2, 4))
                k0 = int(rng.integers(1, N_SEG_PER_WALL - n + 2))
                idx = [segment_index(w, k) for k in range(k0, k0 + n)]
                if hold not in idx:
                    break
            v = _draw_m(rng)
            out.append(_mk(L, W, {i: v for i in idx}, "contiguous_patch", "train", gid))
        for j in range(2):                                   # multi-wall mixed
            rng = np.random.default_rng([seed, gid, 400 + j])
            idx = [int(x) for x in rng.choice(other, size=4, replace=False)]
            out.append(_mk(L, W, {i: _draw_m(rng) for i in idx},
                           "multi_wall", "train", gid))
        for j in range(2):                                   # window
            rng = np.random.default_rng([seed, gid, 500 + j])
            n = 1 if j == 0 else 2
            for _ in range(50):
                w = WALLS_2D[int(rng.integers(len(WALLS_2D)))]
                k0 = int(rng.integers(1, N_SEG_PER_WALL - n + 2))
                idx = [segment_index(w, k) for k in range(k0, k0 + n)]
                if hold not in idx:
                    break
            v = round(float(rng.uniform(*WINDOW_ALPHA_RANGE)), ALPHA_QUANT_DP)
            out.append(_mk(L, W, {i: v for i in idx}, "window", "train", gid))
    return out


def enumerate_test_configs(geoms) -> List[SegConfig]:
    """12 per geometry. Includes the held-out east_3 in three different ways."""
    out: List[SegConfig] = []
    hold = segment_index(*HOLDOUT_SEGMENT)
    for gid, (L, W) in enumerate(geoms):
        out.append(_mk(L, W, {}, "baseline", "test", gid))
        for w in WALLS_2D:                                   # 4 uniform-wall at 0.70
            ed = {segment_index(w, k): TEST_PATCH_ALPHA
                  for k in range(1, N_SEG_PER_WALL + 1)}
            out.append(_mk(L, W, ed, "t_uniform_wall", "test", gid))
        for w in WALLS_2D:                                   # 4 single-segment at 0.70
            out.append(_mk(L, W, {segment_index(w, 3): TEST_PATCH_ALPHA},
                           "t_single_segment", "test", gid))
        out.append(_mk(L, W, {hold: WINDOW_ALPHA}, "t_window_holdout", "test", gid))
        out.append(_mk(L, W, {segment_index("west", 2): WINDOW_ALPHA},
                       "t_window_seen", "test", gid))
        out.append(_mk(L, W, {segment_index("east", k): TEST_PATCH_ALPHA
                              for k in (2, 3, 4)},
                       "t_patch3_holdout", "test", gid))
    return out


def manifest_rows(train, test) -> List[dict]:
    rows = []
    for i, c in enumerate(list(train) + list(test)):
        rows.append({"i": i, "split": c.split, "kind": c.kind, "geom_id": c.geom_id,
                     "L": c.L, "W": c.W, "alphas": list(c.alphas), "m": list(c.m),
                     "edited": list(c.edited), "filename": c.filename, "strata": c.strata})
    return rows


def configs_from_rows(rows, split: Optional[str] = None,
                      kinds: Sequence[str] = ()) -> List[SegConfig]:
    out = []
    for r in rows:
        if split is not None and r["split"] != split:
            continue
        if kinds and r["kind"] not in kinds:
            continue
        out.append(SegConfig(L=float(r["L"]), W=float(r["W"]),
                             alphas=tuple(float(x) for x in r["alphas"]),
                             kind=r["kind"], split=r["split"], geom_id=int(r["geom_id"])))
    return out
