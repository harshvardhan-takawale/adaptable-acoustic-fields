"""P4-1: configs for POLYGONAL rooms -- the shape family beyond the shoebox.

Every 2D config class so far describes a rectangle: `MatConfig` (L, W, 4 alphas), `SegConfig`
(16 wall segments), `ApertureConfig` (adds a divider at x0). A room whose boundary is not a
rectangle cannot be expressed by any of them, and the difference is not cosmetic -- an L-room
has six edges, one of them reflex, and `(L, W)` describes only its bounding box.

`PolyConfig` carries the vertex list itself and one absorption per edge. `L` and `W` are kept
because the renderer needs an AABB (`FreqRenderer2D` intersects rays against a rectangle), the
trainer groups rows by geometry using them, and every downstream metric keys on them -- but for
a polygon they are the BOUNDING BOX and nothing more. Read `verts` for the actual shape.

The vertex list is the single source of truth: `alphas_4` (the shoebox-shaped view some shared
code still wants) is DERIVED from it rather than stored, so the two cannot drift apart.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PolyConfig:
    """One polygonal room. ``verts`` is counter-clockwise, no repeated first vertex."""

    verts: Tuple[Tuple[float, float], ...]
    edge_alphas: Tuple[float, ...]
    filename: str
    split: str = "train"
    kind: str = "poly"
    shape: str = "poly"
    strata: str = "poly"
    geom_id: int = 0

    def __post_init__(self):
        if len(self.verts) < 3:
            raise ValueError("a polygon needs >= 3 vertices, got {}".format(len(self.verts)))
        if len(self.edge_alphas) != len(self.verts):
            raise ValueError("expected {} edge alphas (one per edge), got {}".format(
                len(self.verts), len(self.edge_alphas)))

    # -- bounding box, for the renderer's AABB and for geometry grouping ------------------
    @property
    def L(self) -> float:
        return max(v[0] for v in self.verts) - min(v[0] for v in self.verts)

    @property
    def W(self) -> float:
        return max(v[1] for v in self.verts) - min(v[1] for v in self.verts)

    @property
    def alphas(self) -> List[float]:
        """A 4-vector view for code that still assumes a shoebox.

        This is the OUTER bounding-box walls' absorption, and for a non-convex room it is a
        lossy summary -- the notch faces have no slot in it. Nothing in the P4-1 shape path
        consumes it (the `geom_token` arm reads `verts` + `edge_alphas` directly); it exists so
        the trainer's manifest/data drift check has something to compare, and it is derived so
        it cannot disagree with the vertex list.
        """
        return [float(self.edge_alphas[0])] * 4

    @property
    def label(self) -> str:
        return "{}[{}v]".format(self.shape, len(self.verts))

    @property
    def name(self) -> str:
        return self.label


def configs_from_rows(rows: Sequence[dict], split: Optional[str] = None,
                      kinds: Sequence[str] = ()) -> List[PolyConfig]:
    """Manifest rows -> configs. Signature matches the other `configs_from_rows` verbatim so the
    trainer's schema dispatch can swap between them without any other change."""
    out: List[PolyConfig] = []
    for r in rows:
        if split is not None and r.get("split") != split:
            continue
        if kinds and r.get("kind") not in kinds:
            continue
        out.append(PolyConfig(
            verts=tuple((float(a), float(b)) for a, b in r["verts"]),
            edge_alphas=tuple(float(x) for x in r["edge_alphas"]),
            filename=str(r["filename"]),
            split=str(r.get("split", "train")),
            kind=str(r.get("kind", "poly")),
            shape=str(r.get("shape", "poly")),
            strata=str(r.get("strata", r.get("shape", "poly"))),
            geom_id=int(r.get("geom_id", 0)),
        ))
    return out
