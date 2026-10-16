"""P4-7 prep p1 (D164): orientation audit of the training / test corpora.

For C250, C500, the 15 frozen test shapes and the family corpus, per split:
  * notch placement by corner / side. L-corpora: the region missing from the bounding box is located
    from the verts (the bbox corner it touches, or mid-edge); family corpus: the `notches` sides;
  * reflex-corner positions by bbox quadrant;
  * every other asymmetric feature: source position (all h5 `source_pos`), receiver-lattice anchoring
    (fraction of receivers on 0.12 + 0.06 k from the SW origin), wall alphas, L > W, and the source's
    distance to the nearest notch face.
Read-only; writes outputs/p4_7/p1_orientation.{json,md}.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

CORPORA = {
    "C250": ("configs/sweeps_2d_mat/p4_4_shapes_250.json", "data/track_p4_2_shapes"),
    "C500": ("configs/sweeps_2d_mat/p4_4_shapes_500.json", "data/track_p4_2_shapes"),
    "test15": ("configs/sweeps_2d_mat/p4_2_shapes_manifest.json", "data/track_p4_2_shapes"),
    "family": ("configs/sweeps_2d_mat/p4_4_family_manifest.json", "data/track_p4_4_family"),
}
OUT = Path("outputs/p4_7")


def readers(man):
    if str(man.get("schema", "")).startswith("p4_4.family"):
        from aaf.data.multi_notch import configs_from_rows, reflex_corners
        return configs_from_rows, (lambda c: reflex_corners(c.L, c.W, c.notches)), True
    from aaf.data.shape_configs import configs_from_rows
    return configs_from_rows, (lambda c: [(c.w, c.W - c.d)] if c.d > 0 and c.w > 0 else []), False


def notch_sides_from_verts(c):
    """Bbox corner / mid-edge of the missing region(s), from the polygon (L-corpora)."""
    if not (c.d > 0 and c.w > 0):
        return []
    x0, x1, y0, y1 = 0.0, c.w, c.W - c.d, c.W                    # the region the verts leave out
    ns = "N" if y1 >= c.W - 1e-9 else ("S" if y0 <= 1e-9 else "mid")
    ew = "W" if x0 <= 1e-9 else ("E" if x1 >= c.L - 1e-9 else "mid")
    return [ns + ew if "mid" not in (ns, ew) else "mid" + (ns if ns != "mid" else ew)]


def quadrant(p, L, W):
    return ("N" if p[1] > W / 2 else "S") + ("E" if p[0] > L / 2 else "W")


def seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def main():
    from aaf.data.shape_configs import SRC
    rep = {}
    for name, (mpath, ddir) in CORPORA.items():
        man = json.load(open(mpath))
        cfr, reflex, fam = readers(man)
        rep[name] = {}
        for split in ("train", "test"):
            cs = cfr(man["configs"], split=split)
            if not cs:
                continue
            sides, quads, src_pos, on_lat, alphas = Counter(), Counter(), Counter(), [], Counter()
            lw, d_src = 0, []
            for c in cs:
                s = [n.side for n in c.notches] if fam else notch_sides_from_verts(c)
                sides.update(s if s else ["none (rect)"])
                quads.update(quadrant(p, c.L, c.W) for p in reflex(c))
                lw += c.L > c.W
                alphas.update(round(float(a), 4) for a in c.edge_alphas)
                rcs = reflex(c)
                if rcs:
                    d_src.append(min(np.hypot(SRC[0] - x, SRC[1] - y) for x, y in rcs))
                f = Path(ddir) / c.filename
                if f.exists():
                    with h5py.File(f) as h:
                        src_pos[tuple(json.loads(h.attrs["source_pos"]))] += 1
                        rx = np.asarray(json.loads(h.attrs["receiver_pos"]))
                    k = (rx - 0.12) / 0.06
                    on_lat.append(float(np.all(np.abs(k - np.round(k)) < 1e-6, axis=1).mean()))
            rep[name][split] = {
                "n": len(cs), "notch_sides": dict(sides), "reflex_corner_quadrants": dict(quads),
                "source_positions": {str(k): v for k, v in src_pos.items()},
                "rx_on_SW_anchored_lattice_frac_mean": float(np.mean(on_lat)) if on_lat else None,
                "edge_alphas": {str(k): v for k, v in alphas.items()}, "frac_L_gt_W": lw / len(cs),
                "src_to_nearest_reflex_corner_m": [float(np.min(d_src)), float(np.median(d_src))] if d_src else None,
            }
            print(name, split, json.dumps(rep[name][split]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p1_orientation.json").write_text(json.dumps(rep, indent=1))
    md = ["# P4-7 prep p1 -- orientation audit (D164)", "",
          "| corpus | split | n | notch sides | reflex-corner quadrants | source | rx on SW-anchored lattice | L > W |",
          "|---|---|---|---|---|---|---|---|"]
    for name, d in rep.items():
        for split, r in d.items():
            md.append("| {} | {} | {} | {} | {} | {} | {:.3f} | {:.2f} |".format(
                name, split, r["n"], ", ".join("{} {}".format(k, v) for k, v in sorted(r["notch_sides"].items())),
                ", ".join("{} {}".format(k, v) for k, v in sorted(r["reflex_corner_quadrants"].items())) or "--",
                "; ".join("{} x{}".format(k, v) for k, v in r["source_positions"].items()),
                r["rx_on_SW_anchored_lattice_frac_mean"] or float("nan"), r["frac_L_gt_W"]))
    (OUT / "p1_orientation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
