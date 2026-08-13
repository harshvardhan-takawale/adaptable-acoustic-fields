"""P3-2b dataset gate (BLOCKING) -> outputs/p3_2b/DATASET_GATE.md.

Guards the four things that would silently invalidate the chunk:
  G1  no training config falls inside a held-out slab (else S2 is not a held-out test)
  G2  no training alpha equals a demo preset (else "unseen exact value" is false)
  G3  per-wall m coverage is ~uniform OUTSIDE the slabs and EMPTY inside
  G4  the physics signature still reproduces on configs from the NEW generator
      (guards against a sampling-code regression that quietly changes the target)

Exit 0 = PASS, 3 = STOP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from aaf.data.mat_configs_cont import (
    HOLDOUT_SLABS,
    M_RANGE,
    configs_from_rows,
    in_slab,
    m_of_alpha,
)
from aaf.walls import WALLS_2D, WALL_INDEX

MANIFEST = "configs/sweeps_2d_mat/p3_2b_manifest.json"
PRESETS = (0.05, 0.15, 0.30, 0.50, 0.70)


def main() -> int:
    man = json.load(open(MANIFEST))
    cfgs = configs_from_rows(man["configs"])
    train = [c for c in cfgs if c.split == "train"]
    test = [c for c in cfgs if c.split == "test"]
    out = Path("outputs/p3_2b")
    out.mkdir(parents=True, exist_ok=True)

    rep: dict = {"n_train": len(train), "n_test": len(test),
                 "rows_sha256": man.get("rows_sha256", ""), "seed": man.get("seed")}
    fails: list = []

    # G1/G2 -------------------------------------------------------------------
    slab_hits, preset_hits, per_wall = [], [], {w: [] for w in WALLS_2D}
    for c in train:
        for w in c.edited:
            a = c.alphas[WALL_INDEX[w]]
            per_wall[w].append(m_of_alpha(a))
            if in_slab(w, a):
                slab_hits.append((c.filename, w, a))
            if any(abs(a - p) <= 1e-6 for p in PRESETS):
                preset_hits.append((c.filename, w, a))
    rep["G1_slab_violations"] = len(slab_hits)
    rep["G2_preset_collisions"] = len(preset_hits)
    if slab_hits:
        fails.append(f"G1: {len(slab_hits)} training draws inside a slab")
    if preset_hits:
        fails.append(f"G2: {len(preset_hits)} training draws on a preset alpha")

    # G3 ----------------------------------------------------------------------
    hist = {}
    for w in WALLS_2D:
        ms = np.asarray(per_wall[w])
        h, edges = np.histogram(ms, bins=12, range=M_RANGE)
        sl = HOLDOUT_SLABS.get(w)
        centres = 0.5 * (edges[:-1] + edges[1:])
        inside = np.array([bool(sl and sl[0] <= x <= sl[1]) for x in centres])
        outside = h[~inside]
        # uniformity: no outside bin should be wildly off the mean
        ratio = float(outside.max() / max(outside.mean(), 1e-9))
        hist[w] = {"n": int(ms.size), "counts": h.tolist(),
                   "slab_bins": inside.tolist(), "max_over_mean_outside": ratio,
                   "n_in_slab": int(sum(in_slab(w, 1 - np.exp(-m)) for m in ms))}
        if hist[w]["n_in_slab"]:
            fails.append(f"G3: wall {w} has {hist[w]['n_in_slab']} draws in its slab")
        if ratio > 2.0:
            fails.append(f"G3: wall {w} coverage is non-uniform outside the slab "
                         f"(max/mean = {ratio:.2f})")
    rep["G3_per_wall"] = hist

    # G4 ----------------------------------------------------------------------
    # Re-measure the block-diagonal signature on two configs from the NEW generator.
    import pyroomacoustics as pra  # noqa: F401  (import cost only when the gate runs)
    from scripts.p3_2_physics_gate import measure, receiver_grid
    sig = {}
    for c in [c for c in train if c.kind == "single"][:2]:
        rx = receiver_grid(c.L, c.W)
        base = tuple([0.15] * 4)
        fam_b, _, _, _ = measure(c.L, c.W, base, rx)
        fam_e, _, _, _ = measure(c.L, c.W, c.alphas, rx)
        w = c.edited[0]
        own = "x_axial" if w in ("west", "east") else "y_axial"
        oth = "y_axial" if own == "x_axial" else "x_axial"
        d_own = fam_e[own]["bw"] - fam_b[own]["bw"]
        d_oth = fam_e[oth]["bw"] - fam_b[oth]["bw"]
        sel = abs(d_own) / max(abs(d_oth), 0.15)
        sig[c.label] = {"wall": w, "alpha": c.alphas[WALL_INDEX[w]],
                        "d_bw_own": d_own, "d_bw_other": d_oth, "selectivity": sel}
        if not (d_own > 0.5 and sel >= 3.0):
            fails.append(f"G4: {c.label} lost the block-diagonal signature "
                         f"(d_own={d_own:.3f}, sel={sel:.1f})")
    rep["G4_signature"] = sig

    rep["verdict"] = "PASS" if not fails else "STOP"
    rep["failures"] = fails
    (out / "dataset_gate.json").write_text(json.dumps(rep, indent=2, default=float))

    lines = [f"# P3-2b dataset gate — **{rep['verdict']}**", "",
             f"Manifest `{MANIFEST}` (seed {rep['seed']}, rows_sha256 "
             f"`{rep['rows_sha256'][:16]}`): **{len(train)} train + {len(test)} test**.", "",
             "| check | result |", "|---|---|",
             f"| G1 no training draw inside a held-out slab | "
             f"{'PASS' if not slab_hits else 'FAIL'} ({len(slab_hits)} violations) |",
             f"| G2 no training alpha equals a demo preset (1e-6) | "
             f"{'PASS' if not preset_hits else 'FAIL'} ({len(preset_hits)} collisions) |",
             f"| G3 per-wall m coverage uniform outside slabs, empty inside | "
             f"{'PASS' if all(v['n_in_slab'] == 0 for v in hist.values()) else 'FAIL'} |",
             f"| G4 block-diagonal signature reproduces on new-generator configs | "
             f"{'PASS' if sig and all(v['selectivity'] >= 3.0 for v in sig.values()) else 'FAIL'} |",
             "", "## Per-wall m coverage (12 bins over [0.02, 1.61]; `X` = slab bin)", "",
             "| wall | n draws | histogram | slab bins | in-slab |", "|---|---:|---|---|---:|"]
    for w in WALLS_2D:
        v = hist[w]
        marks = "".join("X" if b else "." for b in v["slab_bins"])
        lines.append(f"| {w} | {v['n']} | {v['counts']} | `{marks}` | {v['n_in_slab']} |")
    lines += ["", "## Physics signature on new-generator configs", "",
              "| config | wall | alpha | ΔBW own | ΔBW other | selectivity |",
              "|---|---|---:|---:|---:|---:|"]
    for k, v in sig.items():
        lines.append(f"| {k} | {v['wall']} | {v['alpha']:.4f} | {v['d_bw_own']:+.3f} | "
                     f"{v['d_bw_other']:+.3f} | {v['selectivity']:.1f} |")
    lines += ["", "Because alpha is drawn continuously, the demo presets (0.05 / 0.50 / 0.70) "
              "have probability zero of appearing exactly in training — **every preset "
              "evaluation is at an unseen exact value by construction**.", ""]
    if fails:
        lines += ["**Failures:** " + "; ".join(fails), ""]
    (out / "DATASET_GATE.md").write_text("\n".join(lines))
    print(json.dumps({"verdict": rep["verdict"], "failures": fails}, indent=1))
    return 0 if rep["verdict"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
