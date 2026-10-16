"""P4-7 prep p4 (D164): how reciprocal is the corpus-resolution FDTD? (pre-flight for `aug_recip_p`)

The swap augmentation keeps the target H(rx; src) when it renders H(src; rx). The continuous physics is
reciprocal; the KW solver is exactly reciprocal in a rectangle between interior nodes, and only
approximately so near masked notch faces (the wall-tip energy-weight residual, D65(b)) -- measured on a
small room in `tests/test_augment_2d.py`. This measures it at the CORPUS settings (dx 0.02, fs 30720,
n 61440, wall alpha 0.15): one room per family (D111's pre-registered median test rooms), the corpus
source (0.5, 1.6) and 10 of the room's own dense receivers (interior nodes, by stride). For each receiver,
simulate source -> receiver and receiver -> source and compare in band (bins 1-600, by band).
Writes outputs/p4_7/p4_reciprocity.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.augment_2d import is_interior_node
from aaf.data.multi_notch import solid_blocks
from aaf.data.shape_configs import ALPHA, SRC
from scripts.build_p4_5_rooms import OUT as ROOT, family_test_rooms

DX, FS, N = 0.02, 30720.0, 61440
BANDS = {"0-20": (1, 40), "20-100": (40, 200), "100-200": (200, 400), "200-300": (400, 601)}
N_RX = 10
OUT = Path("outputs/p4_7")


def main():
    rep = {"settings": {"dx": DX, "fs": FS, "n": N, "wall_alpha": ALPHA}, "rooms": []}
    for fam, c in family_test_rooms():
        nx, dxx = F._fit_axis(c.L, DX, "L")
        ny, dxy = F._fit_axis(c.W, DX, "W")
        m = solid_blocks(nx, ny, dxx, dxy, c.L, c.W, c.notches)
        extra = [{"type": "mask", "solid": m, "alpha": ALPHA}] if m.any() else None
        with h5py.File(Path(ROOT["famdense"]) / c.filename) as f:
            rx_all = np.asarray(json.loads(f.attrs["receiver_pos"]))
        inter = is_interior_node(c.verts, rx_all, dxx, dxy)
        rx = rx_all[inter][np.linspace(0, inter.sum() - 1, N_RX).astype(int)]
        fwd = F.simulate(c.L, c.W, [ALPHA] * 4, SRC, rx, dx=DX, fs=FS, n=N, extra_walls=extra)["H_complex"][:, :601]
        bands = {b: [] for b in BANDS}
        for i, r in enumerate(rx):
            bwd = F.simulate(c.L, c.W, [ALPHA] * 4, tuple(r), np.asarray([SRC]), dx=DX, fs=FS, n=N,
                             extra_walls=extra)["H_complex"][0, :601]
            for b, (lo, hi) in BANDS.items():
                bands[b].append(float(np.linalg.norm(fwd[i, lo:hi] - bwd[lo:hi]) / np.linalg.norm(fwd[i, lo:hi])))
        row = {"family": fam, "room": c.label, "n_rx": int(len(rx)),
               "rel_l2_per_band_median": {b: float(np.median(v)) for b, v in bands.items()},
               "rel_l2_per_band_max": {b: float(np.max(v)) for b, v in bands.items()}}
        rep["rooms"].append(row)
        print(fam, c.label, json.dumps({b: "{:.1e} (max {:.1e})".format(row["rel_l2_per_band_median"][b],
                                                                        row["rel_l2_per_band_max"][b]) for b in BANDS}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p4_reciprocity.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
