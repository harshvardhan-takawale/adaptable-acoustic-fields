"""P4-9 S-3 diagnostic (descriptive): is R0's second-source render any better than its ORIGINAL-source render?
r(R0's original-source field, second-source truth) and the truth-only baseline r(original-source truth, second-source
truth), off-lattice (Rule 11), same scorer as p4_9_src2_eval. -> outputs/p4_9/src2/diag_source_blind.json
  PYTHONPATH=$PWD python scripts/p4_9_src2_diag.py"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from scripts.p4_9_src2_eval import _load, on_lattice, SRC1_DATA, SRC2_DATA
from scripts.p4_8_unseen_eval import rooms
from scripts.p4_6_family_oof import BANDS, scores
from aaf.eval.modal_projection import enumerate_modes
from scripts.p4_4_family_eval import N_MODES
rs, _ = rooms()
cache = np.load("outputs/p4_8/cache/unseen_R0_120K.npz")
rows = []
for j, c in enumerate(rs):
    T1, rx, los1, keep, s1 = _load(SRC1_DATA / c.filename)
    T2, _, los2, _, s2 = _load(SRC2_DATA / c.filename)
    off = ~on_lattice(rx)
    modes = enumerate_modes(c.L, c.W, f_max=200.0)[:N_MODES]
    P1 = cache["P%d" % j].astype(np.complex128)
    a = scores(P1[off], T2[off], los2[off], modes)
    b = scores(T1[off], T2[off], los2[off], modes)
    rows.append({"room": c.label, "R0src1_vs_T2": {k: a[k] for k in BANDS}, "T1_vs_T2": {k: b[k] for k in BANDS}})
    print(c.label[:40], {k: round(a[k], 3) for k in BANDS}, {k: round(b[k], 3) for k in BANDS}, flush=True)
m = {v: {k: float(np.mean([r[v][k] for r in rows])) for k in BANDS} for v in ("R0src1_vs_T2", "T1_vs_T2")}
print("MEAN", json.dumps(m))
json.dump({"rows": rows, "mean": m}, open("outputs/p4_9/src2/diag_source_blind.json", "w"), indent=1)
