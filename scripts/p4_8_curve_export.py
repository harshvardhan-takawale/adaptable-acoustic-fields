"""P4-8 meeting deck (manager, 2026-10-19): per-bin off-lattice r curves for C250 @60K, B5 @60K and R0 @120K, in the
JSON structure of outputs/p4_5/FREQ_CURVE_60K.json (scripts/p4_5_freq_curve.py), for the manager's own charts.

Per room: dB spatial Pearson per bin over OFF-lattice receivers (Rule 11), bins 0-600 (0-300 Hz); per bin the mean over
rooms (unsmoothed, 601 values; NaN -> null). Marks every 25 Hz from 25 to 300 Hz = the mean over +-2.5 Hz (11 bins;
300 Hz one-sided, bins 590-600), with a 95% CI from a 10k bootstrap over rooms (seed 20261013, drawn once per model and
room set, exactly as p4_5_freq_curve.g15). Band LSD 200-300 Hz = mean |dB error| over bins 400-600, same bootstrap.
Room sets:
  G15       the 15 frozen in-family test shapes, dense (outputs/p4_5/gate2_dense/cache/<NAME>__track_p4_5_testdense.npz)
  UNSEEN16  the frozen unseen-family set (configs/sweeps_2d_mat/p4_8_unseen_manifest.json; outputs/p4_8/cache/unseen_*)
  U8, C3_8  its U and C3 halves (manifest order: 0-7 U, 8-15 C3, checked against the manifest)
The A2 (3-shape) protocol of FREQ_CURVE_60K.json is not exported (not requested). Network models only.
  python scripts/p4_8_curve_export.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.p4_5_freq_curve import DF, MARKS, NB, SEED, colr, db, window
from scripts.p4_5_gate2_dense import on_lattice

MODELS = ["C250_60K", "B5_60K", "R0_120K"]
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
UN = "outputs/p4_8/cache/unseen_{}.npz"
OUT = Path("outputs/p4_8/meeting_v2/FREQ_CURVES.json")


def rooms_curves(path, idx_keep=None):
    z = np.load(path, allow_pickle=False)
    per, lsd = [], []
    i = 0
    while "P%d" % i in z.files:
        if idx_keep is None or i in idx_keep:
            P, T, rx = db(z["P%d" % i][:, :601]), db(z["T%d" % i][:, :601]), z["rx%d" % i]
            off = ~on_lattice(rx)
            per.append(colr(P[off], T[off]))
            lsd.append(float(np.mean(np.abs(P[off, 400:601] - T[off, 400:601]))))
        i += 1
    per, lsd = np.array(per), np.array(lsd)
    n = len(per)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, (NB, n))
    out = {}
    for hz in MARKS:
        v = np.nanmean(per[:, slice(*window(hz))], axis=1)
        out[str(hz)] = [float(v.mean()), *map(float, np.percentile(v[idx].mean(1), [2.5, 97.5]))]
    out["lsd_200_300"] = [float(lsd.mean()), *map(float, np.percentile(lsd[idx].mean(1), [2.5, 97.5]))]
    out["n_rooms"] = int(n)
    curve = [None if not np.isfinite(x) else float(x) for x in np.nanmean(per, axis=0)]
    return out, curve


def main():
    man = json.load(open("configs/sweeps_2d_mat/p4_8_unseen_manifest.json"))
    fam = [r["family"] for r in man["configs"]]
    iu = {i for i, f in enumerate(fam) if f == "U"}
    ic = {i for i, f in enumerate(fam) if f == "C3"}
    assert iu == set(range(8)) and ic == set(range(8, 16)), fam
    rep = {"method": __doc__.strip().splitlines()[0], "top_of_band": "295-300 Hz, bins 590-600, one-sided",
           "marks_hz": MARKS, "bin_hz": DF, "n_bins": 601, "unseen_manifest_sha": man["rows_sha256"],
           "sets": {"G15": "15 frozen in-family test shapes (dense)", "UNSEEN16": "frozen unseen set, 8 U + 8 C3",
                    "U8": "unseen set, U rooms (manifest 0-7)", "C3_8": "unseen set, C3 rooms NW+NE+SE (manifest 8-15)"},
           "rows": {}}
    for m in MODELS:
        row = {}
        for key, path, keep in (("G15", G2.format(m), None), ("UNSEEN16", UN.format(m), None),
                                ("U8", UN.format(m), iu), ("C3_8", UN.format(m), ic)):
            marks, curve = rooms_curves(path, keep)
            row[key] = marks
            row[key + "_per_bin_off"] = curve
        rep["rows"][m] = row
        print(m, " ".join("{}:{:.3f}".format(k, row[k]["200"][0]) for k in ("G15", "UNSEEN16", "U8", "C3_8")), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
