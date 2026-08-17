"""P3-3-FAST Track 2b dataset gate. Run BEFORE any training compute is committed.

Four items, all thresholds fixed here before the numbers were looked at:

  (i)   every manifest config is built, opens, and carries the expected shape/attrs;
  (ii)  filenames are unique -- (L, W, x0, a) is the whole name, so a duplicated geometry
        would alias two configs onto one HDF5 file (the P3-2c collision hazard);
  (iii) ZERO training apertures inside the held-out band [0.9, 1.1] and >= 3 test apertures
        inside it, so "did it learn the aperture law?" is answerable;
  (iv)  the divider plumbing works end to end: a SEALED divider must drive room-B energy to
        (numerically) zero -- room B is disconnected, so H_B is identically 0, not merely
        small -- while the FULLY OPEN config of the same domain must show the two sub-rooms
        at a comparable level. This is the one item that tests the simulator rather than the
        bookkeeping: if ``extra_walls`` were silently dropped, (i)-(iii) would still pass.

The room-A / room-B split is by receiver x against the divider position x0, and the level
difference is 20 log10(mean |H| over B / mean |H| over A) across 20-300 Hz, the same
observable FT-B used for the sqrt(a) fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from aaf.data.aperture_configs import A_HOLDOUT, configs_from_rows, in_holdout

MANIFEST = "configs/sweeps_2d_mat/p3_3fast_trackB_manifest.json"
DATA_DIR = "data/track_p3_3fast_B"
OUT = "outputs/p3_3fast/trackB/DATASET_GATE.json"

BAND_LO_HZ, BAND_HI_HZ = 20.0, 300.0
DF_HZ = 0.5
N_RX = 64
N_BINS = 601

# --- thresholds, fixed a priori -------------------------------------------------------
SEALED_MAX_RATIO = 1e-6        # room B / room A amplitude ratio for a sealed divider
OPEN_MAX_ABS_LD_DB = 3.0       # |level difference| for a fully-open (no divider) room
MIN_TEST_IN_BAND = 3
A_REALIZED_TOL_M = 0.02        # 2 grid cells at dx = 0.01


def level_difference_db(H: np.ndarray, rx: np.ndarray, x0: float) -> Dict[str, float]:
    lo, hi = int(round(BAND_LO_HZ / DF_HZ)), int(round(BAND_HI_HZ / DF_HZ)) + 1
    mag = np.abs(H[:, lo:hi])
    sel_a, sel_b = rx[:, 0] < x0, rx[:, 0] > x0
    ma, mb = float(mag[sel_a].mean()), float(mag[sel_b].mean())
    ratio = mb / ma if ma > 0 else float("inf")
    return {"mean_abs_H_roomA": ma, "mean_abs_H_roomB": mb, "ratio_B_over_A": ratio,
            "ld_db": (20.0 * np.log10(ratio)) if ratio > 0 else float("-inf"),
            "n_rx_A": int(sel_a.sum()), "n_rx_B": int(sel_b.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    rows = man["configs"]
    cfgs = configs_from_rows(rows)
    d = Path(args.data_dir)

    # ---- (i) all sims built + readable --------------------------------------------
    missing, malformed = [], []
    for c in cfgs:
        p = d / c.filename
        if not p.exists() or not (d / (c.filename + ".done")).exists():
            missing.append(c.filename)
            continue
        try:
            with h5py.File(p, "r") as f:
                H = f["ism/H_complex"]
                shape = tuple(H.shape)
                a_attr, x0_attr = float(f.attrs["a"]), float(f.attrs["x0"])
                a_real = float(f.attrs["a_realized"])
                nan = not np.isfinite(np.asarray(H[:, :4])).all()
        except Exception as e:                       # noqa: BLE001 - report, do not crash
            malformed.append({"file": c.filename, "error": str(e)[:160]})
            continue
        if shape != (N_RX, N_BINS):
            malformed.append({"file": c.filename, "error": "shape {}".format(shape)})
        elif abs(a_attr - c.a) > 1e-9 or abs(x0_attr - c.x0) > 1e-9:
            malformed.append({"file": c.filename,
                              "error": "attr drift a={} x0={}".format(a_attr, x0_attr)})
        elif not c.fully_open and abs(a_real - c.a) > A_REALIZED_TOL_M:
            malformed.append({"file": c.filename,
                              "error": "a_realized {:.4f} vs {:.4f}".format(a_real, c.a)})
        elif nan:
            malformed.append({"file": c.filename, "error": "non-finite H"})
    n_built = len(cfgs) - len(missing)
    item_i = {"pass": not missing and not malformed, "n_total": len(cfgs),
              "n_built": n_built, "n_missing": len(missing), "n_malformed": len(malformed),
              "missing_examples": missing[:8], "malformed_examples": malformed[:8]}

    # ---- (ii) filename uniqueness --------------------------------------------------
    names = [c.filename for c in cfgs]
    dupes = sorted({n for n in names if names.count(n) > 1}) if len(set(names)) != len(names) \
        else []
    item_ii = {"pass": not dupes, "n_rows": len(names), "n_unique": len(set(names)),
               "duplicates": dupes[:8]}

    # ---- (iii) the hold-out band is exact -------------------------------------------
    tr_in = [c.filename for c in cfgs
             if c.split == "train" and not c.sealed and in_holdout(c.a)]
    te_in = [c.filename for c in cfgs if c.split == "test" and in_holdout(c.a)]
    item_iii = {"pass": (not tr_in) and len(te_in) >= MIN_TEST_IN_BAND,
                "band": list(A_HOLDOUT), "n_train_in_band": len(tr_in),
                "n_test_in_band": len(te_in), "min_test_in_band": MIN_TEST_IN_BAND,
                "train_offenders": tr_in[:8],
                "test_a_values_in_band": sorted({c.a for c in cfgs
                                                 if c.split == "test" and in_holdout(c.a)})}

    # ---- (iv) sealed vs fully open: the divider actually divides ---------------------
    sealed_rep: List[dict] = []
    open_rep: List[dict] = []
    for c in cfgs:
        if not (c.sealed or c.fully_open):
            continue
        p = d / c.filename
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            H = np.asarray(f["ism/H_complex"][:])
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), dtype=float)
        rec = {"file": c.filename, "split": c.split}
        rec.update(level_difference_db(H, rx, c.x0))
        (sealed_rep if c.sealed else open_rep).append(rec)

    worst_sealed = max((r["ratio_B_over_A"] for r in sealed_rep), default=float("inf"))
    worst_open = max((abs(r["ld_db"]) for r in open_rep), default=float("inf"))
    item_iv = {
        "pass": bool(sealed_rep and open_rep
                     and worst_sealed <= SEALED_MAX_RATIO
                     and worst_open <= OPEN_MAX_ABS_LD_DB),
        "n_sealed_checked": len(sealed_rep), "n_open_checked": len(open_rep),
        "sealed_max_ratio_B_over_A": worst_sealed, "sealed_threshold": SEALED_MAX_RATIO,
        "open_max_abs_ld_db": worst_open, "open_threshold_db": OPEN_MAX_ABS_LD_DB,
        "sealed_examples": sealed_rep[:3], "open_examples": open_rep[:3],
        "note": ("sealed room-B energy is EXACTLY zero when the divider disconnects room B; "
                 "any non-zero value here means extra_walls did not reach the solver"),
    }

    gate = {"gate": "P3-3-FAST Track 2b dataset",
            "manifest": args.manifest, "data_dir": args.data_dir,
            "rows_sha256": man.get("rows_sha256", ""),
            "band_hz": [BAND_LO_HZ, BAND_HI_HZ],
            "i_all_built": item_i, "ii_filenames_unique": item_ii,
            "iii_holdout_band_exact": item_iii, "iv_divider_physics": item_iv}
    gate["pass"] = all(gate[k]["pass"] for k in
                       ("i_all_built", "ii_filenames_unique", "iii_holdout_band_exact",
                        "iv_divider_physics"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(gate, indent=1))
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v)
                      for k, v in gate.items() if k.startswith(("i", "pass"))}, indent=1))
    print("built {}/{} | sealed max ratio {:.3e} | open max |LD| {:.2f} dB -> {}".format(
        n_built, len(cfgs), worst_sealed, worst_open, "PASS" if gate["pass"] else "FAIL"))
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
