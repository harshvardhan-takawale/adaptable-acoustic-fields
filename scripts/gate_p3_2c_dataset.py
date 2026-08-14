"""P3-2c dataset gate: G1-G5, run before any training compute is committed.

Each gate guards a failure mode that would train happily and produce a *plausible* density
curve, which is the dangerous kind of bug:

  G1 slab emptiness      a training draw inside the held-out slab makes the "gap" a fiction,
                         so the arm sits at a smaller realized gap than its label claims.
  G2 preset collisions   a training draw landing on a test alpha turns that test point from a
                         holdout into a memorized value -- and every arm's S2 would inflate.
  G3 coverage            the repair stream must not pile draws against a slab edge; if it did,
                         the "wider gap" arm would also be the "worse-covered" arm and the two
                         explanations would be inseparable.
  G4 physics signature   the SIMULATIONS must show per-wall selectivity. A materials dict wired
                         to the wrong wall order still yields smooth, learnable data.
  G5 baseline immunity   alpha=0.15 must survive in every arm even when the slab contains it
                         (W100's [0.193, 1.193] does not, but W060's would for a wider centre).
                         If the baseline were ever repaired, the reference room would differ
                         between arms and every paired delta would be measured against a
                         different anchor.

G1/G2/G3/G5 are manifest-only (fast). G4 reads HDF5 and is a property of the FROZEN test set,
which is shared by every arm, so it runs once rather than per arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from aaf.data.mat_configs_cont import M_RANGE, m_of_alpha
from aaf.data.mat_configs_p3_2c import ALL_UNSEEN_ALPHAS, SPECS, realized_gap
from aaf.walls import ALPHA_BASELINE, WALL_INDEX, WALLS_2D

C_SOUND = 343.0
BAND_HI_HZ = 300.0
DF = 0.5                      # rfft bin spacing of the frozen 2 s / fs pipeline

# Predicted manifest deltas from the derivation design. Asserting the EXACT count (not a
# range) is what makes the byte-identity argument testable: any change to the repair stream,
# the seed, or the slab edges moves these numbers.
EXPECTED_DELTA: Dict[str, int] = {"W030": 31, "W060": 120, "W100": 236, "XTRAP": 156}


# --------------------------------------------------------------------------- helpers
def _train(rows):
    return [r for r in rows if r["split"] == "train"]


def _edited_ms(rows, wall):
    slot = WALL_INDEX[wall]
    return np.array([m_of_alpha(r["alphas"][slot]) for r in rows if wall in r["edited"]])


# --------------------------------------------------------------------------- G1
def g1_slab_emptiness(rows, spec) -> dict:
    """No training draw may lie inside any held-out region of this arm."""
    viol: List[dict] = []
    for r in _train(rows):
        for wall in r["edited"]:
            a = float(r["alphas"][WALL_INDEX[wall]])
            if spec.rejects(wall, a):
                viol.append({"i": r["i"], "wall": wall, "alpha": a, "m": m_of_alpha(a)})
    margins = {}
    for wall, (lo, hi) in spec.slabs().items():
        ms = _edited_ms(_train(rows), wall)
        below, above = ms[ms < lo], ms[ms > hi]
        margins[wall] = {
            "slab": [float(lo), float(hi)],
            "closest_below": float(below.max()) if below.size else None,
            "closest_above": float(above.min()) if above.size else None,
            "margin_below": float(lo - below.max()) if below.size else None,
            "margin_above": float(above.min() - hi) if above.size else None,
        }
    return {"pass": not viol, "n_violations": len(viol),
            "violations": viol[:10], "edge_margins": margins}


# --------------------------------------------------------------------------- G2
def g2_preset_collisions(rows, tol: float = 1e-6) -> dict:
    """No training draw may coincide with an alpha used as a test point."""
    hits: List[dict] = []
    for r in _train(rows):
        for wall in r["edited"]:
            a = float(r["alphas"][WALL_INDEX[wall]])
            for p in ALL_UNSEEN_ALPHAS:
                if abs(a - p) <= tol:
                    hits.append({"i": r["i"], "wall": wall, "alpha": a, "preset": float(p)})
    nearest = {}
    for p in ALL_UNSEEN_ALPHAS:
        d = min(abs(float(r["alphas"][WALL_INDEX[w]]) - p)
                for r in _train(rows) for w in r["edited"]) if _train(rows) else None
        nearest[f"{p:.2f}"] = float(d)
    return {"pass": not hits, "n_collisions": len(hits), "collisions": hits[:10],
            "nearest_train_alpha_to_each_test_preset": nearest}


# --------------------------------------------------------------------------- G3
def g3_coverage(rows, spec, n_bins: int = 16) -> dict:
    """Draws must stay spread over the admissible support, not pile against a slab edge.

    The test is deliberately density-normalized: a wider slab REMOVES support, so a raw
    histogram would flag the wide arms for having no mass where no mass is possible. We
    compare each arm's occupancy to the admissible bins only.
    """
    out = {}
    ok = True
    for wall in WALLS_2D:
        ms = _edited_ms(_train(rows), wall)
        edges = np.linspace(M_RANGE[0], M_RANGE[1], n_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        admissible = np.array(
            [not spec.rejects(wall, 1.0 - np.exp(-c)) for c in centres])
        counts, _ = np.histogram(ms, bins=edges)
        adm_counts = counts[admissible]
        empty = int((adm_counts == 0).sum())
        # No admissible bin may be empty, and no bin may hold >3x the admissible mean.
        mean = adm_counts.mean() if adm_counts.size else 0.0
        hot = int((adm_counts > 3.0 * mean).sum()) if mean else 0
        wall_ok = empty == 0 and hot == 0
        ok &= wall_ok
        out[wall] = {"pass": bool(wall_ok), "n_draws": int(ms.size),
                     "n_admissible_bins": int(admissible.sum()),
                     "n_empty_admissible_bins": empty, "n_overfull_bins": hot,
                     "min_bin": int(adm_counts.min()) if adm_counts.size else 0,
                     "max_bin": int(adm_counts.max()) if adm_counts.size else 0}
    geoms = sorted({r["geom_id"] for r in _train(rows)})
    out["geometry"] = {"n_train_geoms": len(geoms),
                       "contiguous": geoms == list(range(len(geoms)))}
    ok &= bool(out["geometry"]["contiguous"])
    return {"pass": bool(ok), **out}


# --------------------------------------------------------------------------- G5
def g5_baseline_immunity(rows, spec) -> dict:
    """Every non-edited wall must still sit at exactly ALPHA_BASELINE.

    True by construction (``_mk`` hard-sets non-edited walls and ``draw_alpha`` is called only
    for edited ones) -- which is precisely why it is worth asserting: the property lives in a
    different function from the one this chunk rewrote.
    """
    bad: List[dict] = []
    for r in _train(rows):
        for wall in WALLS_2D:
            if wall in r["edited"]:
                continue
            a = float(r["alphas"][WALL_INDEX[wall]])
            if abs(a - ALPHA_BASELINE) > 1e-12:
                bad.append({"i": r["i"], "wall": wall, "alpha": a})
    # Would this arm have rejected the baseline had it been drawn? Recording the answer makes
    # the gate informative rather than merely green.
    would_reject = {w: bool(spec.rejects(w, ALPHA_BASELINE)) for w in WALLS_2D}
    n_base = sum(1 for r in _train(rows) if not r["edited"])
    return {"pass": not bad, "n_violations": len(bad), "violations": bad[:10],
            "baseline_alpha": float(ALPHA_BASELINE),
            "baseline_m": float(m_of_alpha(ALPHA_BASELINE)),
            "arm_would_reject_baseline_if_drawn": would_reject,
            "n_all_baseline_train_rooms": n_base}


# --------------------------------------------------------------------------- G4
def _energy_spectrum(path: Path) -> np.ndarray:
    """Receiver-averaged power spectrum, |H|^2 meaned over receivers.

    Averaging POWER over all 64 receivers rather than picking one keeps every low mode
    visible: a single receiver can sit on a nodal line of the very mode being measured.
    """
    with h5py.File(str(path), "r") as f:
        H = np.asarray(f["ism/H_complex"][...])
    return (np.abs(H) ** 2).mean(axis=0)


def _bw_at(spec_pow: np.ndarray, f_target: float, search_hz: float = 2.0) -> dict:
    """-3 dB full width of the peak nearest ``f_target``, by linear interpolation in dB."""
    n = spec_pow.size
    freqs = np.arange(n) * DF
    lo = max(1, int((f_target - search_hz) / DF))
    hi = min(n - 1, int((f_target + search_hz) / DF) + 1)
    if hi <= lo:
        return {"f_hz": None, "bw_hz": None}
    k = lo + int(np.argmax(spec_pow[lo:hi]))
    db = 10.0 * np.log10(np.maximum(spec_pow, 1e-300))
    target = db[k] - 3.0

    def cross(step: int):
        j = k
        while 0 < j < n - 1 and db[j] > target:
            j += step
        if db[j] > target:
            return None
        j0, j1 = (j - step, j)
        d0, d1 = db[j0], db[j1]
        t = 0.0 if d0 == d1 else (d0 - target) / (d0 - d1)
        return freqs[j0] + t * (freqs[j1] - freqs[j0])

    f_lo, f_hi = cross(-1), cross(+1)
    return {"f_hz": float(freqs[k]),
            "bw_hz": None if f_lo is None or f_hi is None else float(f_hi - f_lo)}


def g4_physics_signature(manifest: dict, data_dir: Path, alpha_hi: float = 0.70) -> dict:
    """Per-wall SELECTIVITY in the simulations themselves.

    west is x-normal and south is y-normal, so absorbing west must broaden the (1,0) mode and
    barely touch (0,1), and south must do the reverse. A wall-order bug (a materials dict keyed
    in the wrong order) produces data that is still smooth and still learnable -- it just
    encodes the wrong physics -- so nothing downstream would catch it.

    west vs EAST would be no test at all: both are x-normal, so they damp the same modes.
    """
    rows = [r for r in manifest["configs"] if r["split"] == "test"]
    by_geom: Dict[int, Dict[str, dict]] = {}
    for r in rows:
        if len(r["edited"]) != 1:
            if not r["edited"]:
                by_geom.setdefault(r["geom_id"], {})["baseline"] = r
            continue
        w = r["edited"][0]
        a = float(r["alphas"][WALL_INDEX[w]])
        if abs(a - alpha_hi) < 1e-9:
            by_geom.setdefault(r["geom_id"], {})[w] = r

    per_geom, ratios_x, ratios_y = [], [], []
    for gid in sorted(by_geom):
        g = by_geom[gid]
        if not {"baseline", "west", "south"} <= set(g):
            continue
        L, W = float(g["baseline"]["L"]), float(g["baseline"]["W"])
        f10, f01 = C_SOUND / (2.0 * L), C_SOUND / (2.0 * W)
        if abs(f10 - f01) < 2.0:      # near-square: the two modes are not separable here
            continue
        m = {}
        for key in ("baseline", "west", "south"):
            p = data_dir / g[key]["filename"]
            if not p.exists():
                break
            s = _energy_spectrum(p)
            m[key] = {"10": _bw_at(s, f10), "01": _bw_at(s, f01)}
        if len(m) != 3:
            continue
        try:
            d_w10 = m["west"]["10"]["bw_hz"] - m["baseline"]["10"]["bw_hz"]
            d_w01 = m["west"]["01"]["bw_hz"] - m["baseline"]["01"]["bw_hz"]
            d_s10 = m["south"]["10"]["bw_hz"] - m["baseline"]["10"]["bw_hz"]
            d_s01 = m["south"]["01"]["bw_hz"] - m["baseline"]["01"]["bw_hz"]
        except TypeError:
            continue
        rx = d_w10 / d_s10 if d_s10 > 1e-9 else np.inf
        ry = d_s01 / d_w01 if d_w01 > 1e-9 else np.inf
        ratios_x.append(rx)
        ratios_y.append(ry)
        per_geom.append({"geom_id": gid, "L": L, "W": W, "f10": f10, "f01": f01,
                         "d_bw10_west": d_w10, "d_bw10_south": d_s10,
                         "d_bw01_south": d_s01, "d_bw01_west": d_w01,
                         "selectivity_x": float(rx), "selectivity_y": float(ry)})
    med_x = float(np.median(ratios_x)) if ratios_x else 0.0
    med_y = float(np.median(ratios_y)) if ratios_y else 0.0
    ok = bool(per_geom) and med_x > 3.0 and med_y > 3.0
    return {"pass": ok, "n_geoms": len(per_geom),
            "median_selectivity_x": med_x, "median_selectivity_y": med_y,
            "threshold": 3.0, "alpha_hi": alpha_hi, "per_geom": per_geom}


# --------------------------------------------------------------------------- driver
def gate_arm(name: str, manifest_path: Path, data_dir: Path, run_g4: bool) -> dict:
    manifest = json.load(open(manifest_path))
    spec = SPECS[name]
    rows = manifest["configs"]
    res = {
        "arm": name, "manifest": str(manifest_path),
        "n_train": manifest["n_train"], "n_test": manifest["n_test"],
        "G1_slab_emptiness": g1_slab_emptiness(rows, spec),
        "G2_preset_collisions": g2_preset_collisions(rows),
        "G3_coverage": g3_coverage(rows, spec),
        "G5_baseline_immunity": g5_baseline_immunity(rows, spec),
        "realized_gap_west": realized_gap(_train(rows), "west"),
        "realized_gap_north": realized_gap(_train(rows), "north"),
        "delta_vs_W015": manifest["delta_vs_W015"],
        # Which number is this arm's x-axis. For XTRAP, ``realized_gap_west`` is the ordinary
        # ~0.026 sampling gap of a contiguous region -- a real quantity, but NOT the distance
        # its test points sit beyond training support. Carrying the axis name next to both
        # numbers keeps the density plot from silently mixing the two.
        "gap_axis": manifest["gap_axis"],
        "edge_distances_west": manifest["edge_distances_west"],
    }
    exp = EXPECTED_DELTA.get(name)
    got = manifest["delta_vs_W015"]["n_changed"]
    res["DELTA_matches_design"] = {"pass": exp is None or got == exp,
                                   "expected": exp, "got": got}
    if run_g4:
        res["G4_physics_signature"] = g4_physics_signature(manifest, data_dir)
    # Missing simulations are a gate failure, not a runtime surprise 12 h into training.
    missing = [r["filename"] for r in rows
               if not (data_dir / (r["filename"] + ".done")).exists()]
    res["G0_sims_built"] = {"pass": not missing, "n_missing": len(missing),
                            "missing": missing[:10]}
    keys = [k for k in res if k.startswith("G") or k == "DELTA_matches_design"]
    res["pass"] = all(res[k]["pass"] for k in keys)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default="configs/sweeps_2d_mat")
    ap.add_argument("--data-dir", default="data/track_c_2d")
    ap.add_argument("--out", default="outputs/p3_2c/dataset_gate.json")
    ap.add_argument("--arms", nargs="*", default=["W030", "W060", "W100", "XTRAP"])
    args = ap.parse_args()

    mdir, ddir = Path(args.manifest_dir), Path(args.data_dir)
    results = []
    for i, name in enumerate(args.arms):
        # G4 is a property of the frozen test set, shared by every arm -> run it once.
        r = gate_arm(name, mdir / f"p3_2c_{name}_manifest.json", ddir, run_g4=(i == 0))
        results.append(r)
        flags = " ".join(
            f"{k.split('_')[0]}:{'ok' if v['pass'] else 'FAIL'}"
            for k, v in r.items() if isinstance(v, dict) and "pass" in v)
        print(f"[{name}] {'PASS' if r['pass'] else 'FAIL'}  {flags}")
        if r["gap_axis"] == "beyond_edge":
            e = r["edge_distances_west"]
            pts = "  ".join(f"a{p['alpha']:.2f}:+{p['beyond_edge_m']:.4f}"
                            for p in e["points"])
            print(f"       BEYOND-EDGE axis (train edge m={e['train_edge_m']:.4f})  {pts}"
                  f"   delta {r['delta_vs_W015']['n_changed']} configs")
        else:
            g = r["realized_gap_west"]
            print(f"       west gap {g['max_gap_m']:.4f} m  bracketed by "
                  f"[{g['bracketing_m'][0]:.4f}, {g['bracketing_m'][1]:.4f}]  "
                  f"delta {r['delta_vs_W015']['n_changed']} configs")
        if "G4_physics_signature" in r:
            s = r["G4_physics_signature"]
            print(f"       G4 selectivity  x {s['median_selectivity_x']:.1f}x  "
                  f"y {s['median_selectivity_y']:.1f}x  over {s['n_geoms']} geoms")

    ok = all(r["pass"] for r in results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"pass": ok, "arms": results}, open(out, "w"), indent=1)
    print(f"\n{'GATE PASS' if ok else 'GATE FAIL'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
