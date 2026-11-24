"""P4-10 A8 (mode-shift slope, spec Step 1 / D197 A8) and D2 (D201: linewidths of the first 25-100 Hz peaks). Both use one peak finder on the
receiver-averaged power spectrum S(f) = mean over receivers |H|^2 (0.5 Hz bins), peaks = local maxima with >= 3 dB prominence (dB domain).
  --d2 M1 M2 ...    15 in-family original rooms (cached renders): for each truth peak in 25-100 Hz (first 10 by frequency) the nearest model
                    peak within 3 Hz; half-power full width by linear interpolation in power (a peak whose half-power points are not reached
                    before a valley is 'overlapped' and dropped, counted). Model / truth linewidth ratio per matched peak; the error spectrum
                    (model per-bin |P - T|^2 over |T|^2, 25-100 Hz) beside the time-domain floor. A ratio > 1 = model peaks broader.
  --a8 MODEL        mode-shift slope on the geometry cells (Stretch, Wall, Notch): per cell, truth original peaks (first 10 in 25-150 Hz) are
                    Hungarian-matched to the model's original peaks (cost |df|, matches > 3 Hz dropped); each field's original peaks are matched
                    to its own edited peaks (> 8 Hz dropped); the shift pairs (truth, predicted) are pooled per family: slope of predicted on
                    true, Pearson, MAE in Hz, matched-peak count; a second reading on 25-100 Hz, labelled. Descriptive (A8).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/p4_10/modes")
DF = 0.5
FAM_GEO = ("StretchU", "StretchA", "Wall", "Notch")
SEED, NB = 20261013, 10000


def spec(H):
    return (np.abs(np.asarray(H)) ** 2).mean(0)


def peaks(S, lo, hi, prom_db=3.0, n=10):
    """indices (float frequencies by parabolic interpolation on dB) of the first n peaks by frequency in [lo, hi]"""
    from scipy.signal import find_peaks
    d = 10 * np.log10(np.maximum(S, 1e-30))
    f0, f1 = int(round(lo / DF)), int(round(hi / DF))
    k, _ = find_peaks(d[f0:f1 + 1], prominence=prom_db)
    k = k[:n] + f0
    out = []
    for i in k:
        a, b, c = d[i - 1], d[i], d[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        out.append((i + np.clip(off, -0.5, 0.5)) * DF)
    return np.array(out), k


def half_width(S, i):
    """half-power full width (Hz) of the peak at bin i, or NaN if a half-power point is not reached before the valley"""
    half = S[i] / 2
    w = []
    for sgn in (-1, 1):
        j = i
        while 0 < j + sgn < len(S) - 1 and S[j + sgn] > half:
            if S[j + sgn] > S[j] and j != i:
                return np.nan                                                      # rose again before reaching half power: overlapped
            j += sgn
        a, b = j, j + sgn
        if not (0 <= b < len(S)):
            return np.nan
        x = (S[a] - half) / max(S[a] - S[b], 1e-30)
        w.append((abs(a - i) + x) * DF)
    return w[0] + w[1]


def hungarian(a, b, cap):
    from scipy.optimize import linear_sum_assignment
    if not len(a) or not len(b):
        return []
    D = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None])
    ii, jj = linear_sum_assignment(D)
    return [(i, j) for i, j in zip(ii, jj) if D[i, j] <= cap]


def boot(v):
    v = np.asarray(v, float)
    m = v[np.random.default_rng(SEED).integers(0, len(v), (NB, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def d2(models):
    from scripts.p4_10_d1_decay import rooms_of
    from scripts.p4_10_scorecard import on_lattice
    rep = {}
    for name in models:
        rows, ratios, n_t, n_match, n_drop = [], [], 0, 0, 0
        errs, floors = [], []
        for P, T, rx in rooms_of(name, "test"):
            sel = ~on_lattice(rx)
            St, Sm = spec(np.asarray(T)[sel]), spec(np.asarray(P)[sel])
            ft, it = peaks(St, 25.0, 100.0)
            fm, im = peaks(Sm, 20.0, 110.0, n=40)
            wt, wm, rr = [], [], []
            for a, b in hungarian(ft, fm, 3.0):
                n_match += 1
                w1, w2 = half_width(St, it[a]), half_width(Sm, im[b])
                if np.isfinite(w1) and np.isfinite(w2):
                    rr.append(w2 / w1)
                    wt.append(w1)
                    wm.append(w2)
                else:
                    n_drop += 1
            n_t += len(ft)
            if rr:
                ratios.append(float(np.median(rr)))
                rows.append({"n_peaks": int(len(ft)), "n_used": len(rr), "truth_width_hz": float(np.median(wt)), "model_width_hz": float(np.median(wm))})
            lo, hi = 50, 200
            errs.append(10 * np.log10(np.mean(np.abs(np.asarray(P)[sel][:, lo:hi] - np.asarray(T)[sel][:, lo:hi]) ** 2) / np.mean(np.abs(np.asarray(T)[sel][:, lo:hi]) ** 2)))
        rep[name] = {"rooms": rows, "n_rooms": len(rows), "truth_peaks": n_t, "matched": n_match, "overlapped_dropped": n_drop,
                     "ratio_of_medians_by_room": boot(ratios), "median_ratio": float(np.median(ratios)),
                     "frac_rooms_model_broader": float(np.mean(np.array(ratios) > 1.0)), "error_level_db_25_100": boot(errs),
                     "truth_width_hz": float(np.median([r["truth_width_hz"] for r in rows])), "model_width_hz": float(np.median([r["model_width_hz"] for r in rows]))}
    return rep


def a8(name):
    from scripts.p4_10_scorecard import PRED, TRUTH, cells, cid, on_lattice, stored
    fams = {}
    for c in cells(list(FAM_GEO), False):
        pz, tz = PRED / name / (cid(c) + ".npz"), TRUTH / (cid(c) + ".npz")
        if not pz.exists() or not tz.exists():
            continue
        z, p = np.load(tz), np.load(pz)
        base = stored(c)
        o = np.load(PRED / name / "orig" / (c["room"] + ".npz"))
        rx = base["rx"]
        s0 = ~on_lattice(rx)
        Td, Pd = z["Hd"], p["Pd"]
        offd = ~on_lattice(z["rx"])
        T0, P0 = spec(base["H"][s0]), spec(o["P"][s0])
        Te, Pe = spec(Td[offd]), spec(Pd[offd])
        res = {}
        for tag, hi in (("25_150", 150.0), ("25_100", 100.0)):
            f0t, _ = peaks(T0, 25.0, hi)
            f0p, _ = peaks(P0, 25.0, hi, n=40)
            fet, _ = peaks(Te, 25.0, hi, n=40)
            fep, _ = peaks(Pe, 25.0, hi, n=40)
            mt = dict(hungarian(f0t, fet, 8.0))
            mp = dict(hungarian(f0p, fep, 8.0))
            pairs = []
            for i, j in hungarian(f0t, f0p, 3.0):
                if i in mt and j in mp:
                    pairs.append((fet[mt[i]] - f0t[i], fep[mp[j]] - f0p[j]))
            res[tag] = {"pairs": pairs, "n_truth_peaks": int(len(f0t)), "n_matched": len(pairs)}
        fams.setdefault((c["family"], c["group"]), []).append(res)
    out = {}
    for (fam, g), rows in fams.items():
        e = {"n_cells": len(rows)}
        for tag in ("25_150", "25_100"):
            pr = np.array([q for r in rows for q in r[tag]["pairs"]], float).reshape(-1, 2)
            e[tag] = {"n_truth_peaks": int(sum(r[tag]["n_truth_peaks"] for r in rows)), "n_pairs": int(len(pr))}
            if len(pr) >= 8 and np.ptp(pr[:, 0]) > 0:
                e[tag].update({"slope": float(np.polyfit(pr[:, 0], pr[:, 1], 1)[0]), "pearson": float(np.corrcoef(pr[:, 0], pr[:, 1])[0, 1]),
                               "mae_hz": float(np.mean(np.abs(pr[:, 1] - pr[:, 0]))), "true_shift_std_hz": float(pr[:, 0].std())})
        out["{}|{}".format(fam, "in-family" if g == "test" else "unseen")] = e
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2", nargs="*")
    ap.add_argument("--a8")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.d2:
        r = d2(a.d2)
        (OUT / "d2{}.json".format(a.tag)).write_text(json.dumps(r, indent=1))
        for m, v in r.items():
            print("{}: {} rooms; {} truth peaks, {} matched, {} overlapped dropped; width truth {:.2f} Hz / model {:.2f} Hz; ratio of medians per room {:.2f} [{:.2f}, {:.2f}], median {:.2f}; "
                  "model broader in {:.0%} of rooms; error level 25-100 Hz {:+.1f} dB".format(m, v["n_rooms"], v["truth_peaks"], v["matched"], v["overlapped_dropped"],
                  v["truth_width_hz"], v["model_width_hz"], *v["ratio_of_medians_by_room"], v["median_ratio"], v["frac_rooms_model_broader"], v["error_level_db_25_100"][0]))
    if a.a8:
        r = a8(a.a8)
        (OUT / "a8_{}{}.json".format(a.a8, a.tag)).write_text(json.dumps(r, indent=1))
        for k, v in r.items():
            print(k, v["n_cells"], {t: v[t] for t in ("25_150", "25_100")})
