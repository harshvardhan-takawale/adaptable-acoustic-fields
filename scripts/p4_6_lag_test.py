"""P4-6 t1 (D160(h), pre-registered): is the train-to-test transfer error MODE MISPLACEMENT?

At each receiver, the mean-removed dB spectra of prediction and FDTD are cross-correlated over a
+-25 Hz window centred at 125 Hz and at 250 Hz; the lag (searched over +-10 Hz) is the argmax of the
normalised correlation, refined by a parabola. POSITIVE lag = the prediction's spectral structure
sits ABOVE FDTD's in frequency (pred(f + lag) ~ FDTD(f)).

Rooms:
  TEST     the 15 dense test shapes, off-lattice receivers (the `p4_5_gate2_dense` cache);
  TRAINING the fixed 23 training rooms of D125 at B6-corpus receivers off the lattice
           (`p4_5_train_offlat_fit.rooms`), rendered by `--render`.

Per room: median lag m_r and sign consistency s_r; SYSTEMATIC if |m_r| >= 0.5 Hz and s_r >= 0.70.
"Mode misplacement" is declared only if (1) >= 2/3 of test rooms are systematic at 250 Hz, (2) over
test rooms systematic at both centres the median |m_r(250)| / |m_r(125)| is in [1.5, 2.5], and
(3) on training rooms median |m_r| <= 0.25 Hz at both centres and <= 1/3 of them are systematic.
If declared: dL_eff = -L * m_r / f per test room (cm; negative = the model's room acts SMALLER).

  python scripts/p4_6_lag_test.py --render NAME=CKPT        # GPU: training-room renders -> cache
  python scripts/p4_6_lag_test.py --analyse PB3_60K,B5_60K  # CPU: lags, verdict, JSON
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from aaf.data.shape_configs import SRC, configs_from_rows
from scripts.p4_5_gate2_dense import MANIFEST, on_lattice
from scripts.p4_5_train_offlat_fit import DATA, rooms

DF = 0.5
CENTRES = {"125": 250, "250": 500}           # Hz label -> centre bin
HALF = 50                                    # +-25 Hz window
MAXLAG = 20                                  # +-10 Hz search
CACHE = Path("outputs/p4_6/cache")
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
OUT = Path("outputs/p4_6/t1_lag")
SYS_LAG, SYS_SIGN = 0.5, 0.70


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def lags(dP, dT, c):
    """Per-receiver lag (Hz) and peak correlation for the window centred at bin c."""
    t = dT[:, c - HALF:c + HALF]
    t = t - t.mean(1, keepdims=True)
    tn = np.sqrt((t * t).sum(1))
    L = np.arange(-MAXLAG, MAXLAG + 1)
    R = np.empty((len(L), dP.shape[0]))
    for j, l in enumerate(L):
        p = dP[:, c - HALF + l:c + HALF + l]
        p = p - p.mean(1, keepdims=True)
        R[j] = (p * t).sum(1) / np.maximum(np.sqrt((p * p).sum(1)) * tn, 1e-12)
    k = R.argmax(0)
    lag = L[k].astype(float)
    inner = (k > 0) & (k < len(L) - 1)
    i = np.nonzero(inner)[0]
    y0, y1, y2 = R[k[i] - 1, i], R[k[i], i], R[k[i] + 1, i]
    den = y0 - 2 * y1 + y2
    lag[i] += np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-12, den, 1), 0.0)
    return lag * DF, R.max(0), ~inner


def room_stats(lag, peak, edge):
    m = float(np.median(lag))
    s = float(np.mean(np.sign(lag) == np.sign(m))) if m != 0 else 0.0
    return {"median_lag_hz": m, "sign_consistency": s, "iqr_hz": [float(np.percentile(lag, 25)),
                                                                  float(np.percentile(lag, 75))],
            "mean_abs_lag_hz": float(np.mean(np.abs(lag))), "mean_peak_corr": float(np.mean(peak)),
            "frac_edge": float(np.mean(edge)), "n_rx": int(len(lag)),
            "systematic": bool(abs(m) >= SYS_LAG and s >= SYS_SIGN)}


def render(name, ck):
    import torch

    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    dev = torch.device("cuda")
    model, renderer, cfg, _m, it = load_model(Path(ck), dev)
    model.eval(); renderer.eval()
    blob = {}
    for i, c in enumerate(rooms()):
        with h5py.File(DATA / c.filename) as f:
            T = np.asarray(f["ism/H_complex"])[:, :601]
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        off = ~on_lattice(rx)
        T, rx = T[off], rx[off]
        cond = build_cond_vector_2d("geom_token", c.L, c.W, c.edge_alphas, verts=c.verts, device=dev)
        R = torch.tensor(rx, dtype=torch.float32, device=dev)
        S = torch.tensor(np.asarray(SRC, float), dtype=torch.float32, device=dev).unsqueeze(0)
        P = np.zeros_like(T)
        for s0 in range(0, len(rx), 8):
            r = R[s0:s0 + 8]
            with torch.no_grad():
                H = renderer(model, r, S.expand(r.shape[0], -1), torch.zeros(2, device=dev),
                             torch.tensor([c.L, c.W], device=dev, dtype=torch.float32),
                             z_s=cond.unsqueeze(0).expand(r.shape[0], -1))
            P[s0:s0 + 8] = H[:, :601].cpu().numpy()
        blob["P%d" % i], blob["T%d" % i], blob["rx%d" % i] = P, T, rx
        blob["L%d" % i], blob["label%d" % i] = c.L, c.label
        print("  rendered", c.label, len(rx), flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "t1_train_{}.npz".format(name), key="{}|{}".format(ck, it), **blob)


def analyse_set(rooms_iter):
    out = {k: [] for k in CENTRES}
    for label, L, P, T in rooms_iter:
        dP, dT = db(P), db(T)
        for k, c in CENTRES.items():
            lag, peak, edge = lags(dP, dT, c)
            st = room_stats(lag, peak, edge)
            st.update({"room": label, "L": float(L), "dL_eff_cm": float(-100.0 * L * st["median_lag_hz"] / int(k))})
            out[k].append(st)
    return out


def test_rooms(name):
    z = np.load(G2.format(name), allow_pickle=False)
    test = configs_from_rows(json.load(open(MANIFEST))["configs"], split="test")
    for i, c in enumerate(test):
        rx = z["rx%d" % i]
        off = ~on_lattice(rx)
        yield c.label, c.L, z["P%d" % i][off, :601], z["T%d" % i][off, :601]


def train_rooms(name):
    z = np.load(CACHE / "t1_train_{}.npz".format(name), allow_pickle=False)
    i = 0
    while "P%d" % i in z.files:
        yield str(z["label%d" % i]), float(z["L%d" % i]), z["P%d" % i], z["T%d" % i]
        i += 1


def verdict(test, train):
    n = len(test["250"])
    sys250 = sum(r["systematic"] for r in test["250"])
    both = [(a, b) for a, b in zip(test["125"], test["250"]) if a["systematic"] and b["systematic"]]
    ratio = float(np.median([abs(b["median_lag_hz"]) / abs(a["median_lag_hz"]) for a, b in both])) if both else float("nan")
    tr_med = {k: float(np.median([abs(r["median_lag_hz"]) for r in train[k]])) for k in CENTRES}
    tr_sys = {k: float(np.mean([r["systematic"] for r in train[k]])) for k in CENTRES}
    c1 = sys250 >= 2 * n / 3
    c2 = bool(both) and 1.5 <= ratio <= 2.5
    c3 = all(tr_med[k] <= 0.25 and tr_sys[k] <= 1 / 3 for k in CENTRES)
    return {"test_systematic_250": "{}/{}".format(sys250, n), "c1": bool(c1),
            "test_both_systematic": len(both), "ratio_250_over_125_median": ratio, "c2": bool(c2),
            "train_median_abs_lag_hz": tr_med, "train_frac_systematic": tr_sys, "c3": bool(c3),
            "mode_misplacement": bool(c1 and c2 and c3)}


def summary(rows):
    m = np.array([r["median_lag_hz"] for r in rows])
    return {"median_of_room_medians_hz": float(np.median(m)), "mean_abs_room_median_hz": float(np.mean(np.abs(m))),
            "frac_systematic": float(np.mean([r["systematic"] for r in rows])),
            "mean_sign_consistency": float(np.mean([r["sign_consistency"] for r in rows])),
            "mean_peak_corr": float(np.mean([r["mean_peak_corr"] for r in rows])),
            "mean_abs_lag_hz_per_rx": float(np.mean([r["mean_abs_lag_hz"] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default="")
    ap.add_argument("--analyse", default="")
    a = ap.parse_args()
    if a.render:
        name, ck = a.render.split("=", 1)
        render(name, ck)
        return
    rep = {"pre_registration": "D160(h)", "window_hz": HALF * DF, "max_lag_hz": MAXLAG * DF, "models": {}}
    for name in a.analyse.replace("+", ",").split(","):
        te, tr = analyse_set(test_rooms(name)), analyse_set(train_rooms(name))
        v = verdict(te, tr)
        rep["models"][name] = {"test": te, "train": tr, "verdict": v,
                               "summary": {s: {k: summary(d[k]) for k in CENTRES} for s, d in (("test", te), ("train", tr))}}
        for s, d in (("test", te), ("train", tr)):
            for k in CENTRES:
                sm = summary(d[k])
                print("{} {:5s} {:>3s} Hz: room-median lag median {:+.2f} Hz, mean|.| {:.2f}; systematic {:.2f}; "
                      "sign-consistency {:.2f}; peak r {:.2f}; per-rx mean|lag| {:.2f} Hz".format(
                          name, s, k, sm["median_of_room_medians_hz"], sm["mean_abs_room_median_hz"],
                          sm["frac_systematic"], sm["mean_sign_consistency"], sm["mean_peak_corr"],
                          sm["mean_abs_lag_hz_per_rx"]), flush=True)
        print(name, "VERDICT", json.dumps(v), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t1_lag.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
