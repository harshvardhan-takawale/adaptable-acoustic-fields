"""P4-5: the per-bin spec-term weight vectors for B1 and B1b. One file per arm, one trainer path.

B1 (`rms`) -- the spec as written: w_b = 1 / max(RMS_b, floor), where RMS_b is the RMS of the
target spectrum at bin b over the 250 training configs and their TRAINING receivers only (the
`val_rx_indices` hold-out is excluded, so validation never informed the weights). The floor is
the 5th percentile of RMS across bins, so no near-silent bin can be amplified without bound.
D92 records why this RE-WEIGHTS toward high frequency rather than equalising: the spec terms
are L1, whose output gradient is already uniform across bins.

B1b (`calibrated`) -- the stated intent, "every frequency contributes comparable gradient",
using A1's MEASURED spec-term parameter-gradient norm share n_G per 10 Hz band at C250 @ 60K:
w_G = (|G| / 601) / n_G, so a band's share of gradient becomes proportional to its width.
The norm share, not the projection share: a projection can be zero or negative and cannot be
inverted into a weight. The max/min weight ratio is CAPPED at B1's, so neither arm is allowed a
more extreme re-weighting than the other and the comparison is about the SHAPE of w.

ONE SCALING RULE FOR BOTH, stated: w is multiplied by c so that mean_b(c w_b m_b) = mean_b(m_b),
m_b = mean |T_b|. At initialisation the error is roughly the target, so the weighted spec loss
then equals the unweighted one -- preserving its balance against L_amp and L_phase and the
grad-clip regime. (Adam is otherwise invariant to an overall loss scale, so this is about the
balance and the clipping, not the learning rate.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from aaf.data.shape_configs import configs_from_rows

N_BINS = 601
FLOOR_PCT = 5.0
MANIFEST = "configs/sweeps_2d_mat/p4_4_shapes_250.json"
DATA = "data/track_p4_2_shapes"


def _val_rx(n_rx):
    return set(range(3, n_rx, 8))               # = multi_room_2d_mat.val_rx_indices


def corpus_stats():
    rows = json.load(open(MANIFEST))["configs"]
    s2, s1, n = np.zeros(N_BINS), np.zeros(N_BINS), 0
    for c in configs_from_rows(rows, split="train"):
        with h5py.File(Path(DATA) / c.filename) as f:
            H = np.asarray(f["ism/H_complex"][:, :N_BINS])
        keep = [i for i in range(H.shape[0]) if i not in _val_rx(H.shape[0])]
        A = np.abs(H[keep])
        s2 += (A ** 2).sum(0)
        s1 += A.sum(0)
        n += len(keep)
    return np.sqrt(s2 / n), s1 / n, n


def _scale(w, m):
    return w * (m.mean() / (w * m).mean())


def rms_weights():
    rms, m, n = corpus_stats()
    floor = float(np.percentile(rms, FLOOR_PCT))
    w = _scale(1.0 / np.maximum(rms, floor), m)
    info = {"arm": "B1", "rule": "w = 1/max(RMS_b, P{:g}(RMS)), scaled".format(FLOOR_PCT),
            "n_rows": int(n), "floor": floor, "floor_pct": FLOOR_PCT,
            "n_bins_floored": int((rms < floor).sum()),
            "rms_min": float(rms.min()), "rms_max": float(rms.max()),
            "rms_bin_argmin_hz": float(rms.argmin() * 0.5),
            "w_min": float(w.min()), "w_max": float(w.max()),
            "max_over_min": float(w.max() / w.min()),
            "check_mean_wm_over_mean_m": float((w * m).mean() / m.mean())}
    return w.astype(np.float32), info, rms, m


def calibrated_weights(a1_json, b1_ratio):
    rms, m, _n = corpus_stats()
    a1 = json.load(open(a1_json))["states"]["C250_60K"]
    edges = a1["band_edges_hz"]
    share = np.asarray(a1["param_norm_share_spec"], float)
    w = np.zeros(N_BINS)
    for (lo, hi), sh in zip(edges, share):
        b0, b1 = int(round(lo / 0.5)), min(int(round(hi / 0.5)), N_BINS)
        if hi >= 300.0:
            b1 = N_BINS                          # bin 600 belongs to the last band
        width = (b1 - b0) / float(N_BINS)
        w[b0:b1] = width / max(sh, 1e-12)
    capped = int((w > w.min() * b1_ratio).sum())
    w = np.minimum(w, w.min() * b1_ratio)        # never more extreme than B1
    w = _scale(w, m)
    info = {"arm": "B1b", "rule": "w_G = (|G|/601) / spec-norm-share_G from A1 @ C250-60K, "
                                  "ratio capped at B1's, scaled",
            "a1_json": str(a1_json), "b1_ratio_cap": float(b1_ratio),
            "n_bins_capped": capped, "spec_norm_share_by_band": share.tolist(),
            "w_min": float(w.min()), "w_max": float(w.max()),
            "max_over_min": float(w.max() / w.min()),
            "check_mean_wm_over_mean_m": float((w * m).mean() / m.mean())}
    return w.astype(np.float32), info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("rms", "calibrated"))
    ap.add_argument("--a1", default="outputs/p4_5/diagnostics/a1_C250_60K.json")
    ap.add_argument("--b1-info", default="outputs/p4_5/arms/weights/B1_weights.json")
    ap.add_argument("--out", default="outputs/p4_5/arms/weights")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.kind == "rms":
        w, info, rms, m = rms_weights()
        np.save(out / "B1_weights.npy", w)
        info["rms_per_bin"] = rms.tolist()
        info["mean_abs_T_per_bin"] = m.tolist()
        json.dump(info, open(out / "B1_weights.json", "w"), indent=1)
        name = "B1"
    else:
        ratio = json.load(open(a.b1_info))["max_over_min"]
        w, info = calibrated_weights(a.a1, ratio)
        np.save(out / "B1b_weights.npy", w)
        json.dump(info, open(out / "B1b_weights.json", "w"), indent=1)
        name = "B1b"
    print("{}: w min {:.4g} max {:.4g} ratio {:.1f}  scale check {:.6f} (must be 1)".format(
        name, info["w_min"], info["w_max"], info["max_over_min"],
        info["check_mean_wm_over_mean_m"]))
    for k in ("n_bins_floored", "floor", "rms_bin_argmin_hz", "n_bins_capped"):
        if k in info:
            print("  {} = {}".format(k, info[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
