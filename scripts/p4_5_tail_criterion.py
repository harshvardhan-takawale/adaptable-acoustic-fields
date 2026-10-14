"""P4-5: the pre-registered "moved the tail" criterion, scored at 60K.

Plan (pre-registered before any arm trained): an arm MOVED THE TAIL if arm minus baseline at
200 Hz exceeds the receiver-bootstrap 95% CI of the difference. "At 200 Hz" is the A2 mark: the
mean of the per-bin dB spatial Pearson over +-2.5 Hz (11 bins, 197.5-202.5 Hz), on the 3 frozen
dense test shapes, averaged over shapes. Baseline: C250 @ 60K, not retrained.

Since D99 / Rule 11 the PRIMARY figure is OFF the training lattice; the all-receiver version,
the literal pre-registered one, is reported beside it. The bootstrap is PAIRED: each resample
draws one set of receiver indices per shape and scores both models on it, so the CI is the CI of
the difference. 2000 resamples, seed 20261013. Stated limitation (plan): no seed replicates, so
the CI covers receiver sampling only, not training-seed variance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.build_p4_5_rooms import test_shapes_for_a2
from scripts.p4_5_eval import DENSE_DIR, _db

B0, HALF, NB, SEED = 400, 5, 2000, 20261013
CACHE = "outputs/p4_5/eval/cache/render_{}.npz"


def onlat(rx):
    k = (rx - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def colr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def load(name):
    z = np.load(CACHE.format(name), allow_pickle=False)
    sl = slice(B0 - HALF, B0 + HALF + 1)
    out = []
    for i, (_tag, c) in enumerate(test_shapes_for_a2()):
        with h5py.File(Path(DENSE_DIR) / c.filename) as f:
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        out.append((_db(z["dP%d" % i][:, sl]), _db(z["dT%d" % i][:, sl]), rx))
    return out


def score(arm, base, subset):
    rng = np.random.default_rng(SEED)
    pa, pb, idxs = [], [], []
    for (A, T, rx), (Bm, T2, rx2) in zip(arm, base):
        assert np.array_equal(rx, rx2) and np.allclose(T, T2)
        sel = np.flatnonzero(~onlat(rx)) if subset == "off" else np.arange(len(rx))
        pa.append(A[sel]); pb.append(Bm[sel]); idxs.append(T[sel])
    ra = np.mean([np.nanmean(colr(a, t)) for a, t in zip(pa, idxs)])
    rb = np.mean([np.nanmean(colr(b, t)) for b, t in zip(pb, idxs)])
    d = []
    for _ in range(NB):
        da, db_ = [], []
        for a, b, t in zip(pa, pb, idxs):
            i = rng.integers(0, len(t), len(t))
            da.append(np.nanmean(colr(a[i], t[i]))); db_.append(np.nanmean(colr(b[i], t[i])))
        d.append(np.mean(da) - np.mean(db_))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"arm_r200": float(ra), "baseline_r200": float(rb), "diff": float(ra - rb),
            "diff_ci95": [float(lo), float(hi)], "moved_the_tail": bool(lo > 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--baseline", default="baseline_60K")
    ap.add_argument("--out", default="outputs/p4_5/TAIL_CRITERION.json")
    a = ap.parse_args()
    base = load(a.baseline)
    rep = json.load(open(a.out)) if Path(a.out).exists() else {"method": __doc__.strip().splitlines()[0], "rows": {}}
    for n in a.arms.split(","):
        arm = load(n)
        rep["rows"][n] = {s: score(arm, base, s) for s in ("off", "all")}
        for s, r in rep["rows"][n].items():
            print("{:10s} {:3s} r200 {:+.3f} vs baseline {:+.3f}: diff {:+.3f} [{:+.3f}, {:+.3f}] -> {}".format(
                n, s, r["arm_r200"], r["baseline_r200"], r["diff"], *r["diff_ci95"],
                "MOVED THE TAIL" if r["moved_the_tail"] else "not moved"))
    json.dump(rep, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
