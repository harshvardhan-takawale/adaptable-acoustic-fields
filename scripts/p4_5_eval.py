"""P4-5 evaluation, identical for the baseline and every arm: is the high-frequency tail fitted?

THE PRIMARY METRIC IS A CURVE, NOT A NUMBER. Spatial Pearson (predicted vs FDTD |H| in dB, across
receivers) at EVERY 0.5 Hz bin from 25 to 300 Hz -- 551 bins, none selected. Prediction and
truth are always read at the same bin, so the comparison is valid whether or not a mode sits
there (D79); no modal enumeration is attempted, because a non-rectangular room has no analytic
modes. This is the Arm C v2 mode screen (`scripts/armC_v2_mode_screen.py`, the spec's
"appendix A9") ported bin by bin.

Two curves, because receiver density is not neutral (P4-4 demo v2, §5.3: the same rooms read up
to 0.22 higher on 800 scattered receivers than on a dense grid):
  * DENSE  -- 3 frozen test shapes on a 64x64 lattice (`data/track_p4_5_testdense`), chosen by
              the pre-registered rule in `build_p4_5_rooms.test_shapes_for_a2`. The headline.
  * SPARSE -- all 15 frozen test shapes at their existing 800 receivers. Broader, and on the 3
              overlapping shapes the sparse-vs-dense gap is measured directly.

"r at 50/100/150/200/250 Hz" means the MEAN over +-2.5 Hz (11 bins), pre-registered so that no
lucky bin between two resonances can be picked. Receiver-bootstrap 95% CIs on those means.
Limitation, stated: no training-seed replicates exist, so the CI covers receiver sampling only.

Also per checkpoint: band LSD split 0-100 / 100-200 / 200-300 Hz (bins [0,200) [200,400)
[400,601)), and RIR Pearson TWO ways -- Gate-2 style (receiver 0, full-band inverse transform,
comparable with D70's 0.9989 -> 0.0630) and modal (20-300 Hz, every receiver) -- because a fix
that lifts high frequency while wrecking the time response is not a fix. The frozen Gate-2
numbers come from `scripts/p4_2_stage2_eval.py`, run unchanged, not from here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from aaf.data.shape_configs import configs_from_rows
from aaf.eval.modal_decay import band_limited_rir
from aaf.eval.p3_2_eval import load_model
from scripts.build_p4_5_rooms import TEST_MANIFEST, test_shapes_for_a2
from scripts.p4_3_demo_pack import render

DF = 0.5
N_BINS = 601
F_LO, F_HI = 25.0, 300.0
MARKS_HZ = (50.0, 100.0, 150.0, 200.0, 250.0)
HALF_WIN = 5                                    # +-5 bins = +-2.5 Hz -> 11-bin window
SUBBANDS = {"0-100": (0, 200), "100-200": (200, 400), "200-300": (400, N_BINS)}
RIR_FS, RIR_N, MODAL_LO = 600.0, 1200, 20.0     # = p4_3_demo_pack / p4_4_family_eval
N_BOOT = 200
DENSE_DIR = "data/track_p4_5_testdense"
SPARSE_DIR = "data/track_p4_2_shapes"


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def _pearson_cols(A, B):
    """Pearson per column (bin), vectorised. A, B: [n_rx, n_bins] real."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    den = np.sqrt((A * A).sum(0) * (B * B).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (A * B).sum(0) / den, np.nan)


def _curve(P, T):
    return {"db": _pearson_cols(_db(P), _db(T)), "lin": _pearson_cols(np.abs(P), np.abs(T))}


def _marks(r):
    out = {}
    for f in MARKS_HZ:
        b = int(round(f / DF))
        out["{:g}".format(f)] = float(np.nanmean(r[b - HALF_WIN:b + HALF_WIN + 1]))
    return out


def _boot_marks(P, T, rng):
    """95% CI of each window mean, resampling RECEIVERS with replacement."""
    n = P.shape[0]
    idx_b = [int(round(f / DF)) for f in MARKS_HZ]
    cols = sorted({b + k for b in idx_b for k in range(-HALF_WIN, HALF_WIN + 1)})
    Pd, Td = _db(P[:, cols]), _db(T[:, cols])
    pos = {c: i for i, c in enumerate(cols)}
    draws = np.zeros((N_BOOT, len(MARKS_HZ)))
    for k in range(N_BOOT):
        s = rng.integers(0, n, n)
        r = _pearson_cols(Pd[s], Td[s])
        for j, b in enumerate(idx_b):
            draws[k, j] = np.nanmean([r[pos[b + d]] for d in range(-HALF_WIN, HALF_WIN + 1)])
    return {"{:g}".format(f): [float(np.percentile(draws[:, j], 2.5)),
                               float(np.percentile(draws[:, j], 97.5))]
            for j, f in enumerate(MARKS_HZ)}, draws


def _rooms(cfgs, data_dir, model, renderer, dev):
    out = []
    for tag, c in cfgs:
        with h5py.File(Path(data_dir) / c.filename) as f:
            T = np.asarray(f["ism/H_complex"][:, :N_BINS])
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        P = render(model, renderer, c, rx, N_BINS, dev)
        out.append({"tag": tag, "cfg": c, "P": P, "T": T, "n_rx": len(rx)})
        print("  rendered {:8s} {:30s} n_rx {:5d}".format(tag, c.filename, len(rx)), flush=True)
    return out


def load_or_render(ck, model, renderer, dev, cache):
    """Cache keyed on checkpoint AND data directories (D89: the dense test rooms share filenames
    with the 800-receiver corpus, so a key without the directory would alias them)."""
    key = "|".join([str(ck), DENSE_DIR, SPARSE_DIR])
    dense_cfgs = test_shapes_for_a2()
    sparse_cfgs = [(c.filename, c) for c in sorted(
        configs_from_rows(json.load(open(TEST_MANIFEST))["configs"], split="test"),
        key=lambda c: c.filename)]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            dense = [{"tag": t, "cfg": c, "P": z["dP{}".format(i)], "T": z["dT{}".format(i)],
                      "n_rx": int(z["dP{}".format(i)].shape[0])}
                     for i, (t, c) in enumerate(dense_cfgs)]
            sparse = [{"tag": t, "cfg": c, "P": z["sP{}".format(i)], "T": z["sT{}".format(i)],
                       "n_rx": int(z["sP{}".format(i)].shape[0])}
                      for i, (t, c) in enumerate(sparse_cfgs)]
            print("[cache] reused", cache, flush=True)
            return dense, sparse
    print("dense (3 frozen test shapes, 64x64 lattice):", flush=True)
    dense = _rooms(dense_cfgs, DENSE_DIR, model, renderer, dev)
    print("sparse (15 frozen test shapes, 800 receivers):", flush=True)
    sparse = _rooms(sparse_cfgs, SPARSE_DIR, model, renderer, dev)
    blob = {"key": key}
    for i, r in enumerate(dense):
        blob["dP{}".format(i)], blob["dT{}".format(i)] = r["P"], r["T"]
    for i, r in enumerate(sparse):
        blob["sP{}".format(i)], blob["sT{}".format(i)] = r["P"], r["T"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **blob)
    return dense, sparse


def evaluate(dense, sparse):
    rng = np.random.default_rng(0)
    lo, hi = int(round(F_LO / DF)), N_BINS
    rep = {"freqs_hz": (np.arange(N_BINS) * DF).tolist(), "band_hz": [F_LO, F_HI],
           "mark_window_hz": HALF_WIN * DF * 2, "n_boot": N_BOOT}
    # ---- dense curve, per shape and pooled-by-mean ------------------------------------------
    dc = []
    rep["dense"] = {}
    for r in dense:
        cv = _curve(r["P"], r["T"])
        ci, _ = _boot_marks(r["P"], r["T"], rng)
        rep["dense"][r["tag"]] = {"room": r["cfg"].filename, "d_hat": float(r["cfg"].d_hat),
                                  "n_rx": r["n_rx"], "r_db": cv["db"].tolist(),
                                  "r_lin": cv["lin"].tolist(), "marks": _marks(cv["db"]),
                                  "marks_ci95": ci}
        dc.append(cv["db"])
    mean_dense = np.nanmean(np.stack(dc), axis=0)
    rep["dense_mean_r_db"] = mean_dense.tolist()
    rep["dense_mean_marks"] = _marks(mean_dense)
    # pooled CI for the 3-shape mean: resample receivers independently within each shape
    draws = []
    for r in dense:
        _ci, d = _boot_marks(r["P"], r["T"], np.random.default_rng(1))
        draws.append(d)
    md = np.mean(np.stack(draws), axis=0)
    rep["dense_mean_marks_ci95"] = {"{:g}".format(f): [float(np.percentile(md[:, j], 2.5)),
                                                       float(np.percentile(md[:, j], 97.5))]
                                    for j, f in enumerate(MARKS_HZ)}
    rep["dense_mean_marks_boot"] = md.tolist()      # kept so arm-vs-baseline CIs can be formed
    rep["dense_mean_r_25_300"] = float(np.nanmean(mean_dense[lo:hi]))
    # ---- sparse curve over the 15 frozen test shapes ----------------------------------------
    sc = np.stack([_curve(r["P"], r["T"])["db"] for r in sparse])
    mean_sparse = np.nanmean(sc, axis=0)
    rep["sparse_mean_r_db"] = mean_sparse.tolist()
    rep["sparse_mean_marks"] = _marks(mean_sparse)
    rep["sparse_mean_r_25_300"] = float(np.nanmean(mean_sparse[lo:hi]))
    # same 3 shapes, sparse vs dense -- the §5.3 sampling effect on frozen data
    dense_files = {r["cfg"].filename for r in dense}
    ov = [_curve(r["P"], r["T"])["db"] for r in sparse if r["cfg"].filename in dense_files]
    rep["sparse_on_dense_shapes_marks"] = _marks(np.nanmean(np.stack(ov), axis=0))
    # ---- sub-band LSD and RIR on the 15 frozen test shapes ----------------------------------
    rep["band_lsd_db"] = {}
    for name, (b0, b1) in SUBBANDS.items():
        rep["band_lsd_db"][name] = float(np.mean([np.mean(np.abs(_db(r["P"][:, b0:b1])
                                                                  - _db(r["T"][:, b0:b1])))
                                                  for r in sparse]))
    rep["band_lsd_db"]["0-300"] = float(np.mean([np.mean(np.abs(_db(r["P"]) - _db(r["T"])))
                                                 for r in sparse]))
    g2, mod = [], []
    for r in sparse:
        n = 2 * (N_BINS - 1)
        a, b = np.fft.irfft(r["P"][0], n=n), np.fft.irfft(r["T"][0], n=n)
        g2.append(float(np.corrcoef(a, b)[0, 1]))
        rp = band_limited_rir(r["P"], RIR_FS, RIR_N, f_lo=MODAL_LO, f_hi=F_HI)
        rt = band_limited_rir(r["T"], RIR_FS, RIR_N, f_lo=MODAL_LO, f_hi=F_HI)
        mod.append(float(np.nanmean([np.corrcoef(rp[i], rt[i])[0, 1] for i in range(len(rp))])))
    rep["rir_pearson_gate2_style"] = float(np.mean(g2))
    rep["rir_pearson_modal_20_300"] = float(np.mean(mod))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True, help="e.g. baseline_60K, B1_30K")
    ap.add_argument("--out", default="outputs/p4_5/eval")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = Path(a.checkpoint)
    model, renderer, cfg, _meta, it = load_model(ck, dev)
    active = int(model.level_mask.sum())
    print("[ckpt] {} iter {} | pool {} | active hash levels {}/{}".format(
        ck, it, cfg.get("token_pool"), active, model.level_mask.numel()), flush=True)
    out = Path(a.out)
    dense, sparse = load_or_render(ck, model, renderer, dev,
                                   out / "cache" / "render_{}.npz".format(a.name))
    rep = evaluate(dense, sparse)
    rep.update({"name": a.name, "checkpoint": str(ck), "iter": it,
                "active_hash_levels": active, "token_pool": cfg.get("token_pool"),
                "loss_spec_bin_weights": cfg.get("loss_spec_bin_weights", ""),
                "loss_curriculum": cfg.get("loss_curriculum", ""),
                "c2f_levels": cfg.get("c2f_levels", ""), "data_dir": cfg.get("data_dir")})
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out / "{}.json".format(a.name), "w"), indent=1)
    m, ci = rep["dense_mean_marks"], rep["dense_mean_marks_ci95"]
    print("DENSE (3 shapes) r_db at:  " + "  ".join(
        "{} Hz {:+.4f} [{:+.3f},{:+.3f}]".format(k, m[k], *ci[k]) for k in m), flush=True)
    print("SPARSE (15 shapes) r_db:   " + "  ".join(
        "{} Hz {:+.4f}".format(k, v) for k, v in rep["sparse_mean_marks"].items()), flush=True)
    print("sparse on the 3 dense shapes: " + "  ".join(
        "{} Hz {:+.4f}".format(k, v) for k, v in rep["sparse_on_dense_shapes_marks"].items()))
    print("band LSD dB: " + "  ".join("{} {:.2f}".format(k, v)
                                      for k, v in rep["band_lsd_db"].items()))
    print("RIR Pearson: gate-2 style {:+.4f} | modal 20-300 Hz {:+.4f}".format(
        rep["rir_pearson_gate2_style"], rep["rir_pearson_modal_20_300"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
