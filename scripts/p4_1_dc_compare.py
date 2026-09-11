"""P4-1: did masking the DC bins actually help? A fair before/after on ONE metric set.

The training-time val curves CANNOT answer this. After the model-selection fix, the masked run
reports `lsd_db` over bins 40-600 while the original A2 reports it over 0-600, so the two
numbers are not the same quantity. Comparing them directly would be the same class of mistake
the fix was made to prevent.

This scores BOTH checkpoints, at the SAME iteration, on the SAME metrics, over the SAME bins:

  * modal-band LSD  -- bins 40-600, everything above the 20 Hz cut. This is the band the masked
    run was trained on and the band the original was asked to fit as well; any benefit has to
    show up here.
  * modal-PEAK LSD  -- only the local maxima of the receiver-RMS spectrum. A raw band LSD
    averages over deep nulls, and this corpus has been misread that way before (D61d): a
    -40 dB peak-relative floor kept only bins 0-12 and measured the near-DC term, not modal
    content. Peaks are the closest thing to "error on what carries the physics".
  * full-band LSD   -- reported for completeness, expected to favour the unmasked run BY
    CONSTRUCTION since the masked run never fits bins 0-39.
  * spatial Pearson -- pred vs GT across receivers at each modal bin.

Iteration 12000 is the only checkpoint both runs share; the masked run early-stopped there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.p3_2_eval import load_model
from aaf.eval.p3_2b_eval import render_config_arm

CUT_BIN = 40          # 20 Hz at df = 0.5 Hz
DF_HZ = 0.5


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def _lsd(p, t):
    return float(np.mean(np.abs(_db(p) - _db(t))))


def _pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def peak_bins(T, lo=CUT_BIN):
    """Local maxima of the receiver-RMS spectrum, above the cut.

    Referenced to LOCAL maxima rather than the global peak on purpose: in this corpus the global
    peak IS the bin-0 compliance term, and anchoring to it is exactly what made the earlier
    floored-LSD measure the near-DC region instead of the modes (D61d).
    """
    rms = np.sqrt(np.mean(np.abs(T) ** 2, axis=0))
    out = [b for b in range(lo + 1, T.shape[1] - 1)
           if rms[b] > rms[b - 1] and rms[b] > rms[b + 1]]
    return np.array(out, dtype=int)


def score(ck, configs, data_dir, dev):
    model, renderer, cfg, meta, it = load_model(Path(ck), dev)
    model.eval()
    renderer.eval()                       # D49 C3
    cs = str(cfg["cond_source"])
    acc = {k: [] for k in ("lsd_full", "lsd_modal", "lsd_peak", "sp")}
    for c in configs:
        f = Path(data_dir) / c.filename
        if not f.exists():
            continue
        with h5py.File(f) as h:
            T = np.asarray(h["ism/H_complex"])
            rx = np.asarray(json.loads(h.attrs["receiver_pos"]))
            src = np.asarray(json.loads(h.attrs["source_pos"]))
        with torch.no_grad():
            P = np.asarray(render_config_arm(model, renderer, cs, c.L, c.W, c.alphas,
                                             rx, src, dev))[:, :T.shape[1]]
        acc["lsd_full"].append(_lsd(P, T))
        acc["lsd_modal"].append(_lsd(P[:, CUT_BIN:], T[:, CUT_BIN:]))
        pk = peak_bins(T)
        if pk.size:
            acc["lsd_peak"].append(_lsd(P[:, pk], T[:, pk]))
        for m in enumerate_modes(c.L, c.W, f_max=200.0)[:6]:
            b = int(round(m.f / DF_HZ))
            if CUT_BIN <= b < T.shape[1]:
                acc["sp"].append(_pearson(_db(P[:, b]), _db(T[:, b])))
    sc = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in acc.items()}
    return int(it), cs, sc, len(acc["lsd_full"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masked",
                    default="outputs/p4_1/dc_fix/p4_1_A2_dcmask/ckpt_iter0012000.pt")
    ap.add_argument("--baseline",
                    default="outputs/p3_3fast/p3_3fast_trackA2/ckpt_iter0012000.pt")
    ap.add_argument("--data-dir", default="data/track_p3_3fast_A")
    ap.add_argument("--manifest",
                    default="configs/sweeps_2d_mat/p3_3fast_trackA_manifest.json")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--out", default="outputs/p4_1/dc_fix/DC_COMPARISON.json")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from aaf.data.seg_configs import configs_from_rows
    rows = json.load(open(a.manifest))["configs"]
    configs = [c for c in configs_from_rows(rows) if getattr(c, "split", "") != "train"][:a.limit]
    print("[data] {} held-out configs".format(len(configs)), flush=True)

    res = {}
    for tag, ck in (("baseline_unmasked", a.baseline), ("dc_masked", a.masked)):
        it, cs, sc, n = score(ck, configs, a.data_dir, dev)
        row = {"checkpoint": ck, "iter": it, "cond_source": cs, "n_configs": n}
        row.update(sc)
        res[tag] = row
        print("  {:18s} iter {} | modal LSD {:.4f} | peak LSD {:.4f} | full LSD {:.4f} | "
              "spatial R {:+.4f}".format(tag, it, sc["lsd_modal"], sc["lsd_peak"],
                                         sc["lsd_full"], sc["sp"]), flush=True)

    b, m = res["baseline_unmasked"], res["dc_masked"]
    delta = {k: m[k] - b[k] for k in ("lsd_full", "lsd_modal", "lsd_peak", "sp")}
    helped = bool(delta["lsd_modal"] < 0 and delta["lsd_peak"] < 0)
    print("\n  delta (masked - unmasked; negative = masking HELPED for LSD):")
    for k in ("lsd_modal", "lsd_peak", "lsd_full", "sp"):
        print("    {:10s} {:+.4f}".format(k, delta[k]))
    print("\n  VERDICT: masking {} the modal band at matched iteration {}".format(
        "HELPED" if helped else "did NOT help", b["iter"]))

    json.dump({"comparison": "p4_1.dc_mask/1", "matched_iter": b["iter"],
               "cut_bin": CUT_BIN, "cut_hz": CUT_BIN * DF_HZ,
               "baseline_unmasked": b, "dc_masked": m,
               "delta_masked_minus_baseline": delta,
               "masking_helped_modal": helped,
               "note": ("Both checkpoints scored by ONE code path on IDENTICAL bins. The "
                        "training-time val curves cannot be compared directly: after the "
                        "model-selection fix the masked run reports lsd_db over bins 40-600 "
                        "and the baseline over 0-600, which are different quantities. "
                        "full-band LSD favours the baseline by construction.")},
              open(a.out, "w"), indent=1, default=float)
    print("-> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
