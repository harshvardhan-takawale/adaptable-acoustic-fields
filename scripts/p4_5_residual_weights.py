"""P4-5 (manager, D129): ROUND-2 per-bin spec weights, from where B1 is STILL wrong at 60K.

B1's vector (D96) was w = 1/RMS(target), derived before anything was known about where the
model would fail; across 200-250 Hz it is roughly flat (10.8-11.8). Round 2, the user's decision:
RELATIVE-RESIDUAL weighting,

    w_b = w_B1(b) * e_b,    e_b = RMS(P_b - T_b) / RMS(T_b)

i.e. B1's normalisation times how wrong B1 @60K still is, relative to the target, in each bin.
Bins B1 already fits lose weight; bins it still misses -- the top of the band -- keep or gain it.

Rows = what the loss sees, the same population as D96: training configs (every `--every`-th of the
250 by filename; run for B12 with --every 10 = 25 configs, 17,500 rows), their TRAINING receivers (`val_rx_indices` excluded), rendered with the
polygon path of `p4_5_gate2_dense.render_all`. Residual is complex (the spec terms are L1 on real
and imaginary parts), RMS over rows.

Then D96 / B1b's rules: e floored at its 5th percentile (no bin's weight driven to ~0 by a lucky
fit); max/min ratio capped at B1's (3146.8) by RAISING THE MINIMUM -- the cap protects the bins
being de-emphasised, never the top-of-band bins the vector exists to press on; finally D92's
scaling, mean_b(w_b m_b) = mean_b(m_b), m_b = mean |T_b| over the same rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.p4_5_bin_weights import FLOOR_PCT, MANIFEST, N_BINS, _scale, _val_rx

B1_W = "outputs/p4_5/arms/weights/B1_weights.npy"
DATA = Path("data/track_p4_2_shapes")
BANDS = {"0-10": (0, 21), "10-100": (21, 200), "100-200": (200, 400), "200-300": (400, 601)}


def main() -> int:
    import torch

    from aaf.data.shape_configs import SRC, configs_from_rows
    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/p4_5/arms/B1/ckpt_iter0060000.pt")
    ap.add_argument("--out", default="outputs/p4_5/arms/weights/B12_weights")
    ap.add_argument("--every", type=int, default=5)
    a = ap.parse_args()
    dev = torch.device("cuda")
    model, renderer, cfg, _m, it = load_model(Path(a.checkpoint), dev)
    model.eval(); renderer.eval()
    assert model.token_pool == cfg.get("token_pool")
    cs = sorted(configs_from_rows(json.load(open(MANIFEST))["configs"], split="train"), key=lambda c: c.filename)[::a.every]
    sr, st, sm, n = np.zeros(N_BINS), np.zeros(N_BINS), np.zeros(N_BINS), 0
    S = torch.tensor(np.asarray(SRC, float), dtype=torch.float32, device=dev).unsqueeze(0)
    for c in cs:
        with h5py.File(DATA / c.filename) as f:
            T = np.asarray(f["ism/H_complex"])[:, :N_BINS]
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        keep = np.array([i for i in range(len(rx)) if i not in _val_rx(len(rx))])
        T, rx = T[keep], rx[keep]
        cond = build_cond_vector_2d("geom_token", c.L, c.W, c.edge_alphas, verts=c.verts, device=dev)
        R = torch.tensor(rx, dtype=torch.float32, device=dev)
        P = np.zeros_like(T)
        for s0 in range(0, len(rx), 8):
            r = R[s0:s0 + 8]
            with torch.no_grad():
                H = renderer(model, r, S.expand(r.shape[0], -1), torch.zeros(2, device=dev),
                             torch.tensor([c.L, c.W], device=dev, dtype=torch.float32),
                             z_s=cond.unsqueeze(0).expand(r.shape[0], -1))
            P[s0:s0 + 8] = H[:, :N_BINS].cpu().numpy()
        sr += (np.abs(P - T) ** 2).sum(0); st += (np.abs(T) ** 2).sum(0); sm += np.abs(T).sum(0); n += len(T)
        print("  rendered {} ({} rows)".format(c.label, len(T)), flush=True)
    rms_res, rms_t, m = np.sqrt(sr / n), np.sqrt(st / n), sm / n
    e = rms_res / rms_t
    e_floor = float(np.percentile(e, FLOOR_PCT))
    w1 = np.load(B1_W).astype(np.float64)
    ratio_cap = float(w1.max() / w1.min())
    w = w1 * np.maximum(e, e_floor)
    n_raised = int((w < w.max() / ratio_cap).sum())
    w = np.maximum(w, w.max() / ratio_cap)
    w = _scale(w, m)
    w1s = _scale(w1, m)

    def shares(v):
        tot = v.sum()
        return {b: float(v[lo:hi].sum() / tot) for b, (lo, hi) in BANDS.items()}

    info = {"arm": "B12", "rule": "w = w_B1 * max(e, P{:g}(e)), e = RMS(P-T)/RMS(T) of B1 @60K; ratio capped at B1's by raising "
                                  "the minimum; D92 scaling".format(FLOOR_PCT),
            "source_checkpoint": a.checkpoint, "source_iter": int(it), "n_configs": len(cs), "n_rows": int(n),
            "e_floor": e_floor, "n_bins_e_floored": int((e < e_floor).sum()), "ratio_cap": ratio_cap,
            "n_bins_min_raised": n_raised, "w_min": float(w.min()), "w_max": float(w.max()),
            "max_over_min": float(w.max() / w.min()),
            "check_mean_wm_over_mean_m": float((w * m).mean() / m.mean()),
            "weight_share_by_band": {"B12": shares(w), "B1": shares(w1s)},
            "e_by_band": {b: float(e[lo:hi].mean()) for b, (lo, hi) in BANDS.items()},
            "w_at_hz": {str(hz): [float(w1s[int(hz * 2)]), float(w[int(hz * 2)])] for hz in (0, 5, 10, 50, 100, 150, 200, 250, 300)},
            "rms_res_per_bin": rms_res.tolist(), "rms_target_per_bin": rms_t.tolist(), "e_per_bin": e.tolist(),
            "weights": [float(x) for x in w]}
    np.save(a.out + ".npy", w.astype(np.float32))
    info["sha256_npy"] = hashlib.sha256(open(a.out + ".npy", "rb").read()).hexdigest()
    json.dump(info, open(a.out + ".json", "w"), indent=1)
    print("relative residual e by band:", {k: round(v, 3) for k, v in info["e_by_band"].items()})
    print("weight share by band  B1 :", {k: round(v, 4) for k, v in info["weight_share_by_band"]["B1"].items()})
    print("weight share by band  B12:", {k: round(v, 4) for k, v in info["weight_share_by_band"]["B12"].items()})
    print("w at Hz (B1, B12):", {k: (round(v[0], 4), round(v[1], 4)) for k, v in info["w_at_hz"].items()})
    print("max/min {:.1f} (cap {:.1f}), D92 check {:.6f}, sha {}".format(
        info["max_over_min"], ratio_cap, info["check_mean_wm_over_mean_m"], info["sha256_npy"][:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
