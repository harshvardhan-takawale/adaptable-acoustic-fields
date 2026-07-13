"""P3-1 edit-mechanism head-to-head eval (band-limited 0-300 Hz).

Arm-agnostic zero-shot eval on the frozen interior test set. Reuses the Phase-2
primitives; adds the in-band suite + the 3D mode-shape extractor + band-limited RIR.
No per-room optimization for any arm (L via RBF lookup; G/G+ via direct feature compute).

Usage:
  python -m aaf.eval.p3_1_eval --train-dir outputs/p3_1/arm_Gplus \
      --out outputs/p3_1/eval/arm_Gplus_best --rooms-yaml configs/sweeps_3d/test_rooms_interior_frozen.yaml
"""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path

import numpy as np
import torch

from aaf.eval.zero_shot_3d import _load_trained_model
from aaf.eval.known_geometry import _build_renderer, render_full, _load_room, build_lookup_maps
from aaf.eval.band_limited import band_indices, compute_band_limited_metrics
from aaf.eval.signal_level import (
    magnitude_correlation, phase_correlation_mag_weighted, rir_pearson, envelope_corr,
)
from aaf.eval.spatial_modes import bin_index_for_freq, spatial_correlation_complex
from aaf.eval.modal_verifier import pick_peaks, modal_error_metrics
from aaf.sim.analytical_modal_3d import eigenfrequencies_3d, C_DEFAULT
from aaf.models.conditioning import fourier_features, eigen_features, resonance_map

SRC = np.array([0.5, 0.5, 0.5], dtype=np.float32)   # source offset (matches sim)
PROM_DB, MIN_DIST_HZ, TOL_HZ, TOL_PCT = 3.0, 2.0, 4.0, 0.02   # Phase-2 settings


# ---------------------------------------------------------------- metrics
def band_limited_rir(H: np.ndarray, fs: float, n_time: int, f_hi: float = 300.0) -> np.ndarray:
    """Zero all bins > f_hi (identical mask both sides), force DC real, irfft."""
    _, hi = band_indices(fs, H.shape[-1], 0.0, f_hi)
    Hf = H.copy(); Hf[..., hi:] = 0.0
    Hf[..., 0] = Hf[..., 0].real
    return np.fft.irfft(Hf, n=n_time, axis=-1).astype(np.float32)


def _distinct_modes(L, W, Hd, f_max):
    return [m for m in eigenfrequencies_3d(L, W, Hd, c=C_DEFAULT, f_max=f_max) if m.f > 0.0]


def modal_placement(H_center, f_axis, L, W, Hd, f_max):
    sel = f_axis <= f_max
    peaks = pick_peaks(H_center[sel], f_axis[sel], prominence_db=PROM_DB, min_distance_hz=MIN_DIST_HZ)
    modes = _distinct_modes(L, W, Hd, f_max)
    m = modal_error_metrics(peaks, modes, tolerance_hz=TOL_HZ, tolerance_pct=TOL_PCT)
    n_pick = m.get("n_picked", len(peaks))
    return {
        "recall": float(m.get("recall_at_tol", 0.0)),
        "precision": float(m.get("n_matched", 0) / max(n_pick, 1)),
        "mae_hz": float(m.get("mae_hz", float("nan"))),
        "n_picked": int(n_pick), "n_analytical": int(m.get("n_analytical", len(modes))),
        "n_matched": int(m.get("n_matched", 0)),
    }


def mode_shape_corrs(H_pred, H_target, L, W, Hd, fs, n_freq, n_modes=6, f_max=300.0):
    modes = _distinct_modes(L, W, Hd, f_max)[:n_modes]
    out = []
    for md in modes:
        b = bin_index_for_freq(md.f, fs, n_freq)
        out.append({"f": float(md.f),
                    "corr": float(spatial_correlation_complex(H_pred[:, b], H_target[:, b]))})
    return out


def inband_signal(H_pred, H_target, fs, n_time, f_hi=300.0):
    lo, hi = band_indices(fs, H_pred.shape[-1], 0.0, f_hi)
    hp, ht = H_pred[:, lo:hi], H_target[:, lo:hi]
    band = compute_band_limited_metrics(H_pred, H_target, fs, H_pred.shape[-1], bands=((0.0, f_hi),))
    rp, rt = band_limited_rir(H_pred, fs, n_time, f_hi), band_limited_rir(H_target, fs, n_time, f_hi)
    return {
        "band_lsd_db": float(band.get(f"lsd_band_0_{int(f_hi)}_db",
                              band.get("lsd_band_0_300_db", float("nan")))),
        "phase_corr_mw": float(phase_correlation_mag_weighted(hp, ht)),
        "rir_pearson": float(rir_pearson(rp, rt)),
        "env_corr": float(envelope_corr(rp, rt)),
        "mag_corr": float(magnitude_correlation(hp, ht)),   # sanity (blur-inflatable), reported last
    }


def compute_room_suite(H_pred, H_target, receiver_pos, L, W, Hd, fs, n_time):
    n_freq = H_pred.shape[-1]
    f_axis = np.arange(n_freq) * (fs / n_time)
    c_rx = int(np.argmin(np.linalg.norm(receiver_pos - np.array([L/2, W/2, Hd/2]), axis=1)))
    d = {"L": L, "W": W, "H": Hd,
         "modal_300": modal_placement(H_pred[c_rx], f_axis, L, W, Hd, 300.0),
         "modal_250": modal_placement(H_pred[c_rx], f_axis, L, W, Hd, 250.0),
         "mode_shape": mode_shape_corrs(H_pred, H_target, L, W, Hd, fs, n_freq),
         **inband_signal(H_pred, H_target, fs, n_time)}
    d["mode_shape_mean"] = float(np.mean([m["corr"] for m in d["mode_shape"]]))
    return d


# ---------------------------------------------------------------- zero-shot render
def _zero_shot_cond(arm, L, W, Hd, model, train_meta, maps):
    if arm == "latent":
        return torch.as_tensor(maps["rbf"]((L, W, Hd)), dtype=torch.float32, device=model.w.device
                               if model.w is not None else next(model.parameters()).device)
    if arm == "geom_fourier":
        return fourier_features(L, W, Hd)
    if arm == "eigen":
        return eigen_features(L, W, Hd)
    raise ValueError(arm)


def run_arm_eval(train_dir, out_dir, rooms_yaml, data_dir="data/track_a_3d", device="cuda"):
    import yaml
    train_dir, out_dir = Path(train_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, meta = _load_trained_model(train_dir, device=device)
    cfg = meta["cfg"]; arm = cfg.get("cond_source", "latent")
    fs, n_time, n_freq = int(cfg["fs"]), int(cfg["n_time_samples"]), model.n_freq_bins
    renderer = _build_renderer(cfg, fs, n_time, device)
    dev = next(model.parameters()).device
    # arm L: build the RBF (L,W,H)->latent map from this model's trained latents
    maps = None
    if arm == "latent":
        train_LWH = np.stack([meta["L_list"], meta["W_list"], meta["H_list"]], axis=1)
        maps = build_lookup_maps(train_LWH, model.latents.weight.detach().cpu().numpy())
    # eigen arm: precompute padded resonance R per band
    _, hi = band_indices(fs, n_freq, 0.0, float(cfg.get("band_max_hz", 300.0)))
    rooms = yaml.safe_load(open(rooms_yaml))["rooms"]
    per_room, agg = {}, []
    for r in rooms:
        L, W, Hd = float(r["L"]), float(r["W"]), float(r["H"])
        h5 = Path(data_dir) / f"L{L:.2f}_W{W:.2f}_H{Hd:.2f}.h5"
        room = _load_room(h5)
        room_min = torch.zeros(3, device=dev)
        room_max = torch.tensor([L, W, Hd], dtype=torch.float32, device=dev)
        z = _zero_shot_cond(arm, L, W, Hd, model, meta, maps).to(dev)
        if arm == "eigen":
            R = torch.zeros(n_freq, device=dev)
            Rb = resonance_map(L, W, Hd, n_bins=hi, df=fs / n_time, device=dev)
            R[:Rb.numel()] = Rb
            model.set_resonance(R)
        H_pred = render_full(model, renderer, z, room_min, room_max,
                             room["receiver_pos"], room["src"], dev)
        suite = compute_room_suite(H_pred, room["H_target"], room["receiver_pos"], L, W, Hd, fs, n_time)
        key = f"L{L:.2f}_W{W:.2f}_H{Hd:.2f}"
        per_room[key] = suite
        agg.append(suite)
    # aggregate
    def mn(path):
        vals = [_get(s, path) for s in agg]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")
    summary = {
        "arm": arm, "n_rooms": len(agg), "train_dir": str(train_dir),
        "recall_300": mn("modal_300.recall"), "precision_300": mn("modal_300.precision"),
        "mae_hz_300": mn("modal_300.mae_hz"),
        "recall_250": mn("modal_250.recall"), "mae_hz_250": mn("modal_250.mae_hz"),
        "mode_shape_mean": mn("mode_shape_mean"),
        "band_lsd_db": mn("band_lsd_db"), "phase_corr_mw": mn("phase_corr_mw"),
        "rir_pearson": mn("rir_pearson"), "env_corr": mn("env_corr"), "mag_corr": mn("mag_corr"),
    }
    (out_dir / "per_room.json").write_text(json.dumps(per_room, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _get(d, path):
    cur = d
    for p in path.split("."):
        cur = cur[p] if isinstance(cur, dict) and p in cur else None
        if cur is None:
            return None
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rooms-yaml", default="configs/sweeps_3d/test_rooms_interior_frozen.yaml")
    ap.add_argument("--data-dir", default="data/track_a_3d")
    a = ap.parse_args()
    run_arm_eval(a.train_dir, a.out, a.rooms_yaml, a.data_dir)


if __name__ == "__main__":
    main()
