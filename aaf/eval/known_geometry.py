"""Known-geometry rendering + oracle-latent ceiling (Chunk P2-3.5).

Two routes to render an unseen room with the converged P3 model, both reusing
the frozen decoder + renderer (no retraining):

  lookup : predict the room's latent from its KNOWN (L,W,H) via a map fit on the
           45 training (latent, geometry) pairs — (a) RBF interpolation,
           (b) linear regression — then render. No measurement-based search.
  oracle : optimize z* against a rich receiver subset of the room (best latent
           the decoder can represent), then render. The ceiling.

Plus a leave-one-out (LOO) sanity on the training rooms: predict each training
room's latent from the other 44 and render — measures the (L,W,H)->latent route
on known-good interpolation points, independent of the test set.

Reuses aaf.eval.zero_shot_3d (_load_trained_model, _losses), signal_level,
band_limited, modal_verifier. Evaluates on ALL 512 receivers (known-geometry has
no "observed" set).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from aaf.data.dataset_builder import read_room_h5
from aaf.eval.zero_shot_3d import _load_trained_model, _losses, C_DEFAULT
from aaf.eval.band_limited import band_indices, compute_band_limited_metrics
from aaf.eval.modal_verifier import pick_peaks, modal_error_metrics
from aaf.eval.signal_level import (
    DEFAULT_BANDS,
    compute_signal_metrics,
    magnitude_correlation,
    make_signal_plots,
)
from aaf.renderers.freq_3d import FreqRenderer3D
from aaf.sim.analytical_modal_3d import eigenfrequencies_3d

BANDS_3 = ((0.0, 250.0), (250.0, 500.0), (500.0, 2000.0))


# ----------------------------------------------------------------------
# Render + eval (no obs/held split — evaluate on all receivers)
# ----------------------------------------------------------------------

def _build_renderer(cfg: dict, fs: float, n_time: int, device: str) -> FreqRenderer3D:
    return FreqRenderer3D(
        n_azi=int(cfg.get("n_azi", 16)),
        n_ele=int(cfg.get("n_ele", 16)),
        n_pts_per_ray=int(cfg.get("n_pts_per_ray", 16)),
        near=float(cfg.get("near", 1e-3)),
        fs=int(fs),
        n_time_samples=n_time,
        c=float(cfg.get("c", C_DEFAULT)),
        use_geometric_attn=False,
    ).to(device).eval()


def render_full(model, renderer, z, room_min, room_max, receiver_pos, src,
                device, eval_chunk: int = 4) -> np.ndarray:
    """Render H [n_rx, n_freq] for a single latent z across all receivers."""
    z_s_static = z.reshape(1, -1)
    rx = torch.from_numpy(receiver_pos).to(device)
    tx = torch.from_numpy(np.tile(src, (rx.size(0), 1))).to(device)
    out = []
    with torch.no_grad():
        for s in range(0, rx.size(0), eval_chunk):
            sub_rx, sub_tx = rx[s:s + eval_chunk], tx[s:s + eval_chunk]
            z_s = z_s_static.expand(sub_rx.size(0), -1)
            Hc = renderer(model, sub_rx, sub_tx, room_min, room_max, z_s=z_s)
            out.append(Hc.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.complex64)


def _per_band_mag_corr(H_pred, H_target, fs, n_freq, bands=BANDS_3) -> dict:
    d = {}
    for lo, hi in bands:
        a, b = band_indices(fs, n_freq, lo, hi)
        d[f"mag_corr_{int(lo)}_{int(hi)}"] = float(
            magnitude_correlation(H_pred[:, a:b + 1], H_target[:, a:b + 1])
        )
    d["mag_corr_full"] = float(magnitude_correlation(H_pred, H_target))
    return d


def eval_full(H_pred, H_target, rir_target, receiver_pos, L, W, H_dim,
              fs, n_time, n_freq, f_schroeder, bands=DEFAULT_BANDS) -> dict:
    """Full eval on ALL receivers: signal metrics + per-band LSD + per-band mag
    corr + full LSD + modal MAE at the centre receiver."""
    rir_pred = np.fft.irfft(H_pred, n=n_time, axis=-1).astype(np.float32)
    eps = 1e-8
    lsd = float(np.mean(np.abs(20 * np.log10(
        np.maximum(np.abs(H_pred), eps) / np.maximum(np.abs(H_target), eps)))))
    sig = compute_signal_metrics(
        H_pred, H_target, fs=fs, n_time_samples=n_time, bands=bands,
        rir_pred=rir_pred, rir_target=rir_target,
    )
    band = compute_band_limited_metrics(H_pred, H_target, fs, n_freq, bands)
    pbm = _per_band_mag_corr(H_pred, H_target, fs, n_freq)

    # Modal MAE at the centre receiver (D18 cap = clip(f_S, 100, 250)).
    f_modal_cap = max(100.0, min(f_schroeder, 250.0))
    centre = int(np.argmin(np.linalg.norm(
        receiver_pos - np.array([L / 2.0, W / 2.0, H_dim / 2.0]), axis=1)))
    f_axis = np.arange(n_freq) * (fs / n_time)
    f_mask = f_axis <= f_modal_cap
    modes = [m for m in eigenfrequencies_3d(L=L, W=W, H=H_dim, c=C_DEFAULT,
                                            f_max=f_modal_cap) if m.f > 0]
    if f_mask.sum() > 0 and modes:
        peaks = pick_peaks(H_pred[centre, f_mask], f_axis[f_mask],
                           prominence_db=3.0, min_distance_hz=2.0)
        modal = modal_error_metrics(peaks, modes, tolerance_hz=4.0, tolerance_pct=0.02)
    else:
        modal = {"mae_hz": float("nan"), "recall_at_tol": 0.0,
                 "n_picked": 0, "n_analytical": 0, "n_matched": 0}
    return {
        "L": L, "W": W, "H": H_dim,
        "lsd_db_full": lsd,
        "signal_metrics": sig,
        "band_metrics": band,
        "per_band_mag_corr": pbm,
        "modal_mae_hz": modal["mae_hz"],
        "modal_recall": modal["recall_at_tol"],
    }


# ----------------------------------------------------------------------
# (L,W,H) -> latent maps
# ----------------------------------------------------------------------

def build_lookup_maps(train_LWH, train_latents):
    """Return {'rbf': fn, 'linear': fn}, each mapping (L,W,H)->z[latent_dim].
    Inputs normalised to [0,1] over the training ranges first."""
    from scipy.interpolate import RBFInterpolator
    from sklearn.linear_model import LinearRegression

    X = np.asarray(train_LWH, dtype=np.float64)
    Z = np.asarray(train_latents, dtype=np.float64)
    lo, hi = X.min(0), X.max(0)
    rng = np.maximum(hi - lo, 1e-6)
    Xn = (X - lo) / rng
    rbf = RBFInterpolator(Xn, Z, kernel="thin_plate_spline")
    lin = LinearRegression().fit(Xn, Z)

    def _norm(lwh):
        return (np.asarray(lwh, dtype=np.float64).reshape(1, 3) - lo) / rng

    return {
        "rbf": lambda lwh: np.asarray(rbf(_norm(lwh))[0], dtype=np.float32),
        "linear": lambda lwh: np.asarray(lin.predict(_norm(lwh))[0], dtype=np.float32),
    }


# ----------------------------------------------------------------------
# Oracle latent optimisation
# ----------------------------------------------------------------------

def optimize_oracle_latent(model, renderer, room_min, room_max, rx_obs, tx_obs,
                           H_obs, z_init_vec, n_iters, lr, lambda_latent,
                           weights, device, eval_chunk=4):
    z_anchor = torch.as_tensor(z_init_vec, dtype=torch.float32, device=device)
    z_star = nn.Parameter(z_anchor.clone())
    opt = torch.optim.Adam([z_star], lr=lr)
    w_r, w_i, w_a, w_p = weights
    n = rx_obs.size(0)
    chunks = [(c, min(c + eval_chunk, n)) for c in range(0, n, eval_chunk)]
    for _ in range(int(n_iters)):
        opt.zero_grad(set_to_none=True)
        for c0, c1 in chunks:
            z_s = z_star.unsqueeze(0).expand(c1 - c0, -1)
            Hc = renderer(model, rx_obs[c0:c1], tx_obs[c0:c1], room_min, room_max, z_s=z_s)
            lc = _losses(Hc, H_obs[c0:c1])
            loss = ((c1 - c0) / n) * (w_r * lc["L_spec_real"] + w_i * lc["L_spec_imag"]
                                      + w_a * lc["L_amp"] + w_p * lc["L_phase"])
            loss.backward()
        l_lat = ((z_star - z_anchor) ** 2).mean()
        (lambda_latent * l_lat).backward()
        if z_star.grad is not None:
            z_star.grad = torch.nan_to_num(z_star.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_([z_star], 1.0)
        opt.step()
    return z_star.detach()


# ----------------------------------------------------------------------
# Room IO
# ----------------------------------------------------------------------

def _load_room(h5_path):
    rt = read_room_h5(h5_path)
    a = rt["attrs"]
    return {
        "L": float(a["L"]), "W": float(a["W"]), "H": float(a["H"]),
        "receiver_pos": np.asarray(a["receiver_pos"], dtype=np.float32),
        "src": np.asarray(a["source_pos"], dtype=np.float32),
        "H_target": rt["ism_H"].astype(np.complex64),
        "rir_target": rt["ism_rir"].astype(np.float32),
        "f_schroeder": float(a.get("schroeder_freq_hz", 200.0)),
    }


def _room_name(L, W, H):
    return f"L{L:.2f}_W{W:.2f}_H{H:.2f}"


def _parse_rooms(s):
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        L, W, H = (float(x) for x in part.split())
        out.append((L, W, H))
    return out


# ----------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------

def run_lookup(train_output_dir, rooms, data_dir, out_dir, device, do_loo, n_loo,
               plot_rooms=()):
    model, train_meta = _load_trained_model(Path(train_output_dir), device=device)
    cfg = train_meta["cfg"]
    fs = float(cfg["fs"]); n_time = int(cfg["n_time_samples"]); n_freq = n_time // 2 + 1
    renderer = _build_renderer(cfg, fs, n_time, device)
    for p in model.parameters():
        p.requires_grad_(False)
    train_latents = model.latents.weight.detach().cpu().numpy()  # [45,16]
    train_LWH = np.stack([train_meta["L_list"], train_meta["W_list"],
                          train_meta["H_list"]], axis=1)  # [45,3]
    out_dir = Path(out_dir)

    results = {"loo": {}, "lookup": {}}

    # --- LOO sanity on training rooms ---
    if do_loo:
        idxs = list(range(len(train_LWH)))[: int(n_loo)]
        loo_rows = []
        for i in idxs:
            keep = [j for j in range(len(train_LWH)) if j != i]
            maps = build_lookup_maps(train_LWH[keep], train_latents[keep])
            L, W, H = (float(x) for x in train_LWH[i])
            h5 = Path(data_dir) / f"{_room_name(L, W, H)}.h5"
            if not h5.exists():
                continue
            room = _load_room(h5)
            rmin = torch.tensor([0., 0., 0.], device=device)
            rmax = torch.tensor([room["L"], room["W"], room["H"]], device=device)
            for mp in ("rbf", "linear"):
                z = torch.as_tensor(maps[mp]((L, W, H)), device=device)
                Hp = render_full(model, renderer, z, rmin, rmax,
                                 room["receiver_pos"], room["src"], device)
                ev = eval_full(Hp, room["H_target"], room["rir_target"],
                               room["receiver_pos"], room["L"], room["W"], room["H"],
                               fs, n_time, n_freq, room["f_schroeder"])
                loo_rows.append({"room": _room_name(L, W, H), "map": mp,
                                 "mag_corr_full": ev["per_band_mag_corr"]["mag_corr_full"],
                                 "mag_corr_0_250": ev["per_band_mag_corr"]["mag_corr_0_250"],
                                 "lsd_db_full": ev["lsd_db_full"]})
        (out_dir / "loo").mkdir(parents=True, exist_ok=True)
        (out_dir / "loo" / "loo_rows.json").write_text(json.dumps(loo_rows, indent=2))
        for mp in ("rbf", "linear"):
            r = [x for x in loo_rows if x["map"] == mp]
            if r:
                results["loo"][mp] = {
                    "n": len(r),
                    "mean_mag_corr_full": float(np.mean([x["mag_corr_full"] for x in r])),
                    "mean_mag_corr_0_250": float(np.mean([x["mag_corr_0_250"] for x in r])),
                    "mean_lsd_db_full": float(np.mean([x["lsd_db_full"] for x in r])),
                }
        print(f"# LOO: {results['loo']}")

    # --- Test/interior room lookup (maps fit on ALL 45) ---
    maps_full = build_lookup_maps(train_LWH, train_latents)
    plot_set = {_room_name(*r) for r in plot_rooms}
    for (L, W, H) in rooms:
        h5 = Path(data_dir) / f"{_room_name(L, W, H)}.h5"
        if not h5.exists():
            print(f"# SKIP {_room_name(L, W, H)} (no h5)")
            continue
        room = _load_room(h5)
        rmin = torch.tensor([0., 0., 0.], device=device)
        rmax = torch.tensor([room["L"], room["W"], room["H"]], device=device)
        for mp in ("rbf", "linear"):
            z = torch.as_tensor(maps_full[mp]((L, W, H)), device=device)
            Hp = render_full(model, renderer, z, rmin, rmax,
                             room["receiver_pos"], room["src"], device)
            ev = eval_full(Hp, room["H_target"], room["rir_target"],
                           room["receiver_pos"], room["L"], room["W"], room["H"],
                           fs, n_time, n_freq, room["f_schroeder"])
            ev["z_norm"] = float(np.linalg.norm(maps_full[mp]((L, W, H))))
            rd = out_dir / "lookup" / f"{_room_name(L, W, H)}__{mp}"
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "metrics.json").write_text(json.dumps(ev, indent=2))
            np.save(rd / "H_pred_all.npy", Hp)
            results["lookup"][f"{_room_name(L, W, H)}__{mp}"] = ev["per_band_mag_corr"]
            # signal plots for the chosen interpolative rooms (rbf only)
            if mp == "rbf" and _room_name(L, W, H) in plot_set:
                make_signal_plots(Hp, room["H_target"], fs=fs, n_time_samples=n_time,
                                  output_dir=rd / "figures", bands=DEFAULT_BANDS,
                                  rir_pred=np.fft.irfft(Hp, n=n_time, axis=-1).astype(np.float32),
                                  rir_target=room["rir_target"])
        print(f"# lookup {_room_name(L, W, H)}: "
              f"rbf mag={results['lookup'][f'{_room_name(L,W,H)}__rbf']['mag_corr_full']:.3f} "
              f"lin mag={results['lookup'][f'{_room_name(L,W,H)}__linear']['mag_corr_full']:.3f}")
    (out_dir / "lookup_summary.json").write_text(json.dumps(results, indent=2))
    return results


def run_oracle(train_output_dir, rooms, data_dir, out_dir, device,
               n_oracle_recv, n_adapt_iters, lr, lambda_latent):
    from aaf.eval.zero_shot_3d import select_obs_indices_3d
    model, train_meta = _load_trained_model(Path(train_output_dir), device=device)
    cfg = train_meta["cfg"]
    fs = float(cfg["fs"]); n_time = int(cfg["n_time_samples"]); n_freq = n_time // 2 + 1
    renderer = _build_renderer(cfg, fs, n_time, device).train()
    for p in model.parameters():
        p.requires_grad_(False)
    z_mean = model.latents.weight.detach().mean(0).cpu().numpy()
    out_dir = Path(out_dir)
    for (L, W, H) in rooms:
        h5 = Path(data_dir) / f"{_room_name(L, W, H)}.h5"
        if not h5.exists():
            print(f"# SKIP {_room_name(L, W, H)} (no h5)")
            continue
        room = _load_room(h5)
        rmin = torch.tensor([0., 0., 0.], device=device)
        rmax = torch.tensor([room["L"], room["W"], room["H"]], device=device)
        n_rx = room["receiver_pos"].shape[0]
        obs_idx = select_obs_indices_3d(int(n_oracle_recv), total=n_rx)
        rx_obs = torch.from_numpy(room["receiver_pos"][obs_idx]).to(device)
        tx_obs = torch.from_numpy(np.tile(room["src"], (rx_obs.size(0), 1))).to(device)
        H_obs = torch.from_numpy(room["H_target"][obs_idx]).to(device)
        z_star = optimize_oracle_latent(model, renderer, rmin, rmax, rx_obs, tx_obs,
                                        H_obs, z_mean, n_adapt_iters, lr, lambda_latent,
                                        (1.0, 1.0, 1.0, 0.1), device)
        renderer.eval()
        Hp = render_full(model, renderer, z_star, rmin, rmax,
                         room["receiver_pos"], room["src"], device)
        ev = eval_full(Hp, room["H_target"], room["rir_target"], room["receiver_pos"],
                       room["L"], room["W"], room["H"], fs, n_time, n_freq, room["f_schroeder"])
        ev["z_norm"] = float(z_star.norm().item())
        ev["n_oracle_recv"] = int(n_oracle_recv)
        ev["n_adapt_iters"] = int(n_adapt_iters)
        rd = out_dir / "oracle" / _room_name(L, W, H)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "metrics.json").write_text(json.dumps(ev, indent=2))
        np.save(rd / "H_pred_all.npy", Hp)
        print(f"# oracle {_room_name(L, W, H)} (n_recv={n_oracle_recv}): "
              f"mag={ev['per_band_mag_corr']['mag_corr_full']:.3f} "
              f"0-250={ev['per_band_mag_corr']['mag_corr_0_250']:.3f} "
              f"lsd={ev['lsd_db_full']:.2f} z*norm={ev['z_norm']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["lookup", "oracle"])
    ap.add_argument("--train-output-dir", required=True)
    ap.add_argument("--output-dir", default="outputs/known_geometry")
    ap.add_argument("--data-dir", default="data/track_a_3d")
    ap.add_argument("--rooms", required=True, help='"L W H; L W H; ..."')
    ap.add_argument("--plot-rooms", default="", help='"L W H; ..." interpolative rooms to plot')
    ap.add_argument("--loo", action="store_true")
    ap.add_argument("--n-loo", type=int, default=45)
    ap.add_argument("--n-oracle-recv", type=int, default=32)
    ap.add_argument("--n-adapt-iters", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lambda-latent", type=float, default=1e-4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rooms = _parse_rooms(args.rooms)
    if args.mode == "lookup":
        run_lookup(args.train_output_dir, rooms, args.data_dir, args.output_dir,
                   device, args.loo, args.n_loo, _parse_rooms(args.plot_rooms))
    else:
        run_oracle(args.train_output_dir, rooms, args.data_dir, args.output_dir,
                   device, args.n_oracle_recv, args.n_adapt_iters, args.lr, args.lambda_latent)


if __name__ == "__main__":
    main()
