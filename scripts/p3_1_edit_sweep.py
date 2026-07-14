"""P3-1 edit-sweep money-figure: render each arm across an unseen L-sweep (W,H fixed),
produce (i) a waterfall |H(f)| vs L with the analytic mode trajectories overlaid, and
(ii) a tracked-peak plot for the first 3 axial-L modes with a tracking MAE.

Correct editing behaviour: predicted spectral ridges slide along the analytic curves
(axial L-modes f=c*n/2L; W/H-only modes stay vertical). Usage:
  python scripts/p3_1_edit_sweep.py --arm-dir outputs/p3_1/arm_Gplus --out outputs/p3_1/edits/arm_Gplus
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from aaf.eval.zero_shot_3d import _load_trained_model
from aaf.eval.known_geometry import _build_renderer, render_full, _load_room, build_lookup_maps
from aaf.eval.band_limited import band_indices
from aaf.eval.modal_verifier import pick_peaks
from aaf.sim.analytical_modal_3d import C_DEFAULT
from aaf.models.conditioning import fourier_features, eigen_features, resonance_map

W_FIX, H_FIX = 4.0, 3.25
F_MAX = 300.0


def _cond(arm, L, W, Hd, model, meta, maps, dev):
    if arm == "latent":
        return torch.as_tensor(maps["rbf"]((L, W, Hd)), dtype=torch.float32, device=dev)
    if arm == "geom_fourier":
        return fourier_features(L, W, Hd, device=dev)
    return eigen_features(L, W, Hd, device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rooms-yaml", default="configs/sweeps_3d/p3_1_edit_sweep.yaml")
    ap.add_argument("--data-dir", default="data/track_a_3d")
    a = ap.parse_args()
    import yaml
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, meta = _load_trained_model(Path(a.arm_dir), device="cuda")
    cfg = meta["cfg"]; arm = cfg.get("cond_source", "latent")
    fs, n_time, n_freq = int(cfg["fs"]), int(cfg["n_time_samples"]), model.n_freq_bins
    dev = next(model.parameters()).device
    renderer = _build_renderer(cfg, fs, n_time, "cuda")
    _, hi = band_indices(fs, n_freq, 0.0, float(cfg.get("band_max_hz", 300.0)))
    f_axis = np.arange(n_freq) * (fs / n_time)
    maps = None
    if arm == "latent":
        train_LWH = np.stack([meta["L_list"], meta["W_list"], meta["H_list"]], 1)
        maps = build_lookup_maps(train_LWH, model.latents.weight.detach().cpu().numpy())

    rooms = sorted(yaml.safe_load(open(a.rooms_yaml))["rooms"], key=lambda r: r["L"])
    Ls, mags, peaks_by_L = [], [], []
    for r in rooms:
        L = float(r["L"])
        h5 = Path(a.data_dir) / f"L{L:.2f}_W{W_FIX:.2f}_H{H_FIX:.2f}.h5"
        if not h5.exists():
            continue
        room = _load_room(h5)
        z = _cond(arm, L, W_FIX, H_FIX, model, meta, maps, dev)
        if arm == "eigen":
            R = torch.zeros(n_freq, device=dev)
            Rb = resonance_map(L, W_FIX, H_FIX, n_bins=hi, df=fs / n_time, device=dev)
            R[:Rb.numel()] = Rb; model.set_resonance(R)
        rmin = torch.zeros(3, device=dev); rmax = torch.tensor([L, W_FIX, H_FIX], device=dev)
        H = render_full(model, renderer, z, rmin, rmax, room["receiver_pos"], room["src"], dev)
        c = int(np.argmin(np.linalg.norm(room["receiver_pos"] - [L/2, W_FIX/2, H_FIX/2], axis=1)))
        Hc = np.abs(H[c, :hi])
        Ls.append(L); mags.append(20 * np.log10(np.maximum(Hc, 1e-8)))
        pk = pick_peaks(H[c, :hi], f_axis[:hi], prominence_db=3.0, min_distance_hz=2.0)
        peaks_by_L.append([p.f for p in pk])
    Ls = np.array(Ls); mags = np.array(mags); fa = f_axis[:hi]

    # analytic axial-L modes f=c*n/2L and a couple of W/H-only (vertical) modes
    def axial_L(n, L): return C_DEFAULT * n / (2 * L)
    wmode = C_DEFAULT / (2 * W_FIX); hmode = C_DEFAULT / (2 * H_FIX)

    # --- (i) waterfall ---
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.pcolormesh(fa, Ls, mags, shading="auto", cmap="magma")
    Lg = np.linspace(Ls.min(), Ls.max(), 100)
    for n in (1, 2, 3, 4):
        ax.plot(axial_L(n, Lg), Lg, "c-", lw=1.2, alpha=0.8, label=f"axial ({n},0,0)=c·{n}/2L" if n == 1 else None)
    ax.axvline(wmode, color="lime", ls="--", lw=1.0, alpha=0.7, label="(0,1,0)=c/2W (invariant)")
    ax.axvline(hmode, color="yellow", ls=":", lw=1.0, alpha=0.7, label="(0,0,1)=c/2H (invariant)")
    ax.set_xlim(0, F_MAX); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("L (m) — edited dimension")
    ax.set_title(f"Edit sweep — arm {arm}: predicted |H| (dB) vs L, with analytic mode trajectories\n"
                 f"(W={W_FIX}, H={H_FIX} fixed; correct = ridges follow the cyan 1/L curves, vertical lines stay put)")
    fig.colorbar(im, ax=ax, label="|H| (dB)"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "waterfall.png", dpi=110); plt.close(fig)

    # --- (ii) tracked-peak + tracking MAE for n=1,2,3 ---
    fig, ax = plt.subplots(figsize=(9, 7))
    tracked = {1: [], 2: [], 3: []}; errs = {1: [], 2: [], 3: []}
    for i, L in enumerate(Ls):
        for n in (1, 2, 3):
            fa_n = axial_L(n, L)
            cand = [f for f in peaks_by_L[i] if abs(f - fa_n) <= max(4.0, 0.05 * fa_n)]
            if cand:
                f_hit = min(cand, key=lambda f: abs(f - fa_n))
                tracked[n].append((L, f_hit)); errs[n].append(abs(f_hit - fa_n))
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    for n in (1, 2, 3):
        ax.plot(Lg, axial_L(n, Lg), "-", color=colors[n], alpha=0.5, label=f"analytic ({n},0,0)")
        if tracked[n]:
            LL, ff = zip(*tracked[n]); ax.scatter(LL, ff, color=colors[n], s=28,
                        label=f"pred n={n} (MAE {np.mean(errs[n]):.2f} Hz, {len(ff)}/{len(Ls)})")
    ax.set_xlabel("L (m)"); ax.set_ylabel("mode frequency (Hz)"); ax.set_ylim(0, F_MAX)
    ax.set_title(f"Tracked axial-L peaks — arm {arm}: predicted vs analytic c·n/2L")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out / "tracked_peaks.png", dpi=110); plt.close(fig)

    summary = {"arm": arm, "n_sweep": len(Ls),
               "tracking_mae_hz": {str(n): (float(np.mean(errs[n])) if errs[n] else None) for n in (1, 2, 3)},
               "tracking_recall": {str(n): len(tracked[n]) / max(len(Ls), 1) for n in (1, 2, 3)}}
    (out / "edit_sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
