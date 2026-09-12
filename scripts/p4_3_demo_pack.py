"""P4-3 Task 1: the scoped shape-edit demo pack.

One trained model, no optimisation at demo time, two rooms it has never seen, and the only thing
that changes between panels is the NOTCH DEPTH in the conditioning vector.

SCOPE, STATED EVERYWHERE RATHER THAN BURIED. P4-2's sweep measured accuracy across 20 unseen
depths and found a U: worst at SHALLOW notches (spatial Pearson +0.741 at d_hat 0.21), best at
deep ones (+0.906 at 0.90). This demo shows d_hat >= 0.5 -- the half that works -- and every
caption says so and gives the shallow number. A demo that quietly picked the good half and
implied the whole range would be the easiest possible way to mislead with true numbers.

Checkpoint: the mean_unmasked arm. Not mean_masked, which scores marginally better on in-slab
spatial Pearson (0.798 vs 0.748), because DC-masking collapses RIR Pearson from 0.9989 to 0.0630
(D70) and this pack shows an impulse response.

Three figures, all numbers printed BEFORE any of them is drawn (D62a):
  figL_<tag>  field maps, 4 depths x {predicted, FDTD} at 3 resolvable modes
  figM_<tag>  centre-receiver spectrum overlay + band-limited RIR overlay
  figN_morph  the morph strip, fixed bounding box, depth swept across the working range
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon as MplPolygon

from aaf.data.shape_configs import SRC, ShapeConfig
from aaf.eval.modal_decay import band_limited_rir
from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.p3_2_eval import load_model
from aaf.models.conditioning_2d import build_cond_vector_2d
from scripts.build_p4_3_demo_rooms import DEMO_SHAPES, GRID_N, demo_configs
from scripts.build_p4_2_sweep import sweep_configs

DPI = 200
DF_HZ = 0.5
N_MODES_SHOWN = 3
C_PRED, C_GT = "#0072B2", "#D55E00"
WORKING_MIN = 0.50
SHALLOW_NOTE = ("Scope: d_hat >= 0.50, the regime P4-2's sweep showed working. Accuracy DEGRADES "
                "at shallow notches -- spatial Pearson +0.741 at d_hat 0.21 against +0.906 at "
                "0.90. This pack does not claim the shallow half.")


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def _pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


# The stored band is 0-300 Hz at df = 0.5 Hz, i.e. a 2 s record: n_time = 1200, fs_eff = 600.
RIR_FS, RIR_N, MODAL_LO = 600.0, 1200, 20.0


def _rir(H, f_lo=0.0):
    """Band-limited impulse response, identical mask on prediction and target.

    TWO BANDS, AND THE DIFFERENCE MATTERS. `f_lo = 0` reproduces P4-2's Gate-2 `rir_pearson`
    exactly, so the demo stays comparable with it. But that number is DOMINATED BY THE NEAR-DC
    TERM: the 0-300 Hz inverse transform is a slow ramp on which prediction and target agree
    trivially, and it reads r = +1.000 while the two traces sit at visibly different levels.
    That is Q18's finding in the time domain. `f_lo = 20` removes it and leaves the modal
    response, which is what a claim about the impulse response should actually rest on. Both are
    reported; the figure draws the modal one, because a panel whose headline number measures a
    DC ramp would be misleading with true numbers.
    """
    return band_limited_rir(np.asarray(H), RIR_FS, RIR_N, f_lo=f_lo, f_hi=300.0)


def render(model, renderer, cfg, rx, n_bins, dev, chunk=8):
    cond = build_cond_vector_2d("geom_token", cfg.L, cfg.W, cfg.edge_alphas,
                                verts=cfg.verts, device=dev)
    room_min = torch.zeros(2, device=dev)
    room_max = torch.tensor([cfg.L, cfg.W], device=dev, dtype=torch.float32)
    R = torch.tensor(rx, dtype=torch.float32, device=dev)
    S = torch.tensor(np.asarray(SRC, float), dtype=torch.float32, device=dev).unsqueeze(0)
    P = np.zeros((len(rx), n_bins), dtype=np.complex128)
    for s0 in range(0, len(rx), chunk):
        sl = slice(s0, s0 + chunk)
        r = R[sl]
        with torch.no_grad():
            H = renderer(model, r, S.expand(r.shape[0], -1), room_min, room_max,
                         z_s=cond.unsqueeze(0).expand(r.shape[0], -1))
        P[sl] = H[:, :n_bins].cpu().numpy()
    return P


def load_rooms(cfgs, data_dir, model, renderer, dev, cache=None, label="demo", ck=None):
    """Render every room, CACHING predictions keyed on the checkpoint.

    The render is the expensive half (~2 min/room) and the figures are the half that gets
    iterated on. P4-2 lost 37 minutes of GPU to a typo downstream of an uncached render; this
    makes a re-plot seconds instead.
    """
    import h5py
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["ckpt"]) == str(ck):
            rooms = []
            for i, (tag, cfg) in enumerate(cfgs):
                keep = z["keep{}".format(i)]
                rooms.append({"tag": tag, "cfg": cfg, "T": z["T{}".format(i)],
                              "P": z["P{}".format(i)], "rx": z["rx{}".format(i)],
                              "los": z["los{}".format(i)],
                              "keep": None if keep.size == 0 else keep})
            print("[cache] reused {} {} rooms from {}".format(len(rooms), label, cache),
                  flush=True)
            return rooms
        print("[cache] {} is for another checkpoint -- re-rendering".format(cache), flush=True)
    rooms = []
    for tag, cfg in cfgs:
        p = Path(data_dir) / cfg.filename
        if not p.exists():
            raise FileNotFoundError("room not built: {}".format(p))
        with h5py.File(p) as f:
            T = np.asarray(f["ism/H_complex"])
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
            los = np.asarray(json.loads(f.attrs["line_of_sight"]), dtype=bool)
            keep = (np.asarray(json.loads(f.attrs["lattice_keep"]), dtype=bool)
                    if "lattice_keep" in f.attrs else None)
        P = render(model, renderer, cfg, rx, T.shape[1], dev)
        rooms.append({"tag": tag, "cfg": cfg, "T": T, "P": P, "rx": rx, "los": los,
                      "keep": keep})
        print("  {} d_hat {:.3f}  n_rx {:5d}".format(tag, cfg.d_hat, len(rx)), flush=True)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        blob = {"ckpt": str(ck)}
        for i, r in enumerate(rooms):
            blob["T{}".format(i)], blob["P{}".format(i)] = r["T"], r["P"]
            blob["rx{}".format(i)], blob["los{}".format(i)] = r["rx"], r["los"]
            blob["keep{}".format(i)] = (np.zeros(0, dtype=bool) if r["keep"] is None
                                        else r["keep"])
        np.savez_compressed(cache, **blob)
        print("[cache] wrote {}".format(cache), flush=True)
    return rooms


def common_probe(rooms, frac=(0.60, 0.80)):
    """One receiver present in EVERY depth of a shape, so the spectrum overlay compares the same
    listening position across the edit. Anything that moved would confound the shape change."""
    cfg0 = rooms[0]["cfg"]
    tx = cfg0.w + frac[0] * (cfg0.L - cfg0.w)
    ty = frac[1] * cfg0.W
    idx = []
    for r in rooms:
        dist = np.hypot(r["rx"][:, 0] - tx, r["rx"][:, 1] - ty)
        idx.append(int(np.argmin(dist)))
    pos = np.array([rooms[k]["rx"][i] for k, i in enumerate(idx)])
    spread = float(np.abs(pos - pos[0]).max())
    return idx, pos[0], spread


def measure(rooms, modes, bins):
    out = []
    for r in rooms:
        sp = [_pearson(_db(r["P"][:, b]), _db(r["T"][:, b])) for b in bins
              if b < r["T"].shape[1]]
        rir_p, rir_t = _rir(r["P"]), _rir(r["T"])
        mod_p, mod_t = _rir(r["P"], MODAL_LO), _rir(r["T"], MODAL_LO)
        out.append({
            "tag": r["tag"], "d_hat": r["cfg"].d_hat, "d": r["cfg"].d,
            "L": r["cfg"].L, "W": r["cfg"].W, "w": r["cfg"].w,
            "n_rx": int(len(r["rx"])), "nlos_frac": float(1.0 - r["los"].mean()),
            "spatial_pearson": float(np.nanmean(sp)),
            "spatial_pearson_per_mode": [float(x) for x in sp],
            "band_lsd_db": float(np.mean(np.abs(_db(r["P"]) - _db(r["T"])))),
            "rir_pearson": float(np.mean([_pearson(rir_p[i], rir_t[i])
                                          for i in range(len(rir_p))])),
            "rir_pearson_modal": float(np.mean([_pearson(mod_p[i], mod_t[i])
                                                for i in range(len(mod_p))])),
        })
    return out


# ---------------------------------------------------------------------------------- figures
def _grid_image(r, b):
    """Scatter the kept lattice values back onto the full GRID_N x GRID_N lattice; notch = NaN."""
    img = np.full(GRID_N * GRID_N, np.nan)
    img[r["keep"]] = _db(r["P" if r.get("_which") == "P" else "T"][:, b])
    return img.reshape(GRID_N, GRID_N).T


def fig_fields(rooms, modes, bins, met, out, tag):
    n_d = len(rooms)
    fig, axes = plt.subplots(2 * N_MODES_SHOWN, n_d,
                            figsize=(3.5 * n_d, 3.3 * 2 * N_MODES_SHOWN), dpi=DPI)
    cfg0 = rooms[0]["cfg"]
    ext = [0.12, cfg0.L - 0.12, 0.12, cfg0.W - 0.12]
    for mi in range(N_MODES_SHOWN):
        b = bins[mi]
        vals = np.concatenate([_db(r["T"][:, b]) for r in rooms])
        vmax = float(np.percentile(vals, 99.5)); vmin = vmax - 40.0
        for di, r in enumerate(rooms):
            for row, which, lab in ((0, "P", "predicted"), (1, "T", "FDTD")):
                ax = axes[2 * mi + row, di]
                r["_which"] = which
                im = ax.imshow(_grid_image(r, b), origin="lower", extent=ext,
                               vmin=vmin, vmax=vmax, cmap="magma", aspect="equal",
                               interpolation="nearest")
                ax.add_patch(MplPolygon(r["cfg"].verts, closed=True, fill=False, ec="black",
                                        lw=1.6, zorder=6))
                ax.plot(*SRC, marker="*", ms=11, color="#00E5FF", mec="black", mew=0.5, zorder=9)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlim(-0.15, cfg0.L + 0.15); ax.set_ylim(-0.15, cfg0.W + 0.15)
                if di == 0:
                    ax.set_ylabel("{}\n({},{}) {:.0f} Hz".format(
                        lab, modes[mi].n_x, modes[mi].n_y, modes[mi].f), fontsize=11,
                        fontweight="bold")
                if row == 0:
                    ax.set_title(r"$\hat{{d}}$ = {:.2f}   R = {:+.3f}".format(
                        r["cfg"].d_hat, met[di]["spatial_pearson_per_mode"][mi]
                        if mi < len(met[di]["spatial_pearson_per_mode"]) else float("nan")),
                        fontsize=11)
        fig.colorbar(im, ax=axes[2 * mi:2 * mi + 2, :], fraction=0.015, pad=0.01).set_label(
            "|H| dB", fontsize=9)
    fig.suptitle("Shape edit, room {}: L = {:.2f} m, W = {:.2f} m, notch width {:.2f} m FIXED; "
                 "only the DEPTH changes  |  unseen room, one forward pass per panel".format(
                     tag, cfg0.L, cfg0.W, cfg0.w), fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005, SHALLOW_NOTE + "\n64x64 lattice masked to the polygon; the removed "
             "corner is left blank rather than interpolated. Cyan star = source.",
             ha="center", fontsize=10.5)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_spectrum_rir(rooms, met, out, tag, probe_idx, probe_pos):
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 6.4), dpi=DPI)
    f = np.arange(rooms[0]["T"].shape[1]) * DF_HZ
    off = 18.0
    ax = axes[0]
    for k, r in enumerate(rooms):
        i = probe_idx[k]
        ax.plot(f, _db(r["T"][i]) + k * off, color=C_GT, lw=1.5,
                label="FDTD" if k == 0 else None)
        ax.plot(f, _db(r["P"][i]) + k * off, color=C_PRED, lw=1.3, ls="--",
                label="predicted" if k == 0 else None)
        ax.text(303, _db(r["T"][i])[-1] + k * off, r"$\hat{{d}}$={:.2f}".format(r["cfg"].d_hat),
                fontsize=9, va="center")
    ax.set_xlim(0, 300); ax.set_xlabel("frequency (Hz)", fontsize=12)
    ax.set_ylabel("|H| (dB, traces offset by depth)", fontsize=12)
    ax.set_title("spectrum at a FIXED receiver [{:.2f}, {:.2f}]".format(*probe_pos),
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left"); ax.grid(alpha=0.2)

    ax = axes[1]
    for k, r in enumerate(rooms):
        i = probe_idx[k]
        rt, rp = _rir(r["T"][i], MODAL_LO), _rir(r["P"][i], MODAL_LO)
        t = np.arange(len(rt)) / RIR_FS
        n = int(0.5 * RIR_FS)                      # first 0.5 s, where the structure lives
        sc = 1.6 / max(np.abs(rt[:n]).max(), 1e-12)
        ax.plot(t[:n], rt[:n] * sc + k * 2.4, color=C_GT, lw=1.2, label="FDTD" if not k else None)
        ax.plot(t[:n], rp[:n] * sc + k * 2.4, color=C_PRED, lw=1.0, ls="--",
                label="predicted" if not k else None)
        ax.text(t[n - 1] * 1.01, k * 2.4, r"r={:+.3f}".format(met[k]["rir_pearson_modal"]),
                fontsize=9, va="center")
    ax.set_xlabel("time (s)", fontsize=12)
    ax.set_ylabel("band-limited impulse response (offset)", fontsize=12)
    ax.set_title("MODAL impulse response (20-300 Hz) at the same receiver", fontsize=13,
                 fontweight="bold")
    ax.set_yticks([]); ax.grid(alpha=0.2); ax.legend(fontsize=10, loc="upper right")
    fig.suptitle("Room {}: the edit in the frequency AND time domain  |  mean spatial R "
                 "{:+.3f}, band LSD {:.2f} dB, modal RIR r {:+.3f} (full-band {:+.3f})".format(
                     tag, float(np.mean([m["spatial_pearson"] for m in met])),
                     float(np.mean([m["band_lsd_db"] for m in met])),
                     float(np.mean([m["rir_pearson_modal"] for m in met])),
                     float(np.mean([m["rir_pearson"] for m in met]))),
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005, SHALLOW_NOTE + "\nThe RIR panel is high-passed at 20 Hz. The full-band "
             "0-300 Hz inverse transform reads r = +1.000, but that number is the shared near-DC "
             "ramp (Q18 in the time domain), not the impulse structure -- both are reported.",
             ha="center", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.10, 1, 0.93])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_morph(rooms, met, b, mode, out):
    n = len(rooms)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 5.6), dpi=DPI)
    vals = np.concatenate([_db(r["T"][:, b]) for r in rooms])
    vmax = float(np.percentile(vals, 99.5)); vmin = vmax - 40.0
    cfg0 = rooms[0]["cfg"]
    for k, r in enumerate(rooms):
        for row, which, lab in ((0, "P", "predicted"), (1, "T", "FDTD")):
            ax = axes[row, k]
            s = ax.scatter(r["rx"][:, 0], r["rx"][:, 1], c=_db(r[which][:, b]), s=3,
                           vmin=vmin, vmax=vmax, cmap="magma")
            ax.add_patch(MplPolygon(r["cfg"].verts, closed=True, fill=False, ec="black",
                                    lw=1.5, zorder=6))
            ax.plot(*SRC, marker="*", ms=10, color="#00E5FF", mec="black", mew=0.5, zorder=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            ax.set_xlim(-0.2, cfg0.L + 0.2); ax.set_ylim(-0.2, cfg0.W + 0.2)
            if k == 0:
                ax.set_ylabel(lab, fontsize=12, fontweight="bold")
        axes[0, k].set_title(r"$\hat{{d}}$={:.2f}" "\n" "R={:+.3f}".format(
            r["cfg"].d_hat, met[k]["spatial_pearson"]), fontsize=10)
    fig.colorbar(s, ax=axes, fraction=0.014, pad=0.01).set_label("|H| dB", fontsize=10)
    fig.suptitle("The morph: L = {:.2f} m, W = {:.2f} m, notch width {:.2f} m all FIXED; the "
                 "corner grows and the field reorganises  |  mode ({},{}) {:.0f} Hz".format(
                     cfg0.L, cfg0.W, cfg0.w, mode.n_x, mode.n_y, mode.f),
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.008, SHALLOW_NOTE, ha="center", fontsize=10.5)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/p4_2/stage2/p4_2_s2_mean_unmasked")
    ap.add_argument("--demo-dir", default="data/track_p4_3_demo")
    ap.add_argument("--sweep-dir", default="data/track_p4_2_sweep")
    ap.add_argument("--out", default="outputs/p4_3/demo")
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = Path(a.checkpoint) if a.checkpoint else sorted(Path(a.run_dir).glob("ckpt_iter*.pt"))[-1]
    model, renderer, cfg_d, meta, it = load_model(ck, dev)
    model.eval(); renderer.eval()                                   # D49 C3
    print("[arm] {} iter {} | pool {} | loss_lo {} Hz".format(
        Path(a.run_dir).name, it, cfg_d.get("token_pool") or "masked_mean",
        cfg_d.get("loss_band_lo_hz", 0.0)), flush=True)

    report = {"arm": Path(a.run_dir).name, "checkpoint": str(ck), "iter": int(it),
              "scope": SHALLOW_NOTE, "shapes": {}}
    allc = demo_configs()
    for sh in DEMO_SHAPES:
        tag = sh["tag"]
        cfgs = [(t, c) for t, c in allc if t == tag]
        print("\n=== room {} ({} depths) ===".format(tag, len(cfgs)), flush=True)
        rooms = load_rooms(cfgs, a.demo_dir, model, renderer, dev,
                           cache=str(out / "render_{}.npz".format(tag)),
                           label="demo " + tag, ck=ck)
        c0 = rooms[0]["cfg"]
        modes = enumerate_modes(c0.L, c0.W, f_max=200.0)[:N_MODES_SHOWN]
        bins = [int(round(m.f / DF_HZ)) for m in modes]
        met = measure(rooms, modes, bins)
        pidx, ppos, pspread = common_probe(rooms)
        print("  probe receiver [{:.2f}, {:.2f}] (max drift across depths {:.3f} m)".format(
            ppos[0], ppos[1], pspread))
        print("  d_hat   n_rx  NLOS%   spatialR   LSD dB   RIR r(0-300)  RIR r(20-300)")
        for m in met:
            print("  {:.3f}  {:5d}  {:5.1f}   {:+.4f}   {:6.2f}     {:+.4f}       {:+.4f}".format(
                m["d_hat"], m["n_rx"], 100 * m["nlos_frac"], m["spatial_pearson"],
                m["band_lsd_db"], m["rir_pearson"], m["rir_pearson_modal"]))
        report["shapes"][tag] = {
            "L": c0.L, "W": c0.W, "w": c0.w, "probe": [float(x) for x in ppos],
            "probe_drift_m": pspread,
            "modes": [[int(m.n_x), int(m.n_y), float(m.f)] for m in modes],
            "per_depth": met,
            "mean_spatial_pearson": float(np.mean([m["spatial_pearson"] for m in met])),
            "mean_band_lsd_db": float(np.mean([m["band_lsd_db"] for m in met])),
            "mean_rir_pearson": float(np.mean([m["rir_pearson"] for m in met])),
            "mean_rir_pearson_modal": float(np.mean([m["rir_pearson_modal"] for m in met])),
        }
        fig_fields(rooms, modes, bins, met, out / "figL_{}_fields.png".format(tag), tag)
        fig_spectrum_rir(rooms, met, out / "figM_{}_spectrum_rir.png".format(tag), tag,
                         pidx, ppos)

    # ---- the morph strip: reuse the already-built sweep corpus, working range only ----
    print("\n=== morph strip (working range of the sweep corpus) ===", flush=True)
    sw = [("M", c) for c in sweep_configs() if c.d_hat >= WORKING_MIN][::2]
    rooms = load_rooms(sw, a.sweep_dir, model, renderer, dev,
                       cache=str(out / "render_morph.npz"), label="morph", ck=ck)
    c0 = rooms[0]["cfg"]
    modes = enumerate_modes(c0.L, c0.W, f_max=200.0)[:N_MODES_SHOWN]
    bins = [int(round(m.f / DF_HZ)) for m in modes]
    met = measure(rooms, modes, bins)
    print("  d_hat   n_rx   spatialR   LSD dB   RIR r(0-300)  RIR r(20-300)")
    for m in met:
        print("  {:.3f}  {:5d}   {:+.4f}   {:6.2f}     {:+.4f}       {:+.4f}".format(
            m["d_hat"], m["n_rx"], m["spatial_pearson"], m["band_lsd_db"], m["rir_pearson"],
            m["rir_pearson_modal"]))
    report["morph"] = {"L": c0.L, "W": c0.W, "w": c0.w, "per_depth": met,
                       "mode": [int(modes[1].n_x), int(modes[1].n_y), float(modes[1].f)]}
    fig_morph(rooms, met, bins[1], modes[1], out / "figN_morph.png")

    json.dump(report, open(out / "demo_metrics.json", "w"), indent=1, default=float)
    for f in sorted(out.glob("fig*.png")):
        print("  {}  {:.1f} MB".format(f, f.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
