"""P4-2 the money figure: move ONE shape parameter and watch the prediction follow -- or not.

Gate 2 asked whether a shape-conditioned model reproduces 15 held-out rooms. This asks the
question the gate cannot: is the model's dependence on shape CONTINUOUS and CORRECT, or is it a
lookup that degrades wherever the training corpus was thin? `L, W, w` are pinned; `d` sweeps 20
values through the held-out slab, none of them in any corpus.

Three panels, computed before they are drawn:

  * **figI, accuracy vs depth** -- spatial Pearson and band LSD against d-hat, with the held-out
    slab shaded and the LOS -> NLOS transition of the fixed probe receiver marked. A model that
    interpolates shape shows a flat curve; one that memorizes shows a dip over the slab.
  * **figJ, the waterfall** -- the fixed probe receiver's spectrum stacked against d-hat, FDTD
    beside prediction. Mode migration with depth is the physics; whether the predicted stack
    migrates the same way is the claim.
  * **figK, the field strip** -- five depths at one mode, predicted above, FDTD below, one
    shared colour scale.

The probe receivers never move between rooms (`build_p4_2_sweep` places them at indices 0 and
1), so anything that changes in the waterfall is the SHAPE EDIT and not a change of listening
position. Receiver 1 starts LOS and is swallowed by the growing shadow; receiver 0 stays LOS as
the control.
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

from aaf.data.shape_configs import D_HAT_HOLDOUT, SRC
from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.p3_2_eval import load_model
from aaf.models.conditioning_2d import build_cond_vector_2d
from scripts.build_p4_2_sweep import PROBE_EDGE, PROBE_LOS, sweep_configs

DPI = 200
DF_HZ = 0.5
N_MODES = 6
C_PRED, C_GT, C_SLAB = "#0072B2", "#D55E00", "#CC79A7"


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def _pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def render_room(model, renderer, cond_source, cfg, rx, n_bins, dev, chunk=8):
    cond = build_cond_vector_2d(cond_source, cfg.L, cfg.W, cfg.edge_alphas,
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


def collect(run_dir, data_dir, dev, checkpoint=None, cache=None):
    """Render all 20 rooms, CACHING the predictions to disk.

    The render is ~2 min/room on one GPU; a typo anywhere downstream of it would otherwise cost
    the whole 40 minutes again. It did exactly that once. The cache is keyed on the checkpoint
    path so a different arm cannot silently reuse another's predictions.
    """
    import h5py
    ck = Path(checkpoint) if checkpoint else sorted(Path(run_dir).glob("ckpt_iter*.pt"))[-1]
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["ckpt"]) == str(ck):
            rooms = [{"cfg": c, "T": z["T{}".format(i)], "P": z["P{}".format(i)],
                      "rx": z["rx{}".format(i)], "los": z["los{}".format(i)]}
                     for i, c in enumerate(sweep_configs())]
            print("[cache] reusing {} rendered rooms from {}".format(len(rooms), cache), flush=True)
            return rooms, int(z["iter"]), ck
        print("[cache] {} is for a different checkpoint -- re-rendering".format(cache), flush=True)
    model, renderer, cfg_d, meta, it = load_model(ck, dev)
    model.eval(); renderer.eval()                                  # D49 C3
    print("[arm] {} iter {} | pool {} | loss_lo {} Hz".format(
        Path(run_dir).name, it, cfg_d.get("token_pool") or "masked_mean",
        cfg_d.get("loss_band_lo_hz", 0.0)), flush=True)

    rooms = []
    for c in sweep_configs():
        p = Path(data_dir) / c.filename
        if not p.exists():
            raise FileNotFoundError("sweep room not built: {}".format(p))
        with h5py.File(p) as f:
            T = np.asarray(f["ism/H_complex"])
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
            los = np.asarray(json.loads(f.attrs["line_of_sight"]), dtype=bool)
        P = render_room(model, renderer, "geom_token", c, rx, T.shape[1], dev)
        rooms.append({"cfg": c, "T": T, "P": P, "rx": rx, "los": los})
        print("  d_hat {:.3f}  n_rx {:5d}  probe1 {}".format(
            c.d_hat, len(rx), "LOS" if los[1] else "NLOS"), flush=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        blob = {"ckpt": str(ck), "iter": int(it)}
        for i, r in enumerate(rooms):
            blob["T{}".format(i)], blob["P{}".format(i)] = r["T"], r["P"]
            blob["rx{}".format(i)], blob["los{}".format(i)] = r["rx"], r["los"]
        np.savez_compressed(cache, **blob)
        print("[cache] wrote {}".format(cache), flush=True)
    return rooms, it, ck


def measure(rooms):
    """Every number the figures show, computed first (D62a)."""
    c0 = rooms[0]["cfg"]
    modes = enumerate_modes(c0.L, c0.W, f_max=200.0)[:N_MODES]
    bins = [int(round(m.f / DF_HZ)) for m in modes]
    per = []
    for r in rooms:
        sp = [_pearson(_db(r["P"][:, b]), _db(r["T"][:, b])) for b in bins
              if b < r["T"].shape[1]]
        lsd = float(np.mean(np.abs(_db(r["P"]) - _db(r["T"]))))
        per.append({"d_hat": r["cfg"].d_hat, "d": r["cfg"].d, "n_rx": int(len(r["rx"])),
                    "probe_edge_los": bool(r["los"][1]),
                    "nlos_frac": float(1.0 - r["los"].mean()),
                    "spatial_pearson": float(np.nanmean(sp)), "band_lsd_db": lsd,
                    "probe_edge_lsd_db": float(np.mean(np.abs(
                        _db(r["P"][1]) - _db(r["T"][1])))),
                    "probe_los_lsd_db": float(np.mean(np.abs(
                        _db(r["P"][0]) - _db(r["T"][0]))))})
    dh = np.array([p["d_hat"] for p in per])
    sl = (dh >= D_HAT_HOLDOUT[0]) & (dh <= D_HAT_HOLDOUT[1])
    R = np.array([p["spatial_pearson"] for p in per])
    return {"per_depth": per, "modes": [[int(m.n_x), int(m.n_y), float(m.f)] for m in modes],
            "mode_bins": bins,
            "in_slab_mean_pearson": float(np.nanmean(R[sl])),
            "out_slab_mean_pearson": float(np.nanmean(R[~sl])),
            "slab_deficit": float(np.nanmean(R[~sl]) - np.nanmean(R[sl])),
            "n_in_slab": int(sl.sum()),
            "pearson_min": float(np.nanmin(R)), "pearson_max": float(np.nanmax(R)),
            "probe_edge_nlos_from_d_hat": next(
                (p["d_hat"] for p in per if not p["probe_edge_los"]), None)}


# ------------------------------------------------------------------------------------ figures
def fig_accuracy(M, out):
    per = M["per_depth"]
    dh = [p["d_hat"] for p in per]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), dpi=DPI)
    for ax, key, lab, col in ((axes[0], "spatial_pearson", "spatial Pearson (mean over 6 modes)",
                               C_PRED),
                              (axes[1], "band_lsd_db", "band LSD (dB, 0-300 Hz)", C_GT)):
        ax.plot(dh, [p[key] for p in per], "o-", color=col, lw=2.0, ms=6)
        ax.axvspan(*D_HAT_HOLDOUT, color=C_SLAB, alpha=0.22, zorder=0,
                   label="held-out slab (no training shape)")
        t = M["probe_edge_nlos_from_d_hat"]
        if t is not None:
            ax.axvline(t, color="#444444", ls="--", lw=1.6,
                       label="probe receiver becomes NLOS")
        ax.set_xlabel(r"notch depth $\hat{d}=d/(0.45W)$", fontsize=12)
        ax.set_ylabel(lab, fontsize=12)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9.5, loc="best")
    axes[0].set_title("does accuracy hold through the unseen slab?", fontsize=13,
                      fontweight="bold")
    axes[1].set_title("reconstruction error vs depth", fontsize=13, fontweight="bold")
    fig.suptitle("Shape-edit sweep: L = 6.00 m, W = 5.00 m, w = 2.00 m fixed; 20 unseen depths  "
                 "|  in-slab R {:+.3f} vs out-of-slab {:+.3f} (deficit {:+.3f})".format(
                     M["in_slab_mean_pearson"], M["out_slab_mean_pearson"], M["slab_deficit"]),
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005,
             "Every point is a room the model never saw. A model that INTERPOLATES shape gives a "
             "flat curve; one that memorizes dips over the shaded band, where the corpus is "
             "empty by construction.",
             ha="center", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.045, 1, 0.92])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_waterfall(rooms, M, out, probe=1):
    """The fixed probe receiver's spectrum, stacked against depth. FDTD | prediction."""
    f = np.arange(rooms[0]["T"].shape[1]) * DF_HZ
    off = 12.0
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.4), dpi=DPI, sharey=True)
    for ax, key, title, col in ((axes[0], "T", "FDTD ground truth", C_GT),
                                (axes[1], "P", "predicted (unseen shapes)", C_PRED)):
        for i, r in enumerate(rooms):
            y = _db(r[key][probe]) - np.median(_db(r[key][probe])) + i * off
            nlos = not bool(r["los"][probe])
            ax.plot(f, y, lw=1.15, color=col, alpha=0.95 if nlos else 0.45)
            ax.text(f[-1] + 3, i * off, "{:.2f}{}".format(r["cfg"].d_hat, "*" if nlos else ""),
                    va="center", fontsize=7.5, color="#333333")
        for i, r in enumerate(rooms):
            if D_HAT_HOLDOUT[0] <= r["cfg"].d_hat <= D_HAT_HOLDOUT[1]:
                ax.axhspan(i * off - off / 2, i * off + off / 2, color=C_SLAB, alpha=0.16,
                           zorder=0)
        ax.set_xlim(0, 300); ax.set_xlabel("frequency (Hz)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_yticks([])
    axes[0].set_ylabel(r"notch depth $\hat{d}$  (offset traces, median-centred)", fontsize=12)
    fig.suptitle("Shape-edit waterfall at a FIXED receiver {}  |  bold = the receiver is in "
                 "shadow (NLOS); pink band = the held-out slab".format(list(PROBE_EDGE)),
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005,
             "The receiver does not move between traces, so every change is the shape edit. "
             "Modes migrate with depth in the FDTD stack; the question is whether the predicted "
             "stack migrates with them.",
             ha="center", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_strip(rooms, M, out, n_show=5):
    """Five depths at one mode; predicted above, FDTD below, one shared scale."""
    k = min(3, len(M["mode_bins"]) - 1)
    b = M["mode_bins"][k]
    nx, ny, fhz = M["modes"][k]
    idx = np.linspace(0, len(rooms) - 1, n_show).round().astype(int)
    vals = np.concatenate([_db(rooms[i]["T"][:, b]) for i in idx])
    vmax = float(np.percentile(vals, 99.5)); vmin = vmax - 40.0

    fig, axes = plt.subplots(2, n_show, figsize=(4.0 * n_show, 8.6), dpi=DPI)
    for col, i in enumerate(idx):
        r = rooms[i]
        for row, key, lab in ((0, "P", "predicted"), (1, "T", "FDTD")):
            ax = axes[row, col]
            s = ax.scatter(r["rx"][:, 0], r["rx"][:, 1], c=_db(r[key][:, b]), s=5,
                           vmin=vmin, vmax=vmax, cmap="magma")
            ax.add_patch(MplPolygon(r["cfg"].verts, closed=True, fill=False, ec="black",
                                    lw=1.8, zorder=6))
            ax.plot(*SRC, marker="*", ms=13, color="#00E5FF", mec="black", mew=0.6, zorder=9)
            ax.plot(*PROBE_EDGE, marker="o", ms=8, mfc="none", mec="#00E5FF", mew=2.0, zorder=9)
            ax.set_xlim(-0.25, 6.25); ax.set_ylim(-0.25, 5.25); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(lab, fontsize=13, fontweight="bold")
        in_slab = D_HAT_HOLDOUT[0] <= r["cfg"].d_hat <= D_HAT_HOLDOUT[1]
        axes[0, col].set_title(r"$\hat{{d}}$ = {:.2f}{}".format(
            r["cfg"].d_hat, "  (held-out slab)" if in_slab else ""), fontsize=12,
            fontweight="bold", color=C_SLAB if in_slab else "black")
    cb = fig.colorbar(s, ax=axes, fraction=0.017, pad=0.01)
    cb.set_label("|H| (dB, shared scale)", fontsize=11)
    fig.suptitle("Field at mode ({},{}) = {:.1f} Hz as the notch deepens  |  same model, same "
                 "source, only the shape changes".format(int(nx), int(ny), fhz),
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01,
             "Cyan star = source, cyan ring = the fixed probe receiver. Scatter, not an image: "
             "the receivers fill a non-convex polygon and gridding would invent values inside "
             "the notch.",
             ha="center", fontsize=10.5)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/p4_2/stage2/p4_2_s2_mean_masked")
    ap.add_argument("--data-dir", default="data/track_p4_2_sweep")
    ap.add_argument("--out", default="outputs/p4_2/stage2/sweep")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cache", default=None,
                    help="npz of rendered predictions; reused when the checkpoint matches")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rooms, it, ck = collect(a.run_dir, a.data_dir, dev, a.checkpoint,
                            cache=(a.cache or str(out / "sweep_render.npz")))
    M = measure(rooms)
    M["arm"] = Path(a.run_dir).name
    M["checkpoint"] = str(ck)
    M["iter"] = int(it)
    M["probe_los"], M["probe_edge"] = list(PROBE_LOS), list(PROBE_EDGE)
    json.dump(M, open(out / "sweep_metrics.json", "w"), indent=1, default=float)

    # numbers before pictures (D62a)
    print("\n=========== shape-edit sweep: {} @ iter {} ===========".format(M["arm"], it))
    print("  d_hat   n_rx  NLOS%  probe1   spatialR   LSD dB")
    for p in M["per_depth"]:
        flag = "  <-- held-out slab" if D_HAT_HOLDOUT[0] <= p["d_hat"] <= D_HAT_HOLDOUT[1] else ""
        print("  {:.3f}  {:5d}  {:5.1f}  {:5s}   {:+.4f}   {:6.2f}{}".format(
            p["d_hat"], p["n_rx"], 100 * p["nlos_frac"],
            "LOS" if p["probe_edge_los"] else "NLOS", p["spatial_pearson"],
            p["band_lsd_db"], flag))
    print("\n  in-slab mean R {:+.4f} (n={})   out-of-slab {:+.4f}   deficit {:+.4f}".format(
        M["in_slab_mean_pearson"], M["n_in_slab"], M["out_slab_mean_pearson"],
        M["slab_deficit"]))
    print("  range over the whole sweep: {:+.4f} .. {:+.4f}".format(
        M["pearson_min"], M["pearson_max"]))
    print("  probe receiver enters shadow at d_hat {}".format(
        M["probe_edge_nlos_from_d_hat"]), flush=True)

    fig_accuracy(M, out / "figI_sweep_accuracy.png")
    fig_waterfall(rooms, M, out / "figJ_sweep_waterfall.png")
    fig_strip(rooms, M, out / "figK_sweep_field_strip.png")
    for f in ("figI_sweep_accuracy.png", "figJ_sweep_waterfall.png",
              "figK_sweep_field_strip.png"):
        print("  {}  {:.1f} MB".format(out / f, (out / f).stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
