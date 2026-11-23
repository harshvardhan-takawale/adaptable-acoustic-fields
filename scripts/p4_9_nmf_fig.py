"""P4-9 F5 (D180 2c): NMF's modes and fields in one unseen U room and one unseen C3 room.

Selection rules (fixed before the figure is drawn):
  arm   the NMF arm with the higher unseen 25-100 Hz r at 30K (outputs/p4_9/nmf/{NMF,NMFP}_30K.json);
  rooms per family, the unseen room at that arm's MEDIAN r_25_100 (lower median of 8), so neither best nor worst.
Per room:
  modes   the first 6 TRUE modes (m1 eigenpairs, `data/track_p4_9_eigen/dense`, mode 0 excluded) above the first 6
          PREDICTED modes, both sorted by eigenfrequency, as listed -- no re-matching. Each predicted mode's sign is set
          to correlate positively with the true mode above it; titles give f true / f predicted and |corr| at the dense
          receivers.
  fields  |H| in dB at 50 Hz and 80 Hz: FDTD truth, NMF, R0 @120K (the standard report's unseen cache); one colour
          scale per frequency (truth's 2nd-98th percentile).
  PYTHONPATH=$PWD python scripts/p4_9_nmf_fig.py  -> outputs/p4_9/meeting/F5_nmf_modes.png (+ .json: rooms, corrs, f)
  ... --draw   redraw from outputs/p4_9/nmf/cache/F5_<ARM>.npz (CPU)
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

NMF_OUT = Path("outputs/p4_9/nmf")
FIG = Path("outputs/p4_9/meeting/F5_nmf_modes.png")
C = 343.0
BINS = {50: 100, 80: 160}  # 0.5 Hz bins


def _choose():
    arms = {}
    for a in ("NMF", "NMFP"):
        arms[a] = json.loads((NMF_OUT / "{}_30K.json".format(a)).read_text())
    arm = max(arms, key=lambda a: arms[a]["summary"]["unseen16"]["r_25_100"])
    rows = [r for r in arms[arm]["rooms"] if r["group"] == "unseen"]
    pick = {}
    for fam in ("U", "C3"):
        fr = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["r_25_100"])
        pick[fam] = fr[(len(fr) - 1) // 2]
    return arm, arms[arm], pick


def _grid(rx, L, W, vals):
    """the dense receivers are the 64 x 64 lattice snapped to the 2 cm FDTD nodes (steps 8 / 10 cm): index by rank."""
    xs, ys = np.unique(np.round(rx[:, 0], 4)), np.unique(np.round(rx[:, 1], 4))
    assert len(xs) == 64 and len(ys) == 64, (len(xs), len(ys))
    i = np.searchsorted(xs, np.round(rx[:, 0], 4))
    j = np.searchsorted(ys, np.round(rx[:, 1], 4))
    g = np.full((64, 64), np.nan)
    g[j, i] = vals
    return g


def compute(arm, pick):
    import torch

    from aaf.models.nmf_2d import NMF2D
    from aaf.train.nmf_2d import LO, HI, NMFCfg, Room, predict, room_field
    from scripts.p4_8_unseen_eval import rooms
    dev = torch.device("cuda")
    st = torch.load("outputs/p4_9/arms/{}/ckpt_iter0030000.pt".format(arm), map_location="cpu")
    cfg = NMFCfg(**st["cfg"])
    model = NMF2D(hidden=cfg.hidden).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    rs, _ = rooms()
    r0 = np.load("outputs/p4_8/cache/unseen_R0_120K.npz")
    out = {}
    for fam, row in pick.items():
        fn = row["room"]
        stem = Path(fn).stem
        lab = dict(np.load(Path(cfg.dense_labels_dir) / (stem + ".npz")))
        with h5py.File(Path("data/track_p4_8_unseen") / fn) as f:
            T = np.asarray(f["ism/H_complex"])[:, :601]
            verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
        L, W = max(v[0] for v in verts), max(v[1] for v in verts)
        room = Room(lab, verts, L, W, dev, h_fd=cfg.h_fd)
        with torch.no_grad():
            _, k, _, phi, _ = predict(model, room, room.rx)
            P = np.zeros((len(lab["rx"]), 601), np.complex64)
            for s0 in range(0, len(lab["rx"]), 4096):
                P[s0:s0 + 4096, LO:HI] = room_field(model, room, room.rx[s0:s0 + 4096])[0].cpu().numpy()
        j = [n for n, c in enumerate(rs) if c.filename == fn][0]
        rx = np.asarray(lab["rx"])
        assert np.allclose(r0["rx%d" % j], rx)
        out[fam] = {"room": fn, "L": L, "W": W, "rx": rx, "T": T, "P": P, "R0": r0["P%d" % j],
                    "psi": np.asarray(lab["psi_r"])[:, 1:11], "f_true": C * np.sqrt(lab["k2"][1:11]) / (2 * np.pi),
                    "phi": phi.cpu().numpy()[:, :6], "f_pred": C * k.cpu().numpy()[:6] / (2 * np.pi),
                    "r_25_100": row["r_25_100"]}
    return out


def draw(arm, summ, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888888"})
    fig, axes = plt.subplots(6, 6, figsize=(15, 15.5), dpi=150, facecolor="white")
    meta = {"arm": arm, "rooms": {}}
    for b, fam in enumerate(("U", "C3")):
        e = d[fam]
        rx, L, W = e["rx"], e["L"], e["W"]
        ext = (0, L, 0, W)
        cors, best = [], []
        for m in range(6):
            t, p = e["psi"][:, m], e["phi"][:, m]
            c = float(np.corrcoef(t, p)[0, 1])
            p = p * (1 if c >= 0 else -1)
            cors.append(abs(c))
            cb = [abs(float(np.corrcoef(e["psi"][:, n], e["phi"][:, m])[0, 1])) for n in range(e["psi"].shape[1])]
            best.append((int(np.argmax(cb)) + 1, float(max(cb))))
            for rr, (v, lab) in enumerate(((t, "true"), (p, "NMF"))):
                ax = axes[3 * b + rr, m]
                lim = np.nanmax(np.abs(v))
                ax.imshow(_grid(rx, L, W, v), origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
                ttl = ("{} mode {}: {:.1f} Hz".format(lab, m + 1, e["f_true"][m]) if rr == 0 else
                       "NMF {}: {:.1f} Hz, |r| {:.2f}\n~ true {} ({:.1f} Hz), |r| {:.2f}".format(
                           m + 1, e["f_pred"][m], abs(c), best[m][0], e["f_true"][best[m][0] - 1], best[m][1]))
                ax.set_title(ttl, fontsize=9, color="#222222")
        col = 0
        for hz, bi in BINS.items():
            tdb = 20 * np.log10(np.abs(e["T"][:, bi]) + 1e-12)
            lo, hi = np.percentile(tdb, [2, 98])
            for name, F in (("FDTD truth", e["T"]), ("NMF", e["P"]), ("R0 @120K", e["R0"])):
                ax = axes[3 * b + 2, col]
                db = 20 * np.log10(np.abs(F[:, bi]) + 1e-12)
                ax.imshow(_grid(rx, L, W, db), origin="lower", extent=ext, cmap="Blues", vmin=lo, vmax=hi)
                ax.set_title("{} at {} Hz".format(name, hz), fontsize=9.5, color="#222222")
                col += 1
        axes[3 * b, 0].annotate("{} room (median {} r): {}".format(fam, arm, e["room"].replace(".h5", "")),
                                xy=(0, 1.42), xycoords="axes fraction", fontsize=11.5, color="#222222", ha="left")
        meta["rooms"][fam] = {"room": e["room"], "r_25_100": e["r_25_100"], "abs_corr_modes_1_6": cors,
                              "best_match_true_mode_of_1_10": best, "f_true_1_10": e["f_true"].tolist(),
                              "f_pred": e["f_pred"].tolist()}
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#eeeeee")
    u, i_ = summ["unseen16"], summ["in_family15"]
    fig.text(0.01, 0.995, "F5 -- the neural modal field ({} @30K): predicted vs true modes, and fields at 50 / 80 Hz, in two "
             "unseen rooms".format(arm), va="top", fontsize=14, color="#222222")
    fig.text(0.01, 0.977, "Arm: higher unseen r at 30K. Rooms: each family's median-r unseen room. Modes sorted by "
             "frequency, not re-matched (2nd line: best-correlated of the first 10 true modes); red / blue = sign.\n"
             "Fields: |H| dB, one scale per frequency (truth's 2-98%). {} @30K, 25-100 Hz r: in-family {:.3f}, unseen {:.3f}; eigenfrequency MAE (10 modes) {:.1f} / {:.1f} Hz."
             .format(arm, i_["r_25_100"], u["r_25_100"], i_["k_mae_hz_10"], u["k_mae_hz_10"]),
             va="top", fontsize=9.5, color="#555555", linespacing=1.4)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.01, hspace=0.55, wspace=0.08)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, facecolor="white")
    FIG.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def main():
    import sys
    arm, rep, pick = _choose()
    print("arm", arm, {f: (r["room"], round(r["r_25_100"], 3)) for f, r in pick.items()}, flush=True)
    cache = NMF_OUT / "cache" / "F5_{}.npz".format(arm)
    if "--draw" in sys.argv and cache.exists():   # redraw on a CPU from the cached arrays
        z = np.load(cache, allow_pickle=True)
        d = {f: z[f].item() for f in ("U", "C3")}
    else:
        d = compute(arm, pick)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{f: np.array(v, dtype=object) for f, v in d.items()})
    draw(arm, rep["summary"], d)


if __name__ == "__main__":
    main()
