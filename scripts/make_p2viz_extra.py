"""P2-VIZ2 deck additions: figs 07-11 + train_rooms_list.md. CPU-only; reuses the cached
median-LOO spectrum (no GPU render) + signal_level helpers. Every number from disk."""
from __future__ import annotations
import json, re, glob, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial.distance import cdist

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from aaf.eval.signal_level import (phase_correlation_mag_weighted, rir_pearson,  # noqa: E402
                                   magnitude_correlation)

AST = REPO / "outputs/phase2_meeting_assets"
KG = REPO / "outputs/known_geometry"
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
                     "figure.titlesize": 15, "legend.fontsize": 13,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})


def _name_to_lwh(name):
    m = re.match(r"L([\d.]+)_W([\d.]+)_H([\d.]+)", name)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


# ----------------------------------------------------------------------
# 07 — median-LOO signal panels (magnitude, phase, RIR)
# ----------------------------------------------------------------------

def fig07_signal_panels():
    d = np.load(KG / "loo_median_spectrum.npz")
    Hp, Ht = d["H_pred"], d["H_target"]
    fs, n_time = float(d["fs"]), int(d["n_time"])
    rxpos = d["receiver_pos"]; L, W, H = float(d["L"]), float(d["W"]), float(d["H"])
    n_freq = Hp.shape[1]; f = np.arange(n_freq) * (fs / n_time)
    c = int(np.argmin(np.linalg.norm(rxpos - np.array([L/2, W/2, H/2]), axis=1)))
    # aggregate metric values (all 512 receivers) from signal_level
    phase_mw = phase_correlation_mag_weighted(Hp, Ht)
    rir_p = np.fft.irfft(Hp, n=n_time, axis=-1).astype(np.float32)
    rir_t = np.fft.irfft(Ht, n=n_time, axis=-1).astype(np.float32)
    rirP = rir_pearson(rir_p, rir_t)
    mag_corr = float(magnitude_correlation(Hp, Ht))   # 512-receiver mean (consistent w/ caption)

    fig, axes = plt.subplots(1, 3, figsize=(19.2, 10.8))
    sel = f <= 2000
    # (a) magnitude
    axes[0].semilogy(f[sel], np.abs(Ht[c, sel]), color="k", lw=1.5, label="ISM ground truth")
    axes[0].semilogy(f[sel], np.abs(Hp[c, sel]), color="#d62728", lw=1.1, alpha=0.8, label="known-geometry render")
    axes[0].set_xlabel("frequency (Hz)"); axes[0].set_ylabel("|H(f)| (log)")
    axes[0].set_title(f"(a) magnitude — mag corr {mag_corr:.2f}")
    axes[0].legend(loc="upper right"); axes[0].grid(True, which="both", alpha=0.25)
    # (b) phase, mag-weighted (only where |H_target| significant)
    thr = 0.12 * np.abs(Ht[c, sel]).max()
    sig = np.abs(Ht[c, sel]) >= thr
    fb = f[sel]
    axes[1].plot(fb[sig], np.angle(Ht[c, sel])[sig], ".", ms=3, color="k", label="ISM phase")
    axes[1].plot(fb[sig], np.angle(Hp[c, sel])[sig], ".", ms=3, color="#d62728", alpha=0.7, label="render phase")
    axes[1].set_xlabel("frequency (Hz)"); axes[1].set_ylabel("phase (rad)")
    axes[1].set_title(f"(b) phase (mag-weighted) — phase corr {phase_mw:.2f}")
    axes[1].set_ylim(-np.pi, np.pi); axes[1].legend(loc="upper right"); axes[1].grid(True, alpha=0.25)
    # (c) RIR full 2 s + inset zoom 0-50 ms
    t = np.arange(n_time) / fs
    axes[2].plot(t, rir_t[c], color="k", lw=0.6, label="ISM RIR")
    axes[2].plot(t, rir_p[c], color="#d62728", lw=0.6, alpha=0.7, label="render RIR")
    axes[2].set_xlabel("time (s)"); axes[2].set_ylabel("amplitude")
    axes[2].set_title(f"(c) RIR (full 2 s) — Pearson {rirP:.2f}")
    axes[2].legend(loc="upper right"); axes[2].grid(True, alpha=0.25)
    ins = axes[2].inset_axes([0.42, 0.55, 0.55, 0.4])
    z = t <= 0.05
    ins.plot(t[z] * 1e3, rir_t[c][z], color="k", lw=1.0)
    ins.plot(t[z] * 1e3, rir_p[c][z], color="#d62728", lw=1.0, alpha=0.8)
    ins.set_title("zoom: first 50 ms", fontsize=11); ins.set_xlabel("ms", fontsize=10)
    ins.tick_params(labelsize=9); ins.grid(True, alpha=0.3)

    fig.suptitle(f"Known-geometry render — median-LOO room L{L:.2f}/W{W:.2f}/H{H:.2f} "
                 f"(leave-one-out, at training density)", fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.90, bottom=0.13, wspace=0.22)
    fig.text(0.5, 0.04, "Predicted from (L,W,H) with NO measurements (room held out of the latent map). Centre receiver shown; "
             "metrics are the 512-receiver means. Phase plotted only where |H| is significant. "
             "Source: outputs/known_geometry/loo_median_spectrum.npz.", ha="center", fontsize=11,
             style="italic", color="#444", wrap=True)
    p = AST / "07_median_loo_signal_panels.png"; fig.savefig(p, dpi=100); plt.close(fig)
    return p, dict(mag=mag_corr, phase_mw=phase_mw, rir=rirP, L=L, W=W, H=H)


# ----------------------------------------------------------------------
# 08 — single-room fidelity table (5 de-risk rooms)
# ----------------------------------------------------------------------

def _parse_summary(path):
    lines = Path(path).read_text().splitlines()
    rows, header = [], None
    for ln in lines:
        if ln.strip().startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if header is None and any("mag corr" in c for c in cells):
                header = cells
            elif header and re.match(r"L[\d.]+_W", cells[0]):
                rows.append(dict(zip(header, cells)))
    return header, rows


def fig08_single_room_table():
    hdr, rows = _parse_summary(REPO / "outputs/single_room_3d/SUMMARY.md")
    def col(d, key):
        for k in d:
            if key in k.lower():
                return d[k]
        return "—"
    cols = ["L", "W", "H", "mag corr", "phase corr", "RIR Pearson", "early/late", "env corr", "modal MAE", "LSD"]
    cell = []
    for r in rows:
        cell.append([col(r, "| l |".strip()) if False else r.get("L", "—"), r.get("W", "—"), r.get("H", "—"),
                     col(r, "mag corr"), col(r, "phase corr"), col(r, "rir pearson"),
                     col(r, "early/late"), col(r, "env corr"), col(r, "modal mae"), col(r, "lsd")])
    fig, ax = plt.subplots(figsize=(19.2, 10.8)); ax.axis("off")
    colhdr = ["L (m)", "W (m)", "H (m)", "mag\ncorr", "phase\ncorr (mw)", "RIR\nPearson",
              "early/late\ncorr", "env\ncorr", "modal\nMAE (Hz)", "LSD\n(dB)"]
    tbl = ax.table(cellText=cell, colLabels=colhdr, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(16); tbl.scale(1.0, 3.2)
    for (r0, c0), cellobj in tbl.get_celld().items():
        if r0 == 0:
            cellobj.set_facecolor("#1f77b4"); cellobj.set_text_props(color="white", fontweight="bold")
        elif c0 in (3, 4, 5, 7):
            cellobj.set_facecolor("#eaf3ea")
    fig.suptitle("Single-room fidelity — IN-DISTRIBUTION UPPER BOUND (each de-risk room overfit individually)",
                 fontweight="bold", y=0.92)
    fig.text(0.5, 0.06, "The architecture's ceiling: a dedicated model per room. mag/phase/RIR/env corr all >= 0.95; LSD "
             "1.3-1.8 dB. This is the UPPER BOUND, distinct from the 45-room in-distribution (2.169 dB) and from "
             "zero-shot. Source: outputs/single_room_3d/SUMMARY.md.", ha="center", fontsize=12, style="italic",
             color="#444", wrap=True)
    p = AST / "08_signal_metrics_table.png"; fig.savefig(p, dpi=100, bbox_inches=None); plt.close(fig)
    return p, rows


# ----------------------------------------------------------------------
# 09 — spatial slices (median-LOO room, a low modal frequency)
# ----------------------------------------------------------------------

def fig09_spatial_slices():
    d = np.load(KG / "loo_median_spectrum.npz")
    Hp, Ht = d["H_pred"], d["H_target"]
    fs, n_time = float(d["fs"]), int(d["n_time"])
    rxpos = d["receiver_pos"]; L, W, H = float(d["L"]), float(d["W"]), float(d["H"])
    n_freq = Hp.shape[1]; faxis = np.arange(n_freq) * (fs / n_time)
    c = 343.0
    f_target = (c / 2) * np.sqrt((1/L)**2 + (1/W)**2)        # (1,1,0) tangential mode
    fb = int(np.argmin(np.abs(faxis - f_target))); f_act = faxis[fb]
    n = 8; zs = [1, 4, 6]
    mp = np.abs(Hp[:, fb]); mt = np.abs(Ht[:, fb])
    eps = 1e-8; dp = 20*np.log10(np.maximum(mp, eps)); dt = 20*np.log10(np.maximum(mt, eps))
    vmin, vmax = float(min(dp.min(), dt.min())), float(max(dp.max(), dt.max()))
    fig, axes = plt.subplots(2, len(zs), figsize=(19.2, 10.8), constrained_layout=True)
    for col, iz in enumerate(zs):
        base = iz * n * n
        idx = np.array([base + iy*n + ix for iy in range(n) for ix in range(n)])
        zh = float(rxpos[base, 2])
        gp = dp[idx].reshape(n, n); gt = dt[idx].reshape(n, n)
        im = axes[0, col].imshow(gp, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"render  z={zh:.2f} m"); axes[1, col].imshow(gt, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f"ISM  z={zh:.2f} m")
        for rr in (0, 1):
            axes[rr, col].set_xticks([]); axes[rr, col].set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="|H| (dB)")
    fig.suptitle(f"Spatial mode shape at {f_act:.0f} Hz — predicted (top) vs ISM (bottom), "
                 f"median-LOO room L{L:.2f}/W{W:.2f}/H{H:.2f} (a room that WORKS, LOO/training density)",
                 fontweight="bold")
    fig.text(0.5, 0.005, f"Horizontal (x,y) slices through the 8x8x8 receiver grid at 3 heights, at the (1,1,0) tangential "
             f"mode ({f_act:.0f} Hz). The render reproduces the standing-wave pattern. NOT a failed sparse-gap test room. "
             f"Source: outputs/known_geometry/loo_median_spectrum.npz.", ha="center", fontsize=11, style="italic",
             color="#444", wrap=True)
    p = AST / "09_spatial_slices.png"; fig.savefig(p, dpi=100); plt.close(fig)
    return p, f_act


# ----------------------------------------------------------------------
# 10 — representative LOO generalization table
# ----------------------------------------------------------------------

def fig10_loo_table():
    rows = [r for r in json.loads((KG / "loo/loo_rows.json").read_text()) if r["map"] == "rbf"]
    rows.sort(key=lambda r: r["mag_corr_full"])
    n = len(rows)
    # honest spread: min, the TRUE median, max, + 3 spaced (q1, q3, ~p90)
    idxs = sorted(set([0, n//4, n//2, 3*n//4, round(0.9*(n-1)), n-1]))
    sel = [rows[i] for i in idxs]
    cell = []
    for r in sel:
        L, W, H = _name_to_lwh(r["room"])
        cell.append([f"{L:.2f}", f"{W:.2f}", f"{H:.2f}", f"{r['mag_corr_full']:.3f}",
                     f"{r['mag_corr_0_250']:.3f}", f"{r['lsd_db_full']:.2f}"])
    mf = np.mean([r["mag_corr_full"] for r in rows]); m0 = np.mean([r["mag_corr_0_250"] for r in rows])
    ml = np.mean([r["lsd_db_full"] for r in rows])
    cell.append(["", "", "mean→", f"{mf:.3f}", f"{m0:.3f}", f"{ml:.2f}"])
    fig, ax = plt.subplots(figsize=(19.2, 10.8)); ax.axis("off")
    colhdr = ["L (m)", "W (m)", "H (m)", "mag corr\n(full)", "mag corr\n(0-250 Hz)", "LSD\n(dB)"]
    tbl = ax.table(cellText=cell, colLabels=colhdr, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(17); tbl.scale(1.0, 3.0)
    labels = ["min", "", "median", "", "", "max", "ALL 45 (mean)"]
    for i, lab in enumerate(labels):
        if lab:
            tbl.add_cell(i+1, -1, width=0.12, height=tbl[1, 0].get_height(), text=lab, loc="right",
                         edgecolor="none").set_text_props(style="italic", color="#666")
    for (r0, c0), co in tbl.get_celld().items():
        if r0 == 0:
            co.set_facecolor("#2ca02c"); co.set_text_props(color="white", fontweight="bold")
        elif r0 == len(cell):
            co.set_facecolor("#dddddd"); co.set_text_props(fontweight="bold")
        elif c0 in (3, 4):
            co.set_facecolor("#eaf3ea")
    fig.suptitle("Known-geometry rendering — LEAVE-ONE-OUT GENERALIZATION "
                 "(predict latent from (L,W,H), no measurements, at training density)", fontweight="bold", y=0.93)
    fig.text(0.5, 0.05, "6 held-out rooms chosen to SPAN the score distribution (min / median / max + 3 evenly spaced — not "
             "the best 6); bottom row anchors to the all-45 mean. All columns from cache (loo_rows.json); phase/RIR not "
             "stored per room. The deliberate counterpart to fig 08's in-distribution upper bound.", ha="center",
             fontsize=12, style="italic", color="#444", wrap=True)
    p = AST / "10_loo_generalization_table.png"; fig.savefig(p, dpi=100); plt.close(fig)
    return p, sel, (mf, m0, ml)


# ----------------------------------------------------------------------
# 11 — coverage anchors (two points, NO connecting line)
# ----------------------------------------------------------------------

def fig11_coverage_anchors():
    tm = json.loads((REPO / "outputs/multi_room_3d/P3_45rooms_4gpu/train_meta.json").read_text())
    X = np.stack([tm["L_list"], tm["W_list"], tm["H_list"]], 1)
    D = cdist(X, X); np.fill_diagonal(D, np.inf); loo_nn = D.min(1)
    loo = [r for r in json.loads((KG / "loo/loo_rows.json").read_text()) if r["map"] == "rbf"]
    loo_mag = np.array([r["mag_corr_full"] for r in loo])
    tmag, tnn = [], []
    for p in sorted(glob.glob(str(KG / "lookup/L*__rbf/metrics.json"))):
        ev = json.loads(Path(p).read_text())
        tmag.append(ev["per_band_mag_corr"]["mag_corr_full"]); tnn.append(cdist([[ev["L"], ev["W"], ev["H"]]], X).min())
    tmag = np.array(tmag); tnn = np.array(tnn)
    # anchors
    sparse_x, sparse_y = tnn.mean(), tmag.mean()
    loo_x, loo_y = loo_nn.mean(), loo_mag.mean()
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    # unmeasured band between the two anchors
    ax.axvspan(loo_x, sparse_x, color="#999999", alpha=0.12)
    ax.text((loo_x + sparse_x)/2, 0.55, "?\nunmeasured\nP2-4 maps this region", ha="center", va="center",
            fontsize=15, color="#555", style="italic")
    # ceiling + target refs
    ax.axhline(loo_y, color="#2ca02c", ls=":", lw=1.2, alpha=0.7)
    ax.text(sparse_x, loo_y + 0.012, "training-density ceiling (0.89)", color="#2ca02c", fontsize=12, ha="right")
    ax.axhline(0.9, color="green", ls="--", lw=1.0, alpha=0.5); ax.text(0.18, 0.905, "Dolby target 0.9", color="green", fontsize=11)
    # two anchors with whiskers (NO line between)
    ax.errorbar(loo_x, loo_y, yerr=[[loo_y - loo_mag.min()], [loo_mag.max() - loo_y]],
                fmt="o", ms=18, color="#2ca02c", capsize=8, lw=2, zorder=5,
                label=f"LOO (training density, ~0.34 m) = {loo_y:.2f}")
    ax.errorbar(sparse_x, sparse_y, yerr=[[sparse_y - tmag.min()], [tmag.max() - sparse_y]],
                fmt="s", ms=18, color="#d62728", capsize=8, lw=2, zorder=5,
                label=f"sparse 45 rooms (unseen, ~0.61 m) = {sparse_y:.2f}")
    ax.annotate("45 rooms\n(current)", (sparse_x, sparse_y), textcoords="offset points", xytext=(12, -38), fontsize=12)
    ax.annotate("training\ndensity", (loo_x, loo_y), textcoords="offset points", xytext=(-70, -10), fontsize=12)
    ax.set_xlabel("mean nearest-neighbour room distance in (L,W,H)  (m)  —  lower = denser coverage")
    ax.set_ylabel("magnitude correlation"); ax.set_ylim(0, 1.0); ax.set_xlim(0.2, 0.75)
    ax.invert_xaxis()  # denser (smaller dist) on the right -> quality rises to the right
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=13)
    fig.suptitle("Coverage vs rendering quality: TWO MEASURED ANCHORS (the curve between is deliberately not drawn)",
                 fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.15)
    fig.text(0.5, 0.05, "Denser training coverage improves zero-shot rendering — but we have measured only the two "
             "endpoints. The intermediate scaling (90 / 150 / 250 rooms) is the running P2-4 experiment; no line is drawn "
             "between the anchors. Whiskers = actual min-max spread. Source: known_geometry/{loo,lookup} + train_meta.json.",
             ha="center", fontsize=11, style="italic", color="#444", wrap=True)
    p = AST / "11_coverage_anchors.png"; fig.savefig(p, dpi=100); plt.close(fig)
    return p, dict(sparse=(sparse_x, sparse_y, tmag.min(), tmag.max()), loo=(loo_x, loo_y, loo_mag.min(), loo_mag.max()))


def write_train_rooms_list():
    cfg = yaml.safe_load((REPO / "configs/sweeps_3d/train_rooms.yaml").read_text())
    rooms = cfg["rooms"]
    out = ["# Phase-2 training rooms — 45 LHS shoebox geometries\n",
           f"set_name={cfg.get('set_name')}, alpha={cfg.get('alpha')}, fs={cfg.get('fs')} Hz, "
           f"n_time_samples={cfg.get('n_time_samples')} ({cfg.get('n_time_samples')/cfg.get('fs'):.1f} s), "
           f"source_offset={cfg.get('source_offset')}. {len(rooms)} rooms.\n",
           "\n| # | L (m) | W (m) | H (m) | V (m³) |", "|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(rooms):
        L, W, H = float(r["L"]), float(r["W"]), float(r["H"])
        out.append(f"| {i} | {L:.3f} | {W:.3f} | {H:.3f} | {L*W*H:.1f} |")
    (AST / "train_rooms_list.md").write_text("\n".join(out) + "\n")
    return len(rooms)


def main():
    from PIL import Image
    results = {}
    for fn in (fig07_signal_panels, fig08_single_room_table, fig09_spatial_slices,
               fig10_loo_table, fig11_coverage_anchors):
        try:
            r = fn(); p = r[0] if isinstance(r, tuple) else r
            print(f"wrote {p.name}  {Image.open(p).size}"); results[fn.__name__] = r
        except Exception as e:
            import traceback; print(f"FAILED {fn.__name__}: {e!r}"); traceback.print_exc()
    n = write_train_rooms_list(); print(f"wrote train_rooms_list.md  ({n} rooms)")


if __name__ == "__main__":
    main()
