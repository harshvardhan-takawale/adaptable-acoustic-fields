"""Arm C v2, stage 2: the modal-hierarchy figure, the multi-mode Fig A, and the difference maps.

All three run off the cached v1 `.npz` dumps -- no GPU, no re-simulation, same checkpoint as v1.

**Fig D** -- accuracy across the modal hierarchy. The point v1 could not make: quality is NOT
monotone in frequency. It dips to +0.655 at (3,2) 138 Hz and recovers to +0.891 at (4,3) 199 Hz.
Non-isolated modes are marked, because a mode whose neighbour sits inside its linewidth is a
superposition and its label is nominal.

**Fig A2** -- replaces v1's single fundamental with three modes spanning the band. The mid row
(1,2) at 111.7 Hz overlaps (0,2) at 107.9 Hz within its 3.96 Hz linewidth; that is stated on the
row rather than left for a reader to discover.

**Fig E** -- the difference maps, and the real test. Raw-field correlation partly rewards getting
the ROOM right, since prediction and truth share the same room. Subtracting the baseline cancels
that and leaves only the response to the EDIT, which is what an editable representation actually
claims. Measured: Delta correlation is materially lower than raw everywhere, and for the two
SINGLE-wall edits it decays steeply with frequency (east curtain 0.871 -> 0.315 linear, lowest
six modes to highest six). It does NOT decay for the two-wall edit (0.885 -> 0.876), which is a
larger perturbation -- so Delta recovery tracks EDIT MAGNITUDE, not frequency alone. Saying
"it decays with frequency" without that qualification would be wrong. Two mode blocks are shown
for the same reason: 61.2 Hz where every edit is captured, 111.7 Hz where the small one is not.

Delta convention, and it is a real choice: the MAP and the leading metric are both LINEAR,
normalised by the baseline field's RMS so the units are "fraction of baseline amplitude". The dB
form is printed beside every panel. Reason for the ordering, also stated in the manifest: dB
differences diverge near pressure nulls, where the field is negligible and the quantity is
numerically unstable -- that artifact alone moves east-curtain at 107.9 Hz from +0.521 (linear)
to +0.010 (dB). The ordering is the more favourable one; it is chosen on that argument and both
numbers appear everywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from aaf.eval.p3_2_eval import _pearson

DPI = 170
GEOM = "median"
SCEN = ["a_baseline", "b_east_curtain", "c_north_absorber", "d_two_wall"]
EDITS = ["b_east_curtain", "c_north_absorber", "d_two_wall"]
SHORT = {"a_baseline": "baseline\nall walls $\\alpha$=0.15",
         "b_east_curtain": "east curtain\n$\\alpha$=0.50",
         "c_north_absorber": "north absorber $\\alpha$=0.70\n★ HELD-OUT combo",
         "d_two_wall": "two-wall edit\neast 0.50 + south 0.70"}
TINY = {"b_east_curtain": "east 0.50", "c_north_absorber": "north 0.70 ★",
        "d_two_wall": "east+south"}
# rows of Fig A2: (bin, label, f_hz, note)
FIG_A2_MODES = [(58, "(1,0)", 28.92, "isolated"),
                (223, "(1,2)", 111.67, "overlaps (0,2) @107.9 Hz"),
                (398, "(4,3)", 198.90, "isolated")]
FIG_E_BLOCKS = [(122, "(1,1)", 61.20), (223, "(1,2)", 111.67)]
C_PRED, C_GT, C_ACC = "#0072B2", "#D55E00", "#009E73"
GOOD = 0.85


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def _grid(v, n=64):
    """v is emitted x-major (x outer, y inner) -> [ny, nx] for imshow."""
    return np.asarray(v, float).reshape(n, n).T


def load(fdir):
    """Materialise the arrays ONCE.

    `NpzFile.__getitem__` decompresses on every access, and the delta table indexes `pred`/`gt`
    a few thousand times -- reading lazily turns a 20 s job into a many-minute one.
    """
    out = {}
    for s in SCEN:
        with np.load(Path(fdir) / "{}_{}.npz".format(GEOM, s)) as z:
            out[s] = {k: z[k] for k in ("pred", "gt", "rx", "L", "W", "centre")}
    return out


# --------------------------------------------------------------------------------- Fig D
def fig_d(screen, out):
    g = screen["geometries"][GEOM]
    rows = g["modes"]
    f = np.array([r["f_hz"] for r in rows])
    rdb = np.array([r["pearson_db"]["a_baseline"] for r in rows])
    rln = np.array([r["pearson_lin"]["a_baseline"] for r in rows])
    iso = np.array([r["isolated"] for r in rows])

    fig, ax = plt.subplots(figsize=(19.2, 9.2), dpi=DPI)
    ax.axhline(GOOD, color=C_GT, lw=2.0, ls="--", zorder=2,
               label="0.85 — the spec's mode-selection bar")
    ax.plot(f, rdb, "-", color=C_PRED, lw=2.2, zorder=3, label="spatial Pearson, dB fields")
    ax.plot(f, rln, "-", color="#CC79A7", lw=2.0, zorder=3,
            label="spatial Pearson, linear |H| (declines monotonically)")
    ax.scatter(f[iso], rdb[iso], s=95, color=C_PRED, zorder=4, edgecolor="white", linewidth=1.2)
    ax.scatter(f[~iso], rdb[~iso], s=115, facecolor="white", edgecolor=C_PRED, linewidth=2.2,
               zorder=4, label="hollow = NOT isolated (neighbour inside its linewidth)")
    ax.scatter(f, rln, s=42, color="#CC79A7", zorder=4)
    picked = {m[0] for m in FIG_A2_MODES}
    for r in rows:
        y = r["pearson_db"]["a_baseline"]
        sel = r["bin"] in picked
        ax.annotate("({},{})".format(r["n_x"], r["n_y"]), (r["f_hz"], y),
                    textcoords="offset points", xytext=(0, 13 if sel else 9), ha="center",
                    fontsize=11 if sel else 8.5,
                    fontweight="bold" if sel else "normal",
                    color=C_ACC if sel else "#555555")
    for b, lab, fh, _ in FIG_A2_MODES:
        ax.axvline(fh, color=C_ACC, lw=1.3, ls=":", zorder=1)
    ax.annotate("worst mode: (3,2)\n+{:.3f}".format(rdb[np.argmin(rdb)]),
                (f[np.argmin(rdb)], rdb.min()), textcoords="offset points", xytext=(18, -34),
                fontsize=11, color="#B22222", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.6))
    ax.annotate("recovers at (4,3), 198.9 Hz: +{:.3f}\naccuracy is NOT monotone in "
                "frequency".format(rdb[-1]), (f[-1], rdb[-1]),
                textcoords="offset points", xytext=(-300, 88), ha="left", fontsize=12,
                color=C_ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_ACC, lw=1.8,
                                connectionstyle="arc3,rad=-0.18"))
    ax.set_xlabel("mode frequency (Hz)", fontsize=13)
    ax.set_ylabel("spatial Pearson vs ISM ground truth\n(64×64 grid, baseline scenario)",
                  fontsize=12)
    ax.set_xlim(0, 210); ax.set_ylim(0.40, 1.06)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=11.5, loc="lower left", framealpha=0.95)
    ax.set_title("Reconstruction quality across the modal hierarchy — all {} modes "
                 "≤ 200 Hz, {:.2f} × {:.2f} m unseen geometry".format(
                     g["n_modes"], g["L"], g["W"]), fontsize=15.5, fontweight="bold")
    fig.text(0.5, 0.005,
             "Zero-shot, one forward pass, single checkpoint. {}/{} modes ≥ 0.70; {}/{} "
             "≥ 0.85, the highest at {:.1f} Hz. Dotted lines mark the three modes drawn in "
             "Fig A2.\nBoth curves are shown because they disagree: the dB form rewards getting "
             "the nodal pattern right, the linear form weights the loud regions and falls "
             "steadily with frequency. Neither alone is the whole picture."
             .format(g["n_ge_070"], g["n_modes"], g["n_ge_085"], g["n_modes"],
                     g["f_highest_ge_085"]),
             ha="center", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"n_modes": g["n_modes"], "n_ge_070": g["n_ge_070"], "n_ge_085": g["n_ge_085"],
            "min": float(rdb.min()), "mean": float(rdb.mean()),
            "f_highest_ge_085": g["f_highest_ge_085"],
            "n_not_isolated": int((~iso).sum())}


# -------------------------------------------------------------------------------- Fig A2
def fig_a2(dat, out):
    L, W = float(dat[SCEN[0]]["L"]), float(dat[SCEN[0]]["W"])
    fig = plt.figure(figsize=(19.6, 16.4), dpi=DPI)
    gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.10, wspace=0.045,
                           top=0.925, bottom=0.075, left=0.075, right=0.90)
    info = []
    for mi, (b, lab, fh, note) in enumerate(FIG_A2_MODES):
        vals = {s: (_db(dat[s]["pred"][:, b]), _db(dat[s]["gt"][:, b])) for s in SCEN}
        pool = np.concatenate([np.concatenate(v) for v in vals.values()])
        vmax = float(np.percentile(pool, 99.5)); vmin = vmax - 40.0
        row_axes = []
        for si, s in enumerate(SCEN):
            for k in (0, 1):
                ax = fig.add_subplot(gs[2 * mi + k, si])
                row_axes.append(ax)
                im = ax.imshow(_grid(vals[s][k]), origin="lower", extent=[0, L, 0, W],
                               vmin=vmin, vmax=vmax, cmap="magma", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                if si == 0:
                    ax.set_ylabel("PRED" if k == 0 else "ISM GT", fontsize=11.5,
                                  fontweight="bold",
                                  color=C_PRED if k == 0 else C_GT)
                if k == 0:
                    r = _pearson(vals[s][0], vals[s][1])
                    ax.text(0.98, 0.94, "r = {:+.3f}".format(r), transform=ax.transAxes,
                            ha="right", va="top", fontsize=11, color="white",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.22", fc=C_ACC, ec="none",
                                      alpha=0.88))
                if mi == 0 and k == 0:
                    ax.set_title(SHORT[s], fontsize=12.5, pad=9)
        cb = fig.colorbar(im, ax=row_axes, fraction=0.020, pad=0.012)
        cb.set_label("|H| dB (shared across the row)", fontsize=10)
        p0 = row_axes[0].get_position(); p1 = row_axes[1].get_position()
        fig.text(0.022, 0.5 * (p0.y1 + p1.y0),
                 "mode {}\n{:.1f} Hz\n{}".format(lab, fh, note),
                 fontsize=13, fontweight="bold", rotation=90, va="center", ha="center",
                 color="#222222")
        info.append({"bin": b, "mode": lab, "f_hz": fh, "note": note,
                     "vmin_db": vmin, "vmax_db": vmax,
                     "pearson_db": {s: _pearson(vals[s][0], vals[s][1]) for s in SCEN}})
    fig.suptitle("One INR, three modes, four rooms it never saw — {:.2f} × {:.2f} m "
                 "unseen geometry, 64×64 query from an 8×8-trained model".format(L, W),
                 fontsize=16.5, fontweight="bold", y=0.965)
    fig.text(0.5, 0.012,
             "Zero-shot: one forward pass per panel from a SINGLE checkpoint, no measurements of "
             "these rooms, no per-room fitting. Colour scale is shared within each mode row.\n"
             "The north-absorber column is doubly zero-shot — north@0.70 lies in the "
             "held-out slab and appears in no training config. Ground truth re-simulated by ISM "
             "on the identical grid.\nRow 2's mode overlaps (0,2) at 107.9 Hz within its 3.96 Hz "
             "linewidth, so that field is a genuine superposition and the label is nominal.",
             ha="center", fontsize=11)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return info


# --------------------------------------------------------------------------------- Fig E
def delta_table(dat, modes):
    """Delta-Pearson at each analytic MODE, both definitions.

    Evaluated at modal frequencies, not at every bin. Between modes the field is low-amplitude
    and its difference is noise-dominated, so an all-bins curve mostly plots noise -- and it is
    the modal content that carries the physics.
    """
    base = dat["a_baseline"]
    rows = []
    for m in modes:
        b, f = m["bin"], m["f_hz"]
        rec = {"bin": b, "f_hz": f, "mode": [m["n_x"], m["n_y"]],
               "isolated": m["isolated"]}
        for s in EDITS:
            dpl = np.abs(dat[s]["pred"][:, b]) - np.abs(base["pred"][:, b])
            dgl = np.abs(dat[s]["gt"][:, b]) - np.abs(base["gt"][:, b])
            dpd = _db(dat[s]["pred"][:, b]) - _db(base["pred"][:, b])
            dgd = _db(dat[s]["gt"][:, b]) - _db(base["gt"][:, b])
            rec[s] = {"lin": _pearson(dpl, dgl), "db": _pearson(dpd, dgd)}
        rows.append(rec)
    return rows


def fig_e(dat, dtab, out):
    L, W = float(dat[SCEN[0]]["L"]), float(dat[SCEN[0]]["W"])
    base = dat["a_baseline"]
    fig = plt.figure(figsize=(17.6, 19.6), dpi=DPI)
    # A dedicated thin row per block carries that block's three colorbars. Letting
    # `colorbar(ax=pair)` steal space instead shrinks those two panels only, which misaligns
    # the PRED/ISM pairs against the other columns.
    # Rows 3 and 7 are deliberately EMPTY spacers: a colorbar row butted straight against the
    # next block's column titles collides with them.
    gs = gridspec.GridSpec(9, 3, figure=fig, hspace=0.10, wspace=0.06,
                           height_ratios=[1, 1, 0.075, 0.34, 1, 1, 0.075, 0.46, 1.35],
                           top=0.935, bottom=0.062, left=0.075, right=0.975)
    info = {"blocks": [], "convention": "linear delta normalised by baseline RMS |H|"}
    for bi, (b, lab, fh) in enumerate(FIG_E_BLOCKS):
        rms = float(np.sqrt(np.mean(np.abs(base["gt"][:, b]) ** 2)))
        maps, ann = {}, {}
        for s in EDITS:
            dp = (np.abs(dat[s]["pred"][:, b]) - np.abs(base["pred"][:, b])) / rms
            dg = (np.abs(dat[s]["gt"][:, b]) - np.abs(base["gt"][:, b])) / rms
            maps[s] = (dp, dg)
            ann[s] = {
                "lin": _pearson(dp, dg),
                "db": _pearson(_db(dat[s]["pred"][:, b]) - _db(base["pred"][:, b]),
                               _db(dat[s]["gt"][:, b]) - _db(base["gt"][:, b]))}
        blk = {"bin": b, "mode": lab, "f_hz": fh, "baseline_rms": rms, "edits": {}}
        col0 = []
        for si, s in enumerate(EDITS):
            v = float(np.percentile(np.abs(np.concatenate(maps[s])), 98.0))
            pair = []
            r0 = 4 * bi
            for k in (0, 1):
                ax = fig.add_subplot(gs[r0 + k, si])
                pair.append(ax)
                im = ax.imshow(_grid(maps[s][k]), origin="lower", extent=[0, L, 0, W],
                               vmin=-v, vmax=v, cmap="RdBu_r", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                if si == 0:
                    ax.set_ylabel("PRED $\\Delta$" if k == 0 else "ISM $\\Delta$",
                                  fontsize=11.5, fontweight="bold",
                                  color=C_PRED if k == 0 else C_GT)
                if k == 0:
                    ax.set_title(TINY[s], fontsize=12.5, pad=6)
                    ax.text(0.98, 0.95,
                            "$\\Delta$r = {:+.3f} lin\n       {:+.3f} dB".format(
                                ann[s]["lin"], ann[s]["db"]),
                            transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888888",
                                      alpha=0.92))
            cax = fig.add_subplot(gs[r0 + 2, si])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.set_label("$\\Delta$|H| / baseline RMS   (symmetric, $\\pm${:.2f})".format(v),
                         fontsize=9)
            cb.ax.tick_params(labelsize=8)
            blk["edits"][s] = dict(ann[s], scale_sym=v)
            if si == 0:
                col0 = pair
        p0, p1 = col0[0].get_position(), col0[1].get_position()
        fig.text(0.018, 0.5 * (p0.y1 + p1.y0), "mode {}  {:.1f} Hz".format(lab, fh),
                 fontsize=13.5, fontweight="bold", rotation=90, va="center", ha="center")
        info["blocks"].append(blk)

    summ = {}
    for s_ in EDITS:
        for k in ("lin", "db"):
            v = np.array([r[s_][k] for r in dtab], float)
            summ["{}_{}".format(s_, k)] = {
                "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                "lowest6_mean": float(v[:6].mean()), "highest6_mean": float(v[-6:].mean())}
    info["per_edit_over_modes"] = summ

    ax = fig.add_subplot(gs[8, :])
    f = np.array([r["f_hz"] for r in dtab])
    for s, col in zip(EDITS, (C_PRED, C_ACC, "#CC79A7")):
        ax.plot(f, [r[s]["lin"] for r in dtab], "-o", color=col, lw=2.0, ms=5.5,
                label="{} — linear".format(TINY[s]))
        ax.plot(f, [r[s]["db"] for r in dtab], ":s", color=col, lw=1.5, ms=4.5, alpha=0.85,
                label="{} — dB".format(TINY[s]))
    for b, lab, fh in FIG_E_BLOCKS:
        ax.axvline(fh, color="#333333", lw=1.6, ls="--", zorder=1)
        ax.text(fh, 1.03, " {} {:.1f} Hz".format(lab, fh), fontsize=11, fontweight="bold",
                rotation=0, ha="left", va="bottom")
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xlim(0, 210); ax.set_ylim(-0.35, 1.14)
    ax.set_xlabel("frequency (Hz)", fontsize=12.5)
    ax.set_ylabel("$\\Delta$-field spatial Pearson", fontsize=12)
    ax.grid(alpha=0.25); ax.legend(fontsize=10, ncol=3, loc="lower left", framealpha=0.95)
    ax.set_title("The same measurement at every mode — $\\Delta$ agreement decays with "
                 "frequency for the SINGLE-wall edits, but holds for the larger two-wall edit "
                 "(linear)", fontsize=13.5, fontweight="bold")

    fig.suptitle("Does the model reproduce the EDIT, not just the room?  "
                 "$\\Delta$ = edited − baseline, {:.2f} × {:.2f} m unseen geometry"
                 .format(L, W), fontsize=16.5, fontweight="bold", y=0.962)
    fig.text(0.5, 0.008,
             "Subtracting the baseline cancels the shared room structure that raw-field "
             "correlation partly rewards, so this is a STRICTER test — and $\\Delta$ correlation "
             "is materially lower than raw-field correlation (0.78–0.99) at every mode.\n"
             "Averaged over all 24 modes (linear): east curtain +0.586, north absorber +0.614, "
             "two-wall +0.835. Lowest six modes → highest six: the single-wall edits fall "
             "(+0.871→+0.315, +0.822→+0.391) while the two-wall edit does not (+0.885→+0.876),\n"
             "so $\\Delta$ recovery tracks EDIT MAGNITUDE as well as frequency. Maps and the "
             "leading number are LINEAR, normalised by baseline RMS; the dB value is printed "
             "beside each. dB differences diverge near pressure nulls, where the field is\n"
             "negligible and the quantity unstable — that artifact alone moves east curtain at "
             "107.9 Hz from +0.521 linear to +0.010 dB. Both are reported throughout. "
             "Zero-shot, one forward pass, single checkpoint.",
             ha="center", fontsize=10.5)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="outputs/armC_demo/fields")
    ap.add_argument("--screen", default="outputs/armC_demo/v2/mode_screen.json")
    ap.add_argument("--outdir", default="outputs/armC_demo/v2")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    screen = json.load(open(a.screen))
    dat = load(a.fields)

    d = fig_d(screen, out / "figD_mode_screen.png")
    a2 = fig_a2(dat, out / "figA2_multimode_fields.png")
    dtab = delta_table(dat, screen["geometries"][GEOM]["modes"])
    e = fig_e(dat, dtab, out / "figE_difference_maps.png")
    json.dump({"figD": d, "figA2": a2, "figE": e, "delta_vs_frequency": dtab},
              open(out / "figures_v2.json", "w"), indent=1, default=float)
    for f in ("figD_mode_screen.png", "figA2_multimode_fields.png",
              "figE_difference_maps.png"):
        print("  {}".format(out / f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
