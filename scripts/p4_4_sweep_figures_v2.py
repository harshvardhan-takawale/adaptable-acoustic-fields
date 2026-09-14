"""P4-4 demo v2: figN's format, generalised to any swept shape parameter.

`fig_morph` in `p4_3_demo_pack.py` draws one sweep beautifully and cannot draw any other, for
three reasons that are worth naming because each is a silent failure rather than a crash:

  1. **It introspects `.d` and `.w`.** `FamilyConfig` -- the multi-notch room -- has neither
     (a U has two depths, so a single scalar `d` is not defined) and every U/T/DN frame would
     raise `AttributeError` in the title, in `measure` and in `common_probe`.
  2. **It pins the axes to `rooms[0]`.** `set_xlim(-0.2, cfg0.L + 0.2)` is right when the box is
     fixed and CROPS every taller frame the moment the room size is what is being swept.
  3. **Its render cache is keyed on the checkpoint path alone.** Two different frame sets at one
     checkpoint would silently reuse each other's arrays -- a figure drawn from another sweep's
     fields, with no error anywhere.

So this is a separate script and `p4_3_demo_pack.py` keeps working unchanged. What is preserved
exactly, because the whole point is that a v2 strip can be read beside figN:

  * predicted row above, FDTD row below, one shared colour scale per row taken from the FDTD
    (`vmax` at the 99.5th percentile, 40 dB of range);
  * per-column spatial Pearson averaged over the first `N_MODES_SHOWN = 3` resolvable modes --
    figN's protocol, NOT Gate 2's six, which reads lower. The manifest states this;
  * the polygon outline in black, the source as a cyan star, `magma`, no ticks.

THE MODE IS RESOLVED PER FRAME, NOT ONCE. `enumerate_modes` is the analytic rectangular mode
list of the BOUNDING BOX (D79), so when the box is what is being swept -- `s2_room_width`, and
the gallery, where five different rooms sit side by side -- mode (0,1) sits at a different
frequency in every column. Reading every column at frame 0's bin would compare rooms at a
frequency that is resonant in one of them. Each column is therefore read at its OWN mode bin and
the column label prints the frequency whenever it varies. Prediction and truth are always read
at the SAME bin, so the D79 caveat holds: a bin that misses a true resonance of a non-convex
room costs sensitivity, it cannot flatter the model.
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

from aaf.data.shape_configs import SRC
from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.p3_2_eval import load_model
from scripts.p4_3_demo_pack import DF_HZ, DPI, N_MODES_SHOWN, _db, _pearson, render
from scripts.p4_4_sweeps import SWEEP_CKPT, SWEEPS, gallery_configs, validate

MIN_W, MIN_H = 2560, 1440          # the deck's requirement, asserted after every save
COL_W_IN = 3.40                    # figN's 2.95, widened so 8 columns clear MIN_H once the
GALLERY_COL_W_IN = 4.60            # panel height is driven by the room's aspect (see _figsize)
CHROME_H_IN = 1.95                 # suptitle + per-column titles + the caption block


def _figsize(n, col_w, Lmax, Wmax):
    """Size the canvas from the ROOM's aspect, not the other way round.

    The panels are `aspect="equal"`, so a subplot box that is wider than the room is tall leaves
    the difference as dead vertical space -- which is what a fixed figure height produces, and it
    shrinks the fields to a strip in the middle of a 1640 px canvas. Deriving the height from
    `(W + pad) / (L + pad)` makes the boxes match the rooms, so the height requirement is met by
    LARGER PANELS rather than by more whitespace.
    """
    ar = (Wmax + 0.4) / (Lmax + 0.4)
    # A WIDE, SHORT room (s3c is 5.70 x 4.02) gives a short canvas at any given column width, so
    # the column width is solved for rather than fixed: whatever makes two panel rows plus the
    # chrome clear MIN_H. The 1.08 is headroom for what `bbox_inches="tight"` trims off.
    col_w = max(col_w, ((MIN_H * 1.08) / DPI - CHROME_H_IN) / (2.0 * ar))
    return (col_w * n, 2.0 * col_w * ar + CHROME_H_IN)

CAPTION = ("Zero-shot: unseen shapes, ONE forward pass per frame, nothing optimised at demo "
           "time, and NO measurements of any of these rooms -- the model sees only the boundary "
           "polygon.\nSwept: {}.  Fixed: {}.  Checkpoint: {}.  {}")


def _mode_for(cfg, idx):
    """Mode `idx` of this frame's own bounding box, and its frequency bin.

    Index 1 by default, matching figN: index 0 is (0,0), the DC term, whose 'field' is flat and
    on which any model correlates near-perfectly for free.
    """
    modes = enumerate_modes(cfg.L, cfg.W, f_max=200.0)[:max(N_MODES_SHOWN, idx + 1)]
    return modes, int(round(modes[idx].f / DF_HZ))


def load_frames(frames, data_dir, model, renderer, dev, cache=None, ck=None):
    """Render every frame, caching on (checkpoint, EXACT frame list).

    The frame list is part of the key on purpose. `p4_3_demo_pack.load_rooms` keys on the
    checkpoint alone, which is safe there because one checkpoint draws one strip; here five
    sweeps share two checkpoints and a checkpoint-only key would serve one sweep's fields to
    another with no error raised anywhere.
    """
    import h5py
    key = "|".join(c.filename for _, c in frames)
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["ckpt"]) == str(ck) and str(z["key"]) == key:
            out = [{"label": lab, "cfg": c, "T": z["T{}".format(i)], "P": z["P{}".format(i)],
                    "rx": z["rx{}".format(i)], "los": z["los{}".format(i)]}
                   for i, (lab, c) in enumerate(frames)]
            print("[cache] reused {} frames from {}".format(len(out), cache), flush=True)
            return out
        print("[cache] {} is for another checkpoint or frame list -- re-rendering".format(cache),
              flush=True)
    out = []
    for lab, c in frames:
        p = Path(data_dir) / c.filename
        if not p.exists():
            raise FileNotFoundError("room not built: {}".format(p))
        with h5py.File(p) as f:
            T = np.asarray(f["ism/H_complex"])
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
            los = np.asarray(json.loads(f.attrs["line_of_sight"]), dtype=bool)
        P = render(model, renderer, c, rx, T.shape[1], dev)
        out.append({"label": lab, "cfg": c, "T": T, "P": P, "rx": rx, "los": los})
        print("  {:16s} {} tok  n_rx {:5d}".format(lab, c.n_tokens, len(rx)), flush=True)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        blob = {"ckpt": str(ck), "key": key}
        for i, r in enumerate(out):
            blob["T{}".format(i)], blob["P{}".format(i)] = r["T"], r["P"]
            blob["rx{}".format(i)], blob["los{}".format(i)] = r["rx"], r["los"]
        np.savez_compressed(cache, **blob)
        print("[cache] wrote {}".format(cache), flush=True)
    return out


def measure(rooms, mode_idx):
    """figN's numbers, frame by frame, each frame at its OWN mode bins."""
    out = []
    for r in rooms:
        modes, _b = _mode_for(r["cfg"], mode_idx)
        bins = [int(round(m.f / DF_HZ)) for m in modes[:N_MODES_SHOWN]]
        sp = [_pearson(_db(r["P"][:, b]), _db(r["T"][:, b])) for b in bins
              if b < r["T"].shape[1]]
        # From `modes`, not from `bins`: `bins` is truncated to N_MODES_SHOWN for the 3-mode
        # average, so indexing it with a mode_idx >= 3 would raise rather than draw the mode
        # that was asked for.
        b_show = int(round(modes[mode_idx].f / DF_HZ))
        out.append({
            "label": r["label"], "room": r["cfg"].label, "kind": r["cfg"].kind,
            "L": r["cfg"].L, "W": r["cfg"].W, "n_tokens": int(r["cfg"].n_tokens),
            "d_hat": float(r["cfg"].d_hat),
            "notches": [{"side": n.side, "d": n.d, "w": n.w, "x0": n.x0}
                        for n in r["cfg"].notches],
            "n_rx": int(len(r["rx"])), "nlos_frac": float(1.0 - r["los"].mean()),
            "mode": [int(modes[mode_idx].n_x), int(modes[mode_idx].n_y),
                     float(modes[mode_idx].f)],
            "spatial_pearson": float(np.nanmean(sp)),
            "spatial_pearson_per_mode": [float(x) for x in sp],
            "spatial_pearson_shown_mode": float(
                _pearson(_db(r["P"][:, b_show]), _db(r["T"][:, b_show]))),
            "band_lsd_db": float(np.mean(np.abs(_db(r["P"]) - _db(r["T"])))),
        })
    return out


def fig_sweep(rooms, met, mode_idx, out, swept, fixed, arm, note="", col_w=COL_W_IN,
              title=None):
    n = len(rooms)
    Lmax = max(r["cfg"].L for r in rooms); Wmax = max(r["cfg"].W for r in rooms)
    fig, axes = plt.subplots(2, n, figsize=_figsize(n, col_w, Lmax, Wmax), dpi=DPI,
                             gridspec_kw={"hspace": 0.06, "wspace": 0.06})
    bins = [_mode_for(r["cfg"], mode_idx)[1] for r in rooms]
    # One colour scale for the whole strip, from the FDTD row (figN's rule). Per-panel scaling
    # would make every frame look equally good and hide exactly what the strip is for.
    vals = np.concatenate([_db(r["T"][:, b]) for r, b in zip(rooms, bins)])
    vmax = float(np.percentile(vals, 99.5)); vmin = vmax - 40.0
    # Axes span the LARGEST frame, not the first: in a room-size sweep, pinning to frame 0 crops
    # every later column, and a cropped field map looks like a modelling failure.
    # Marker area scaled to the receiver COUNT. figN's s=3 is tuned for its ~4000-point 0.08 m
    # grid; the family corpus stores 800 scattered receivers per room, and s=3 there renders as
    # confetti with the field structure invisible between the dots.
    fs = [float(np.clip(12000.0 / max(len(r["rx"]), 1), 3.0, 30.0)) for r in rooms]
    freqs = [m["mode"][2] for m in met]
    show_f = (max(freqs) - min(freqs)) > 0.5      # the mode moves when the bbox is swept
    for k, r in enumerate(rooms):
        for row, which, lab in ((0, "P", "predicted"), (1, "T", "FDTD")):
            ax = axes[row, k]
            s = ax.scatter(r["rx"][:, 0], r["rx"][:, 1], c=_db(r[which][:, bins[k]]), s=fs[k],
                           vmin=vmin, vmax=vmax, cmap="magma")
            ax.add_patch(MplPolygon(r["cfg"].verts, closed=True, fill=False, ec="black",
                                    lw=1.5, zorder=6))
            ax.plot(*SRC, marker="*", ms=10, color="#00E5FF", mec="black", mew=0.5, zorder=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            ax.set_xlim(-0.2, Lmax + 0.2); ax.set_ylim(-0.2, Wmax + 0.2)
            if k == 0:
                ax.set_ylabel(lab, fontsize=13, fontweight="bold")
        head = r["label"]
        if show_f:
            head += "\n({},{}) {:.0f} Hz".format(met[k]["mode"][0], met[k]["mode"][1],
                                                 met[k]["mode"][2])
        axes[0, k].set_title("{}\nR={:+.3f}".format(head, met[k]["spatial_pearson"]),
                             fontsize=11)
    fig.colorbar(s, ax=axes, fraction=0.014, pad=0.01).set_label("|H| dB", fontsize=10)
    m0 = met[0]["mode"]
    mode_txt = ("mode ({},{}), {:.0f}-{:.0f} Hz across the row".format(
        m0[0], m0[1], min(freqs), max(freqs)) if show_f
        else "mode ({},{}) {:.0f} Hz".format(m0[0], m0[1], m0[2]))
    fig.suptitle("{}  |  {}  |  ckpt {}".format(title or swept, mode_txt, arm),
                 fontsize=16, fontweight="bold")
    # WRAP the caption to the canvas. `bbox_inches="tight"` grows the saved image to contain
    # every artist, so one long unwrapped line silently widens the PNG well past the panels and
    # leaves large dead margins -- the figure looks padded rather than full.
    import textwrap
    fig_w_in = fig.get_size_inches()[0]
    cols = max(60, int(fig_w_in / 0.082))
    body = "\n".join("\n".join(textwrap.wrap(ln, cols)) if ln else ""
                     for ln in CAPTION.format(swept, fixed, arm, note).split("\n"))
    fig.text(0.5, 0.008, body, ha="center", fontsize=11.5)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _assert_size(out)


def _assert_size(out):
    """A figure below the deck's size, or one that came out blank, must fail here and not on a
    projector. Blank is a real failure mode: an all-NaN field scatters without raising."""
    from PIL import Image
    im = Image.open(out)
    w, h = im.size
    if w < MIN_W or h < MIN_H:
        raise AssertionError("{} is {}x{}, below the required {}x{}".format(
            out, w, h, MIN_W, MIN_H))
    a = np.asarray(im.convert("L"))
    ink = float((a < 250).mean())
    if ink < 0.02:
        raise AssertionError("{} is {:.3%} ink -- effectively blank".format(out, ink))
    print("  {}  {}x{}  ink {:.1%}  {:.1f} MB".format(
        out, w, h, ink, Path(out).stat().st_size / 1e6), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="a key of SWEEPS, or 'gallery'")
    ap.add_argument("--data-dir", default="data/track_p4_4_sweeps")
    ap.add_argument("--family-dir", default="data/track_p4_4_family")
    ap.add_argument("--out", default="outputs/p4_4/demo/v2")
    ap.add_argument("--mode-idx", type=int, default=1,
                    help="which mode of the bounding box to draw; 1 = the lowest non-DC one, "
                         "as figN uses")
    ap.add_argument("--run-dir", default=None, help="override the sweep's checkpoint")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.sweep == "gallery":
        frames = gallery_configs()
        data_dir, run_dir = a.family_dir, "outputs/p4_4/p4_4_FAM"
        # NOT a sweep: nothing varies continuously, so saying "swept: ..." here would misdescribe
        # the figure. Five different FAMILIES, each at its own room's mode (0,1).
        swept = ("nothing -- this is a gallery, not a sweep: five different room FAMILIES "
                 "(rectangle, L, T, double-notch, U) side by side, each drawn at its own "
                 "room's mode (0,1)")
        fixed = ("five HELD-OUT rooms, one per family, chosen for comparable bounding boxes "
                 "(L 5.30-5.72 m, W 4.02-4.64 m) and a visible notch")
        note = ("This is a SELECTED set: 5 of the 50 held-out family rooms, picked for "
                "comparable bounding boxes and a visible notch, NOT the average room. For "
                "context, the family evaluator's means over all 50 are rect +0.847, L +0.766, "
                "T +0.758, DN +0.704, U +0.673 -- but those use its SIX-mode protocol and the "
                "per-panel r above uses figN's THREE-mode one. The two are NOT comparable in "
                "either direction, and neither is a gate result.")
        col_w = GALLERY_COL_W_IN
        title = "One model, five room shapes it never saw"
    else:
        frames, meta = validate(a.sweep)
        data_dir, run_dir = a.data_dir, SWEEP_CKPT[a.sweep]
        swept, fixed, note, col_w = meta["swept"], meta["fixed"], "", COL_W_IN
        title = meta.get("title")
    run_dir = a.run_dir or run_dir

    ck = sorted(Path(run_dir).glob("ckpt_iter*.pt"))[-1]
    model, renderer, cd, _meta, it = load_model(ck, dev)
    model.eval(); renderer.eval()                                   # D49 C3
    arm = Path(run_dir).name
    print("[arm] {} iter {} | pool {}".format(arm, it, cd.get("token_pool") or "masked_mean"),
          flush=True)
    print("=== {} ({} frames) ===".format(a.sweep, len(frames)), flush=True)

    rooms = load_frames(frames, data_dir, model, renderer, dev,
                        cache=str(out / "render_{}_{}.npz".format(a.sweep, arm)), ck=ck)
    met = measure(rooms, a.mode_idx)
    print("  {:16s} {:>4s} {:>6s} {:>6s} {:>10s} {:>9s} {:>8s}".format(
        "frame", "tok", "n_rx", "NLOS%", "spatialR", "R(shown)", "LSD dB"))
    for m in met:
        print("  {:16s} {:4d} {:6d} {:6.1f} {:+10.4f} {:+9.4f} {:8.2f}".format(
            m["label"], m["n_tokens"], m["n_rx"], 100 * m["nlos_frac"],
            m["spatial_pearson"], m["spatial_pearson_shown_mode"], m["band_lsd_db"]))

    # A second mode of the same sweep must not overwrite the first: the render cache is shared
    # (same frames, same checkpoint) so it costs seconds, but only if the outputs are distinct.
    tag = a.sweep if a.mode_idx == 1 else "{}_mode{}".format(a.sweep, a.mode_idx)
    png = out / "fig_{}.png".format(tag)
    fig_sweep(rooms, met, a.mode_idx, png, swept, fixed, arm, note=note, col_w=col_w,
              title=title)
    json.dump({"sweep": a.sweep, "arm": arm, "checkpoint": str(ck), "iter": int(it),
               "n_modes": N_MODES_SHOWN,
               "n_modes_note": ("spatial_pearson is averaged over the first 3 resolvable modes "
                                "of each frame's own bounding box -- figN's protocol. Gate 2 "
                                "uses 6 and reads lower; the two are not comparable."),
               "mode_idx_shown": a.mode_idx, "title": title, "swept": swept,
               "fixed": fixed, "note": note,
               "png": str(png), "per_frame": met},
              open(out / "{}_metrics.json".format(tag), "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
