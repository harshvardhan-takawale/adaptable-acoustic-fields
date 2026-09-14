"""P4-4 demo v2: build FIGURE_MANIFEST.md from the per-sweep metrics each figure emitted.

The manifest is written FROM THE JSON THE FIGURE WROTE, never from the spec. If a sweep was
re-parameterised and the figure re-rendered, the manifest follows automatically; if a figure did
not render, it is listed as missing rather than described as if it existed. A manifest that
documents a figure nobody produced is worse than no manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["s1_notch_width", "s2_room_width", "s3a_rect_L_U", "s3b_rect_L_DN",
         "s3c_rect_L_U_fav", "gallery"]

HEAD = """# P4-4 demo v2 -- morph-figure expansion pack

Every figure below is **zero-shot**: the rooms are unseen shapes, each frame is **one forward
pass**, nothing is optimised at demo time, and **no measurement of any of these rooms** is used
-- the model is given the boundary polygon and nothing else.

## How to read the numbers

`spatial Pearson` is the correlation between predicted and FDTD |H| in dB across all receivers,
**averaged over the first 3 resolvable modes of each frame's own bounding box**. That is
`figN_morph`'s protocol, kept so a v2 strip can be read beside it. **Gate 2, the P4-2/P4-4
sweep curves and `p4_4_family_eval` use 6 modes**, so no number here is comparable to one of
theirs and **none of it may be quoted against a gate threshold**. The difference is not a fixed
offset in a known direction: on the P4-3 demo rooms the 3-mode number read above the gate, while
`s3c`'s terminal frame -- the same room the family evaluator scored +0.7968 -- reads +0.7052
here. Two things differ there, not one: the mode count, and the receiver set (800 scattered
points in the family corpus against ~2.6k on a 0.08 m grid).

`R(shown)` is the correlation at the single mode the figure actually draws. `band LSD` is the
mean absolute dB error over **every** bin in 0-300 Hz and every receiver -- a whole-band number,
not a modal one, which is why it is several dB even where the spatial correlation is high.

`enumerate_modes` is the analytic mode list of the **bounding box** (D79). Prediction and truth
are always read at the same bin, so a bin that misses a true resonance of a non-convex room
costs sensitivity; it cannot flatter the model.

Ground truth is 2D FDTD at `dx = 0.02 m`, `fs = 30720 Hz`, band 0-300 Hz at `df = 0.5 Hz`.
Receivers are on a 0.08 m grid with 0.12 m of wall clearance and 0.15 m cleared around **every**
reflex corner.

"""


def _table(met):
    show_f = (max(m["mode"][2] for m in met) - min(m["mode"][2] for m in met)) > 0.5
    cols = ["frame", "tokens", "n_rx", "NLOS %", "mode", "spatial r (3-mode)", "r (shown mode)",
            "band LSD dB"]
    if not show_f:
        cols.remove("mode")
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for m in met:
        row = [m["label"], str(m["n_tokens"]), str(m["n_rx"]),
               "{:.1f}".format(100 * m["nlos_frac"])]
        if show_f:
            row.append("({},{}) {:.1f} Hz".format(m["mode"][0], m["mode"][1], m["mode"][2]))
        row += ["**{:+.4f}**".format(m["spatial_pearson"]),
                "{:+.4f}".format(m["spatial_pearson_shown_mode"]),
                "{:.2f}".format(m["band_lsd_db"])]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/p4_4/demo/v2")
    ap.add_argument("--raw-base", default="https://raw.githubusercontent.com/"
                    "harshvardhan-takawale/adaptable-acoustic-fields/main")
    a = ap.parse_args()
    d = Path(a.dir)
    parts = [HEAD]

    # figN itself, corrected, so the pack is self-contained and the complement is visible.
    fn = Path("outputs/p4_4/demo/demo_metrics.json")
    if fn.exists():
        mo = json.load(open(fn))["morph"]
        parts.append("## fig 0 -- `figN_morph` (the original, title corrected)\n")
        parts.append("{}/outputs/p4_4/demo/figN_morph.png\n".format(a.raw_base))
        parts.append("* **Swept**: notch DEPTH, d_hat 0.00 -> 1.00 (a rectangle to the deepest "
                     "notch)\n"
                     "* **Fixed**: L = {:.2f} m, W = {:.2f} m, notch width {:.2f} m\n"
                     "* **Mode**: ({},{}) at {:.1f} Hz\n"
                     "* **Checkpoint**: `{}`\n".format(
                         mo["L"], mo["W"], mo["w"], mo["mode"][0], mo["mode"][1], mo["mode"][2],
                         mo["checkpoint"]))
        parts.append("The published title read `notch width 0.00 m`. `fig_morph` took the width "
                     "from frame 0, and frame 0 of a full-range morph is `d = 0` -- a pure "
                     "rectangle, stored with `w = 0.0`. Every notched frame carries `w = 2.00`. "
                     "Only the title changed; the render cache was intact, so the numbers are "
                     "the published ones.\n")
        # d_hat IS the frame label here; the per-room `tag` is "M" for every frame of the
        # morph and a column of identical "M"s tells the reader nothing.
        parts.append("| frame (d_hat) | notch depth d (m) | n_rx | spatial r (3-mode) | "
                     "band LSD dB |")
        parts.append("|---|---|---|---|---|")
        for m in mo["per_depth"]:
            parts.append("| {:.3f} | {:.2f} | {} | **{:+.4f}** | {:.2f} |".format(
                m["d_hat"], m["d"], m["n_rx"], m["spatial_pearson"], m["band_lsd_db"]))
        parts.append("")

    for i, name in enumerate(ORDER, start=1):
        j = d / "{}_metrics.json".format(name)
        if not j.exists():
            parts.append("## fig {} -- `{}`\n\n*Not rendered.*\n".format(i, name))
            continue
        z = json.load(open(j))
        met = z["per_frame"]
        parts.append("## fig {} -- `{}`\n".format(i, name))
        parts.append("{}/{}\n".format(a.raw_base, z["png"]))
        parts.append("* **Swept**: {}\n* **Fixed**: {}\n* **Checkpoint**: `{}` (iter {})\n"
                     "* **Mode drawn**: index {} of the bounding box's analytic list\n".format(
                         z["swept"], z["fixed"], z["checkpoint"], z["iter"],
                         z["mode_idx_shown"]))
        if z.get("note"):
            parts.append("* **Note**: {}\n".format(z["note"]))
        parts.append(_table(met))
        parts.append("\nMean spatial r **{:+.4f}**, mean band LSD **{:.2f} dB** over the "
                     "{} frames.\n".format(
                         sum(m["spatial_pearson"] for m in met) / len(met),
                         sum(m["band_lsd_db"] for m in met) / len(met), len(met)))

    (d / "FIGURE_MANIFEST.md").write_text("\n".join(parts))
    print("wrote {}".format(d / "FIGURE_MANIFEST.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
