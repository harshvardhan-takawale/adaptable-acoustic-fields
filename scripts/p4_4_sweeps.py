"""P4-4 demo v2: the SWEEP SPECS -- one module, imported by both the builder and the figures.

WHY THE SPECS LIVE APART FROM BOTH. The builder simulates the rooms and the figure script
renders and plots them, and they must agree on the frame list EXACTLY: a frame the builder
skipped is a missing .h5, and a frame the figure adds silently is a stale-cache read. Keeping
one list in one place makes that impossible rather than merely unlikely.

EVERY SWEEP IS A LINE THROUGH THE FAMILY, NOT A DRAW FROM IT. `figN_morph` fixes the bounding
box and the notch width and sweeps the DEPTH. These sweep the other knobs:

  s1_notch_width   figN's bounding box, figN's fixed depth, NOTCH WIDTH swept. figN grows the
                   corner downward; this grows it sideways. The direct complement.
  s2_room_width    the notch held fixed IN METRES while the ROOM grows. A size edit and a shape
                   edit go through the same interface, so they belong in the same format.
  s3a_rect_L_U     two-phase: NW depth 0 -> full (rect -> L), then NE depth 0 -> full (L -> U).
                   Boundary-token count steps 4 -> 6 -> 8 across the row.
  s3b_rect_L_DN    IDENTICAL to s3a for its first four frames, then the second notch goes to the
                   SE corner instead of NE. Same bounding box, same widths, same depths -- the
                   only difference is WHERE the second corner is removed, which is what makes
                   the U/DN accuracy gap attributable to placement rather than to the draw.
  s3c_rect_L_U_fav the same rect -> L -> U progression, run in the region of the family where
                   `p4_4_FAM` is measurably strongest. Its terminal frame IS the best-scoring
                   held-out U room (+0.7968), so the endpoint has an independent number.

WHY NEW FDTD RATHER THAN SNAPPING TO EXISTING ROOMS. Asked of the corpora on disk: across the
515 single-notch rooms the largest group sharing `(L, W, d)` holds TWO distinct notch widths and
a sweep needs eight; the largest sharing `(L, d, w)` holds ONE distinct `W`; and the largest U
group holds ONE distinct second-notch depth. Snapping would therefore move three parameters at
once down the row, which destroys the one thing the format exists to show. At figN's receiver
density a room costs ~50 s, so eight of them is ~1 minute as an array -- cheaper than the
compromise.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from aaf.data.multi_notch import FamilyConfig, Notch, geometry_ok
from aaf.data.shape_configs import D_FRAC_MAX, GRID_DX, W_FRAC_MAX

SWEEP_N = 8                    # 8 frames: 20 would be 58 inches wide and unreadable
SHAPE_ID0 = 970000


def _snap(x: float) -> float:
    """Every shape parameter must land on an FDTD node (D67c) or the solver quantises it for us."""
    return round(round(float(x) / GRID_DX) * GRID_DX, 2)


def _steps(lo: float, hi: float, n: int = SWEEP_N) -> List[float]:
    return [_snap(v) for v in np.linspace(lo, hi, n)]


# --------------------------------------------------------------------------- the five sweeps
def s1_notch_width():
    """figN's bounding box and a FIXED depth; the notch WIDTH sweeps 0.30 -> 2.70 m (= 0.45 L)."""
    L, W, d = 6.00, 5.00, 1.50                      # d_hat = 1.50 / (0.45 * 5.00) = 0.667
    out = []
    for i, w in enumerate(_steps(0.30, W_FRAC_MAX * L)):
        out.append(("w={:.2f}".format(w),
                    FamilyConfig(L, W, (Notch("NW", d, w),), split="sweep",
                                 shape_id=SHAPE_ID0 + i)))
    return out, {"title": "The morph, sideways: L = 6.00 m, W = 5.00 m and notch DEPTH "
                 "1.50 m all FIXED; the corner grows ACROSS and the field reorganises",
                 "swept": "notch width w, 0.30 -> 2.70 m",
                 "fixed": "L = 6.00 m, W = 5.00 m, notch depth d = 1.50 m (d_hat = 0.667)"}


def s2_room_width():
    """The notch is held FIXED IN METRES and the room grows around it: W sweeps 4.00 -> 5.50 m.

    `d_hat = d / (0.45 W)` therefore FALLS 0.667 -> 0.485 down the row even though the notch
    never changes. That is the honest reading of the figure and the caption says it: the shape
    edit and the size edit are the same interface, and the normalised depth is a consequence of
    the size, not an independent knob.
    """
    L, d, w = 6.00, 1.20, 2.00
    out = []
    for i, W in enumerate(_steps(4.00, 5.50)):
        out.append(("W={:.2f}".format(W),
                    FamilyConfig(L, W, (Notch("NW", d, w),), split="sweep",
                                 shape_id=SHAPE_ID0 + 100 + i)))
    return out, {"title": "Same notch, bigger room: L = 6.00 m and the notch (d = 1.20 m, "
                 "w = 2.00 m) all FIXED in metres; only the ROOM grows",
                 "swept": "room width W, 4.00 -> 5.50 m",
                 "fixed": "L = 6.00 m, notch d = 1.20 m and w = 2.00 m held in METRES, so "
                          "d_hat falls 0.667 -> 0.485 as a consequence of the size, not as a "
                          "second knob"}


def _two_phase(L, W, side2, w1, w2, d1_end, d2_end, id0, fam):
    """rect -> L -> <fam>. Four frames grow the NW notch, four grow the second one.

    Frame 1 is a true rectangle (4 boundary tokens), frames 2-4 are L-rooms (6) and frames 5-8
    carry the second reflex corner (8). ONE knob moves at a time, but there are two knobs -- so
    this is a two-phase morph and must not be captioned as a single-parameter sweep.
    """
    out = []
    for i, d in enumerate(_steps(0.0, d1_end, 4)):
        ns = () if d <= 0.0 else (Notch("NW", d, w1),)
        out.append(("NW d={:.2f}".format(d),
                    FamilyConfig(L, W, ns, split="sweep", shape_id=id0 + i)))
    for i, d in enumerate(_steps(d2_end / 4.0, d2_end, 4)):
        ns = (Notch("NW", d1_end, w1), Notch(side2, d, w2))
        out.append(("{} d={:.2f}".format(side2, d),
                    FamilyConfig(L, W, ns, split="sweep", shape_id=id0 + 10 + i)))
    return out, {"title": "Rectangle to L to {}: one bounding box, one notch width each; "
                          "the NW corner is removed, then the {} -- boundary tokens 4 -> 6 -> "
                          "8".format(fam, side2),
                 "swept": "notch DEPTH, in TWO phases: frames 1-4 grow the NW notch "
                          "(rect -> L), frames 5-8 grow the {} notch (L -> {}). Two knobs "
                          "move, one at a time -- this is not a single-parameter "
                          "sweep".format(side2, fam),
                 "fixed": "L = {:.2f} m, W = {:.2f} m, NW width {:.2f} m, {} width {:.2f} m; "
                          "boundary tokens 4 -> 6 -> 8".format(L, W, w1, side2, w2)}


def s3a_rect_L_U():
    """The user's chosen form, on figN's own bounding box."""
    return _two_phase(6.00, 5.00, "NE", 2.00, 2.00, 1.50, 1.50, SHAPE_ID0 + 200, "U")


def s3b_rect_L_DN():
    """The placement control: identical to s3a except the second notch is SE, not NE.

    Frames 1-4 are the SAME FOUR ROOMS as s3a's. `p4_4_FAM` scores held-out DN at +0.704 and U
    at +0.673, and the two families differ in the draw as well as the placement; holding the
    bounding box, both widths and both depths fixed removes the draw from the comparison.
    """
    return _two_phase(6.00, 5.00, "SE", 2.00, 2.00, 1.50, 1.50, SHAPE_ID0 + 300, "DN")


def s3c_rect_L_U_fav():
    """The same progression where `p4_4_FAM` is strongest -- terminal frame = its best U room.

    `UL5.70_W4.02_NEd1.74w2.22x0.00_NWd0.68w0.62x0.00` scores +0.7968 on the held-out family
    set, the highest of the ten. Ending the sweep there means the last column has a number
    measured by an evaluator that never saw this figure. NOTE THAT TWO THINGS DIFFER, NOT ONE:
    +0.7968 is the family evaluator's SIX-mode mean over the corpus's 800 scattered receivers,
    and this figure prints figN's THREE-mode mean over ~2.6k receivers on a 0.08 m grid. The
    endpoint is an independent check that the room is a good one to end on; it is NOT a number
    this figure should reproduce, and the gap between them is not evidence about either.
    """
    return _two_phase(5.70, 4.02, "NE", 0.62, 2.22, 0.68, 1.74, SHAPE_ID0 + 400, "U")


SWEEPS = {
    "s1_notch_width": s1_notch_width,
    "s2_room_width": s2_room_width,
    "s3a_rect_L_U": s3a_rect_L_U,
    "s3b_rect_L_DN": s3b_rect_L_DN,
    "s3c_rect_L_U_fav": s3c_rect_L_U_fav,
}

# Sweeps 1 and 2 stay inside the single-notch family P4-2/P4-3 trained on, so they take figN's
# checkpoint. The two-phase sweeps leave it -- a U and a DN have two reflex corners -- so they
# take the family-trained one. Every figure title names the checkpoint it used.
SWEEP_CKPT = {
    "s1_notch_width": "outputs/p4_3/p4_3_D_data",
    "s2_room_width": "outputs/p4_3/p4_3_D_data",
    "s3a_rect_L_U": "outputs/p4_4/p4_4_FAM",
    "s3b_rect_L_DN": "outputs/p4_4/p4_4_FAM",
    "s3c_rect_L_U_fav": "outputs/p4_4/p4_4_FAM",
}

# Five HELD-OUT family rooms already on disk, one per family, for the gallery. Chosen from the
# 50 test rooms for (a) comparable bounding boxes, so the analytic mode lists nearly coincide,
# and (b) a VISIBLE notch -- the tightest-bbox set contains a "T" whose notch is 4 cm deep, which
# is a rectangle wearing a label. This IS a selected set and the caption says so; the per-family
# held-out means are printed beside it so the selection cannot be mistaken for the average, with
# the protocol difference (6-mode there, 3-mode here) stated alongside.
GALLERY = [
    ("rectangle", "rectL5.62_W4.18"),
    ("L",         "LL5.56_W4.52_NWd1.10w2.22x0.00"),
    ("T",         "TL5.72_W4.64_midNd0.56w0.88x1.94"),
    ("double-notch", "DNL5.30_W4.50_NWd1.06w1.02x0.00_SEd0.34w1.70x0.00"),
    ("U",         "UL5.70_W4.02_NEd1.74w2.22x0.00_NWd0.68w0.62x0.00"),
]


FAMILY_MANIFEST = "configs/sweeps_2d_mat/p4_4_family_manifest.json"


def gallery_configs():
    """The five gallery rooms as `FamilyConfig`s, read from the frozen family manifest.

    ONE loader, shared by the builder and the figure script, for the same reason the sweep specs
    are shared: if the builder simulated one set of rooms and the figure drew another, the
    mismatch would surface as a missing file at best and as a figure of the wrong rooms at worst.
    """
    import json

    from aaf.data.multi_notch import configs_from_rows
    rows = json.load(open(FAMILY_MANIFEST))["configs"]
    by = {r["filename"][:-3]: r for r in rows}
    missing = [fn for _lab, fn in GALLERY if fn not in by]
    if missing:
        raise KeyError("gallery rooms absent from {}: {}".format(FAMILY_MANIFEST, missing))
    return [(lab, configs_from_rows([by[fn]])[0]) for lab, fn in GALLERY]


def validate(name: str):
    """Every frame must be a legal room BEFORE an array launches. `geometry_ok` is the gate."""
    frames, meta = SWEEPS[name]()
    assert len(frames) == SWEEP_N, "{}: {} frames, expected {}".format(
        name, len(frames), SWEEP_N)
    fns = [c.filename for _, c in frames]
    assert len(set(fns)) == len(fns), "{}: frame filenames collide: {}".format(name, fns)
    for lab, c in frames:
        if c.notches:
            ok, why = geometry_ok(c.L, c.W, c.notches)
            assert ok, "{} frame {}: {}".format(name, lab, why)
            for n in c.notches:
                assert n.d <= D_FRAC_MAX * c.W + 1e-9, "{} {}: depth over cap".format(name, lab)
                assert n.w <= W_FRAC_MAX * c.L + 1e-9, "{} {}: width over cap".format(name, lab)
    return frames, meta
