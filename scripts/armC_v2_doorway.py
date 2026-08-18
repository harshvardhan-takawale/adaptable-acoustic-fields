"""Arm C v2, stage 0: the doorway-physics motivator. GROUND TRUTH ONLY -- no model anywhere.

This figure exists to motivate the NEXT phase (aperture as a trainable edit axis), not to report
a result. Every panel is FDTD simulation. That is stated on the figure itself, because a reader
who mistakes a simulated field for a prediction draws exactly the wrong conclusion from it.

Domain is FT-B's frozen one (`scripts/p3_3fast_ftb.py`): 8.0 x 4.0, divider at x = 4.0, alpha =
0.15 on every surface including the divider, source at (0.5, 0.5) in room A, dx = 0.01, fs =
61440. Reusing it means the inter-room level differences already published in
`outputs/p3_3fast/trackB/aperture_sweep.json` annotate these panels without recomputation, and
any discrepancy is a real signal rather than a new-setup artifact.

Three apertures, built with the canonical recipe (`aaf/data/aperture_configs.py`):
  * sealed a = 0.0   -- slab with NO `apertures` key; room B disconnects EXACTLY (H_B == 0)
  * mid    a = 1.0   -- slab + one centred aperture
  * open   a = 4.0   -- the EMPTY LIST, not a W-wide aperture (which leaves staircased tips)

Mode: FT-B's own PRIMARY_MODE, sub-room (1,1) at 60.71 Hz. Its even branch ends on full-domain
(2,1) at 60.63 Hz, so the sealed case shows two independent sub-room fields and the open case
shows one field spanning the whole domain -- the coupling story at a single frequency. The
nearest other sub-room mode is 17.7 Hz away, so the short record used here cannot blend them.

Cost note: `simulate` has no whole-grid output, so a dense RECEIVER array is the supported
route. It allocates `ir_t`, `ir`, `H_complex` AND `H_deconv` all at n_rx x n (H_deconv has no
skip flag), so the binding constraint is memory, not time -- shrink `n`, never `n_rx`. n = 30720
gives T = 0.5 s and df = 2.0 Hz, ample when the nearest competing mode is 17.7 Hz away.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import aaf.sim.fdtd_2d as F

# ------------------------------------------------------------------ FT-B's frozen domain
C = 343.0
L, W = 8.0, 4.0
ALPHA = 0.15
DIV_X = 4.0
SRC = (0.5, 0.5)
DX = 0.01
FS = 61440.0                 # fs MUST scale with 1/dx; 12288 at dx=0.01 raises on CFL
N = 30720                    # T = 0.500 s, df = 2.0 Hz  (FT-B used 122880 for 0.5 Hz)
L_SUB = 3.99

#: sub-room (1,1). FT-B's PRIMARY_MODE, chosen there because its nearest sub-room neighbours
#: are 17.7 and 25.0 Hz away, so nothing else can leak into the map.
MODE_F_HZ = (C / 2.0) * np.sqrt((1.0 / L_SUB) ** 2 + (1.0 / W) ** 2)
BAND_LO, BAND_HI = 20.0, 300.0

APERTURES = (0.0, 1.0, 4.0)
#: Published broadband inter-room level difference, `aperture_sweep.json`. Annotated, not recomputed.
PUBLISHED_LD_DB = {0.0: float("-inf"), 1.0: -7.15, 4.0: -1.45}

N_RX_X, N_RX_Y = 128, 64
MARGIN = 0.20

#: FT-B's own protocol, replayed verbatim as a REPRODUCTION check. Its published level
#: differences were measured with `n = 122880` (T = 2.0 s) on a 16x8 grid at a 0.3 m margin,
#: and with the estimator `20 log10(mean|H|)` -- a mean of AMPLITUDE, not of power. The dense
#: pass above differs in all three respects, so the two are not expected to agree to the
#: decimal and the discrepancy must be attributed rather than assumed benign.
FTB_N = 122880
FTB_MARGIN = 0.3


def ftb_grid16():
    """FT-B's `receiver_grids()[1]` verbatim -- 16x8 over the full domain, no nudge needed
    (x = 4.0 is not sampled, and the nearest point snaps to node 395 against a divider at 400)."""
    ys = np.linspace(FTB_MARGIN, W - FTB_MARGIN, 8)
    xs = np.linspace(FTB_MARGIN, L - FTB_MARGIN, 16)
    return np.array([[x, y] for x in xs for y in ys], dtype=float)


def ld_amplitude(H, freqs, sel_a, sel_b, lo=BAND_LO, hi=BAND_HI):
    """FT-B's `band_level_ratio`: 20 log10( <|H|>_B / <|H|>_A ). -inf when room B is silent."""
    m = (freqs >= lo) & (freqs <= hi)
    ma = float(np.mean(np.abs(H[sel_a][:, m])))
    mb = float(np.mean(np.abs(H[sel_b][:, m])))
    if ma <= 0.0:
        return float("nan")
    return float("-inf") if mb <= 0.0 else float(20.0 * np.log10(mb / ma))


def ld_power(H, freqs, sel_a, sel_b, lo=BAND_LO, hi=BAND_HI):
    """Energy-mean variant: 10 log10( <|H|^2>_B / <|H|^2>_A ). Weights loud receivers more."""
    m = (freqs >= lo) & (freqs <= hi)
    pa = float(np.mean(np.abs(H[sel_a][:, m]) ** 2))
    pb = float(np.mean(np.abs(H[sel_b][:, m]) ** 2))
    if pa <= 0.0:
        return float("nan")
    return float("-inf") if pb <= 0.0 else float(10.0 * np.log10(pb / pa))


def extra_walls_for(a):
    """Divider spec for aperture ``a`` -- canonical form, matching aperture_configs.py:156."""
    if a >= W:
        return []                                    # fully open: no wall at all
    spec = {"type": "slab", "axis": "x", "pos": DIV_X, "alpha": ALPHA}
    if a > 0.0:
        spec["apertures"] = [(0.5 * W - 0.5 * a, 0.5 * W + 0.5 * a)]
    return [spec]                                    # a == 0: no 'apertures' key -> sealed


def dense_receivers():
    """Dense grid over the FULL domain, with every receiver pushed clear of the divider column.

    Without the nudge `_snap_nodes` raises ("snaps onto a solid node") for the sealed and mid
    cases while succeeding for the open one -- a per-aperture failure on a property that must
    be per-domain. Same rule as `build_p3_3fast_trackB.receivers`, so all three apertures share
    one receiver array and the panels are directly comparable.
    """
    dxx = float(L) / int(round(float(L) / DX))
    i_div = int(round(DIV_X / dxx))
    xs = []
    for x in np.linspace(MARGIN, L - MARGIN, N_RX_X):
        i = int(round(float(x) / dxx))
        if abs(i - i_div) <= 1:
            i = i_div + 2 if i >= i_div else i_div - 2
            x = i * dxx
        xs.append(float(x))
    ys = np.linspace(MARGIN, W - MARGIN, N_RX_Y)
    return np.array([[x, y] for x in xs for y in ys], dtype=float), np.array(xs), ys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/armC_demo/v2/doorway")
    a_ = ap.parse_args()
    out = Path(a_.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rx, xs, ys = dense_receivers()
    print("[domain] {:.1f} x {:.1f} m | divider x={:.2f} | src {} | dx {} fs {:.0f} n {}".format(
        L, W, DIV_X, SRC, DX, FS, N), flush=True)
    print("[mode]   sub-room (1,1) = {:.2f} Hz | df = {:.2f} Hz".format(MODE_F_HZ, FS / N))
    print("[rx]     {} x {} = {} receivers, spacing {:.4f} x {:.4f} m".format(
        N_RX_X, N_RX_Y, rx.shape[0], xs[1] - xs[0], ys[1] - ys[0]), flush=True)

    meta = {"L": L, "W": W, "div_x": DIV_X, "alpha": ALPHA, "src": list(SRC), "dx": DX,
            "fs": FS, "n": N, "df_hz": FS / N, "mode_f_hz": float(MODE_F_HZ),
            "mode": [1, 1], "mode_basis": "sub-room, L_sub={:.2f}".format(L_SUB),
            "band_hz": [BAND_LO, BAND_HI], "n_rx": int(rx.shape[0]),
            "rx_grid": [N_RX_X, N_RX_Y], "apertures": list(APERTURES),
            "published_ld_db": {str(k): v for k, v in PUBLISHED_LD_DB.items()},
            "source": "FDTD simulation (aaf.sim.fdtd_2d.simulate) -- NOT model output",
            "runs": []}

    print("\n=== pass 1: dense field maps (n={}, {} rx) ===".format(N, rx.shape[0]), flush=True)
    for a in APERTURES:
        t0 = time.time()
        res = F.simulate(L, W, [ALPHA] * 4, SRC, rx, dx=DX, fs=FS, n=N, c=C,
                         extra_walls=extra_walls_for(a))
        H = np.asarray(res["H_complex"])
        freqs = np.asarray(res["freqs"])
        b = int(np.argmin(np.abs(freqs - MODE_F_HZ)))
        band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
        # band-integrated level per receiver -- the literal "how much energy is here" view
        e_band = np.sqrt(np.sum(np.abs(H[:, band]) ** 2, axis=1))
        # mean |H| per receiver, so FT-B's amplitude estimator can be re-derived without re-running
        m_band = np.mean(np.abs(H[:, band]), axis=1)

        ra = rx[:, 0] < DIV_X
        np.savez_compressed(out / "a{:04.0f}.npz".format(1000 * a),
                            H_mode=H[:, b].astype(np.complex64), e_band=e_band, m_band=m_band,
                            rx=rx, xs=xs, ys=ys, a=a, f_mode=freqs[b], bin_mode=b)
        rec = {"a": a, "bin": b, "f_bin_hz": float(freqs[b]),
               "n": N, "n_rx": int(rx.shape[0]), "margin": MARGIN,
               "ld_db_amplitude_ftb_def": ld_amplitude(H, freqs, ra, ~ra),
               "ld_db_power": ld_power(H, freqs, ra, ~ra),
               "ld_db_published_ftb": PUBLISHED_LD_DB[a],
               "room_b_exactly_zero": bool(np.all(H[~ra][:, band] == 0.0)),
               "max_abs_room_b": float(np.max(np.abs(H[~ra][:, band]))),
               "loop_s": round(time.time() - t0, 1)}
        meta["runs"].append(rec)
        print("  a={:.1f}  bin {} ({:.2f} Hz)  LD_amp {:+.2f}  LD_pow {:+.2f}  "
              "roomB_zero={}  {:.0f}s".format(
                  a, b, freqs[b], rec["ld_db_amplitude_ftb_def"], rec["ld_db_power"],
                  rec["room_b_exactly_zero"], rec["loop_s"]), flush=True)

    # ---- pass 2: reproduce FT-B's published numbers on FT-B's exact protocol -------------
    # Cheap (192 rx), so the long record is affordable. If this reproduces the published
    # values, the solver and geometry are right and pass 1's offset is attributable to the
    # record length / receiver set / estimator rather than to a modelling error.
    g16 = ftb_grid16()
    print("\n=== pass 2: FT-B protocol reproduction (n={}, {} rx, margin {}) ===".format(
        FTB_N, g16.shape[0], FTB_MARGIN), flush=True)
    meta["ftb_reproduction"] = []
    for a in APERTURES:
        t0 = time.time()
        res = F.simulate(L, W, [ALPHA] * 4, SRC, g16, dx=DX, fs=FS, n=FTB_N, c=C,
                         extra_walls=extra_walls_for(a))
        H = np.asarray(res["H_complex"])
        freqs = np.asarray(res["freqs"])
        ra = g16[:, 0] < DIV_X
        got = ld_amplitude(H, freqs, ra, ~ra)
        pub = PUBLISHED_LD_DB[a]
        d = float("nan") if not np.isfinite(pub) else got - pub
        meta["ftb_reproduction"].append(
            {"a": a, "n": FTB_N, "n_rx": int(g16.shape[0]), "margin": FTB_MARGIN,
             "ld_db_amplitude_ftb_def": got, "ld_db_published_ftb": pub,
             "delta_vs_published_db": d, "loop_s": round(time.time() - t0, 1)})
        print("  a={:.1f}  LD_amp {:+.2f} dB  published {:+.2f}  delta {:+.3f} dB  {:.0f}s"
              .format(a, got, pub, d, time.time() - t0), flush=True)

    json.dump(meta, open(out / "doorway_meta.json", "w"), indent=1, default=float)
    print("-> {}".format(out / "doorway_meta.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
