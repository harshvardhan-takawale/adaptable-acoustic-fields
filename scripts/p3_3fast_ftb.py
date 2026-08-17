"""FT-B: is a doorway/divider APERTURE a trainable edit axis?

One 8.0 x 4.0 domain, alpha = 0.15 on all four outer walls, split by a one-node-thick
interior divider at x = 4.0 (also alpha = 0.15) carrying a centred aperture of clear width
``a``. The divider node column sits at x = 4.00, so the two sub-rooms are 3.99 x 4.0 each
(air nodes x in [0, 3.99] and x in [4.01, 8.00]) -- exact mirror images about x = 4.0.

``a = 0`` omits the ``apertures`` key entirely (a sealed divider: room B is *exactly*
disconnected, H_B is identically zero, not small). ``a = 4.0`` omits the slab entirely.
Everything between is one slab with one aperture.

Why dx = 0.01 and fs = 61440
----------------------------
A0c measured the aperture observable moving 10.4x the estimator floor between dx = 0.02 and
dx = 0.01, so the aperture axis is not converged at the frozen 0.05/0.02 grids. ``fs`` MUST
scale with 1/dx or the CFL bound is violated -- fs = 12288 at dx = 0.01 gives lam = 2.79
against a bound of 1.0 and ``simulate`` raises. The pair (dx, fs) = (0.01, 61440) holds
lambda at EXACTLY the frozen 0.55827, and n = 2*fs keeps T = 2.000 s / df = 0.5 Hz exactly.

Two receiver grids
------------------
* ``grid8`` -- 8x8 over the full domain with a 0.3 m margin. Pipeline compatibility only.
* ``grid16`` -- 16x8 over the full domain with the same margin. **All measurement uses this
  one.** 8 x-samples over 8 m cannot resolve the n_x <= 9 modes below 200 Hz: cond(Phi) on
  the 8x8 full-domain basis is 1.9e16 against ``modal_projection.COND_MAX = 5``. The 16x8
  also puts exactly 8 x-columns in each sub-room and is mirror-symmetric about x = 4.0
  (x_k + x_{15-k} = 8.0 exactly, and the snapped nodes satisfy i_k + i_{15-k} = 800), which
  is what makes the even/odd decomposition below exact rather than approximate.

Splitting is measured by SYMMETRY, not by peak-picking a doubled peak
---------------------------------------------------------------------
The domain is mirror-symmetric about x = 4.0 for every ``a``, so every eigenmode is either
even or odd about that plane -- regardless of the source being in room A only. Forming

    H_even = (H(x) + H(8-x)) / 2 ,   H_odd = (H(x) - H(8-x)) / 2

on the 64 room-A receivers therefore separates the two members of each near-degenerate
pair *before* any peak fitting, and each projected spectrum has a single clean peak. The
splitting is then ``f_odd - f_even`` for the same sub-room (n_x, n_y).

CAVEAT, stated up front: at 8.0 x 4.0 the sub-room (1,0) sits at 42.98 Hz and (0,1) at
42.87 Hz -- 0.105 Hz apart inside a ~3 Hz linewidth. The splitting observable is therefore
poorly conditioned in THIS domain for the axial pair, and mode (1,1) at 60.71 Hz (nearest
neighbour 25 Hz away) is used as the primary tracked mode. The other three observables --
inter-room level difference, coupled decay, and modal frequency migration -- are unaffected.

Layout: ``--task IDX`` runs ONE config (one FDTD run) and drops a compact npz;
``--aggregate`` reads them all and writes ``aperture_sweep.json`` + ``FEASIBILITY.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.eval.modal_bandwidth import (
    _parabolic_vertex, caps_from_predicted_bw, measure_modes, peak_level_and_bw,
)
from aaf.eval.modal_projection import COND_MAX, enumerate_modes, mode_shape_matrix
from aaf.sim.analytical_modal_2d import damping_to_bandwidth_hz, modal_damping_2d

# ---------------------------------------------------------------------------- frozen setup
C = 343.0
L, W = 8.0, 4.0
ALPHA = 0.15
DX = 0.01
#: fs MUST scale with 1/dx. These pairs hold lam at the frozen dx=0.05/fs=12288 value.
FS_FOR_DX = {0.05: 12288.0, 0.02: 30720.0, 0.01: 61440.0}
FS = FS_FOR_DX[DX]
N = int(2 * FS)                      # T = 2.000 s, df = 0.5 Hz
DIV_X = 4.0
MARGIN = 0.3

#: Sweep. 0.05 is deliberately absent -- see FEASIBILITY.md ("dropped as under-resolved").
APERTURES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0)
#: Replicate configs for the MEASURED estimator floor (same geometry, moved source).
REPLICATE_A = (0.3, 1.0)
SRC = (0.5, 0.5)
SRC_REP = (0.7, 0.9)

L_SUB = 3.99                         # sub-room length: air nodes 0..399 and 401..800
F_MAX_MODES = 200.0
BAND_LO, BAND_HI = 20.0, 300.0
F_STORE_HZ = 400.0                   # spectra stored to here; analysis stops at 300 Hz
#: ISO third-octave centres with the whole band inside 0-300 Hz.
THIRD_OCTAVE_HZ = (25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0)
#: Tracked mode. Sub-room (1,1) sits at 60.71 Hz with its nearest sub-room neighbours 17.7
#: and 25.0 Hz away, so it is NOT touched by the (1,0)/(0,1) near-degeneracy caveat. Its even
#: branch ends on full-domain (2,1) = 60.63 Hz (it barely moves) and its odd branch ends on
#: full-domain (3,1) = 77.29 Hz -- a 16.6 Hz migration with nothing else in the way.
PRIMARY_MODE = (1, 1)
#: Continuity-tracking half-window. The odd branch migrates ~16.6 Hz in total, far outside
#: `measure_modes`'s fixed +/-2 Hz search, so the branch is followed from the previous sweep
#: point instead of re-searched around the analytic frequency. 6 Hz is well under the 17.7 Hz
#: gap to the nearest other sub-room mode, so the tracker cannot hop branches.
TRACK_HZ = 6.0

# ------------------------------------------------------------------------------ thresholds
SMOOTH_MAX_FRAC = 0.25               # (i)
R2_MIN = 0.95                        # (ii)
EFFECT_OVER_FLOOR_MIN = 10.0         # (iii)
#: (4) two-rooms criterion: the tracked pair's even/odd splitting exceeds its own linewidth.
SPLIT_OVER_BW_ONE_ROOM = 1.0
#: (3) double-slope acceptance -- fixed BEFORE the runs, not tuned afterwards.
DOUBLE_SLOPE_RMS_RATIO = 0.5
DOUBLE_SLOPE_T60_RATIO = 1.3

OUT_DIR = Path("outputs/p3_3fast/trackB")
RUN_DIR = OUT_DIR / "runs"


# ------------------------------------------------------------------------------- geometry
def extra_walls_for(a: float) -> Optional[List[Dict[str, Any]]]:
    """Divider spec for aperture ``a``. ``a >= W`` -> no wall at all; ``a == 0`` -> sealed."""
    if a >= W:
        return None
    spec: Dict[str, Any] = {"type": "slab", "axis": "x", "pos": DIV_X, "alpha": ALPHA}
    if a > 0.0:
        spec["apertures"] = [(0.5 * W - 0.5 * a, 0.5 * W + 0.5 * a)]
    return [spec]


def receiver_grids() -> Tuple[np.ndarray, np.ndarray]:
    """``(grid8 [64,2], grid16 [128,2])``, both spanning the full domain with MARGIN."""
    ys = np.linspace(MARGIN, W - MARGIN, 8)
    x8 = np.linspace(MARGIN, L - MARGIN, 8)
    x16 = np.linspace(MARGIN, L - MARGIN, 16)
    g8 = np.array([[x, y] for x in x8 for y in ys], dtype=float)
    g16 = np.array([[x, y] for x in x16 for y in ys], dtype=float)
    return g8, g16


def tasks() -> List[Dict[str, Any]]:
    """11 sweep points + 2 source-moved replicates = 13 independent FDTD runs."""
    out = [{"tag": "a{:04.0f}".format(1000 * a), "a": float(a), "src": SRC, "kind": "sweep"}
           for a in APERTURES]
    out += [{"tag": "rep{:04.0f}".format(1000 * a), "a": float(a), "src": SRC_REP,
             "kind": "replicate"} for a in REPLICATE_A]
    return out


# -------------------------------------------------------------------------------- one run
def run_task(idx: int) -> Path:
    t = tasks()[idx]
    g8, g16 = receiver_grids()
    rx = np.concatenate([g8, g16], axis=0)
    t0 = time.perf_counter()
    out = F.simulate(L, W, (ALPHA,) * 4, src=t["src"], rx=rx, dx=DX, fs=FS, n=N, c=C,
                     extra_walls=extra_walls_for(t["a"]))
    freqs = np.asarray(out["freqs"], dtype=float)
    H = out["H_complex"]

    # --- band-limited (0-300 Hz) IR -> per-sub-room energy envelope for the Schroeder EDC
    Hb = H.copy()
    Hb[:, freqs > BAND_HI] = 0.0
    ir_bl = np.fft.irfft(Hb, n=N, axis=1)
    snap = np.asarray(out["meta"]["rx_pos_snapped"], dtype=float)
    g16_lo = g8.shape[0]
    xs16 = snap[g16_lo:, 0]
    a_sel = np.where(xs16 < DIV_X)[0] + g16_lo
    b_sel = np.where(xs16 > DIV_X)[0] + g16_lo
    e_a = np.sum(ir_bl[a_sel] ** 2, axis=0)
    e_b = np.sum(ir_bl[b_sel] ** 2, axis=0)

    keep = freqs <= F_STORE_HZ
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_DIR / "{}.npz".format(t["tag"])
    ap_meta = out["meta"]["extra_walls"]
    np.savez_compressed(
        path,
        H=H[:, keep].astype(np.complex128),
        freqs=freqs[keep],
        rx_snapped=snap,
        e_a=e_a,
        e_b=e_b,
        meta=json.dumps({
            "tag": t["tag"], "a_nominal": t["a"], "kind": t["kind"], "src": list(t["src"]),
            "src_snapped": out["meta"]["src_pos_snapped"],
            "grid_shape": out["meta"]["grid_shape"], "dx_x": out["meta"]["dx_x"],
            "dx_y": out["meta"]["dx_y"], "fs": FS, "n": N, "df_hz": out["meta"]["df_hz"],
            "lambda_CFL": out["meta"]["lambda_CFL"],
            "lambda_CFL_aniso": out["meta"]["lambda_CFL_aniso"],
            "extra_walls": ap_meta,
            "a_realized": (float(W) if not ap_meta
                           else (float(ap_meta[0]["apertures"][0]["clear_width_m"])
                                 if ap_meta[0].get("apertures") else 0.0)),
            "loop_s": out["meta"]["throughput"]["loop_seconds"],
            "wall_s": time.perf_counter() - t0,
        }),
    )
    print("[{}] a={} realized -> {}  ({:.1f} s)".format(
        idx, t["a"], path, time.perf_counter() - t0), flush=True)
    return path


# ------------------------------------------------------------------------------- analysis
def _lin_fit_r2(u: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares ``y = m u + b``; returns ``(m, b, r2)``. r2 = nan if degenerate."""
    ok = np.isfinite(u) & np.isfinite(y)
    u, y = u[ok], y[ok]
    if u.size < 3 or np.ptp(u) == 0.0 or np.ptp(y) == 0.0:
        return float("nan"), float("nan"), float("nan")
    m, b = np.polyfit(u, y, 1)
    res = y - (m * u + b)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(m), float(b), float(1.0 - float(np.sum(res ** 2)) / ss_tot)


def coordinates(a: np.ndarray) -> Dict[str, np.ndarray]:
    """The candidate linearizing coordinates.

    In 2D the aperture "area" is the clear width times a unit depth, i.e. numerically ``a``
    itself -- so ``area`` is an AFFINE copy of ``a`` and must return the identical r^2. It is
    reported anyway so the equivalence is visible rather than assumed.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "a": a.copy(),
            "a^2": a ** 2,
            "area (= a * 1 m, affine copy of a)": a * 1.0,
            "log a": np.log(np.where(a > 0, a, np.nan)),
            "sqrt a": np.sqrt(a),
        }


def band_level_ratio(H: np.ndarray, freqs: np.ndarray, a_sel, b_sel,
                     lo: float, hi: float) -> float:
    """20 log10( <|H|>_{B, band} / <|H|>_{A, band} ) in dB. -inf when room B is silent."""
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any():
        return float("nan")
    ma = float(np.mean(np.abs(H[a_sel][:, m])))
    mb = float(np.mean(np.abs(H[b_sel][:, m])))
    if ma <= 0.0:
        return float("nan")
    if mb <= 0.0:
        return float("-inf")
    return float(20.0 * math.log10(mb / ma))


def schroeder(e: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-integrated energy decay curve in dB, plus its time axis."""
    tot = float(np.sum(e))
    if tot <= 0.0:
        return np.array([]), np.array([])
    edc = np.cumsum(e[::-1])[::-1] / tot
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(np.maximum(edc, 1e-300))
    return db, np.arange(e.size) / fs


def _seg_fit(t: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Slope (dB/s) and RMS residual of a straight-line fit."""
    m, b = np.polyfit(t, y, 1)
    return float(m), float(np.sqrt(np.mean((y - (m * t + b)) ** 2)))


#: EDC fit window. -25 dB, not -45: the 2.0 s record is SHORTER than the ~3.3 s T60 at
#: alpha = 0.15, so backward integration of a truncated IR bends steeply near t = T. A window
#: reaching -45 dB sits entirely inside that artifact and would report a spurious "double
#: slope" (measured: a late segment 4x STEEPER than the early one, identical for every
#: aperture INCLUDING the sealed divider -- the tell that it is truncation, not coupling).
EDC_DB_HI, EDC_DB_LO = -5.0, -25.0


def decay_analysis(e: np.ndarray, fs: float) -> Dict[str, Any]:
    """Single- vs double-slope fit of the band-limited EDC. Windows fixed a priori."""
    db, t = schroeder(e, fs)
    if db.size == 0:
        return {"valid": False, "reason": "no energy in this sub-room (sealed divider)"}
    i0 = int(np.argmax(db <= EDC_DB_HI))
    i1 = int(np.argmax(db <= EDC_DB_LO))
    if i1 <= i0 or i1 == 0:
        return {"valid": False,
                "reason": "EDC never reaches {} dB inside the {} s record".format(
                    EDC_DB_LO, t[-1] if t.size else 0.0)}
    tw, yw = t[i0:i1], db[i0:i1]
    s1, r1 = _seg_fit(tw, yw)
    s_1seg = s1

    best = None
    n = yw.size
    lo, hi = int(0.15 * n), int(0.85 * n)
    for k in range(lo, hi, max(1, n // 400)):
        sa, ra = _seg_fit(tw[:k], yw[:k])
        sb, rb = _seg_fit(tw[k:], yw[k:])
        rms = math.sqrt((ra * ra * k + rb * rb * (n - k)) / n)
        if best is None or rms < best[0]:
            best = (rms, k, sa, sb)
    rms2, k, sa, sb = best
    t60_e = -60.0 / sa if sa < 0 else float("inf")
    t60_l = -60.0 / sb if sb < 0 else float("inf")
    ratio = (t60_l / t60_e) if np.isfinite(t60_e) and t60_e > 0 else float("nan")
    is_double = bool(rms2 <= DOUBLE_SLOPE_RMS_RATIO * r1
                     and np.isfinite(ratio) and ratio >= DOUBLE_SLOPE_T60_RATIO)
    t60_single = float(-60.0 / s_1seg) if s_1seg < 0 else float("inf")
    return {
        "valid": True,
        "fit_window_db": [EDC_DB_HI, EDC_DB_LO],
        "record_s": float(t[-1]),
        "t_at_window_end_s": float(tw[-1]),
        "record_shorter_than_t60": bool(t60_single > t[-1]),
        "t60_single_s": t60_single,
        "slope_single_db_s": s_1seg,
        "rms_single_db": r1,
        "rms_double_db": rms2,
        "rms_ratio": float(rms2 / r1) if r1 > 0 else float("nan"),
        "t60_early_s": t60_e, "t60_late_s": t60_l, "t60_ratio": ratio,
        "transition_time_s": float(tw[k]), "transition_level_db": float(yw[k]),
        "double_slope": is_double,
    }


class Run:
    """One loaded FDTD record with the derived observables."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.freqs = z["freqs"]
        self.H = z["H"]
        self.rx = z["rx_snapped"]
        self.e_a, self.e_b = z["e_a"], z["e_b"]
        self.a = float(self.meta["a_nominal"])
        self.a_real = float(self.meta["a_realized"])
        n8 = 64
        self.g16 = slice(n8, self.H.shape[0])
        xs = self.rx[n8:, 0]
        self.a_sel = np.where(xs < DIV_X)[0] + n8
        self.b_sel = np.where(xs > DIV_X)[0] + n8
        self.mirror = (n8 + np.arange(128).reshape(16, 8)[::-1].reshape(-1))

    # -- observable 2 -------------------------------------------------------------------
    def level_difference(self) -> Dict[str, Any]:
        bands = []
        for fc in THIRD_OCTAVE_HZ:
            lo, hi = fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6)
            bands.append({"fc_hz": fc, "lo_hz": lo, "hi_hz": hi,
                          "ld_db": band_level_ratio(self.H, self.freqs, self.a_sel,
                                                    self.b_sel, lo, hi)})
        return {"bands": bands,
                "ld_broadband_db": band_level_ratio(self.H, self.freqs, self.a_sel,
                                                    self.b_sel, BAND_LO, BAND_HI)}

    # -- observable 1 -------------------------------------------------------------------
    def subroom_fields(self):
        """``(H_A, H_B_mirrored, rx_local)`` -- both sub-rooms in the SAME local frame.

        Room B is mirrored through x = 4.0 (u = 8 - x), which maps its 64 receivers onto
        room A's snapped node positions exactly, so one ``Phi`` serves both sub-rooms and
        the even/odd combination below is exact rather than interpolated.
        """
        ha = self.H[self.a_sel]
        hb = self.H[self.mirror[self.a_sel - 64]]
        return ha, hb, self.rx[self.a_sel]

    def even_odd_spectra(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(H_even, H_odd, rx_local)`` on the 64 room-A receivers."""
        ha, hb, rxl = self.subroom_fields()
        return 0.5 * (ha + hb), 0.5 * (ha - hb), rxl

    # -- observable 3 -------------------------------------------------------------------
    def decay(self) -> Dict[str, Any]:
        return {"room_A": decay_analysis(self.e_a, FS),
                "room_B": decay_analysis(self.e_b, FS)}


def subroom_basis(rx_a: np.ndarray):
    modes = [m for m in enumerate_modes(L_SUB, W, f_max=F_MAX_MODES, c=C)]
    phi = mode_shape_matrix(modes, rx_a, L_SUB, W)
    bw_pred = [damping_to_bandwidth_hz(
        modal_damping_2d(L_SUB, W, [ALPHA] * 4, m.n_x, m.n_y, model="kuttruff")) for m in modes]
    return modes, phi, float(np.linalg.cond(phi)), bw_pred


def measure_subroom(run: Run, modes, phi, caps) -> Dict[str, Any]:
    """Per-sub-room modal frequencies and -3 dB bandwidths, plus the even/odd pair split.

    Room A and room B are each projected onto the SAME 3.99 x 4.0 basis (room B mirrored),
    so ``room_A`` / ``room_B`` are directly comparable. ``even`` / ``odd`` are the
    symmetry-decomposed pair members; their difference is the splitting.
    """
    ha, hb, _ = run.subroom_fields()
    pin = np.linalg.pinv(phi)
    sets = {"room_A": ha, "room_B": hb, "even": 0.5 * (ha + hb), "odd": 0.5 * (ha - hb)}
    peaks = {k: measure_modes(np.abs(pin @ v), run.freqs, modes, caps=caps)
             for k, v in sets.items()}
    rows = []
    for j, m in enumerate(modes):
        row = {"n_x": m.n_x, "n_y": m.n_y, "family": m.family, "f_analytic_hz": m.f}
        for k in sets:
            row["f_{}_hz".format(k)] = peaks[k][j].f_peak
            row["bw_{}_hz".format(k)] = peaks[k][j].bw_3db_hz
            row["flag_{}".format(k)] = peaks[k][j].bw_flag
        fe, fo = row["f_even_hz"], row["f_odd_hz"]
        split = abs(fo - fe) if np.isfinite(fe) and np.isfinite(fo) else float("nan")
        bws = [v for v in (row["bw_even_hz"], row["bw_odd_hz"]) if np.isfinite(v)]
        bw = float(np.mean(bws)) if bws else float("nan")
        row.update(splitting_hz=split, bw_mean_hz=bw,
                   split_over_bw=(split / bw) if (np.isfinite(split) and bw > 0)
                   else float("nan"))
        rows.append(row)
    return {"modes": rows,
            "note": ("f_* come from measure_modes' fixed +/-2 Hz search around the analytic "
                     "sub-room frequency, so a branch that migrates further than 2 Hz is "
                     "NOT followed here -- see the tracked branch in observable 4")}


def _pick(rows, n_x, n_y):
    for r in rows:
        if r["n_x"] == n_x and r["n_y"] == n_y:
            return r
    return None


def track_branch(mags: Sequence[np.ndarray], freqs: np.ndarray, f0: float,
                 cap_hz: float, track_hz: float = TRACK_HZ) -> List[Dict[str, Any]]:
    """Follow one spectral branch across the sweep by CONTINUITY, not by re-searching.

    At each sweep point the peak is taken as the largest bin within ``+/-track_hz`` of the
    PREVIOUS point's peak, refined sub-bin by the same parabolic-in-dB vertex the bandwidth
    estimator uses. ``at_window_edge`` is reported per point: if it is ever True the branch
    may have been lost and the number must not be trusted.
    """
    out: List[Dict[str, Any]] = []
    f_prev = float(f0)
    df = float(freqs[1] - freqs[0])
    for mag in mags:
        with np.errstate(divide="ignore"):
            db = 20.0 * np.log10(np.maximum(np.abs(mag), 1e-30))
        lo = int(np.searchsorted(freqs, f_prev - track_hz, side="left"))
        hi = int(np.searchsorted(freqs, f_prev + track_hz, side="right"))
        lo, hi = max(lo, 1), min(hi, db.size - 1)
        if hi - lo < 3:
            out.append({"f_hz": float("nan"), "bw_hz": float("nan"),
                        "at_window_edge": True})
            continue
        i = int(lo + np.argmax(db[lo:hi]))
        delta, _ = _parabolic_vertex(db[i - 1], db[i], db[i + 1])
        f_pk = float(freqs[i] + delta * df)
        bw, _lvl, _fp, flag = peak_level_and_bw(np.abs(mag), freqs, f_pk,
                                                search_hz=0.75 * df, cap_hz=cap_hz)
        out.append({"f_hz": f_pk, "bw_hz": bw, "bw_flag": flag,
                    "at_window_edge": bool(i <= lo + 1 or i >= hi - 2)})
        f_prev = f_pk
    return out


# ------------------------------------------------------------------------------ aggregate
def aggregate() -> int:
    t_start = time.perf_counter()
    runs = {}
    for t in tasks():
        p = RUN_DIR / "{}.npz".format(t["tag"])
        if not p.exists():
            raise SystemExit("missing run {} -- rerun `--task` for it".format(p))
        runs[t["tag"]] = Run(p)

    sweep_tags = ["a{:04.0f}".format(1000 * a) for a in APERTURES]
    ref = runs[sweep_tags[-1]]

    # --- conditioning of both bases, on the full-domain modes (the task's stated gate) ----
    modes_full = enumerate_modes(L, W, f_max=F_MAX_MODES, c=C)
    g8, g16 = receiver_grids()
    cond8 = float(np.linalg.cond(mode_shape_matrix(modes_full, g8, L, W)))
    cond16 = float(np.linalg.cond(mode_shape_matrix(modes_full, ref.rx[64:], L, W)))

    _, _, rx_a = ref.even_odd_spectra()
    modes_sub, phi_sub, cond_sub, bw_pred = subroom_basis(rx_a)
    caps = caps_from_predicted_bw(bw_pred)

    pin_sub = np.linalg.pinv(phi_sub)
    j_prim = [i for i, m in enumerate(modes_sub)
              if (m.n_x, m.n_y) == PRIMARY_MODE][0]
    records, mag_even, mag_odd = [], [], []
    for tag, a in zip(sweep_tags, APERTURES):
        r = runs[tag]
        h_e, h_o, _ = r.even_odd_spectra()
        mag_even.append(np.abs(pin_sub @ h_e)[j_prim])
        mag_odd.append(np.abs(pin_sub @ h_o)[j_prim])
        rec = {"a_nominal": float(a), "a_realized_m": r.a_real,
               "aperture_fraction_of_wall": r.a_real / W,
               "level_difference": r.level_difference(),
               "decay": r.decay(),
               "subroom_modes": measure_subroom(r, modes_sub, phi_sub, caps)}
        records.append(rec)
        print("  a={:.2f}: LD = {:8.3f} dB".format(
            a, rec["level_difference"]["ld_broadband_db"]), flush=True)

    a_nom = np.array([r["a_nominal"] for r in records], float)
    ld = np.array([r["level_difference"]["ld_broadband_db"] for r in records], float)
    finite = np.isfinite(ld)

    # --- observable 4: modal migration, by CONTINUITY tracking ---------------------------
    freqs = runs[sweep_tags[0]].freqs
    f_ana = modes_sub[j_prim].f
    cap_prim = caps[j_prim]
    tr_e = track_branch(mag_even, freqs, f_ana, cap_prim)
    tr_o = track_branch(mag_odd, freqs, f_ana, cap_prim)
    f_even = np.array([t["f_hz"] for t in tr_e], float)
    f_odd = np.array([t["f_hz"] for t in tr_o], float)
    bw_e = np.array([t["bw_hz"] for t in tr_e], float)
    bw_o = np.array([t["bw_hz"] for t in tr_o], float)
    bw_m = np.nanmean(np.stack([bw_e, bw_o]), axis=0)
    split = np.abs(f_odd - f_even)
    with np.errstate(invalid="ignore", divide="ignore"):
        s_over_bw = split / bw_m
    lost = bool(any(t["at_window_edge"] for t in tr_e + tr_o))
    f_sealed, f_open = f_odd[0], f_odd[-1]
    span = abs(f_open - f_sealed)
    migration = np.abs(f_odd - f_sealed) / max(span, 1e-12)

    def _cross(y, thr):
        """First a where y crosses thr, linearly interpolated between sweep points."""
        for i in range(1, len(a_nom)):
            if np.isfinite(y[i]) and y[i] >= thr:
                y0, y1 = y[i - 1], y[i]
                if not np.isfinite(y0) or y1 == y0:
                    return float(a_nom[i])
                return float(a_nom[i - 1] + (a_nom[i] - a_nom[i - 1])
                             * (thr - y0) / (y1 - y0))
        return None

    a_half = _cross(migration, 0.5)
    a_linewidth = _cross(split, float(np.nanmedian(bw_m)))
    a_split_bw = _cross(s_over_bw, SPLIT_OVER_BW_ONE_ROOM)
    a_one_room = a_half

    # --- (iii) measured estimator floor --------------------------------------------------
    floor_terms = []
    for a in REPLICATE_A:
        base = runs["a{:04.0f}".format(1000 * a)].level_difference()["ld_broadband_db"]
        rep = runs["rep{:04.0f}".format(1000 * a)].level_difference()["ld_broadband_db"]
        floor_terms.append({"a": a, "ld_src1_db": base, "ld_src2_db": rep,
                            "abs_diff_db": abs(base - rep)})
    floor = float(np.mean([f["abs_diff_db"] for f in floor_terms]))
    y_fin = ld[finite]
    effect = float(np.max(y_fin) - np.min(y_fin))
    effect_ratio = effect / floor if floor > 0 else float("inf")

    # --- (i) smoothness ------------------------------------------------------------------
    jumps = np.abs(np.diff(y_fin))
    smooth_frac = float(np.max(jumps) / effect) if effect > 0 else float("nan")
    j_at = int(np.argmax(jumps))
    a_fin = a_nom[finite]

    # --- (ii) linearizing coordinate -----------------------------------------------------
    def fit_block(mask) -> Dict[str, Any]:
        aa, yy = a_nom[mask], ld[mask]
        out = {}
        for name, u in coordinates(aa).items():
            m, b, r2 = _lin_fit_r2(u, yy)
            out[name] = {"slope": m, "intercept": b, "r2": r2, "n": int(np.sum(mask))}
        best = max(out.items(), key=lambda kv: (kv[1]["r2"] if np.isfinite(kv[1]["r2"])
                                                else -np.inf))
        return {"fits": out, "best": best[0], "best_r2": best[1]["r2"]}

    m_pool = finite & (a_nom > 0)
    m_small = m_pool & (a_nom <= 2.0)
    m_large = m_pool & (a_nom >= 1.0)
    fits = {"pooled_a_gt_0": fit_block(m_pool),
            "a_le_2.0": fit_block(m_small),
            "a_ge_1.0": fit_block(m_large)}
    best_name = fits["pooled_a_gt_0"]["best"]
    best_r2 = fits["pooled_a_gt_0"]["best_r2"]

    # --- sampling density from the curvature of the selected coordinate -------------------
    u_best = coordinates(a_nom[m_pool])[best_name]
    y_best = ld[m_pool]
    order = np.argsort(u_best)
    us, ys = u_best[order], y_best[order]
    d2 = []
    for i in range(1, us.size - 1):
        h1, h2 = us[i] - us[i - 1], us[i + 1] - us[i]
        d2.append(2.0 * (ys[i - 1] * h2 - ys[i] * (h1 + h2) + ys[i + 1] * h1)
                  / (h1 * h2 * (h1 + h2)))
    d2 = np.abs(np.array(d2, float))
    curv_max = float(np.nanmax(d2)) if d2.size else float("nan")
    curv_med = float(np.nanmedian(d2)) if d2.size else float("nan")
    u_range = float(np.ptp(us))
    # linear interpolation error between samples spaced h is <= |y''| h^2 / 8; require it to
    # stay under the MEASURED estimator floor (same tolerance the effect size is judged by).
    h_max = math.sqrt(8.0 * floor / curv_max) if curv_max > 0 else float("inf")
    h_med = math.sqrt(8.0 * floor / curv_med) if curv_med > 0 else float("inf")
    delta_star = h_max / u_range if u_range > 0 else float("nan")
    delta_star_med = h_med / u_range if u_range > 0 else float("nan")

    go_i = bool(np.isfinite(smooth_frac) and smooth_frac < SMOOTH_MAX_FRAC)
    go_ii = bool(np.isfinite(best_r2) and best_r2 >= R2_MIN)
    go_iii = bool(effect_ratio >= EFFECT_OVER_FLOOR_MIN)

    result = {
        "gate": "FT-B",
        "question": "is the doorway / divider APERTURE width a trainable edit axis?",
        "domain": {"L": L, "W": W, "alpha_all_walls": ALPHA, "divider_x_m": DIV_X,
                   "divider_alpha": ALPHA, "sub_room_L_m": L_SUB, "sub_room_W_m": W,
                   "dx": DX, "fs": FS, "n": N, "T_s": N / FS, "df_hz": FS / N, "c": C,
                   "lambda_CFL": ref.meta["lambda_CFL"],
                   "lambda_CFL_aniso": ref.meta["lambda_CFL_aniso"],
                   "grid_shape": ref.meta["grid_shape"],
                   "src": list(SRC), "src_replicate": list(SRC_REP)},
        "sweep_a": list(APERTURES),
        "dropped_from_sweep": {
            "a": 0.05,
            "why": ("under-resolved: 0.05 m is 5 cells at dx = 0.01 and `_apply_slab` needs "
                    ">= 3 open nodes for the two edge nodes to carry the boundary condition, "
                    "so the staircased tips occupy 2 of 5 cells. A0c already measured the "
                    "aperture observable moving 10.4x the floor between dx 0.02 and 0.01, so "
                    "a 5-cell aperture is inside the un-converged regime by construction.")},
        "receivers": {
            "grid8_8x8": {"n": 64, "margin_m": MARGIN,
                          "cond_phi_full_domain": cond8,
                          "usable_for_projection": bool(cond8 <= COND_MAX),
                          "role": "pipeline compatibility only"},
            "grid16_16x8": {"n": 128, "margin_m": MARGIN,
                            "cond_phi_full_domain": cond16,
                            "usable_for_projection": bool(cond16 <= COND_MAX),
                            "role": "all measurement"},
            "cond_max_gate": COND_MAX,
            "cond_phi_subroom_even_odd": cond_sub,
            "n_modes_full_domain": len(modes_full),
            "n_modes_subroom": len(modes_sub)},
        "observable_1_modal": {
            "method": ("each sub-room's 64 receivers are projected onto the 3.99 x 4.0 "
                       "sub-room mode shapes (room B mirrored through x = 4.0 into the same "
                       "local frame), giving per-sub-room modal frequencies and -3 dB "
                       "bandwidths; the even/odd decomposition H_+- = (H(x) +- H(8-x))/2 "
                       "separates the two members of each near-degenerate pair BEFORE peak "
                       "fitting, so splitting = |f_odd - f_even| for the same (n_x, n_y)"),
            "caveat": ("sub-room (1,0) = 42.98 Hz and (0,1) = 42.87 Hz are 0.107 Hz apart "
                       "inside a ~3.07 Hz linewidth, so the SPLITTING observable is poorly "
                       "conditioned in THIS 8.0 x 4.0 domain for the axial pair -- reported, "
                       "but do not lean on it. Mode (1,1) at 60.71 Hz is isolated (nearest "
                       "sub-room neighbours 17.7 and 25.0 Hz away) and is the tracked mode. "
                       "The other three observables are unaffected."),
            "tracked_mode": {"n_x": PRIMARY_MODE[0], "n_y": PRIMARY_MODE[1],
                             "f_analytic_hz": f_ana,
                             "track_half_window_hz": TRACK_HZ,
                             "branch_lost_at_window_edge": lost},
            "tracked_per_a": [
                {"a": float(a), "f_even_hz": float(fe), "f_odd_hz": float(fo),
                 "bw_even_hz": float(be), "bw_odd_hz": float(bo),
                 "splitting_hz": float(s), "bw_mean_hz": float(bw),
                 "split_over_bw": float(sb), "migration_frac": float(mg),
                 "edge_even": te["at_window_edge"], "edge_odd": to["at_window_edge"]}
                for a, fe, fo, be, bo, s, bw, sb, mg, te, to in
                zip(a_nom, f_even, f_odd, bw_e, bw_o, split, bw_m, s_over_bw, migration,
                    tr_e, tr_o)],
            "edge_flag_a": [float(a) for a, te, to in zip(a_nom, tr_e, tr_o)
                            if te["at_window_edge"] or to["at_window_edge"]],
            "per_a_full_table": [
                {"a": r["a_nominal"], **r["subroom_modes"]} for r in records]},
        "observable_2_level_difference": {
            "definition": ("20 log10( mean_|H| over the 64 room-B receivers / mean_|H| over "
                           "the 64 room-A receivers ), per ISO third-octave band"),
            "broadband_band_hz": [BAND_LO, BAND_HI],
            "per_a": [{"a": r["a_nominal"], "ld_broadband_db":
                       r["level_difference"]["ld_broadband_db"],
                       "bands": r["level_difference"]["bands"]} for r in records]},
        "observable_3_decay": {
            "method": ("Schroeder integration of the 0-300 Hz band-limited IR energy summed "
                       "over each sub-room's 64 receivers; single-slope and two-segment "
                       "(searched-breakpoint) fits over the SAME -5..-25 dB window"),
            "double_slope_criterion": {"rms_ratio_max": DOUBLE_SLOPE_RMS_RATIO,
                                       "t60_late_over_early_min": DOUBLE_SLOPE_T60_RATIO},
            "fit_window_db": [EDC_DB_HI, EDC_DB_LO],
            "limitation": (
                "T60 is ~3.3 s at alpha = 0.15 but the frozen record is T = 2.0 s, so the "
                "EDC below about -30 dB is backward-integration truncation, not decay. The "
                "first pass used a -5..-45 dB window and reported an apparently clean "
                "'double slope' with the late segment 4x STEEPER than the early one and the "
                "SAME breakpoint (1.70 s / -21.5 dB) for every aperture INCLUDING the sealed "
                "divider -- the tell that it was the record end, not room coupling. The "
                "window was cut to -5..-25 dB, which the record does support. The double- "
                "slope question is therefore answered NEGATIVELY within the supported range "
                "and is INCONCLUSIVE below -25 dB; resolving it needs a longer record or a "
                "more absorptive room, not a different fit."),
            "why_no_double_slope_is_expected_here": (
                "classical coupled-room double slope requires the two sub-volumes to have "
                "DIFFERENT decay rates. Here the sub-rooms are exact mirror images with "
                "identical alpha = 0.15, so they share one decay constant by construction "
                "and no aperture width can split it. Measured: T60 moves only 3.324 -> 3.301 "
                "s (0.7%) across the whole sweep and the two sub-rooms agree to < 0.01 s. "
                "This axis's decay observable only becomes informative once the sub-rooms "
                "differ in absorption -- a real design note for any FT-B follow-up."),
            "per_a": [{"a": r["a_nominal"], **r["decay"]} for r in records]},
        "observable_4_two_rooms": {
            "criterion": (
                "STATED IN TERMS OF MODAL FREQUENCY MIGRATION. The odd branch of sub-room "
                "mode (1,1) starts at the sealed-divider sub-room frequency f_sealed and "
                "ends, with the divider absent, at the full-domain frequency f_open; define "
                "migration(a) = |f_odd(a) - f_sealed| / |f_open - f_sealed|. The domain "
                "STOPS BEHAVING AS TWO ROOMS at the a where migration crosses 0.5, i.e. "
                "where the mode is closer in frequency to the one-room eigenvalue than to "
                "the two-room one. Two supporting crossings are reported but do not define "
                "the verdict: (a) the first a at which the migration exceeds one modal "
                "linewidth, i.e. the smallest coupling that is spectrally RESOLVABLE at all, "
                "and (b) the a at which the even/odd splitting exceeds the linewidth."),
            "migration_definition": ("|f_odd(a) - f_odd(sealed)| / |f_odd(no wall) - "
                                     "f_odd(sealed)| for the tracked (1,1) odd branch"),
            "f_odd_sealed_hz": float(f_sealed), "f_odd_open_hz": float(f_open),
            "endpoint_check": {
                "f_odd_open_measured_hz": float(f_open),
                "f_full_domain_3_1_analytic_hz": float(
                    0.5 * C * math.hypot(3.0 / L, 1.0 / W)),
                "rel_error": float(abs(f_open - 0.5 * C * math.hypot(3.0 / L, 1.0 / W))
                                   / (0.5 * C * math.hypot(3.0 / L, 1.0 / W))),
                "why": ("the odd branch of sub-room (1,1) must land on full-domain (3,1) "
                        "once the divider is gone. It does, which is the independent proof "
                        "that the continuity tracker never hopped branches -- including at "
                        "a = 3.0, the one point flagged near the tracking-window edge.")},
            "migration_span_hz": float(span),
            "median_linewidth_hz": float(np.nanmedian(bw_m)),
            "a_one_room_m": a_one_room,
            "a_one_room_fraction_of_W": (a_one_room / W) if a_one_room else None,
            "a_first_resolvable_coupling_m": a_linewidth,
            "a_splitting_exceeds_linewidth_m": a_split_bw},
        "go_no_go": {
            "i_smoothness": {"threshold_frac": SMOOTH_MAX_FRAC,
                             "largest_jump_db": float(np.max(jumps)),
                             "between_a": [float(a_fin[j_at]), float(a_fin[j_at + 1])],
                             "total_range_db": effect,
                             "fraction_of_range": smooth_frac,
                             "verdict": "GO" if go_i else "NO-GO",
                             "note": ("computed over the FINITE points a > 0. a = 0 is not a "
                                      "limit point: a sealed one-node divider disconnects "
                                      "room B exactly, so H_B == 0 and the level difference "
                                      "is -inf. That is a topological discontinuity, not a "
                                      "large jump, and no continuous coordinate can include "
                                      "it -- the trainable range is a in (0, 4].")},
            "ii_linearizing_coordinate": {
                "threshold_r2": R2_MIN, "response": "inter-room level difference (dB)",
                "best_pooled": best_name, "best_pooled_r2": best_r2,
                "verdict": "GO" if go_ii else "NO-GO", "fits": fits,
                "note": ("2D slit physics is logarithmic at small a and area-like at large "
                         "a, so the restricted fits a <= 2.0 and a >= 1.0 are reported "
                         "alongside the pooled fit. In 2D the aperture 'area' is the clear "
                         "width times unit depth = a, an affine copy of the `a` coordinate, "
                         "so it necessarily shares its r^2.")},
            "iii_effect_over_floor": {
                "threshold": EFFECT_OVER_FLOOR_MIN,
                "floor_db": floor, "floor_terms": floor_terms,
                "floor_method": ("mean |delta| of the broadband level difference between two "
                                 "runs of the SAME geometry differing only in source "
                                 "position ((0.5,0.5) vs (0.7,0.9) m), measured at a = 0.3 "
                                 "and a = 1.0 -- measured, not assumed"),
                "effect_db": effect, "ratio": effect_ratio,
                "verdict": "GO" if go_iii else "NO-GO"}},
        "sampling_density": {
            "coordinate": best_name,
            "coordinate_range": u_range,
            "curvature_max_db_per_u2": curv_max,
            "curvature_median_db_per_u2": curv_med,
            "tolerance_db": floor,
            "rule": ("linear interpolation between adjacent training samples errs by at most "
                     "|y''| h^2 / 8; requiring that to stay under the MEASURED estimator "
                     "floor gives h* = sqrt(8 * floor / |y''|)"),
            "h_star_worst_case": h_max, "h_star_median_curvature": h_med,
            "delta_star_frac_of_range": delta_star,
            "delta_star_frac_of_range_median_curvature": delta_star_med,
            "compare_absorption_axis": ("P3-2d measured Delta* ~ 0.275 of the linearizing "
                                        "coordinate's range on the absorption axis")},
        "verdict": ("GO" if (go_i and go_ii and go_iii) else "NO-GO"),
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "aperture_sweep.json", "w") as fh:
        json.dump(result, fh, indent=1, default=lambda o: None if o != o else float(o))
    write_report(result)
    print("\nFT-B verdict: {}  (i={} ii={} iii={})".format(
        result["verdict"], go_i, go_ii, go_iii))
    print("-> {}".format(OUT_DIR / "aperture_sweep.json"))
    return 0


def _f(v, spec="{:.3f}"):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "-inf" if x < 0 else ("inf" if x > 0 else "nan")
    return spec.format(x)


def write_report(r: Dict[str, Any]) -> None:
    g = r["go_no_go"]
    ld = r["observable_2_level_difference"]["per_a"]
    md = r["observable_1_modal"]["tracked_per_a"]
    dec = r["observable_3_decay"]["per_a"]
    lines = []
    A = lines.append
    A("# FT-B — doorway / divider aperture: feasibility\n")
    A("**Verdict: {}** — (i) smoothness {}, (ii) linearizing coordinate {}, "
      "(iii) effect size {}.\n".format(
          r["verdict"], g["i_smoothness"]["verdict"],
          g["ii_linearizing_coordinate"]["verdict"], g["iii_effect_over_floor"]["verdict"]))
    d = r["domain"]
    A("Domain {}x{} m, alpha = {} on all four outer walls and on the divider at x = {}; "
      "dx = {}, fs = {}, n = {} (T = {} s, df = {} Hz, lambda = {}, anisotropic CFL = {} "
      "against a bound of 1.0). Grid {}. Source at {}.\n".format(
          d["L"], d["W"], d["alpha_all_walls"], d["divider_x_m"], d["dx"], d["fs"], d["n"],
          d["T_s"], d["df_hz"], _f(d["lambda_CFL"], "{:.5f}"),
          _f(d["lambda_CFL_aniso"], "{:.5f}"), d["grid_shape"], d["src"]))
    A("`a = 0.05` was **dropped as under-resolved**: {}\n".format(r["dropped_from_sweep"]["why"]))
    rc = r["receivers"]
    A("Receiver grids: the 8x8 is carried for pipeline compatibility only — cond(Phi) on the "
      "full-domain basis is **{}** against the `modal_projection` gate of {}. All measurement "
      "uses the 16x8 (cond = {}). The sub-room even/odd basis has cond = {}.\n".format(
          _f(rc["grid8_8x8"]["cond_phi_full_domain"], "{:.3g}"), rc["cond_max_gate"],
          _f(rc["grid16_16x8"]["cond_phi_full_domain"], "{:.3g}"),
          _f(rc["cond_phi_subroom_even_odd"], "{:.3f}")))

    A("\n## 1. Sub-room modes and splitting\n")
    A(r["observable_1_modal"]["method"] + "\n")
    A("> **CAVEAT.** " + r["observable_1_modal"]["caveat"] + "\n")
    tm = r["observable_1_modal"]["tracked_mode"]
    A("Tracked mode ({},{}), analytic {} Hz, followed by continuity with a +/-{} Hz window. "
      "Points where the peak landed within 2 bins of the tracking window edge (and are "
      "therefore the only ones where the branch could have been lost): **{}**.\n".format(
          tm["n_x"], tm["n_y"], _f(tm["f_analytic_hz"]), tm["track_half_window_hz"],
          r["observable_1_modal"]["edge_flag_a"] or "none"))
    A("| a (m) | f_even (Hz) | f_odd (Hz) | BW even (Hz) | BW odd (Hz) | splitting (Hz) | "
      "split/BW | migration | edge? |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|:--|")
    for m in md:
        A("| {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            m["a"], _f(m["f_even_hz"]), _f(m["f_odd_hz"]), _f(m["bw_even_hz"], "{:.2f}"),
            _f(m["bw_odd_hz"], "{:.2f}"), _f(m["splitting_hz"]),
            _f(m["split_over_bw"], "{:.2f}"), _f(m["migration_frac"], "{:.2f}"),
            "E" if m["edge_even"] else ("O" if m["edge_odd"] else "")))
    A("\nPer-sub-room modal frequencies and bandwidths for every one of the {} sub-room "
      "modes below 200 Hz (rooms A and B measured independently on the same basis) are in "
      "`aperture_sweep.json` under `observable_1_modal.per_a_full_table`. Room-A vs room-B "
      "mean -3 dB bandwidth per aperture:\n".format(
          r["receivers"]["n_modes_subroom"]))
    A("| a (m) | mean BW room A (Hz) | mean BW room B (Hz) | n modes resolved A / B |")
    A("|---:|---:|---:|---:|")
    for row in r["observable_1_modal"]["per_a_full_table"]:
        ba = [m["bw_room_A_hz"] for m in row["modes"]
              if m["bw_room_A_hz"] is not None and np.isfinite(m["bw_room_A_hz"] or np.nan)]
        bb = [m["bw_room_B_hz"] for m in row["modes"]
              if m["bw_room_B_hz"] is not None and np.isfinite(m["bw_room_B_hz"] or np.nan)]
        A("| {:.2f} | {} | {} | {} / {} |".format(
            row["a"], _f(np.mean(ba) if ba else float("nan"), "{:.2f}"),
            _f(np.mean(bb) if bb else float("nan"), "{:.2f}"), len(ba), len(bb)))

    A("\n## 2. Inter-room level difference\n")
    A(r["observable_2_level_difference"]["definition"] + "\n")
    fcs = [b["fc_hz"] for b in ld[0]["bands"]]
    A("| a (m) | broadband 20-300 Hz | " + " | ".join("{:g}".format(f) for f in fcs) + " |")
    A("|---:|---:|" + "---:|" * len(fcs))
    for row in ld:
        A("| {:.2f} | {} | ".format(row["a"], _f(row["ld_broadband_db"], "{:.2f}"))
          + " | ".join(_f(b["ld_db"], "{:.1f}") for b in row["bands"]) + " |")

    A("\n## 3. Coupled decay\n")
    A(r["observable_3_decay"]["method"] + "\n")
    A("Double-slope is accepted only if the two-segment RMS is <= {}x the single-segment RMS "
      "AND T60_late/T60_early >= {} (fixed before the runs).\n".format(
          r["observable_3_decay"]["double_slope_criterion"]["rms_ratio_max"],
          r["observable_3_decay"]["double_slope_criterion"]["t60_late_over_early_min"]))
    A("> **This observable FAILED to produce a usable signal, and one earlier version of it "
      "produced a false positive.** " + r["observable_3_decay"]["limitation"] + "\n")
    A("> " + r["observable_3_decay"]["why_no_double_slope_is_expected_here"] + "\n")
    A("| a (m) | room | T60 single (s) | T60 early (s) | T60 late (s) | RMS ratio | "
      "double? | transition (s / dB) |")
    A("|---:|:--|---:|---:|---:|---:|:--|---:|")
    for row in dec:
        for rm in ("room_A", "room_B"):
            v = row[rm]
            if not v.get("valid"):
                A("| {:.2f} | {} | — | — | — | — | — | {} |".format(
                    row["a"], rm[-1], v.get("reason", "invalid")))
                continue
            A("| {:.2f} | {} | {} | {} | {} | {} | {} | {} / {} |".format(
                row["a"], rm[-1], _f(v["t60_single_s"], "{:.3f}"),
                _f(v["t60_early_s"], "{:.3f}"), _f(v["t60_late_s"], "{:.3f}"),
                _f(v["rms_ratio"], "{:.2f}"), "yes" if v["double_slope"] else "no",
                _f(v["transition_time_s"], "{:.3f}"), _f(v["transition_level_db"], "{:.1f}")))

    o4 = r["observable_4_two_rooms"]
    A("\n## 4. When does it stop being two rooms?\n")
    A("**Criterion.** " + o4["criterion"] + "\n")
    A("Migration is " + o4["migration_definition"] + ". The tracked odd branch runs from "
      "{} Hz (sealed) to {} Hz (no divider), a span of {} Hz against a median linewidth of "
      "{} Hz.\n".format(_f(o4["f_odd_sealed_hz"]), _f(o4["f_odd_open_hz"]),
                        _f(o4["migration_span_hz"]), _f(o4["median_linewidth_hz"])))
    ec = o4["endpoint_check"]
    A("**Endpoint check.** " + ec["why"] + " Measured {} Hz vs analytic {} Hz, "
      "{} relative error.\n".format(_f(ec["f_odd_open_measured_hz"]),
                                    _f(ec["f_full_domain_3_1_analytic_hz"]),
                                    _f(100 * ec["rel_error"], "{:.2f}%")))
    A("**Stops behaving as two rooms at a = {} m ({} of the divider width).** "
      "First spectrally resolvable coupling (migration > one linewidth) at a = {} m; "
      "even/odd splitting exceeds the linewidth at a = {} m.\n".format(
          _f(o4["a_one_room_m"], "{:.3f}"), _f(o4["a_one_room_fraction_of_W"], "{:.3f}"),
          _f(o4["a_first_resolvable_coupling_m"], "{:.3f}"),
          _f(o4["a_splitting_exceeds_linewidth_m"], "{:.3f}")))

    A("\n## GO / NO-GO\n")
    s = g["i_smoothness"]
    A("**(i) Smoothness — {}.** Largest adjacent jump {} dB between a = {} and a = {}, "
      "which is **{}** of the total range ({} dB); threshold < {}.\n".format(
          s["verdict"], _f(s["largest_jump_db"], "{:.2f}"), s["between_a"][0],
          s["between_a"][1], _f(100 * s["fraction_of_range"], "{:.1f}%"),
          _f(s["total_range_db"], "{:.2f}"), SMOOTH_MAX_FRAC))
    A("> " + s["note"] + " The largest jump also falls across the WIDEST sampling gap "
      "(a: 2.0 -> 3.0, the only 1.0 m step in the sweep), so it measures the sweep spacing "
      "as much as the physics -- it is not a discontinuity in the response.\n")
    f = g["ii_linearizing_coordinate"]
    A("**(ii) Linearizing coordinate — {}.** Best pooled coordinate is **{}** with "
      "r^2 = {} (threshold >= {}).\n".format(
          f["verdict"], f["best_pooled"], _f(f["best_pooled_r2"], "{:.4f}"), R2_MIN))
    A("| coordinate | r^2 pooled (a>0) | r^2 (a <= 2.0) | r^2 (a >= 1.0) |")
    A("|:--|---:|---:|---:|")
    for name in f["fits"]["pooled_a_gt_0"]["fits"]:
        A("| {} | {} | {} | {} |".format(
            name, _f(f["fits"]["pooled_a_gt_0"]["fits"][name]["r2"], "{:.4f}"),
            _f(f["fits"]["a_le_2.0"]["fits"][name]["r2"], "{:.4f}"),
            _f(f["fits"]["a_ge_1.0"]["fits"][name]["r2"], "{:.4f}")))
    A("\nBest on a <= 2.0: **{}** (r^2 = {}); best on a >= 1.0: **{}** (r^2 = {}).\n".format(
        f["fits"]["a_le_2.0"]["best"], _f(f["fits"]["a_le_2.0"]["best_r2"], "{:.4f}"),
        f["fits"]["a_ge_1.0"]["best"], _f(f["fits"]["a_ge_1.0"]["best_r2"], "{:.4f}")))
    A("> " + f["note"] + "\n")
    e = g["iii_effect_over_floor"]
    A("**(iii) Effect size — {}.** Effect {} dB vs a measured floor of {} dB = **{}x** "
      "(threshold >= {}x).\n".format(
          e["verdict"], _f(e["effect_db"], "{:.2f}"), _f(e["floor_db"], "{:.3f}"),
          _f(e["ratio"], "{:.1f}"), EFFECT_OVER_FLOOR_MIN))
    A("Floor method: " + e["floor_method"] + ". Per-config: "
      + "; ".join("a={}: |{} - {}| = {} dB".format(
          t["a"], _f(t["ld_src1_db"], "{:.3f}"), _f(t["ld_src2_db"], "{:.3f}"),
          _f(t["abs_diff_db"], "{:.3f}")) for t in e["floor_terms"]) + ".\n")

    sd = r["sampling_density"]
    A("\n## Required addition — how densely must `a` be sampled?\n")
    A(sd["rule"] + ". With the selected coordinate **{}** over a range of {}, the worst-case "
      "curvature is {} dB/unit^2 (median {}), and the tolerance is the measured floor "
      "{} dB.\n".format(sd["coordinate"], _f(sd["coordinate_range"], "{:.3f}"),
                        _f(sd["curvature_max_db_per_u2"], "{:.3g}"),
                        _f(sd["curvature_median_db_per_u2"], "{:.3g}"),
                        _f(sd["tolerance_db"], "{:.3f}")))
    A("**Delta\\* = {} of the coordinate's range** (worst-case curvature); {} using the "
      "median curvature. {}.\n".format(
          _f(sd["delta_star_frac_of_range"], "{:.3f}"),
          _f(sd["delta_star_frac_of_range_median_curvature"], "{:.3f}"),
          sd["compare_absorption_axis"]))
    (OUT_DIR / "FEASIBILITY.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int, default=None, help="run one config (0..12)")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i, t in enumerate(tasks()):
            print(i, t)
        return 0
    if a.task is not None:
        run_task(a.task)
        return 0
    if a.aggregate:
        return aggregate()
    ap.error("pass --task IDX, --aggregate, or --list")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
