"""P3-3-FAST Track 2b evaluation: does the field track coupling through UNSEEN apertures?

The Track B model (``outputs/p3_3fast/p3_3fast_trackB``) is conditioned on
``(L, W, x0, sqrt(a))`` -- 55-d ``cond_source = "aperture"``. Training apertures EXCLUDE the
band ``a in [0.9, 1.1]`` exactly (0 train draws inside; nearest draws 0.8969 and 1.1044), and
the test split puts 18 configs inside it (a = 0.95 / 1.00 / 1.05 across the 6 frozen test
domains). **That band is the headline split**: the claim is that the field tracks inter-room
coupling CONTINUOUSLY through aperture widths the model never saw, so every observable here is
reported held-out versus seen, never pooled into one number.

Four observables, predicted versus ground truth, per test config:

1. ``level_difference`` -- ``20 log10(<|H|>_roomB / <|H|>_roomA)`` per ISO third-octave band
   inside 20-300 Hz and pooled over the band. Room membership is by receiver x against the
   domain's divider position ``x0`` (read from the HDF5 attrs). This is the FT-B observable
   and the one the sqrt(a) law was measured on.
2. ``mode_split`` -- each sub-room is projected onto ITS OWN analytic mode shapes and the
   number/separation of peaks near each uncoupled sub-room resonance is measured. Coupling
   splits a sub-room mode into a pair. CAVEAT: near-square sub-rooms put (1,0) and (0,1)
   within a linewidth of each other, which makes the split unidentifiable in that domain; the
   per-mode ``degenerate`` flag and the per-domain ``n_degenerate`` count record exactly where.
3. ``decay`` -- band-limited Schroeder EDC per sub-room, single- vs two-segment slope fit over
   the SAME -5..-25 dB window FT-B froze, so double-slope verdicts are comparable.
4. ``lsd`` -- band-limited log-spectral distance, whole domain and per sub-room, ALWAYS
   reported next to the ground truth's own dynamic range. The FDTD corpus spans ~75 dB where
   the ISM corpus of P3-2b spanned ~22 dB, so an absolute LSD here is NOT comparable to
   P3-2b's ~1.0 dB and that comparison must not be made.

**a = 0 (sealed) is a TOPOLOGICAL discontinuity, not a small aperture.** A sealed one-node
divider disconnects room B exactly, so ``H_B`` is identically zero and the level difference is
-inf, not merely large. Sealed configs were excluded from training (their conditioning
coordinate ``sqrt(0) = 0`` is also the limit of a vanishing doorway -- same input, different
target). Here they are excluded from every continuous fit and from every aggregate, and
reported alone under ``topological_reference``. Every dB is guarded against log(0).

Estimators are imported, not reimplemented: ``load_model`` / ``load_gt`` / ``band_limit`` /
``find_checkpoint`` from :mod:`aaf.eval.p3_2_eval` (``load_model`` puts BOTH model and renderer
in ``eval()``; the renderer flag is load-bearing -- ``FreqRenderer2D`` jitters ray azimuths
while ``self.training``, D49 C3), ``_lsd_db`` / ``band_indices`` from
:mod:`aaf.eval.band_limited`, ``enumerate_modes`` / ``mode_shape_matrix`` from
:mod:`aaf.eval.modal_projection`, and ``band_level_ratio`` / ``decay_analysis`` /
``_lin_fit_r2`` / ``THIRD_OCTAVE_HZ`` / the frozen EDC window from :mod:`scripts.p3_3fast_ftb`
so the FT-B thresholds are inherited rather than re-chosen.

``render_config_arm`` from :mod:`aaf.eval.p3_2b_eval` cannot be reused verbatim: it does not
forward ``x0`` / ``a``, and ``build_cond_vector_2d`` REQUIRES them for ``cond_source =
"aperture"`` (the four wall alphas carry no divider information). :func:`render_aperture`
below is that function with the two arguments threaded through.

Usage
-----
    python scripts/p3_3fast_trackB_eval.py                  # needs a GPU (tinycudann)
    python scripts/p3_3fast_trackB_eval.py --gt-only        # ground truth only, CPU
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aaf.data.aperture_configs import A_HOLDOUT, configs_from_rows, in_holdout
from aaf.eval.band_limited import _lsd_db
from aaf.eval.modal_bandwidth import _parabolic_vertex
from aaf.eval.modal_projection import COND_MAX, enumerate_modes, mode_shape_matrix
from aaf.sim.analytical_modal_2d import damping_to_bandwidth_hz, modal_damping_2d

# FT-B froze these; inheriting them is the point -- the eval must not re-choose a window.
from scripts.p3_3fast_ftb import (
    BAND_HI,
    BAND_LO,
    EDC_DB_HI,
    EDC_DB_LO,
    THIRD_OCTAVE_HZ,
    _lin_fit_r2,
    band_level_ratio,
    decay_analysis,
)
from scripts.p3_3fast_trackA_diag import _f, _mean_sd

# --------------------------------------------------------------------------------- frozen
#: The renderer's own rfft grid: fs 4096 / 8192 samples -> df = 0.5 Hz over a 2.0 s record.
#: The FDTD ground truth is fs 61440 / 122880 samples -> df = 0.5 Hz over a 2.0 s record too,
#: and is stored already truncated to its first 601 bins (0-300 Hz). The two grids therefore
#: coincide bin-for-bin below 300 Hz, which is what makes a direct comparison legitimate.
FS_T = 4096.0
N_TIME = 8192
N_FREQ_FULL = N_TIME // 2 + 1          # 4097
DF_HZ = FS_T / N_TIME                  # 0.5
N_BINS_BAND = int(round(BAND_HI / DF_HZ)) + 1        # 601

EPS = 1e-8
"""Usability floor on |H|, matching the eps inside ``_lsd_db``: a cell the log would clamp is
excluded from every statistic rather than silently floored. ``frac_usable`` is reported next
to every number it gates."""

#: Sub-room projection: try these f_max values in order and keep the first whose mode-shape
#: basis is well conditioned. The 8x8 receiver grid leaves only 3-5 columns inside a sub-room,
#: so cond(Phi) blows up to ~1e16 by 200 Hz -- that is spatial aliasing dressed up as physics
#: and `modal_projection` is right to refuse it. 140 Hz clears cond <= 5 in all 12 sub-rooms.
F_MAX_LADDER = (160.0, 140.0, 120.0, 100.0, 80.0, 60.0)

#: Mode-split peak search. The half-window never exceeds 45% of the distance to the nearest
#: other mode of the same sub-room, so the window cannot contain a DIFFERENT mode's peak.
SPLIT_HALF_WINDOW_HZ = 6.0
SPLIT_WINDOW_GAP_FRAC = 0.45
SPLIT_MIN_HALF_WINDOW_HZ = 1.5         # 3 bins; below this a peak pair is not resolvable
SPLIT_MIN_PROMINENCE_DB = 3.0
#: A mode whose nearest neighbour (own sub-room OR the other sub-room) is inside this many
#: linewidths is flagged `degenerate`: its window holds two DIFFERENT modes, not a split pair.
DEGENERACY_LINEWIDTHS = 1.0
#: Wall absorption on every outer wall AND on the divider (aperture_configs.DIVIDER_ALPHA).
ALPHA_WALL = 0.15

MANIFEST = "configs/sweeps_2d_mat/p3_3fast_trackB_manifest.json"
DATA_DIR = "data/track_p3_3fast_B"
TRAIN_DIR = "outputs/p3_3fast/p3_3fast_trackB"
OUT_DIR = "outputs/p3_3fast/trackB"

GROUP_HELDOUT = "held_out_band"
GROUP_SEEN = "seen_apertures"
GROUP_ORDER = (GROUP_HELDOUT, GROUP_SEEN)


def group_of(cfg) -> str:
    """Which reporting group a NON-SEALED config belongs to. Sealed is neither."""
    return GROUP_HELDOUT if in_holdout(cfg.a) else GROUP_SEEN


# ------------------------------------------------------------------------------- geometry
def room_masks(rx: np.ndarray, x0: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Room-A / room-B receiver masks by x against the divider, plus the on-divider count.

    A receiver sitting exactly on x0 belongs to neither sub-room and is dropped from both
    (never silently folded into one), so the count is returned and reported.
    """
    x = np.asarray(rx, dtype=float)[:, 0]
    sel_a, sel_b = x < float(x0), x > float(x0)
    return sel_a, sel_b, int(np.sum(~(sel_a | sel_b)))


def subroom_frame(rx: np.ndarray, L: float, x0: float, room: str):
    """``(local_rx, L_room, mask)`` -- receivers of one sub-room in ITS OWN [0, L_room] frame."""
    sel_a, sel_b, _ = room_masks(rx, x0)
    if room == "A":
        return np.asarray(rx, dtype=float)[sel_a], float(x0), sel_a
    loc = np.asarray(rx, dtype=float)[sel_b].copy()
    loc[:, 0] -= float(x0)
    return loc, float(L) - float(x0), sel_b


def subroom_basis(loc: np.ndarray, L_room: float, W: float):
    """Mode list + pinv(Phi) at the highest ``f_max`` whose basis passes ``cond <= COND_MAX``.

    Returns ``(modes, pinv, info)``. ``info["ok"]`` is False when even the lowest rung is
    ill-conditioned, in which case the sub-room contributes no split measurements at all.
    """
    for f_max in F_MAX_LADDER:
        modes = enumerate_modes(L_room, W, f_max=f_max)
        if not modes:
            continue
        phi = mode_shape_matrix(modes, loc, L_room, W)
        cond = float(np.linalg.cond(phi))
        if cond <= COND_MAX:
            return modes, np.linalg.pinv(phi), {
                "ok": True, "f_max_hz": float(f_max), "cond_phi": cond,
                "n_modes": len(modes), "n_rx": int(loc.shape[0]),
                "L_room_m": float(L_room), "cond_max": COND_MAX,
            }
    return [], None, {"ok": False, "f_max_hz": None, "cond_phi": float("nan"),
                      "n_modes": 0, "n_rx": int(loc.shape[0]), "L_room_m": float(L_room),
                      "reason": "cond(Phi) > {} at every rung of {}".format(
                          COND_MAX, F_MAX_LADDER)}


# ------------------------------------------------------------------------------- spectra
def pad_to_full(H: np.ndarray) -> np.ndarray:
    """``[n_rx, 601] -> [n_rx, 4097]`` zero-padded, so ``irfft`` yields the 2.0 s record."""
    H = np.asarray(H)
    out = np.zeros(H.shape[:-1] + (N_FREQ_FULL,), dtype=np.complex128)
    out[..., :H.shape[-1]] = H
    out[..., 0] = out[..., 0].real
    return out


def subroom_energy(H_band: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Per-sample energy summed over one sub-room's receivers, band-limited by construction."""
    if not np.any(sel):
        return np.zeros(N_TIME)
    rir = np.fft.irfft(pad_to_full(H_band[sel]), n=N_TIME, axis=-1)
    return np.asarray(np.sum(rir * rir, axis=0), dtype=float)


def dynamic_range_db(H_band: np.ndarray, lo_i: int, hi_i: int) -> Dict[str, float]:
    """Span of ``20 log10|H|`` over the in-band cells that clear the log floor.

    Reported beside every LSD. ``max_minus_min_db`` is the headline (~75 dB on this FDTD
    corpus versus ~22 dB on the P3-2b ISM corpus) and the percentile span is the
    outlier-insensitive companion.
    """
    m = np.abs(np.asarray(H_band)[:, lo_i:hi_i])
    m = m[m > EPS]
    if m.size < 16:
        return {"max_minus_min_db": float("nan"), "p999_minus_p001_db": float("nan"),
                "n_cells": int(m.size)}
    db = 20.0 * np.log10(m)
    return {"max_minus_min_db": float(db.max() - db.min()),
            "p999_minus_p001_db": float(np.percentile(db, 99.9) - np.percentile(db, 0.1)),
            "n_cells": int(m.size)}


def frac_usable(*arrays: np.ndarray) -> float:
    """Fraction of in-band cells finite AND above the log floor in EVERY argument."""
    ok = None
    for a in arrays:
        m = np.isfinite(np.asarray(a)) & (np.abs(np.asarray(a)) > EPS)
        ok = m if ok is None else (ok & m)
    return _f(np.mean(ok)) if ok is not None else float("nan")


# ------------------------------------------------------------------- observable 1: level
def level_difference(H_band: np.ndarray, freqs: np.ndarray, sel_a: np.ndarray,
                     sel_b: np.ndarray) -> Dict[str, object]:
    """Inter-room level difference, per third-octave band and pooled over 20-300 Hz.

    ``band_level_ratio`` already returns -inf when room B is silent (the sealed case) and nan
    when room A is, so no dB is ever taken of a zero here.
    """
    bands = []
    for fc in THIRD_OCTAVE_HZ:
        lo, hi = fc * 2 ** (-1.0 / 6.0), fc * 2 ** (1.0 / 6.0)
        bands.append({"fc_hz": float(fc), "lo_hz": float(lo), "hi_hz": float(hi),
                      "ld_db": band_level_ratio(H_band, freqs, sel_a, sel_b, lo, hi)})
    return {"bands": bands,
            "ld_broadband_db": band_level_ratio(H_band, freqs, sel_a, sel_b,
                                                BAND_LO, BAND_HI)}


# -------------------------------------------------------------- observable 2: mode split
def _peak_pair(mag: np.ndarray, freqs: np.ndarray, f0: float, hw: float
               ) -> Dict[str, float]:
    """Peak structure of one mode-resolved spectrum inside ``f0 +/- hw``.

    Two numbers come out, and they have very different resolution:

    ``f_peak_hz`` -- the dominant peak, refined sub-bin by the same parabolic-in-dB vertex the
    bandwidth estimator uses. A peak POSITION is identifiable far below the linewidth, so this
    is the well-conditioned half of the modal observable and drives ``migration``.

    ``split_hz`` -- the separation of the two dominant peaks when the valley between them is at
    least ``SPLIT_MIN_PROMINENCE_DB`` below the weaker of the two (a genuine doublet); 0.0 when
    the window holds a single peak (a legitimate "no split", not missing data); nan when no
    interior maximum exists at all. Resolving a doublet REQUIRES a separation of order the
    linewidth, so this observable is blind to the sub-linewidth splitting FT-B could see with
    its even/odd decomposition -- which is unavailable here because x0 varies and the two
    sub-rooms are not congruent, so there is no mirror symmetry to decompose along.
    """
    lo = int(np.searchsorted(freqs, f0 - hw, side="left"))
    hi = int(np.searchsorted(freqs, f0 + hw, side="right"))
    if hi - lo < 5:
        return {"split_hz": float("nan"), "f_peak_hz": float("nan"), "n_peaks": 0,
                "reason": "window < 5 bins"}
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(np.abs(mag[lo:hi]), 1e-30))
    f = freqs[lo:hi]
    idx = [i for i in range(1, db.size - 1)
           if db[i] >= db[i - 1] and db[i] > db[i + 1]]
    if not idx:
        return {"split_hz": float("nan"), "f_peak_hz": float("nan"), "n_peaks": 0,
                "reason": "no interior maximum"}
    order = sorted(idx, key=lambda i: -db[i])

    def refine(i: int) -> float:
        d, _ = _parabolic_vertex(db[i - 1], db[i], db[i + 1])
        return float(f[i] + d * DF_HZ)

    top = order[0]
    out = {"f_peak_hz": refine(top), "peak_level_db": float(db[top])}
    for j in order[1:]:
        a, b = (top, j) if top < j else (j, top)
        valley = float(np.min(db[a:b + 1]))
        if min(db[a], db[b]) - valley >= SPLIT_MIN_PROMINENCE_DB:
            out.update({"split_hz": abs(refine(b) - refine(a)), "n_peaks": 2,
                        "f_lo_hz": refine(a), "f_hi_hz": refine(b),
                        "prominence_db": float(min(db[a], db[b]) - valley)})
            return out
    out.update({"split_hz": 0.0, "n_peaks": 1, "f_lo_hz": refine(top),
                "f_hi_hz": refine(top)})
    return out


def mode_linewidth_hz(L_room: float, W: float, m) -> float:
    """Kuttruff -3 dB width of one sub-room mode at the baseline absorption.

    The same estimator FT-B used to size its bandwidth caps. Preferred over 2.2/T60 for the
    degeneracy flag because it is PER MODE: the caveat is about two particular modes landing
    inside each other's width, not about the room's average decay.
    """
    return float(damping_to_bandwidth_hz(
        modal_damping_2d(L_room, W, [ALPHA_WALL] * 4, m.n_x, m.n_y, model="kuttruff")))


def mode_split(H_gt: np.ndarray, H_pred: Optional[np.ndarray], rx: np.ndarray,
               L: float, W: float, x0: float, freqs: np.ndarray,
               linewidth_hz: float) -> Dict[str, object]:
    """Per-sub-room modal peak position and splitting, predicted versus ground truth.

    Each sub-room is projected onto its OWN analytic basis (room B shifted into its local
    frame -- the sub-rooms are NOT congruent here because x0 varies, so FT-B's mirror/even-odd
    trick does not apply). Two observables come out of each mode's window: the peak POSITION
    (well conditioned, sub-linewidth) and the two-peak SPLITTING (needs a separation of order
    the linewidth). Modes whose nearest neighbour in EITHER sub-room is inside one Kuttruff
    linewidth are flagged ``degenerate`` -- their window holds a different mode, not a split
    partner -- and are excluded from the statistics while staying visible in the count.
    """
    bases, mode_f = {}, []
    for room in ("A", "B"):
        loc, L_room, sel = subroom_frame(rx, L, x0, room)
        modes, pinv, info = subroom_basis(loc, L_room, W)
        bases[room] = {"modes": modes, "pinv": pinv, "sel": sel, "info": info,
                       "L_room": L_room}
        mode_f.extend(float(m.f) for m in modes)
    mode_f_all = np.asarray(sorted(mode_f), dtype=float)

    rows: List[dict] = []
    for room in ("A", "B"):
        b = bases[room]
        if not b["info"]["ok"] or not np.any(b["sel"]):
            continue
        own = np.asarray([float(m.f) for m in b["modes"]], dtype=float)
        a_gt = np.abs(b["pinv"] @ H_gt[b["sel"]])
        a_pr = np.abs(b["pinv"] @ H_pred[b["sel"]]) if H_pred is not None else None
        for j, m in enumerate(b["modes"]):
            gap_own = float(np.min(np.abs(np.delete(own, j) - m.f))) if own.size > 1 \
                else float("inf")
            others = mode_f_all[np.abs(mode_f_all - m.f) > 1e-9]
            gap_any = float(np.min(np.abs(others - m.f))) if others.size else float("inf")
            lw = mode_linewidth_hz(b["L_room"], W, m)
            hw = min(SPLIT_HALF_WINDOW_HZ, SPLIT_WINDOW_GAP_FRAC * gap_own)
            rec = {"room": room, "n_x": int(m.n_x), "n_y": int(m.n_y),
                   "family": m.family, "f_analytic_hz": float(m.f),
                   "linewidth_kuttruff_hz": lw,
                   "gap_own_subroom_hz": gap_own, "gap_any_subroom_hz": gap_any,
                   "half_window_hz": float(hw),
                   "degenerate": bool(gap_any < DEGENERACY_LINEWIDTHS * lw)}
            if hw < SPLIT_MIN_HALF_WINDOW_HZ:
                rec.update({"gt": {"split_hz": float("nan"), "f_peak_hz": float("nan"),
                                   "n_peaks": 0,
                                   "reason": "neighbour mode within {:.2f} Hz".format(gap_own)},
                            "dropped": True})
                rows.append(rec)
                continue
            rec["gt"] = _peak_pair(a_gt[j], freqs, m.f, hw)
            rec["migration_gt_hz"] = _f(rec["gt"]["f_peak_hz"] - m.f)
            if a_pr is not None:
                rec["pred"] = _peak_pair(a_pr[j], freqs, m.f, hw)
                rec["migration_pred_hz"] = _f(rec["pred"]["f_peak_hz"] - m.f)
            rec["dropped"] = bool(not np.isfinite(rec["gt"]["split_hz"])
                                  or (a_pr is not None
                                      and not np.isfinite(rec["pred"]["split_hz"])))
            rows.append(rec)

    kept = [r for r in rows if not r["dropped"]]
    usable = [r for r in kept if not r["degenerate"]]
    out = {
        "bases": {k: bases[k]["info"] for k in ("A", "B")},
        "linewidth_from_decay_hz": float(linewidth_hz),
        "linewidth_kuttruff_median_hz": _f(np.median(
            [r["linewidth_kuttruff_hz"] for r in rows])) if rows else float("nan"),
        "modes": rows,
        "n_modes": len(rows),
        "n_kept": len(kept),
        "n_usable_modes": len(usable),
        "n_degenerate": int(sum(1 for r in rows if r["degenerate"])),
        "frac_modes_dropped": _f(1.0 - len(kept) / float(len(rows))) if rows
        else float("nan"),
        "gt_mean_split_hz": _f(np.mean([r["gt"]["split_hz"] for r in usable]))
        if usable else float("nan"),
        "gt_mean_abs_migration_hz": _f(np.mean(
            [abs(r["migration_gt_hz"]) for r in usable])) if usable else float("nan"),
        "frac_split_gt": _f(np.mean([r["gt"]["n_peaks"] == 2 for r in usable]))
        if usable else float("nan"),
    }
    if H_pred is not None and usable:
        g = np.asarray([r["gt"]["split_hz"] for r in usable], dtype=float)
        p = np.asarray([r["pred"]["split_hz"] for r in usable], dtype=float)
        fg = np.asarray([r["gt"]["f_peak_hz"] for r in usable], dtype=float)
        fp = np.asarray([r["pred"]["f_peak_hz"] for r in usable], dtype=float)
        mg = np.asarray([r["migration_gt_hz"] for r in usable], dtype=float)
        mp = np.asarray([r["migration_pred_hz"] for r in usable], dtype=float)
        out.update({
            "pred_mean_split_hz": _f(np.mean(p)),
            "frac_split_pred": _f(np.mean([r["pred"]["n_peaks"] == 2 for r in usable])),
            "mean_abs_split_error_hz": _f(np.mean(np.abs(p - g))),
            "split_pearson": _pearson(g, p),
            "pred_mean_abs_migration_hz": _f(np.mean(np.abs(mp))),
            "mean_abs_peak_error_hz": _f(np.mean(np.abs(fp - fg))),
            "migration_pearson": _pearson(mg, mp),
        })
    elif H_pred is not None:
        out.update({"pred_mean_split_hz": float("nan"),
                    "pred_mean_abs_migration_hz": float("nan"),
                    "mean_abs_peak_error_hz": float("nan")})
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r; 0.0 for a constant vector (a real result), nan for < 3 finite pairs."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    if np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


# ------------------------------------------------------------------- observable 3: decay
def decay(H_band: np.ndarray, sel_a: np.ndarray, sel_b: np.ndarray) -> Dict[str, dict]:
    """Single- vs double-slope EDC fit per sub-room, over FT-B's frozen -5..-25 dB window."""
    return {"room_A": decay_analysis(subroom_energy(H_band, sel_a), FS_T),
            "room_B": decay_analysis(subroom_energy(H_band, sel_b), FS_T)}


def linewidth_from_decay(dec: Dict[str, dict]) -> float:
    """Modal linewidth (Hz) implied by the measured decay: BW_3dB = gamma/pi = 2.2/T60."""
    t = [d["t60_single_s"] for d in dec.values()
         if d.get("valid") and np.isfinite(d.get("t60_single_s", np.nan))
         and d["t60_single_s"] > 0]
    return float(2.2 / np.mean(t)) if t else float("nan")


# --------------------------------------------------------------------- observable 4: LSD
def lsd_block(H_pred: np.ndarray, H_gt: np.ndarray, sel_a: np.ndarray, sel_b: np.ndarray,
              lo_i: int, hi_i: int) -> Dict[str, object]:
    """Band-limited LSD whole-domain and per sub-room, each with its usable-cell fraction.

    ``dynamic_range_db`` of the GROUND TRUTH travels with every LSD: on this FDTD corpus the
    in-band span is ~75 dB against the ISM corpus's ~22 dB, so P3-2b's ~1.0 dB is NOT a
    comparable reference and ``lsd_over_dynamic_range`` is the only cross-corpus quantity.
    """
    out: Dict[str, object] = {}
    for name, sel in (("all", np.ones(H_gt.shape[0], dtype=bool)),
                      ("room_A", sel_a), ("room_B", sel_b)):
        if not np.any(sel):
            out[name] = {"lsd_db": float("nan"), "n_rx": 0}
            continue
        p, g = H_pred[sel][:, lo_i:hi_i], H_gt[sel][:, lo_i:hi_i]
        dr = dynamic_range_db(H_gt[sel], lo_i, hi_i)
        lsd = _f(_lsd_db(p, g))
        out[name] = {
            "lsd_db": lsd,
            "n_rx": int(np.sum(sel)),
            "frac_usable": frac_usable(p, g),
            "gt_dynamic_range_db": dr["max_minus_min_db"],
            "gt_dynamic_range_p999_db": dr["p999_minus_p001_db"],
            "lsd_over_dynamic_range": _f(lsd / dr["max_minus_min_db"])
            if np.isfinite(dr["max_minus_min_db"]) and dr["max_minus_min_db"] > 0
            else float("nan"),
        }
    return out


# --------------------------------------------------------------------------- model access
def render_aperture(model, renderer, cond_source: str, L: float, W: float, x0: float,
                    a: float, alphas: Sequence[float], rx: np.ndarray, src: np.ndarray,
                    device, rx_chunk: int = 8) -> np.ndarray:
    """ZERO-SHOT render of one aperture config -> ``[n_rx, n_freq]`` complex64.

    ``p3_2b_eval.render_config_arm`` with ``x0`` and ``a`` threaded into
    ``build_cond_vector_2d``; the aperture arm raises without them because the four wall
    alphas carry no divider information. Nothing about this config's ground truth is read.
    """
    import torch

    from aaf.models.conditioning_2d import build_cond_vector_2d

    with torch.no_grad():
        cond = build_cond_vector_2d(cond_source, L, W, alphas, device=device,
                                    x0=float(x0), a=float(a))
        room_min = torch.zeros(2, device=device)
        room_max = torch.tensor([float(L), float(W)], dtype=torch.float32, device=device)
        rx_t = torch.as_tensor(np.asarray(rx, dtype=np.float32), device=device)
        src_t = torch.as_tensor(np.asarray(src, dtype=np.float32), device=device)
        out = []
        for s in range(0, rx_t.shape[0], rx_chunk):
            r = rx_t[s:s + rx_chunk]
            tx = src_t.unsqueeze(0).expand(r.shape[0], -1)
            z = cond.unsqueeze(0).expand(r.shape[0], -1)
            out.append(renderer(model, r, tx, room_min, room_max, z_s=z).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.complex64)


# ------------------------------------------------------------------------ the sqrt(a) fit
def _fit(u: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """OLS ``y = m u + b`` with r, r^2, n AND the span of ``u`` the fit had to work with.

    ``u_range`` is not decoration. The held-out band is a in [0.9, 1.1], i.e. sqrt(a) in
    [0.949, 1.049] -- a span of 0.10 against ~1.2 for the seen apertures. A regression run
    INSIDE the band is therefore near-degenerate by construction and its r^2 says almost
    nothing about the model; the seen-line residual test below is the one that does.
    """
    u, y = np.asarray(u, dtype=float), np.asarray(y, dtype=float)
    m, b, r2 = _lin_fit_r2(u, y)
    ok = np.isfinite(u) & np.isfinite(y)
    return {"slope": m, "intercept": b, "r2": r2,
            "pearson": _pearson(u[ok], y[ok]),
            "u_range": _f(np.ptp(u[ok])) if int(ok.sum()) else float("nan"),
            "n": int(ok.sum())}


def continuity_fit(rows: Sequence[dict], with_pred: bool) -> Dict[str, object]:
    """The headline: does the level difference sit on ONE sqrt(a) line through the hold-out?

    sqrt(a) is FT-B's measured linearizing coordinate (pooled r^2 = 0.9870 on ground truth;
    raw ``a`` gives 0.905). Three questions, kept apart:

    * ``by_group`` -- fit sqrt(a) separately inside and outside the held-out band. A model
      that memorized aperture values shows a broken slope across the two.
    * ``seen_line`` -- fit the SEEN points only, then evaluate the held-out points against
      that line. This is the sharp test: if the field tracks coupling continuously, the
      held-out residuals are the size of the seen residuals, not larger.
    * ``pred_vs_gt`` -- regress the predicted level difference on the ground-truth one, per
      group. Slope 1 / r 1 is perfect edit transfer regardless of the aperture law.

    Sealed configs never enter here: ``a = 0`` is a topological discontinuity whose level
    difference is -inf, and no continuous coordinate contains it.
    """
    rows = [r for r in rows if not r["sealed"] and np.isfinite(r["ld_gt_db"])]
    out: Dict[str, object] = {
        "coordinate": "sqrt(a)",
        "response": "inter-room level difference, 20-300 Hz pooled (dB)",
        "excluded": "sealed (a = 0): topological discontinuity, LD = -inf",
        "ft_b_reference": {"pooled_r2_sqrt_a": 0.9870078558194965, "slope": 6.807838364670918,
                           "source": "outputs/p3_3fast/trackB/aperture_sweep.json"},
    }
    u = np.asarray([math.sqrt(r["a"]) for r in rows], dtype=float)
    g = np.asarray([r["ld_gt_db"] for r in rows], dtype=float)
    grp = np.asarray([r["group"] for r in rows])
    dom = np.asarray([r["geom_id"] for r in rows], dtype=int)
    p = np.asarray([r.get("ld_pred_db", np.nan) for r in rows], dtype=float) if with_pred \
        else None

    out["pooled"] = {"gt": _fit(u, g)}
    out["by_group"] = {k: {"gt": _fit(u[grp == k], g[grp == k])} for k in GROUP_ORDER}
    if with_pred:
        out["pooled"]["pred"] = _fit(u, p)
        for k in GROUP_ORDER:
            out["by_group"][k]["pred"] = _fit(u[grp == k], p[grp == k])
            out["by_group"][k]["pred_vs_gt"] = _fit(g[grp == k], p[grp == k])
        out["pooled"]["pred_vs_gt"] = _fit(g, p)

    # -- seen-only line, held-out points scored against it -------------------------------
    seen = grp == GROUP_SEEN
    hold = grp == GROUP_HELDOUT
    seen_line: Dict[str, object] = {}
    for name, y in (("gt", g),) + ((("pred", p),) if with_pred else ()):
        fit = _fit(u[seen], y[seen])
        res_s = y[seen] - (fit["slope"] * u[seen] + fit["intercept"])
        res_h = y[hold] - (fit["slope"] * u[hold] + fit["intercept"])
        seen_line[name] = {
            "fit_on_seen": fit,
            "rms_residual_seen_db": _f(np.sqrt(np.mean(res_s ** 2))),
            "rms_residual_heldout_db": _f(np.sqrt(np.mean(res_h ** 2))),
            "mean_residual_heldout_db": _f(np.mean(res_h)),
            "heldout_over_seen_rms": _f(np.sqrt(np.mean(res_h ** 2))
                                        / np.sqrt(np.mean(res_s ** 2)))
            if res_s.size and np.sqrt(np.mean(res_s ** 2)) > 0 else float("nan"),
            "n_seen": int(seen.sum()), "n_heldout": int(hold.sum()),
        }
    out["seen_line"] = seen_line

    # -- per domain: pooling 6 geometries adds (L, W, x0) scatter the law does not own ----
    per_dom = []
    for d in sorted(set(dom.tolist())):
        s = dom == d
        rec = {"geom_id": int(d), "n": int(s.sum()), "gt": _fit(u[s], g[s])}
        if with_pred:
            rec["pred"] = _fit(u[s], p[s])
            rec["pred_vs_gt"] = _fit(g[s], p[s])
        sd = s & seen
        if int(sd.sum()) >= 3:
            f_gt = _fit(u[sd], g[sd])
            hd = s & hold
            rec["heldout_residual_vs_seen_line_gt_db"] = [
                _f(v) for v in (g[hd] - (f_gt["slope"] * u[hd] + f_gt["intercept"]))]
            if with_pred:
                f_pr = _fit(u[sd], p[sd])
                rec["heldout_residual_vs_seen_line_pred_db"] = [
                    _f(v) for v in (p[hd] - (f_pr["slope"] * u[hd] + f_pr["intercept"]))]
        per_dom.append(rec)
    out["per_domain"] = per_dom
    out["per_domain_summary"] = {
        "gt_r2": _mean_sd([r["gt"]["r2"] for r in per_dom]),
        "gt_slope": _mean_sd([r["gt"]["slope"] for r in per_dom]),
    }
    if with_pred:
        out["per_domain_summary"]["pred_r2"] = _mean_sd([r["pred"]["r2"] for r in per_dom])
        out["per_domain_summary"]["pred_slope"] = _mean_sd(
            [r["pred"]["slope"] for r in per_dom])
    return out


# ------------------------------------------------------------------------------ aggregate
def aggregate(rows: Sequence[dict], with_pred: bool) -> Dict[str, object]:
    """Every observable, split held-out band versus seen apertures. Sealed never enters."""
    out: Dict[str, object] = {}
    for grp in GROUP_ORDER:
        sel = [r for r in rows if not r["sealed"] and r["group"] == grp]
        blk: Dict[str, object] = {"n_configs": len(sel),
                                  "a_values": sorted({r["a"] for r in sel})}
        blk["level_difference"] = {
            "gt_db": _mean_sd([r["ld_gt_db"] for r in sel]),
            "frac_usable": _mean_sd([r["ld_frac_usable"] for r in sel]),
        }
        blk["mode_split"] = {
            "gt_mean_split_hz": _mean_sd([r["modal"]["gt_mean_split_hz"] for r in sel]),
            "gt_mean_abs_migration_hz": _mean_sd([r["modal"]["gt_mean_abs_migration_hz"]
                                                  for r in sel]),
            "frac_modes_dropped": _mean_sd([r["modal"]["frac_modes_dropped"]
                                            for r in sel]),
            "n_modes": _mean_sd([r["modal"]["n_modes"] for r in sel]),
            "n_usable_modes": _mean_sd([r["modal"]["n_usable_modes"] for r in sel]),
            "n_degenerate": _mean_sd([r["modal"]["n_degenerate"] for r in sel]),
            "frac_split_gt": _mean_sd([r["modal"]["frac_split_gt"] for r in sel]),
            "linewidth_kuttruff_median_hz": _mean_sd(
                [r["modal"]["linewidth_kuttruff_median_hz"] for r in sel]),
        }
        blk["decay"] = {
            "gt_t60_single_s_roomA": _mean_sd([_valid(r["decay_gt"], "room_A") for r in sel]),
            "gt_t60_single_s_roomB": _mean_sd([_valid(r["decay_gt"], "room_B") for r in sel]),
            "gt_n_double_slope_roomB": int(sum(
                1 for r in sel if r["decay_gt"]["room_B"].get("double_slope"))),
            "gt_n_valid_roomB": int(sum(
                1 for r in sel if r["decay_gt"]["room_B"].get("valid"))),
        }
        if with_pred:
            blk["level_difference"].update({
                "pred_db": _mean_sd([r["ld_pred_db"] for r in sel]),
                "abs_error_db": _mean_sd([abs(r["ld_pred_db"] - r["ld_gt_db"])
                                          for r in sel]),
                "signed_error_db": _mean_sd([r["ld_pred_db"] - r["ld_gt_db"] for r in sel]),
                "pearson_pred_vs_gt": _pearson(
                    np.asarray([r["ld_gt_db"] for r in sel]),
                    np.asarray([r["ld_pred_db"] for r in sel])),
            })
            blk["level_difference"]["by_third_octave"] = _band_table(sel)
            blk["mode_split"].update({
                "pred_mean_split_hz": _mean_sd([r["modal"]["pred_mean_split_hz"]
                                                for r in sel]),
                "mean_abs_split_error_hz": _mean_sd([r["modal"].get(
                    "mean_abs_split_error_hz", float("nan")) for r in sel]),
                "split_pearson": _mean_sd([r["modal"].get("split_pearson", float("nan"))
                                           for r in sel]),
                "pred_mean_abs_migration_hz": _mean_sd(
                    [r["modal"]["pred_mean_abs_migration_hz"] for r in sel]),
                "mean_abs_peak_error_hz": _mean_sd(
                    [r["modal"]["mean_abs_peak_error_hz"] for r in sel]),
                "migration_pearson": _mean_sd([r["modal"].get("migration_pearson",
                                                              float("nan")) for r in sel]),
            })
            blk["decay"].update({
                "pred_t60_single_s_roomA": _mean_sd([_valid(r["decay_pred"], "room_A")
                                                     for r in sel]),
                "pred_t60_single_s_roomB": _mean_sd([_valid(r["decay_pred"], "room_B")
                                                     for r in sel]),
                "abs_t60_error_s_roomA": _mean_sd([
                    abs(_valid(r["decay_pred"], "room_A") - _valid(r["decay_gt"], "room_A"))
                    for r in sel]),
                "abs_t60_error_s_roomB": _mean_sd([
                    abs(_valid(r["decay_pred"], "room_B") - _valid(r["decay_gt"], "room_B"))
                    for r in sel]),
                "pred_n_double_slope_roomB": int(sum(
                    1 for r in sel if r["decay_pred"]["room_B"].get("double_slope"))),
            })
            blk["lsd"] = {
                k: {"lsd_db": _mean_sd([r["lsd"][k]["lsd_db"] for r in sel]),
                    "gt_dynamic_range_db": _mean_sd([r["lsd"][k]["gt_dynamic_range_db"]
                                                     for r in sel]),
                    "lsd_over_dynamic_range": _mean_sd([r["lsd"][k]["lsd_over_dynamic_range"]
                                                        for r in sel]),
                    "frac_usable": _mean_sd([r["lsd"][k]["frac_usable"] for r in sel])}
                for k in ("all", "room_A", "room_B")}
        out[grp] = blk
    return out


def _valid(dec: Dict[str, dict], room: str) -> float:
    d = dec.get(room, {})
    return float(d["t60_single_s"]) if d.get("valid") else float("nan")


def _band_table(sel: Sequence[dict]) -> List[dict]:
    """Per third-octave band: GT and predicted level difference and the error."""
    out = []
    for i, fc in enumerate(THIRD_OCTAVE_HZ):
        g = [r["ld_bands_gt"][i]["ld_db"] for r in sel]
        p = [r["ld_bands_pred"][i]["ld_db"] for r in sel]
        e = [abs(pp - gg) for pp, gg in zip(p, g)
             if np.isfinite(pp) and np.isfinite(gg)]
        out.append({"fc_hz": float(fc), "gt_db": _mean_sd(g), "pred_db": _mean_sd(p),
                    "abs_error_db": _mean_sd(e),
                    "frac_usable": _mean_sd([r["ld_bands_frac"][i] for r in sel])})
    return out


def topological_reference(rows: Sequence[dict], with_pred: bool) -> Dict[str, object]:
    """The sealed configs, reported ALONE and never as a point on the coupling curve."""
    sel = [r for r in rows if r["sealed"]]
    blk: Dict[str, object] = {
        "n_configs": len(sel),
        "why_separate": ("a = 0 disconnects room B EXACTLY, so H_B == 0 and the level "
                         "difference is -inf. sqrt(0) = 0 is also the limit of a vanishing "
                         "doorway, so the conditioning cannot separate the two cases; sealed "
                         "configs were excluded from training and are excluded from every "
                         "continuous fit and aggregate here."),
        "gt_room_b_energy_is_zero": bool(all(r["gt_roomB_all_zero"] for r in sel)),
        "gt_ld_db": [r["ld_gt_db"] for r in sel],
    }
    if with_pred:
        blk.update({
            "pred_ld_db": _mean_sd([r["ld_pred_db"] for r in sel]),
            "pred_ld_db_each": [r["ld_pred_db"] for r in sel],
            "lsd_room_A_db": _mean_sd([r["lsd"]["room_A"]["lsd_db"] for r in sel]),
            "note": ("room-B LSD is undefined for a sealed divider (the target is "
                     "identically zero) and is not reported; the predicted level difference "
                     "is finite by construction and is the MEASURE OF THE GAP between the "
                     "continuous law the model learned and the topological truth"),
        })
    return blk


# ------------------------------------------------------------------------------ per config
def evaluate_config(cfg, data_dir: Path, hi_idx: int, model, renderer, cond_source: str,
                    device, rx_chunk: int) -> dict:
    from aaf.eval.p3_2_eval import band_limit, load_gt

    H_raw, rx, src, gt_split = load_gt(data_dir / cfg.filename)
    if gt_split != cfg.split:
        raise ValueError("{} carries split {!r} but the manifest says {!r}".format(
            cfg.filename, gt_split, cfg.split))
    if H_raw.shape[-1] < hi_idx:
        raise ValueError("GT {} has {} bins, need {}".format(
            cfg.filename, H_raw.shape[-1], hi_idx))
    H_gt = band_limit(H_raw, hi_idx)[:, :hi_idx]
    freqs = np.arange(hi_idx) * DF_HZ
    lo_i = int(round(BAND_LO / DF_HZ))
    sel_a, sel_b, n_on = room_masks(rx, cfg.x0)

    with_pred = model is not None
    H_pred = None
    if with_pred:
        H_pred = band_limit(
            render_aperture(model, renderer, cond_source, cfg.L, cfg.W, cfg.x0, cfg.a,
                            cfg.alphas, rx, src, device, rx_chunk), hi_idx)[:, :hi_idx]

    ld_gt = level_difference(H_gt, freqs, sel_a, sel_b)
    dec_gt = decay(H_gt, sel_a, sel_b)
    lw = linewidth_from_decay(dec_gt)
    modal = mode_split(H_gt, H_pred, rx, cfg.L, cfg.W, cfg.x0, freqs, lw)

    rec = {
        "filename": cfg.filename, "split": gt_split, "kind": cfg.kind,
        "geom_id": cfg.geom_id, "L": cfg.L, "W": cfg.W, "x0": cfg.x0, "a": cfg.a,
        "sqrt_a": math.sqrt(cfg.a), "sealed": bool(cfg.sealed),
        "fully_open": bool(cfg.fully_open),
        "group": "sealed" if cfg.sealed else group_of(cfg),
        "n_rx_A": int(sel_a.sum()), "n_rx_B": int(sel_b.sum()), "n_rx_on_divider": n_on,
        "ld_gt_db": ld_gt["ld_broadband_db"],
        "ld_bands_gt": ld_gt["bands"],
        "ld_frac_usable": frac_usable(H_gt[:, lo_i:hi_idx]),
        "ld_bands_frac": [_band_frac(H_gt, freqs, b["lo_hz"], b["hi_hz"])
                          for b in ld_gt["bands"]],
        "gt_roomB_all_zero": bool(np.all(np.abs(H_gt[sel_b]) <= 0.0))
        if np.any(sel_b) else True,
        "gt_dynamic_range_db": dynamic_range_db(H_gt, lo_i, hi_idx)["max_minus_min_db"],
        "decay_gt": dec_gt,
        "modal": modal,
    }
    if with_pred:
        ld_pr = level_difference(H_pred, freqs, sel_a, sel_b)
        rec.update({
            "ld_pred_db": ld_pr["ld_broadband_db"],
            "ld_bands_pred": ld_pr["bands"],
            "ld_abs_error_db": _f(abs(ld_pr["ld_broadband_db"] - ld_gt["ld_broadband_db"]))
            if np.isfinite(ld_pr["ld_broadband_db"])
            and np.isfinite(ld_gt["ld_broadband_db"]) else float("nan"),
            "decay_pred": decay(H_pred, sel_a, sel_b),
            "lsd": lsd_block(H_pred, H_gt, sel_a, sel_b, lo_i, hi_idx),
        })
    return rec


def _band_frac(H: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    m = (freqs >= lo) & (freqs <= hi)
    return frac_usable(H[:, m]) if m.any() else float("nan")


# ---------------------------------------------------------------------------------- driver
def run(train_dir: str, manifest: str, data_dir: str, out_dir: str,
        checkpoint: Optional[str], gt_only: bool, rx_chunk: int,
        limit: Optional[int]) -> dict:
    t0 = time.time()
    man = json.loads(Path(manifest).read_text())
    rows_in = man["configs"] if isinstance(man, dict) else man
    cfgs = configs_from_rows(rows_in, split="test")
    if limit:
        cfgs = cfgs[:limit]

    meta = {
        "eval": "P3-3-FAST Track 2b aperture generalization",
        "manifest": manifest, "data_dir": data_dir, "train_dir": train_dir,
        "n_test_configs": len(cfgs),
        "holdout_band_a": list(A_HOLDOUT),
        "band_hz": [BAND_LO, BAND_HI],
        "third_octave_fc_hz": list(THIRD_OCTAVE_HZ),
        "edc_fit_window_db": [EDC_DB_HI, EDC_DB_LO],
        "df_hz": DF_HZ, "eps_usable": EPS, "gt_only": bool(gt_only),
        "f_max_ladder_hz": list(F_MAX_LADDER), "cond_max": COND_MAX,
        "lsd_scoping": ("this FDTD corpus spans ~75 dB in band against the P3-2b ISM "
                        "corpus's ~22 dB, so absolute LSD is NOT comparable to P3-2b's "
                        "~1.0 dB; use lsd_over_dynamic_range for any cross-chunk statement"),
    }

    model = renderer = device = None
    cond_source = "aperture"
    hi_idx = N_BINS_BAND
    if not gt_only:
        import torch

        from aaf.eval.p3_2_eval import find_checkpoint, load_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = Path(checkpoint) if checkpoint else find_checkpoint(train_dir)
        model, renderer, cfg, tmeta, it = load_model(ckpt, device)
        cond_source = str(cfg["cond_source"])
        if cond_source != "aperture" or int(cfg["cond_dim"]) != 55:
            raise ValueError("expected aperture/55 conditioning, got {}/{}".format(
                cond_source, cfg["cond_dim"]))
        n_freq = int(cfg["n_time_samples"]) // 2 + 1
        df = float(cfg["fs"]) / float(cfg["n_time_samples"])
        if abs(df - DF_HZ) > 1e-9 or n_freq != N_FREQ_FULL:
            raise ValueError(
                "renderer grid (df {:.6f} Hz, {} bins) does not match the GT grid "
                "(df {} Hz, {} bins); the two spectra are not bin-aligned".format(
                    df, n_freq, DF_HZ, N_FREQ_FULL))
        sp = Path(train_dir) / "scalars.json"
        val = [r for r in json.loads(sp.read_text())
               if r.get("phase") == "val" and int(r.get("iter", 0)) <= it] \
            if sp.exists() else []
        meta.update({
            "checkpoint": str(ckpt), "iter": int(it), "cond_source": cond_source,
            "cond_dim": int(cfg["cond_dim"]), "device": str(device),
            "conditioning_type": str(cfg.get("conditioning_type", "film")),
            "in_dist_val_lsd_db": float(val[-1]["lsd_db"]) if val else None,
            "n_train_configs": int(tmeta.get("n_configs", 0)),
        })
    meta["hi_idx"] = int(hi_idx)

    records: List[dict] = []
    for i, c in enumerate(cfgs):
        records.append(evaluate_config(c, Path(data_dir), hi_idx, model, renderer,
                                       cond_source, device, rx_chunk))
        if (i + 1) % 6 == 0 or i + 1 == len(cfgs):
            print("  [{}/{}] {} ({:.1f}s)".format(i + 1, len(cfgs), c.filename,
                                                  time.time() - t0), flush=True)

    with_pred = not gt_only
    res = {
        "meta": meta,
        "groups": aggregate(records, with_pred),
        "continuity": continuity_fit(records, with_pred),
        "topological_reference": topological_reference(records, with_pred),
        "per_config": records,
    }
    n_sealed = sum(1 for r in records if r["sealed"])
    res["meta"].update({
        "n_sealed_excluded": n_sealed,
        "n_heldout": sum(1 for r in records if r["group"] == GROUP_HELDOUT),
        "n_seen": sum(1 for r in records if r["group"] == GROUP_SEEN),
        "wall_seconds": round(time.time() - t0, 1),
    })

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = "EVAL_GT_ONLY.json" if gt_only else "EVAL.json"
    (out / name).write_text(json.dumps(res, indent=1, default=float))
    print("[wrote] {}".format(out / name), flush=True)
    write_report(res, out / ("EVAL_GT_ONLY.md" if gt_only else "EVAL.md"), with_pred)
    return res


# ----------------------------------------------------------------------------- the report
def _n(v, spec: str = "{:.3f}") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(f):
        return "-inf" if f == float("-inf") else ("inf" if f > 0 else "n/a")
    return spec.format(f)


def _ms(d: dict, spec: str = "{:.3f}") -> str:
    if not isinstance(d, dict):
        return "n/a"
    return "{} +/- {}".format(_n(d.get("mean"), spec), _n(d.get("sd"), spec))


def write_report(res: dict, path: Path, with_pred: bool) -> None:
    m, g, c = res["meta"], res["groups"], res["continuity"]
    L: List[str] = []
    sec = [0]

    def head(title: str) -> None:
        sec[0] += 1
        L.append("## {}. {}".format(sec[0], title))
        L.append("")

    L.append("# P3-3-FAST Track 2b evaluation")
    L.append("")
    L.append("{} test configs | held-out band a in {} (n={}) vs seen (n={}) | "
             "{} sealed excluded from every fit and aggregate".format(
                 m["n_test_configs"], m["holdout_band_a"], m["n_heldout"], m["n_seen"],
                 m["n_sealed_excluded"]))
    if with_pred:
        L.append("")
        L.append("checkpoint `{}` (iter {}) | in-dist val LSD {} dB".format(
            m.get("checkpoint"), m.get("iter"), _n(m.get("in_dist_val_lsd_db"))))
    else:
        L.append("")
        L.append("**GROUND TRUTH ONLY** -- no model was loaded. Every `pred` column is "
                 "absent by construction; this run validates the harness and reports the "
                 "GT-side observables that the model will be scored against.")
    L.append("")
    L.append("Band {}-{} Hz, df {} Hz. EDC fit window {} dB (inherited from FT-B). "
             "Sub-room mode basis: highest f_max in {} Hz with cond(Phi) <= {}."
             .format(m["band_hz"][0], m["band_hz"][1], m["df_hz"], m["edc_fit_window_db"],
                     m["f_max_ladder_hz"], m["cond_max"]))
    L.append("")

    # -- headline continuity -------------------------------------------------------------
    head("The continuity claim: level difference vs sqrt(a)")
    L.append("sqrt(a) is FT-B's measured linearizing coordinate "
             "(pooled r^2 = {:.4f} on GT in the single 8.0 x 4.0 domain)."
             .format(c["ft_b_reference"]["pooled_r2_sqrt_a"]))
    L.append("")
    L.append("| fit | n | sqrt(a) span | slope (dB per sqrt m) | intercept dB | r | r^2 |")
    L.append("|---|---|---|---|---|---|---|")
    fits = [("GT pooled (6 domains)", c["pooled"]["gt"])]
    for k in GROUP_ORDER:
        fits.append(("GT {}".format(k), c["by_group"][k]["gt"]))
    if with_pred:
        fits.append(("PRED pooled", c["pooled"]["pred"]))
        for k in GROUP_ORDER:
            fits.append(("PRED {}".format(k), c["by_group"][k]["pred"]))
    for name, f in fits:
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            name, f["n"], _n(f["u_range"], "{:.3f}"), _n(f["slope"]), _n(f["intercept"]),
            _n(f["pearson"], "{:.4f}"), _n(f["r2"], "{:.4f}")))
    L.append("")
    L.append("> **Read the held-out row's r^2 with care.** The band is a in [0.9, 1.1], so "
             "sqrt(a) spans only {} against {} for the seen apertures. A regression run "
             "INSIDE the band is near-degenerate by construction -- its r^2 measures "
             "across-domain (L, W, x0) scatter, not the aperture law. The two tests that "
             "actually answer the question are the seen-line residuals and (with a model) "
             "the pred-vs-GT regression, both below."
             .format(_n(c["by_group"][GROUP_HELDOUT]["gt"]["u_range"], "{:.3f}"),
                     _n(c["by_group"][GROUP_SEEN]["gt"]["u_range"], "{:.3f}")))
    L.append("")
    L.append("Per-domain fits (the pooled fit carries (L, W, x0) scatter the aperture law "
             "does not own): GT r^2 {} over {} domains, GT slope {}.".format(
                 _ms(c["per_domain_summary"]["gt_r2"], "{:.4f}"),
                 len(c["per_domain"]), _ms(c["per_domain_summary"]["gt_slope"])))
    L.append("")
    L.append("**Do the held-out points sit on the seen line?** Fit sqrt(a) on the SEEN "
             "apertures only, then score the held-out points against that line.")
    L.append("")
    L.append("| side | seen-line slope | RMS resid seen dB | RMS resid held-out dB | ratio |")
    L.append("|---|---|---|---|---|")
    for k in ("gt", "pred"):
        if k not in c["seen_line"]:
            continue
        s = c["seen_line"][k]
        L.append("| {} | {} | {} | {} | {} |".format(
            k.upper(), _n(s["fit_on_seen"]["slope"]), _n(s["rms_residual_seen_db"]),
            _n(s["rms_residual_heldout_db"]), _n(s["heldout_over_seen_rms"])))
    L.append("")
    for k, label in (("gt", "GROUND TRUTH"), ("pred", "PREDICTION")):
        if k not in c["seen_line"]:
            continue
        r = c["seen_line"][k]["heldout_over_seen_rms"]
        if not np.isfinite(r):
            verdict = "not determinable"
        elif r <= 1.5:
            verdict = ("YES -- the held-out points sit on the seen line (held-out residuals "
                       "are {:.2f}x the seen residuals)".format(r))
        elif r <= 3.0:
            verdict = ("PARTLY -- held-out residuals are {:.2f}x the seen residuals, so the "
                       "band is fit worse than the trained range but not detached".format(r))
        else:
            verdict = ("NO -- held-out residuals are {:.2f}x the seen residuals; the band "
                       "is off the line".format(r))
        L.append("- **{}: {}**".format(label, verdict))
    L.append("")
    if with_pred:
        L.append("| pred vs GT level difference | n | slope | r | r^2 |")
        L.append("|---|---|---|---|---|")
        for k in GROUP_ORDER:
            f = c["by_group"][k]["pred_vs_gt"]
            L.append("| {} | {} | {} | {} | {} |".format(
                k, f["n"], _n(f["slope"]), _n(f["pearson"], "{:.4f}"),
                _n(f["r2"], "{:.4f}")))
        L.append("")

    # -- per-observable table ------------------------------------------------------------
    head("Observables, held-out vs seen")
    L.append("| observable | {} | {} |".format(GROUP_HELDOUT, GROUP_SEEN))
    L.append("|---|---|---|")

    def row(label, fn):
        L.append("| {} | {} | {} |".format(label, fn(g[GROUP_HELDOUT]), fn(g[GROUP_SEEN])))

    row("n configs", lambda b: str(b["n_configs"]))
    row("level difference GT (dB)", lambda b: _ms(b["level_difference"]["gt_db"]))
    row("LD usable-cell frac", lambda b: _ms(b["level_difference"]["frac_usable"], "{:.4f}"))
    if with_pred:
        row("level difference PRED (dB)", lambda b: _ms(b["level_difference"]["pred_db"]))
        row("|LD error| (dB)", lambda b: _ms(b["level_difference"]["abs_error_db"]))
        row("LD pearson pred vs GT",
            lambda b: _n(b["level_difference"]["pearson_pred_vs_gt"], "{:.4f}"))
    row("mode split GT (Hz)", lambda b: _ms(b["mode_split"]["gt_mean_split_hz"]))
    row("mode |migration| GT (Hz)",
        lambda b: _ms(b["mode_split"]["gt_mean_abs_migration_hz"]))
    row("modes / config", lambda b: _ms(b["mode_split"]["n_modes"], "{:.1f}"))
    row("usable modes / config", lambda b: _ms(b["mode_split"]["n_usable_modes"], "{:.1f}"))
    row("frac_modes_dropped", lambda b: _ms(b["mode_split"]["frac_modes_dropped"], "{:.4f}"))
    row("n degenerate modes / config", lambda b: _ms(b["mode_split"]["n_degenerate"],
                                                     "{:.2f}"))
    row("Kuttruff linewidth (Hz)",
        lambda b: _ms(b["mode_split"]["linewidth_kuttruff_median_hz"], "{:.2f}"))
    if with_pred:
        row("mode split PRED (Hz)", lambda b: _ms(b["mode_split"]["pred_mean_split_hz"]))
        row("|split error| (Hz)", lambda b: _ms(b["mode_split"]["mean_abs_split_error_hz"]))
        row("mode |migration| PRED (Hz)",
            lambda b: _ms(b["mode_split"]["pred_mean_abs_migration_hz"]))
        row("|peak position error| (Hz)",
            lambda b: _ms(b["mode_split"]["mean_abs_peak_error_hz"]))
        row("migration pearson", lambda b: _ms(b["mode_split"]["migration_pearson"],
                                               "{:.4f}"))
    row("T60 room A GT (s)", lambda b: _ms(b["decay"]["gt_t60_single_s_roomA"]))
    row("T60 room B GT (s)", lambda b: _ms(b["decay"]["gt_t60_single_s_roomB"]))
    row("room B double-slope (GT)", lambda b: "{}/{}".format(
        b["decay"]["gt_n_double_slope_roomB"], b["decay"]["gt_n_valid_roomB"]))
    if with_pred:
        row("T60 room A PRED (s)", lambda b: _ms(b["decay"]["pred_t60_single_s_roomA"]))
        row("T60 room B PRED (s)", lambda b: _ms(b["decay"]["pred_t60_single_s_roomB"]))
        row("room B double-slope (PRED)", lambda b: str(
            b["decay"]["pred_n_double_slope_roomB"]))
        for k in ("all", "room_A", "room_B"):
            row("LSD {} (dB)".format(k), lambda b, k=k: _ms(b["lsd"][k]["lsd_db"]))
            row("  GT dynamic range {} (dB)".format(k),
                lambda b, k=k: _ms(b["lsd"][k]["gt_dynamic_range_db"], "{:.1f}"))
            row("  LSD / dynamic range {}".format(k),
                lambda b, k=k: _ms(b["lsd"][k]["lsd_over_dynamic_range"], "{:.4f}"))
            row("  usable-cell frac {}".format(k),
                lambda b, k=k: _ms(b["lsd"][k]["frac_usable"], "{:.4f}"))
    L.append("")

    # -- third octave --------------------------------------------------------------------
    if with_pred:
        head("Level difference per ISO third-octave band")
        L.append("| fc Hz | GT held-out | PRED held-out | |err| | GT seen | PRED seen "
                 "| |err| | usable frac |")
        L.append("|---|---|---|---|---|---|---|---|")
        bh = g[GROUP_HELDOUT]["level_difference"]["by_third_octave"]
        bs = g[GROUP_SEEN]["level_difference"]["by_third_octave"]
        for h, s in zip(bh, bs):
            L.append("| {:.0f} | {} | {} | {} | {} | {} | {} | {} |".format(
                h["fc_hz"], _n(h["gt_db"]["mean"], "{:.2f}"),
                _n(h["pred_db"]["mean"], "{:.2f}"), _n(h["abs_error_db"]["mean"], "{:.2f}"),
                _n(s["gt_db"]["mean"], "{:.2f}"), _n(s["pred_db"]["mean"], "{:.2f}"),
                _n(s["abs_error_db"]["mean"], "{:.2f}"),
                _n(h["frac_usable"]["mean"], "{:.3f}")))
        L.append("")

    # -- topological reference -----------------------------------------------------------
    t = res["topological_reference"]
    head("Sealed (a = 0): topological reference, NOT a point on the curve")
    L.append(t["why_separate"])
    L.append("")
    L.append("- sealed configs: {} | GT room-B field identically zero in all of them: {}"
             .format(t["n_configs"], t["gt_room_b_energy_is_zero"]))
    L.append("- GT level difference: -inf in every sealed config (room B is disconnected)")
    if with_pred:
        L.append("- PRED level difference: {} dB -- finite by construction. The model's "
                 "coordinate is sqrt(a), and sqrt(0) = 0 is also the limit of a vanishing "
                 "doorway, so this number is the size of the gap between the continuous law "
                 "and the topological truth, not an error the model could have avoided."
                 .format(_ms(t["pred_ld_db"], "{:.2f}")))
        L.append("- room-A LSD on sealed configs: {} dB (room-B LSD is undefined -- the "
                 "target is identically zero)".format(_ms(t["lsd_room_A_db"], "{:.2f}")))
    L.append("")

    # -- caveats -------------------------------------------------------------------------
    head("Caveats that condition every number above")
    L.append("1. **Mode split is poorly conditioned in near-square sub-rooms.** When a "
             "sub-room's (1,0) and (0,1) fall within one linewidth of each other (FT-B "
             "measured 0.107 Hz apart inside a ~3.07 Hz linewidth in its 8.0 x 4.0 domain), "
             "the peak-search window contains a DIFFERENT mode rather than the split partner "
             "and the observable is unidentifiable. Such modes are flagged `degenerate` "
             "per mode and counted per config (`n_degenerate`); they are excluded from the "
             "split statistics but the count is reported so the exclusion is visible. The "
             "other three observables are unaffected.")
    L.append("2. **The two-peak split is resolution-limited; the peak POSITION is not.** "
             "Resolving a doublet needs a separation of order the linewidth ({} Hz median "
             "here, and FT-B measured that the splitting only exceeds the linewidth for "
             "a >= 1.66 m), so `split_hz` is legitimately 0 across most of the aperture "
             "range. FT-B could see sub-linewidth splitting because its divider sat at "
             "x0 = L/2 and the two sub-rooms were exact mirror images, which let it "
             "decompose the field into even/odd branches BEFORE peak fitting. Track 2b "
             "varies x0, the sub-rooms are not congruent, and that decomposition does not "
             "exist -- so the well-conditioned modal observable here is the peak MIGRATION "
             "(`migration_*_hz`, `mean_abs_peak_error_hz`), and `split_hz` is reported "
             "beside it rather than leaned on."
             .format(_ms(g[GROUP_SEEN]["mode_split"]["linewidth_kuttruff_median_hz"],
                         "{:.2f}")))
    L.append("3. **Absolute LSD is not comparable to earlier ISM chunks.** {}"
             .format(m["lsd_scoping"]))
    L.append("4. **Every table is conditioned on a resolvable subset.** `frac_modes_dropped` "
             "and `frac_usable` sit next to the numbers they gate; a mode whose window is "
             "narrower than 3 bins, or a cell below the {:.0e} log floor, never enters a "
             "statistic.".format(m["eps_usable"]))
    L.append("5. **The 8x8 receiver grid leaves 3-5 columns per sub-room**, so the sub-room "
             "mode basis blows up (cond ~1e16) above ~160 Hz. The f_max ladder backs off "
             "until cond(Phi) <= {}; the chosen f_max and cond are recorded per sub-room in "
             "`per_config[*].modal.bases`.".format(m["cond_max"]))
    L.append("6. **The record is 2.0 s and T60 is longer**, so the EDC below about -30 dB is "
             "backward-integration truncation rather than decay. The fit window stops at "
             "{} dB for exactly that reason (FT-B's finding, inherited unchanged)."
             .format(m["edc_fit_window_db"][1]))
    L.append("")
    path.write_text("\n".join(L) + "\n")
    print("[wrote] {}".format(path), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="P3-3-FAST Track 2b aperture evaluation")
    ap.add_argument("--train-dir", default=TRAIN_DIR)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--checkpoint", default=None,
                    help="default: newest ckpt_iter*.pt in --train-dir")
    ap.add_argument("--gt-only", action="store_true",
                    help="ground-truth observables only; runs without a GPU")
    ap.add_argument("--rx-chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N test configs")
    a = ap.parse_args()
    res = run(a.train_dir, a.manifest, a.data_dir, a.out, a.checkpoint, a.gt_only,
              a.rx_chunk, a.limit)
    c = res["continuity"]
    print("[continuity] GT sqrt(a): pooled r^2 {} | held-out r^2 {} (n={}) | "
          "seen r^2 {} (n={})".format(
              _n(c["pooled"]["gt"]["r2"], "{:.4f}"),
              _n(c["by_group"][GROUP_HELDOUT]["gt"]["r2"], "{:.4f}"),
              c["by_group"][GROUP_HELDOUT]["gt"]["n"],
              _n(c["by_group"][GROUP_SEEN]["gt"]["r2"], "{:.4f}"),
              c["by_group"][GROUP_SEEN]["gt"]["n"]), flush=True)


if __name__ == "__main__":
    main()
