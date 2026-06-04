"""Signal-level evaluation suite (Dolby-requested).

3-layer factoring (mirrors `aaf.eval.band_limited`):

  Layer 1 — pure component functions
    magnitude_correlation, phase_correlation_mag_weighted,
    per_band_lsd, rir_pearson, edc_db, edc_error, early_late_corr,
    envelope_corr

  Layer 2 — aggregator
    compute_signal_metrics(H_pred, H_target, fs, n_time_samples, ...)
      returns a flat dict of metric values.

  Layer 3 — plotting
    make_signal_plots(H_pred, H_target, fs, n_time_samples, output_dir, ...)
      writes the 5 figures the eval suite emits.

Phase 2's stable eval surface. P2-2 and beyond reuse this for zero-shot eval;
do not break the public API without updating callers.

Conventions
-----------
- All ``H_*`` arrays are RFFT-layout complex (last dim = n_freq_bins).
- All ``rir_*`` arrays are time-domain real float (last dim = n_time_samples).
- Per-receiver metrics are averaged across receivers (axis 0). Single-
  receiver inputs are accepted via the [F] / [T] 1-D shapes.
- Eps for log-magnitude: 1e-8 (consistent with band_limited.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.signal import hilbert

from aaf.eval.band_limited import band_indices, compute_band_limited_metrics


DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
)
EPS = 1e-8


def _as_2d(arr: np.ndarray) -> np.ndarray:
    """Promote a [F] or [T] 1-D array to [1, F] / [1, T] for uniform handling."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D input, got shape {arr.shape}")
    return arr


def _ensure_rir(
    H: np.ndarray, rir: Optional[np.ndarray], n_time_samples: int
) -> np.ndarray:
    """Return the time-domain RIR matching ``H``; compute via irfft if absent."""
    if rir is not None:
        rir = _as_2d(rir).astype(np.float32, copy=False)
        if rir.shape[-1] != n_time_samples:
            raise ValueError(
                f"rir last dim {rir.shape[-1]} != n_time_samples {n_time_samples}"
            )
        return rir
    return np.fft.irfft(_as_2d(H), n=n_time_samples, axis=-1).astype(np.float32)


# ----------------------------------------------------------------------
# Layer 1: pure component functions
# ----------------------------------------------------------------------


def _pearson_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two [N, K] real arrays. Returns [N]."""
    A = A - A.mean(axis=-1, keepdims=True)
    B = B - B.mean(axis=-1, keepdims=True)
    num = (A * B).sum(axis=-1)
    den = np.sqrt((A * A).sum(axis=-1) * (B * B).sum(axis=-1)) + EPS
    return num / den


def magnitude_correlation(H_pred: np.ndarray, H_target: np.ndarray) -> float:
    """Pearson correlation of |H_pred(f)| vs |H_target(f)| across freq, per
    receiver, then averaged over receivers.

    Both inputs ``[N_rx, F]`` (or ``[F]``).
    """
    Hp = _as_2d(H_pred)
    Ht = _as_2d(H_target)
    if Hp.shape != Ht.shape:
        raise ValueError(f"H_pred {Hp.shape} vs H_target {Ht.shape}")
    rho = _pearson_rows(np.abs(Hp), np.abs(Ht))
    return float(np.mean(rho))


def phase_correlation_mag_weighted(
    H_pred: np.ndarray, H_target: np.ndarray
) -> float:
    """Magnitude-weighted phase agreement.

    Avoids penalizing phase error where |H_target| is negligible (phase is
    meaningless near zero). Formula:

      score = mean_rx [ Σ_f (|H_target| · cos(∠H_pred - ∠H_target)) /
                        Σ_f |H_target| ]

    Range: [-1, 1]; 1.0 = perfect phase agreement weighted by where there's
    actually signal.
    """
    Hp = _as_2d(H_pred)
    Ht = _as_2d(H_target)
    if Hp.shape != Ht.shape:
        raise ValueError(f"H_pred {Hp.shape} vs H_target {Ht.shape}")
    cos_d = np.cos(np.angle(Hp) - np.angle(Ht))
    w = np.abs(Ht)
    num = np.sum(w * cos_d, axis=-1)
    den = np.sum(w, axis=-1) + EPS
    return float(np.mean(num / den))


def per_band_lsd(
    H_pred: np.ndarray,
    H_target: np.ndarray,
    fs: float,
    n_freq_bins: int,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
) -> dict:
    """Per-band LSD via the existing ``compute_band_limited_metrics``.

    Returns a dict of ``{"lsd_band_<lo>_<hi>_db": float, ...}``.
    """
    return compute_band_limited_metrics(
        _as_2d(H_pred), _as_2d(H_target), fs=fs,
        n_freq_bins=n_freq_bins, bands=bands,
    )


def rir_pearson(rir_pred: np.ndarray, rir_target: np.ndarray) -> float:
    """Pearson correlation between predicted and target RIR in the time
    domain, per receiver, then averaged."""
    A = _as_2d(rir_pred).astype(np.float64)
    B = _as_2d(rir_target).astype(np.float64)
    if A.shape != B.shape:
        raise ValueError(f"rir_pred {A.shape} vs rir_target {B.shape}")
    return float(np.mean(_pearson_rows(A, B)))


def edc_db(rir: np.ndarray, fs: float) -> np.ndarray:
    """Schroeder integration → energy decay curve in dB.

    EDC(t) = 10·log10( ∫_t^∞ |rir|² dτ / ∫_0^∞ |rir|² dτ )

    Returns array of same shape as ``rir`` (each row a per-receiver EDC).
    """
    r = _as_2d(rir).astype(np.float64)
    p2 = r * r
    # Reverse cumulative integration (energy remaining from index i onward).
    rev = np.cumsum(p2[..., ::-1], axis=-1)[..., ::-1]
    total = rev[..., :1] + EPS
    edc = 10.0 * np.log10(rev / total + EPS)
    return edc.astype(np.float32)


def _t_reverse(rir: np.ndarray, fs: float, drop_db: float, ref_db: float = -5.0) -> float:
    """Backward-integration ``T_drop`` estimate (seconds) extrapolated to -60 dB.

    ``T20`` corresponds to drop_db=20 (slope from -5 to -25 dB → scale by 3).
    ``T30`` corresponds to drop_db=30 (slope from -5 to -35 dB → scale by 2).
    Returns ``nan`` if the curve never reaches the lower bound.
    """
    edc = edc_db(rir, fs)                          # [N, T]
    n_rx, T = edc.shape
    t_axis = np.arange(T) / fs
    out = np.empty(n_rx, dtype=np.float64)
    for i in range(n_rx):
        e = edc[i]
        upper = ref_db
        lower = ref_db - drop_db
        # Find first crossing below `upper` and `lower`.
        i_up = np.argmax(e <= upper)
        if e[i_up] > upper:
            out[i] = np.nan
            continue
        i_lo = np.argmax(e <= lower)
        if e[i_lo] > lower:
            out[i] = np.nan
            continue
        # Slope from upper to lower; extrapolate to -60 dB.
        dt = t_axis[i_lo] - t_axis[i_up]
        if dt <= 0:
            out[i] = np.nan
            continue
        slope_db_per_s = (lower - upper) / dt        # negative
        t60 = (-60.0 - upper) / slope_db_per_s + t_axis[i_up]
        # Anchored at t_axis[i_up], so subtract that to give "time to -60 dB".
        out[i] = t60 - t_axis[i_up]
    return float(np.nanmean(out))


def edc_error(
    rir_pred: np.ndarray, rir_target: np.ndarray, fs: float
) -> dict:
    """Compare EDCs of predicted vs target RIRs.

    Returns ``{
        "edc_max_db": max absolute EDC deviation (dB),
        "edc_rmse_db": rms EDC deviation (dB),
        "t20_delta_s": mean(|T20_pred - T20_target|) seconds,
        "t30_delta_s": mean(|T30_pred - T30_target|) seconds,
    }``.
    """
    e_p = edc_db(rir_pred, fs)
    e_t = edc_db(rir_target, fs)
    # Clamp to a sensible dynamic range so floor noise doesn't dominate.
    floor = -80.0
    e_p = np.maximum(e_p, floor)
    e_t = np.maximum(e_t, floor)
    diff = e_p - e_t
    edc_max = float(np.mean(np.max(np.abs(diff), axis=-1)))
    edc_rmse = float(np.sqrt(np.mean(diff * diff)))
    try:
        t20_p = _t_reverse(rir_pred, fs, drop_db=20.0)
        t20_t = _t_reverse(rir_target, fs, drop_db=20.0)
        t30_p = _t_reverse(rir_pred, fs, drop_db=30.0)
        t30_t = _t_reverse(rir_target, fs, drop_db=30.0)
        t20_d = float(abs(t20_p - t20_t)) if np.isfinite(t20_p) and np.isfinite(t20_t) else float("nan")
        t30_d = float(abs(t30_p - t30_t)) if np.isfinite(t30_p) and np.isfinite(t30_t) else float("nan")
    except Exception:
        t20_d = t30_d = float("nan")
    return {
        "edc_max_db": edc_max,
        "edc_rmse_db": edc_rmse,
        "t20_delta_s": t20_d,
        "t30_delta_s": t30_d,
    }


def early_late_corr(
    rir_pred: np.ndarray,
    rir_target: np.ndarray,
    fs: float,
    split_ms: float = 50.0,
) -> tuple[float, float]:
    """Split each RIR at ``split_ms`` and compute Pearson on the early
    (direct + first reflections) and late (reverberant tail) halves
    separately. Returns (early_corr, late_corr).
    """
    A = _as_2d(rir_pred).astype(np.float64)
    B = _as_2d(rir_target).astype(np.float64)
    if A.shape != B.shape:
        raise ValueError(f"rir_pred {A.shape} vs rir_target {B.shape}")
    split = int(round(split_ms / 1000.0 * fs))
    split = max(1, min(split, A.shape[-1] - 1))
    early = float(np.mean(_pearson_rows(A[:, :split], B[:, :split])))
    late = float(np.mean(_pearson_rows(A[:, split:], B[:, split:])))
    return early, late


def envelope_corr(rir_pred: np.ndarray, rir_target: np.ndarray) -> float:
    """Hilbert-envelope Pearson correlation, per receiver, averaged.

    Envelope = |analytic_signal| = |hilbert(rir)|. Smooths out the fine
    structure so this metric captures "do the RIRs decay the same way" while
    rir_pearson captures "are the reflections at the same times".
    """
    A = _as_2d(rir_pred).astype(np.float64)
    B = _as_2d(rir_target).astype(np.float64)
    if A.shape != B.shape:
        raise ValueError(f"rir_pred {A.shape} vs rir_target {B.shape}")
    env_p = np.abs(hilbert(A, axis=-1))
    env_t = np.abs(hilbert(B, axis=-1))
    return float(np.mean(_pearson_rows(env_p, env_t)))


# ----------------------------------------------------------------------
# Layer 2: aggregator
# ----------------------------------------------------------------------


def compute_signal_metrics(
    H_pred: np.ndarray,
    H_target: np.ndarray,
    fs: float,
    n_time_samples: int,
    *,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
    early_late_split_ms: float = 50.0,
    rir_pred: Optional[np.ndarray] = None,
    rir_target: Optional[np.ndarray] = None,
) -> dict:
    """One-call wrapper returning a flat dict of all signal-level metrics.

    Args:
        H_pred, H_target: complex arrays of shape ``[N_rx, F]`` or ``[F]``.
        fs: time-domain sample rate (Hz).
        n_time_samples: implies ``n_freq_bins = n_time_samples // 2 + 1``.
        bands: per-band LSD bands; defaults to (0-250, 250-500, 500-1000,
            1000-2000) Hz.
        early_late_split_ms: split for early/late RIR Pearson.
        rir_pred, rir_target: optional pre-computed time-domain RIRs. If
            None, irfft is computed internally.

    Returns:
        flat dict with keys:
          - mag_corr (float)
          - phase_corr_mw (float)
          - lsd_band_<lo>_<hi>_db (one per band)
          - lsd_band_<lo>_<hi>_n_bins (one per band)
          - rir_pearson (float)
          - edc_max_db, edc_rmse_db, t20_delta_s, t30_delta_s
          - early_corr, late_corr
          - envelope_corr
    """
    Hp = _as_2d(H_pred)
    Ht = _as_2d(H_target)
    n_freq_bins = Hp.shape[-1]
    if n_freq_bins != n_time_samples // 2 + 1:
        raise ValueError(
            f"n_freq_bins {n_freq_bins} != n_time_samples // 2 + 1 "
            f"({n_time_samples // 2 + 1})"
        )

    rp = _ensure_rir(Hp, rir_pred, n_time_samples)
    rt = _ensure_rir(Ht, rir_target, n_time_samples)

    out: dict = {}
    out["mag_corr"] = magnitude_correlation(Hp, Ht)
    out["phase_corr_mw"] = phase_correlation_mag_weighted(Hp, Ht)
    out.update(per_band_lsd(Hp, Ht, fs=fs, n_freq_bins=n_freq_bins, bands=bands))
    out["rir_pearson"] = rir_pearson(rp, rt)
    out.update(edc_error(rp, rt, fs=fs))
    early, late = early_late_corr(rp, rt, fs=fs, split_ms=early_late_split_ms)
    out["early_corr"] = early
    out["late_corr"] = late
    out["envelope_corr"] = envelope_corr(rp, rt)
    return out


# ----------------------------------------------------------------------
# Layer 3: plotting
# ----------------------------------------------------------------------


def make_signal_plots(
    H_pred: np.ndarray,
    H_target: np.ndarray,
    fs: float,
    n_time_samples: int,
    output_dir: str | Path,
    *,
    representative_rx_idx: Optional[int] = None,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
    f_max_plot: float = 2000.0,
    rir_pred: Optional[np.ndarray] = None,
    rir_target: Optional[np.ndarray] = None,
) -> dict[str, Path]:
    """Write the 5 signal-level figures to ``output_dir``.

    Files:
      - magnitude_overlay.png
      - phase_overlay.png
      - rir_time_overlay.png
      - edc_overlay.png
      - signal_metrics_summary.png

    Returns ``{name: Path}`` of the written files.
    """
    import matplotlib.pyplot as plt   # local import — keeps the module GPU-only-free

    Hp = _as_2d(H_pred)
    Ht = _as_2d(H_target)
    rp = _ensure_rir(Hp, rir_pred, n_time_samples)
    rt = _ensure_rir(Ht, rir_target, n_time_samples)

    n_rx = Hp.shape[0]
    if representative_rx_idx is None:
        # Pick the receiver with the largest |H_target| total energy — a
        # "loud" receiver where errors are most visible.
        rx_idx = int(np.argmax(np.sum(np.abs(Ht) ** 2, axis=-1)))
    else:
        rx_idx = int(representative_rx_idx) % n_rx

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_freq_bins = Hp.shape[-1]
    f_axis = np.arange(n_freq_bins) * (fs / n_time_samples)
    t_axis = np.arange(n_time_samples) / fs

    out: dict[str, Path] = {}

    # 1. magnitude_overlay
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(f_axis, np.abs(Ht[rx_idx]) + EPS, label="target (ISM)",
                color="steelblue", lw=1.0)
    ax.semilogy(f_axis, np.abs(Hp[rx_idx]) + EPS, label="predicted",
                color="tab:red", lw=0.8, alpha=0.85)
    ax.set_xlim(0, f_max_plot)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|H(f)|")
    ax.set_title(f"Magnitude overlay (rx={rx_idx})")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    p = output_dir / "magnitude_overlay.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out["magnitude_overlay"] = p

    # 2. phase_overlay (with magnitude above for context)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                              gridspec_kw={"height_ratios": [1, 2]})
    axes[0].semilogy(f_axis, np.abs(Ht[rx_idx]) + EPS, color="steelblue", lw=0.8)
    axes[0].set_ylabel("|H_target|")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].set_xlim(0, f_max_plot)
    axes[1].plot(f_axis, np.angle(Ht[rx_idx]), color="steelblue", lw=0.6,
                 label="target (ISM)")
    axes[1].plot(f_axis, np.angle(Hp[rx_idx]), color="tab:red", lw=0.5,
                 alpha=0.85, label="predicted")
    axes[1].set_xlim(0, f_max_plot)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("∠H(f) (rad)")
    axes[1].set_ylim(-np.pi - 0.2, np.pi + 0.2)
    axes[1].legend(loc="upper right", frameon=False)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Phase overlay (rx={rx_idx}; phase meaningful where |H| is large)")
    fig.tight_layout()
    p = output_dir / "phase_overlay.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out["phase_overlay"] = p

    # 3. rir_time_overlay (full + 0-50 ms zoom)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t_axis * 1000, rt[rx_idx], color="steelblue",
                 lw=0.6, label="target (ISM)")
    axes[0].plot(t_axis * 1000, rp[rx_idx], color="tab:red", lw=0.5,
                 alpha=0.7, label="predicted")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("RIR amplitude")
    axes[0].set_title(f"Full RIR overlay (rx={rx_idx})")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(True, alpha=0.3)
    # Zoom to first 50 ms
    n_zoom = int(round(0.05 * fs))
    axes[1].plot(t_axis[:n_zoom] * 1000, rt[rx_idx, :n_zoom],
                 color="steelblue", lw=1.0, label="target (ISM)")
    axes[1].plot(t_axis[:n_zoom] * 1000, rp[rx_idx, :n_zoom],
                 color="tab:red", lw=0.8, alpha=0.85, label="predicted")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("RIR amplitude")
    axes[1].set_title("First 50 ms (early reflections)")
    axes[1].legend(loc="upper right", frameon=False)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p = output_dir / "rir_time_overlay.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out["rir_time_overlay"] = p

    # 4. edc_overlay
    edc_p = edc_db(rp, fs)
    edc_t = edc_db(rt, fs)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_axis * 1000, edc_t[rx_idx], color="steelblue",
            lw=1.0, label="target (ISM)")
    ax.plot(t_axis * 1000, edc_p[rx_idx], color="tab:red",
            lw=0.8, alpha=0.85, label="predicted")
    ax.axhline(-60.0, color="gray", ls="--", lw=0.7, label="-60 dB")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("EDC (dB)")
    ax.set_title(f"Schroeder energy decay curve (rx={rx_idx})")
    ax.set_ylim(-80, 5)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = output_dir / "edc_overlay.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out["edc_overlay"] = p

    # 5. signal_metrics_summary — bar chart of metrics
    metrics = compute_signal_metrics(
        Hp, Ht, fs=fs, n_time_samples=n_time_samples,
        bands=bands, rir_pred=rp, rir_target=rt,
    )
    # Group into two clusters for readability.
    corr_keys = [
        "mag_corr", "phase_corr_mw", "rir_pearson",
        "early_corr", "late_corr", "envelope_corr",
    ]
    band_keys = [k for k in metrics if k.startswith("lsd_band_") and k.endswith("_db")]
    band_labels = [k.replace("lsd_band_", "").replace("_db", "").replace("_", "-") for k in band_keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Correlations panel
    vals = [metrics[k] for k in corr_keys]
    axes[0].bar(corr_keys, vals, color="steelblue")
    axes[0].axhline(1.0, color="gray", ls="--", lw=0.7, label="perfect (1.0)")
    axes[0].set_ylim(-0.2, 1.05)
    axes[0].set_ylabel("Pearson / cos-correlation")
    axes[0].set_title("Correlation metrics")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(frameon=False)
    # Per-band LSD panel
    vals = [metrics[k] for k in band_keys]
    axes[1].bar(band_labels, vals, color="tab:orange")
    axes[1].set_ylabel("LSD (dB)")
    axes[1].set_title("Per-band LSD")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = output_dir / "signal_metrics_summary.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out["signal_metrics_summary"] = p

    return out
