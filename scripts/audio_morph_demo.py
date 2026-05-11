"""V3: length-morphing audio demo (Chunk 3.7).

For three unseen L values, render the predicted H(f) at the centre receiver
from the C2_latent_jitter + B6 zero-shot run, IRFFT to a time-domain RIR, and
convolve with a synthetic source (impulse + 3 sinusoids at modal-ish
frequencies). Write WAV files demonstrating smooth length-morphing.

Quality gate: if the rendered audio's peak-to-median-amplitude ratio is < 3
(signal is swamped by noise), write an ``audio_SKIPPED.txt`` marker and
emit no WAVs — V4 will then omit V3 from the deck.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from aaf.data.dataset_builder import read_room_h5, room_filename


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LS = (3.25, 4.25, 5.75)
DEFAULT_TONE_HZ = (80.0, 120.0, 180.0)
DEFAULT_TONE_AMP = (0.3, 0.2, 0.1)


def _source(duration_s: float, fs: int, tones_hz, tones_amp, impulse_amp: float = 1.0):
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    x = np.zeros(n, dtype=np.float64)
    x[0] = impulse_amp
    for f, a in zip(tones_hz, tones_amp):
        x += a * np.sin(2 * math.pi * f * t)
    return x / max(np.abs(x).max(), 1e-12)


def _centre_idx(L: float, W: float, n_grid: int = 8, margin: float = 0.5) -> int:
    xs = np.linspace(margin, L - margin, n_grid)
    ys = np.linspace(margin, W - margin, n_grid)
    iy = int(np.argmin(np.abs(ys - W / 2.0)))
    ix = int(np.argmin(np.abs(xs - L / 2.0)))
    return iy * n_grid + ix


def _rir_from_Hpred(H_pred: np.ndarray, centre: int, n_time: int) -> np.ndarray:
    """Inverse-rfft the centre receiver's predicted complex spectrum to a real
    time-domain impulse response of length n_time."""
    H_one = H_pred[centre]                                # [n_freq] complex
    h = np.fft.irfft(H_one, n=n_time).astype(np.float64)
    return h


def _convolve_norm(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear convolve and normalise to peak 0.9."""
    y = np.convolve(x, h, mode="full")
    peak = float(np.abs(y).max())
    if peak < 1e-12:
        return y
    return (0.9 / peak) * y


def _write_wav(path: Path, y: np.ndarray, fs: int):
    y16 = np.clip(y, -1.0, 1.0)
    y16 = (y16 * 32767).astype(np.int16)
    wavfile.write(str(path), fs, y16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="C2_latent_jitter")
    ap.add_argument("--inner_loop", type=str, default="B6")
    ap.add_argument("--Ls", nargs="+", type=float, default=list(DEFAULT_LS))
    ap.add_argument("--sweep_root", type=str,
                    default=str(REPO_ROOT / "outputs/multi_room/sweep"))
    ap.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "data/track_a"))
    ap.add_argument("--out_dir", type=str,
                    default=str(REPO_ROOT / "outputs/meeting_assets/07_audio_demo"))
    ap.add_argument("--duration_s", type=float, default=1.0)
    ap.add_argument("--snr_ratio_min", type=float, default=3.0,
                    help="minimum |peak| / median(|y|) ratio; below this triggers SKIP")
    args = ap.parse_args()

    run_dir = Path(args.sweep_root) / args.run
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_meta = json.loads((run_dir / "train_meta.json").read_text())
    cfg = train_meta["cfg"]
    fs = int(cfg["fs"])
    n_time = int(cfg["n_time_samples"])

    sub = "zero_shot" if args.inner_loop == "B1" else f"zero_shot_{args.inner_loop}"
    summary_rows = []
    skip_reasons = []
    for L in args.Ls:
        H_pred_path = run_dir / sub / f"L{L}" / "H_pred_all.pt"
        h5_path = Path(args.data_dir) / room_filename(L=L, W=4.0, alpha=0.15)
        if not H_pred_path.exists():
            print(f"# skipping L={L}: missing {H_pred_path}")
            continue
        H_pred = torch.load(H_pred_path, map_location="cpu").detach().cpu().numpy()
        rt = read_room_h5(h5_path)
        W = float(rt["attrs"].get("W", 4.0))
        centre = _centre_idx(L=L, W=W)
        h = _rir_from_Hpred(H_pred, centre, n_time)
        x = _source(args.duration_s, fs, DEFAULT_TONE_HZ, DEFAULT_TONE_AMP)
        y = _convolve_norm(x, h)
        peak = float(np.abs(y).max())
        median = float(np.median(np.abs(y)) + 1e-12)
        ratio = peak / median
        wav_path = out_dir / f"morph_L{L:.2f}.wav"
        if ratio < args.snr_ratio_min:
            skip_reasons.append(f"L={L:.2f}: peak/median ratio {ratio:.2f} < "
                                f"{args.snr_ratio_min}")
        else:
            _write_wav(wav_path, y, fs)
            print(f"# wrote {wav_path}  (peak/median={ratio:.2f})")
        summary_rows.append((L, peak, median, ratio, str(wav_path) if ratio >= args.snr_ratio_min else "SKIPPED"))

    if skip_reasons:
        # If any L produced bad audio, skip the whole demo per the chunk spec.
        (out_dir / "audio_SKIPPED.txt").write_text(
            "Audio demo skipped — at least one L had peak/median below the "
            f"{args.snr_ratio_min}× threshold:\n  " + "\n  ".join(skip_reasons)
        )
        # Remove any partially-written WAVs so the deck doesn't pick them up.
        for p in out_dir.glob("morph_L*.wav"):
            p.unlink()
        print(f"# AUDIO SKIPPED: {len(skip_reasons)} L(s) below SNR threshold")
    else:
        readme = [
            f"# V3 length-morphing audio demo ({args.run} + {args.inner_loop})",
            "",
            "Source: 1.0-sec, fs=4096, x(t) = impulse + 0.3·sin(2π·80·t) + "
            "0.2·sin(2π·120·t) + 0.1·sin(2π·180·t).",
            "",
            "For each unseen L the predicted RIR at the centre receiver was "
            "produced by inverse-rfft of the saved `H_pred_all.pt`, then "
            "convolved with the source and peak-normalised. Files:",
            "",
        ]
        for L, peak, median, ratio, path in summary_rows:
            readme.append(f"- `morph_L{L:.2f}.wav` — peak/median = {ratio:.2f}")
        readme.append("")
        readme.append("Quality caveat: full-band held-LSD ≈ 5 dB on these "
                      "models. The audio is a qualitative demo, not a faithful "
                      "RIR — it'll be audibly imperfect. We're shipping it "
                      "anyway to demonstrate smooth latent morphing across L.")
        (out_dir / "README.md").write_text("\n".join(readme) + "\n")
        print(f"# wrote {out_dir/'README.md'}")


if __name__ == "__main__":
    main()
