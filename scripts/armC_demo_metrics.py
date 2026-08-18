"""Arm C demo pack, stage 1: the metrics, computed BEFORE any figure is drawn.

The demo claim is that one INR renders many unseen scenarios zero-shot, including dense spatial
fields queried far off the 8x8 grid it trained on. That claim has an explicit kill switch: if
pointwise spatial Pearson between prediction and ISM ground truth comes out below ~0.7, the
spatial part of the demo is not supportable and must be reported rather than illustrated. So
this script computes and prints the whole table first; figure assembly is a separate stage that
reads its JSON. Ordering the work this way is what makes the abort rule real instead of
decorative.

That risk is not hypothetical. The FDTD-corpus model scored 0.24-0.60 on exactly this metric.
Arm C is expected to do far better -- it trained on the clean ISM corpus, reaches ~1.01 dB
in-distribution, and P3-2b already measured a mode-shape score of 0.9921 for it -- but that
score is `mode_shape_invariance`, i.e. agreement with the analytic cosine shape on the 8x8
training grid. It is NOT pointwise agreement with ground truth on a 64x64 grid, and the two are
reported separately here rather than conflated.

Three scenario notes worth carrying into the captions:
  * `north@0.70` is inside Arm C's held-out slab north (1.13, 1.28) in m, so scenario (c) is
    doubly zero-shot -- unseen geometry AND an unseen wall/material combination.
  * scenario (d) (east 0.50 + south 0.70) is not one of P3-2b's four frozen two-wall test
    combos, which is harmless because every ground truth here is re-simulated at 64x64 anyway.
  * ground truth keeps the corpus protocol (fs 4096, n 8192, max_order 60) so it is the same
    physics the model was trained against; its truncation warning is the frozen protocol's own
    behaviour, not a new defect.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import yaml

from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.p3_2_eval import band_limit, load_model
from aaf.eval.p3_2b_eval import render_config_arm
from aaf.sim.ism_2d import simulate_room_2d
from aaf.walls import WALL_INDEX, WALLS_2D

BAND_HI_HZ = 300.0
FS, N_TIME = 4096.0, 8192
MAX_ORDER = 60
SRC = (0.5, 0.5)
DENSE = 64
N_MODES_PLOT = 3
PEARSON_ABORT = 0.70

SCENARIOS = [
    ("a_baseline", {}, "baseline, all walls 0.15"),
    ("b_east_curtain", {"east": 0.50}, "east curtain 0.50 (trained combo)"),
    ("c_north_absorber", {"north": 0.70}, "north absorber 0.70 (HELD-OUT combo)"),
    ("d_two_wall", {"east": 0.50, "south": 0.70}, "east 0.50 + south 0.70"),
]


def alphas_for(edits: Dict[str, float]) -> List[float]:
    a = [0.15] * len(WALLS_2D)
    for w, v in edits.items():
        a[WALL_INDEX[w]] = float(v)
    return a


def dense_grid(L: float, W: float, n: int = DENSE, margin: float = 0.15) -> np.ndarray:
    """n x n receivers spanning the room. Training used 8x8 at 0.3 m margin, so this is both
    denser and closer to the walls -- the continuity claim."""
    xs = np.linspace(margin, L - margin, n)
    ys = np.linspace(margin, W - margin, n)
    return np.array([[x, y] for x in xs for y in ys], dtype=float)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _circ_corr(pa: np.ndarray, pb: np.ndarray) -> float:
    """Circular correlation of two phase fields (radians)."""
    a = np.asarray(pa, float).ravel()
    b = np.asarray(pb, float).ravel()
    a = a - np.angle(np.mean(np.exp(1j * a)))
    b = b - np.angle(np.mean(np.exp(1j * b)))
    num = np.sum(np.sin(a) * np.sin(b))
    den = np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2))
    return float(num / den) if den > 0 else float("nan")


def _lsd_db(pred: np.ndarray, gt: np.ndarray) -> float:
    p = 20.0 * np.log10(np.maximum(np.abs(pred), 1e-30))
    g = 20.0 * np.log10(np.maximum(np.abs(gt), 1e-30))
    return float(np.mean(np.abs(p - g)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="outputs/p3_2/p3_2b_C_cont_mlinear/ckpt_iter0060000.pt")
    ap.add_argument("--test-yaml", default="configs/sweeps_2d_mat/p3_2_test_frozen.yaml")
    ap.add_argument("--out", default="outputs/armC_demo/metrics.json")
    ap.add_argument("--npz-dir", default="outputs/armC_demo/fields")
    ap.add_argument("--dense", type=int, default=DENSE)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer, cfg, meta, it = load_model(Path(a.checkpoint), dev)
    model.eval()
    renderer.eval()                    # required as well as model.eval() -- D49 C3
    cond_source = str(cfg["cond_source"])
    print("[arm] {} iter {} | cond {} ({}d) | device {}".format(
        Path(a.checkpoint).parent.name, it, cond_source, cfg["cond_dim"], dev), flush=True)

    geoms = [(round(float(g["L"]), 2), round(float(g["W"]), 2))
             for g in yaml.safe_load(Path(a.test_yaml).read_text())["geometries"]]
    order = np.argsort([L * W for L, W in geoms])
    picked = [("small", geoms[order[0]]), ("median", geoms[order[len(order) // 2]]),
              ("large", geoms[order[-1]])]
    print("[geoms] " + " | ".join("{} {}x{}".format(t, L, W) for t, (L, W) in picked))

    df = FS / N_TIME
    hi = int(round(BAND_HI_HZ / df)) + 1
    npz_dir = Path(a.npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []

    for tag, (L, W) in picked:
        rx = dense_grid(L, W, a.dense)
        modes = [m for m in enumerate_modes(L, W, f_max=BAND_HI_HZ)
                 if not (m.n_x == 0 and m.n_y == 0)][:N_MODES_PLOT]
        mode_bins = [int(round(m.f / df)) for m in modes]
        centre = int(np.argmin(np.linalg.norm(rx - np.array([L / 2, W / 2]), axis=1)))

        for name, edits, desc in SCENARIOS:
            al = alphas_for(edits)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")     # corpus protocol truncation warning
                gt_out = simulate_room_2d(dict(
                    L=L, W=W, source_pos=np.asarray(SRC, float), receiver_pos=rx,
                    alphas=tuple(al), fs=FS, n_time_samples=N_TIME, max_order=MAX_ORDER))
            gt = band_limit(np.asarray(gt_out["H_complex"])[:, :hi], hi)
            with torch.no_grad():
                pr = np.asarray(render_config_arm(model, renderer, cond_source, L, W, al,
                                                  rx, np.asarray(SRC, float), dev))
            pr = band_limit(pr[:, :hi], hi)

            rec = {"geometry": tag, "L": L, "W": W, "scenario": name, "desc": desc,
                   "alphas": al, "n_rx": int(rx.shape[0]),
                   "held_out_combo": bool(name == "c_north_absorber"),
                   "modes": [], "spatial_pearson": []}
            for m, b in zip(modes, mode_bins):
                gm, pm = np.abs(gt[:, b]), np.abs(pr[:, b])
                sp = _pearson(20 * np.log10(np.maximum(pm, 1e-30)),
                              20 * np.log10(np.maximum(gm, 1e-30)))
                rec["modes"].append({"mode": [m.n_x, m.n_y], "f_hz": float(m.f), "bin": b,
                                     "spatial_pearson_db": sp,
                                     "spatial_pearson_lin": _pearson(pm, gm)})
                rec["spatial_pearson"].append(sp)
            rec["spatial_pearson_mean"] = float(np.nanmean(rec["spatial_pearson"]))
            rec["magnitude_corr"] = _pearson(np.abs(pr), np.abs(gt))
            rec["band_lsd_db"] = _lsd_db(pr[centre], gt[centre])
            rec["band_lsd_db_allrx"] = _lsd_db(pr, gt)
            rec["phase_circ_corr"] = _circ_corr(np.angle(pr), np.angle(gt))
            rir_p = np.fft.irfft(pr[centre], n=N_TIME)
            rir_g = np.fft.irfft(gt[centre], n=N_TIME)
            rec["rir_pearson"] = _pearson(rir_p, rir_g)
            rows.append(rec)

            np.savez_compressed(npz_dir / "{}_{}.npz".format(tag, name),
                                pred=pr.astype(np.complex64), gt=gt.astype(np.complex64),
                                rx=rx, mode_bins=np.array(mode_bins),
                                mode_f=np.array([m.f for m in modes]),
                                mode_nx=np.array([m.n_x for m in modes]),
                                mode_ny=np.array([m.n_y for m in modes]),
                                centre=centre, L=L, W=W, alphas=np.array(al))
            print("  {:7s} {:18s} spatialR {:+.3f} | magR {:+.3f} | LSD {:5.2f} dB | "
                  "phaseR {:+.3f} | rirR {:+.3f}".format(
                      tag, name, rec["spatial_pearson_mean"], rec["magnitude_corr"],
                      rec["band_lsd_db"], rec["phase_circ_corr"], rec["rir_pearson"]),
                  flush=True)

    allsp = np.array([r["spatial_pearson_mean"] for r in rows], float)
    worst, mean = float(np.nanmin(allsp)), float(np.nanmean(allsp))
    ok = bool(worst >= PEARSON_ABORT)
    out = {
        "checkpoint": a.checkpoint, "iter": int(it), "cond_source": cond_source,
        "dense_grid": a.dense, "training_grid": 8, "n_rx_dense": int(a.dense ** 2),
        "band_hz": [0.0, BAND_HI_HZ], "protocol": {"fs": FS, "n": N_TIME,
                                                   "max_order": MAX_ORDER, "src": list(SRC)},
        "abort_threshold_spatial_pearson": PEARSON_ABORT,
        "spatial_pearson_worst": worst, "spatial_pearson_mean": mean,
        "proceed_to_figures": ok,
        "prior_note": ("P3-2b recorded mode_shape_invariance 0.9921 (pred) for this arm on the "
                       "8x8 grid; that is agreement with the ANALYTIC cosine shape, NOT the "
                       "pointwise pred-vs-GT Pearson reported here. Distinct quantities."),
        "scenarios": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("\nspatial Pearson: worst {:.3f}, mean {:.3f} (abort below {:.2f}) -> {}".format(
        worst, mean, PEARSON_ABORT, "PROCEED" if ok else "STOP AND REPORT"))
    print("-> {}".format(a.out))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
