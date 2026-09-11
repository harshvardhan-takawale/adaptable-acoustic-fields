"""P4-1 Stage 1 evaluation: can the architecture represent a non-convex room at all?

GATE 1: NLOS spatial Pearson >= 0.70 AND a LOS-NLOS gap <= 0.15. The verdict is computed and
printed BEFORE any figure is drawn (the D62a ordering rule) -- a figure must not exist for an
unemitted verdict.

The split is the whole question, so nothing is pooled: every metric is reported separately for
receivers the source reaches directly and receivers it can only reach by diffracting around the
reflex corner.

WHAT THE sigma PROFILE ACTUALLY TESTS
-------------------------------------
The chunk spec frames Stage 1 as "the renderer integrates along source->receiver rays and that
path crosses solid wall". It does not. `FreqRenderer2D` fans 64 rays OUTWARD FROM THE RECEIVER
to the room AABB and feeds the source in as a network input; no ray is traced from tx and
|rx - tx| is never computed. Two consequences shape this evaluation:

  * Transmittance accumulates ONLY along the receiver->point leg, so sigma can express an
    occluder between the RECEIVER and a sample point. A wall between the SOURCE and that point
    has no structural representation at all -- it can only be absorbed into signal(pts, tx).
  * The geometric phase uses d = |rx - pt|, so a diffracted arrival whose true path is LONGER
    than |rx - pt| can only be emitted from a point whose receiver-distance equals the true
    path length -- an image-source-like trick the model would have to discover unaided.

So the sigma profile is probed along a ray FROM A RECEIVER THROUGH THE NOTCH: that is the one
direction in which sigma is structurally capable of representing the wall. If sigma does not
rise inside the notch there, the representation has not found the occluder even where it could.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.p3_2_eval import load_model
from aaf.models.conditioning_2d import build_cond_vector_2d
from scripts.build_p4_1_lroom import NOTCH_X, NOTCH_Y, VERTS, inside_polygon

GATE_NLOS_PEARSON = 0.70
GATE_MAX_GAP = 0.15
N_MODES = 6
DF_HZ = 0.5


def _pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def _lsd(p, t):
    return float(np.mean(np.abs(_db(p) - _db(t))))


def sigma_profile(model, cond, rx0, direction, src, dev, n=400, reach=6.5):
    """sigma(f) sampled along one ray from `rx0`, at the model's own band centre.

    Queries the model directly rather than through the renderer, because the renderer returns
    only the integrated H and the question here is what the FIELD looks like along the path.
    """
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    t = np.linspace(1e-3, reach, n)
    pts = np.asarray(rx0, float)[None, :] + t[:, None] * d[None, :]
    P = torch.tensor(pts, dtype=torch.float32, device=dev).unsqueeze(0)
    V = torch.tensor(np.repeat(d[None, :], n, axis=0), dtype=torch.float32,
                     device=dev).unsqueeze(0)
    TX = torch.tensor(np.repeat(np.asarray(src, float)[None, :], n, axis=0),
                      dtype=torch.float32, device=dev).unsqueeze(0)
    z = cond.unsqueeze(0).expand(1, -1)
    with torch.no_grad():
        attn, _ = model(P, V, TX, tx_view=None, z_s=z)
        sig = attn.real.clamp(min=0)[0].mean(dim=-1).cpu().numpy()   # mean over frequency
    return t, pts, sig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/p4_1/stage1/p4_1_stage1_lroom")
    ap.add_argument("--data", default="data/track_p4_1_lroom/lroom.h5")
    ap.add_argument("--out", default="outputs/p4_1/stage1")
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()

    ck = Path(a.checkpoint) if a.checkpoint else sorted(
        Path(a.run_dir).glob("ckpt_iter*.pt"))[-1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, renderer, cfg, meta, it = load_model(ck, dev)
    model.eval()
    renderer.eval()                       # D49 C3
    print("[ckpt] {} iter {} | cond {} | world_scale {}".format(
        ck.name, it, cfg["cond_source"], cfg.get("world_scale")), flush=True)

    with h5py.File(a.data) as f:
        T = np.asarray(f["ism/H_complex"])
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        src = np.asarray(json.loads(f.attrs["source_pos"]))
        los = np.asarray(json.loads(f.attrs["line_of_sight"]), dtype=bool)
        L, W = float(f.attrs["L"]), float(f.attrs["W"])
    print("[data] {} receivers | LOS {} | NLOS {} ({:.1%})".format(
        len(rx), int(los.sum()), int((~los).sum()), 1 - los.mean()), flush=True)

    cond = build_cond_vector_2d("geom_token", L, W, [0.15] * len(VERTS),
                                verts=VERTS, device=dev)
    room_min = torch.zeros(2, device=dev)
    room_max = torch.tensor([L, W], device=dev, dtype=torch.float32)
    P = np.zeros_like(T)
    R = torch.tensor(rx, dtype=torch.float32, device=dev)
    S = torch.tensor(src, dtype=torch.float32, device=dev).unsqueeze(0)
    for s0 in range(0, len(rx), 8):
        sl = slice(s0, s0 + 8)
        r = R[sl]
        z = cond.unsqueeze(0).expand(r.shape[0], -1)
        with torch.no_grad():
            H = renderer(model, r, S.expand(r.shape[0], -1), room_min, room_max, z_s=z)
        P[sl] = H[:, :T.shape[1]].cpu().numpy()

    # --- TRAINED vs HELD-OUT receivers ----------------------------------------------------
    # Stage 1 is a per-scene FIT, so the trainer supervised 1-in-8-complement of these
    # receivers. Scoring all of them answers the question the spec actually asks ("can the
    # representation fit one such room at all") -- but 87.5% of them were trained on, so the
    # headline number is a FIT metric and would be badly misread as generalization. Both are
    # reported; the gate is on the fit metric, per the spec.
    from aaf.train.multi_room_2d_mat import val_rx_indices
    held = np.zeros(len(rx), dtype=bool)
    held[list(val_rx_indices(len(rx)))] = True
    print("[split] trained on {} receivers | held out {}".format(
        int((~held).sum()), int(held.sum())), flush=True)

    # --- per-mode spatial Pearson, split by line of sight ---------------------------------
    modes = [m for m in enumerate_modes(L, W, f_max=200.0)][:N_MODES]
    rows = []
    for m in modes:
        b = int(round(m.f / DF_HZ))
        if b >= T.shape[1]:
            continue
        pm, tm = _db(P[:, b]), _db(T[:, b])
        rows.append({"mode": [m.n_x, m.n_y], "f_hz": float(m.f), "bin": b,
                     "pearson_all": _pearson(pm, tm),
                     "pearson_los": _pearson(pm[los], tm[los]),
                     "pearson_nlos": _pearson(pm[~los], tm[~los]),
                     "pearson_los_heldout": _pearson(pm[los & held], tm[los & held]),
                     "pearson_nlos_heldout": _pearson(pm[~los & held], tm[~los & held]),
                     # Pearson is affine-invariant, so a correct SHAPE with a wrong LEVEL still
                     # scores 1.0. The residual level error is recorded so a high R cannot be
                     # read as a perfect fit without checking it.
                     "mean_abs_db_err": float(np.mean(np.abs(pm - tm))),
                     "level_offset_db": float(np.mean(pm - tm))})
    r_los = float(np.nanmean([r["pearson_los"] for r in rows]))
    r_nlos = float(np.nanmean([r["pearson_nlos"] for r in rows]))
    r_los_h = float(np.nanmean([r["pearson_los_heldout"] for r in rows]))
    r_nlos_h = float(np.nanmean([r["pearson_nlos_heldout"] for r in rows]))
    lsd_los, lsd_nlos = _lsd(P[los], T[los]), _lsd(P[~los], T[~los])
    lsd_los_h, lsd_nlos_h = _lsd(P[los & held], T[los & held]), _lsd(P[~los & held], T[~los & held])
    lsd_trained, lsd_held = _lsd(P[~held], T[~held]), _lsd(P[held], T[held])
    gap = r_los - r_nlos
    gap_h = r_los_h - r_nlos_h

    print("\n  mode      f_Hz |  R_all   R_LOS  R_NLOS")
    for r in rows:
        print("  ({},{})  {:7.2f} | {:+.3f}  {:+.3f}  {:+.3f}".format(
            r["mode"][0], r["mode"][1], r["f_hz"], r["pearson_all"],
            r["pearson_los"], r["pearson_nlos"]))
    print("\n  mean spatial Pearson   LOS {:+.3f}   NLOS {:+.3f}   gap {:+.3f}".format(
        r_los, r_nlos, gap))
    print("  band LSD               LOS {:5.2f} dB  NLOS {:5.2f} dB".format(lsd_los, lsd_nlos))
    print("\n  --- the same metrics on HELD-OUT receivers only ({} of {}) ---".format(
        int(held.sum()), len(rx)))
    print("  mean spatial Pearson   LOS {:+.3f}   NLOS {:+.3f}   gap {:+.3f}".format(
        r_los_h, r_nlos_h, gap_h))
    print("  band LSD               LOS {:5.2f} dB  NLOS {:5.2f} dB".format(lsd_los_h, lsd_nlos_h))
    print("  band LSD  trained {:.2f} dB  vs  held-out {:.2f} dB  ({:.1f}x)".format(
        lsd_trained, lsd_held, lsd_held / max(lsd_trained, 1e-9)))

    passed = bool(r_nlos >= GATE_NLOS_PEARSON and gap <= GATE_MAX_GAP)
    crit = {"nlos_pearson": {"value": r_nlos, "op": ">=", "threshold": GATE_NLOS_PEARSON,
                             "pass": bool(r_nlos >= GATE_NLOS_PEARSON)},
            "los_nlos_gap": {"value": gap, "op": "<=", "threshold": GATE_MAX_GAP,
                             "pass": bool(gap <= GATE_MAX_GAP)}}
    print("\n  GATE 1: {}".format("PASS" if passed else "FAIL"))
    for k, v in crit.items():
        print("    {:16s} {:+.3f} {} {:.2f} -> {}".format(
            k, v["value"], v["op"], v["threshold"], "pass" if v["pass"] else "FAIL"))

    # --- sigma along a ray from an NLOS receiver THROUGH the notch ------------------------
    # Pick the NLOS receiver whose straight line to the source passes deepest through solid.
    cand = rx[~los]
    depth = []
    for p in cand:
        t = np.linspace(0, 1, 200)[:, None]
        seg = p[None, :] + t * (src[None, :] - p[None, :])
        depth.append(float((~inside_polygon(seg)).mean()))
    rx0 = cand[int(np.argmax(depth))]
    t_ax, pts, sig = sigma_profile(model, cond, rx0, src - rx0, src, dev)
    in_solid = ~inside_polygon(pts)
    sig_solid = float(np.mean(sig[in_solid])) if in_solid.any() else float("nan")
    sig_air = float(np.mean(sig[~in_solid]))
    ratio = sig_solid / sig_air if sig_air > 0 else float("nan")
    print("\n  sigma along a ray from NLOS rx {} toward the source:".format(np.round(rx0, 2)))
    print("    mean sigma inside the notch {:.4g} | in air {:.4g} | ratio {:.2f}".format(
        sig_solid, sig_air, ratio))
    print("    (ratio > 1 = the model placed attenuation where the wall is)")

    out = {
        "gate": "p4_1.stage1/1", "checkpoint": str(ck), "iter": int(it),
        "cond_source": cfg["cond_source"], "world_scale": cfg.get("world_scale"),
        "n_rx": int(len(rx)), "n_los": int(los.sum()), "n_nlos": int((~los).sum()),
        "nlos_fraction": float(1 - los.mean()),
        "per_mode": rows,
        "mean_pearson_los": r_los, "mean_pearson_nlos": r_nlos, "los_nlos_gap": gap,
        "band_lsd_los_db": lsd_los, "band_lsd_nlos_db": lsd_nlos,
        "heldout": {"n_trained": int((~held).sum()), "n_heldout": int(held.sum()),
                    "mean_pearson_los": r_los_h, "mean_pearson_nlos": r_nlos_h,
                    "los_nlos_gap": gap_h,
                    "band_lsd_los_db": lsd_los_h, "band_lsd_nlos_db": lsd_nlos_h,
                    "band_lsd_trained_db": lsd_trained, "band_lsd_heldout_db": lsd_held,
                    "note": ("Stage 1 is a per-scene FIT and 87.5% of receivers were "
                             "supervised. The gate is on the all-receiver metric, which is what "
                             "the spec asks for, but that is a FIT number -- these held-out "
                             "values are the generalization one.")},
        "sigma_probe": {"rx": rx0.tolist(), "mean_sigma_solid": sig_solid,
                        "mean_sigma_air": sig_air, "solid_over_air": ratio,
                        "n_samples_in_solid": int(in_solid.sum())},
        "criteria": crit, "passed": passed,
        "thresholds": {"nlos_pearson_min": GATE_NLOS_PEARSON, "max_gap": GATE_MAX_GAP},
        "renderer_note": ("FreqRenderer2D fans rays outward FROM THE RECEIVER; there is no "
                          "source->receiver ray. sigma can occlude the receiver->point leg "
                          "only, so the sigma probe is run along that direction."),
    }
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.out) / "GATE1.json", "w"), indent=1, default=float)
    np.savez_compressed(Path(a.out) / "stage1_fields.npz", pred=P.astype(np.complex64),
                        gt=T.astype(np.complex64), rx=rx, los=los,
                        sigma_t=t_ax, sigma_pts=pts, sigma=sig, sigma_rx=rx0,
                        mode_bins=np.array([r["bin"] for r in rows]),
                        mode_f=np.array([r["f_hz"] for r in rows]))
    print("\n-> {}".format(Path(a.out) / "GATE1.json"))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
