"""P4-7 prep p3 (D164): does the geometry encoder react to SEGMENT COUNT rather than PHYSICAL SIZE?

D111's dense U room (`UL5.00_W5.26_NEd0.14w0.48x0.00_NWd1.22w1.86x0.00`, shape_id 8002) and the same room
with its 14 cm NE notch REMOVED (an L with the same NW notch: 8 -> 6 boundary tokens). Both are scored at
the U's own 3680 dense receivers (all on-grid and in air in the L; the L's solid mask is a strict subset
of the U's), off the 0.12 + 0.06 k lattice (Rule 11).

  ground truth : dT = mean |dB T_L - dB T_U|   (FDTD, `build_p4_5_rooms._simulate`)
  model        : dP = mean |dB P_L - dB P_U|   (PB3 / B5 @60K, `p4_4_family_eval._render` with each ring)
at 50 / 100 / 200 Hz (single bins 100 / 200 / 400, and +-2.5 Hz means), plus the spatial Pearson of the
change fields (does the model change in the RIGHT places) and the same excluding receivers within
0.5 m of the removed notch. Verdict (D164): segment-count sensitivity if dP/dT >= 3 at >= 2 of the 3
frequencies, for either model.

  python scripts/p4_7_geometry_sensitivity.py --simulate      # CPU: FDTD of the notch-removed room
  python scripts/p4_7_geometry_sensitivity.py --analyse       # GPU: renders + metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.build_p4_5_rooms import OUT as ROOT, family_test_rooms
from scripts.p4_5_gate2_dense import on_lattice

DATA = Path("data/track_p4_7_p3")
OUT = Path("outputs/p4_7")
FREQS = {"50 Hz": 100, "100 Hz": 200, "200 Hz": 400}
HW = 5                                        # +-2.5 Hz
NOTCH_BOX = (4.52, 5.12, 5.0, 5.26)           # the removed NE notch: x0, y0, x1, y1
MODELS = {"PB3_60K": "outputs/p4_5/arms/PB3/ckpt_iter0060000.pt",
          "B5_60K": "outputs/p4_5/arms/B5/ckpt_iter0060000.pt"}


def u_room():
    rooms = family_test_rooms()
    j = [k for k, (fam, _c) in enumerate(rooms) if fam == "U"][0]
    return j, rooms[j][1]


def l_cfg(U):
    from aaf.data.multi_notch import FamilyConfig, Notch
    nw = [n for n in U.notches if n.side == "NW"]
    assert len(nw) == 1 and len(U.notches) == 2
    return FamilyConfig(L=U.L, W=U.W, notches=(Notch("NW", d=nw[0].d, w=nw[0].w, x0=0.0),),
                        split="test", shape_id=U.shape_id)


def load_u():
    _j, U = u_room()
    with h5py.File(Path(ROOT["famdense"]) / U.filename) as f:
        T = np.asarray(f["ism/H_complex"])[:, :601]
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool)
    return U, T, rx, keep


def simulate():
    import aaf.sim.fdtd_2d as F
    from aaf.data.multi_notch import solid_blocks
    from aaf.data.shape_configs import ALPHA, SRC
    from aaf.sim.polygon_geom import line_of_sight, polygon_contains
    from scripts.build_p4_5_rooms import _simulate
    U, _T, rx, keep = load_u()
    cfg = l_cfg(U)
    dx = 0.02
    nx, dxx = F._fit_axis(cfg.L, dx, "L")
    ny, dxy = F._fit_axis(cfg.W, dx, "W")
    m = solid_blocks(nx, ny, dxx, dxy, cfg.L, cfg.W, cfg.notches)
    extra = [{"type": "mask", "solid": m, "alpha": ALPHA}]
    geom = F.build_geometry(cfg.L, cfg.W, [ALPHA] * 4, dx=dx, extra_walls=extra)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    assert (geom.air == polygon_contains(cfg.verts, np.stack([ii * dxx, jj * dxy], -1))).all()
    H, rx_snap, _res, fs, n, loop_s = _simulate(cfg, rx, dx, geom, extra)
    assert np.abs(rx_snap - rx).max() < 1e-9
    los = line_of_sight(cfg.verts, SRC, rx_snap)
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / cfg.filename
    with h5py.File(path, "w") as f:
        f.create_dataset("ism/H_complex", data=H, compression="gzip", compression_opts=4)
        a = f.attrs
        a["source_pos"] = json.dumps(list(SRC))
        a["receiver_pos"] = json.dumps(rx_snap.tolist())
        a["line_of_sight"] = json.dumps([bool(x) for x in los])
        a["lattice_keep"] = json.dumps([bool(x) for x in keep])       # the U's own lattice
        a["label"], a["kind"] = cfg.label, cfg.kind
        a["verts"] = json.dumps([list(v) for v in cfg.verts])
        a["L"], a["W"], a["n_rx"], a["loop_s"], a["dx"], a["fs"], a["n_time_samples"] = (
            cfg.L, cfg.W, int(len(rx)), round(loop_s, 1), dx, fs, n)
        a["derived_from"] = U.filename + " with the NE notch removed (P4-7 prep p3, D164)"
    print("simulated", path, H.shape, "{:.0f}s".format(loop_s), flush=True)


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def change(A, B, sel, b, hw=0):
    """mean |dB A - dB B| over receivers `sel` at bin b (or the +-hw bin mean of per-bin values)."""
    bins = range(b - hw, b + hw + 1)
    return float(np.mean([np.mean(np.abs(db(A[sel, k]) - db(B[sel, k]))) for k in bins]))


def corr_change(dA, dB):
    a, b = dA - dA.mean(), dB - dB.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def analyse():
    import torch

    from aaf.eval.p3_2_eval import load_model
    from scripts.p4_4_family_eval import _render
    j, _U0 = u_room()
    U, T_U, rx, _keep = load_u()
    cfg = l_cfg(U)
    with h5py.File(DATA / cfg.filename) as f:
        T_L = np.asarray(f["ism/H_complex"])[:, :601]
        assert np.allclose(np.asarray(json.loads(f.attrs["receiver_pos"])), rx)
    off = ~on_lattice(rx)
    x, y = rx[:, 0], rx[:, 1]
    x0, y0, x1, y1 = NOTCH_BOX
    dist = np.hypot(np.clip(x0 - x, 0, None) + np.clip(x - x1, 0, None),
                    np.clip(y0 - y, 0, None) + np.clip(y - y1, 0, None))
    far = off & (dist > 0.5)
    dev = torch.device("cuda")
    rep = {"room_U": U.label, "room_L": cfg.label, "n_off": int(off.sum()), "n_off_far": int(far.sum()),
           "tokens": {"U": len(U.verts), "L": len(cfg.verts)}, "freqs": {}, "verdict": {}}
    preds = {}
    for name, ck in MODELS.items():
        z = np.load("outputs/p4_6/cache/famdense_{}.npz".format(name))
        assert np.allclose(z["rx%d" % j], rx), "famdense cache receiver order differs"
        model, renderer, _cfg, _m, _it = load_model(Path(ck), dev)
        model.eval(); renderer.eval()
        P_L = _render(model, renderer, cfg, rx, 601, dev)[:, :601]
        preds[name] = (z["P%d" % j][:, :601], P_L)
        del model, renderer
        torch.cuda.empty_cache()
    for lab, b in FREQS.items():
        row = {}
        for sel_name, sel in (("off", off), ("off_far_from_notch", far)):
            dT = change(T_L, T_U, sel, b); dTw = change(T_L, T_U, sel, b, HW)
            r = {"dT_db": dT, "dT_db_pm2.5": dTw}
            for name, (P_U, P_L) in preds.items():
                dP = change(P_L, P_U, sel, b); dPw = change(P_L, P_U, sel, b, HW)
                fT = db(T_L[sel, b]) - db(T_U[sel, b]); fP = db(P_L[sel, b]) - db(P_U[sel, b])
                r[name] = {"dP_db": dP, "dP_db_pm2.5": dPw, "ratio": dP / dT, "ratio_pm2.5": dPw / dTw,
                           "change_field_r": corr_change(fP, fT)}
            row[sel_name] = r
        rep["freqs"][lab] = row
        print(lab, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else
                                        {a: round(c, 3) for a, c in vv.items()})
                                   for kk, vv in v.items()} for k, v in row.items()}), flush=True)
    for name in MODELS:
        ratios = [rep["freqs"][lab]["off"][name]["ratio"] for lab in FREQS]
        rep["verdict"][name] = {"ratios_off": ratios, "n_ge_3": int(sum(r >= 3 for r in ratios)),
                                "segment_count_sensitive": bool(sum(r >= 3 for r in ratios) >= 2)}
    rep["verdict"]["any_model"] = any(v["segment_count_sensitive"] for k, v in rep["verdict"].items() if k in MODELS)
    rep["confound"] = "the L version has d_hat 0.515 (inside C250's held-out slab) and L < W"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p3_sensitivity.json").write_text(json.dumps(rep, indent=1))
    print("VERDICT", json.dumps(rep["verdict"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--simulate", action="store_true")
    g.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    simulate() if a.simulate else analyse()
