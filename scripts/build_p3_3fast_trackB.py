"""Build the Track 2b (doorway aperture) FDTD dataset.

Writes the EXISTING HDF5 schema, including the ``ism/H_complex`` key, even though the solver
is FDTD and there is no image-source model anywhere near this data. That key name is now
misleading and the choice is deliberate (D56): every loader, the trainer and the whole P3-2b
eval battery read that path, so reusing it means zero changes anywhere downstream. A
``solver`` attr records the truth. Two attrs are NEW relative to Track A -- ``x0`` and ``a``
-- because the divider is not expressible in the 4-alpha vector.

**dx and fs are coupled.** The aperture axis needs dx = 0.01 (FT-1b A0c measured the aperture
observable moving 10.4x the estimator floor between dx = 0.02 and dx = 0.01, i.e. dx = 0.02 is
inside the un-converged regime). fs MUST scale with 1/dx or the CFL condition fails outright:
at dx = 0.01 a fixed fs = 12288 raises a ValueError from the solver. fs = 61440 with
n = 122880 holds lambda at the frozen 0.55827 and keeps T = 2.000 s, df = 0.5 Hz.

Idempotent via ``.done`` sentinels. The worklist is the FULL stable list -- filtering on
``.done`` here would shrink it as the build progresses and race the array index against the
config mapping, which silently left 79 of 479 P3-2c configs unbuilt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.aperture_configs import configs_from_rows

DX = 0.01
#: fs MUST scale with 1/dx (see module docstring). Holds lambda at the frozen 0.55827.
FS, N = 61440.0, 122880          # T = 2.000 s, df = 0.5 Hz
C = 343.0
BAND_HI_HZ = 300.0
SRC = (0.5, 0.5)                 # always in room A: x0 >= 0.4 * 7.0 = 2.8 m
MANIFEST = "configs/sweeps_2d_mat/p3_3fast_trackB_manifest.json"


def receivers(L, W, x0, n_side=8, margin=0.3, dx=DX):
    """8x8 grid with a 0.3 m margin spanning the FULL domain (both sub-rooms).

    Any receiver whose x-node lands on or beside the divider column is pushed two nodes clear
    of it, away from the divider. Without that, a receiver snapping onto the slab makes
    ``simulate`` raise ("snaps onto a solid node") for the sealed and small-aperture configs
    while succeeding for the wide ones. The nudge depends only on (L, W, x0), so all 20
    configs of a domain share one receiver array regardless of ``a``.
    """
    dxx = float(L) / int(round(float(L) / dx))
    i_div = int(round(float(x0) / dxx))
    xs = []
    for x in np.linspace(margin, L - margin, n_side):
        i = int(round(float(x) / dxx))
        if abs(i - i_div) <= 1:
            i = i_div + 2 if i >= i_div else i_div - 2
            x = i * dxx
        xs.append(float(x))
    ys = np.linspace(margin, W - margin, n_side)
    return np.array([[x, y] for x in xs for y in ys])


def build_one(cfg, out_dir: Path, force=False):
    out_path = out_dir / cfg.filename
    done = out_dir / (cfg.filename + ".done")
    if done.exists() and out_path.exists() and not force:
        return "skip"
    rx = receivers(cfg.L, cfg.W, cfg.x0)
    res = F.simulate(cfg.L, cfg.W, cfg.alphas, src=SRC, rx=rx,
                     dx=DX, fs=FS, n=N, c=C, extra_walls=cfg.extra_walls)
    hi = int(round(BAND_HI_HZ / (FS / N))) + 1
    H = np.asarray(res["H_complex"])[:, :hi].astype(np.complex64)
    specs = res["meta"]["extra_walls"]
    if not specs:
        a_real = float(cfg.W)
    elif specs[0].get("apertures"):
        a_real = float(specs[0]["apertures"][0]["clear_width_m"])
    else:
        a_real = 0.0
    with h5py.File(out_path, "w") as f:
        g = f.create_group("ism")            # legacy key, FDTD data -- see module docstring
        g.create_dataset("H_complex", data=H, compression="gzip", compression_opts=4)
        f.attrs["source_pos"] = json.dumps(list(SRC))
        f.attrs["receiver_pos"] = json.dumps(np.asarray(
            res["meta"]["rx_pos_snapped"], float).tolist())
        f.attrs["split"] = cfg.split
        f.attrs["alphas"] = json.dumps(list(cfg.alphas))
        f.attrs["L"] = float(cfg.L)
        f.attrs["W"] = float(cfg.W)
        f.attrs["x0"] = float(cfg.x0)
        f.attrs["a"] = float(cfg.a)
        f.attrs["a_realized"] = a_real
        f.attrs["sealed"] = bool(cfg.sealed)
        f.attrs["fully_open"] = bool(cfg.fully_open)
        f.attrs["kind"] = cfg.kind
        f.attrs["solver"] = "fdtd_2d_slf_kw"
        f.attrs["dx"] = DX
        f.attrs["fs"] = FS
        f.attrs["n_time_samples"] = N
        f.attrs["band_hi_hz"] = BAND_HI_HZ
        f.attrs["n_freq_bins"] = int(H.shape[1])
        f.attrs["extra_walls"] = json.dumps(specs)
    done.touch()
    return "built"


def worklist(manifest=MANIFEST, data_dir="data/track_p3_3fast_B"):
    """FULL stable list, deduplicated by filename and sorted. NEVER filtered on ``.done``."""
    rows = json.load(open(manifest))["configs"]
    cfgs = configs_from_rows(rows)
    seen, out = set(), []
    for c in cfgs:
        if c.filename in seen:
            continue
        seen.add(c.filename)
        out.append(c)
    out.sort(key=lambda c: c.filename)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--out-dir", default="data/track_p3_3fast_B")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    work = worklist(data_dir=a.out_dir)
    d = Path(a.out_dir)
    if a.plan or a.idx is None:
        pend = [c for c in work if not (d / (c.filename + ".done")).exists()]
        n_tasks = (len(work) + a.chunk - 1) // a.chunk
        print(json.dumps({"n_total": len(work), "n_pending": len(pend),
                          "chunk": a.chunk, "array_range": "0-{}".format(n_tasks - 1)}, indent=1))
        return
    d.mkdir(parents=True, exist_ok=True)
    lo = a.idx * a.chunk
    for c in work[lo:lo + a.chunk]:
        print(build_one(c, d), c.filename, flush=True)


if __name__ == "__main__":
    main()
