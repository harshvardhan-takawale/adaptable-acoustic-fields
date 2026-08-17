"""Build the Track 1 (per-segment absorption + window) FDTD dataset.

Writes the EXISTING HDF5 schema, including the ``ism/H_complex`` key, even though the solver
is FDTD. That key name is now misleading and the choice is deliberate (D56): every loader, the
trainer and the whole P3-2b eval battery read that path, so reusing it means zero changes
anywhere downstream. A ``solver`` attr records the truth, and the results doc states the
simulator split explicitly.

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
from aaf.data.seg_configs import configs_from_rows, seg_alphas_to_wall_specs

DX = 0.02
FS, N = 30720.0, 61440          # holds lambda at the frozen 0.55827; T=2.000 s, df=0.5 Hz
C = 343.0
BAND_HI_HZ = 300.0
MANIFEST = "configs/sweeps_2d_mat/p3_3fast_trackA_manifest.json"


def receivers(L, W, n_side=8, margin=0.3):
    xs = np.linspace(margin, L - margin, n_side)
    ys = np.linspace(margin, W - margin, n_side)
    return np.array([[x, y] for x in xs for y in ys])


def build_one(cfg, out_dir: Path, force=False):
    out_path = out_dir / cfg.filename
    done = out_dir / (cfg.filename + ".done")
    if done.exists() and out_path.exists() and not force:
        return "skip"
    rx = receivers(cfg.L, cfg.W)
    res = F.simulate(cfg.L, cfg.W, (0.15,) * 4, src=(0.5, 0.5), rx=rx,
                     dx=DX, fs=FS, n=N, c=C,
                     extra_walls=seg_alphas_to_wall_specs(cfg.alphas))
    hi = int(round(BAND_HI_HZ / (FS / N))) + 1
    H = np.asarray(res["H_complex"])[:, :hi].astype(np.complex64)
    with h5py.File(out_path, "w") as f:
        g = f.create_group("ism")            # legacy key, FDTD data -- see module docstring
        g.create_dataset("H_complex", data=H, compression="gzip", compression_opts=4)
        f.attrs["source_pos"] = json.dumps([0.5, 0.5])
        f.attrs["receiver_pos"] = json.dumps(np.asarray(
            res["meta"]["rx_pos_snapped"], float).tolist())
        f.attrs["split"] = cfg.split
        f.attrs["alphas"] = json.dumps(list(cfg.alphas))
        f.attrs["L"] = float(cfg.L)
        f.attrs["W"] = float(cfg.W)
        f.attrs["kind"] = cfg.kind
        f.attrs["solver"] = "fdtd_2d_slf_kw"
        f.attrs["dx"] = DX
        f.attrs["fs"] = FS
        f.attrs["n_time_samples"] = N
        f.attrs["band_hi_hz"] = BAND_HI_HZ
        f.attrs["n_freq_bins"] = int(H.shape[1])
        f.attrs["segments_tile"] = json.dumps(
            [s.get("tiles_exactly", False) for s in res["meta"]["extra_walls"]])
    done.touch()
    return "built"


def worklist(manifest=MANIFEST, data_dir="data/track_p3_3fast_A"):
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
    ap.add_argument("--chunk", type=int, default=13)
    ap.add_argument("--out-dir", default="data/track_p3_3fast_A")
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
