"""P4-9 0a (D180; D181 A3, A4, E5): latency by edit type -- network (R0 @120K) vs physics with cached modes vs a modal
re-solve to 600 Hz vs FDTD.

--gpu (one named GPU; fp32 = the shipped dtype, D181 A4): R0 @120K, eval mode, the CURRENT path (4097 bins computed)
      and the 601-BIN path (D181 c4; parity-tested in tests/test_p4_9_render.py), on shape 1000 (test index 0) and on the
      largest of the 31 dense rooms (most receivers):
        listener move   render 1 receiver (single-listener update rate = 1 / this) and the room's 64x64 grid
        source move     the same renders with another source (no re-encode: the source is a render input)
        material change rebuild the tokens with alpha 0.30 on every edge, then render (LATENCY ONLY: the network was
                        never trained on a material change, so its accuracy there is untested)
        geometry change move one notch vertex (the reflex corner) by 0.3 m, rebuild the tokens, render the grid
      plus the token build alone and the render-time projection on the 64x64 lattice (551 bins), separately.
--cpu (one named CPU node; FDTD and modal timed on the SAME node, D180): FDTD of the dense room (its stored
      receivers; every edit type needs a full re-run); m1's modal solve to 600 Hz (operator + eigenpairs, what a
      geometry change costs); with CACHED modes (the eigenvectors on every air node): a listener / source move is a
      lookup + a [modes x 601] product, a material change rescales eta (beta enters linearly) and re-forms the sum.
Repeats: the median of 20 runs after 3 warm-ups for every case under 5 s per run; 5 runs after 1 warm-up above that
(the spread of a multi-second render is far below its median; stated in the report).
--report: outputs/p4_9/latency.{json,md} and F2 (outputs/p4_9/meeting/F2_latency.png, log scale).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

OUT = Path("outputs/p4_9/latency")
CK = "outputs/p4_8/arms/R0/ckpt_iter0120000.pt"
SHAPE1000 = "data/track_p4_5_testdense/L5.54_W5.28_d1.42_w1.52.h5"


def rooms():
    """shape 1000 and the dense room with the most receivers."""
    import h5py

    from scripts.p4_9_fdtd_stencils import dense_rooms
    best, n = None, -1
    for _g, _f, p in dense_rooms():
        with h5py.File(p) as f:
            k = len(json.loads(f.attrs["receiver_pos"]))
        if k > n:
            best, n = p, k
    return {"shape1000": Path(SHAPE1000), "largest": best}


def timeit(fn, sync=None, budget_s=5.0):
    """Median wall time: 3 warm-ups + 20 runs, or 1 + 5 when one run exceeds budget_s."""
    def one():
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        return time.perf_counter() - t0
    first = one()
    n_warm, n_run = (1, 5) if first > budget_s else (3, 20)
    for _ in range(n_warm - 1):
        one()
    ts = [one() for _ in range(n_run)]
    return {"median_s": float(np.median(ts)), "min_s": float(np.min(ts)), "max_s": float(np.max(ts)), "runs": n_run,
            "warmups": n_warm}


def room_info(path):
    import h5py
    with h5py.File(path) as f:
        return {"verts": [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])],
                "rx": np.asarray(json.loads(f.attrs["receiver_pos"]), float),
                "keep": np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64),
                "L": float(f.attrs["L"]), "W": float(f.attrs["W"]), "name": Path(path).name}


def moved_vertex(verts, L, W):
    """Move the first reflex (notch) corner by 0.3 m into the notch's solid (its notch grows), keeping the ring
    axis-aligned: the reflex corner's two incident edges shift with it."""
    from aaf.sim.polygon_geom import polygon_contains
    n = len(verts)
    for i in range(n):
        a, b, c = np.asarray(verts[i - 1]), np.asarray(verts[i]), np.asarray(verts[(i + 1) % n])
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross < 0:                                          # reflex corner of a CCW ring
            for d in ((0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3)):
                v = [tuple(p) for p in verts]
                nb = b + np.asarray(d)
                # move the corner and the neighbouring vertices that share its x or y
                v[i] = tuple(nb)
                if abs(a[0] - b[0]) < 1e-9:
                    v[i - 1] = (nb[0], a[1])
                else:
                    v[i - 1] = (a[0], nb[1])
                if abs(c[0] - b[0]) < 1e-9:
                    v[(i + 1) % n] = (nb[0], c[1])
                else:
                    v[(i + 1) % n] = (c[0], nb[1])
                xs, ys = [p[0] for p in v], [p[1] for p in v]
                if min(xs) < -1e-9 or min(ys) < -1e-9 or max(xs) > L + 1e-9 or max(ys) > W + 1e-9:
                    continue
                area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(v, v[1:] + v[:1]))
                if area2 <= 0 or not bool(polygon_contains(v, np.asarray([[0.5, 1.6]]))[0]):
                    continue
                return v, {"vertex": i, "move_m": list(d)}
    raise RuntimeError("no movable reflex vertex")


def run_gpu():
    import torch

    from aaf.eval.p3_2_eval import load_model
    from aaf.eval.physical_projection import project_field
    from aaf.models.conditioning_2d import build_cond_vector_2d
    from aaf.sim.polygon_geom import polygon_contains
    dev = torch.device("cuda")
    model, renderer, *_ = load_model(Path(CK), dev)
    sync = torch.cuda.synchronize
    rep = {"gpu": torch.cuda.get_device_name(0), "dtype": "fp32 (tinycudann sm_52 build: CutlassMLP, fp16 compiled "
           "out; D181 A4)", "checkpoint": CK, "rooms": {}}

    def render(rx, src, z, L, W, n_out, chunk):
        R = torch.tensor(rx, dtype=torch.float32, device=dev)
        S = torch.tensor(src, dtype=torch.float32, device=dev).view(1, 2)
        lo, hi = torch.zeros(2, device=dev), torch.tensor([L, W], device=dev, dtype=torch.float32)
        kw = {} if n_out is None else {"n_out": n_out}
        out = []
        with torch.no_grad():
            for s in range(0, len(R), chunk):
                r = R[s:s + chunk]
                out.append(renderer(model, r, S.expand(len(r), -1), lo, hi,
                                    z_s=z.unsqueeze(0).expand(len(r), -1), **kw))
        return out

    for tag, path in rooms().items():
        ri = room_info(path)
        z = build_cond_vector_2d("geom_token", ri["L"], ri["W"], [0.15] * len(ri["verts"]), verts=ri["verts"],
                                 device=dev)
        res = {"room": ri["name"], "n_grid": int(len(ri["rx"]))}
        one = ri["rx"][len(ri["rx"]) // 2][None]
        src2 = (ri["L"] - 0.5, 1.6)
        for pname, n_out, chunk in (("current_4097", None, 8), ("bins_601", 601, 16)):
            # tinycudann allocates from its OWN arena and cannot reuse torch's cached blocks (D177): free them
            # between phases, or the larger chunk of the next phase runs out of memory.
            torch.cuda.empty_cache()
            p = {}
            p["listener_1rx"] = timeit(lambda: render(one, (0.5, 1.6), z, ri["L"], ri["W"], n_out, 1), sync)
            p["listener_grid"] = timeit(lambda: render(ri["rx"], (0.5, 1.6), z, ri["L"], ri["W"], n_out, chunk), sync)
            p["source_1rx"] = timeit(lambda: render(one, src2, z, ri["L"], ri["W"], n_out, 1), sync)
            p["source_grid"] = p["listener_grid"]          # identical work: the source is a render input
            zm_fn = lambda: build_cond_vector_2d("geom_token", ri["L"], ri["W"], [0.30] * len(ri["verts"]),  # noqa
                                                 verts=ri["verts"], device=dev)
            p["material_1rx"] = timeit(lambda: render(one, (0.5, 1.6), zm_fn(), ri["L"], ri["W"], n_out, 1), sync)
            p["material_grid"] = timeit(lambda: render(ri["rx"], (0.5, 1.6), zm_fn(), ri["L"], ri["W"], n_out, chunk),
                                        sync)
            if tag == "shape1000":
                vg, how = moved_vertex(ri["verts"], ri["L"], ri["W"])
                inside = ri["rx"][polygon_contains(vg, ri["rx"])]
                zg_fn = lambda: build_cond_vector_2d("geom_token", ri["L"], ri["W"], [0.15] * len(vg), verts=vg,  # noqa
                                                     device=dev)
                p["geometry_grid"] = timeit(lambda: render(inside, (0.5, 1.6), zg_fn(), ri["L"], ri["W"], n_out,
                                                           chunk), sync)
                p["geometry_1rx"] = timeit(lambda: render(one, (0.5, 1.6), zg_fn(), ri["L"], ri["W"], n_out, 1), sync)
                p["geometry_edit"] = dict(how, n_grid_after=int(len(inside)))
            p["chunk_rows"] = chunk
            res[pname] = p
            print(tag, pname, {k: round(v["median_s"], 4) for k, v in p.items() if isinstance(v, dict)
                               and "median_s" in v}, flush=True)
        res["token_build"] = timeit(lambda: build_cond_vector_2d("geom_token", ri["L"], ri["W"],
                                                                 [0.15] * len(ri["verts"]), verts=ri["verts"],
                                                                 device=dev), sync)
        torch.cuda.empty_cache()
        P = np.concatenate([h.cpu().numpy() for h in render(ri["rx"], (0.5, 1.6), z, ri["L"], ri["W"], 601, 16)])
        res["projection_64x64"] = timeit(lambda: project_field(P, ri["keep"], (ri["L"] - 0.24) / 63,
                                                               (ri["W"] - 0.24) / 63), sync)
        rep["rooms"][tag] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gpu.json").write_text(json.dumps(rep, indent=1))
    print("wrote", OUT / "gpu.json")


def run_cpu():
    import aaf.sim.fdtd_2d as F
    from types import SimpleNamespace

    from scripts.p4_7_m1_modal import F_PRIMARY, eigen, operator
    from scripts.p4_9_fdtd_stencils import ALPHA, C, DX, geometry, load_room
    cpu = subprocess.run(["bash", "-c", "lscpu | grep 'Model name'"], capture_output=True, text=True).stdout.strip()
    rep = {"node": platform.node(), "cpu": cpu, "threads": "OMP/MKL/OPENBLAS = 4", "rooms": {}}
    for tag, path in rooms().items():
        r = load_room(path)
        geom, extra, dxx, dxy = geometry(r)
        res = {"room": path.name, "n_rx": int(len(r["rx"]))}
        res["fdtd_full_room"] = timeit(lambda: F.simulate(r["L"], r["W"], [ALPHA] * 4, r["src"], r["rx"], dx=DX,
                                                          fs=r["fs"], n=r["n"], c=C, extra_walls=extra))
        cfg = SimpleNamespace(L=r["L"], W=r["W"], verts=r["verts"], label=path.stem)
        holder = {}

        def solve():
            op = operator(cfg)
            holder["op"] = op
            holder["eig"] = eigen(op, F_PRIMARY)
        res["modal_resolve_600Hz"] = timeit(solve)
        op = holder["op"]
        k2, psi = holder["eig"][0], holder["eig"][1]
        res["n_modes_600"] = int(len(k2))
        eta = ((op["mu"] * op["D"])[:, None] * psi ** 2).sum(0) * op["dA"]
        f = np.arange(601) * 0.5
        k = 2 * np.pi * f / 343.0
        I = np.rint(r["rx"][:, 0] / op["dxx"]).astype(int)
        J = np.rint(r["rx"][:, 1] / op["dxy"]).astype(int)
        jr = op["idx"][I, J]
        js = op["idx"][int(round(r["src"][0] / op["dxx"])), int(round(r["src"][1] / op["dxy"]))]

        def field(rows, src_row, eta_scale=1.0):
            den = k2[:, None] - k[None] ** 2 + 1j * k[None] * (eta_scale * eta)[:, None]
            den[:, 0] = 1.0
            return psi[rows] @ (psi[src_row][:, None] / den)
        one = jr[len(jr) // 2:len(jr) // 2 + 1]
        res["cached_listener_1rx"] = timeit(lambda: field(one, js))
        res["cached_listener_grid"] = timeit(lambda: field(jr, js))
        res["cached_source_grid"] = timeit(lambda: field(jr, jr[0]))
        res["cached_material_grid"] = timeit(lambda: field(jr, js, eta_scale=2.0))
        res["cached_material_1rx"] = timeit(lambda: field(one, js, eta_scale=2.0))
        rep["rooms"][tag] = res
        print(tag, {kk: round(v["median_s"], 5) for kk, v in res.items() if isinstance(v, dict)}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cpu.json").write_text(json.dumps(rep, indent=1))
    print("wrote", OUT / "cpu.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.gpu:
        run_gpu()
    if a.cpu:
        run_cpu()


if __name__ == "__main__":
    main()
