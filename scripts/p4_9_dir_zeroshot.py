"""P4-9 0c (D180; D181 A13, A14, E2, E3, c2): first-order directional output, zero-shot -- three readouts.

  per-ray     D = sum_j u_j s_j from the renderer's per-ray decomposition (before the ray sum; the DC head is omni only),
              read as grad p ~= j k D under the renderer's e^{-j 2 pi f d / c} delay (u_j points receiver -> sample).
  omni-FD     the central finite-difference gradient of the RENDERED omni field, neighbours at +-h, h = 2 FDTD nodes
              (~4 cm, E2), eval mode (no jitter).
  proj-FD     the gradient of the PROJECTED omni field (render-time projection, |k| <= 1.5 k0) by central differences on
              the 64x64 dense lattice (interior nodes; truncation error ~(k h)^2 / 6 = 3.5% at 300 Hz for h ~ 8.4 cm).
Truth: FDTD central differences at the same receivers (`scripts/p4_9_fdtd_stencils.py --set dense`): h = 1 node for
per-ray and proj-FD (D180: "at the solver dx"), h = 2 nodes for omni-FD (E2: "at the same h").
Score (c2): per bin, over OFF-lattice receivers >= 0.5 m from the source and both components, the normalised complex
inner product rho = sum conj(a) b / sqrt(sum |a|^2 sum |b|^2) between the readout and grad p / (j k); reported as
Re(rho) (PRIMARY: sign- and phase-sensitive) and |rho|; band means 25-100 / 100-200 / 200-300 Hz; in-family (15) and
unseen (16; U 8, C3 8).
Unit test first (E3): for an ideal decomposition of a free-field point source (all of p on the ray towards the
source), the per-ray reading's error relative to the reading, |grad p / (j k D) - 1| = |H1(kr) / (j H0(kr)) - 1|, must
track 1/(2 kr) within 10% for kr in [1, 30]; the code's error must equal that closed form, and the omni-FD code must
reproduce the analytic gradient to O((k h)^2).
  --unit                      CPU: the unit test -> outputs/p4_9/dirzs/unit.json
  --render NAME=CKPT          GPU: per-ray D and omni-FD at the 31 dense rooms -> outputs/p4_9/dirzs/cache/NAME__room.npz
  --score NAMES               CPU: outputs/p4_9/directional_zero_shot.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path("outputs/p4_9/dirzs")
STEN = Path("outputs/p4_9/fdtd_stencils/dense")
C0, DF = 343.0, 0.5
BANDS = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
SRC = np.array([0.5, 1.6])
MIN_SRC = 0.5
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
UN = "outputs/p4_8/cache/unseen_{}.npz"


# ------------------------------------------------------------------------------------------- the readouts
def readout_per_ray(D, k):
    """grad p estimate from the per-ray first-order channel: j k D. D [..., 2, F], k [F]."""
    return 1j * k * D


def readout_fd(Hxp, Hxm, Hyp, Hym, hx, hy):
    """central differences [..., 2, F]."""
    return np.stack([(Hxp - Hxm) / (2 * hx), (Hyp - Hym) / (2 * hy)], axis=-2)


def rho(a, b):
    """per-bin normalised complex inner product over rows and components: a, b [n, 2, F] -> [F] complex."""
    num = (np.conj(a) * b).sum(axis=(0, 1))
    den = np.sqrt((np.abs(a) ** 2).sum(axis=(0, 1)) * (np.abs(b) ** 2).sum(axis=(0, 1)))
    return num / np.maximum(den, 1e-300)


# ------------------------------------------------------------------------------------------- unit test (E3)
def unit_test():
    from scipy.special import hankel2
    out = {"kr": [], "code_err": [], "closed_form": [], "one_over_2kr": [], "fd_err": []}
    f = 100.0
    k = 2 * np.pi * f / C0
    src = np.zeros(2)
    for kr in np.geomspace(1.0, 30.0, 40):
        r = kr / k
        x = np.array([r, 0.0])                                 # receiver on the +x axis; source at the origin
        p = -0.25j * hankel2(0, kr)                            # outgoing under e^{-jkd}
        u_s = (src - x) / r                                    # the ray from the receiver towards the source
        D = (u_s[:, None] * p)[None]                           # [1, 2, 1]: all of p on that ray
        R = readout_per_ray(D, np.array([k]))[0, :, 0]         # j k D
        T = 0.25j * k * hankel2(1, kr) * (x - src) / r          # analytic grad p = -(j/4) d/dr H0 = (j k / 4) H1 r_hat
        e = float(np.abs(T[0] / R[0] - 1.0))                   # error relative to the reading
        h = 0.04
        pf = lambda y: np.array([-0.25j * hankel2(0, k * np.linalg.norm(y - src))])  # noqa  (one bin)
        G = readout_fd(pf(x + [h, 0]), pf(x - [h, 0]), pf(x + [0, h]), pf(x - [0, h]), h, h)[:, 0]
        out["kr"].append(float(kr))
        out["code_err"].append(e)
        out["closed_form"].append(float(np.abs(hankel2(1, kr) / (1j * hankel2(0, kr)) - 1.0)))
        out["one_over_2kr"].append(float(1.0 / (2 * kr)))
        out["fd_err"].append(float(np.linalg.norm(G - T) / np.linalg.norm(T)))
    ce, cf, ref = (np.asarray(out[k_]) for k_ in ("code_err", "closed_form", "one_over_2kr"))
    out["code_equals_closed_form"] = bool(np.allclose(ce, cf, rtol=1e-9, atol=1e-12))
    out["max_rel_dev_from_1_over_2kr"] = float(np.max(np.abs(cf - ref) / ref))
    out["passes_E3"] = bool(out["code_equals_closed_form"] and out["max_rel_dev_from_1_over_2kr"] <= 0.10)
    out["fd_max_rel_err"] = float(np.max(out["fd_err"]))
    out["sign_check"] = ("a sign error in j k D would make the error ~2 (the reading would point away from the "
                         "gradient); a scale error s would make it ~|s - 1| at large kr")
    return out


# ------------------------------------------------------------------------------------------- rendering
def dense_rooms():
    from scripts.p4_9_fdtd_stencils import dense_rooms as dr
    out, it, iu = [], 0, 0
    for g, fam, p in dr():
        out.append({"group": g, "family": fam, "path": p, "cache_i": it if g == "test" else iu})
        it, iu = (it + 1, iu) if g == "test" else (it, iu + 1)
    return out


def render(name, ck):
    import torch

    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    dev = torch.device("cuda")
    model, renderer, *_ = load_model(Path(ck), dev)
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    for r in dense_rooms():
        dst = OUT / "cache" / "{}__{}.npz".format(name, r["path"].stem)
        if dst.exists():
            continue
        st = np.load(STEN / (r["path"].stem + ".npz"))
        with h5py.File(r["path"]) as f:
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
            verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
            L, W = float(f.attrs["L"]), float(f.attrs["W"])
        assert np.allclose(st["rx"], rx)
        hx, hy = 2 * float(st["dxx"]), 2 * float(st["dxy"])
        z = build_cond_vector_2d("geom_token", L, W, [0.15] * len(verts), verts=verts, device=dev)
        lo, hi = torch.zeros(2, device=dev), torch.tensor([L, W], device=dev, dtype=torch.float32)
        S1 = torch.tensor([[0.5, 1.6]], device=dev)

        def rend(P, per_ray=False):
            Rt = torch.tensor(P, dtype=torch.float32, device=dev)
            Hs, Ds = [], []
            with torch.no_grad():
                for s0 in range(0, len(Rt), 32):
                    rr = Rt[s0:s0 + 32]
                    zz = z.unsqueeze(0).expand(len(rr), -1)
                    if per_ray:
                        H, pr, dirs = renderer(model, rr, S1.expand(len(rr), -1), lo, hi, z_s=zz, n_out=601,
                                               return_per_ray=True)
                        Ds.append(torch.einsum("baf,ad->bdf", pr, dirs.to(pr.dtype)).cpu().numpy())
                    else:
                        H = renderer(model, rr, S1.expand(len(rr), -1), lo, hi, z_s=zz, n_out=601)
                    Hs.append(H.cpu().numpy())
            return np.concatenate(Hs), (np.concatenate(Ds) if per_ray else None)
        H, D = rend(rx, per_ray=True)
        nb = [rend(rx + d)[0] for d in ([hx, 0], [-hx, 0], [0, hy], [0, -hy])]
        G = readout_fd(nb[0], nb[1], nb[2], nb[3], hx, hy)
        np.savez_compressed(dst, D=D.astype(np.complex64), G=G.astype(np.complex64), H=H.astype(np.complex64), rx=rx,
                            hx=hx, hy=hy)
        print("rendered", dst.name, flush=True)


# ------------------------------------------------------------------------------------------- scoring
def lattice_grad(F, keep, dx, dy):
    """central differences of a lattice field F [n_keep, F] (np.nonzero(keep) order) at interior nodes.
    Returns (rows, G [n_rows, 2, F])."""
    idx = -np.ones(keep.shape, dtype=np.int64)
    idx[keep] = np.arange(int(keep.sum()))
    rows, G = [], []
    for i in range(1, keep.shape[0] - 1):
        for j in range(1, keep.shape[1] - 1):
            c, a, b, d, e = idx[i, j], idx[i - 1, j], idx[i + 1, j], idx[i, j - 1], idx[i, j + 1]
            if min(c, a, b, d, e) < 0:
                continue
            rows.append(c)
            G.append(np.stack([(F[b] - F[a]) / (2 * dx), (F[e] - F[d]) / (2 * dy)]))
    return np.asarray(rows), np.asarray(G)


def score(names):
    from aaf.eval.physical_projection import project_field
    from scripts.p4_5_gate2_dense import on_lattice
    k = 2 * np.pi * np.arange(601) * DF / C0
    jk = 1j * k
    jk[0] = 1.0
    rows = []
    for r in dense_rooms():
        st = np.load(STEN / (r["path"].stem + ".npz"))
        with h5py.File(r["path"]) as f:
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
            keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
            L, W = float(f.attrs["L"]), float(f.attrs["W"])
        ok = (~on_lattice(rx)) & (np.hypot(*(rx - SRC).T) >= MIN_SRC)
        T1 = np.stack([st["g1x"], st["g1y"]], axis=1) / jk                       # [n, 2, F] grad p / (jk), h = 1
        T2 = np.stack([st["g2x"], st["g2y"]], axis=1) / jk
        row = {"room": r["path"].name, "group": r["group"], "family": r["family"], "n_rx": int(ok.sum()),
               "readouts": {}}
        for name in names:
            c = np.load(OUT / "cache" / "{}__{}.npz".format(name, r["path"].stem))
            reads = {"per-ray": (c["D"], T1, ok), "omni-FD": (c["G"] / jk, T2, ok)}
            cache = np.load((G2 if r["group"] == "test" else UN).format(name))
            P = cache["P%d" % r["cache_i"]][:, :601]
            Pp = project_field(P, keep, (L - 0.24) / 63, (W - 0.24) / 63)
            lr, LG = lattice_grad(Pp, keep, (L - 0.24) / 63, (W - 0.24) / 63)
            m = np.zeros(len(rx), bool)
            m[lr] = True
            sel = ok[lr]
            reads["proj-FD"] = (LG[sel] / jk, T1[lr][sel], None)
            for rn, (A, T, msk) in reads.items():
                a, b = (A[msk], T[msk]) if msk is not None else (A, T)
                rr = rho(a, b)
                row["readouts"]["{} {}".format(name, rn)] = {
                    **{"re_" + bn: float(np.mean(rr.real[lo:hi])) for bn, (lo, hi) in BANDS.items()},
                    **{"abs_" + bn: float(np.mean(np.abs(rr[lo:hi]))) for bn, (lo, hi) in BANDS.items()},
                    "n": int(len(a))}
        rows.append(row)
        print(r["path"].name[:40], {k_: round(v["re_25-100"], 3) for k_, v in row["readouts"].items()}, flush=True)
    keys = list(rows[0]["readouts"])
    sets = {"in_family15": lambda x: x["group"] == "test", "unseen16": lambda x: x["group"] == "unseen",
            "U8": lambda x: x["family"] == "U", "C3_8": lambda x: x["family"] == "C3"}
    summ = {s: {k_: {m: float(np.mean([x["readouts"][k_][m] for x in rows if f(x)]))
                     for m in rows[0]["readouts"][k_] if m != "n"} for k_ in keys} for s, f in sets.items()}
    unit = json.loads((OUT / "unit.json").read_text()) if (OUT / "unit.json").exists() else unit_test()
    rep = {"definition": __doc__.split("\n\n")[0].strip(), "unit_test": {k_: unit[k_] for k_ in (
        "passes_E3", "code_equals_closed_form", "max_rel_dev_from_1_over_2kr", "fd_max_rel_err")}, "summary": summ,
        "rows": rows}
    Path("outputs/p4_9").mkdir(parents=True, exist_ok=True)
    Path("outputs/p4_9/directional_zero_shot.json").write_text(json.dumps(rep, indent=1))
    Lm = ["# P4-9 0c: first-order directional output, zero-shot", "",
          "Unit test (E3): passes = {passes_E3}; the code's per-ray error equals the closed form |H1/(jH0) - 1| = "
          "{code_equals_closed_form}; its largest deviation from 1/(2kr) over kr 1-30 is {max_rel_dev_from_1_over_2kr:.3f}; "
          "omni-FD reproduces the analytic gradient to {fd_max_rel_err:.2e}.".format(**rep["unit_test"]), "",
          "Directional r = Re of the normalised complex inner product with FDTD grad p / (jk), per bin over off-lattice "
          "receivers >= 0.5 m from the source and both components, band means (|rho| in the JSON).", ""]
    for s, title in (("in_family15", "In-family (15)"), ("unseen16", "Unseen (16)"), ("U8", "U (8)"), ("C3_8", "C3 (8)")):
        Lm += ["### " + title, "", "| readout | Re rho 25-100 / 100-200 / 200-300 | abs rho 25-100 |", "|---|---|---|"]
        for k_ in keys:
            d = summ[s][k_]
            Lm.append("| {} | {:.3f} / {:.3f} / {:.3f} | {:.3f} |".format(k_, d["re_25-100"], d["re_100-200"],
                                                                         d["re_200-300"], d["abs_25-100"]))
        Lm.append("")
    Path("outputs/p4_9/directional_zero_shot.md").write_text("\n".join(Lm) + "\n")
    print("\n".join(Lm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", action="store_true")
    ap.add_argument("--render", default="")
    ap.add_argument("--score", default="")
    a = ap.parse_args()
    if a.unit:
        OUT.mkdir(parents=True, exist_ok=True)
        u = unit_test()
        (OUT / "unit.json").write_text(json.dumps(u, indent=1))
        print({k: u[k] for k in ("passes_E3", "code_equals_closed_form", "max_rel_dev_from_1_over_2kr",
                                 "fd_max_rel_err")})
    for item in [x for x in a.render.split("+") if x]:
        n, ck = item.split("=")
        render(n, ck)
    if a.score:
        score(a.score.split("+"))


if __name__ == "__main__":
    main()
