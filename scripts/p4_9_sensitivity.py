"""P4-9 0d (D180) with D181 A1 / A2 / A7: physics-input sensitivity, physics AND network under the same wrong inputs.

Physics = m1's exact field (`aaf.eval.modal_prior.compute`'s formula, re-formed from the stored eigenpairs of
`scripts/p4_9_modal.py`: C_FIT, the + i k eta damping, every mode to 600 Hz, the zero mode by its exact discrete
record) on each geometry variant; beta x0.5 / x2 rescale eta (beta enters linearly; the zero-mode record is
re-synthesised). Network = R0 @120K and S2 @120K given the SAME wrong geometry: tokens built from the variant polygon
(and its bounding box as the render box), at the room's true dense receivers (601-bin path, D181 c4). The network has
no beta input it was ever trained to use (one material in every corpus, A2), so the beta rows are physics-only.

Inputs (D180 0d): (i) beta x0.5, x2 (physics only); (ii) vertices snapped to 10 cm; (iii) the bounding rectangle;
(iv) the smallest notch removed (the 16 unseen rooms only, A7). Scores: per room, OFF-lattice receivers (Rule 11),
the frozen metric (`scripts/p4_7_m1_modal.scores` = p4_6's bands and colr: band-mean per-bin dB Pearson r, band LSD) at 25-100 / 100-200 /
200-300 Hz, against FDTD truth (PRIMARY, A1) and against m1-exact (D180's reference); means over the 15 in-family
rooms and the 16 unseen (U 8, C3 8).
  --render   GPU: network renders of the variants -> outputs/p4_9/sensitivity/cache/
  --score    CPU: outputs/p4_9/physics_sensitivity.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from aaf.eval.modal_prior import C0, C_FIT, DT, zero_mode_record

OUT = Path("outputs/p4_9")
CACHE = OUT / "sensitivity" / "cache"
MODAL = OUT / "modal"
MODELS = {"R0_120K": "outputs/p4_8/arms/R0/ckpt_iter0120000.pt", "S2_120K": "outputs/p4_8/arms/S2/ckpt_iter0120000.pt"}
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
UN = "outputs/p4_8/cache/unseen_{}.npz"
GEOM_VARIANTS = ("snap10", "bbox", "notch_rm")
BANDS = ("25-100", "100-200", "200-300")


def rooms():
    from scripts.p4_9_fdtd_stencils import dense_rooms
    out, it, iu = [], 0, 0
    for g, fam, p in dense_rooms():
        out.append({"group": g, "family": fam, "path": p, "cache_i": it if g == "test" else iu})
        if g == "test":
            it += 1
        else:
            iu += 1
    return out


def modal_field(z, eta_scale=1.0, second_source=False):
    """m1's exact transfer function from stored eigenpairs (modal_prior.compute's formula)."""
    k2, eta, psi_r, S = z["k2"], z["eta"] * eta_scale, z["psi_r"], z["S"]
    psi_s = z["psi_s2"] if second_source else z["psi_s"]
    meta = json.loads(str(z["meta"]))
    f = np.arange(601) * 0.5
    kk = 2 * np.pi * f / C0
    den = k2[1:, None] - kk[None] ** 2 + 1j * kk[None] * eta[1:, None]
    den[:, 0] = 1.0
    G = psi_r[:, 1:] @ (psi_s[1:, None] / den)
    G = G * (S * np.exp(1j * 2 * np.pi * f * DT))[None]
    G[:, 0] = 0.0
    c_pred = meta["dA"] / (C0 * DT) ** 2
    td0 = z["td0_s2"] if second_source else z["td0"]
    if eta_scale != 1.0:
        from aaf.eval.modal_prior import pulse
        s, _ = pulse()
        g0 = float(psi_s[0] * meta["dA"])                     # the source is an interior node (mu = 1)
        assert np.allclose(zero_mode_record(float(z["eta"][0]), g0, s), td0, rtol=1e-6, atol=0), "g0 convention"
        td0 = zero_mode_record(float(eta[0]), g0, s)
    return C_FIT * G + psi_r[:, :1] * td0[None] * (C_FIT / c_pred)


def render_variants():
    import torch

    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    from scripts.p4_9_modal import variant_verts
    dev = torch.device("cuda")
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, ck in MODELS.items():
        model, renderer, *_ = load_model(Path(ck), dev)
        for r in rooms():
            with h5py.File(r["path"]) as f:
                rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
                verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
            for v in GEOM_VARIANTS:
                if v == "notch_rm" and r["group"] != "unseen":
                    continue
                dst = CACHE / "{}__{}__{}.npz".format(name, r["path"].stem, v)
                if dst.exists():
                    continue
                vv = variant_verts(verts, v, r["group"], r["path"].name)
                L, W = max(p[0] for p in vv), max(p[1] for p in vv)
                z = build_cond_vector_2d("geom_token", L, W, [0.15] * len(vv), verts=vv, device=dev)
                lo, hi = torch.zeros(2, device=dev), torch.tensor([L, W], device=dev, dtype=torch.float32)
                R = torch.tensor(rx, dtype=torch.float32, device=dev)
                S = torch.tensor([[0.5, 1.6]], device=dev)
                P = []
                with torch.no_grad():
                    for s0 in range(0, len(R), 32):
                        rr = R[s0:s0 + 32]
                        P.append(renderer(model, rr, S.expand(len(rr), -1), lo, hi,
                                          z_s=z.unsqueeze(0).expand(len(rr), -1), n_out=601).cpu().numpy())
                np.savez_compressed(dst, P=np.concatenate(P).astype(np.complex64), rx=rx, verts=np.asarray(vv))
                print("rendered", dst.name, flush=True)


def score():
    from scripts.p4_5_gate2_dense import on_lattice
    from scripts.p4_7_m1_modal import scores          # the frozen band r / LSD (same BANDS and colr as p4_6)
    rows = []
    caches = {m: {"test": np.load(G2.format(m)), "unseen": np.load(UN.format(m))} for m in MODELS}
    for r in rooms():
        with h5py.File(r["path"]) as f:
            T = np.asarray(f["ism/H_complex"])[:, :601]
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
        off = ~on_lattice(rx)
        ze = np.load(MODAL / "{}__exact.npz".format(r["path"].stem))
        assert np.allclose(ze["rx"], rx)
        m1 = modal_field(ze)
        fields = {"physics exact": m1, "physics beta x0.5": modal_field(ze, 0.5),
                  "physics beta x2": modal_field(ze, 2.0)}
        for v in GEOM_VARIANTS:
            if v == "notch_rm" and r["group"] != "unseen":
                continue
            fields["physics " + v] = modal_field(np.load(MODAL / "{}__{}.npz".format(r["path"].stem, v)))
            for m in MODELS:
                fields["{} {}".format(m, v)] = np.load(CACHE / "{}__{}__{}.npz".format(m, r["path"].stem, v))["P"]
        for m in MODELS:
            z = caches[m][r["group"]]
            assert np.allclose(z["rx%d" % r["cache_i"]], rx)
            fields["{} exact".format(m)] = z["P%d" % r["cache_i"]][:, :601]
        row = {"room": r["path"].name, "group": r["group"], "family": r["family"], "n_off": int(off.sum()), "vs_fdtd": {},
               "vs_m1": {}}
        for k, F in fields.items():
            row["vs_fdtd"][k] = scores(F[off], T[off])
            row["vs_m1"][k] = scores(F[off], m1[off])
        rows.append(row)
        print(r["path"].name[:40], " ".join("{}:{:.3f}".format(k, row["vs_fdtd"][k]["25-100"])
                                            for k in ("physics exact", "physics bbox", "R0_120K exact", "R0_120K bbox")),
              flush=True)
    keys = sorted({k for row in rows for k in row["vs_fdtd"]})
    sets = {"in_family15": lambda x: x["group"] == "test", "unseen16": lambda x: x["group"] == "unseen",
            "U8": lambda x: x["family"] == "U", "C3_8": lambda x: x["family"] == "C3"}
    summ = {}
    for ref in ("vs_fdtd", "vs_m1"):
        summ[ref] = {}
        for sname, sel in sets.items():
            summ[ref][sname] = {}
            for k in keys:
                rs = [x for x in rows if sel(x) and k in x[ref]]
                if rs:
                    summ[ref][sname][k] = {b: float(np.mean([x[ref][k][b] for x in rs])) for b in BANDS}
                    summ[ref][sname][k].update({"lsd_" + b: float(np.mean([x[ref][k]["lsd_" + b] for x in rs]))
                                                for b in BANDS})
                    summ[ref][sname][k]["n_rooms"] = len(rs)
    rep = {"definition": __doc__.split("\n\n")[0].strip(), "rows": rows, "summary": summ}
    (OUT / "physics_sensitivity.json").write_text(json.dumps(rep, indent=1))
    order = ["physics exact", "physics beta x0.5", "physics beta x2", "physics snap10", "physics bbox",
             "physics notch_rm"] + ["{} {}".format(m, v) for m in MODELS for v in ("exact",) + GEOM_VARIANTS]
    L = ["# P4-9 0d: physics-input sensitivity -- physics and the network under the same wrong inputs", "",
         rep["definition"], "",
         "Off-lattice band-mean r (and LSD dB), mean over rooms, against **FDTD truth** (primary). Network = given the "
         "same wrong geometry as physics; the beta rows are physics-only (A2).", ""]
    for sname, title in (("in_family15", "In-family (15 test shapes)"), ("unseen16", "Unseen (16: 8 U + 8 C3)"),
                         ("U8", "Unseen U (8)"), ("C3_8", "Unseen C3 (8)")):
        L += ["### " + title, "", "| input | r 25-100 / 100-200 / 200-300 | LSD dB | r vs m1-exact (25-100) |",
              "|---|---|---|---|"]
        for k in order:
            d = summ["vs_fdtd"][sname].get(k)
            if d is None:
                continue
            dm = summ["vs_m1"][sname][k]
            L.append("| {} | {:.3f} / {:.3f} / {:.3f} | {:.2f} / {:.2f} / {:.2f} | {:.3f} |".format(
                k, d["25-100"], d["100-200"], d["200-300"], d["lsd_25-100"], d["lsd_100-200"], d["lsd_200-300"],
                dm["25-100"]))
        L.append("")
    (OUT / "physics_sensitivity.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.render:
        render_variants()
    if a.score:
        score()


if __name__ == "__main__":
    main()
