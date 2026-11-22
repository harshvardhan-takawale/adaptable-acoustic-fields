"""P4-9 2c feasibility (D180 S-b; D181 A10; plan I9), PHYSICS ONLY, before any NMF code trains.

From m1's eigenpairs on the 31 dense rooms (`scripts/p4_9_modal.py`, exact geometry; every mode to 600 Hz; psi at the
dense receivers and the source), the field is m1's exact formula (`p4_9_sensitivity.modal_field`) restricted to the
zero mode (its exact record, as NMF keeps it) plus the first N nonzero modes:
  S-b         N = 24 (NMF-LB), scored at 25-100 Hz against FDTD (off-lattice, frozen metric): pre-registered >= 0.95 in
              family. Also N = 12, 16, 32, 48, all; and N = 24 + the STATIC correction of the omitted modes
              (sum_{n>24} phi_n(x) phi_n(x0) / k_n^2, frequency-independent -- D180's fallback, here with the TRUE
              higher modes, i.e. its ceiling).
  error budget  N = 24 with errors a learned model would make (5 random draws each, mean):
              eigenfrequency: k_n -> k_n (1 + eps xi_n), xi_n ~ N(0, 1), eps in {0.25, 0.5, 1, 2, 5}%;
              mode shape: phi_n -> renormalised phi_n + delta sum_m c_nm phi_m over the 24 (c ~ N(0, 1/24)), delta in
              {0.05, 0.1, 0.2, 0.4} (delta ~ the subspace error per mode); the damping eta_n is kept (secondary).
  Read: the smallest errors that still reach N1's bars (unseen >= 0.60, in-family >= 0.90) say how accurate NMF must be.
Writes outputs/p4_9/nmf/feasibility.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from scripts.p4_9_sensitivity import modal_field

OUT = Path("outputs/p4_9/nmf")
MODAL = Path("outputs/p4_9/modal")
EPS = (0.0025, 0.005, 0.01, 0.02, 0.05)
DELTA = (0.05, 0.1, 0.2, 0.4)
NS = (12, 16, 24, 32, 48)


def on_lattice(rx):
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def truncated(z, n, k2=None, psi_r=None, psi_s=None, static=False):
    """m1's field with the zero mode + the first n nonzero modes (optionally modified k2 / psi), + optional static
    correction from the omitted true modes."""
    zz = dict(z)
    m = slice(0, n + 1)
    zz["k2"] = (z["k2"] if k2 is None else k2)[m]
    zz["eta"] = z["eta"][m]
    zz["psi_r"] = (z["psi_r"] if psi_r is None else psi_r)[:, m]
    zz["psi_s"] = (z["psi_s"] if psi_s is None else psi_s)[m]
    F = modal_field(zz)
    if static:
        from aaf.eval.modal_prior import C_FIT, DT
        f = np.arange(601) * 0.5
        S = z["S"] * np.exp(1j * 2 * np.pi * f * DT)
        corr = z["psi_r"][:, n + 1:] @ (z["psi_s"][n + 1:] / z["k2"][n + 1:])
        F = F + C_FIT * corr[:, None] * S[None]
    return F


def main():
    from scripts.p4_7_m1_modal import scores
    from scripts.p4_9_dir_zeroshot import dense_rooms
    rng = np.random.default_rng(20261013)
    rows = []
    for r in dense_rooms():
        with h5py.File(r["path"]) as f:
            T = np.asarray(f["ism/H_complex"])[:, :601]
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
        off = ~on_lattice(rx)
        z = dict(np.load(MODAL / "{}__exact.npz".format(r["path"].stem)))
        row = {"room": r["path"].name, "group": r["group"], "family": r["family"], "n_modes_600": int(len(z["k2"])),
               "f24_hz": float(343.0 * np.sqrt(z["k2"][24]) / (2 * np.pi)), "trunc": {}, "eps": {}, "delta": {}}
        row["trunc"]["all"] = scores(modal_field(z)[off], T[off])["25-100"]
        for n in NS:
            row["trunc"][str(n)] = scores(truncated(z, n)[off], T[off])["25-100"]
        row["trunc"]["24+static"] = scores(truncated(z, 24, static=True)[off], T[off])["25-100"]
        k = np.sqrt(z["k2"])
        for e in EPS:
            v = []
            for _ in range(5):
                kk = k.copy()
                kk[1:25] = k[1:25] * (1 + e * rng.standard_normal(24))
                v.append(scores(truncated(z, 24, k2=kk ** 2)[off], T[off])["25-100"])
            row["eps"][str(e)] = float(np.mean(v))
        for d in DELTA:
            v = []
            for _ in range(5):
                C = np.eye(25)
                C[1:25, 1:25] += d * rng.standard_normal((24, 24)) / np.sqrt(24)
                pr, ps = z["psi_r"][:, :25] @ C.T, z["psi_s"][:25] @ C.T
                nr = np.sqrt((C ** 2).sum(1))                       # renormalise each mixed mode (orthonormal basis)
                pr, ps = pr / nr[None], ps / nr
                PR = z["psi_r"].copy()
                PS = z["psi_s"].copy()
                PR[:, :25], PS[:25] = pr, ps
                v.append(scores(truncated(z, 24, psi_r=PR, psi_s=PS)[off], T[off])["25-100"])
            row["delta"][str(d)] = float(np.mean(v))
        rows.append(row)
        print(r["path"].name[:40], "all {:.3f} 24 {:.3f} 24+static {:.3f} f24 {:.0f} Hz".format(
            row["trunc"]["all"], row["trunc"]["24"], row["trunc"]["24+static"], row["f24_hz"]), flush=True)

    def mean(sel, key, sub):
        return float(np.mean([x[key][sub] for x in rows if sel(x)]))
    sets = {"in_family15": lambda x: x["group"] == "test", "unseen16": lambda x: x["group"] == "unseen"}
    summ = {s: {"trunc": {k_: mean(f, "trunc", k_) for k_ in rows[0]["trunc"]},
                "eps": {k_: mean(f, "eps", k_) for k_ in rows[0]["eps"]},
                "delta": {k_: mean(f, "delta", k_) for k_ in rows[0]["delta"]},
                "f24_hz_median": float(np.median([x["f24_hz"] for x in rows if f(x)]))} for s, f in sets.items()}
    sb = summ["in_family15"]["trunc"]["24"]
    rep = {"definition": __doc__.split("\n\n")[0].strip(), "summary": summ,
           "S_b": {"test": "N = 24 at 25-100 Hz in family >= 0.95", "value": sb, "holds": bool(sb >= 0.95)},
           "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feasibility.json").write_text(json.dumps(rep, indent=1))
    L = ["# P4-9 NMF feasibility (physics only; plan I9)", "", rep["definition"], "",
         "**S-b (pre-registered, D180): N = 24 modes reach {:.3f} in family at 25-100 Hz -> {}.**".format(
             sb, "HOLDS" if sb >= 0.95 else "FAILS"), "",
         "| r at 25-100 Hz (off-lattice, vs FDTD) | in-family 15 | unseen 16 |", "|---|---|---|"]
    for k_ in summ["in_family15"]["trunc"]:
        L.append("| modes: {} | {:.3f} | {:.3f} |".format(k_, summ["in_family15"]["trunc"][k_],
                                                         summ["unseen16"]["trunc"][k_]))
    for k_ in summ["in_family15"]["eps"]:
        L.append("| 24 modes, eigenfrequency error {:.2%} | {:.3f} | {:.3f} |".format(
            float(k_), summ["in_family15"]["eps"][k_], summ["unseen16"]["eps"][k_]))
    for k_ in summ["in_family15"]["delta"]:
        L.append("| 24 modes, mode-shape mixing delta {} | {:.3f} | {:.3f} |".format(
            k_, summ["in_family15"]["delta"][k_], summ["unseen16"]["delta"][k_]))
    L += ["", "The 24th nonzero mode lies at {:.0f} Hz (in-family median) / {:.0f} Hz (unseen median).".format(
        summ["in_family15"]["f24_hz_median"], summ["unseen16"]["f24_hz_median"]),
        "N1's bars: unseen >= 0.60, in-family >= 0.90 (NMF or NMF-P). The error rows say how accurate a learned modal "
        "field must be to clear them."]
    (OUT / "feasibility.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
