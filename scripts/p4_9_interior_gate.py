"""P4-9 0b(i) (D180): the interior Helmholtz residual of R0 @120K and S2 @120K, RAW and PROJECTED, against FDTD truth.

The S1 relative residual exactly as `scripts/p4_8_s1_gate.py` defines it (imported, not re-implemented): on each dense
room's 64x64 lattice, per bin 25-300 Hz, rel(f) = ||lap_h F + k^2 (F - <F>)|| / ||k^2 (F - <F>)|| over interior
stencils (4 lattice neighbours present; stencils within 0.3 m of the source dropped; unequal spacings from the stored
receiver coordinates), then band means 25-100 / 100-200 / 200-300 Hz. Projected = the render-time physical projection
(`aaf.eval.physical_projection.project_field`, |k| <= 1.5 k0, the evaluators' call), applied to the cached dense
renders. FDTD truth is also projected, as a check that the projection itself does not create a residual.

Rooms: the 15 in-family test shapes + the 16 frozen unseen rooms (31). Pre-registered reading (D180): if projected R0
is at most 2x the FDTD value in EVERY band (means over the 31 rooms), projection already removes the interior term's
target, and the interior Helmholtz loss is closed for good. Wording (D181 E6): a ratio to FDTD's residual is stated as
"N x FDTD's"; the residual itself is relative to the k^2 term.
  python scripts/p4_9_interior_gate.py         (GPU recommended: the projection runs ~100 lattice fields)
Writes outputs/p4_9/gates/interior.json and the interior section of outputs/p4_9/physics_gates.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from aaf.eval.physical_projection import project_field
from scripts.p4_8_s1_gate import HI, residual, rooms_test, rooms_unseen

MODELS = ["R0_120K", "S2_120K"]
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
UN = "outputs/p4_8/cache/unseen_{}.npz"
OUT = Path("outputs/p4_9/gates")
BANDS = ("25-100", "100-200", "200-300")


def room_lw(f):
    v = np.asarray(json.loads(f.attrs["verts"]), dtype=float)
    return float(v[:, 0].max()), float(v[:, 1].max())


def main():
    groups = {"test": rooms_test(), "unseen": rooms_unseen()}
    rooms = []
    for g, rs in groups.items():
        for i, (fam, p) in enumerate(rs):
            with h5py.File(p) as f:
                T = np.asarray(f["ism/H_complex"])[:, :HI]
                rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
                keep = np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64)
                L, W = room_lw(f)
            assert len(rx) == keep.sum(), p.name
            rooms.append({"group": g, "family": fam, "room": p.name, "i": i, "T": T, "rx": rx, "keep": keep,
                          "dx": (L - 0.24) / 63, "dy": (W - 0.24) / 63})
    caches = {m: {"test": np.load(G2.format(m)), "unseen": np.load(UN.format(m))} for m in MODELS}
    per = []
    for r in rooms:
        row = {k: r[k] for k in ("group", "family", "room")}
        fields = {"FDTD": r["T"]}
        for m in MODELS:
            z = caches[m][r["group"]]
            assert np.allclose(z["rx%d" % r["i"]], r["rx"]), (m, r["room"])
            fields[m] = z["P%d" % r["i"]][:, :HI]
        for name, F in fields.items():
            for kind in ("raw", "proj"):
                G = F if kind == "raw" else project_field(F, r["keep"], r["dx"], r["dy"])
                _, bands, n = residual(G, r["keep"], r["rx"])
                row["{}_{}".format(name, kind)] = bands
                row["n_stencils"] = n
        per.append(row)
        print("{:6s} {:3s} {:48s} FDTD {} | R0 raw {} proj {}".format(
            r["group"], r["family"], r["room"][:48],
            " ".join("{:.3f}".format(row["FDTD_raw"][b]) for b in BANDS),
            " ".join("{:.2f}".format(row["R0_120K_raw"][b]) for b in BANDS),
            " ".join("{:.3f}".format(row["R0_120K_proj"][b]) for b in BANDS)), flush=True)

    def mean(key, sel=lambda r: True):
        rs = [r for r in per if sel(r)]
        return {b: float(np.mean([r[key][b] for r in rs])) for b in BANDS}

    sets = {"all31": lambda r: True, "in_family15": lambda r: r["group"] == "test",
            "unseen16": lambda r: r["group"] == "unseen", "U8": lambda r: r["family"] == "U",
            "C3_8": lambda r: r["family"] == "C3"}
    keys = ["FDTD_raw", "FDTD_proj"] + ["{}_{}".format(m, k) for m in MODELS for k in ("raw", "proj")]
    summ = {s: {k: mean(k, f) for k in keys} for s, f in sets.items()}
    fd = summ["all31"]["FDTD_raw"]
    ratios = {k: {b: summ["all31"][k][b] / fd[b] for b in BANDS} for k in keys if k != "FDTD_raw"}
    holds = all(ratios["R0_120K_proj"][b] <= 2.0 for b in BANDS)
    rep = {"definition": __doc__.split("\n\n")[0].strip(), "rooms": len(per), "per_room": per, "summary": summ,
           "ratio_to_fdtd_all31": ratios,
           "reading": {"test": "projected R0 <= 2x FDTD in every band (means over the 31 rooms)",
                       "holds": bool(holds),
                       "consequence": ("projection already removes the interior term's target: the interior "
                                       "Helmholtz loss is CLOSED for good") if holds else
                                      ("projection does NOT remove the interior term's target in every band: "
                                       "the interior Helmholtz loss is not closed by this reading")}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "interior.json").write_text(json.dumps(rep, indent=1))

    def f3(d):
        return " / ".join("{:.3f}".format(d[b]) for b in BANDS)

    L = ["## (i) Interior Helmholtz residual (D180 0b(i))", "",
         "Relative residual ||lap F + k^2 (F - <F>)|| / ||k^2 (F - <F>)||, the S1 definition (`scripts/p4_8_s1_gate.py`), "
         "band means 25-100 / 100-200 / 200-300 Hz, mean over rooms. Projected = the render-time projection "
         "(|k| <= 1.5 k0). FDTD truth is the reference level (Rule 12).", "",
         "| field | all 31 | in-family 15 | unseen 16 | U 8 | C3 8 | x FDTD's (all 31) |", "|---|---|---|---|---|---|---|"]
    for k in keys:
        rr = "--" if k == "FDTD_raw" else " / ".join("{:.1f}x".format(ratios[k][b]) for b in BANDS)
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            k.replace("_", " "), f3(summ["all31"][k]), f3(summ["in_family15"][k]), f3(summ["unseen16"][k]),
            f3(summ["U8"][k]), f3(summ["C3_8"][k]), rr))
    L += ["", "**Pre-registered reading (D180):** projected R0 <= 2x FDTD's residual in every band -> **{}**. {}.".format(
        "HOLDS" if holds else "DOES NOT HOLD", rep["reading"]["consequence"][0].upper() + rep["reading"]["consequence"][1:]),
        "", "Per-room values: `outputs/p4_9/gates/interior.json`."]
    (OUT / "interior.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
