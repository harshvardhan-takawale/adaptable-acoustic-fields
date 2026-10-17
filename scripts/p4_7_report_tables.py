"""P4-7 (D171): the standard-report tables for every finished P4-7 arm, with paired bootstraps against the parent.

Reads the per-arm JSONs the watchers wrote (no rendering):
  dense Gate-2 + CI (GAP_CI_*), risk metrics (LSD < 100 Hz, modal / full RIR), per-band test r / LSD and
  speckle d1 (p4_5_speckle per_shape / summary), training-room fit (C250 or family off-lattice rooms),
  out of family (p4_6_family_oof, raw and projected).
Paired shape bootstrap over the 15 test shapes (10k, seed 20261013), and a ROOM bootstrap over the 6
out-of-family rooms (n = 6: reported, flagged as weak). Writes outputs/p4_7/report_tables.json and prints.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

B = ["25-100", "100-200", "200-300"]
SEED, NB = 20261013, 10000
OUT = Path("outputs/p4_7/report_tables.json")
FAM_FIT = {"AFAM", "G2", "M3FAM"}
PAIRS = [("AFAM_60K", "G2_60K"), ("AFAM_30K", "G2_30K"), ("A500_60K", "G1_60K"), ("A500_30K", "G1_30K"),
         ("GX_60K", "G1_60K"), ("GX_30K", "G1_30K"), ("G3_120K", "G3_60K"), ("G3_120K", "PB3_60K"),
         ("M3FAM_20K", "G2_60K")]
ARMS = ["AFAM_30K", "AFAM_60K", "A500_30K", "A500_60K", "GX_30K", "GX_60K", "G3_120K", "M3FAM_20K",
        "G1_30K", "G1_60K", "G2_30K", "G2_60K", "G3_60K", "PB3_60K"]


def _load(*paths):
    for p in paths:
        if Path(p).exists():
            return json.load(open(p))
    return None


def boot(diff):
    v = np.asarray(diff, float)
    idx = np.random.default_rng(SEED).integers(0, len(v), (NB, len(v)))
    m = v[idx].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def arm(n):
    stem = n.split("_")[0]
    out = {"name": n}
    sp = _load("outputs/p4_5/speckle/{}.json".format(n))
    if sp:
        out["test_r"] = {b: sp["summary"]["r"][b][0] for b in B}
        out["test_lsd"] = {b: sp["summary"]["lsd"][b][0] for b in B}
        out["test_r_per_shape"] = {b: sp["per_shape"]["r"][b] for b in B}
        out["d1_raw_200_300"] = sp["summary"]["d1_off"]["200-300"][0]
        if "d1_off_proj" in sp["summary"]:
            out["d1_proj_200_300"] = sp["summary"]["d1_off_proj"]["200-300"][0]
        out["abs_c_200_300"] = sp["summary"]["d2_ls_c_abs"]["200-300"][0]
    ci = _load("outputs/p4_7/GAP_CI_{}.json".format(n), "outputs/p4_6/GAP_CI_{}.json".format(n),
               "outputs/p4_6/GAP_CI_{}.json".format(n.split("_")[1]))
    if ci:
        rows = [r for r in ci["rows"] if r["checkpoint"] == n]
        if rows:
            r = rows[0]
            out["gate2"] = [r["off_in_slab"]] + list(r["off_in_ci95"])
            out["gate2_gap"] = r["gap_in"]
    rk = _load("outputs/p4_7/RISK_METRICS_{}.json".format(n), "outputs/p4_6/RISK_METRICS_{}.json".format(n),
               "outputs/p4_6/RISK_METRICS_{}.json".format(n.split("_")[1]))
    if rk:
        rows = [r for r in rk["rows"] if r["name"] == n]
        if rows:
            r = rows[0]
            out["lsd_0_100"] = r["lsd_0_100_off"]
            out["rir_modal_off"] = r["rir_modal_off"]
            out["rir_full_off"] = r["rir_fullband_off"]
    fit = _load(("outputs/p4_6/train_fit/{}.json" if stem in FAM_FIT else "outputs/p4_5/train_offlat_fit/{}.json").format(n))
    if fit:
        out["fit_r"] = {"25-100": fit["r_25_100"], "100-200": fit["r_100_200"], "200-300": fit["r_200_300"]}
        out["fit_lsd_0_100"] = fit["lsd_0_100"]
        out["fit_rooms"] = "family (20)" if stem in FAM_FIT else "C250 (off-lattice)"
        if "test_r" in out:
            out["gap"] = {b: out["fit_r"][b] - out["test_r"][b] for b in B}
    oof = _load("outputs/p4_6/family_oof/{}.json".format(n))
    if oof:
        pf = oof["per_family"]
        out["family"] = {f: {b: pf[f]["raw"][b] for b in B} for f in pf}
        out["family_proj"] = {f: {b: pf[f]["projected"][b] for b in B} for f in pf}
        out["oof_mean"] = {b: oof["shape_corpus_out_of_family_T_DN_U"]["raw"][b] for b in B}
        out["oof_mean_proj"] = {b: oof["shape_corpus_out_of_family_T_DN_U"]["projected"][b] for b in B}
        out["U_100Hz"] = pf["U"]["raw"]["r_100Hz"]
        out["oof_rooms"] = {r["room"]: {b: r["raw"][b] for b in B} for r in oof["rooms"] if r["family"] in ("T", "DN", "U")}
    return out


def main():
    arms = {n: arm(n) for n in ARMS}
    pairs = {}
    for a, p in PAIRS:
        A, P = arms.get(a, {}), arms.get(p, {})
        row = {}
        if "test_r_per_shape" in A and "test_r_per_shape" in P:
            row["test_r"] = {b: boot(np.array(A["test_r_per_shape"][b]) - np.array(P["test_r_per_shape"][b])) for b in B}
        if "oof_rooms" in A and "oof_rooms" in P and set(A["oof_rooms"]) == set(P["oof_rooms"]):
            ks = sorted(A["oof_rooms"])
            row["oof_rooms_n6"] = {b: boot([A["oof_rooms"][k][b] - P["oof_rooms"][k][b] for k in ks]) for b in B}
        if "family" in A and "family" in P:
            row["U"] = {b: A["family"]["U"][b] - P["family"]["U"][b] for b in B}
        pairs["{} - {}".format(a, p)] = row
    for a in arms.values():
        a.pop("test_r_per_shape", None)
    OUT.write_text(json.dumps({"arms": arms, "pairs": pairs, "bootstrap": "paired over 15 test shapes / 6 OOF rooms, "
                                                                        "10k, seed 20261013"}, indent=1))
    f3 = lambda d: "/".join("%.3f" % d[b] for b in B) if d else "--"
    print("arm        Gate2 [CI] gap | test r | LSD<100 | fit r (gap 200-300) | OOF T/DN/U mean | U | T | DN | rect | L | d1 raw/proj")
    for n, a in arms.items():
        if "test_r" not in a and "family" not in a:
            print(n, "(missing)"); continue
        g = a.get("gate2")
        print("{:10s} {} {} | {} | {} | {} ({}) | {} | {} | {} | {} | {} | {} | {}".format(
            n, "%.3f [%.3f, %.3f]" % tuple(g) if g else "--", "%+.3f" % a["gate2_gap"] if "gate2_gap" in a else "",
            f3(a.get("test_r")), "%.2f" % a["lsd_0_100"] if "lsd_0_100" in a else "--", f3(a.get("fit_r")),
            "%.3f" % a["gap"]["200-300"] if "gap" in a else "--", f3(a.get("oof_mean")),
            f3(a.get("family", {}).get("U")), f3(a.get("family", {}).get("T")), f3(a.get("family", {}).get("DN")),
            f3(a.get("family", {}).get("rect")), f3(a.get("family", {}).get("L")),
            "%.3f/%.3f" % (a["d1_raw_200_300"], a.get("d1_proj_200_300", float("nan"))) if "d1_raw_200_300" in a else "--"))
    for k, row in pairs.items():
        print(k, json.dumps({kk: {b: [round(x, 3) for x in v[b]] if isinstance(v[b], list) else round(v[b], 3) for b in B}
                             for kk, v in row.items()}))


if __name__ == "__main__":
    main()
