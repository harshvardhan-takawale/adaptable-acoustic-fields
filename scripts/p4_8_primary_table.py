"""P4-8 (D174(a), (e), (h)): the single primary table and the pre-registered verdicts.

Per arm-checkpoint, from the JSONs the watchers wrote (no rendering):
  HF  = in-family test r at 200-300 Hz on the 15 frozen shapes (p4_5_speckle per_shape), paired over shapes;
  UF  = unseen-family r at 25-100 (and 100-200, 200-300) Hz on the frozen 16-room set (p4_8_unseen_eval), paired
        over rooms; U and C3 reported separately; T / DN (D111) labelled "present in GX training";
  training-room fit r per band and the train-to-test gap per band (C250 off-lattice rooms), test LSD per band,
  modal RIR off-lattice (risk metrics), speckle d1 200-300 Hz raw / projected.
Paired bootstrap 10k, seed 20261013, against GX@60K (pre-registrations) and GX@30K (the 30K kill rule). D176: the
kill rule does NOT apply to LS2 (its one pre-registered read is its 30K end point vs GX@60K); LS2C is LS2's batching and
schedule with lambda = 0 (the control, run only if an LS2 arm passes); S2 is read at 60K and 120K (final) vs GX@60K.
Writes outputs/p4_8/primary_table.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

B = ["25-100", "100-200", "200-300"]
SEED, NB = 20261013, 10000
OUT = Path("outputs/p4_8")
REFS = ["C250_60K", "G1_60K", "GX_30K", "GX_60K", "PB3_60K", "B5_60K"]          # C250: start-of-phase reference (deck)
ARMS = (["R0_%dK" % k for k in (30, 60, 90, 120)]
        + ["%s_%dK" % (a, k) for a in ("MB1", "MB2", "LS1A", "LS1B", "LS3") for k in (30, 60)]
        + ["%s_%dK" % (a, k) for a in ("LS2A", "LS2B", "LS2C") for k in (15, 30)]
        + ["S2_60K", "S2_120K"])


def _j(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def boot(v):
    v = np.asarray(v, float)
    idx = np.random.default_rng(SEED).integers(0, len(v), (NB, len(v)))
    m = v[idx].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load(n):
    sp = _j("outputs/p4_5/speckle/{}.json".format(n))
    un = _j("outputs/p4_8/unseen/{}.json".format(n))
    if sp is None or un is None:
        return None
    a = {"name": n,
         "hf_shapes": sp["per_shape"]["r"]["200-300"], "test_r_shapes": {b: sp["per_shape"]["r"][b] for b in B},
         "test_r": {b: sp["summary"]["r"][b][0] for b in B}, "test_lsd": {b: sp["summary"]["lsd"][b][0] for b in B},
         "test_lsd_shapes": {b: sp["per_shape"]["lsd"][b] for b in B},
         "d1_raw": sp["summary"]["d1_off"]["200-300"][0],
         "d1_proj": sp["summary"].get("d1_off_proj", {}).get("200-300", [float("nan")])[0],
         "uf_rooms": {b: [r["raw"][b] for r in un["rooms"]] for b in B},
         "uf": {b: un["unseen_all"]["raw"][b] for b in B}, "U": {b: un["U"]["raw"][b] for b in B},
         "C3": {b: un["C3"]["raw"][b] for b in B}, "uf_proj": {b: un["unseen_all"]["projected"][b] for b in B}}
    fo = _j("outputs/p4_6/family_oof/{}.json".format(n))
    if fo:
        a["T"] = {b: fo["per_family"]["T"]["raw"][b] for b in B}
        a["DN"] = {b: fo["per_family"]["DN"]["raw"][b] for b in B}
    fit = _j("outputs/p4_5/train_offlat_fit/{}.json".format(n))
    if fit:
        a["fit"] = {"25-100": fit["r_25_100"], "100-200": fit["r_100_200"], "200-300": fit["r_200_300"]}
        a["gap"] = {b: a["fit"][b] - a["test_r"][b] for b in B}
    for d in ("outputs/p4_8", "outputs/p4_7", "outputs/p4_6"):
        rk = _j("{}/RISK_METRICS_{}.json".format(d, n)) or _j("{}/RISK_METRICS_60K.json".format(d))
        rows = [r for r in (rk or {}).get("rows", []) if r["name"] == n]
        if rows:
            a["rir_modal_off"] = rows[0]["rir_modal_off"]
            break
    return a


def paired(a, r):
    out = {"HF": boot(np.array(a["hf_shapes"]) - np.array(r["hf_shapes"]))}
    for b in B:
        out["UF_" + b] = boot(np.array(a["uf_rooms"][b]) - np.array(r["uf_rooms"][b]))
        out["test_" + b] = boot(np.array(a["test_r_shapes"][b]) - np.array(r["test_r_shapes"][b]))
        out["lsd_" + b] = boot(np.array(a["test_lsd_shapes"][b]) - np.array(r["test_lsd_shapes"][b]))
    return out


def verdict(arm, d, a):
    """The pre-registered tests of D174(e) against GX@60K (point estimates; CIs reported beside)."""
    hf, uf = d["HF"][0], d["UF_25-100"][0]
    kind = arm.split("_")[0]
    if kind == "S2":
        ok = uf >= 0.05 and a["test_r"]["200-300"] >= 0.504 - 0.02
        return {"test": "UF >= GX@60K + 0.05 AND HF >= 0.504 - 0.02", "holds": bool(ok)}
    if kind == "R0":
        ok = a["test_r"]["200-300"] >= 0.56 and d["UF_25-100"][2] >= 0.0
        return {"test": "HF >= 0.56 AND UF not below GX@60K within CI", "holds": bool(ok)}
    if kind in ("MB1", "MB2"):
        ok = (uf >= 0.05 and hf >= -0.02) or (hf >= 0.03 and uf >= -0.02)
        return {"test": "UF >= +0.05 & HF >= -0.02, or HF >= +0.03 & UF >= -0.02", "holds": bool(ok)}
    others = [d["test_" + b][0] for b in B] + [d["UF_" + b][0] for b in B]
    ok = (hf >= 0.03 or uf >= 0.05) and min(others) >= -0.02
    t = "HF >= +0.03 or UF >= +0.05, no primary / band more than 0.02 worse"
    if kind.startswith("LS2"):
        ok = ok and max(d["lsd_" + b][0] for b in B) <= 0.2
        t += ", band LSD <= +0.2 dB"
    return {"test": t, "holds": bool(ok)}


def kill(d):
    """30K kill rule: stop only if BOTH primaries are below the reference with CIs excluding 0."""
    return bool(d["HF"][2] < 0 and d["UF_25-100"][2] < 0)


def main():
    rows = {n: load(n) for n in REFS + ARMS}
    rows = {n: a for n, a in rows.items() if a}
    rep = {"rows": {}, "pairs": {}, "verdicts": {}, "kill": {}}
    for n, a in rows.items():
        rep["rows"][n] = {k: v for k, v in a.items() if not k.endswith("_shapes") and k != "uf_rooms"}
    for n, a in rows.items():
        if n in REFS:
            continue
        it = int(n.split("_")[1][:-1])
        if "GX_60K" in rows:
            rep["pairs"][n + " - GX_60K"] = d = paired(a, rows["GX_60K"])
            final = (n.startswith("LS2") and it == 30) or (n.startswith(("R0", "S2")) and it == 120) or \
                    (not n.startswith(("LS2", "R0", "S2")) and it == 60)
            if final or (n.startswith("S2") and it == 60):
                rep["verdicts"][n] = dict(verdict(n, d, a), **({} if final else {"read": "interim (60K of 120K)"}))
        early = (not n.startswith(("LS2", "S2"))) and it == 30          # D176: no kill rule for LS2 (or S2)
        if early and "GX_30K" in rows:
            rep["pairs"][n + " - GX_30K"] = d = paired(a, rows["GX_30K"])
            rep["kill"][n] = {"stop": kill(d), "HF": d["HF"], "UF_25-100": d["UF_25-100"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "primary_table.json").write_text(json.dumps(rep, indent=1))
    f3 = lambda d: "/".join("%.3f" % d[b] for b in B) if d else "--"
    L = ["# P4-8 primary table (D174(h))", "",
         "Off-lattice. HF = in-family test r 200-300 Hz (15 frozen shapes); UF = unseen-family r (8 U + 8 C3, frozen "
         "set) at 25-100 / 100-200 / 200-300 Hz. T / DN are present in GX-corpus training (CX500). Network-only "
         "(D174 scope).", "",
         "| model | HF | UF | U | C3 | test r (3 bands) | test LSD dB | fit r | gap | modal RIR | d1 raw / proj | T* | DN* |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, a in rows.items():
        L.append("| {} | {:.3f} | {} | {} | {} | {} | {} | {} | {} | {} | {:.3f} / {:.3f} | {} | {} |".format(
            n, a["test_r"]["200-300"], f3(a["uf"]), f3(a["U"]), f3(a["C3"]), f3(a["test_r"]),
            "/".join("%.2f" % a["test_lsd"][b] for b in B), f3(a.get("fit")), f3(a.get("gap")),
            "%.3f" % a["rir_modal_off"] if "rir_modal_off" in a else "--", a["d1_raw"], a["d1_proj"],
            f3(a.get("T")), f3(a.get("DN"))))
    L += ["", "**Paired differences** (mean [95% CI])", "", "| pair | HF | UF 25-100 | UF 100-200 | UF 200-300 |", "|---|---|---|---|---|"]
    ci = lambda v: "%+.3f [%+.3f, %+.3f]" % tuple(v)
    for k, d in rep["pairs"].items():
        L.append("| {} | {} | {} | {} | {} |".format(k, ci(d["HF"]), ci(d["UF_25-100"]), ci(d["UF_100-200"]), ci(d["UF_200-300"])))
    L += ["", "**Pre-registered verdicts:** " + json.dumps(rep["verdicts"]), "", "**30K kill rule:** " + json.dumps(
        {k: v["stop"] for k, v in rep["kill"].items()})]
    (OUT / "primary_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
