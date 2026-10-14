"""P4-5, D112 items 1-3: does MEAN pooling recover the family numbers off the lattice, is it
invariant under re-tokenisation, and does the D109(f) candidate survive ten rooms?

Two stages, so the GPU part is one render per (model, room, tokenisation) and every statistic is
recomputed on CPU from the cache:

  --render NAME=CKPT   render the ten dense family rooms (D111's five median rooms + D112's
                       second median room per family) at every tokenisation the D84 control
                       uses, and -- for rooms with published FAM values -- the ORIGINAL 800
                       receivers too, so FAM's published numbers are reproduced before use.
  --analyse            D112(a) recovery fraction, (b) token cost with paired receiver-bootstrap
                       CIs, (c) Spearman(reflex count, FAM shift). Writes FAMILY_POOLING.json.

Metric: `p4_4_family_eval._metrics` (6 bounding-box modes, dB spatial Pearson) -- D111's, verbatim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_p4_5_rooms import OUT, family_test_rooms
from scripts.p4_5_family_dense import PUB, TOL, load, onlat

CACHE = Path("outputs/p4_5/gate2_dense/cache")
OUTJ = Path("outputs/p4_5/gate2_dense/FAMILY_POOLING.json")
REFLEX = {"rect": 0, "L": 1, "T": 2, "DN": 2, "U": 2}
IN_DIST_FOR_L_FAMILY = ("rect", "L")          # D112(a): what Arm D / B7 were trained on
NB_TOKEN = 2000


def rooms():
    return ([(f, c, 1) for f, c in family_test_rooms()]
            + [(f, c, 2) for f, c in family_test_rooms(second=True)])


def ks_for(kind):
    from scripts.p4_4_family_eval import REFINE
    return (1,) + tuple(k for k in REFINE.get(kind, ()) if k != 1)


def render(name, ck):
    import torch

    from aaf.eval.modal_projection import enumerate_modes  # noqa: F401  (import check)
    from aaf.eval.p3_2_eval import load_model
    from scripts.p4_4_family_eval import MAX_SEG_POLY, _render, refine_verts
    dev = torch.device("cuda")
    model, renderer, cfg, _m, it = load_model(Path(ck), dev)
    model.eval(); renderer.eval()
    # Rule 8 / D73: the pooling the checkpoint was trained with must be the one it renders with.
    assert getattr(model, "token_pool", None) == cfg.get("token_pool"), (model.token_pool, cfg.get("token_pool"))
    assert cfg.get("cond_source") == "geom_token", cfg.get("cond_source")
    blob = {"key": "|".join([name, str(ck)]), "iter": int(it), "token_pool": str(cfg.get("token_pool"))}
    for fam, c, pick in rooms():
        Td, rxd, losd = load(Path(OUT["famdense"]) / c.filename)
        T8 = rx8 = None
        p8 = Path("data/track_p4_4_family") / c.filename
        if p8.exists():
            T8, rx8, _l = load(p8)
        for k in ks_for(c.kind):
            v = refine_verts(c.verts, k)
            assert len(v) <= MAX_SEG_POLY
            tag = "{}__k{}".format(c.label, k)
            blob["P_" + tag] = _render(model, renderer, c, rxd, Td.shape[1], dev, verts=v)
            if T8 is not None and name == "FAM":
                blob["P8_" + tag] = _render(model, renderer, c, rx8, T8.shape[1], dev, verts=v)
            print("  {} {:5s} {:52s} k{} ({} tokens) rendered".format(name, fam, c.label, k, len(v)), flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "family_pooling_{}.npz".format(name), **blob)     # complex128: exact vs D111
    print("wrote", CACHE / "family_pooling_{}.npz".format(name))


def _sp(P, T, los, modes):
    from scripts.p4_4_family_eval import _metrics
    return _metrics(P, T, los, modes)["spatial_pearson"]


def _sp_fast(Pdb, Tdb):
    """Mean over modes of the per-mode Pearson -- `_metrics`'s spatial_pearson, on precomputed dB
    columns (n_rx x n_modes), so the bootstrap does not redo FFT-free work 2000 times."""
    a = Pdb - Pdb.mean(0); b = Tdb - Tdb.mean(0)
    r = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return float(np.nanmean(r))


def analyse(names):
    from aaf.eval.modal_projection import enumerate_modes
    from scripts.p4_4_family_eval import DF_HZ, N_MODES, _db
    pubj = json.load(open(PUB))
    pub = {r["label"]: r["spatial_pearson"] for r in pubj["per_room"]}
    pubk = {(r["label"], r["refine_k"]): r["spatial_pearson"] for r in pubj["token_control_rooms"]}
    Z = {n: np.load(CACHE / "family_pooling_{}.npz".format(n), allow_pickle=False) for n in names}
    rep = {"metric": "p4_4_family_eval._metrics spatial_pearson, N_MODES = {}".format(N_MODES),
           "models": {n: {"iter": int(Z[n]["iter"]), "token_pool": str(Z[n]["token_pool"])} for n in names},
           "rooms": [], "token_cost": [], "reproduction": {}}
    rng = np.random.default_rng(20261014)
    worst = worst_d111 = 0.0
    d111 = {r["room"]: r for r in json.load(open("outputs/p4_5/gate2_dense/FAMILY_DENSE_BOUND.json"))["rooms"]}
    for fam, c, pick in rooms():
        Td, rxd, losd = load(Path(OUT["famdense"]) / c.filename)
        on = onlat(rxd)
        modes = enumerate_modes(c.L, c.W, f_max=200.0)[:N_MODES]
        bins = [int(round(m.f / DF_HZ)) for m in modes if int(round(m.f / DF_HZ)) < Td.shape[1]]
        T8 = los8 = None
        if (Path("data/track_p4_4_family") / c.filename).exists():
            T8, _rx8, los8 = load(Path("data/track_p4_4_family") / c.filename)
        row = {"family": fam, "room": c.label, "pick": pick, "reflex": REFLEX[fam],
               "in_dist_for_L_family_models": fam in IN_DIST_FOR_L_FAMILY,
               "fam_published_800rx": pub.get(c.label), "n_dense": int(len(rxd)), "n_on": int(on.sum())}
        for n in names:
            P = Z[n]["P_{}__k1".format(c.label)]
            row[n] = {"on": _sp(P[on], Td[on], losd[on], modes), "off": _sp(P[~on], Td[~on], losd[~on], modes)}
            # the bootstrap's fast Pearson must BE the evaluator's (to summation-order rounding, ~2e-9)
            assert abs(_sp_fast(_db(P[~on][:, bins]), _db(Td[~on][:, bins])) - row[n]["off"]) < 1e-7
            if n == "FAM" and T8 is not None:
                r8 = _sp(Z[n]["P8_{}__k1".format(c.label)], T8, los8, modes)
                worst = max(worst, abs(r8 - pub[c.label]))
                row[n]["reproduced_800rx"] = r8
        if "FAM" in names and c.label in d111:
            # D111's own numbers, from its own run: the same model on the same data must agree
            for q in ("on", "off"):
                worst_d111 = max(worst_d111, abs(row["FAM"][q] - d111[c.label]["dense_" + q]))
        if "FAM" in names:
            row["fam_shift_off_minus_published"] = row["FAM"]["off"] - pub[c.label]
            den = pub[c.label] - row["FAM"]["off"]
            for n in names:
                if n != "FAM":
                    rho = (row[n]["off"] - row["FAM"]["off"]) / den
                    row[n]["recovery_rho"] = rho
                    row[n]["reading"] = "RECOVERED" if rho >= 0.75 else ("GONE" if rho <= 0.25 else "PARTIAL")
        rep["rooms"].append(row)
        # ---- (b) token cost, off-lattice, paired receiver bootstrap ----
        ks = [k for k in ks_for(c.kind) if k != 1]
        if not ks:
            continue
        off = np.flatnonzero(~on)
        Tdb = _db(Td[off][:, bins])
        for n in names:
            P1db = _db(Z[n]["P_{}__k1".format(c.label)][off][:, bins])
            for k in ks:
                Pkdb = _db(Z[n]["P_{}__k{}".format(c.label, k)][off][:, bins])
                cost = _sp_fast(Pkdb, Tdb) - _sp_fast(P1db, Tdb)
                bs = []
                for _ in range(NB_TOKEN):
                    i = rng.integers(0, len(off), len(off))
                    bs.append(_sp_fast(Pkdb[i], Tdb[i]) - _sp_fast(P1db[i], Tdb[i]))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                t = {"model": n, "family": fam, "room": c.label, "k": k,
                     "tokens": "{}->{}".format({"rect": 4, "L": 6}[fam], {"rect": 4, "L": 6}[fam] * k),
                     "off_lattice_cost": float(cost), "ci95": [float(lo), float(hi)]}
                if n == "FAM" and (c.label, k) in pubk:
                    t["published_cost_800rx"] = pubk[(c.label, k)] - pubk[(c.label, 1)]
                rep["token_cost"].append(t)
    rep["reproduction"] = {"fam_800rx_max_abs_error": worst, "fam_dense_vs_D111_max_abs_error": worst_d111,
                           "tolerance": TOL, "ok": bool(worst <= TOL and worst_d111 <= TOL)}
    # ---- (a) summary ----
    for n in names:
        if n == "FAM":
            continue
        for grp, sel in (("in_distribution_rect_L", True), ("zero_shot_T_DN_U", False)):
            rs = [r for r in rep["rooms"] if r["in_dist_for_L_family_models"] == sel]
            rep["models"][n][grp] = {
                "mean_off": float(np.mean([r[n]["off"] for r in rs])),
                "fam_mean_off": float(np.mean([r["FAM"]["off"] for r in rs])),
                "fam_mean_published": float(np.mean([r["fam_published_800rx"] for r in rs])),
                "mean_rho": float(np.mean([r[n]["recovery_rho"] for r in rs])),
                "readings": [r[n]["reading"] for r in rs]}
        tc = [t for t in rep["token_cost"] if t["model"] == n]
        if tc:
            rep["models"][n]["token_invariant_D112b"] = bool(all(abs(t["off_lattice_cost"]) <= 0.02 for t in tc))
    # ---- (c) D109(f) candidate vs ten rooms ----
    if "FAM" in names:
        x = np.array([r["reflex"] for r in rep["rooms"]], float)
        y = np.array([r["fam_shift_off_minus_published"] for r in rep["rooms"]])
        from scipy.stats import spearmanr                                 # handles the tied counts
        rho_s, p_s = spearmanr(x, y)
        rect = [r["fam_shift_off_minus_published"] for r in rep["rooms"] if r["family"] == "rect"]
        rest = [r["fam_shift_off_minus_published"] for r in rep["rooms"] if r["family"] != "rect"]
        rep["d109_candidate"] = {"spearman_reflex_vs_shift": float(rho_s), "p": float(p_s),
                                 "candidate_predicts": "negative",
                                 "rect_shifts": rect, "other_mean_shift": float(np.mean(rest)),
                                 "rect_mean_shift": float(np.mean(rect))}
    json.dump(rep, open(OUTJ, "w"), indent=1, default=float)
    _print(rep, names)


def _print(rep, names):
    print("reproduction: FAM published 800-rx max |dr| {:.1e}; FAM dense vs D111 max |dr| {:.1e} -> {}".format(
        rep["reproduction"]["fam_800rx_max_abs_error"], rep["reproduction"]["fam_dense_vs_D111_max_abs_error"],
        "OK" if rep["reproduction"]["ok"] else "FAIL"))
    hdr = "{:5s} {:2s} {:46s} {:>7s}".format("fam", "#", "room", "FAMpub")
    for n in names:
        hdr += " | {:>6s} on {:>6s} off".format(n, n)
    print(hdr + " | rho per non-FAM model")
    for r in rep["rooms"]:
        s = "{:5s} {:2d} {:46s} {:+7.3f}".format(r["family"], r["pick"], r["room"][:46], r["fam_published_800rx"])
        for n in names:
            s += " | {:+9.3f} {:+10.3f}".format(r[n]["on"], r[n]["off"])
        s += " | " + " ".join("{} {:+.2f} {}".format(n, r[n]["recovery_rho"], r[n]["reading"]) for n in names if n != "FAM")
        print(s)
    for n in names:
        if n == "FAM":
            continue
        for grp in ("in_distribution_rect_L", "zero_shot_T_DN_U"):
            g = rep["models"][n][grp]
            print("{} {:24s}: off {:+.3f} vs FAM off {:+.3f} (FAM published {:+.3f}); mean rho {:+.2f}  {}".format(
                n, grp, g["mean_off"], g["fam_mean_off"], g["fam_mean_published"], g["mean_rho"], g["readings"]))
    print("\ntoken cost, OFF-lattice [paired receiver-bootstrap 95% CI]")
    for t in rep["token_cost"]:
        print("  {:4s} {:4s} {:34s} {:6s} {:+.3f} [{:+.3f}, {:+.3f}]{}".format(
            t["model"], t["family"], t["room"][:34], t["tokens"], t["off_lattice_cost"], *t["ci95"],
            "   (published 800-rx {:+.3f})".format(t["published_cost_800rx"]) if "published_cost_800rx" in t else ""))
    for n in names:
        if "token_invariant_D112b" in rep["models"][n]:
            print("{}: D112(b) INVARIANT (all |cost| <= 0.02)? {}".format(n, rep["models"][n]["token_invariant_D112b"]))
    if "d109_candidate" in rep:
        d = rep["d109_candidate"]
        print("\nD109(f) candidate: Spearman(reflex count, FAM shift) {:+.3f} (p {:.2f}; candidate predicts negative); "
              "rect mean shift {:+.3f} vs others {:+.3f}".format(d["spearman_reflex_vs_shift"], d["p"],
                                                               d["rect_mean_shift"], d["other_mean_shift"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default=None, help="NAME=CKPT")
    ap.add_argument("--analyse", default=None, help="comma-separated model names, FAM first")
    a = ap.parse_args()
    if a.render:
        n, ck = a.render.split("=", 1)
        render(n, ck)
    if a.analyse:
        analyse(a.analyse.split(","))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
