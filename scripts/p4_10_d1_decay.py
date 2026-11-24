"""P4-10 D1 (D201): the decay audit on the ORIGINAL rooms. Per octave (63 / 125 / 250 Hz), model vs truth: EDT, T20 (kept only where the
model's own decay clears its late floor by >= 10 dB at -25 dB, D202; dropped counts printed) and the late floor. Off-lattice receivers >= 1 m
from the source, valid in the truth (A7), every 4th receiver (documented); per room the receiver median, then the mean over rooms (bootstrap
10k, seed 20261013). Fields: cached renders (in-family 15: the dense Gate-2 set; unseen 16), truth from the same caches. No model is run here.
  python scripts/p4_10_d1_decay.py --model R0_120K      -> outputs/p4_10/d1/<model>.json
  python scripts/p4_10_d1_decay.py --report M1 M2 ...    -> outputs/p4_10/d1/report.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/p4_10/d1")
CACHE = {"test": "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz", "unseen": "outputs/p4_8/cache/unseen_{}.npz"}
BANDS3 = ("63", "125", "250")
SEED, NB = 20261013, 10000
SRC = np.array([0.5, 1.6])


def rooms_of(name, group):
    z = np.load(CACHE[group].format(name))
    i = 0
    while "P{}".format(i) in z.files:
        yield z["P%d" % i], z["T%d" % i], z["rx%d" % i]
        i += 1


def audit(name):
    from aaf.eval.octave_decay import decay_metrics, floor_mask, valid_mask
    from scripts.p4_10_scorecard import on_lattice
    rep = {"model": name, "groups": {}}
    for g in CACHE:
        rep["groups"][g] = []
        for j, (P, T, rx) in enumerate(rooms_of(name, g)):
            sel = np.nonzero(~on_lattice(rx) & (np.hypot(rx[:, 0] - SRC[0], rx[:, 1] - SRC[1]) >= 1.0))[0][::4]
            row = {"room": j, "n_rx": int(len(sel)), "bands": {}}
            for b in BANDS3:
                t, m = decay_metrics(np.asarray(T)[sel], b), decay_metrics(np.asarray(P)[sel], b)
                ok = valid_mask(t["t35"])
                kp = ok & floor_mask(m)
                e = {"n_valid": int(ok.sum()), "n_t20_kept": int(kp.sum())}
                for k, d in (("truth", t), ("model", m)):
                    e["EDT_" + k] = float(np.nanmedian(d["EDT"][ok])) if ok.any() else np.nan
                    e["floor_db_" + k] = float(np.nanmedian(d["floor_db"][ok])) if ok.any() else np.nan
                    e["T20_" + k] = float(np.nanmedian(d["T20"][kp])) if kp.any() else np.nan
                    e["T30_truth"] = float(np.nanmedian(t["T30"][ok])) if ok.any() else np.nan
                rel = lambda x, y, mask: float(np.nanmedian(np.abs(x[mask] - y[mask]) / np.maximum(y[mask], 1e-6))) if mask.any() else np.nan  # noqa: E731
                e["EDT_err"] = rel(m["EDT"], t["EDT"], ok)
                e["EDT_bias"] = float(np.nanmedian((m["EDT"][ok] - t["EDT"][ok]) / np.maximum(t["EDT"][ok], 1e-6))) if ok.any() else np.nan
                e["T20_err"] = rel(m["T20"], t["T20"], kp)
                row["bands"][b] = e
            rep["groups"][g].append(row)
        print(name, g, len(rep["groups"][g]), "rooms", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (name + ".json")).write_text(json.dumps(rep, indent=1))


def boot(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not len(v):
        return [float("nan")] * 3
    m = v[np.random.default_rng(SEED).integers(0, len(v), (NB, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def report(models):
    rep = {"models": models, "definition": __doc__.split("\n")[0], "groups": {}}
    L = ["# P4-10 D1: decay on the original rooms (model vs truth)", "", "JNDs: EDT 5% (ISO 3382-1), about 18% measured; T20 not used as a headline. Room mean of receiver medians, 95% CI over rooms.",
         "", "| model | rooms | octave | EDT truth / model (s) | EDT error (median, %) | EDT bias (%) | T20 kept receivers | T20 error | floor truth / model (dB) |", "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        d = json.loads((OUT / (m + ".json")).read_text())
        for g in ("test", "unseen"):
            rows = d["groups"].get(g, [])
            if not rows:
                continue
            for b in BANDS3:
                q = lambda k: [r["bands"][b][k] for r in rows]  # noqa: E731
                e = {k: boot(q(k)) for k in ("EDT_truth", "EDT_model", "EDT_err", "EDT_bias", "T20_err", "floor_db_truth", "floor_db_model")}
                kept, valid = int(sum(q("n_t20_kept"))), int(sum(q("n_valid")))
                rep["groups"].setdefault(m, {}).setdefault(g, {})[b] = {**e, "n_t20_kept": kept, "n_valid": valid}
                L.append("| {} | {} | {} | {:.3f} / {:.3f} | {:.1%} [{:.1%}, {:.1%}] | {:+.1%} | {} of {} | {} | {:.0f} / {:.0f} |".format(
                    m, "in-family" if g == "test" else "unseen", b, e["EDT_truth"][0], e["EDT_model"][0], *e["EDT_err"], e["EDT_bias"][0], kept, valid,
                    "{:.0%}".format(e["T20_err"][0]) if np.isfinite(e["T20_err"][0]) else "n/a", e["floor_db_truth"][0], e["floor_db_model"][0]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--report", nargs="*")
    a = ap.parse_args()
    if a.model:
        audit(a.model)
    elif a.report:
        report(a.report)
