"""P4-9 S-3 (D181 E8; D182(b)): a second source position for the 16 frozen unseen rooms -- descriptive, no pass rule.

The frozen rooms use source (0.5, 1.6) (D194: R0's corpus itself has one source PER ROOM, 499 distinct in CX500). Per room, `data/track_p4_9_unseen_src2`
holds the FDTD at a second source (D182(b): the first of (L - 0.5, 1.6), (L / 2, 1.0), (L / 2, W / 2) on an air node
>= 0.3 m from every wall) and the SAME dense receivers. Scored exactly as `p4_8_unseen_eval` (off-lattice, Rule 11; raw
and after the render-time projection, POST-PROCESSING; band-mean r over 25-100 / 100-200 / 200-300 Hz and band LSD):
  network items   rendered at the second source (the renderer's source input; nothing else changes); the original-source
                  field is the standard report's cache (`outputs/p4_8/cache/unseen_<name>.npz`), so both sources are
                  scored by this script on identical receivers.
  m1              the exact-geometry eigenpairs (`outputs/p4_9/modal/<room>__exact.npz`, psi at both sources).
Paired per-room delta (second - original source), paired bootstrap over the 16 rooms (10k, seed 20261013).
  python scripts/p4_9_src2_eval.py --items "R0_120K=outputs/p4_8/arms/R0/ckpt_iter0120000.pt,m1=m1"
  -> outputs/p4_9/src2/<name>.json ; python scripts/p4_9_src2_eval.py --report -> outputs/p4_9/src2.md
X3 (D191): `--set test` scores the 15 in-family test shapes at their second source (`data/track_p4_9_test_src2`)
-> outputs/p4_9/src2_test/<name>.json, report outputs/p4_9/src2_test.md. A network item with no standard-report cache
for the original source is rendered at it here (same function, same receivers).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.physical_projection import project_field
from scripts.p4_4_family_eval import N_MODES
from scripts.p4_6_family_oof import BANDS, scores
from scripts.p4_8_unseen_eval import DATA as SRC1_DATA
from scripts.p4_8_unseen_eval import rooms

SRC2_DATA = Path("data/track_p4_9_unseen_src2")
MODAL = Path("outputs/p4_9/modal")
OUT = Path("outputs/p4_9/src2")
SETS = {"unseen": {"src1": SRC1_DATA, "src2": SRC2_DATA, "out": OUT, "md": Path("outputs/p4_9/src2.md"),
                   "cache": "outputs/p4_8/cache/unseen_{}.npz", "fams": ("U", "C3")},
        "test": {"src1": Path("data/track_p4_5_testdense"), "src2": Path("data/track_p4_9_test_src2"),
                 "out": Path("outputs/p4_9/src2_test"), "md": Path("outputs/p4_9/src2_test.md"),
                 "cache": "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz", "fams": ()}}


def set_rooms(which):
    """(configs, manifest sha) for a set; test configs carry what `render_at` and `_score` read."""
    if which == "unseen":
        return rooms()
    from types import SimpleNamespace

    from scripts.p4_8_s1_gate import rooms_test
    out = []
    for _, p in rooms_test():
        with h5py.File(p) as f:
            verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
            out.append(SimpleNamespace(L=float(f.attrs["L"]), W=float(f.attrs["W"]), verts=verts, filename=p.name,
                                       label=p.stem, family="test"))
    return out, None
SEED, NBOOT = 20261013, 10000


def on_lattice(rx):
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def render_at(model, renderer, c, rx, src, dev, n_bins=601, chunk=8):
    """`p4_4_family_eval._render` with the source as an argument (the standard 4097-bin path)."""
    import torch

    from aaf.models.conditioning_2d import build_cond_vector_2d
    cond = build_cond_vector_2d("geom_token", c.L, c.W, [0.15] * len(c.verts), verts=list(c.verts), device=dev)
    room_min = torch.zeros(2, device=dev)
    room_max = torch.tensor([c.L, c.W], device=dev, dtype=torch.float32)
    R = torch.tensor(rx, dtype=torch.float32, device=dev)
    S = torch.tensor(np.asarray(src, float), dtype=torch.float32, device=dev).view(1, 2)
    P = np.zeros((len(rx), n_bins), np.complex128)
    for s0 in range(0, len(rx), chunk):
        r = R[s0:s0 + chunk]
        with torch.no_grad():
            H = renderer(model, r, S.expand(r.shape[0], -1), room_min, room_max,
                         z_s=cond.unsqueeze(0).expand(r.shape[0], -1))
        P[s0:s0 + chunk] = H[:, :n_bins].cpu().numpy()
    return P


def _load(path):
    with h5py.File(path) as f:
        return (np.asarray(f["ism/H_complex"])[:, :601], np.asarray(json.loads(f.attrs["receiver_pos"])),
                np.asarray(json.loads(f.attrs["line_of_sight"]), bool),
                np.asarray(json.loads(f.attrs["lattice_keep"]), bool).reshape(64, 64),
                np.asarray(json.loads(f.attrs["source_pos"]), float))


def _score(P, T, los, keep, c, off):
    modes = enumerate_modes(c.L, c.W, f_max=200.0)[:N_MODES]
    Pp = project_field(P, keep, (c.L - 0.24) / 63, (c.W - 0.24) / 63)
    return {"raw": scores(P[off], T[off], los[off], modes), "projected": scores(Pp[off], T[off], los[off], modes)}


def evaluate(name, ck, which="unseen"):
    S = SETS[which]
    rs, sha = set_rooms(which)
    rep = {"name": name, "checkpoint": ck, "manifest_sha": sha, "set": which, "rooms": []}
    if ck != "m1":
        import torch

        from aaf.eval.p3_2_eval import load_model
        dev = torch.device("cuda")
        model, renderer, *_ = load_model(Path(ck), dev)
        model.eval()
        renderer.eval()
        cp = Path(S["cache"].format(name))
        cache = np.load(cp) if cp.exists() else None
        rep["src1_from"] = str(cp) if cache is not None else "rendered here"
    else:
        from scripts.p4_9_sensitivity import modal_field
    for j, c in enumerate(rs):
        T1, rx1, los1, keep1, s1 = _load(S["src1"] / c.filename)
        T2, rx2, los2, keep2, s2 = _load(S["src2"] / c.filename)
        assert np.allclose(rx1, rx2) and (keep1 == keep2).all(), c.filename
        off = ~on_lattice(rx1)
        if ck == "m1":
            z = dict(np.load(MODAL / "{}__exact.npz".format(Path(c.filename).stem), allow_pickle=True))
            P1, P2 = modal_field(z), modal_field(z, second_source=True)
        elif cache is not None:
            assert np.allclose(cache["rx%d" % j], rx1)
            P1 = cache["P%d" % j][:, :601].astype(np.complex128)
            P2 = render_at(model, renderer, c, rx1, s2, dev)
        else:
            P1 = render_at(model, renderer, c, rx1, s1, dev)
            P2 = render_at(model, renderer, c, rx1, s2, dev)
        row = {"family": c.family, "room": c.label, "src1": s1.tolist(), "src2": s2.tolist(), "n_off": int(off.sum()),
               "s1": _score(P1, T1, los1, keep1, c, off), "s2": _score(P2, T2, los2, keep2, c, off),
               "ignore_move": _score(T1, T2, los2, keep2, c, off)}
        rep["rooms"].append(row)
        print("{:8s} {:3s} {:56s} src1 {} src2 {} raw r src1 {:.3f}/{:.3f}/{:.3f} src2 {:.3f}/{:.3f}/{:.3f}".format(
            name, c.family, c.label, s1.tolist(), s2.tolist(), *[row["s1"]["raw"][b] for b in BANDS],
            *[row["s2"]["raw"][b] for b in BANDS]), flush=True)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rep["rooms"]), (NBOOT, len(rep["rooms"])))
    for v in ("raw", "projected"):
        rep[v] = {}
        for b in BANDS:
            a1 = np.array([r["s1"][v][b] for r in rep["rooms"]])
            a2 = np.array([r["s2"][v][b] for r in rep["rooms"]])
            d = a2 - a1
            bs = d[idx].mean(1)
            rep[v][b] = {"src1": float(a1.mean()), "src2": float(a2.mean()), "delta": float(d.mean()),
                         "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                         "lsd_src1": float(np.mean([r["s1"][v]["lsd_" + b] for r in rep["rooms"]])),
                         "lsd_src2": float(np.mean([r["s2"][v]["lsd_" + b] for r in rep["rooms"]]))}
            rep[v][b]["ignore_move_vs_src2"] = float(np.mean([r["ignore_move"][v][b] for r in rep["rooms"]]))
            for fam in S["fams"]:
                rep[v][b][fam] = {s: float(np.mean([r[s][v][b] for r in rep["rooms"] if r["family"] == fam]))
                                  for s in ("s1", "s2")}
    S["out"].mkdir(parents=True, exist_ok=True)
    (S["out"] / "{}.json".format(name)).write_text(json.dumps(rep, indent=1))
    print(name, {b: (round(rep["raw"][b]["src1"], 3), round(rep["raw"][b]["src2"], 3)) for b in BANDS}, flush=True)


def report(which="unseen"):
    S = SETS[which]
    OUT = S["out"]
    title = ("# P4-9 S-3: a second source position on the 16 frozen unseen rooms (D181 E8, D182(b); descriptive)"
             if which == "unseen" else
             "# P4-9 X3 (D191(b)(ii)): a second source position on the 15 in-family test shapes (descriptive)")
    lines = [title, "",
             "Each training room has ONE source (D194; R0's CX500 half has 499 distinct ones). Here each frozen room is re-simulated with a second "
             "source (D182(b)) on the same dense receivers; off-lattice r (Rule 11), mean over the set's rooms, band-mean per-bin "
             "Pearson of dB. Delta = second - original source, paired bootstrap over rooms (10k, seed 20261013).", ""]
    for v in ("raw", "projected"):
        lines += ["**{}** (r: original source -> second source; delta [95% CI]; LSD dB original / second)".format(v), "",
                  "| field | 25-100 Hz | 100-200 Hz | 200-300 Hz |", "|---|---|---|---|"]
        for p in sorted([q for q in OUT.glob("*.json") if not q.stem.startswith("diag")], key=lambda q: (q.stem != "m1", q.stem)):
            d = json.load(open(p))
            cells = []
            for b in BANDS:
                e = d[v][b]
                cells.append("{:.3f} -> {:.3f}; {:+.3f} [{:+.3f}, {:+.3f}]; {:.1f} / {:.1f}".format(
                    e["src1"], e["src2"], e["delta"], *e["ci95"], e["lsd_src1"], e["lsd_src2"]))
            lines.append("| {} | {} |".format(d["name"], " | ".join(cells)))
        lines.append("")
    dg = OUT / "diag_source_blind.json"
    if dg.exists():
        m = json.load(open(dg))["mean"]
        lines += ["**Reference fields** (`scripts/p4_9_src2_diag.py`; raw, off-lattice, scored against the SECOND-source "
                  "truth): R0's ORIGINAL-source render {} ; the original-source TRUTH (a field that ignores the move) "
                  "{} (25-100 / 100-200 / 200-300 Hz).".format(
                      " / ".join("{:.3f}".format(m["R0src1_vs_T2"][b]) for b in BANDS),
                      " / ".join("{:.3f}".format(m["T1_vs_T2"][b]) for b in BANDS)), ""]
    ign = [json.load(open(p)) for p in sorted(OUT.glob("*.json")) if not p.stem.startswith("diag")]
    ign = [d for d in ign if "ignore_move_vs_src2" in d["raw"][next(iter(BANDS))]]
    if ign:
        lines += ["**'Ignore the move'** (the original-source truth scored against the second-source truth, raw): " +
                  " / ".join("{:.3f}".format(ign[0]["raw"][b]["ignore_move_vs_src2"]) for b in BANDS), ""]
    lines += ["Per room (sources, families, both variants): `{}/<name>.json`.".format(OUT)]
    S["md"].write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--set", default="unseen", choices=sorted(SETS))
    a = ap.parse_args()
    for item in [s for s in a.items.replace("+", ",").split(",") if s]:
        name, ck = item.split("=", 1)
        evaluate(name, ck, a.set)
    if a.report:
        report(a.set)


if __name__ == "__main__":
    main()
