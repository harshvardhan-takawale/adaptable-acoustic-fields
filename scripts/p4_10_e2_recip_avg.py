"""P4-10 E2 (D206, D207): reciprocal averaging on R1's Source cells, zero training. For each Source cell (second and third source, all 31 rooms) and the original source: three renders
against the same truth, on the off-lattice receivers that are interior FDTD nodes:
  (a) normal  F(s', r): the stored prediction (scorecard pred for the cell; the original-room render for the original source);
  (b) swapped F(r, s'): receiver and source exchanged in the renderer, source = each receiver in turn;
  (c) average (a + b) / 2 of the complex fields.
r = the frozen scorer (band_r) per cell; per-room means; paired bootstrap over rooms (10k, seed 20261013). Pre-registered reading (D206): averaging is a P4-11 candidate only if (c) beats (a) by
>= 0.03 with CI above zero in a band above 100 Hz (100-200 or 200-300) at the third source, AND (c) loses no more than 0.01 to (a) at the original source (mean over rooms, any band is checked).
  --render --idx I   GPU: swapped renders of R1 for Source cells I*chunk ... and original rooms -> PRED/<name>/swap/<cid or room>.npz (chunk = --chunk cells per task)
  --score            CPU: outputs/p4_10/e2/report.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.p4_10_scorecard import BAND, PRED, Renderer, TRUTH, band_r, boot, cells, cid, on_lattice, stored

OUT = Path("outputs/p4_10/e2")
B = ("25-100", "100-200", "200-300")


def interior(verts, rx):
    from aaf.data.augment_2d import is_interior_node
    return np.asarray(is_interior_node([tuple(map(float, v)) for v in verts], rx, 0.02, 0.02))


def swap_render(rd, verts, L, W, alphas, src, rx, chunk=32):
    """renderer with receiver = src (fixed) and source = each rx in turn; [len(rx), 601]"""
    from aaf.models.conditioning_2d import build_cond_vector_2d
    t = rd.torch
    z = build_cond_vector_2d("geom_token", L, W, list(alphas), verts=[tuple(map(float, v)) for v in verts], device=rd.dev)
    lo, hi = t.zeros(2, device=rd.dev), t.tensor([L, W], device=rd.dev, dtype=t.float32)
    S = t.tensor(np.asarray(rx), dtype=t.float32, device=rd.dev)
    R = t.tensor([list(map(float, src))], dtype=t.float32, device=rd.dev)
    out = []
    with t.no_grad():
        for s0 in range(0, len(S), chunk):
            s = S[s0:s0 + chunk]
            out.append(rd.renderer(rd.model, R.expand(len(s), -1), s, lo, hi, z_s=z.unsqueeze(0).expand(len(s), -1), n_out=601).cpu().numpy())
    return np.concatenate(out).astype(np.complex64)


def render(name, ck, idx, chunk):
    M = cells(["Source"], False)
    M = M[idx * chunk:(idx + 1) * chunk]
    if not M:
        return
    rd = Renderer(name, ck)
    (PRED / name / "swap").mkdir(parents=True, exist_ok=True)
    for c in M:
        z = np.load(TRUTH / (cid(c) + ".npz"))
        base = stored(c)
        for key, verts, L, W, alphas, src, rx in (
                (cid(c), c["verts"], c["L"], c["W"], c["edge_alphas"], c["src"], z["rx"]),
                ("orig_" + c["room"], c["orig_verts"], c["orig_L"], c["orig_W"], [0.15] * len(c["orig_verts"]), (0.5, 1.6), base["rx"])):
            dst = PRED / name / "swap" / (key + ".npz")
            if dst.exists():
                continue
            k = np.nonzero(~on_lattice(rx) & interior(verts, rx))[0]
            Ps = swap_render(rd, verts, L, W, alphas, src, rx[k])
            tmp = dst.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, P=Ps, k=k, rx=rx[k])
            tmp.replace(dst)
            print("swap", key, len(k), flush=True)


def rowcells(name):
    """per cell/room: (a), (b), (c) r per band on the same receivers"""
    out = []
    for c in cells(["Source"], False):
        sw = PRED / name / "swap" / (cid(c) + ".npz")
        pr = PRED / name / (cid(c) + ".npz")
        if not sw.exists() or not pr.exists():
            continue
        z, p, s = np.load(TRUTH / (cid(c) + ".npz")), np.load(pr), np.load(sw)
        k = s["k"]
        T, Pa = z["Hd"][k], p["Pd"][k]
        Pb = s["P"]
        out.append({"id": c["id"], "room": c["room"], "group": c["group"], "mag": c["mag"], "n": int(len(k)),
                    **{lab: band_r(q, T) for lab, q in (("a", Pa), ("b", Pb), ("c", 0.5 * (Pa + Pb)))}})
    done = set()
    for c in cells(["Source"], False):
        if c["room"] in done:
            continue
        done.add(c["room"])
        so = PRED / name / "swap" / ("orig_" + c["room"] + ".npz")
        op = PRED / name / "orig" / (c["room"] + ".npz")
        if not so.exists() or not op.exists():
            continue
        base, o, s = stored(c), np.load(op), np.load(so)
        k = s["k"]
        T, Pa = base["H"][k], o["P"][k]
        Pb = s["P"]
        out.append({"id": "orig|" + c["room"], "room": c["room"], "group": c["group"], "mag": "original", "n": int(len(k)),
                    **{lab: band_r(q, T) for lab, q in (("a", Pa), ("b", Pb), ("c", 0.5 * (Pa + Pb)))}})
    return out


def score(name):
    rows = rowcells(name)
    rep = {"definition": __doc__.split("\n\n")[0], "name": name, "n_rows": len(rows), "strata": {}}
    L = ["# P4-10 E2: reciprocal averaging on {} (normal / swapped / average)".format(name), "",
         "| set | position | rooms | (a) normal | (b) swapped | (c) average | (c) - (a) [95% CI] |", "|---|---|---|---|---|---|---|"]
    for g, lab in (("test", "in-family 15"), ("unseen", "unseen 16"), (None, "all 31")):
        for mag in ("second", "third", "original"):
            sel = [r for r in rows if r["mag"] == mag and (g is None or r["group"] == g)]
            rooms = sorted({r["room"] for r in sel})
            if not rooms:
                continue
            e = {"n_rooms": len(rooms)}
            for lbl in ("a", "b", "c"):
                e[lbl] = {b: float(np.mean([np.mean([r[lbl][b] for r in sel if r["room"] == rm]) for rm in rooms])) for b in B}
            e["c_minus_a"] = {b: boot([np.mean([r["c"][b] - r["a"][b] for r in sel if r["room"] == rm]) for rm in rooms]) for b in B}
            e["b_minus_a"] = {b: boot([np.mean([r["b"][b] - r["a"][b] for r in sel if r["room"] == rm]) for rm in rooms]) for b in B}
            rep["strata"]["{}|{}".format(lab, mag)] = e
            f = lambda d: " / ".join("{:.3f}".format(d[b]) for b in B)  # noqa: E731
            L.append("| {} | {} | {} | {} | {} | {} | {} |".format(lab, mag, e["n_rooms"], f(e["a"]), f(e["b"]), f(e["c"]), " / ".join("{:+.3f} [{:+.3f}, {:+.3f}]".format(*e["c_minus_a"][b]) for b in B)))
    all3 = rep["strata"].get("all 31|third", {})
    orig = rep["strata"].get("all 31|original", {})
    if all3 and orig:
        gain = {b: bool(all3["c_minus_a"][b][0] >= 0.03 and all3["c_minus_a"][b][1] > 0) for b in ("100-200", "200-300")}
        harm = {b: bool(orig["c_minus_a"][b][0] >= -0.01) for b in B}
        rep["reading"] = {"gain_at_third_source_above_100Hz": gain, "no_harm_at_original_per_band": harm, "candidate": bool(any(gain.values()) and all(harm.values()))}
        L += ["", "**Pre-registered reading (D206; read on all 31 rooms, D207):** gain at the third source above 100 Hz {} ; original-source loss <= 0.01 in every band {} -> averaging is a P4-11 candidate: **{}**".format(
            gain, harm, "YES" if rep["reading"]["candidate"] else "NO")]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--name", default="R1_150K")
    ap.add_argument("--ck", default="outputs/p4_10/arms/R1/ckpt_iter0150000.pt")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4)
    a = ap.parse_args()
    if a.render:
        render(a.name, a.ck, a.idx, a.chunk)
    elif a.score:
        score(a.name)
