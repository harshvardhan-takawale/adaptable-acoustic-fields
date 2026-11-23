"""P4-10 Step 1: the editability scorecard (D195, D197 A6 / A8 / A12).

Edit operator: geometry = re-encode the edited polygon's tokens; material = per-edge m in the tokens; source = the source
input. All renders use the 601-bin path.
  --render --model NAME=CKPT [--families F+F] [--lite] --idx I --chunk N
        GPU: cell (dense + common receivers) and the model's original render -> scratch/pred/NAME/<cell id>.npz
  --score --model NAME [--families ...] [--lite]
        CPU: per cell and band: edited-field r / LSD (dense, off-lattice), r_delta, g_delta, ignore-the-edit, warp
        (Stretch), m1 reference (10 cm, 350 Hz; geometry and source cells) -> scratch/score/NAME/<cell id>.json
  --aggregate --models A+B      per family, group, band: mean over rooms of each room's average over magnitudes, paired CIs
                                (10k, seed 20261013) -> outputs/p4_10/scorecard_<tag>.{json,md}
r_delta: per-bin spatial Pearson over the common receivers of delta-dB = dB(edited) - dB(original), clipped to +/-20 dB,
band mean. g_delta = LSD(model's original render vs edited truth) / LSD(model's edited render vs edited truth) on the same
receivers; > 1 means the edit moved the model toward the truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.p4_10_truth import OUT as TRUTH, cid, on_lattice, stored

MAN = Path("configs/p4_10/edit_manifest.json")
PRED = Path("/fs/nexus-scratch/htakawal/aaf_p410/pred")
SC = Path("/fs/nexus-scratch/htakawal/aaf_p410/score")
OUT = Path("outputs/p4_10")
BAND = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
FAMS = ("StretchU", "StretchA", "Wall", "Notch", "Source", "Material")
SEED, NB = 20261013, 10000


def cells(families=None, lite=False):
    M = json.loads(MAN.read_text())["cells"]
    M = [c for c in M if families is None or c["family"] in families]
    if lite:
        tr = sorted({c["room"] for c in M if c["group"] == "test"})[::2]
        u = sorted({c["room"] for c in M if c["group"] == "unseen" and c["room_family"] == "U"})[::2][:4]
        c3 = sorted({c["room"] for c in M if c["group"] == "unseen" and c["room_family"] == "C3"})[::2][:4]
        keep = set(tr) | set(u) | set(c3)
        M = [c for c in M if c["lite"] and c["room"] in keep]
    return M


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def colr(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def band_r(P, T):
    r = colr(db(P), db(T))
    return {b: float(np.nanmean(r[lo:hi])) for b, (lo, hi) in BAND.items()}


def band_lsd(P, T):
    return {b: float(np.mean(np.abs(db(P[:, lo:hi]) - db(T[:, lo:hi])))) for b, (lo, hi) in BAND.items()}


# ------------------------------------------------------------------------------------------- rendering
class Renderer:
    def __init__(self, name, ck):
        import torch

        from aaf.eval.p3_2_eval import load_model
        self.torch = torch
        self.dev = torch.device("cuda")
        self.model, self.renderer, *_ = load_model(Path(ck), self.dev)
        self.model.eval()
        self.renderer.eval()
        self.name = name

    def render(self, verts, L, W, alphas, src, rx, chunk=None):
        import os

        from aaf.models.conditioning_2d import build_cond_vector_2d
        chunk = int(os.environ.get("P410_CHUNK", "32")) if chunk is None else chunk
        t = self.torch
        z = build_cond_vector_2d("geom_token", L, W, list(alphas), verts=[tuple(map(float, v)) for v in verts],
                                 device=self.dev)
        lo, hi = t.zeros(2, device=self.dev), t.tensor([L, W], device=self.dev, dtype=t.float32)
        R = t.tensor(np.asarray(rx), dtype=t.float32, device=self.dev)
        S = t.tensor([list(map(float, src))], device=self.dev, dtype=t.float32)
        out = []
        with t.no_grad():
            for s0 in range(0, len(R), chunk):
                r = R[s0:s0 + chunk]
                out.append(self.renderer(self.model, r, S.expand(len(r), -1), lo, hi, z_s=z.unsqueeze(0).expand(len(r), -1),
                                         n_out=601).cpu().numpy())
        return np.concatenate(out).astype(np.complex64)


def save_atomic(path, **kw):
    """several array tasks may render the same room's original field: write to a private name, then rename"""
    import os
    tmp = path.parent / "{}.{}.{}.tmp.npz".format(path.stem, os.getpid(), os.uname().nodename.split(".")[0])
    np.savez_compressed(tmp, **kw)
    os.replace(tmp, path)


def render(name, ck, families, lite, idx, chunk):
    M = cells(families, lite)
    M = M[idx * chunk:(idx + 1) * chunk]
    if not M:
        return
    rd = Renderer(name, ck)
    (PRED / name / "orig").mkdir(parents=True, exist_ok=True)
    for c in M:
        dst = PRED / name / (cid(c) + ".npz")
        tp = TRUTH / (cid(c) + ".npz")
        if dst.exists():
            continue
        if not tp.exists():
            print("no truth yet:", c["id"], flush=True)
            continue
        base = stored(c)
        op = PRED / name / "orig" / (c["room"] + ".npz")
        if not op.exists():                                    # the model's original render at every original receiver
            P0 = rd.render(c["orig_verts"], c["orig_L"], c["orig_W"], [0.15] * len(c["orig_verts"]), (0.5, 1.6), base["rx"])
            save_atomic(op, P=P0, rx=base["rx"])
        z = np.load(tp)
        Pd = rd.render(c["verts"], c["L"], c["W"], c["edge_alphas"], c["src"], z["rx"])
        Pc = Pd[z["common_idx"]] if c["family"] in ("Source", "Material") else rd.render(
            c["verts"], c["L"], c["W"], c["edge_alphas"], c["src"], z["common"])
        save_atomic(dst, Pd=Pd, Pc=Pc, id=c["id"])
        print("rendered", c["id"], len(Pd), len(Pc), flush=True)


# ------------------------------------------------------------------------------------------- scoring
def m1_field(c, rx, src, orig=False):
    from aaf.eval.modal_prior import pulse
    from scripts.p4_9_fast_physics import field, solve
    V = [tuple(map(float, v)) for v in (c["orig_verts"] if orig else c["verts"])]
    L, W = (c["orig_L"], c["orig_W"]) if orig else (c["L"], c["W"])
    op, k2, psi, eta = solve(V, L, W, 0.10, 350.0)
    sig, S = pulse()
    return field(op, k2, psi, eta, np.asarray(rx), src, 350.0, sig, S)[0]


def warp_field(c, T0, rx0, rx_new):
    """the original truth mapped through the stretch at the same frequency: p(x') = p0(x'/sx, y'/sy) (linear interpolation
    of the complex field; points outside the original hull take the nearest receiver)."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    q = np.asarray(rx_new) / [c["sx"], c["sy"]]
    lin, near = LinearNDInterpolator(rx0, T0), NearestNDInterpolator(rx0, T0)
    out = lin(q)
    bad = np.isnan(out.real).any(1)
    out[bad] = near(q[bad])
    return out


def score_cell(name, c, m1=True):
    z = np.load(TRUTH / (cid(c) + ".npz"))
    p = np.load(PRED / name / (cid(c) + ".npz"))
    base = stored(c)
    o = np.load(PRED / name / "orig" / (c["room"] + ".npz"))
    rx = z["rx"]
    off = ~on_lattice(rx)
    Td, Pd = z["Hd"][off], p["Pd"][off]
    row = {"id": c["id"], "group": c["group"], "room": c["room"], "family": c["family"], "mag": c["mag"],
           "n_off": int(off.sum()), "edited": {"r": band_r(Pd, Td), "lsd": band_lsd(Pd, Td)}}
    ofs = ~on_lattice(base["rx"])
    row["orig_r"] = band_r(o["P"][ofs], base["H"][ofs])                     # the model's static r on the ORIGINAL room
    ci = z["common_idx"]
    T0, P0 = base["H"][ci], o["P"][ci]                         # the original room's truth and the model's original render
    Te, Pe = z["Hc"], p["Pc"]
    clip = lambda x: np.clip(x, -20.0, 20.0)  # noqa: E731
    dT, dP = clip(db(Te) - db(T0)), clip(db(Pe) - db(P0))
    r = colr(dP, dT)
    row["r_delta"] = {b: float(np.nanmean(r[lo:hi])) for b, (lo, hi) in BAND.items()}
    l0, l1 = band_lsd(P0, Te), band_lsd(Pe, Te)
    row["g_delta"] = {b: l0[b] / max(l1[b], 1e-9) for b in BAND}
    row["edited_lsd_common"] = l1
    row["ignore"] = {"r": band_r(T0, Te), "lsd": band_lsd(T0, Te)}
    row["model_orig_vs_edited_truth"] = {"r": band_r(P0, Te), "lsd": l0}
    if c["family"] in ("StretchU", "StretchA"):
        Wp = warp_field(c, base["H"], base["rx"], rx)[off]
        row["warp"] = {"r": band_r(Wp, Td), "lsd": band_lsd(Wp, Td)}
    if c["family"] == "Material":
        row["decay"] = decay_cell(c, base, o, z, p)
    if m1 and c["family"] != "Material":
        Pm = m1_field(c, rx, tuple(c["src"]))[off]
        row["m1"] = {"r": band_r(Pm, Td), "lsd": band_lsd(Pm, Td)}
        dM = clip(db(m1_field(c, z["common"], tuple(c["src"]))) - db(m1_field(c, z["common"], (0.5, 1.6), orig=True)))
        rm = colr(dM, dT)
        row["m1"]["r_delta"] = {b: float(np.nanmean(rm[lo:hi])) for b, (lo, hi) in BAND.items()}
    return row


def decay_cell(c, base, o, z, p):
    """Octave-band decay (A7) on the Material cell's receivers: T0 / Te (truth before / after the edit), P0 / Pe (model).
    Receivers: off-lattice, >= 1 m from the source, valid in BOTH truths. Per band: mean decay rate 1/T30 (1/s) of each field,
    median |T30 error| / truth, median |EDT error| / truth, median |C50 error| dB (all vs the edited truth)."""
    from aaf.eval.octave_decay import BANDS, decay_metrics, valid_mask
    rx = base["rx"]
    src = np.array([0.5, 1.6])
    sel = ~on_lattice(rx) & (np.hypot(rx[:, 0] - src[0], rx[:, 1] - src[1]) >= 1.0)
    F = {"T0": base["H"][sel], "Te": z["Hd"][sel], "P0": o["P"][sel], "Pe": p["Pd"][sel]}
    out = {"n_rx": int(sel.sum())}
    for b in BANDS:
        m = {k: decay_metrics(v, b) for k, v in F.items()}
        ok = valid_mask(m["T0"]["t35"]) & valid_mask(m["Te"]["t35"])
        e = {"n_valid": int(ok.sum())}
        for k in F:
            T = m[k]["T30"]
            e["rate_" + k] = float(np.nanmean(1.0 / T[ok])) if ok.any() else np.nan
            e["T30_" + k] = float(np.nanmedian(T[ok])) if ok.any() else np.nan
        if ok.any():
            t30 = np.abs(m["Pe"]["T30"] - m["Te"]["T30"]) / np.maximum(m["Te"]["T30"], 1e-6)
            e["t30_err"] = float(np.nanmedian(t30[ok]))
            e["edt_err"] = float(np.nanmedian((np.abs(m["Pe"]["EDT"] - m["Te"]["EDT"]) / np.maximum(m["Te"]["EDT"], 1e-6))[ok]))
            e["c50_err_db"] = float(np.nanmedian(np.abs(m["Pe"]["C50"] - m["Te"]["C50"])[ok]))
            e["t30_err_orig_model"] = float(np.nanmedian(np.abs(m["P0"]["T30"] - m["Te"]["T30"])[ok] / np.maximum(m["Te"]["T30"], 1e-6)[ok]))
        out[b] = e
    return out


def score(name, families, lite, idx=None, chunk=None, m1=True):
    M = cells(families, lite)
    if idx is not None:
        M = M[idx * chunk:(idx + 1) * chunk]
    (SC / name).mkdir(parents=True, exist_ok=True)
    for c in M:
        dst = SC / name / (cid(c) + ".json")
        if dst.exists() or not (PRED / name / (cid(c) + ".npz")).exists():
            continue
        dst.write_text(json.dumps(score_cell(name, c, m1=m1)))
        print("scored", c["id"], flush=True)


# ------------------------------------------------------------------------------------------- aggregation
def source_gap(name, lite=False, groups=("test", "unseen")):
    """moved-source gap per magnitude: {group: {mag: {room: {band: orig r - moved r}}}} (R1's primary metric, RS1)."""
    out = {}
    for c in cells(["Source"], lite):
        f = SC / name / (cid(c) + ".json")
        if f.exists():
            r = json.loads(f.read_text())
            out.setdefault(c["group"], {}).setdefault(c["mag"], {})[c["room"]] = {
                b: r["orig_r"][b] - r["edited"]["r"][b] for b in BAND}
    return out


def material_slope(name, lite=False, group="test"):
    """decay-rate edit slope (MM1): over every Material cell of the group, the regression of the predicted change in mean
    1/T30 (Pe - P0) on the true change (Te - T0), per octave band and pooled, with Pearson; NaN cells (truth too long for
    the gate) are dropped and counted."""
    pts = {b: [] for b in ("63", "125", "250")}
    n_drop = 0
    for c in cells(["Material"], lite):
        if c["group"] != group:
            continue
        f = SC / name / (cid(c) + ".json")
        if not f.exists():
            continue
        d = json.loads(f.read_text()).get("decay", {})
        for b in pts:
            e = d.get(b, {})
            t, p = e.get("rate_Te", np.nan) - e.get("rate_T0", np.nan), e.get("rate_Pe", np.nan) - e.get("rate_P0", np.nan)
            if np.isfinite(t) and np.isfinite(p):
                pts[b].append((t, p))
            else:
                n_drop += 1
    out = {"n_dropped_cells_bands": n_drop, "bands": {}}
    allt, allp = [], []
    for b, v in pts.items():
        if len(v) >= 4:
            x, y = np.array(v).T
            allt += list(x)
            allp += list(y)
            sl, ic = np.polyfit(x, y, 1)
            out["bands"][b] = {"slope": float(sl), "pearson": float(np.corrcoef(x, y)[0, 1]), "n": len(v)}
    if len(allt) >= 4:
        x, y = np.array(allt), np.array(allp)
        out["pooled"] = {"slope": float(np.polyfit(x, y, 1)[0]), "pearson": float(np.corrcoef(x, y)[0, 1]), "n": len(allt)}
    return out


def boot(v):
    v = np.asarray(v, float)
    idx = np.random.default_rng(SEED).integers(0, len(v), (NB, len(v)))
    m = v[idx].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def room_table(name, families, lite):
    """{family: {group: {room: {metric: {band: value}}}}} -- each room's average over its magnitudes"""
    acc = {}
    for c in cells(families, lite):
        f = SC / name / (cid(c) + ".json")
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        acc.setdefault((c["family"], c["group"], c["room"]), []).append(r)
    out = {}
    for (fam, g, room), rows in acc.items():
        e = {"n_cells": len(rows)}
        for nm, pick in (("edited_r", lambda r, b: r["edited"]["r"][b]), ("edited_lsd", lambda r, b: r["edited"]["lsd"][b]),
                         ("r_delta", lambda r, b: r["r_delta"][b]), ("g_delta", lambda r, b: r["g_delta"][b]),
                         ("ignore_r", lambda r, b: r["ignore"]["r"][b]), ("ignore_lsd", lambda r, b: r["ignore"]["lsd"][b]),
                         ("m1_r", lambda r, b: r.get("m1", {}).get("r", {}).get(b, np.nan)),
                         ("m1_r_delta", lambda r, b: r.get("m1", {}).get("r_delta", {}).get(b, np.nan)),
                         ("warp_r", lambda r, b: r.get("warp", {}).get("r", {}).get(b, np.nan)),
                         ("orig_r_vs_edit", lambda r, b: r["model_orig_vs_edited_truth"]["r"][b]),
                         ("orig_r", lambda r, b: r["orig_r"][b])):
            e[nm] = {b: float(np.nanmean([pick(r, b) for r in rows])) for b in BAND}
        out.setdefault(fam, {}).setdefault(g, {})[room] = e
    return out


def aggregate(models, families, lite, tag):
    tabs = {m: room_table(m, families, lite) for m in models}
    rep = {"definition": __doc__.split("\n\n")[0], "models": list(models), "lite": lite, "families": {}}
    L = ["# P4-10 editability scorecard ({}): {}".format("lite" if lite else "full", tag), "",
         "Off-lattice dense receivers (edited-field r / LSD) or the common receivers (r_delta, g_delta). Mean over rooms "
         "of each room's average over magnitudes; CIs = paired bootstrap over rooms (10k, seed {}). Bands 25-100 / "
         "100-200 / 200-300 Hz. In-family 15 is primary, unseen beside. 'ignore the edit' = the original room's truth "
         "scored against the edited truth. m1 = 10 cm / 350 Hz physics (reference, not a baseline to beat).".format(SEED), ""]
    for fam in (families or FAMS):
        for g in ("test", "unseen"):
            rooms = sorted(set.intersection(*[set(tabs[m].get(fam, {}).get(g, {})) for m in models])) if models else []
            if not rooms:
                continue
            key = "{}|{}".format(fam, g)
            rep["families"][key] = {"n_rooms": len(rooms)}
            L += ["## {} -- {} ({} rooms)".format(fam, "in-family" if g == "test" else "unseen", len(rooms)), "",
                  "| field | edited r | r_delta | g_delta |", "|---|---|---|---|"]
            for m in models:
                e = rep["families"][key].setdefault(m, {})
                for met in ("edited_r", "edited_lsd", "r_delta", "g_delta", "ignore_r", "m1_r", "m1_r_delta", "warp_r", "orig_r_vs_edit", "orig_r"):
                    e[met] = {b: boot([tabs[m][fam][g][r][met][b] for r in rooms]) for b in BAND}
                f = lambda met: " / ".join("{:.3f}".format(e[met][b][0]) for b in BAND)  # noqa: E731
                L.append("| {} | {} | {} | {} |".format(m, f("edited_r"), f("r_delta"), f("g_delta")))
            e0 = rep["families"][key][models[0]]
            g3 = lambda met: " / ".join("{:.3f}".format(e0[met][b][0]) for b in BAND)  # noqa: E731
            L.append("| ignore the edit (original truth) | {} | -- | 1 (by definition) |".format(g3("ignore_r")))
            if np.isfinite(e0["m1_r"]["25-100"][0]):
                L.append("| m1 physics (reference) | {} | {} | -- |".format(g3("m1_r"), g3("m1_r_delta")))
            if fam == "Material":
                for m in models:
                    ms = material_slope(m, lite, g if g == "test" else "unseen")
                    if "pooled" in ms:
                        L.append("decay-rate edit slope (true -> predicted change in 1/T30) -- {}: {} (pooled over cells and octaves; Pearson {:.2f}, n {})".format(
                            m, "{:.2f}".format(ms["pooled"]["slope"]), ms["pooled"]["pearson"], ms["pooled"]["n"]))
                        rep["families"][key].setdefault(m, {})["material_slope"] = ms
            if fam in ("StretchU", "StretchA"):
                L.append("| naive warp of the original truth | {} | -- | -- |".format(g3("warp_r")))
            L.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scorecard_{}.json".format(tag)).write_text(json.dumps(rep, indent=1))
    (OUT / "scorecard_{}.md".format(tag)).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    for k in ("--render", "--score", "--aggregate", "--plan"):
        ap.add_argument(k, action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--families", default="")
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--no-m1", action="store_true")
    ap.add_argument("--tag", default="R0")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=20)
    a = ap.parse_args()
    fam = a.families.split("+") if a.families else None
    if a.plan:
        n = len(cells(fam, a.lite))
        print(n, "cells ->", (n - 1) // a.chunk + 1, "tasks")
    elif a.render:
        name, ck = a.model.split("=", 1)
        render(name, ck, fam, a.lite, a.idx, a.chunk)
    elif a.score:
        score(a.model, fam, a.lite, a.idx, a.chunk, not a.no_m1)
    elif a.aggregate:
        aggregate(a.models.split("+"), fam, a.lite, a.tag)


if __name__ == "__main__":
    main()
