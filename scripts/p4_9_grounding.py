"""P4-9 Step 1 (D180; D181 A1, A5, A6, A12, E1, E7, E8/S-2, c3, c4): sparse-measurement grounding, network weights frozen.

Rooms: the 15 in-family test shapes + the 16 frozen unseen rooms (reported); the 2 T + 2 DN family dense rooms
(`data/track_p4_5_famdense`, in neither reported set) for CALIBRATION only (E7).
Per room (c3): the "< 0.3 m" stratum = every off-lattice dense receiver within 0.3 m of the source; the ">= 0.3 m"
stratum = 500 off-lattice receivers >= 0.3 m from the source reserved at random (seed 20261013); the DRAW POOL = the
remaining off-lattice receivers >= 0.3 m. K in {8, 16, 32, 64} measurements per draw, seeds 0 / 1 / 2. Every method,
K and seed scores on the same strata (pooled = both); the headline K = 32 clean fits are also scored on ALL non-drawn
off-lattice receivers (D180's definition) -- the set P1 / P2 are read on. Metric: the frozen band r / LSD
(`scripts/p4_7_m1_modal.scores`).
Methods (every fit minimises the TRAINING loss -- the 4-term spectral loss, bins 0-600 -- on its K measurements):
  M0     zero-shot R0 / S2.
  M1     the per-token embeddings (12 x 64, `segment_encoder` output, A5) fitted by Adam from their zero-shot value with a
         relative L2 pull lam ||E - E0||^2 / ||E0||^2; 8 measurements per step; 601-bin path (c4); eval-mode renderer.
  M2     M1 + the FiLM layers of the three signal heads (`film_signal`, `_signal_films.0/.1`) and the DC head MLP (per-
         room copies, restored after each fit), same relative pull, lr / 10.
  M1-iii / M2-iii  as M1 / M2 with tokens from the BOUNDING RECTANGLE (A1: the network given the same wrong geometry as
         M4-iii). R0 only.
  M3     m1's exact field (exact geometry) with ONE beta scale fitted (A2: one material) by a 1-D search on the loss.
  M4-ii / M4-iii   as M3 on the 10 cm-snapped polygon / the bounding rectangle.
  M5     p(x) = a(f) G0(x, x0; f) + sum_i alpha_i J0(k |x - x_i|) per bin (E1), G0 = -(j/4) H0^(2)(k r) (outgoing under
         e^{-jkd}), a unpenalised, alpha ridge-penalised (lam alpha^H K alpha); no geometry.
Conditions: clean; real2 = 20 dB SNR (complex Gaussian, per bin, relative to the measured receivers' mean power) + 2 cm
position error (N(0, 2 cm) per axis on the positions GIVEN to the method; data from the true positions); real5 = the
same with 5 cm. Realism at K = 32 (S-2): M1 / M2 (R0) real2; M3 / M4 / M5 real2 and real5.
Calibration (E7), once, on the 4 T / DN rooms at K = 32, seed 0, R0: M1 / M2 lr in {1e-3, 3e-3} x lam in {0, 1e-2,
1e-1}, scores logged at 50 / 100 / 200 / 400 steps -> the (lr, lam, steps) maximising the mean pooled r over the three
bands (ties -> fewer steps); M5 lam in {1e-4, 1e-3, 1e-2, 1e-1, 1}. Frozen, then applied to every reported fit.
  --calibrate                 GPU (+ CPU for M5): outputs/p4_9/step1/calibration.json
  --net MODEL --room I        GPU: every network fit of one reported room (requeue-safe, one JSON per fit)
  --phys --room I             CPU: every physics / KRR fit of one reported room
  --aggregate                 outputs/p4_9/step1/grounding.{json,md}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

OUT = Path("outputs/p4_9/step1")
FITS = OUT / "fits"
MODELS = {"R0_120K": "outputs/p4_8/arms/R0/ckpt_iter0120000.pt", "S2_120K": "outputs/p4_8/arms/S2/ckpt_iter0120000.pt"}
KS = (8, 16, 32, 64)
SEEDS = (0, 1, 2)
SRC = np.array([0.5, 1.6])
NEAR = 0.3
N_FAR = 500
SEED = 20261013
C0, DF = 343.0, 0.5
BANDS = ("25-100", "100-200", "200-300")
CAL_GRID = [(lr, lam) for lr in (1e-3, 3e-3) for lam in (0.0, 1e-2, 1e-1)]
CAL_STEPS = (50, 100, 200, 400)
M5_LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FAMDENSE = Path("data/track_p4_5_famdense")


# ------------------------------------------------------------------------------------------- rooms / splits / draws
def eval_rooms():
    from scripts.p4_9_dir_zeroshot import dense_rooms
    return dense_rooms()


def calib_rooms():
    out = []
    for p in sorted(FAMDENSE.glob("*.h5")):
        if p.name[:1] == "T" or p.name[:2] == "DN":
            out.append({"group": "calib", "family": "T" if p.name[0] == "T" else "DN", "path": p})
    assert len(out) == 4, [r["path"].name for r in out]
    return out


def on_lattice(rx, tol=1e-6):
    """`scripts/p4_5_gate2_dense.on_lattice`, inlined (that module imports torch and the evaluator stack, which a
    CPU node does not need): a receiver on the training lattice 0.12 + 0.06 k in both coordinates."""
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < tol, axis=1)


def load(path):
    with h5py.File(path) as f:
        r = {"T": np.asarray(f["ism/H_complex"])[:, :601], "rx": np.asarray(json.loads(f.attrs["receiver_pos"]), float),
             "verts": [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])],
             "L": float(f.attrs["L"]), "W": float(f.attrs["W"]), "name": Path(path).name, "stem": Path(path).stem}
    r["off"] = ~on_lattice(r["rx"])
    r["dist"] = np.hypot(*(r["rx"] - SRC).T)
    return r


def _rhash(name):
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def split(r):
    near = np.nonzero(r["off"] & (r["dist"] < NEAR))[0]
    farpool = np.nonzero(r["off"] & (r["dist"] >= NEAR))[0]
    rng = np.random.default_rng([SEED, _rhash(r["name"])])
    far = np.sort(rng.choice(farpool, size=min(N_FAR, len(farpool) // 2), replace=False))
    pool = np.setdiff1d(farpool, far)
    return {"near": near, "far": far, "pool": pool}


def draw(r, sp, K, seed):
    rng = np.random.default_rng([20261122, K, seed, _rhash(r["name"])])
    return np.sort(rng.choice(sp["pool"], size=K, replace=False))


def perturb(r, idx, cond, K, seed):
    """(positions GIVEN to the method, measured spectra) under a condition."""
    from aaf.sim.polygon_geom import polygon_contains
    P = r["rx"][idx].copy()
    Y = r["T"][idx].copy()
    if cond == "clean":
        return P, Y
    sig = {"real2": 0.02, "real5": 0.05}[cond]
    rng = np.random.default_rng([31337, K, seed, _rhash(r["name"]), int(sig * 100)])
    for i in range(len(P)):
        while True:
            q = r["rx"][idx[i]] + rng.normal(0.0, sig, 2)
            if polygon_contains(r["verts"], q[None])[0]:
                P[i] = q
                break
    pw = (np.abs(Y) ** 2).mean(axis=0)                                   # per bin, over the measured receivers
    sd = np.sqrt(pw * 10 ** (-20 / 10) / 2.0)
    Y = Y + sd * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    return P, Y.astype(np.complex64)


def score_sets(r, sp, idx, headline):
    sets = {"near": sp["near"], "far": sp["far"], "pooled": np.concatenate([sp["near"], sp["far"]])}
    if headline:
        sets["all_nondrawn"] = np.setdiff1d(np.nonzero(r["off"])[0], idx)
    return sets


def scores_of(F, r, sets):
    from scripts.p4_7_m1_modal import scores
    return {k: scores(F[v], r["T"][v]) for k, v in sets.items()}


def fit_path(model, method, room, K, seed, cond):
    return FITS / "{}__{}__{}__K{}__s{}__{}.json".format(model, method, room, K, seed, cond)


# ------------------------------------------------------------------------------------------- network
class Net:
    def __init__(self, name):
        import torch

        from aaf.eval.p3_2_eval import load_model
        self.torch = torch
        self.dev = torch.device("cuda")
        self.name = name
        self.model, self.renderer, *_ = load_model(Path(MODELS[name]), self.dev)
        self.model.eval()
        self.renderer.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.m2_mods = [self.model.film_signal, *self.model._signal_films, self.model.dc_head]

    def cond(self, verts, L, W):
        from aaf.models.conditioning_2d import build_cond_vector_2d
        return build_cond_vector_2d("geom_token", L, W, [0.15] * len(verts), verts=verts, device=self.dev)

    def render(self, z, L, W, pts, E=None, chunk=16):
        torch = self.torch
        R = torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=self.dev)
        S = torch.tensor([[0.5, 1.6]], device=self.dev)
        lo, hi = torch.zeros(2, device=self.dev), torch.tensor([L, W], device=self.dev, dtype=torch.float32)
        out = []
        for s0 in range(0, len(R), chunk):
            rr = R[s0:s0 + chunk]
            kw = {"tok_override": E.expand(len(rr), -1, -1)} if E is not None else {}
            out.append(self.renderer(self.model, rr, S.expand(len(rr), -1), lo, hi,
                                     z_s=z.unsqueeze(0).expand(len(rr), -1), n_out=601, **kw))
        return torch.cat(out)

    def fit(self, z, L, W, P, Y, method, lr, lam, steps, log_steps=(), log_fn=None, seed=0):
        """M1 / M2 fit; returns the fitted embeddings (and leaves M2's modules adapted -- the caller restores)."""
        torch = self.torch
        from aaf.train.multi_room_2d_mat import _losses
        with torch.no_grad():
            E0 = self.model.token_embeddings(z.unsqueeze(0)).detach()
        E = E0.clone().requires_grad_(True)
        groups = [{"params": [E], "lr": lr}]
        extra, base = [], []
        if method == "M2":
            for m in self.m2_mods:
                for p in m.parameters():
                    p.requires_grad_(True)
                    extra.append(p)
                    base.append(p.detach().clone())
            groups.append({"params": extra, "lr": lr / 10.0})
        opt = torch.optim.Adam(groups)
        Yt = torch.as_tensor(Y, dtype=torch.complex64, device=self.dev)
        Pt = np.asarray(P)
        g = np.random.default_rng([seed, len(P)])
        nE0 = float((E0 ** 2).sum())
        for step in range(1, steps + 1):
            sel = g.choice(len(Pt), size=min(8, len(Pt)), replace=False)
            H = self.render(z, L, W, Pt[sel], E=E)
            t = _losses(H, Yt[sel])
            loss = t["L_spec_real"] + t["L_spec_imag"] + t["L_amp"] + 0.1 * t["L_phase"]
            if lam > 0:
                loss = loss + lam * ((E - E0) ** 2).sum() / nE0
                for p, b in zip(extra, base):
                    loss = loss + lam * ((p - b) ** 2).sum() / max(float((b ** 2).sum()), 1e-12)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if log_fn is not None and step in log_steps:
                log_fn(step, E.detach())
        for p in extra:
            p.requires_grad_(False)
        return E.detach()

    def snapshot(self):
        return [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in self.m2_mods]

    def restore(self, snap):
        for m, sd in zip(self.m2_mods, snap):
            m.load_state_dict(sd)


def net_fit_and_score(net, r, sp, idx, method, cond, K, seed, geom, cfg_fit, headline):
    torch = net.torch
    verts = r["verts"] if geom == "exact" else [(0.0, 0.0), (r["L"], 0.0), (r["L"], r["W"]), (0.0, r["W"])]
    z = net.cond(verts, r["L"], r["W"])
    sets = score_sets(r, sp, idx, headline)
    allpts = np.unique(np.concatenate(list(sets.values())))
    if method == "M0":
        with torch.no_grad():
            F = np.zeros((len(r["rx"]), 601), np.complex64)
            F[allpts] = net.render(z, r["L"], r["W"], r["rx"][allpts]).cpu().numpy()
        return {"scores": scores_of(F, r, sets), "fit_s": 0.0}
    P, Y = perturb(r, idx, cond, K, seed)
    snap = net.snapshot()
    torch.cuda.synchronize()
    t0 = time.time()
    E = net.fit(z, r["L"], r["W"], P, Y, method, cfg_fit["lr"], cfg_fit["lam"], cfg_fit["steps"], seed=seed)
    torch.cuda.synchronize()
    dt = time.time() - t0
    with torch.no_grad():
        F = np.zeros((len(r["rx"]), 601), np.complex64)
        F[allpts] = net.render(z, r["L"], r["W"], r["rx"][allpts], E=E).cpu().numpy()
    net.restore(snap)
    return {"scores": scores_of(F, r, sets), "fit_s": dt, "steps": cfg_fit["steps"], "lr": cfg_fit["lr"],
            "lam": cfg_fit["lam"]}


def calibrate():
    net = Net("R0_120K")
    res = {"grid": [], "rooms": [c["path"].name for c in calib_rooms()]}
    for (lr, lam) in CAL_GRID:
        for method in ("M1", "M2"):
            curve = {s: [] for s in CAL_STEPS}
            for c in calib_rooms():
                r = load(c["path"])
                sp = split(r)
                idx = draw(r, sp, 32, 0)
                z = net.cond(r["verts"], r["L"], r["W"])
                sets = score_sets(r, sp, idx, False)
                pts = np.concatenate([sets["near"], sets["far"]])
                snap = net.snapshot()

                def log_fn(step, E, r=r, z=z, pts=pts, sets=sets):
                    with net.torch.no_grad():
                        F = np.zeros((len(r["rx"]), 601), np.complex64)
                        F[pts] = net.render(z, r["L"], r["W"], r["rx"][pts], E=E).cpu().numpy()
                    s = scores_of(F, r, {"pooled": sets["pooled"]})["pooled"]
                    curve[step].append(float(np.mean([s[b] for b in BANDS])))
                net.fit(z, r["L"], r["W"], r["rx"][idx], r["T"][idx], method, lr, lam, max(CAL_STEPS),
                        log_steps=CAL_STEPS, log_fn=log_fn)
                net.restore(snap)
            for s in CAL_STEPS:
                res["grid"].append({"method": method, "lr": lr, "lam": lam, "steps": s, "mean_r": float(np.mean(curve[s]))})
            print(method, lr, lam, {s: round(float(np.mean(v)), 4) for s, v in curve.items()}, flush=True)
    best = {}
    for method in ("M1", "M2"):
        g = [x for x in res["grid"] if x["method"] == method]
        top = max(x["mean_r"] for x in g)
        best[method] = min([x for x in g if x["mean_r"] >= top - 1e-9], key=lambda x: x["steps"])
    res["chosen"] = best
    # M5's ridge on the same rooms / draws
    m5 = {}
    for lam in M5_LAMS:
        v = []
        for c in calib_rooms():
            r = load(c["path"])
            sp = split(r)
            idx = draw(r, sp, 32, 0)
            sets = score_sets(r, sp, idx, False)
            F = krr(r["rx"][idx], r["T"][idx], r["rx"], lam)
            s = scores_of(F, r, {"pooled": sets["pooled"]})["pooled"]
            v.append(float(np.mean([s[b] for b in BANDS])))
        m5[str(lam)] = float(np.mean(v))
    res["m5_lam"] = m5
    res["chosen"]["M5"] = {"lam": float(max(M5_LAMS, key=lambda lam: m5[str(lam)]))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(res, indent=1))
    print("chosen", res["chosen"], flush=True)


def run_net(model, ri):
    cal = json.loads((OUT / "calibration.json").read_text())["chosen"]
    rr = eval_rooms()[ri]
    r = load(rr["path"])
    sp = split(r)
    net = Net(model)
    FITS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for K in KS:
        for seed in SEEDS:
            jobs.append(("M1", "exact", K, seed, "clean"))
            if K == 32 or seed == 0:
                jobs.append(("M2", "exact", K, seed, "clean"))
    if model == "R0_120K":
        for seed in SEEDS:
            jobs += [("M1", "bbox", 32, seed, "clean"), ("M2", "bbox", 32, seed, "clean"),
                     ("M1", "exact", 32, seed, "real2"), ("M2", "exact", 32, seed, "real2")]
    jobs = [("M0", "exact", 32, 0, "clean"), ("M0", "bbox", 32, 0, "clean")] + jobs
    for method, geom, K, seed, cond in jobs:
        mname = method + ("-iii" if geom == "bbox" else "")
        dst = fit_path(model, mname, r["stem"], K, seed, cond)
        if dst.exists():
            continue
        idx = draw(r, sp, K, seed)
        cfg_fit = {"lr": cal[method]["lr"], "lam": cal[method]["lam"], "steps": cal[method]["steps"]} \
            if method != "M0" else {}
        out = net_fit_and_score(net, r, sp, idx, method, cond, K, seed, geom, cfg_fit,
                                headline=(K == 32 and cond == "clean"))
        out.update({"model": model, "method": mname, "room": r["name"], "group": rr["group"], "family": rr["family"],
                    "K": K, "seed": seed, "cond": cond, "measured": idx.tolist()})
        dst.write_text(json.dumps(out))
        print(dst.name, round(out["scores"]["pooled"]["25-100"], 3), round(out.get("fit_s", 0), 1), flush=True)


# ------------------------------------------------------------------------------------------- physics / KRR
def krr(P, Y, Q, lam, src=SRC):
    """E1: per bin, p(x) = a G0(x, x0) + sum_i alpha_i J0(k |x - x_i|); a unpenalised, alpha ridge-penalised."""
    from scipy.special import hankel2, j0
    out = np.zeros((len(Q), Y.shape[1]), np.complex128)
    dPP = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
    dQP = np.hypot(*(Q[:, None] - P[None]).transpose(2, 0, 1))
    rP = np.maximum(np.hypot(*(P - src).T), 1e-3)
    rQ = np.maximum(np.hypot(*(Q - src).T), 1e-3)
    out[:, 0] = Y[:, 0].mean()
    for b in range(1, Y.shape[1]):
        k = 2 * np.pi * b * DF / C0
        A = j0(k * dPP) + lam * np.eye(len(P))
        g = -0.25j * hankel2(0, k * rP)
        y = Y[:, b]
        Ai_y = np.linalg.solve(A, y)
        Ai_g = np.linalg.solve(A, g)
        a = (np.conj(g) @ Ai_y) / (np.conj(g) @ Ai_g)
        alpha = Ai_y - a * Ai_g
        out[:, b] = a * (-0.25j * hankel2(0, k * rQ)) + j0(k * dQP) @ alpha
    return out


class Modal:
    """m1's exact field on one geometry variant, with psi at the dense receivers and (for perturbed positions) by
    bilinear interpolation on the full air grid."""

    def __init__(self, stem, variant):
        self.z = dict(np.load(Path("outputs/p4_9/modal") / "{}__{}.npz".format(stem, variant)))
        self.stem, self.variant = stem, variant
        self.grid = None

    def psi_at(self, pts):
        if self.grid is None:
            gd = Path("data/track_p4_9_modal_grid")
            g = np.load(gd / "{}__{}__grid.npz".format(self.stem, self.variant))
            self.grid = {"psi": np.load(gd / "{}__{}__psi.npy".format(self.stem, self.variant)), "idx": g["idx"],
                         "dxx": float(g["dxx"]), "dxy": float(g["dxy"])}
        G = self.grid
        u, v = pts[:, 0] / G["dxx"], pts[:, 1] / G["dxy"]
        i0, j0_ = np.floor(u).astype(int), np.floor(v).astype(int)
        out = np.zeros((len(pts), G["psi"].shape[1]))
        wsum = np.zeros(len(pts))
        for di in (0, 1):
            for dj in (0, 1):
                ii, jj = np.clip(i0 + di, 0, G["idx"].shape[0] - 1), np.clip(j0_ + dj, 0, G["idx"].shape[1] - 1)
                q = G["idx"][ii, jj]
                w = (1 - np.abs(u - (i0 + di))) * (1 - np.abs(v - (j0_ + dj)))
                w = np.where(q >= 0, w, 0.0)
                out += w[:, None] * G["psi"][np.maximum(q, 0)]
                wsum += w
        return out / np.maximum(wsum, 1e-12)[:, None]

    def field(self, psi_rows, s):
        from aaf.eval.modal_prior import C_FIT, DT, pulse, zero_mode_record
        z = self.z
        meta = json.loads(str(z["meta"]))
        f = np.arange(601) * DF
        kk = 2 * np.pi * f / C0
        eta = z["eta"] * s
        den = z["k2"][1:, None] - kk[None] ** 2 + 1j * kk[None] * eta[1:, None]
        den[:, 0] = 1.0
        G = psi_rows[:, 1:] @ (z["psi_s"][1:, None] / den)
        G = G * (z["S"] * np.exp(1j * 2 * np.pi * f * DT))[None]
        G[:, 0] = 0.0
        if s == 1.0:
            td0 = z["td0"]
        else:
            sig, _ = pulse()
            td0 = zero_mode_record(float(eta[0]), float(z["psi_s"][0] * meta["dA"]), sig)
        return C_FIT * G + psi_rows[:, :1] * td0[None] * (C_FIT / (meta["dA"] / (C0 * DT) ** 2))


def train_loss_np(P, T, eps=1e-6):
    """The trainer's 4-term loss (`_losses` default path, weights 1 / 1 / 1 / 0.1) in numpy -- the trainer module imports
    tinycudann, which cannot load on a CPU node."""
    return float(np.mean(np.abs(P.real - T.real)) + np.mean(np.abs(P.imag - T.imag))
                 + np.mean(np.abs(np.log10(np.abs(P) + eps) - np.log10(np.abs(T) + eps)))
                 + 0.1 * np.mean(1.0 - np.cos(np.angle(P) - np.angle(T))))


def beta_fit(mod, psi_meas, Y):
    """one beta scale by a 1-D search on the training loss at the measured receivers (log grid, then golden)."""
    Yc = np.asarray(Y, np.complex64)

    def loss(s):
        return train_loss_np(mod.field(psi_meas, s).astype(np.complex64), Yc)
    grid = np.geomspace(1 / 8, 8, 49)
    vals = [loss(s) for s in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(25):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = loss(d)
    return float((a + b) / 2)


def run_phys(ri):
    cal = json.loads((OUT / "calibration.json").read_text())["chosen"]
    rr = eval_rooms()[ri]
    r = load(rr["path"])
    sp = split(r)
    FITS.mkdir(parents=True, exist_ok=True)
    mods = {"M3": Modal(r["stem"], "exact"), "M4-ii": Modal(r["stem"], "snap10"), "M4-iii": Modal(r["stem"], "bbox")}
    jobs = [(m, K, seed, "clean") for m in ("M3", "M4-ii", "M4-iii", "M5") for K in KS for seed in SEEDS]
    jobs += [(m, 32, seed, c) for m in ("M3", "M4-ii", "M4-iii", "M5") for seed in SEEDS for c in ("real2", "real5")]
    for method, K, seed, cond in jobs:
        dst = fit_path("physics", method, r["stem"], K, seed, cond)
        if dst.exists():
            continue
        idx = draw(r, sp, K, seed)
        P, Y = perturb(r, idx, cond, K, seed)
        sets = score_sets(r, sp, idx, headline=(K == 32 and cond == "clean"))
        allpts = np.unique(np.concatenate(list(sets.values())))
        t0 = time.time()
        if method == "M5":
            F = np.zeros((len(r["rx"]), 601), np.complex128)
            F[allpts] = krr(P, Y, r["rx"][allpts], cal["M5"]["lam"])
            extra = {"lam": cal["M5"]["lam"]}
        else:
            mod = mods[method]
            psi_meas = mod.z["psi_r"][idx] if cond == "clean" else mod.psi_at(P)
            s = beta_fit(mod, psi_meas, Y)
            F = np.zeros((len(r["rx"]), 601), np.complex128)
            F[allpts] = mod.field(mod.z["psi_r"][allpts], s)
            extra = {"beta_scale": s}
        dt = time.time() - t0
        out = {"scores": scores_of(F, r, sets), "fit_s": dt, "model": "physics", "method": method, "room": r["name"],
               "group": rr["group"], "family": rr["family"], "K": K, "seed": seed, "cond": cond, **extra}
        dst.write_text(json.dumps(out))
        print(dst.name, round(out["scores"]["pooled"]["25-100"], 3), round(dt, 2), flush=True)


# ------------------------------------------------------------------------------------------- aggregation (D183(e))
def _boot(v, nb=10000):
    v = np.asarray(v, float)
    idx = np.random.default_rng(SEED).integers(0, len(v), (nb, len(v)))
    m = v[idx].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def aggregate():
    fits = [json.loads(p.read_text()) for p in sorted(FITS.glob("*.json"))]
    # per (model, method, K, cond, stratum, band, room): mean over seeds
    acc = {}
    for f in fits:
        for st, sc in f["scores"].items():
            for b in BANDS:
                key = (f["model"], f["method"], f["K"], f["cond"], st, b)
                acc.setdefault(key, {}).setdefault(f["room"], []).append(sc[b])
    per_room = {k: {r: float(np.mean(v)) for r, v in d.items()} for k, d in acc.items()}
    groups = {}
    for f in fits:
        groups[f["room"]] = (f["group"], f["family"])
    sets = {"unseen16": lambda r: groups[r][0] == "unseen", "in_family15": lambda r: groups[r][0] == "test",
            "U8": lambda r: groups[r][1] == "U", "C3_8": lambda r: groups[r][1] == "C3"}
    table = {}
    for k, d in per_room.items():
        for sname, sel in sets.items():
            v = [x for r, x in d.items() if sel(r)]
            if v:
                table.setdefault(sname, {})["|".join(map(str, k))] = {"mean_ci": _boot(v), "n_rooms": len(v)}
    times = {}
    for f in fits:
        times.setdefault((f["model"], f["method"], f["K"]), []).append(f.get("fit_s", 0.0))
    fit_time = {"|".join(map(str, k)): float(np.median(v)) for k, v in times.items()}

    def rooms_of(key, sel):
        d = per_room.get(key, {})
        return {r: x for r, x in d.items() if sel(r)}

    def paired(ka, kb, sel):
        a, b = rooms_of(ka, sel), rooms_of(kb, sel)
        common = sorted(set(a) & set(b))
        return _boot([a[r] - b[r] for r in common]), len(common)

    un = sets["unseen16"]
    readings = {}
    for m in MODELS:
        d = rooms_of((m, "M1", 32, "clean", "all_nondrawn", "25-100"), un)
        v = float(np.mean(list(d.values()))) if d else float("nan")
        readings["P1_" + m] = {"test": "M1, K = 32, all non-drawn off-lattice, unseen 25-100 Hz >= 0.65", "value": v,
                               "n_rooms": len(d), "holds": bool(v >= 0.65)}
        p3 = {}
        for K in (8, 16, 32):
            ci, n = paired((m, "M1", K, "clean", "pooled", "200-300"), ("physics", "M5", K, "clean", "pooled", "200-300"), un)
            p3[str(K)] = {"M1_minus_M5": ci, "n": n}
        readings["P3_" + m] = {"test": "M1 - M5 at 200-300 Hz, pooled strata, CI above 0 at K = 8, 16 and 32", "by_K": p3,
                               "holds": bool(all(p3[k]["M1_minus_M5"][1] > 0 for k in p3))}
    p2 = {}
    for meth in ("M1-iii", "M2-iii", "M1"):
        bands = {}
        for b in BANDS:
            ci, n = paired(("R0_120K", meth, 32, "clean", "all_nondrawn", b), ("physics", "M4-iii", 32, "clean",
                                                                                  "all_nondrawn", b), un)
            bands[b] = {"minus_M4iii": ci, "n": n}
        p2[meth] = {"bands": bands, "wins": int(sum(bands[b]["minus_M4iii"][1] > 0 for b in BANDS))}
    readings["P2"] = {"test": "R0's M1-iii or M2-iii beats M4-iii (same geometry, A1) in >= 2 of 3 bands, paired CI > 0 "
                              "(K = 32, all non-drawn off-lattice, unseen 16); exact-geometry M1 shown beside",
                      "by_method": p2, "holds": bool(p2["M1-iii"]["wins"] >= 2 or p2["M2-iii"]["wins"] >= 2)}
    rep = {"definition": __doc__.split("\n\n")[0].strip(), "n_fits": len(fits), "readings": readings,
           "table": table, "fit_time_median_s": fit_time}
    (OUT / "grounding.json").write_text(json.dumps(rep, indent=1))

    def cell(sname, model, meth, K, cond, st, b):
        e = table.get(sname, {}).get("|".join(map(str, (model, meth, K, cond, st, b))))
        return "--" if e is None else "{:.3f}".format(e["mean_ci"][0])
    rows = [("R0_120K", "M0"), ("S2_120K", "M0"), ("R0_120K", "M1"), ("S2_120K", "M1"), ("R0_120K", "M2"),
            ("S2_120K", "M2"), ("R0_120K", "M1-iii"), ("R0_120K", "M2-iii"), ("physics", "M3"), ("physics", "M4-ii"),
            ("physics", "M4-iii"), ("physics", "M5")]
    L = ["# P4-9 Step 1: sparse-measurement grounding (D180; D183)", "", rep["definition"], "",
         "## Pre-registered readings", ""]
    for k, v in readings.items():
        L.append("* **{}** -- {}: **{}**".format(k, v["test"], "HOLDS" if v["holds"] else "does not hold")
                 + (" (value {:.3f}, {} rooms)".format(v["value"], v["n_rooms"]) if "value" in v else ""))
    L += ["", "## Unseen 16, K = 32, clean: r per band on all non-drawn off-lattice receivers (headline)", "",
          "| model | method | 25-100 | 100-200 | 200-300 | median fit time (s) |", "|---|---|---|---|---|---|"]
    for model, meth in rows:
        K0 = 32
        L.append("| {} | {} | {} | {} | {} | {} |".format(model, meth, *[cell("unseen16", model, meth, K0, "clean",
                                                                              "all_nondrawn", b) for b in BANDS],
                                                         "{:.1f}".format(fit_time.get("|".join(map(str, (model, meth, K0))), float("nan")))))
    for sname, title in (("unseen16", "Unseen 16"), ("in_family15", "In-family 15")):
        L += ["", "## {}: r vs K, pooled strata (both strata; 25-100 / 100-200 / 200-300)".format(title), "",
              "| model | method | " + " | ".join("K={}".format(K) for K in KS) + " |", "|---|---|" + "---|" * len(KS)]
        for model, meth in rows:
            cells = []
            for K in KS:
                c = [cell(sname, model, meth, K, "clean", "pooled", b) for b in BANDS]
                cells.append(" / ".join(c))
            L.append("| {} | {} | {} |".format(model, meth, " | ".join(cells)))
    L += ["", "## Strata (unseen 16, K = 32, clean, 25-100 Hz): < 0.3 m / >= 0.3 m / pooled", "",
          "| model | method | near | far | pooled |", "|---|---|---|---|---|"]
    for model, meth in rows:
        L.append("| {} | {} | {} | {} | {} |".format(model, meth, *[cell("unseen16", model, meth, 32, "clean", st,
                                                                          "25-100") for st in ("near", "far", "pooled")]))
    L += ["", "## Realism at K = 32 (unseen 16, pooled, 25-100 / 100-200 / 200-300)", "",
          "| model | method | clean | real2 (20 dB + 2 cm) | real5 (20 dB + 5 cm) |", "|---|---|---|---|---|"]
    for model, meth in rows:
        L.append("| {} | {} | {} |".format(model, meth, " | ".join(
            " / ".join(cell("unseen16", model, meth, 32, cond, "pooled", b) for b in BANDS)
            for cond in ("clean", "real2", "real5"))))
    (OUT / "grounding.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:20]))
    figure_f1(table)


def figure_f1(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = [("R0_120K", "M0", "M0 zero-shot (R0)"), ("R0_120K", "M1", "M1 z-fit (R0)"), ("S2_120K", "M1", "M1 z-fit (S2)"),
              ("R0_120K", "M2", "M2 z+FiLM (R0)"), ("physics", "M3", "M3 physics, exact geometry"),
              ("physics", "M4-iii", "M4-iii physics, bounding box"), ("physics", "M5", "M5 kernel ridge (no geometry)")]
    col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=160, facecolor="white", sharey=True)
    for ax, b in zip(axes, BANDS):
        ax.set_facecolor("white")
        ax.grid(True, color="#e6e6e6")
        ax.set_axisbelow(True)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        for (model, meth, lab), c in zip(series, col):
            xs, ys, lo, hi = [], [], [], []
            for K in KS:
                Kk = 32 if meth == "M0" else K
                e = table.get("unseen16", {}).get("|".join(map(str, (model, meth, Kk, "clean", "pooled", b))))
                if e:
                    xs.append(K)
                    ys.append(e["mean_ci"][0])
                    lo.append(e["mean_ci"][1])
                    hi.append(e["mean_ci"][2])
            if xs:
                ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=8, markeredgecolor="white",
                        markeredgewidth=2, label=lab, linestyle=(0, (5, 3)) if meth == "M0" else "-")
                ax.fill_between(xs, lo, hi, color=c, alpha=0.10, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS)
        ax.set_xticklabels([str(K) for K in KS])
        ax.set_xlabel("measurements K")
        ax.set_title("{} Hz".format(b), loc="left", fontsize=15)
    axes[0].set_ylabel("r, unseen 16 rooms (off-lattice, both strata)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=11)
    fig.suptitle("F1 -- sparse-measurement grounding on unseen rooms: r vs K (95% CI bands over rooms)", x=0.01,
                 ha="left", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path("outputs/p4_9/meeting").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/p4_9/meeting/F1_grounding.png", facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--net", default="")
    ap.add_argument("--phys", action="store_true")
    ap.add_argument("--room", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    if a.calibrate:
        calibrate()
    if a.net:
        run_net(a.net, a.room)
    if a.phys:
        run_phys(a.room)


if __name__ == "__main__":
    main()
