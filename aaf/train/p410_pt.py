"""P4-10 P-1 (D217 / D218): R1-P(t), a 10K-step fine-tune of R1 @150K with the single-radius mean-value (Helmholtz) penalty, and its
matched control R1-ctl (the same class with the penalty off). Nothing in aaf/train/multi_room_2d_mat.py is changed: this is a subclass.

Config: a P32TrainCfg YAML (R1's, with the warm-start keys) plus one extra mapping `pt:` read by this module's `main` only:
  pt: {frac: 0.20, k: 2, bins: 8, force_w: null, seed_off: 50021}
  frac 0 (or force_w 0) = the control: the data step is the parent's, unchanged.
Penalty (D204's single-radius stencil): at centre x the mean over 8 points on a circle of radius r equals J0(k r) p(x) for any Helmholtz
field with the disk inside the room. r = 0.35 m for bins up to 187.5 Hz (where 1.2 / k >= 0.35 m); above, one f0 per step is drawn uniform in
[187.5, 300] Hz, r = 1.2 / k(f0), and the bins are drawn within +/- 5% of f0. Per bin: RMS over centres of (circle mean - J0(k r) p) divided by
the RMS over centres of |p| -- NOT detached (A13). Per step: 8 bins from 25-100 Hz, 8 from 100-187.5 Hz (the two share the r = 0.35 m renders) and
8 around f0. ONE class per step (2 centres, 18 renders: all that fits under 60% step-time overhead on a 24 GB A5000): odd steps the r = 0.35 m class (strata 25-100 and 100-187.5 Hz),
even steps the r(f0) class; each contributes (2/3) x the sum of its stratum means, so the expected per-step penalty is the mean of the three strata. Two centres pool the
per-bin RMS: with one, the normaliser is a single |p| and near a node the ratio has infinite variance. Bins < 41 (the DC head) are never used.
Draws come from a DEDICATED CPU generator (saved in every checkpoint): the batch / slot / jitter / swap streams are R1's own, continued.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.special import j0

from aaf.sim.polygon_geom import polygon_contains
from aaf.train.multi_room_2d_mat import P32TrainCfg, P32Trainer, _losses

DF, C0 = 0.5, 343.0
R_MAX, KR_MAX = 0.35, 1.2
F_SPLIT = 375                          # bin: 187.5 Hz; below it 1.2 / k >= 0.35 m, so r = 0.35 m
NPT = 8
BINS_LO, BINS_MID, BINS_HI = (50, 200), (200, F_SPLIT), (F_SPLIT, 601)
MARGIN_WALL, MARGIN_SRC = 0.05, 0.30
LATTICE_TOL = 0.01                     # centres stay >= 1 cm from every node of the training receiver lattice 0.12 + 0.06 k


def radius(f_hz: float) -> float:
    k = 2 * math.pi * f_hz / C0
    return min(KR_MAX / k, R_MAX)


def meanvalue_ratio(Hc: torch.Tensor, Hm: torch.Tensor, kr: torch.Tensor) -> torch.Tensor:
    """Hc [K, B] centre values, Hm [K, 8, B] circle values, kr [B] -> [B] RMS_K(|mean_8 Hm - J0(kr) Hc|) / RMS_K(|Hc|), neither detached"""
    J = torch.as_tensor(j0(kr.cpu().numpy()), dtype=Hc.real.dtype, device=Hc.device)
    R = Hm.mean(1) - J * Hc
    ab = lambda z: z.real ** 2 + z.imag ** 2  # noqa: E731
    return torch.sqrt(ab(R).mean(0) + 1e-24) / torch.sqrt(ab(Hc).mean(0) + 1e-24)


def poly_dist(p: np.ndarray, verts) -> np.ndarray:
    v = np.asarray(verts, float)
    a, b = v, np.roll(v, -1, 0)
    ab = b - a
    t = np.clip(((p[:, None, :] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(p[:, None, :] - (a + t[..., None] * ab), axis=-1).min(1)


class PTTrainer(P32Trainer):
    def __init__(self, cfg: P32TrainCfg, output_dir, pt: dict = None):
        self.pt = {"frac": 0.20, "k": 2, "bins": 8, "force_w": None, "seed_off": 50021}
        self.pt.update(pt or {})
        self._pt_ckpt = None
        super().__init__(cfg, output_dir)
        self._pt_gen = torch.Generator(device="cpu")
        self._pt_gen.manual_seed(int(cfg.seed) + int(self.pt["seed_off"]))
        self._pt_w = None
        if self._pt_ckpt is not None:                                          # a resume: the checkpoint's generator and weight win
            self._pt_gen.set_state(self._pt_ckpt["gen"])
            self._pt_w = self._pt_ckpt["w"]
            if self._pt_ckpt["pt"] != {k: v for k, v in self.pt.items() if k != "force_w"}:
                raise RuntimeError("the checkpoint was written with other penalty settings: {} vs {}".format(self._pt_ckpt["pt"], self.pt))
        if self._pt_w is None:
            self._pt_w = float(self.pt["force_w"]) if self.pt["force_w"] is not None else (0.0 if self.pt["frac"] <= 0 else self._calibrate())
        print("[pt] weight {:.5g} settings {}".format(self._pt_w, self.pt), flush=True)

    # ------------------------------------------------------------------------------------ warm start / resume
    def _init_from(self, path):
        """the parent's warm start (weights + Adam moments + explicit LR) and, additionally, R1's own generator states: the 10K steps continue R1's
        batch / swap / slot / jitter streams instead of replaying its first ones"""
        it0 = super()._init_from(path)
        st = torch.load(Path(path).resolve(), map_location="cpu")
        self._p49_gen.set_state(st["p49_gen_state"])
        self._p49_swap_gen.set_state(st["p49_swap_gen_state"])
        for k, g in self._p410_gens.items():
            g.set_state(st["p410_gen_states"][k])
        print("[pt] continued the generator states of {}".format(Path(path).name), flush=True)
        return it0

    def _maybe_resume(self):
        it = super()._maybe_resume()
        c = self.out / "ckpt_iter{:07d}.pt".format(it)
        if c.exists():                                                         # a resumed run of THIS arm
            st = torch.load(c, map_location="cpu")
            if "pt_state" not in st:
                raise RuntimeError("refusing to resume {}: no penalty state".format(c.name))
            self._pt_ckpt = st["pt_state"]
            del st
        return it

    def save_ckpt(self, it: int):
        super().save_ckpt(it)
        path = self.out / "ckpt_iter{:07d}.pt".format(it)
        st = torch.load(path, map_location="cpu")
        st["pt_state"] = {"gen": self._pt_gen.get_state(), "w": float(self._pt_w), "pt": {k: v for k, v in self.pt.items() if k != "force_w"}}
        tmp = path.with_suffix(".pt.pttmp")
        torch.save(st, tmp)
        tmp.replace(path)

    # ------------------------------------------------------------------------------------ penalty
    def _pt_centres(self, c: int, src, r: float, n: int, gen) -> np.ndarray:
        """[n, 2] points: in the polygon, >= r + 0.05 m from every wall, >= r + 0.3 m from the source, not within 1 cm of a lattice node"""
        room = self.configs[c]
        L, W = float(room.L), float(room.W)
        out = []
        for _ in range(200):
            p = torch.rand(256, 2, generator=gen, dtype=torch.float64).numpy() * np.array([L, W])
            kx = (p - 0.12) / 0.06
            near = np.hypot(*((kx - np.round(kx)) * 0.06).T)
            ok = np.asarray(polygon_contains(room.verts, p), bool) & (poly_dist(p, room.verts) >= r + MARGIN_WALL) \
                & (np.hypot(p[:, 0] - src[0], p[:, 1] - src[1]) >= r + MARGIN_SRC) & (near >= LATTICE_TOL)
            out += list(p[ok])
            if len(out) >= n:
                return np.asarray(out[:n])
        return None                                                            # no admissible point (narrow room / source in the middle): skip

    def _pt_draw(self, gen, kind: str):
        """one penalty evaluation: a room, one of its sources, `k` centres of one class (kind 'lo': r = 0.35 m; 'hi': r = 1.2 / k(f0), f0 uniform in [187.5, 300] Hz), rotations, jitters, bins"""
        K, nb = int(self.pt["k"]), int(self.pt["bins"])
        for _ in range(20):
            c = int(torch.randint(0, len(self.configs), (1,), generator=gen))
            s = int(torch.randint(0, self._src_ms.shape[1], (1,), generator=gen)) if self._src_ms is not None else 0
            src = (self._src_ms[c, s] if self._src_ms is not None else self.src_per[c]).cpu().numpy().astype(float)
            fi = int(torch.randint(F_SPLIT, 601, (1,), generator=gen))
            r = R_MAX if kind == "lo" else radius(fi * DF)
            ctr = self._pt_centres(c, src, r, K, gen)
            if ctr is not None:
                break
        else:
            return None
        ang = torch.rand(K, generator=gen, dtype=torch.float64).numpy() * 2 * math.pi
        jit = torch.rand(K, generator=gen) * 2 * math.pi / self.renderer.n_azi
        pick = lambda lo, hi: np.sort(lo + torch.randperm(hi - lo, generator=gen)[:nb].numpy())  # noqa: E731
        if kind == "lo":
            bins = (pick(*BINS_LO), pick(*BINS_MID))
        else:
            bins = (pick(max(F_SPLIT, int(round(fi * 0.95))), min(601, int(round(fi * 1.05)) + 1)),)
        return {"kind": kind, "c": c, "src": src, "ctr": ctr, "r": r, "ang": ang, "jit": jit, "bins": bins, "f0_bin": fi}

    def _pt_penalty(self, d: dict):
        """the step's penalty (a scalar that backpropagates) and its stratum values"""
        dev = self.device
        K = len(d["ctr"])
        th = np.arange(NPT) * 2 * np.pi / NPT
        pts, jr = [], []
        for i, x in enumerate(d["ctr"]):
            circ = x + d["r"] * np.stack([np.cos(th + d["ang"][i]), np.sin(th + d["ang"][i])], 1)
            pts.append(np.concatenate([x[None], circ]))                        # [9, 2]
            jr += [float(d["jit"][i])] * 9                                      # ONE jitter per centre: a shared ray fan, or the stencil mixes fans
        P = torch.tensor(np.concatenate(pts), dtype=torch.float32, device=dev)
        S = torch.tensor(d["src"], dtype=torch.float32, device=dev).expand(len(P), 2)
        z = self.cond[d["c"]].unsqueeze(0).expand(len(P), -1)
        room = self.configs[d["c"]]
        hi = torch.tensor([float(room.L), float(room.W)], device=dev, dtype=torch.float32)
        jt = torch.tensor(jr, dtype=torch.float32, device=dev)
        H = self.renderer(self.model, P, S, torch.zeros(2, device=dev), hi, z_s=z, jitter=jt, n_out=601).view(K, 9, -1)
        vals = []
        for b in d["bins"]:
            bt = torch.as_tensor(np.asarray(b), device=dev)
            kr = torch.as_tensor(np.asarray(b) * DF * 2 * math.pi / C0 * d["r"], dtype=torch.float64)
            vals.append(meanvalue_ratio(H[:, 0][:, bt], H[:, 1:][:, :, bt], kr).mean())
        return sum(vals) * (2.0 / 3.0), {("lo" if d["kind"] == "lo" and i == 0 else "mid" if d["kind"] == "lo" else "hi"): float(v) for i, v in enumerate(vals)}

    def _calibrate(self) -> float:
        """w = f / (1 - f) * L_data / penalty at step 0, over 8 micro-batches and 16 draws, from separate generators; every stream state is restored"""
        f = float(self.pt["frac"])
        st = [self._p49_gen.get_state(), self._p49_swap_gen.get_state(), {k: g.get_state() for k, g in self._p410_gens.items()}]
        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.cfg.seed) + 777)
        w_r, w_i, w_a, w_p = self.cfg.weights
        data, pen = [], []
        self.model.train()
        with torch.no_grad():
            for _ in range(4):
                for ci, ri in self._p49_batch():
                    rx, tx, z, dims, T, jitter = self._p410_rows(ci, ri, {})
                    t = _losses(self._render_rows(rx, tx, z, dims, jitter=jitter), T, self.loss_lo_bin, w_spec=self.spec_w, amp_eps=self.amp_eps)
                    data.append(float(w_r * t["L_spec_real"] + w_i * t["L_spec_imag"] + w_a * t["L_amp"] + w_p * t["L_phase"]))
            i = 0
            while len(pen) < 24 and i < 200:
                i += 1
                d = self._pt_draw(g, "lo" if len(pen) % 2 == 0 else "hi")            # the two classes alternate: the mean of pen is the expected step penalty
                if d is not None:
                    pen.append(float(self._pt_penalty(d)[0]))
        self._p49_gen.set_state(st[0])
        self._p49_swap_gen.set_state(st[1])
        for k, gg in self._p410_gens.items():
            gg.set_state(st[2][k])
        w = f / (1 - f) * float(np.mean(data)) / max(float(np.mean(pen)), 1e-12)
        self._pt_calib = {"L_data": float(np.mean(data)), "penalty": float(np.mean(pen)), "n_data": len(data), "n_pen": len(pen)}
        print("[pt] calibration: L_data {:.4g}, penalty {:.4g} -> weight {:.5g} ({:.0%} of the total at step 0)".format(
            self._pt_calib["L_data"], self._pt_calib["penalty"], w, f), flush=True)
        return w

    def _step_p410(self, it: int) -> dict:
        """R1's data step; with a weight > 0 the penalty of `pt['rooms']` draws is added in its own backward, before the clipping and the update"""
        if self._pt_w == 0.0 and self.pt.get("force_w") is None:
            return super()._step_p410(it)
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        agg, cfg = {}, self.cfg
        w_r, w_i, w_a, w_p = cfg.weights
        for ci, ri in self._p49_batch():
            rx, tx, z, dims, T, jitter = self._p410_rows(ci, ri, agg)
            H = self._render_rows(rx, tx, z, dims, jitter=jitter)
            t = _losses(H, T, self.loss_lo_bin, w_spec=self.spec_w, amp_eps=self.amp_eps)
            loss = w_r * t["L_spec_real"] + w_i * t["L_spec_imag"] + w_a * t["L_amp"] + w_p * t["L_phase"]
            (loss / cfg.grad_accum_steps).backward()
            for kk, v in t.items():
                agg[kk] = agg.get(kk, 0.0) + float(v) / cfg.grad_accum_steps
            agg["loss"] = agg.get("loss", 0.0) + float(loss) / cfg.grad_accum_steps
        d = self._pt_draw(self._pt_gen, "lo" if it % 2 == 1 else "hi")
        if d is None:
            agg["pt_skipped"] = 1.0
        else:
            pen, strata = self._pt_penalty(d)
            (self._pt_w * pen).backward()
            agg["L_pt"] = float(pen)
            for nm, v in strata.items():
                agg["pt_" + nm] = v
        agg["pt_w"] = float(self._pt_w)
        self._apply_grads()
        return agg


def load_cfg(path):
    d = yaml.safe_load(open(path))
    pt = d.pop("pt", {}) or {}
    known = {f for f in P32TrainCfg.__dataclass_fields__}
    bad = [k for k in d if k not in known]
    if bad:
        raise SystemExit("unknown config keys {}".format(bad))
    return P32TrainCfg(**d), pt


def bench(cfg, pt, out, n=25):
    """step time of the control and of penalty settings on this GPU; no checkpoint is written"""
    t0 = PTTrainer(cfg, out, dict(pt, frac=0.0))
    t0._step_fn = t0._step_p410
    def tm(tr, k=n, warm=5):  # noqa: E306
        for _ in range(warm):
            tr._step_p410(1)
        torch.cuda.synchronize()
        s = time.time()
        for _ in range(k):
            tr._step_p410(1)
        torch.cuda.synchronize()
        return (time.time() - s) / k
    base = tm(t0)
    print("[bench] control {:.4f} s/step (peak {:.1f} GB)".format(base, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    res = {"control": base}
    torch.cuda.reset_peak_memory_stats()
    t0._pt_w, t0.pt = 1.0, dict(t0.pt, force_w=None, frac=0.2)
    s = tm(t0, 25, 5)
    res["k2"] = s
    print("[bench] k = 2 centres of one class per step (18 penalty rows): {:.4f} s/step = +{:.0%} over control (peak {:.1f} GB)".format(s, s / base - 1, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    cfg, pt = load_cfg(a.config)
    if a.bench:
        bench(cfg, pt, a.output_dir)
        return
    PTTrainer(cfg, a.output_dir, pt).train()


if __name__ == "__main__":
    main()
