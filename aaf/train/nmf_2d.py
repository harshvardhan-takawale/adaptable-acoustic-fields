"""P4-9 2c trainer (D180 NMF-LB; D181 A10, E4): the neural modal field on the GX corpus, 25-100 Hz.

Losses (per room, mean over the step's rooms):
  field     the 4-term spectral loss (`multi_room_2d_mat._losses`) on bins 50-199 at rx_per_room sampled receivers
  subspace  ||P_hat - P||^2 / ||P||^2 with soft weights (cosine roll-off between the TRUE k_20 and k_24, applied to each
            set's own wavenumbers -- continuous when modes cross, A10), quadrature inner product
  k         mean |k_hat_n - k_n| / k_n over the 24 sorted modes
  ortho     ||Phi^T W Phi - I||^2 / 24 (soft orthonormality, A10)
  NMF-P     (phys_weight > 0) the per-mode relative residuals of lap phi + k^2 phi = 0 (5-point stencil) at interior
            collocation points and of d phi / dn = 0 (3-point one-sided) at wall nodes > 0.15 m from vertices, both by
            finite differences at h = 12.4 cm = the grid's finest cell (E4)
Kill rule (D180): at 10K, if the in-family (15 test shapes) off-lattice r at 25-100 Hz < 0.70, the run stops and says so.
  python -m aaf.train.nmf_2d --config configs/sweep_2d_mat/P4_9_NMF.yaml --output_dir outputs/p4_9/arms/NMF
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from aaf.eval.modal_prior import C_FIT, DT
from aaf.models.nmf_2d import NMF2D, orthonormality_loss, soft_weights, subspace_loss

C0 = 343.0
LO, HI = 50, 200                               # 25-100 Hz (bins of 0.5 Hz)
NMF_KEYS = ("labels_dir", "manifest", "data_dir", "init_encoder", "n_iters", "rooms_per_step", "rx_per_room", "lr",
            "lr_encoder", "w_field", "w_sub", "w_k", "w_orth", "phys_weight", "h_fd", "seed", "hidden")


@dataclass
class NMFCfg:
    run_id: str = "p4_9_NMF"
    labels_dir: str = "data/track_p4_9_eigen/gx"
    manifest: str = "configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"
    data_dir: str = "data/track_p4_7_mix"
    init_encoder: str = "outputs/p4_8/arms/R0/ckpt_iter0120000.pt"
    dense_labels_dir: str = "data/track_p4_9_eigen/dense"
    n_iters: int = 30000
    rooms_per_step: int = 8
    rx_per_room: int = 32
    lr: float = 1e-3
    lr_encoder: float = 1e-4
    eta_min: float = 1e-5
    w_field: float = 1.0
    w_sub: float = 1.0
    w_k: float = 1.0
    w_orth: float = 0.1
    phys_weight: float = 0.0
    h_fd: float = 0.124
    n_colloc: int = 256
    hidden: int = 128
    seed: int = 0
    ckpt_every: int = 2000
    val_every: int = 2000
    log_every: int = 200
    kill_iter: int = 10000
    kill_r: float = 0.70


def _tokens(verts, L, W, dev):
    from aaf.models.conditioning_2d import build_cond_vector_2d
    return build_cond_vector_2d("geom_token", L, W, [0.15] * len(verts), verts=verts, device=dev)


def _wall_normals(pos, verts):
    """inward unit normal of the nearest (axis-aligned) polygon edge, and the distance to the nearest vertex."""
    V = np.asarray(verts, float)
    nrm = np.zeros_like(pos)
    best = np.full(len(pos), np.inf)
    for (x0, y0), (x1, y1) in zip(V, np.roll(V, -1, axis=0)):
        a, b = np.array([x0, y0]), np.array([x1, y1])
        d = b - a
        t = np.clip(((pos - a) @ d) / max(d @ d, 1e-12), 0, 1)
        dist = np.linalg.norm(pos - (a + t[:, None] * d), axis=1)
        n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-12)    # CCW ring: the interior is on the left
        upd = dist < best
        best[upd] = dist[upd]
        nrm[upd] = n
    vdist = np.min(np.linalg.norm(pos[:, None] - V[None], axis=2), axis=1)
    return nrm, vdist


class Room:
    """One room's labels and (training rooms) FDTD targets, on the device."""

    def __init__(self, lab, verts, L, W, dev, H=None, h_fd=0.124):
        from aaf.sim.polygon_geom import polygon_contains
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)  # noqa
        self.z = _tokens(verts, L, W, dev)
        self.k_true = t(np.sqrt(lab["k2"]))                         # [29] incl. the zero mode
        self.q_pos, self.w_q, self.psi_q = t(lab["q_pos"]), t(lab["w_q"]), t(lab["psi_q"])
        self.w_pos, self.w_w = t(lab["w_pos"]), t(lab["w_w"])
        self.rx = t(lab["rx"])
        self.src = t(lab["src"]).view(1, 2)
        self.psi0 = float(lab["psi_s"][0])                          # the zero mode is constant: 1 / sqrt(area)
        self.td0 = torch.as_tensor(lab["td0"], dtype=torch.complex128, device=dev)
        self.c_pred = float(lab["dA"]) / (C0 * DT) ** 2
        self.S = torch.as_tensor(lab["S"], dtype=torch.complex128, device=dev)
        self.H = None if H is None else torch.as_tensor(H, dtype=torch.complex64, device=dev)
        # NMF-P geometry: interior collocation (all four +-h neighbours inside) and wall stencils (> 0.15 m from vertices)
        q = np.asarray(lab["q_pos"], float)
        ok = np.ones(len(q), bool)
        for d in ((h_fd, 0), (-h_fd, 0), (0, h_fd), (0, -h_fd)):
            ok &= polygon_contains(verts, q + np.asarray(d))
        self.colloc = t(q[ok])
        wp = np.asarray(lab["w_pos"], float)
        nrm, vd = _wall_normals(wp, verts)
        keep = vd > 0.15
        inside = polygon_contains(verts, wp + 2 * h_fd * nrm)
        keep &= inside
        self.wall_fd = t(wp[keep])
        self.wall_n = t(nrm[keep])


def predict(model, room, x, k_override=None):
    """normalised mode shapes at x [N, 2], plus k [24] and eta [24], for one room."""
    c = model.room_code(room.z.view(1, -1))[0]
    k = model.k_modes(c.view(1, -1))[0] if k_override is None else k_override
    nq = model.norms(room.q_pos, room.w_q, c)
    phi_x = model.phi_raw(x, c) / nq
    phi_w = model.phi_raw(room.w_pos, c) / nq
    eta = (room.w_w[:, None] * phi_w ** 2).sum(0)
    return c, k, nq, phi_x, eta


def room_field(model, room, x):
    """H at x [N, 2], bins 50-199 -> [N, 150] complex64 (+ the pieces for the losses)."""
    c, k, nq, phi_x, eta = predict(model, room, x)
    phi_s = model.phi_raw(room.src, c)[0] / nq
    bins = torch.arange(LO, HI, device=x.device)
    # td0 already carries psi_0(x0) (its forcing is mu psi_0(x0) dA), so the receiver side adds psi_0(x) only (m1)
    zero = (room.psi0 * room.td0[LO:HI] * (C_FIT / room.c_pred))[None].expand(len(x), -1)
    H = NMF2D.field(k, eta, phi_x, phi_s, bins, room.S[LO:HI], C_FIT, zero)
    return H, c, k, nq


def phys_residuals(model, room, c, k, nq, h, n_colloc, gen):
    """E4: relative residuals at h = 12.4 cm. interior: ||lap phi + k^2 phi|| / ||k^2 phi||; walls: ||d_n phi|| / ||k phi||."""
    i = torch.randint(0, len(room.colloc), (min(n_colloc, len(room.colloc)),), device=room.colloc.device, generator=gen)
    x = room.colloc[i]
    ex, ey = torch.tensor([h, 0.0], device=x.device), torch.tensor([0.0, h], device=x.device)
    f = lambda p: model.phi_raw(p, c) / nq  # noqa
    p0 = f(x)
    lap = (f(x + ex) + f(x - ex) + f(x + ey) + f(x - ey) - 4 * p0) / h ** 2
    k2 = (k ** 2)[None]
    r_int = (torch.linalg.norm(lap + k2 * p0, dim=0) / torch.linalg.norm(k2 * p0, dim=0).clamp(min=1e-9)).mean()
    w, n = room.wall_fd, room.wall_n
    dn = (-3 * f(w) + 4 * f(w + h * n) - f(w + 2 * h * n)) / (2 * h)
    r_wall = (torch.linalg.norm(dn, dim=0) / torch.linalg.norm(k[None] * f(w), dim=0).clamp(min=1e-9)).mean()
    return r_int, r_wall


def load_rooms(cfg, dev):
    rows = json.load(open(cfg.manifest))["configs"]
    rooms = []
    for k, r in enumerate(rows):
        lab = dict(np.load(Path(cfg.labels_dir) / "{:04d}.npz".format(k)))
        assert str(lab["filename"]) == r["filename"]
        with h5py.File(Path(cfg.data_dir) / r["filename"]) as f:
            H = np.asarray(f["ism/H_complex"])[:, LO:HI]
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
        assert np.allclose(rx, lab["rx"])
        verts = [tuple(v) for v in r["verts"]]
        rooms.append(Room(lab, verts, max(v[0] for v in verts), max(v[1] for v in verts), dev, H=H, h_fd=cfg.h_fd))
    return rooms


def dense_eval(model, cfg, dev, groups=("test",)):
    """off-lattice r at 25-100 Hz on the dense rooms of the given groups (the frozen metric)."""
    from scripts.p4_7_m1_modal import scores
    from scripts.p4_9_fdtd_stencils import dense_rooms
    out = []
    model.eval()
    with torch.no_grad():
        for g, fam, p in dense_rooms():
            if g not in groups:
                continue
            lab = dict(np.load(Path(cfg.dense_labels_dir) / (p.stem + ".npz")))
            with h5py.File(p) as f:
                T = np.asarray(f["ism/H_complex"])[:, :601]
                verts = [tuple(map(float, v)) for v in json.loads(f.attrs["verts"])]
            room = Room(lab, verts, max(v[0] for v in verts), max(v[1] for v in verts), dev, h_fd=cfg.h_fd)
            P = np.zeros((len(lab["rx"]), 601), np.complex64)
            for s0 in range(0, len(lab["rx"]), 4096):
                P[s0:s0 + 4096, LO:HI] = room_field(model, room, room.rx[s0:s0 + 4096])[0].cpu().numpy()
            k = (np.asarray(lab["rx"]) - 0.12) / 0.06
            off = ~np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)
            out.append({"room": p.name, "group": g, "family": fam, "r_25_100": scores(P[off], T[off])["25-100"],
                        "P": P})
    model.train()
    return out


def train(cfg: NMFCfg, out_dir: str):
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = NMF2D(hidden=cfg.hidden).to(dev)
    st0 = torch.load(cfg.init_encoder, map_location="cpu")["model"]
    model.segment_encoder.load_state_dict({k.split("segment_encoder.")[1]: v for k, v in st0.items()
                                           if k.startswith("segment_encoder.")})
    enc = list(model.segment_encoder.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("segment_encoder.")]
    opt = torch.optim.Adam([{"params": enc, "lr": cfg.lr_encoder}, {"params": rest, "lr": cfg.lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_iters, eta_min=cfg.eta_min)
    from aaf.train.multi_room_2d_mat import _losses
    rooms = load_rooms(cfg, dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed)
    start, scal = 0, []
    ck = sorted(out.glob("ckpt_iter*.pt"))
    if ck:
        st = torch.load(ck[-1], map_location="cpu")
        for k in NMF_KEYS:
            if st["cfg"].get(k) != getattr(cfg, k):
                raise RuntimeError("refusing to resume {}: {} differs".format(ck[-1].name, k))
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        gen.set_state(st["gen"])
        start, scal = int(st["iter"]), st.get("scalars", [])
        print("[resume]", ck[-1].name, flush=True)
    (out / "train_meta.json").write_text(json.dumps({"cfg": asdict(cfg), "n_rooms": len(rooms),
                                                     "grid": model.grid_cfg}, indent=1))
    t0 = time.time()
    for it in range(start + 1, cfg.n_iters + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        ids = torch.randint(0, len(rooms), (cfg.rooms_per_step,), device=dev, generator=gen).tolist()
        agg = {}
        for i in ids:
            room = rooms[i]
            ri = torch.randint(0, len(room.rx), (cfg.rx_per_room,), device=dev, generator=gen)
            H, c, k, nq = room_field(model, room, room.rx[ri])
            t = _losses(H, room.H[ri])
            l_field = t["L_spec_real"] + t["L_spec_imag"] + t["L_amp"] + 0.1 * t["L_phase"]
            kt = room.k_true[1:25]
            phi_q = model.phi_raw(room.q_pos, c) / nq
            ka, kb = float(room.k_true[20]), float(room.k_true[24])
            l_sub = subspace_loss(phi_q, soft_weights(k, ka, kb), room.psi_q[:, 1:25],
                                  soft_weights(kt, ka, kb), room.w_q)
            l_k = ((k - kt).abs() / kt).mean()
            l_orth = orthonormality_loss(phi_q, room.w_q)
            loss = cfg.w_field * l_field + cfg.w_sub * l_sub + cfg.w_k * l_k + cfg.w_orth * l_orth
            terms = {"L_field": l_field, "L_sub": l_sub, "L_k": l_k, "L_orth": l_orth}
            if cfg.phys_weight > 0:
                r_int, r_wall = phys_residuals(model, room, c, k, nq, cfg.h_fd, cfg.n_colloc, gen)
                loss = loss + cfg.phys_weight * (r_int + r_wall)
                terms.update({"R_int": r_int, "R_wall": r_wall})
            (loss / len(ids)).backward()
            for kk_, v in terms.items():
                agg[kk_] = agg.get(kk_, 0.0) + float(v) / len(ids)
            agg["loss"] = agg.get("loss", 0.0) + float(loss) / len(ids)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % cfg.log_every == 0:
            scal.append({"iter": it, **agg})
            print("[{:6d}/{}] ".format(it, cfg.n_iters) + " ".join("{}={:.4f}".format(k, v) for k, v in agg.items())
                  + " ({:.3f}s/it)".format((time.time() - t0) / max(it - start, 1)), flush=True)
        if it % cfg.val_every == 0 or it == cfg.kill_iter:
            ev = dense_eval(model, cfg, dev)
            r = float(np.mean([e["r_25_100"] for e in ev]))
            scal.append({"iter": it, "phase": "val", "in_family_r_25_100": r})
            print("  [val {}] in-family off-lattice r 25-100 Hz = {:.4f}".format(it, r), flush=True)
            if it == cfg.kill_iter and r < cfg.kill_r:
                (out / "KILLED.json").write_text(json.dumps({"iter": it, "in_family_r_25_100": r,
                                                             "rule": "D180: < {} at {}".format(cfg.kill_r, it)}))
                save(out, it, model, opt, sched, gen, cfg, scal)
                print("[kill] in-family r {:.4f} < {} at {}: stopped (D180)".format(r, cfg.kill_r, it), flush=True)
                return
        if it % cfg.ckpt_every == 0 or it == cfg.n_iters:
            save(out, it, model, opt, sched, gen, cfg, scal)


def save(out, it, model, opt, sched, gen, cfg, scal):
    p = out / "ckpt_iter{:07d}.pt".format(it)
    tmp = p.with_suffix(".pt.tmp")
    torch.save({"iter": it, "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "gen": gen.get_state(), "cfg": asdict(cfg), "scalars": scal}, tmp)
    tmp.replace(p)
    (out / "scalars.json").write_text(json.dumps(scal, indent=1))
    for old in sorted(out.glob("ckpt_iter*.pt"))[:-1]:          # keep the newest + every 10K
        if int(old.stem.replace("ckpt_iter", "")) % 10000:
            old.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    a = ap.parse_args()
    d = yaml.safe_load(open(a.config))
    known = set(NMFCfg.__dataclass_fields__)
    bad = [k for k in d if k not in known]
    if bad:
        raise SystemExit("unknown config keys {}".format(bad))
    cfg = NMFCfg(**d)
    print(json.dumps(asdict(cfg), indent=1))
    train(cfg, a.output_dir)


if __name__ == "__main__":
    main()
