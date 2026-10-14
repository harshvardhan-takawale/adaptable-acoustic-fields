"""P4-5 A1: where in the band does the training GRADIENT actually come from?

Scored against D92, which was committed before this ran. Three quantities per frequency, because
they answer different questions and the hypothesis being tested conflates them:

  1. LOSS-VALUE share -- what `p4_1_dc_diagnostic.py` measured and called "gradient". Reported so
     the old 23.7% can be reconciled, NOT as the scored quantity.
  2. OUTPUT-gradient share |dL/dP_b| -- the gradient arriving at the network's output. For the
     L1 spec terms this is uniform in b by construction (D92(c)(i)); measured anyway, so the
     claim is checked rather than asserted.
  3. PARAMETER-gradient share -- what the optimiser actually receives. THE SCORED QUANTITY.
     The total weighted loss is split exactly into (term, 10 Hz band) pieces whose sum is the
     training loss, each piece is back-propagated to the parameters, and its share is the
     PROJECTION <g_tG, g_tot> / ||g_tot||^2 onto the net gradient. Those shares sum to exactly 1.
     A band can be negative: it pulls against the step the optimiser takes. The norm share
     ||g_G|| / sum ||g_G'|| is reported beside it, and B1b is calibrated from the spec-term norm
     share (a projection can be <= 0 and cannot be inverted into a weight).

THE FORWARD IS THE TRAINER'S. Rows are drawn as `_step` draws them (configs uniform with
replacement, train receivers only, `val_rx_indices` held out), rendered as `_render` renders them
(grouped by geometry, room_max = [L, W], conditioning from `build_cond_vector_2d` with the
polygon), at the trainer's 8 rows per backward -- 16 OOMs a 24 GB card (P4-1, measured). The
weights are read from the checkpoint, not assumed.

ONE DEPARTURE, AND WHY. The renderer runs in eval mode, which removes the trainer's azimuth
jitter. The decomposition needs the net gradient g_tot BEFORE it can project anything onto it,
so every batch is rendered twice (pass 1 for g_tot, pass 2 for the pieces), and the projection
is only exact if both passes see the identical graph. Jitter would make them differ.

Two model states: `p4_4_C250` at 60K (the baseline every arm is compared with) and a fresh
seed-0 model (what every arm sees at iteration 0, since the arms train from scratch).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from aaf.data.shape_configs import configs_from_rows
from aaf.models.conditioning_2d import build_cond_vector_2d
from aaf.renderers.freq_2d import FreqRenderer2D
from aaf.train.multi_room_2d_mat import val_rx_indices

DF = 0.5
N_BINS = 601
BAND_HZ = 10.0
TERMS = ("L_spec_real", "L_spec_imag", "L_amp", "L_phase")
CUM_EDGES_HZ = (50.0, 100.0, 150.0, 200.0, 300.0)
PREDICTION = {"above_hz": 100.0, "max_share": 0.15,
              "text": "bins above 100 Hz contribute under 15% of total gradient combined"}


def bands():
    """30 bands of 10 Hz; bin 600 (exactly 300 Hz) joins the last band."""
    per = int(round(BAND_HZ / DF))
    out = [list(range(k * per, (k + 1) * per)) for k in range(N_BINS // per)]
    out[-1].append(N_BINS - 1)
    assert sorted(b for g in out for b in g) == list(range(N_BINS))
    return out


def elementwise_terms(P, T):
    """`_losses` BEFORE its mean -- the identical formulas, kept elementwise [rows, bins]."""
    eps = 1e-6
    return {
        "L_spec_real": (P.real - T.real).abs(),
        "L_spec_imag": (P.imag - T.imag).abs(),
        "L_amp": (torch.log10(P.abs() + eps) - torch.log10(T.abs() + eps)).abs(),
        "L_phase": 1.0 - torch.cos(P.angle() - T.angle()),
    }


def build_model(ckpt, dev, fresh):
    """load_model's construction, verbatim; `fresh` skips the weights and seeds 0."""
    from aaf.models.inr_2d import INR2D_AutoDecoder
    st = torch.load(str(ckpt), map_location=dev)
    cfg = dict(st["cfg"])
    meta_p = Path(ckpt).parent / "train_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    hg = dict(otype="HashGrid", n_levels=int(cfg["n_levels"]), n_features_per_level=2,
              log2_hashmap_size=int(cfg["log2_hashmap_size"]),
              base_resolution=int(cfg.get("base_resolution", 16)),
              per_level_scale=float(cfg["per_level_scale"]))
    if fresh:
        torch.manual_seed(0)
    model = INR2D_AutoDecoder(
        n_rooms=int(meta.get("n_configs", 1)), latent_dim=int(cfg["latent_dim"]),
        n_freq_bins=int(cfg["n_time_samples"]) // 2 + 1, hash_grid_config=hg,
        conditioning_type=str(cfg.get("conditioning_type", "film")),
        cond_source=str(cfg["cond_source"]), cond_dim=int(cfg["cond_dim"]),
        world_scale=cfg.get("world_scale"), token_pool=(cfg.get("token_pool") or None),
        dc_head_bins=int(cfg.get("dc_head_bins", 0) or 0),
        l_head_enabled=False).to(dev)
    if not fresh:
        model.load_state_dict(st["model"])
    model.train()                                       # as `_step`; the model has no dropout
    renderer = FreqRenderer2D(
        n_azi=int(cfg["n_azi"]), n_pts_per_ray=int(cfg["n_pts_per_ray"]),
        near=float(cfg["near"]), fs=int(cfg["fs"]),
        n_time_samples=int(cfg["n_time_samples"]), c=float(cfg["c"])).to(dev)
    renderer.eval()                                     # see module docstring
    w = tuple(float(x) for x in cfg.get("weights", (1.0, 1.0, 1.0, 0.1)))
    return model, renderer, cfg, w, int(st["iter"])


class Corpus:
    """The training corpus exactly as the trainer preloads it."""

    def __init__(self, manifest, data_dir, cfg, dev):
        man = json.load(open(manifest))
        rows = man["configs"]
        if str(man.get("schema", "")).startswith("p3_3fast.trackB"):
            # D137 item 4: the doorway corpus -- the trainer's own config branch and kinds filter
            from aaf.data.aperture_configs import configs_from_rows as aper_rows
            kinds = cfg.get("config_kinds")
            self.configs = aper_rows(rows, split="train", **({"kinds": tuple(kinds)} if kinds else {}))
        else:
            self.configs = configs_from_rows(rows, split="train")
        H, rx, cond, gid, self.geom = [], [], [], [], []
        gidx = {}
        for c in self.configs:
            with h5py.File(Path(data_dir) / c.filename) as f:
                H.append(np.asarray(f["ism/H_complex"][:, :N_BINS], dtype=np.complex64))
                rx.append(np.asarray(json.loads(f.attrs["receiver_pos"]), dtype=np.float32))
                src = np.asarray(json.loads(f.attrs["source_pos"]), dtype=np.float32)
            if (c.L, c.W) not in gidx:
                gidx[(c.L, c.W)] = len(self.geom)
                self.geom.append((c.L, c.W))
            gid.append(gidx[(c.L, c.W)])
            # the trainer's call (multi_room_2d_mat.py:366-371), schema-agnostic
            _v = getattr(c, "verts", None)
            cond.append(build_cond_vector_2d(cfg["cond_source"], c.L, c.W,
                                             c.edge_alphas if _v is not None else c.alphas,
                                             x0=getattr(c, "x0", None), a=getattr(c, "a", None),
                                             verts=_v).numpy())
        self.H = torch.tensor(np.stack(H), device=dev)
        self.rx = torch.tensor(np.stack(rx), device=dev)
        self.src = torch.tensor(src, device=dev)
        self.cond = torch.tensor(np.stack(cond), dtype=torch.float32, device=dev)
        self.gid = torch.tensor(gid, device=dev)
        n_rx = self.rx.shape[1]
        val = set(val_rx_indices(n_rx))
        self.train_rx = torch.tensor([i for i in range(n_rx) if i not in val], device=dev)
        self.dev = dev

    def sample(self, n_batches, rows, rx_per_cfg, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        out = []
        for _ in range(n_batches):
            ci = torch.randint(0, len(self.configs), (rows // rx_per_cfg,), generator=g)
            ci = ci.repeat_interleave(rx_per_cfg)
            ri = self.train_rx.cpu()[torch.randint(0, len(self.train_rx), (ci.numel(),),
                                                   generator=g)]
            out.append((ci.to(self.dev), ri.to(self.dev)))
        return out

    def render(self, model, renderer, ci, ri):
        """`_render`, verbatim in substance."""
        out = torch.zeros(ci.numel(), N_BINS, dtype=torch.complex64, device=self.dev)
        g = self.gid[ci]
        for gid in torch.unique(g):
            sel = torch.nonzero(g == gid).squeeze(1)
            L, W = self.geom[int(gid)]
            room_min = torch.zeros(2, device=self.dev)
            room_max = torch.tensor([L, W], device=self.dev, dtype=torch.float32)
            H = renderer(model, self.rx[ci[sel], ri[sel]],
                         self.src.unsqueeze(0).expand(sel.numel(), -1), room_min, room_max,
                         z_s=self.cond[ci[sel]])
            out[sel] = H[:, :N_BINS]
        return out


def _flat(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, params)])


def measure(model, renderer, w, corpus, batches):
    params = [p for p in model.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in params)
    B = bands()
    wt = dict(zip(TERMS, w))
    n_rows = sum(b[0].numel() for b in batches)

    loss_val = np.zeros((len(TERMS), N_BINS))
    out_grad = np.zeros((len(TERMS), N_BINS))

    # ---- pass 1: the net gradient over the whole sample (+ the cheap per-bin quantities) -----
    g_tot = torch.zeros(n_par, device=corpus.dev)
    for ci, ri in batches:
        P, T = corpus.render(model, renderer, ci, ri), corpus.H[ci, ri]
        e = elementwise_terms(P, T)
        denom = float(P.shape[0] * N_BINS)               # `.mean()` over rows x bins
        for k, t in enumerate(TERMS):
            loss_val[k] += wt[t] * e[t].detach().sum(0).cpu().numpy() / denom
            gP = torch.autograd.grad(wt[t] * e[t].mean(), P, retain_graph=True)[0]
            out_grad[k] += gP.abs().sum(0).cpu().numpy()
        loss = sum(wt[t] * e[t].mean() for t in TERMS)
        # allow_unused: parameters the geom-token path never touches (e.g. a latent table)
        # get no gradient and must count as zero, not raise.
        g_tot += _flat(torch.autograd.grad(loss, params, allow_unused=True), params)
        del P, e, loss
        # tiny-cuda-nn allocates its workspace through its OWN arena (cuMemCreate), outside
        # PyTorch's caching allocator. Freed graph memory that PyTorch keeps cached is invisible
        # to it, so after many renders tcnn's next allocation fails even though PyTorch holds
        # gigabytes it is not using. The full run died exactly so, on its first pass-2 backward
        # after 48 pass-1 batches; the 2-batch smoke never built the cache up. Return it.
        torch.cuda.empty_cache()
    gg = float(g_tot @ g_tot)

    # ---- pass 2: every (term, band) piece, BAND-OUTER --------------------------------------
    # Band outer, batch inner, so only TWO full-length accumulators are alive at once. The
    # batch-outer order needed one per band per kind -- 60 vectors of n_params floats, ~19 GB
    # at this parameter count -- and OOMed a 24 GB card in the smoke run. The price is one
    # re-render per (band, batch), ~30% more time; the two model states run as parallel jobs.
    proj = np.zeros((len(TERMS), len(B)))               # exact, additive
    proj_batch = np.zeros((len(batches), len(B)))       # for a jackknife SE
    nt, ns = np.zeros(len(B)), np.zeros(len(B))
    t0 = time.time()
    for j, G in enumerate(B):
        acc_tot = torch.zeros(n_par, device=corpus.dev)
        acc_spec = torch.zeros(n_par, device=corpus.dev)
        for bi, (ci, ri) in enumerate(batches):
            P, T = corpus.render(model, renderer, ci, ri), corpus.H[ci, ri]
            e = elementwise_terms(P, T)
            denom = float(P.shape[0] * N_BINS)
            for k, t in enumerate(TERMS):
                piece = wt[t] * e[t][:, G].sum() / denom
                g = _flat(torch.autograd.grad(piece, params, retain_graph=True,
                                              allow_unused=True), params)
                d = float(g @ g_tot) / gg
                proj[k, j] += d
                proj_batch[bi, j] += d
                acc_tot += g
                if t in ("L_spec_real", "L_spec_imag"):
                    acc_spec += g
                del g
            del P, e
            torch.cuda.empty_cache()                     # see pass 1: the tcnn arena
        nt[j], ns[j] = float(acc_tot.norm()), float(acc_spec.norm())
        del acc_tot, acc_spec
        print("  band {:2d}/{} [{:5.1f},{:5.1f}) Hz  proj {:+.4f}  {:.0f}s".format(
            j + 1, len(B), G[0] * DF, (G[-1] + 1) * DF, proj[:, j].sum(),
            time.time() - t0), flush=True)
    # the decomposition is exact by construction; check it rather than trust it
    closure = float(proj.sum())
    nb = len(batches)
    jk = np.array([proj_batch.sum(0) - proj_batch[i] * 1.0 for i in range(nb)])
    # jackknife over batches of the share-above-100-Hz (approximate: g_tot held fixed)
    hi_idx = [j for j, G in enumerate(B) if G[0] * DF >= PREDICTION["above_hz"]]
    jk_hi = jk[:, hi_idx].sum(1) * nb / (nb - 1)
    se_hi = float(np.sqrt((nb - 1) / nb * ((jk_hi - jk_hi.mean()) ** 2).sum()))
    return {"n_rows": n_rows, "n_params": int(n_par), "projection_closure": closure,
            "band_edges_hz": [[G[0] * DF, (G[-1] + 1) * DF] for G in B],
            "param_proj_share": {t: proj[k].tolist() for k, t in enumerate(TERMS)},
            "param_proj_share_total": proj.sum(0).tolist(),
            "param_norm_share_total": (nt / nt.sum()).tolist(),
            "param_norm_share_spec": (ns / ns.sum()).tolist(),
            "se_share_above_100hz": se_hi,
            "loss_value_share": {t: (loss_val[k] / loss_val.sum()).tolist()
                                 for k, t in enumerate(TERMS)},
            "loss_value_share_total": (loss_val.sum(0) / loss_val.sum()).tolist(),
            "output_grad_share": {t: (out_grad[k] / out_grad[k].sum()).tolist()
                                  for k, t in enumerate(TERMS)}}


def cumulative(per_bin_or_band, edges, is_band):
    out = {}
    for f in CUM_EDGES_HZ:
        if is_band:
            out["below_{:g}".format(f)] = float(sum(v for (lo, _hi), v in zip(edges, per_bin_or_band)
                                                    if lo < f))
        else:
            out["below_{:g}".format(f)] = float(sum(per_bin_or_band[:int(round(f / DF))]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/p4_4/p4_4_C250/ckpt_iter0060000.pt")
    ap.add_argument("--manifest", default="configs/sweeps_2d_mat/p4_4_shapes_250.json")
    ap.add_argument("--data-dir", default="data/track_p4_2_shapes")
    ap.add_argument("--out", default="outputs/p4_5/diagnostics")
    ap.add_argument("--n-batches", type=int, default=48)
    # 4 rows per backward, not the trainer's 8: at 8 the renderer graph alone is ~20 GB (the
    # trainer sits right at the 24 GB edge), and this script's ~1 GB of gradient accumulators
    # tipped it over twice. The decomposition is a sum over rows, so the batch split changes
    # nothing but the jackknife granularity; 48 x 4 = the same 192 rows.
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--rx-per-config", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="prefix for the output file (e.g. trackB_20K)")
    ap.add_argument("--smoke", action="store_true", help="2 batches, to fail fast")
    ap.add_argument("--state", choices=("C250_60K", "fresh_init", "both"), default="both",
                    help="one state per job lets the two run in parallel")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nb = 2 if a.smoke else a.n_batches

    report = {"prediction": PREDICTION, "checkpoint": a.checkpoint, "rows_per_batch": a.rows,
              "n_batches": nb, "renderer_mode": "eval (no azimuth jitter; see docstring)",
              "states": {}}
    corpus = None
    states = ("C250_60K", "fresh_init") if a.state == "both" else (a.state,)
    for state in states:
        model, renderer, cfg, w, it = build_model(a.checkpoint, dev, fresh=(state == "fresh_init"))
        if corpus is None:
            corpus = Corpus(a.manifest, a.data_dir, cfg, dev)
            Habs = corpus.H[:, corpus.train_rx].abs()
            report["mean_abs_H_per_bin"] = Habs.mean((0, 1)).cpu().numpy().tolist()
            report["weights"] = list(w)
            batches = corpus.sample(nb, a.rows, a.rx_per_config, a.seed)
        print("=== state {} (iter {}) ===".format(state, 0 if state == "fresh_init" else it),
              flush=True)
        m = measure(model, renderer, w, corpus, batches)
        edges = m["band_edges_hz"]
        m["cum_param_proj_total"] = cumulative(m["param_proj_share_total"], edges, True)
        m["cum_param_norm_total"] = cumulative(m["param_norm_share_total"], edges, True)
        m["cum_loss_value_total"] = cumulative(m["loss_value_share_total"], edges, False)
        above = 1.0 - m["cum_param_proj_total"]["below_100"]
        m["scored"] = {"share_above_100hz_param_proj": above,
                       "se": m["se_share_above_100hz"],
                       "share_above_100hz_param_norm": 1.0 - m["cum_param_norm_total"]["below_100"],
                       "share_above_100hz_loss_value": 1.0 - m["cum_loss_value_total"]["below_100"],
                       "prediction_holds": bool(above < PREDICTION["max_share"])}
        report["states"][state] = m
        print("  closure (should be 1.000): {:.6f}".format(m["projection_closure"]), flush=True)
        print("  share >= 100 Hz  param-proj {:.4f} +- {:.4f} | param-norm {:.4f} | loss-value "
              "{:.4f}  -> prediction (< 0.15) {}".format(
                  above, m["se_share_above_100hz"],
                  m["scored"]["share_above_100hz_param_norm"],
                  m["scored"]["share_above_100hz_loss_value"],
                  "HOLDS" if above < 0.15 else "FAILS"), flush=True)
        del model, renderer
        torch.cuda.empty_cache()
    pre = "a1_{}_".format(a.tag) if a.tag else "a1_"
    name = "a1_smoke_{}.json".format(a.state) if a.smoke else "{}{}.json".format(pre, a.state)
    json.dump(report, open(out / name, "w"), indent=1)
    print("wrote", out / name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
