"""P4-8 (D174): multi-band models (MB1 / MB2) and the spatial losses (LS1 / LS2), tested on BEHAVIOUR (Rule 8).

MB adds parameters, but MB1 vs MB2 vs a mis-rebuilt grid can share parameter COUNTS, so the grid numbers are a
checked buffer and the tests assert outputs: band locality (a band's modules move only its bins), the shared
direction encoder moving every band, FiLM woken only in its band (the init-vacuity trap), the fill above the band,
per-band seeds, and an exact load_model round trip whose teeth is a tampered grid spec.
LS1 / LS2 add no parameters at all: loss values and invariances, the pair sampler's invariants, the renderer's
jitter override (no draw, bit-identical repeat), and the two-pass LS2 step's gradients equal to a one-pass
reference (teeth: a wrong chunk weight; a pass-2 that does not reproduce pass 1 is refused).
GPU tests import tinycudann.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
MIX = "configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"
MIX_DIR = "data/track_p4_7_mix"
NB = 601


# ------------------------------------------------------------------------------------------- MB
def _cfg(**kw):
    from aaf.train.multi_room_2d_mat import P32TrainCfg
    base = dict(world_scale=10.0, base_resolution=80, n_levels=20, log2_hashmap_size=18, per_level_scale=1.5,
                signal_heads=3, signal_band_edges_hz="173.0,245.0", signal_xfade_hz=10.0, cond_source="geom_token",
                cond_dim=324, token_pool="attn_residual_extent", dc_head_bins=41)
    base.update(kw)
    return P32TrainCfg(**base)


def _model(mb=""):
    from aaf.models.inr_2d import INR2D_AutoDecoder
    from aaf.train.multi_room_2d_mat import multiband_kwargs, signal_kwargs
    c = _cfg(mb_grids=mb)
    torch.manual_seed(0)
    hg = dict(otype="HashGrid", n_levels=c.n_levels, n_features_per_level=2, log2_hashmap_size=c.log2_hashmap_size,
              base_resolution=c.base_resolution, per_level_scale=c.per_level_scale)
    m = INR2D_AutoDecoder(n_rooms=2, latent_dim=16, n_freq_bins=4097, hash_grid_config=hg, conditioning_type="film",
                          cond_source="geom_token", cond_dim=324, world_scale=10.0, token_pool="attn_residual_extent",
                          dc_head_bins=41, **signal_kwargs(c, NB), **multiband_kwargs(c))
    return m.cuda().eval()


def _inputs(seed=1, B=2, N=40):
    g = torch.Generator().manual_seed(seed)
    pts = (torch.rand(B, N, 2, generator=g) * 4 + 0.5).cuda()
    view = torch.nn.functional.normalize(torch.randn(B, N, 2, generator=g), dim=-1).cuda()
    tx = torch.tensor([0.5, 1.6]).view(1, 1, 2).expand(B, N, 2).contiguous().cuda()
    z = torch.rand(B, 324, generator=g).cuda()
    return pts, view, tx, z


def _raw(m, z=None, seed=1):
    pts, view, tx, z0 = _inputs(seed)
    with torch.no_grad():
        return m(pts, view, tx, z_s=z0 if z is None else z)


def _out(m, z=None, seed=1):
    a, s = _raw(m, z, seed)
    return a.to(torch.complex64), s.to(torch.complex64)


def _bins_changed(x, y):
    d = (x - y).abs().reshape(-1, x.shape[-1]).amax(0)
    return torch.nonzero(d > 0).squeeze(1).cpu()


def test_mb1_grid_numbers():
    from aaf.train.multi_room_2d_mat import mb_grid_table, multiband_kwargs
    g = multiband_kwargs(_cfg(mb_grids="matched"))["mb_grid_cfgs"]
    assert [x["base_resolution"] for x in g] == [11, 11, 11] and [x["n_levels"] for x in g] == [6, 6, 6]
    np.testing.assert_allclose([x["per_level_scale"] for x in g], [1.4933, 1.6000, 1.6655], atol=2e-4)
    tab = mb_grid_table(g, 10.0)
    for t, f in zip(tab, (173.0, 245.0, 300.0)):
        assert abs(t["levels"][0]["cell_m"] - 1.0) < 1e-5
        assert abs(t["levels"][-1]["cell_m"] - 343.0 / (16 * f)) < 1e-5 * 343.0 / (16 * f) + 1e-6
        assert all(lv["dense"] for lv in t["levels"])
    assert multiband_kwargs(_cfg()) == {}
    d = multiband_kwargs(_cfg(mb_grids="default"))["mb_grid_cfgs"]
    assert all(x["n_levels"] == 20 and x["base_resolution"] == 80 for x in d)


@GPU
def test_mb1_encoders_are_small_dense_tables():
    from aaf.train.multi_room_2d_mat import mb_grid_table, multiband_kwargs
    m = _model("matched")
    tab = mb_grid_table(multiband_kwargs(_cfg(mb_grids="matched"))["mb_grid_cfgs"], 10.0)
    for d, t in zip(m._mb_bands, tab):
        assert d["pos"].n_output_dims == 12
        want = sum(int(math.ceil(lv["res"] ** 2 / 8.0) * 8) * 2 for lv in t["levels"])
        assert d["pos"].params.numel() == want, (d["pos"].params.numel(), want)


@GPU
def test_default_model_has_no_mb_modules_and_mb_releases_the_shared_trunk():
    base, mb = _model(), _model("default")
    assert base._mb_bands is None and not any(k.startswith(("_mb_bands", "mb_grid_spec")) for k in base.state_dict())
    assert mb._pos_encoding is None and mb._model_decoder_sigma is None and mb._signal_nets is None
    # the SHARED modules start exactly as in the default model (built before any MB module)
    for k in ("segment_encoder", "q_proj", "k_proj", "v_proj", "o_proj", "dc_head", "_dir_encoding"):
        sb, sm = getattr(base, k).state_dict(), getattr(mb, k).state_dict()
        assert all(torch.equal(sb[n], sm[n]) for n in sb), k
    # MB2's band-0 grid has the default config AND the default seed -> the default initialisation
    assert torch.equal(mb._mb_bands[0]["pos"].params, base._pos_encoding.params)


@GPU
def test_mb_band_locality_and_shared_direction_encoder():
    m = _model("matched")
    a0, s0 = _out(m)
    lohi = m.signal_head_lohi.cpu()
    for k in range(3):
        lo, hi = int(lohi[k, 0]), int(lohi[k, 1])
        d = m._mb_bands[k]
        keep = d["dec_sigma"].params.data.clone()
        d["dec_sigma"].params.data += 0.05 * torch.randn_like(keep)
        a1, s1 = _out(m)
        d["dec_sigma"].params.data.copy_(keep)
        ch = _bins_changed(a1, a0)
        assert ch.numel() > 0 and ch.min() >= lo and ch.max() <= hi and torch.equal(s1, s0)
        keep = d["net"].params.data.clone()
        d["net"].params.data += 0.05 * torch.randn_like(keep)
        a1, s1 = _out(m)
        d["net"].params.data.copy_(keep)
        ch = _bins_changed(s1, s0)
        assert ch.numel() > 0 and ch.min() >= lo and ch.max() <= hi and torch.equal(a1, a0)
        keep = d["pos_sig"].params.data.clone()
        d["pos_sig"].params.data += 0.1 * torch.randn_like(keep)
        a1, s1 = _out(m)
        d["pos_sig"].params.data.copy_(keep)
        assert torch.equal(a1, a0)                                       # a signal grid never moves sigma
    keep = m._dir_encoding.params.data.clone()
    m._dir_encoding.params.data += 0.1 * torch.randn_like(keep)
    a1, s1 = _out(m)
    m._dir_encoding.params.data.copy_(keep)
    ch = _bins_changed(s1, s0)
    assert torch.equal(a1, a0)
    for k in range(3):                                                    # the shared encoder moves every band
        lo, hi = int(lohi[k, 0]), int(lohi[k, 1])
        assert ((ch >= lo) & (ch <= hi)).any()


@GPU
def test_mb_film_vacuity_then_only_its_band():
    m = _model("matched")
    _p, _v, _t, z = _inputs(1)
    z2 = torch.rand_like(z)
    a1, s1 = _out(m, z)
    a2, s2 = _out(m, z2)
    assert torch.equal(a1, a2) and torch.equal(s1, s2)                   # identity FiLM: z has no effect at init
    with torch.no_grad():
        torch.nn.init.normal_(m._mb_bands[1]["film_sigma"].weight, std=0.05)
    a1, _ = _out(m, z)
    a2, _ = _out(m, z2)
    lo, hi = (int(x) for x in m.signal_head_lohi[1])
    ch = _bins_changed(a1, a2)
    assert ch.numel() > 0 and ch.min() >= lo and ch.max() <= hi


@GPU
def test_mb_fill_dtype_and_seeds():
    m, pb3 = _model("default"), _model()
    a, s = _raw(m)
    ap, _sp = _raw(pb3)
    assert a.dtype == ap.dtype and s.dtype == ap.dtype
    fill = torch.tensor(m.mb_sigma_fill, dtype=a.real.dtype)
    a, s = a.to(torch.complex64), s.to(torch.complex64)
    assert torch.all(a.real[..., NB:] == fill.float().to(a.device)) and torch.all(a.imag[..., NB:] == 0)
    assert torch.all(s[..., NB:] == 0) and torch.all(s.imag[..., 0] == 0)
    p = [d["pos"].params for d in m._mb_bands]
    assert not torch.equal(p[0], p[1]) and not torch.equal(p[1], p[2]) and not torch.equal(p[0], p[2])


@GPU
def test_mb_refusals():
    m = _model("matched")
    with pytest.raises(ValueError, match="coarse-to-fine"):
        m.set_active_levels(3)
    from aaf.models.inr_2d import INR2D_AutoDecoder
    from aaf.train.multi_room_2d_mat import multiband_kwargs
    with pytest.raises(ValueError, match="three band-only heads"):
        INR2D_AutoDecoder(n_rooms=1, latent_dim=16, n_freq_bins=4097, conditioning_type="film", cond_source="geom_token",
                          cond_dim=324, world_scale=10.0, token_pool="attn_residual_extent",
                          **multiband_kwargs(_cfg(mb_grids="default")))
    with pytest.raises(ValueError, match="world_scale"):
        multiband_kwargs(_cfg(mb_grids="matched", world_scale=None))


# ------------------------------------------------------------------------------------------- LS1 / LS2 losses
def test_ls1_loss_values():
    from aaf.train.multi_room_2d_mat import ls1_loss
    g = torch.Generator().manual_seed(0)
    T = torch.complex(torch.randn(8, NB, generator=g), torch.randn(8, NB, generator=g))
    m_b = torch.rand(NB - 40, generator=g) + 0.5
    assert float(ls1_loss(T, T, 40, NB, m_b, 1.3)) == 0.0
    P = T + torch.complex(torch.randn(8, NB, generator=g), torch.randn(8, NB, generator=g)) * 0.1
    base = ls1_loss(P, T, 40, NB, m_b, 1.3)
    Q = P.clone()
    Q[0::2] += 5.0
    Q[1::2] += 5.0                                                        # a common shift of both pair members
    assert torch.allclose(ls1_loss(Q, T, 40, NB, m_b, 1.3), base, rtol=1e-5)
    Q = P.clone()
    Q[0::2, 100] += 5.0                                                   # one member only
    assert not torch.allclose(ls1_loss(Q, T, 40, NB, m_b, 1.3), base)
    Q = P.clone()
    Q[:, :40] += 7.0                                                      # bins outside [40, 601) are not seen
    assert torch.equal(ls1_loss(Q, T, 40, NB, m_b, 1.3), base)
    assert torch.allclose(ls1_loss(torch.zeros_like(T), T, 40, NB, m_b, 2.0),
                          2.0 * ((T[0::2, 40:] - T[1::2, 40:]).abs() / m_b).mean())


def test_ls2_loss_invariances():
    from aaf.train.multi_room_2d_mat import ls2_loss
    bands = [(40, 346), (346, 490), (490, 601)]
    g = torch.Generator().manual_seed(1)
    T = torch.complex(torch.randn(32, NB, generator=g), torch.randn(32, NB, generator=g))
    assert float(ls2_loss(T, T, bands)) < 1e-6
    gain = (torch.rand(NB, generator=g) + 0.5)
    assert float(ls2_loss(T * gain, T, bands)) < 1e-5                     # a per-bin gain is a dB offset: r unchanged
    perm = T[torch.randperm(32, generator=g)]
    assert float(ls2_loss(perm, T, bands)) > 0.3
    Q = T.clone()
    Q[:, 490:] = perm[:, 490:]                                            # only band C wrong -> 1/3 of its value
    full_c = ls2_loss(perm[:, 490:], T[:, 490:], [(0, NB - 490)])
    assert torch.allclose(ls2_loss(Q, T, bands), full_c / 3.0, atol=1e-6)
    Q = T.clone()
    Q[:, :40] = perm[:, :40]
    assert float(ls2_loss(Q, T, bands)) < 1e-6                            # bins outside the bands are not seen


# ------------------------------------------------------------------------------------------- CPU stubs for the steps
class _StubRenderer(torch.nn.Module):
    """Deterministic, jitter-dependent stand-in for model + renderer: H = Lin([rx, tx, cos j, sin j])."""
    n_azi = 64

    def __init__(self, noisy_second_call=False):
        super().__init__()
        self.calls = []
        self.noisy = noisy_second_call
        self.seen = {}

    def forward(self, model, rx, tx, room_min, room_max, z_s=None, jitter=None):
        self.calls.append(rx.detach().clone())
        j = torch.zeros(1) if jitter is None else jitter
        feat = torch.cat([rx, tx, torch.cos(j).expand(rx.shape[0], 1), torch.sin(j).expand(rx.shape[0], 1)], -1)
        o = model(feat)
        H = torch.complex(o[:, :NB], o[:, NB:])
        key = tuple(rx.flatten().tolist())
        if self.noisy and key in self.seen:
            H = H + 1e-3
        self.seen[key] = True
        return H


def _stub(ls_mode, **kw):
    from aaf.train.multi_room_2d_mat import P32TrainCfg, P32Trainer
    torch.manual_seed(0)
    t = P32Trainer.__new__(P32Trainer)
    t.cfg = P32TrainCfg(ls_mode=ls_mode, batch_size=8, rx_per_config=2, grad_accum_steps=2, max_rows_per_render=8,
                        signal_band_edges_hz="173.0,245.0", fs=4096, n_time_samples=8192, **kw)
    t.device = torch.device("cpu")
    t.model = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2 * NB))
    t.opt = torch.optim.SGD(t.model.parameters(), lr=0.0)
    t.sched = torch.optim.lr_scheduler.LambdaLR(t.opt, lambda s: 1.0)
    C, n_rx = 3, 60
    g = torch.Generator().manual_seed(3)
    xs = torch.arange(10) * 0.12 + 0.3                                   # a 0.12 m lattice -> neighbours within 0.30
    grid = torch.stack(torch.meshgrid(xs, xs[:6], indexing="ij"), -1).reshape(-1, 2)
    t.rx = torch.stack([grid + 0.01 * c for c in range(C)])
    t.H = torch.complex(torch.randn(C, n_rx, NB, generator=g), torch.randn(C, n_rx, NB, generator=g))
    t.configs = list(range(C))
    t.train_rx_idx = torch.tensor([i for i in range(n_rx) if i % 8 != 3])
    t.geom_dims = [(5.0, 4.0), (6.0, 4.5), (5.5, 5.0)]
    t.geom_id = torch.arange(C)
    t.cond = torch.rand(C, 324, generator=g)
    t.src = torch.tensor([0.5, 1.6])
    t.band, t.n_band = (0, NB), NB
    t.loss_lo_bin, t.spec_w, t.amp_eps = 0, None, None
    t.renderer = _StubRenderer()
    t._ls = t._init_ls()
    return t


def test_ls1_sampler_and_normaliser():
    t = _stub("ls1", ls1_weight=0.3)
    L = t._ls
    tr = t.train_rx_idx
    for _ in range(50):
        ci, ri, dist = t._ls1_draw()
        assert torch.equal(ci[0::2], ci[1::2])                             # both members from the same room
        assert all(int(r) in set(tr.tolist()) for r in ri)
        assert (dist <= 0.30 + 1e-6).all() and (dist > 0).all()
    # m_b against brute force over the sampler's distribution
    lo, hi = L["lo"], L["hi"]
    want = torch.zeros(hi - lo)
    for c in range(3):
        P = t.rx[c, tr].double()
        d2 = ((P[:, None] - P[None]) ** 2).sum(-1)
        acc, na = torch.zeros(hi - lo), 0
        for a in range(len(tr)):
            nb = [j for j in range(len(tr)) if j != a and d2[a, j] <= 0.09 + 1e-9]
            if nb:
                acc += torch.stack([(t.H[c, tr[a], lo:hi] - t.H[c, tr[j], lo:hi]).abs() for j in nb]).mean(0)
                na += 1
        want += acc / na / 3
    assert torch.allclose(L["m_b"], want, rtol=1e-4)
    S = sum(float(t.H[c, tr].real.abs().mean() + t.H[c, tr].imag.abs().mean()) for c in range(3)) / 3
    assert abs(L["S"] - S) < 1e-5 * S


def test_ls1_pairs_share_a_renderer_call():
    t = _stub("ls1", ls1_weight=0.3)
    t.max_calls = []
    rows, ri, _ = t._ls1_draw()
    t.renderer.calls.clear()
    t._render(rows, ri)
    flat = {tuple(r.tolist()) for call in t.renderer.calls for r in call}
    for call in t.renderer.calls:
        assert call.shape[0] % 2 == 0
    for k in range(0, ri.numel(), 2):                                     # every pair lands in ONE call
        a, n = tuple(t.rx[rows[k], ri[k]].tolist()), tuple(t.rx[rows[k + 1], ri[k + 1]].tolist())
        assert any(a in {tuple(r.tolist()) for r in c} and n in {tuple(r.tolist()) for r in c} for c in t.renderer.calls)
    assert flat


def _grads(t):
    return {n: p.grad.detach().clone() for n, p in t.model.named_parameters()}


@pytest.mark.parametrize("frac_bug", [False, True])
def test_ls2_two_pass_equals_one_pass(frac_bug):
    from aaf.train import multi_room_2d_mat as M
    t = _stub("ls2", ls2_weight=0.3, ls2_rx_per_step=20)
    got = {}
    t._apply_grads = lambda: got.update(_grads(t))
    st = torch.get_rng_state()
    if frac_bug:                                                          # teeth: the PB3 term not averaged over chunks
        orig = M._losses
        M._losses = lambda *a, **k: {kk: v * 2.0 for kk, v in orig(*a, **k).items()}
    try:
        t._step_ls2(1)
    finally:
        if frac_bug:
            M._losses = orig
    torch.set_rng_state(st)
    c = int(torch.randint(0, 3, (1,)))
    ri = t.train_rx_idx[torch.randperm(t.train_rx_idx.numel())[:20]]
    j = torch.empty(1).uniform_(0.0, 2.0 * math.pi / 64)
    t.model.zero_grad(set_to_none=True)
    H = t._render_room(c, ri, j)
    T = t.H[c, ri]
    terms = M._losses(H, T, 0)
    w_r, w_i, w_a, w_p = t.cfg.weights
    ref = (w_r * terms["L_spec_real"] + w_i * terms["L_spec_imag"] + w_a * terms["L_amp"] + w_p * terms["L_phase"]
           + 0.3 * M.ls2_loss(H, T, t._ls["bands"], t.cfg.ls2_floor_db))
    ref.backward()
    want = _grads(t)
    close = all(torch.allclose(got[n], want[n], rtol=1e-4, atol=1e-6) for n in want)
    assert close != frac_bug


def test_ls2_pass2_must_reproduce_pass1():
    t = _stub("ls2", ls2_weight=0.3, ls2_rx_per_step=16)
    t.renderer = _StubRenderer(noisy_second_call=True)
    t._apply_grads = lambda: None
    with pytest.raises(RuntimeError, match="does not reproduce pass 1"):
        t._step_ls2(1)


@GPU
def test_renderer_jitter_override():
    from aaf.renderers.freq_2d import FreqRenderer2D
    r = FreqRenderer2D(n_azi=16, n_pts_per_ray=8, near=0.0, fs=4096, n_time_samples=8192, c=343.0).cuda().train()
    dev = torch.device("cuda")
    s0 = torch.cuda.get_rng_state()
    d_default = r._ray_directions_2d(dev)
    s1 = torch.cuda.get_rng_state()
    assert not torch.equal(s0, s1)                                        # the default path draws once
    torch.cuda.set_rng_state(s0)
    j = torch.empty(1, device=dev).uniform_(0.0, 2.0 * math.pi / 16)      # the value the default drew
    assert torch.equal(torch.cuda.get_rng_state(), s1)
    s2 = torch.cuda.get_rng_state()
    d_over = r._ray_directions_2d(dev, j)
    assert torch.equal(torch.cuda.get_rng_state(), s2)                    # the override draws nothing
    assert torch.equal(d_over, d_default)
    assert torch.equal(r._ray_directions_2d(dev, j), d_over)
    assert not torch.equal(r._ray_directions_2d(dev, j + 0.01), d_over)


# ------------------------------------------------------------------------------------------- real trainer (GPU)
def _tiny_mix(tmp_path, n_c500=2, n_cx=2):
    m = json.load(open(MIX))
    rows = [r for r in m["configs"] if r["origin"] == "C500"][:n_c500]
    rows += [r for r in m["configs"] if r["origin"] == "CX500"][:n_cx]
    out = tmp_path / "man.json"
    out.write_text(json.dumps(dict(m, configs=rows)))
    return str(out)


def _trainer(tmp_path, name, **kw):
    import yaml

    from aaf.train.multi_room_2d_mat import P32TrainCfg, P32Trainer
    base = yaml.safe_load(open("configs/sweep_2d_mat/P4_7_GX.yaml"))
    base = {k: v for k, v in base.items() if k in P32TrainCfg.__dataclass_fields__}
    base.update({"configs_manifest": _tiny_mix(tmp_path), "n_iters": 4, "val_max_configs": 2})
    base.update(kw)
    return P32Trainer(P32TrainCfg(**base), tmp_path / name)


@GPU
def test_real_trainer_default_step_fn_and_refusals(tmp_path):
    t = _trainer(tmp_path, "gx")
    assert t._step_fn == t._step and t._ls is None and t.model._mb_bands is None
    with pytest.raises(ValueError, match="c2f_levels|coarse-to-fine"):
        _trainer(tmp_path, "bad", mb_grids="matched", c2f_levels="10:1000")
    with pytest.raises(ValueError, match="ls_mode must be"):
        _trainer(tmp_path, "bad2", ls_mode="ls3")


@GPU
def test_real_trainer_ls1_and_ls2_steps_run(tmp_path):
    t1 = _trainer(tmp_path, "ls1", ls_mode="ls1", ls1_weight=0.3)
    m = t1._step_fn(1)
    assert m["ls1_pairs_per_step"] == 8.0 and m["L_ls1"] > 0 and 0 < m["ls1_frac"] < 1
    assert t1._ls["info"]["anchors_without"] < 0.01 * t1._ls["info"]["anchors_with_neighbour"]
    t2 = _trainer(tmp_path, "ls2", ls_mode="ls2", ls2_weight=0.1, ls2_rx_per_step=16, ls2_det_check_steps=5)
    for it in (1, 2):                                                     # the bitwise pass-1 / pass-2 check on the real model
        m = t2._step_fn(it)
    assert m["ls2_rx_per_room"] == 16.0 and 0 <= m["L_ls2"] <= 2


@GPU
def test_real_trainer_mb_round_trip_and_spec_teeth(tmp_path):
    from aaf.eval.p3_2_eval import load_model
    t = _trainer(tmp_path, "mb1", mb_grids="matched")
    assert t.model._mb_bands is not None
    t.model.eval(); t.renderer.eval()
    ci = torch.tensor([0, 1, 2, 3], device=t.device); ri = torch.tensor([3, 50, 9, 700], device=t.device)
    with torch.no_grad():
        want = t._render(ci, ri)
    t.save_ckpt(4)
    (t.out / "train_meta.json").write_text(json.dumps({"n_configs": len(t.configs)}))
    ck = t.out / "ckpt_iter0000004.pt"
    m, r, _cfg_, _meta, _it = load_model(ck, t.device)
    t2_model, t.model = t.model, m
    with torch.no_grad():
        got = t._render(ci, ri)
    t.model = t2_model
    assert torch.equal(got, want)
    st = torch.load(str(ck), map_location="cpu")
    st["model"]["mb_grid_spec"][0, 2] += 1e-4                            # same table size, different grid
    torch.save(st, str(ck))
    with pytest.raises(RuntimeError, match="grid spec mismatch"):
        load_model(ck, t.device)


@GPU
def test_resume_guard_and_unknown_keys_are_fatal(tmp_path, monkeypatch):
    import inspect
    import sys

    from aaf.train import multi_room_2d_mat as M
    src = inspect.getsource(M)
    guard = src[src.index("def _maybe_resume"):src.index("def _early_stop")]
    assert "P48_KEYS" in guard and set(M.P48_KEYS) <= set(M.P32TrainCfg.__dataclass_fields__)
    cfgf = tmp_path / "c.yaml"
    cfgf.write_text("run_id: x\nls_mod: ls1\n")
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfgf), "--output_dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="unknown config keys"):
        M.main()
