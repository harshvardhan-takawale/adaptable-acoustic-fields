"""P4-7 m3 (D170(g)): the physics modal prior and its plumbing, tested on BEHAVIOUR (Rule 8).

The prior adds a buffer and the features add one zero-initialised layer, so a checkpoint round-trip
alone proves nothing -- every claim is an OUTPUT claim:
  * library: H_modal against the ACTUAL FDTD on a small room (teeth: the wrong damping sign, the continuous
    zero-mode term); degenerate modes summed, so the features are invariant to the basis chosen inside a
    degenerate eigenspace (teeth: the un-summed per-mode kernel is not);
  * model: with the zero-initialised input map the output is BIT-IDENTICAL to the model without it; woken,
    the output changes and depends on the features (the FiLM-at-init vacuity, Rule 8);
  * renderer: the prior is added exactly and only to its bins; a prior-trained model is refused without a
    prior, and a prior is refused for a model without the flag;
  * wrapper: rows map to their room (by conditioning) and receiver (by coordinates) in any order; unknown
    rooms, receivers and sources raise;
  * trainer + load_model: the trainer's render minus the prior equals the prior-free trainer's render at
    init; the EVALUATOR path (load_model -> wrapper lookup) reproduces the TRAINER path on the same rows.
GPU tests import tinycudann; the library tests are CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
FAM_PRIOR = Path("data/track_p4_7_modal/track_p4_4_family")


# ------------------------------------------------------------------------------------------- library
def _small_room_fdtd(L=1.6, W=1.2, src=(0.42, 0.36), rx=((1.1, 0.8), (0.3, 0.9), (1.4, 0.2), (0.8, 0.6))):
    import aaf.sim.fdtd_2d as F
    from aaf.eval.modal_prior import ALPHA, C0, DX, FS, N
    res = F.simulate(L, W, [ALPHA] * 4, src, np.asarray(rx), dx=DX, fs=FS, n=N, c=C0)
    return np.asarray(res["H_complex"])[:, :601], np.asarray(rx), np.asarray(src)


def test_prior_matches_the_fdtd_on_a_small_room():
    import aaf.eval.modal_prior as MP
    L, W = 1.6, 1.2
    T, rx, src = _small_room_fdtd(L, W)
    verts = [(0, 0), (L, 0), (L, W), (0, W)]
    d = MP.compute(verts, L, W, rx, src)
    band = slice(50, 601)                                     # 25-300 Hz
    err = np.linalg.norm(d["H"][:, band] - T[:, band]) / np.linalg.norm(T[:, band])
    assert err < 0.15, err
    # teeth 1: the continuous zero-mode term instead of its exact record
    op = MP.operator(verts, L, W)
    k2, psi, _ = MP.eigen(op)
    jr, js = MP.nodes(op, rx), int(MP.nodes(op, src[None])[0])
    eta = ((op["mu"] * op["D"])[:, None] * psi ** 2).sum(0) * op["dA"]
    _s, S = MP.pulse()
    f = np.arange(601) * MP.DF
    kk = 2 * np.pi * f / MP.C0

    def formula(sign):
        den = k2[:, None] - kk[None] ** 2 + sign * 1j * kk[None] * eta[:, None]
        den[:, 0] = 1.0
        G = psi[jr] @ (psi[js][:, None] / den) * (S * np.exp(1j * 2 * np.pi * f * MP.DT))[None]
        return MP.C_FIT * G

    e_cont = np.linalg.norm(formula(+1)[:, band] - T[:, band]) / np.linalg.norm(T[:, band])
    e_sign = np.linalg.norm(formula(-1)[:, band] - T[:, band]) / np.linalg.norm(T[:, band])
    assert e_cont > 1.5 * err, (e_cont, err)
    assert e_sign > 0.5, e_sign                               # teeth 2: the wrong damping sign


def test_degenerate_pairs_are_summed_so_features_are_basis_invariant():
    import aaf.eval.modal_prior as MP
    L = W = 1.2                                               # a square: (m, n) / (n, m) are degenerate
    op = MP.operator([(0, 0), (L, 0), (L, W), (0, W)], L, W)
    k2, psi, _ = MP.eigen(op)
    rx = np.asarray([[0.3, 0.5], [0.9, 0.2], [0.62, 0.84]])
    jr, js = MP.nodes(op, rx), int(MP.nodes(op, np.asarray([[0.24, 0.4]]))[0])
    cl = MP.clusters(k2)
    pair = next(c for c in cl[:12] if len(c) == 2)
    s0, _kc, sizes = MP.features(psi[jr], psi[js], k2, op["area"], 12)
    assert 2 in sizes
    th = 0.7                                                   # rotate the basis inside the eigenspace
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    psi2 = psi.copy()
    psi2[:, pair] = psi[:, pair] @ R
    s1, _, _ = MP.features(psi2[jr], psi2[js], k2, op["area"], 12)
    np.testing.assert_allclose(s1, s0, atol=1e-9)
    a, b = psi[jr][:, pair[0]] * psi[js][pair[0]], psi2[jr][:, pair[0]] * psi2[js][pair[0]]
    assert np.abs(a - b).max() > 1e-3                          # teeth: one mode alone is NOT invariant


# ------------------------------------------------------------------------------------------- model / renderer
def _model(modal_feat_dim=0, modal_prior=False):
    from aaf.models.inr_2d import INR2D_AutoDecoder
    torch.manual_seed(0)
    hg = dict(otype="HashGrid", n_levels=8, n_features_per_level=2, log2_hashmap_size=14,
              base_resolution=16, per_level_scale=1.5)
    from aaf.train.multi_room_2d_mat import P32TrainCfg, signal_kwargs
    c = P32TrainCfg(signal_heads=3, signal_band_edges_hz="173.0,245.0", signal_xfade_hz=10.0)
    m = INR2D_AutoDecoder(n_rooms=2, latent_dim=16, n_freq_bins=30721, hash_grid_config=hg,
                          conditioning_type="film", cond_source="geom_token", cond_dim=324, world_scale=10.0,
                          token_pool="attn_residual_extent", dc_head_bins=41, **signal_kwargs(c, 601),
                          **({"modal_feat_dim": modal_feat_dim, "modal_prior": modal_prior}
                             if (modal_feat_dim or modal_prior) else {}))
    return m.cuda().eval()


def _rows(B=3, F=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    rx = (torch.rand(B, 2, generator=g) * 3 + 0.5).cuda()
    tx = torch.tensor([[0.5, 1.6]] * B).cuda()
    z = torch.rand(B, 324, generator=g).cuda()
    f = torch.randn(B, F, generator=g).cuda()
    return rx, tx, z, f


def _renderer():
    from aaf.renderers.freq_2d import FreqRenderer2D
    return FreqRenderer2D(n_azi=16, n_pts_per_ray=16, near=0.0, fs=30720, n_time_samples=61440, c=343.0).cuda().eval()


BOX = (torch.zeros(2).cuda() if torch.cuda.is_available() else None, torch.tensor([5.0, 4.5]).cuda()
       if torch.cuda.is_available() else None)


@GPU
def test_zero_init_input_map_is_bit_identical_and_waking_it_changes_the_output():
    base, withf = _model(), _model(modal_feat_dim=8)
    assert all(torch.equal(base.state_dict()[k], v) for k, v in withf.state_dict().items() if k in base.state_dict())
    r = _renderer()
    rx, tx, z, f = _rows()
    with torch.no_grad():
        a = r(base, rx, tx, *BOX, z_s=z)
        b = r(withf, rx, tx, *BOX, z_s=z, modal_feat=f)
        assert torch.equal(a, b)
        torch.nn.init.normal_(withf.modal_proj.weight, std=0.05)          # wake it
        c = r(withf, rx, tx, *BOX, z_s=z, modal_feat=f)
        d = r(withf, rx, tx, *BOX, z_s=z, modal_feat=-f)
    assert not torch.equal(c, a) and not torch.equal(c, d)


@GPU
def test_model_refusals():
    r = _renderer()
    rx, tx, z, f = _rows()
    with pytest.raises(ValueError, match="modal-kernel inputs"):
        r(_model(modal_feat_dim=8), rx, tx, *BOX, z_s=z)
    with pytest.raises(ValueError, match="without the modal input map"):
        r(_model(), rx, tx, *BOX, z_s=z, modal_feat=f)


@GPU
def test_prior_is_added_exactly_to_its_bins_and_refusals():
    m = _model(modal_prior=True)
    r = _renderer()
    rx, tx, z, _f = _rows()
    p = torch.complex(torch.randn(3, 601), torch.randn(3, 601)).cuda() * 100
    with torch.no_grad():
        h0 = r(m, rx, tx, *BOX, z_s=z, h_prior=torch.zeros_like(p))
        h1 = r(m, rx, tx, *BOX, z_s=z, h_prior=p)
    assert torch.allclose(h1[:, :601] - h0[:, :601], p, rtol=0, atol=1e-3)
    assert torch.equal(h1[:, 601:], h0[:, 601:])
    with pytest.raises(ValueError, match="trained with a physics modal prior"):
        r(m, rx, tx, *BOX, z_s=z)
    with pytest.raises(ValueError, match="not trained with a modal prior"):
        r(_model(), rx, tx, *BOX, z_s=z, h_prior=p)


# ------------------------------------------------------------------------------------------- wrapper
def _fake_rooms(root):
    from aaf.eval.modal_prior import cond_key
    rng = np.random.default_rng(0)
    rooms = []
    for j in range(2):
        cond = rng.random(324).astype(np.float32)
        rx = rng.random((6, 2)) * 4 + 0.3
        H = (rng.standard_normal((6, 601)) + 1j * rng.standard_normal((6, 601))).astype(np.complex64)
        s = rng.standard_normal((6, 48)).astype(np.float32)
        kc = np.linspace(0.05, 0.9, 48).astype(np.float32)
        d = root / "set{}".format(j)
        d.mkdir(parents=True)
        np.savez(d / "room.npz", H=H, s=s, kc=kc, rx=rx, src=np.array([0.5, 1.6]), cond=cond,
                 cond_key=np.asarray(cond_key(cond)), meta=np.asarray("{}"))
        rooms.append((cond, rx, H, s, kc))
    return rooms


def test_wrapper_lookup_maps_rows_and_refuses_misses(tmp_path):
    from aaf.eval.modal_prior import ModalPriorRenderer, feat_rows
    rooms = _fake_rooms(tmp_path)
    w = ModalPriorRenderer(torch.nn.Identity(), root=tmp_path)

    class M:
        modal_feat_dim = 8
    order = [(1, 4), (0, 2), (1, 0), (0, 5)]
    z = torch.tensor(np.stack([rooms[j][0] for j, _ in order]))
    rx = torch.tensor(np.stack([rooms[j][1][i] for j, i in order]), dtype=torch.float32)
    tx = torch.tensor([[0.5, 1.6]] * len(order))
    h, f = w.lookup(M, z, rx, tx)
    for b, (j, i) in enumerate(order):
        assert np.array_equal(h[b].numpy(), rooms[j][2][i])
        assert np.array_equal(f[b].numpy(), feat_rows(rooms[j][3][i:i + 1], rooms[j][4], 4)[0])
    with pytest.raises(KeyError, match="not among the precomputed"):
        w.lookup(M, z[:1], rx[:1] + 0.01, tx[:1])
    with pytest.raises(KeyError, match="not among the precomputed"):
        w.lookup(M, z[:1], rx[:1], tx[:1] + 0.5)
    with pytest.raises(KeyError, match="no precomputed modal prior"):
        w.lookup(M, z[:1] + 0.1, rx[:1], tx[:1])


# ------------------------------------------------------------------------------------------- trainer + load_model
def _tiny_family(tmp_path, n=3):
    m = json.load(open("configs/sweeps_2d_mat/p4_6_family_holdU_manifest.json"))
    rows = [r for r in m["configs"] if r["split"] == "train"]
    rows = [r for r in rows if (FAM_PRIOR / (r["filename"] + ".npz")).exists()][:n]
    if len(rows) < n:
        pytest.skip("family modal priors not precomputed yet")
    out = tmp_path / "man.json"
    out.write_text(json.dumps(dict(m, configs=rows)))
    return str(out)


def _trainer(tmp_path, man, name, **kw):
    import yaml

    from aaf.train.multi_room_2d_mat import P32TrainCfg, P32Trainer
    base = yaml.safe_load(open("configs/sweep_2d_mat/P4_6_G2.yaml"))
    base = {k: v for k, v in base.items() if k in P32TrainCfg.__dataclass_fields__}
    base.update({"configs_manifest": man, "n_iters": 2, "val_max_configs": 2})
    base.update(kw)
    return P32Trainer(P32TrainCfg(**base), tmp_path / name)


@GPU
def test_trainer_render_is_prior_plus_the_unchanged_network_at_init(tmp_path):
    from aaf.eval.modal_prior import feat_rows, load_room
    man = _tiny_family(tmp_path)
    off = _trainer(tmp_path, man, "off")
    on = _trainer(tmp_path, man, "on", modal_prior_dir=str(FAM_PRIOR), modal_k=48)
    assert off.H_prior is None and on.H_prior.shape == on.H.shape and on.modal_feat.shape[-1] == 96
    for ci, c in enumerate(on.configs):
        d = load_room(FAM_PRIOR / (c.filename + ".npz"))
        assert np.array_equal(on.H_prior[ci].cpu().numpy(), d["H"][:, :on.n_band])
        assert np.array_equal(on.modal_feat[ci].cpu().numpy(), feat_rows(d["s"], d["kc"], 48))
    nb = on.cfg.dc_head_bins
    want = (on.H - on.H_prior)[:, on.train_rx_idx, :nb].mean(1).abs().pow(2).mean(0).sqrt().clamp(min=1e-6)
    assert torch.allclose(on.model.dc_scale, want.float())
    for t in (off, on):
        t.model.eval(); t.renderer.eval()
    ci = torch.tensor([0, 1, 2, 2], device=on.device); ri = torch.tensor([3, 50, 7, 700], device=on.device)
    with torch.no_grad():
        h_on = on._render(ci, ri)
        h_off = off._render(ci, ri)
    # the same float op the renderer does (net + prior): BIT-identical. (Subtracting the ~1e3 prior back
    # out in float32 would lose the network's ~1e-3 low bins to cancellation.)
    assert torch.equal(h_on, h_off + on.H_prior[ci, ri])
    assert not torch.equal(h_on, h_off)                                    # the prior is live


@GPU
def test_load_model_evaluator_path_reproduces_the_trainer_path(tmp_path):
    from aaf.eval.modal_prior import ModalPriorRenderer
    from aaf.eval.p3_2_eval import load_model
    man = _tiny_family(tmp_path)
    on = _trainer(tmp_path, man, "on", modal_prior_dir=str(FAM_PRIOR), modal_k=48)
    with torch.no_grad():
        torch.nn.init.normal_(on.model.modal_proj.weight, std=0.02)       # a non-trivial, trained-like map
    on.save_ckpt(2)
    (on.out / "train_meta.json").write_text(json.dumps({"n_configs": len(on.configs)}))
    m, r, cfg, _meta, _it = load_model(on.out / "ckpt_iter0000002.pt", on.device)
    assert isinstance(r, ModalPriorRenderer) and m.modal_feat_dim == 96 and bool(m.requires_modal_prior)
    on.model.eval(); on.renderer.eval()
    ci = torch.tensor([0, 0, 1, 2], device=on.device); ri = torch.tensor([3, 9, 50, 700], device=on.device)
    with torch.no_grad():
        want = on._render(ci, ri)
        rx, tx, z = on.rx[ci, ri], on.src.unsqueeze(0).expand(4, -1), on.cond[ci]
        got = torch.cat([r(m, rx[i:i + 1], tx[i:i + 1], torch.zeros(2, device=on.device),
                           torch.tensor(on.geom_dims[int(on.geom_id[ci[i]])], device=on.device), z_s=z[i:i + 1])
                         for i in range(4)])[:, :on.n_band]
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-4 * float(want.abs().max()))
        _h, f = r.lookup(m, z[:1], rx[:1], tx[:1])
        box = (torch.zeros(2, device=on.device), torch.tensor(on.geom_dims[int(on.geom_id[ci[0]])], device=on.device))
        with pytest.raises(ValueError, match="trained with a physics modal prior"):
            r.base(m, rx[:1], tx[:1], *box, z_s=z[:1], modal_feat=f)              # the bare renderer refuses
        with pytest.raises(ValueError, match="modal-kernel inputs"):
            r.base(m, rx[:1], tx[:1], *box, z_s=z[:1], h_prior=_h)                # the model refuses
