"""P4-5 behavioural tests (CLAUDE.md rule 8) for the four high-frequency mechanisms.

Each mechanism changes behaviour WITHOUT changing the parameter set -- a loss weighting, a loss
band, a mask buffer, a checkpoint rewrite -- which is exactly the class of change a checkpoint
round-trip cannot catch (D73). So every test asserts on OUTPUTS or GRADIENTS, and each guard is
paired with a case where it must FIRE, because a guard nobody has seen fail is not known to work
(D91).

All tests need a GPU: importing the trainer imports tinycudann.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
N_BAND = 601


def _pre_change_losses(H_pred, H_target):
    """An independent copy of `_losses` exactly as it stood before P4-5."""
    eps = 1e-6
    return {
        "L_spec_real": F.l1_loss(H_pred.real, H_target.real),
        "L_spec_imag": F.l1_loss(H_pred.imag, H_target.imag),
        "L_amp": F.l1_loss(torch.log10(H_pred.abs() + eps), torch.log10(H_target.abs() + eps)),
        "L_phase": (1.0 - torch.cos(H_pred.angle() - H_target.angle())).mean(),
    }


def _pair(seed=0, rows=8):
    g = torch.Generator().manual_seed(seed)
    P = torch.complex(torch.randn(rows, N_BAND, generator=g),
                      torch.randn(rows, N_BAND, generator=g)).requires_grad_(True)
    T = torch.complex(torch.randn(rows, N_BAND, generator=g),
                      torch.randn(rows, N_BAND, generator=g))
    return P, T


# --------------------------------------------------------------------------- B1 / B1b: weights
@GPU
def test_default_loss_is_bit_identical_to_the_pre_change_function():
    from aaf.train.multi_room_2d_mat import _losses
    P, T = _pair()
    new, old = _losses(P, T), _pre_change_losses(P, T)
    for k in old:
        assert torch.equal(new[k], old[k]), k


@GPU
def test_spec_weights_scale_the_output_gradient_of_each_bin_exactly():
    """dL/dRe(P)[:, b] must be w_b times its unweighted value -- and amp/phase untouched."""
    from aaf.train.multi_room_2d_mat import _losses
    w = torch.linspace(0.2, 5.0, N_BAND)
    P, T = _pair(1)
    g_w = torch.autograd.grad(_losses(P, T, w_spec=w)["L_spec_real"], P)[0].real
    g_0 = torch.autograd.grad(_losses(P, T)["L_spec_real"], P)[0].real
    assert torch.allclose(g_w, g_0 * w, rtol=1e-6, atol=1e-12)
    for k in ("L_amp", "L_phase"):
        assert torch.allclose(_losses(P, T, w_spec=w)[k], _losses(P, T)[k], rtol=1e-6)


@GPU
def test_unit_weights_reproduce_the_unweighted_loss():
    from aaf.train.multi_room_2d_mat import _losses
    P, T = _pair(2)
    a, b = _losses(P, T, w_spec=torch.ones(N_BAND)), _losses(P, T)
    for k in a:
        assert torch.allclose(a[k], b[k], rtol=1e-6), k


@GPU
def test_spec_weights_actually_change_the_loss():
    """The firing case: a non-uniform w MUST move the spec terms."""
    from aaf.train.multi_room_2d_mat import _losses
    P, T = _pair(3)
    w = torch.linspace(0.2, 5.0, N_BAND)
    assert not torch.allclose(_losses(P, T, w_spec=w)["L_spec_real"],
                              _losses(P, T)["L_spec_real"])


# --------------------------------------------------------------------------- B3: curriculum
@GPU
def test_curriculum_gives_exactly_zero_gradient_above_the_edge():
    from aaf.train.multi_room_2d_mat import _losses
    P, T = _pair(4)
    hi = 201
    terms = _losses(P, T, hi_bin=hi)
    g = torch.autograd.grad(sum(terms.values()), P)[0]
    assert torch.count_nonzero(g[:, hi:]) == 0
    assert torch.count_nonzero(g[:, :hi]) > 0                     # and it does fire below


@GPU
def test_curriculum_keeps_each_active_bin_at_its_baseline_weight():
    """The denominator stays the FULL band: an active bin's gradient must EQUAL the baseline's.
    Plain slicing would scale it by n/hi_bin (~3x here) and this test would fail."""
    from aaf.train.multi_room_2d_mat import _losses
    P, T = _pair(5)
    hi = 201
    for k in ("L_spec_real", "L_spec_imag", "L_amp", "L_phase"):
        g_c = torch.autograd.grad(_losses(P, T, hi_bin=hi)[k], P)[0][:, :hi]
        g_0 = torch.autograd.grad(_losses(P, T)[k], P)[0][:, :hi]
        assert torch.allclose(g_c, g_0, rtol=1e-5, atol=1e-12), k


@GPU
def test_curriculum_and_c2f_schedules():
    from aaf.train.multi_room_2d_mat import P32Trainer
    me = types.SimpleNamespace(cfg=types.SimpleNamespace(fs=4096, n_time_samples=8192,
                                                         n_levels=20), n_band=N_BAND)
    me.curriculum = P32Trainer._parse_curriculum(me, "0:100,15000:200,35000:300")
    assert me.curriculum == [(0, 201), (15000, 401), (35000, None)]
    me.c2f = (6, 2800)
    st = lambda it: P32Trainer._stage(me, it)                     # noqa: E731
    assert st(0) == (201, 6) and st(14999) == (201, 11) and st(15000) == (401, 11)
    assert st(35000) == (None, 18) and st(39200) == (None, 20) and st(60000) == (None, 20)
    with pytest.raises(ValueError):                               # must start at iteration 0
        P32Trainer._parse_curriculum(me, "5000:100,20000:300")


# --------------------------------------------------------------------------- B2: level mask
def _model(seed=0):
    """FiLM woken: at init it is identity, so the output would not depend on the conditioning
    at all and a mask wired to the wrong columns could pass unnoticed."""
    from aaf.models.inr_2d import INR2D_AutoDecoder
    torch.manual_seed(seed)
    hg = dict(otype="HashGrid", n_levels=20, n_features_per_level=2, log2_hashmap_size=18,
              base_resolution=80, per_level_scale=1.5)
    m = INR2D_AutoDecoder(n_rooms=4, latent_dim=16, n_freq_bins=N_BAND, hash_grid_config=hg,
                          cond_source="geom_token", cond_dim=324, conditioning_type="film",
                          token_pool="attn_residual_extent", world_scale=10.0).cuda().eval()
    g = torch.Generator(device="cuda").manual_seed(1234)
    with torch.no_grad():
        for lin in (m.film_sigma, m.film_signal):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g, device="cuda") * 0.05)
    return m


def _fwd(m, n=257, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    P = torch.rand(1, n, 2, generator=g, device="cuda") * 6.0
    V = torch.zeros(1, n, 2, device="cuda"); V[..., 0] = 1.0
    TX = torch.tensor([0.5, 1.6], device="cuda").view(1, 1, 2).expand(1, n, 2)
    z = torch.randn(1, 324, generator=g, device="cuda") * 0.1
    return m(P, V, TX, tx_view=None, z_s=z)


@GPU
def test_level_mask_defaults_to_all_ones_and_is_a_saved_buffer():
    m = _model()
    assert torch.equal(m.level_mask, torch.ones(20, device="cuda"))
    assert "level_mask" in m.state_dict()


@GPU
def test_masking_a_level_changes_the_output_and_unmasking_restores_it_exactly():
    m = _model()
    # 14 levels off, not 1: the finest level's hash features start near +-1e-4, so masking it
    # alone could move the output by less than float tolerance and make the firing case flaky.
    with torch.no_grad():
        a = [t.clone() for t in _fwd(m)]
        m.set_active_levels(6)
        b = [t.clone() for t in _fwd(m)]
        m.set_active_levels(20)
        c = [t.clone() for t in _fwd(m)]
    assert not torch.allclose(a[0], b[0]) or not torch.allclose(a[1], b[1]), \
        "switching off 14 levels changed nothing -- the mask is not wired in"
    assert all(torch.equal(x, y) for x, y in zip(a, c)), "unmasking is not exact"


@GPU
def test_masked_levels_receive_exactly_zero_gradient():
    """Captured at the encoder OUTPUT: hash params only get gradient through these columns."""
    m = _model().train()
    k = 6
    m.set_active_levels(k)
    caught = {}

    def hook(_mod, _inp, out):
        out.retain_grad()
        caught["e"] = out

    h = m._pos_encoding.register_forward_hook(hook)
    attn, sig = _fwd(m)
    (attn.real.sum() + sig.real.sum()).backward()
    h.remove()
    gr = caught["e"].grad
    assert torch.count_nonzero(gr[:, 2 * k:]) == 0, "masked levels received gradient"
    assert torch.count_nonzero(gr[:, :2 * k]) > 0, "active levels received none"


@GPU
def test_a_partially_masked_checkpoint_reloads_with_its_mask(tmp_path):
    """The D73 hazard: a 30K B2 checkpoint must come back with ITS levels, not all 20."""
    from aaf.eval.p3_2_eval import load_model
    ref = Path("outputs/p4_4/p4_4_C250/ckpt_iter0060000.pt")
    m, r, cfg, _meta, _it = load_model(ref, torch.device("cuda"))
    m.set_active_levels(7)
    with torch.no_grad():
        want = [t.clone() for t in _fwd(m)]
    torch.save({"iter": 30000, "model": m.state_dict(), "cfg": cfg}, tmp_path / "ck.pt")
    m2, *_ = load_model(tmp_path / "ck.pt", torch.device("cuda"))
    assert int(m2.level_mask.sum()) == 7
    with torch.no_grad():
        got = _fwd(m2)
    assert all(torch.equal(x, y) for x, y in zip(want, got))


@GPU
def test_pre_p4_5_checkpoint_loads_and_renders_as_before():
    """`p4_3_D_data` predates the buffer. It must (a) load at all -- a strict load would refuse
    a missing key -- with an all-ones mask, and (b) re-render receivers whose predictions the
    PRE-CHANGE code cached in the P4-4 demo v2 render, to within float noise across GPUs."""
    from aaf.eval.p3_2_eval import load_model
    from scripts.p4_3_demo_pack import render
    from scripts.p4_4_sweeps import validate
    cache = Path("outputs/p4_4/demo/v2/render_s1_notch_width_p4_3_D_data.npz")
    if not cache.exists():
        pytest.skip("pre-change render cache not present")
    m, r, *_ = load_model(Path("outputs/p4_3/p4_3_D_data/ckpt_iter0060000.pt"),
                          torch.device("cuda"))
    assert torch.equal(m.level_mask, torch.ones_like(m.level_mask))
    z = np.load(cache)
    rx, P_old = z["rx0"][:16], z["P0"][:16]
    cfg = validate("s1_notch_width")[0][0][1]
    P_new = render(m, r, cfg, rx, P_old.shape[1], torch.device("cuda"))
    rel = float(np.abs(P_new - P_old).max() / np.abs(P_old).max())
    print("max rel diff vs pre-change render: {:.3e}".format(rel))
    assert rel < 1e-4, rel


# --------------------------------------------------------------------------- checkpoint thinning
def _fake_ckpt(path, it):
    torch.save({"iter": it, "model": {"w": torch.arange(4.0)}, "cfg": {"a": 1},
                "optimizer": {"state": torch.zeros(1000)}, "scheduler": {"s": 1}}, path)


@GPU
def test_thinning_strips_only_older_checkpoints_and_resume_skips_them(tmp_path):
    from aaf.train.multi_room_2d_mat import P32Trainer
    a, b = tmp_path / "ckpt_iter0002000.pt", tmp_path / "ckpt_iter0004000.pt"
    _fake_ckpt(a, 2000)
    _fake_ckpt(b, 4000)
    me = types.SimpleNamespace(out=tmp_path)
    P32Trainer._thin_older_than(me, b)
    sa, sb = torch.load(a), torch.load(b)
    assert sa["thinned"] and "optimizer" not in sa and "scheduler" not in sa
    assert torch.equal(sa["model"]["w"], torch.arange(4.0)) and sa["iter"] == 2000
    assert "optimizer" in sb and not sb.get("thinned")
    # resume must never pick a thinned file: with only thinned files present it starts fresh
    b.unlink()
    me = types.SimpleNamespace(out=tmp_path, device="cpu")
    assert P32Trainer._maybe_resume(me) == 0
