"""P4-10 P-1 (D217 / D218): the mean-value penalty fine-tune. Rule 8: nothing here adds a parameter, so a checkpoint cannot see a mistake.
  * the residual: ~0 for a plane wave (a Helmholtz solution), 1 - J0(kr) for a constant field (the single-radius form is NOT blind to a constant), and
    invariant to the field's scale (the normaliser is not detached: shrinking the field cannot lower it).
  * weight 0 with the penalty path running is the control: same batches, rows, generator states, same first-step loss; the parameters after one step differ from
    the control's only by the control's own run-to-run noise (tinycudann's hash-grid gradients are not bit-reproducible); weight > 0 is different.
  * draws: centres obey the wall / source / lattice rules, the streams of the data step are untouched by the calibration and by the penalty's own generator.
  * resume continues the penalty generator and the weight.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aaf.train.p410_pt import C0, DF, F_SPLIT, R_MAX, meanvalue_ratio, poly_dist, radius
from tests.test_p4_10_stream import _ms
from tests.test_p4_9_dir import _cfg

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")


@pytest.fixture(autouse=True)
def _free_gpu():
    """several trainers are built per test: give the memory back between tests (a 24 GB card otherwise runs out in the last one)"""
    yield
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stencil(field, centres, r, kdirs=None):
    th = np.arange(8) * 2 * np.pi / 8
    out = []
    for x in centres:
        out.append(np.concatenate([x[None], x + r * np.stack([np.cos(th), np.sin(th)], 1)]))
    return np.stack([field(q) for q in out])                # [K, 9, B]


def test_plane_wave_is_zero_and_constant_is_not():
    rng = np.random.default_rng(0)
    f = np.array([40.0, 63.0, 100.0, 150.0, 187.0])
    k = 2 * np.pi * f / C0
    d = rng.standard_normal((1, 2))
    d /= np.linalg.norm(d)
    ctr = rng.uniform(0, 5, (6, 2))
    r = R_MAX
    H = _stencil(lambda q: np.exp(-1j * (q @ d[0])[:, None] * k[None]), ctr, r)
    Hc, Hm = torch.tensor(H[:, 0]), torch.tensor(H[:, 1:])
    kr = torch.tensor(k * r)
    v = meanvalue_ratio(Hc, Hm, kr)
    assert (v < 1e-3).all(), v
    Hk = np.stack([np.ones((9, len(f))) * 2.5 for _ in ctr])                 # a constant
    vc = meanvalue_ratio(torch.tensor(Hk[:, 0] + 0j), torch.tensor(Hk[:, 1:] + 0j), kr)
    from scipy.special import j0
    assert np.allclose(vc.numpy(), np.abs(1 - j0(k * r)), atol=1e-6)
    assert float(vc[1]) == pytest.approx(0.0404, abs=0.003)                   # the manager's "about 4%" at 63 Hz, r = 0.35 m
    rnd = torch.randn(6, 8, 5, dtype=torch.complex128)                       # noise is not a solution: large
    assert (meanvalue_ratio(torch.randn(6, 5, dtype=torch.complex128), rnd, kr) > 0.2).all()


def test_scale_invariant_not_detached():
    rng = np.random.default_rng(1)
    Hc = torch.tensor(rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3)), requires_grad=True)
    Hm = torch.tensor(rng.standard_normal((4, 8, 3)) + 1j * rng.standard_normal((4, 8, 3)))
    kr = torch.tensor([0.5, 0.8, 1.1], dtype=torch.float64)
    a = meanvalue_ratio(Hc, Hm, kr)
    assert torch.allclose(meanvalue_ratio(3.0 * Hc, 3.0 * Hm, kr), a, rtol=1e-9)     # shrinking the field cannot lower the penalty
    s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    meanvalue_ratio(s * Hc.detach(), s * Hm, kr).sum().backward()
    assert abs(float(s.grad)) < 1e-8                                                  # no gradient along the scale
    # detached denominator (the rejected A13 form) would have one
    t = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    R = (t * Hm).mean(1) - torch.as_tensor(__import__("scipy.special", fromlist=["j0"]).j0(kr.numpy())) * t * Hc.detach()
    (torch.sqrt((R.abs() ** 2).mean(0)) / torch.sqrt((t.detach() * Hc.detach()).abs().pow(2).mean(0))).sum().backward()
    assert abs(float(t.grad)) > 1e-2


def _trainer(tmp_path, name, d, pt=None, **kw):
    from aaf.train.p410_pt import PTTrainer
    cfg = _cfg(tmp_path, p49_mode="ctl", p410_ms_dir=str(d), p410_row_jitter=True, p49_recip_p=0.25, **kw)
    return PTTrainer(cfg, tmp_path / name, pt)


def _base(tmp_path):
    from aaf.train.multi_room_2d_mat import P32Trainer
    return _ms(tmp_path, P32Trainer(_cfg(tmp_path, p49_mode="ctl"), tmp_path / "base"))


def _params(tr):
    return torch.cat([p.detach().reshape(-1).cpu() for p in tr.model.parameters()])


@GPU
def test_weight_zero_is_the_control(tmp_path):
    d = _base(tmp_path)
    ctl, ctl2 = _trainer(tmp_path, "c", d, {"frac": 0.0}), _trainer(tmp_path, "c2", d, {"frac": 0.0})
    w0 = _trainer(tmp_path, "w0", d, {"frac": 0.2, "force_w": 0.0, "k": 2})
    assert torch.equal(_params(ctl), _params(w0))
    m, m2, m0 = ctl._step_fn(1), ctl2._step_fn(1), w0._step_fn(1)
    assert m["loss"] == m2["loss"] == m0["loss"], (m["loss"], m2["loss"], m0["loss"])   # the first step's forward is deterministic
    assert m0["L_pt"] > 0 and "L_pt" not in m                          # the penalty path really ran, with weight 0
    for k in ctl._p410_gens:
        assert torch.equal(ctl._p410_gens[k].get_state(), w0._p410_gens[k].get_state()), k
    assert torch.equal(ctl._p49_gen.get_state(), w0._p49_gen.get_state()) and torch.equal(ctl._p49_swap_gen.get_state(), w0._p49_swap_gen.get_state())
    noise = float((_params(ctl) - _params(ctl2)).abs().max())                              # the control's own run-to-run difference
    assert float((_params(ctl) - _params(w0)).abs().max()) <= max(2.5 * noise, 1e-6)


@GPU
def test_weight_positive_moves_and_calibration_is_clean(tmp_path):
    d = _base(tmp_path)
    a = _trainer(tmp_path, "a", d, {"frac": 0.2, "k": 2})
    assert a._pt_w > 0 and math.isfinite(a._pt_w)
    b = _trainer(tmp_path, "b", d, {"frac": 0.0})
    for k in a._p410_gens:                                                                 # the calibration restored every stream
        assert torch.equal(a._p410_gens[k].get_state(), b._p410_gens[k].get_state()), k
    assert torch.equal(a._p49_gen.get_state(), b._p49_gen.get_state())
    m, m_c = a._step_fn(1), b._step_fn(1)
    assert m["loss"] == m_c["loss"] and m["L_pt"] > 0                                      # same data step; the penalty is extra
    assert float((_params(a) - _params(b)).abs().max()) > 1e-5
    # the penalty moves parameters through its own gradient
    c = _trainer(tmp_path, "g", d, {"frac": 0.2, "k": 2})
    c.opt.zero_grad(set_to_none=True)
    pen, _ = c._pt_penalty(c._pt_draw(c._pt_gen, "lo"))
    pen.backward()
    g = [float(p.grad.abs().sum()) for p in c.model.parameters() if p.grad is not None]
    assert math.isfinite(sum(g)) and sum(g) > 0


@GPU
def test_draw_rules(tmp_path):
    d = _base(tmp_path)
    a = _trainer(tmp_path, "d", d, {"frac": 0.2, "k": 3})
    for i in range(40):
        kind = "lo" if i % 2 else "hi"
        x = a._pt_draw(a._pt_gen, kind)
        assert x is not None and x["kind"] == kind and len(x["ctr"]) == 3
        room, src, ctr, r = a.configs[x["c"]], x["src"], x["ctr"], x["r"]
        assert F_SPLIT <= x["f0_bin"] <= 600
        assert (poly_dist(ctr, room.verts) >= r + 0.05 - 1e-9).all()
        assert (np.hypot(*(ctr - src).T) >= r + 0.3 - 1e-9).all()
        kk = (ctr - 0.12) / 0.06
        assert (np.hypot(*((kk - np.round(kk)) * 0.06).T) >= 0.01 - 1e-9).all()              # off the receiver lattice
        assert r == R_MAX if kind == "lo" else abs(r - radius(x["f0_bin"] * DF)) < 1e-12
        assert len(x["bins"]) == (2 if kind == "lo" else 1) and min(min(b) for b in x["bins"]) >= 50
        if kind == "hi":
            for b in x["bins"][0]:
                assert abs(b - x["f0_bin"]) <= 0.06 * x["f0_bin"]


@GPU
def test_resume_continues_the_penalty_state(tmp_path):
    d = _base(tmp_path)
    a = _trainer(tmp_path, "r", d, {"frac": 0.2, "k": 2})
    a._step_fn(1)
    a.save_ckpt(1)
    nxt, w_a = a._pt_draw(a._pt_gen, "lo"), a._pt_w
    del a
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    again = _trainer(tmp_path, "r", d, {"frac": 0.2, "k": 2})
    assert again.start_iter == 1 and again._pt_w == w_a
    y = again._pt_draw(again._pt_gen, "lo")
    assert y["c"] == nxt["c"] and np.array_equal(y["bins"][0], nxt["bins"][0]) and np.allclose(y["ctr"][0], nxt["ctr"][0])
    with pytest.raises(RuntimeError):                                                        # other penalty settings must not resume silently
        _trainer(tmp_path, "r", d, {"frac": 0.2, "k": 3})


@GPU
def test_warm_start_continues_the_generators(tmp_path):
    """R1's generator states are continued, not re-seeded: the first batch after the warm start is the one R1 would have drawn next"""
    d = _base(tmp_path)
    a = _trainer(tmp_path, "w", d, {"frac": 0.0})
    for _ in range(3):
        a._step_fn(1)
    a.save_ckpt(3)
    nxt = [t.clone() for t in a._p49_batch()[0]]
    b = _trainer(tmp_path, "w2", d, {"frac": 0.0}, init_ckpt=str(tmp_path / "w" / "ckpt_iter0000003.pt"), lr_restart_peak=5e-5, lr_warmup_iters=0,
                 lr_restart_from_iter=3, n_iters=6, early_stop_warmup=6)
    assert b.start_iter == 3 and abs(b.sched.get_last_lr()[0] - 5e-5) < 1e-12
    got = b._p49_batch()[0]
    assert all(torch.equal(x, y) for x, y in zip(nxt, got))
