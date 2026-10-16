"""P4-7 prep p4 (D164): the trainer's augmentation wiring, tested on BEHAVIOUR (Rule 8).

* Flags OFF: `_step` must be the pre-change `_step` -- same losses, same parameter updates, same CPU and
  CUDA RNG states after three steps -- checked against a FROZEN VERBATIM COPY of the pre-change function
  on a CPU stub (teeth: one extra global draw is detected). Nothing is imported, drawn or allocated.
* Flags ON: the draws come from a private generator (global RNG untouched, deterministic per seed);
  identity draws reproduce `_render` bit-for-bit; swapped rows exchange rx / tx with an UNCHANGED target;
  every transformed row lies inside its transformed room box (teeth: untransposed boxes fail);
  unsupported set-ups are refused; the resume guard covers the new keys.
* Integration (real trainer, family corpus): an OFF and an ON trainer hold identical default tensors,
  and the tables' identity slice equals them exactly.
All tests import the trainer (tinycudann), so they need a GPU node.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
N_BAND = 601


# ----------------------------------------------------------------------- frozen pre-change _step
def _frozen_step(self, it):
    """VERBATIM copy of P32Trainer._step before P4-7 (main @ c6ceded); sigma_aux block omitted (weight 0)."""
    from aaf.train.multi_room_2d_mat import _losses
    cfg = self.cfg
    self.model.train()
    self.opt.zero_grad(set_to_none=True)
    n_cfg = max(1, cfg.batch_size // cfg.rx_per_config)
    agg = {}
    hi_bin, levels = self._stage(it)
    if levels is not None:
        self.model.set_active_levels(levels)
        agg["c2f_active_levels"] = float(levels)
    if hi_bin is not None:
        agg["curriculum_hi_bin"] = float(hi_bin)
    for _ in range(cfg.grad_accum_steps):
        ci = torch.randint(0, len(self.configs), (n_cfg,), device=self.device)
        ci = ci.repeat_interleave(cfg.rx_per_config)
        ri = self.train_rx_idx[
            torch.randint(0, self.train_rx_idx.numel(), (ci.numel(),), device=self.device)]
        H_pred = self._render(ci, ri)
        H_tgt = self.H[ci, ri]
        terms = _losses(H_pred, H_tgt, self.loss_lo_bin, w_spec=self.spec_w, hi_bin=hi_bin,
                        amp_eps=self.amp_eps)
        w_r, w_i, w_a, w_p = cfg.weights
        loss = (w_r * terms["L_spec_real"] + w_i * terms["L_spec_imag"]
                + w_a * terms["L_amp"] + w_p * terms["L_phase"])
        (loss / cfg.grad_accum_steps).backward()
        for k, v in terms.items():
            agg[k] = agg.get(k, 0.0) + float(v) / cfg.grad_accum_steps
        agg["loss"] = agg.get("loss", 0.0) + float(loss) / cfg.grad_accum_steps
    for p in self.model.parameters():
        if p.grad is not None:
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_max_norm)
    self.opt.step()
    self.sched.step()
    return agg


class _TinyField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2 * N_BAND)

    def forward(self, rx, tx):
        o = self.lin(torch.cat([rx, tx], -1))
        return torch.complex(o[:, :N_BAND], o[:, N_BAND:])


def _cpu_stub(seed=0, extra_draw=False):
    from aaf.train.multi_room_2d_mat import P32TrainCfg, P32Trainer
    torch.manual_seed(seed)
    t = P32Trainer.__new__(P32Trainer)
    t.cfg = P32TrainCfg(batch_size=4, rx_per_config=2, grad_accum_steps=2)
    t.device = torch.device("cpu")
    t.model = _TinyField()
    t.opt = torch.optim.Adam(t.model.parameters(), lr=1e-3)
    t.sched = torch.optim.lr_scheduler.CosineAnnealingLR(t.opt, T_max=10)
    t.configs = list(range(3))
    t.train_rx_idx = torch.arange(10)
    g = torch.Generator().manual_seed(7)
    t.H = torch.complex(torch.randn(3, 10, N_BAND, generator=g), torch.randn(3, 10, N_BAND, generator=g))
    t.rx = torch.rand(3, 10, 2, generator=g) * 4
    t.src = torch.tensor([0.5, 1.6])
    t.loss_lo_bin, t.spec_w, t.amp_eps = 0, None, None
    t._stage = lambda it: (None, None)

    def _render(ci, ri):
        if extra_draw:
            torch.rand(1)
        return t.model(t.rx[ci, ri], t.src.unsqueeze(0).expand(ci.numel(), -1))
    t._render = _render
    return t


def _run(step_fn, t, n=3):
    aggs = [step_fn(t, it) for it in range(1, n + 1)]
    return aggs, [p.detach().clone() for p in t.model.parameters()], torch.get_rng_state()


@GPU
def test_defaults_are_off():
    from aaf.train.multi_room_2d_mat import P32TrainCfg, _aug_requested
    c = P32TrainCfg()
    assert c.aug_d4_p == 0.0 and c.aug_recip_p == 0.0 and not _aug_requested(c)


@GPU
def test_off_step_equals_the_frozen_pre_change_step():
    from aaf.train.multi_room_2d_mat import P32Trainer
    a = _cpu_stub()
    assert getattr(a, "_aug", None) is None
    cuda0 = torch.cuda.get_rng_state()
    ra = _run(lambda t, it: P32Trainer._step(t, it), a)
    b = _cpu_stub()
    rb = _run(_frozen_step, b)
    assert ra[0] == rb[0]
    assert all(torch.equal(x, y) for x, y in zip(ra[1], rb[1]))
    assert torch.equal(ra[2], rb[2]) and torch.equal(torch.cuda.get_rng_state(), cuda0)
    c = _cpu_stub(extra_draw=True)                                    # teeth
    rc = _run(_frozen_step, c)
    assert not torch.equal(rc[2], rb[2])


# --------------------------------------------------------------------- on-path plumbing (stub renderer)
class _BoxRenderer(torch.nn.Module):
    """Deterministic stand-in: depends on |rx - tx|, the room box and z; asserts rows lie in the box."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, model, rx, tx, room_min, room_max, z_s=None):
        assert (rx >= room_min - 1e-5).all() and (rx <= room_max + 1e-5).all(), "row outside its room box"
        self.calls.append((rx.clone(), tx.clone(), room_max.clone()))
        d = (rx - tx).norm(dim=-1, keepdim=True)
        f = torch.arange(N_BAND, dtype=torch.float32, device=rx.device)[None]
        amp = 1.0 + d * 0.01 * f + 0.001 * z_s.sum(-1, keepdim=True) + room_max.prod()
        return amp.to(torch.complex64)


def _real_stub(elems="fx,fy,r180", p_d4=0.0, p_rec=0.0, n_cfg=4, sigma=0.0, cond_source="geom_token"):
    from aaf.data.multi_notch import configs_from_rows
    from aaf.models.conditioning_2d import build_cond_vector_2d
    from aaf.train.multi_room_2d_mat import P32TrainCfg, P32Trainer
    man = "configs/sweeps_2d_mat/p4_6_family_holdU_manifest.json"
    cs = configs_from_rows(json.load(open(man))["configs"], split="train")[:n_cfg]
    t = P32Trainer.__new__(P32Trainer)
    t.cfg = P32TrainCfg(cond_source=cond_source, cond_dim=324, data_dir="data/track_p4_4_family",
                        configs_manifest=man, aug_d4_p=p_d4, aug_d4_elems=elems, aug_recip_p=p_rec,
                        aug_seed=3, sigma_aux_weight=sigma, batch_size=8, rx_per_config=2, grad_accum_steps=1)
    t.device = torch.device("cuda")
    t.configs = cs
    t.start_iter = 0
    rx, H = [], []
    import h5py
    for c in cs:
        with h5py.File(Path(t.cfg.data_dir) / c.filename) as f:
            rx.append(np.asarray(json.loads(f.attrs["receiver_pos"]), dtype=np.float32))
            H.append(np.asarray(f["ism/H_complex"])[:, :N_BAND])
    t.rx = torch.tensor(np.stack(rx), device=t.device)
    t.H = torch.tensor(np.stack(H), device=t.device)
    t.src = torch.tensor([0.5, 1.6], device=t.device)
    t.cond = torch.stack([build_cond_vector_2d("geom_token", c.L, c.W, c.edge_alphas, verts=c.verts).float()
                          for c in cs]).to(t.device)
    dims = {}
    t.geom_id = torch.tensor([dims.setdefault((c.L, c.W), len(dims)) for c in cs], device=t.device)
    t.geom_dims = list(dims)
    t.band, t.n_band = (0, N_BAND), N_BAND
    t.model = None
    t.renderer = _BoxRenderer()
    return t


@GPU
def test_identity_draws_reproduce_render_and_tables_match_preload():
    t = _real_stub(p_d4=0.5)
    t._aug = t._init_augment()
    A = t._aug
    assert torch.equal(A["rx"][:, 0], t.rx) and torch.equal(A["cond"][:, 0], t.cond)
    ci = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=t.device)
    ri = torch.tensor([5, 9, 11, 40, 3, 77, 12, 13], device=t.device)
    ref = t._render(ci, ri)
    g0 = torch.zeros_like(ci)
    got = t._render_rows(A["rx"][ci, g0, ri], A["src"][ci, g0], A["cond"][ci, g0], A["dims"][ci, g0])
    assert torch.equal(ref, got)


@GPU
def test_aug_draw_uses_a_private_generator():
    t = _real_stub(p_d4=0.5, p_rec=0.5)
    t._aug = t._init_augment()
    s_cpu, s_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    g1, w1 = t._aug_draw(4, 8)
    assert torch.equal(torch.get_rng_state(), s_cpu) and torch.equal(torch.cuda.get_rng_state(), s_cuda)
    u = _real_stub(p_d4=0.5, p_rec=0.5)
    u._aug = u._init_augment()
    g2, w2 = u._aug_draw(4, 8)
    assert torch.equal(g1, g2) and torch.equal(w1, w2)


@GPU
def test_swap_exchanges_rx_and_tx_and_keeps_the_target():
    t = _real_stub(p_rec=1.0)
    t._aug = t._init_augment()
    ci = torch.tensor([0, 0, 1, 1], device=t.device)
    ri = torch.tensor([5, 9, 11, 40], device=t.device)
    H_pred, H_tgt = t._aug_forward(ci, ri, {})
    assert torch.equal(H_tgt, t.H[ci, ri])
    rx_seen = torch.cat([c[0] for c in t.renderer.calls]); tx_seen = torch.cat([c[1] for c in t.renderer.calls])
    assert torch.allclose(rx_seen, t.src.expand_as(rx_seen))
    assert set(map(tuple, tx_seen.tolist())) == set(map(tuple, t.rx[ci, ri].tolist()))


@GPU
@pytest.mark.parametrize("elems", ["fx,fy,r180"])
def test_transformed_rows_lie_in_their_boxes_and_field_is_invariant(elems):
    t = _real_stub(elems=elems, p_d4=1.0)
    t._aug = t._init_augment()
    ci = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=t.device)
    ri = torch.tensor([5, 9, 11, 40, 3, 77, 12, 13], device=t.device)
    G = len(t._aug["elems"])
    for g in range(G):
        gg = torch.full_like(ci, g)
        A = t._aug
        # identity cond on purpose: the stub field depends on z, and this test checks boxes + |rx - tx|
        H = t._render_rows(A["rx"][ci, gg, ri], A["src"][ci, gg], A["cond"][ci, torch.zeros_like(ci)],
                           A["dims"][ci, gg])
        if g == 0:
            H0 = H
        assert torch.allclose(H, H0, rtol=1e-5, atol=1e-5)               # |rx - tx| and box area are D4-invariant


@GPU
def test_untransposed_boxes_fail_for_transposing_elements():
    t = _real_stub(elems="r90", p_d4=1.0)
    t.cfg.configs_manifest = ""                                       # skip the slab guard for this check
    t._aug = t._init_augment()
    A = t._aug
    ci = torch.tensor([0, 1], device=t.device)
    # receivers with x > W: after r90 their y' = x exceeds the UNtransposed box's height W
    ri = torch.stack([torch.argmax(t.rx[int(c), :, 0]) for c in ci])
    assert all(float(t.rx[int(c), int(r), 0]) > t.configs[int(c)].W for c, r in zip(ci, ri))
    gg = torch.ones_like(ci)
    wrong_dims = A["dims"][ci, torch.zeros_like(ci)]                  # the ORIGINAL (L, W)
    with pytest.raises(AssertionError):
        t._render_rows(A["rx"][ci, gg, ri], A["src"][ci, gg], A["cond"][ci, gg], wrong_dims)


@GPU
def test_refusals():
    with pytest.raises(ValueError, match="sigma_aux"):
        _real_stub(p_d4=0.5, sigma=0.1)._init_augment()
    with pytest.raises(ValueError, match="held-out d_hat slab"):
        _real_stub(elems="r90,tr", p_d4=0.5, n_cfg=60)._init_augment()
    with pytest.raises(ValueError, match="geometry-token"):
        _real_stub(p_d4=0.5, cond_source="aperture")._init_augment()
    t = _real_stub(p_rec=0.5)
    t.rx = t.rx.clone(); t.rx[0, 0] = torch.tensor([0.0, 1.0], device=t.device)    # a wall node
    with pytest.raises(ValueError, match="wall nodes"):
        t._init_augment()


@GPU
def test_resume_guard_covers_the_aug_keys():
    import inspect

    from aaf.train import multi_room_2d_mat as M
    src = inspect.getsource(M)
    guard = src[src.index("def _maybe_resume"):src.index("def _early_stop")]
    for k in ("aug_d4_p", "aug_d4_elems", "aug_recip_p"):
        assert '"%s"' % k in guard and k in M.P32TrainCfg.__dataclass_fields__, k


@GPU
@pytest.mark.skipif(not Path("data/track_p4_4_family").exists(), reason="family corpus not on disk")
def test_integration_off_and_on_trainers_share_their_default_tensors(tmp_path):
    import yaml

    from aaf.train.multi_room_2d_mat import P32TrainCfg, P32Trainer
    base = yaml.safe_load(open("configs/sweep_2d_mat/P4_6_G2.yaml"))
    keep = {k: v for k, v in base.items() if k in P32TrainCfg.__dataclass_fields__}
    keep.update({"n_iters": 2, "val_max_configs": 2})
    off = P32Trainer(P32TrainCfg(**keep), tmp_path / "off")
    on = P32Trainer(P32TrainCfg(**{**keep, "aug_d4_p": 0.5, "aug_recip_p": 0.25}), tmp_path / "on")
    assert off._aug is None and on._aug is not None
    for k in ("H", "rx", "cond", "src", "geom_id", "train_rx_idx"):
        assert torch.equal(getattr(off, k), getattr(on, k)), k
    assert torch.equal(on._aug["rx"][:, 0], on.rx) and torch.equal(on._aug["cond"][:, 0], on.cond)
