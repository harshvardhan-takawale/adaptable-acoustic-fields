"""P4-9 (D181 c4, A5; D180 0c): the 601-bin inference path, the per-ray output and the per-token override.

Run on the TRAINED R0 @120K checkpoint, so FiLM and the attention o_proj are awake: on a freshly built model FiLM and
o_proj are zero-initialised, the field does not depend on the token embeddings at all, and an override test would pass
on a completely broken implementation (Rule 8's vacuous version).

Behaviours:
  * n_out=601 returns the full path's bins 0-600 (renders AND gradients), including bin 600's imaginary signal part,
    which the full spectrum's Nyquist zeroing must not touch on the shorter buffer.
  * return_per_ray: the per-ray signals sum (plus the omni DC term) to H; the directions are the renderer's own.
  * tok_override: the encoder's own embeddings reproduce the render; perturbed embeddings change it.
  * The default call is unchanged (a tensor of all 4097 bins).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="tinycudann needs a GPU")
CK = Path("outputs/p4_8/arms/R0/ckpt_iter0120000.pt")
ROOM = Path("data/track_p4_5_testdense/L5.54_W5.28_d1.42_w1.52.h5")
HAVE = pytest.mark.skipif(not (CK.exists() and ROOM.exists()), reason="needs R0 @120K and the shape-1000 room")


def _setup(n_rx=4):
    import h5py

    from aaf.eval.p3_2_eval import load_model
    from aaf.models.conditioning_2d import build_cond_vector_2d
    dev = torch.device("cuda")
    model, renderer, *_ = load_model(CK, dev)
    with h5py.File(ROOM) as f:
        verts = [tuple(v) for v in json.loads(f.attrs["verts"])]
        rx = torch.tensor(json.loads(f.attrs["receiver_pos"]), dtype=torch.float32)
        L, W = float(f.attrs["L"]), float(f.attrs["W"])
    pick = torch.linspace(0, len(rx) - 1, n_rx).round().long()
    R = rx[pick].to(dev)
    S = torch.tensor([[0.5, 1.6]], device=dev).expand(n_rx, -1)
    z = build_cond_vector_2d("geom_token", L, W, [0.15] * len(verts), verts=verts, device=dev)
    Z = z.unsqueeze(0).expand(n_rx, -1)
    lo, hi = torch.zeros(2, device=dev), torch.tensor([L, W], device=dev)
    return model, renderer, R, S, Z, lo, hi


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-30))


@GPU
@HAVE
def test_trained_model_is_awake():
    model, *_ = _setup(1)
    assert float(model.o_proj.weight.norm()) > 0.1
    assert float(model.film_signal.weight.norm()) > 0.1


@GPU
@HAVE
def test_601_bin_path_equals_the_full_path():
    model, renderer, R, S, Z, lo, hi = _setup()
    with torch.no_grad():
        Hf = renderer(model, R, S, lo, hi, z_s=Z)
        H6 = renderer(model, R, S, lo, hi, z_s=Z, n_out=601)
    assert Hf.shape[-1] == 4097 and H6.shape[-1] == 601
    assert _rel(H6, Hf[:, :601]) < 1e-6
    # The model's own outputs: bin 600's IMAGINARY signal part is kept (it is not the Nyquist bin of 601).
    pts = R.view(-1, 1, 2).expand(-1, 3, 2).contiguous()
    view = torch.zeros_like(pts)
    view[..., 0] = 1.0
    tx = S.view(-1, 1, 2).expand(-1, 3, 2).contiguous()
    with torch.no_grad():
        af, sf = model(pts, view, tx, z_s=Z)
        a6, s6 = model(pts, view, tx, z_s=Z, n_out=601)
    assert torch.equal(a6, af[..., :601]) or _rel(a6, af[..., :601]) < 1e-6
    assert float(sf[..., 600].imag.abs().max()) > 0                          # not vacuous: bin 600 has an imag part
    assert _rel(s6, sf[..., :601]) < 1e-6
    assert float(sf[..., -1].imag.abs().max()) == 0.0                         # the real Nyquist bin stays zeroed


@GPU
@HAVE
def test_601_bin_gradients_equal_the_full_path():
    model, renderer, R, S, Z, lo, hi = _setup(2)
    renderer.train()
    jit = torch.tensor(0.0123, device=R.device)
    params = [model.film_signal.weight, model.o_proj.weight, model._signal_films[1].weight]
    grads = []
    for n_out in (None, 601):
        model.zero_grad(set_to_none=True)
        kw = {} if n_out is None else {"n_out": n_out}
        H = renderer(model, R, S, lo, hi, z_s=Z, jitter=jit, **kw)[:, :601]
        (H.real.abs().sum() + H.imag.abs().sum()).backward()
        grads.append([p.grad.detach().clone() for p in params])
    for gf, g6 in zip(*grads):
        assert float(gf.abs().max()) > 0
        assert _rel(g6, gf) < 1e-5


@GPU
@HAVE
def test_per_ray_output_sums_to_the_render():
    model, renderer, R, S, Z, lo, hi = _setup()
    with torch.no_grad():
        H, per, dirs = renderer(model, R, S, lo, hi, z_s=Z, n_out=601, return_per_ray=True)
        dc = model.dc_term(Z, 601)
    assert per.shape == (R.shape[0], renderer.n_azi, 601) and dirs.shape == (renderer.n_azi, 2)
    assert _rel(per.sum(dim=1) + dc, H) < 1e-6
    assert torch.equal(dirs, renderer._ray_directions_2d(R.device))           # eval mode: the fixed grid
    assert float(per.abs().std(dim=1).max()) > 0                              # rays genuinely differ


@GPU
@HAVE
def test_token_override_identity_and_effect():
    model, renderer, R, S, Z, lo, hi = _setup()
    with torch.no_grad():
        e0 = model.token_embeddings(Z[:1])
        assert e0.shape[1] == model._tok_n
        H = renderer(model, R, S, lo, hi, z_s=Z, n_out=601)
        Hi = renderer(model, R, S, lo, hi, z_s=Z, n_out=601, tok_override=e0.expand(R.shape[0], -1, -1))
        g = torch.Generator(device=R.device).manual_seed(0)
        ep = e0 + 0.2 * e0.std() * torch.randn(e0.shape, device=R.device, generator=g)
        Hp = renderer(model, R, S, lo, hi, z_s=Z, n_out=601, tok_override=ep.expand(R.shape[0], -1, -1))
    assert _rel(Hi, H) < 1e-6
    # The effect is measured on 25-300 Hz (bins 50-600): bins 0-40 carry the DC head, which reads the RAW tokens and
    # dominates max |H| (~1e3 at bin 0), so a max-normalised difference would hide a real change of the field.
    d = float((Hp - H)[:, 50:].abs().mean() / H[:, 50:].abs().mean())
    assert d > 1e-2, d


@GPU
@HAVE
def test_default_call_is_unchanged():
    model, renderer, R, S, Z, lo, hi = _setup(2)
    with torch.no_grad():
        H = renderer(model, R, S, lo, hi, z_s=Z)
    assert torch.is_tensor(H) and H.shape == (2, 4097)
    with pytest.raises(ValueError):
        model(R.view(-1, 1, 2), torch.zeros(2, 1, 2, device=R.device), S.view(-1, 1, 2), z_s=Z, n_out=0)
