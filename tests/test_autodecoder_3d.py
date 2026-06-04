"""Tests for INR3D_AutoDecoder (P2-2 model)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="INR3D_AutoDecoder needs tcnn (CUDA).",
)


def _make_model(n_rooms=5, latent_dim=16, n_freq_bins=257):
    from aaf.models.inr_3d import INR3D_AutoDecoder
    hg = {
        "otype": "HashGrid",
        "n_levels": 4,
        "n_features_per_level": 2,
        "log2_hashmap_size": 12,
        "base_resolution": 8,
        "per_level_scale": 1.5,
    }
    return INR3D_AutoDecoder(
        n_rooms=n_rooms,
        latent_dim=latent_dim,
        n_freq_bins=n_freq_bins,
        hash_grid_config=hg,
        conditioning_type="film",
        latent_jitter_sigma=0.1,
        l_head_enabled=True,
    )


def test_autodecoder_3d_forward_shape():
    m = _make_model(n_rooms=5, latent_dim=16, n_freq_bins=257).cuda()
    B, N = 2, 16
    z_s = torch.randn(B, 16, device="cuda")
    pts = torch.rand(B, N, 3, device="cuda") * 2 - 1
    view = torch.rand(B, N, 3, device="cuda") * 2 - 1
    tx = torch.rand(B, N, 3, device="cuda") * 2 - 1
    attn, signal = m(pts, view, tx, z_s=z_s)
    assert attn.shape == (B, N, 257)
    assert signal.shape == (B, N, 257)
    assert attn.is_complex() and signal.is_complex()


def test_autodecoder_3d_requires_z_s():
    m = _make_model(n_rooms=5, latent_dim=16, n_freq_bins=257).cuda()
    pts = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    view = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    tx = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    with pytest.raises(ValueError):
        m(pts, view, tx, z_s=None)


def test_autodecoder_3d_predict_geometry_shape():
    m = _make_model(n_rooms=5, latent_dim=16, n_freq_bins=129).cuda()
    z_s = torch.randn(3, 16, device="cuda")
    out = m.predict_geometry(z_s)
    assert out is not None
    assert out.shape == (3, 3)


def test_autodecoder_3d_predict_geometry_1d_input():
    """Should accept [latent_dim] and promote to [1, 3]."""
    m = _make_model(n_rooms=5, latent_dim=16, n_freq_bins=129).cuda()
    z_s = torch.randn(16, device="cuda")
    out = m.predict_geometry(z_s)
    assert out.shape == (1, 3)


def test_autodecoder_3d_different_z_different_geom():
    """At construction, FiLM is identity (γ=1, β=0) so z_s has no effect on
    the forward path until trained. The geometry head IS initialized non-
    trivially, so predict_geometry should differ for different z values.
    Once a single backward pass perturbs the FiLM weights from zero, the
    forward output will differ too."""
    m = _make_model(n_rooms=5, latent_dim=16, n_freq_bins=129).cuda()
    m.eval()
    z1 = torch.randn(1, 16, device="cuda")
    z2 = torch.randn(1, 16, device="cuda") + 1.0
    g1 = m.predict_geometry(z1)
    g2 = m.predict_geometry(z2)
    assert (g1 - g2).abs().max().item() > 0

    # After one training step, the FiLM weights are non-zero → forward
    # output must respond to z.
    m.train()
    z_s = m.get_latent(torch.tensor([0, 1], device="cuda"))
    pts = torch.rand(2, 8, 3, device="cuda") * 2 - 1
    view = torch.rand(2, 8, 3, device="cuda") * 2 - 1
    tx = torch.rand(2, 8, 3, device="cuda") * 2 - 1
    attn, signal = m(pts, view, tx, z_s=z_s)
    loss = signal.abs().mean() + attn.abs().mean()
    loss.backward()
    # Walk a few optimization steps to break FiLM identity.
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        z_s = m.get_latent(torch.tensor([0, 1], device="cuda"))
        attn, signal = m(pts, view, tx, z_s=z_s)
        loss = signal.abs().mean() + attn.abs().mean()
        loss.backward()
        opt.step()

    m.eval()
    pts1 = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    view1 = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    tx1 = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    a1, s1 = m(pts1, view1, tx1, z_s=z1)
    a2, s2 = m(pts1, view1, tx1, z_s=z2)
    assert (a1 - a2).abs().max().item() > 0
    assert (s1 - s2).abs().max().item() > 0


def test_autodecoder_3d_gradient_flow_latents_film_l_head():
    """Backward through loss must populate gradients on latent table, FiLM
    sigma/signal, and the geometry head."""
    m = _make_model(n_rooms=5, latent_dim=16, n_freq_bins=129).cuda()
    z_s = m.get_latent(torch.tensor([0, 1], device="cuda"))                  # [2, 16]
    pts = torch.rand(2, 8, 3, device="cuda") * 2 - 1
    view = torch.rand(2, 8, 3, device="cuda") * 2 - 1
    tx = torch.rand(2, 8, 3, device="cuda") * 2 - 1
    attn, signal = m(pts, view, tx, z_s=z_s)
    geom_pred = m.predict_geometry(z_s)
    loss = signal.abs().mean() + attn.abs().mean() + geom_pred.abs().mean()
    loss.backward()
    assert m.latents.weight.grad is not None
    assert torch.isfinite(m.latents.weight.grad).any()
    assert m.film_sigma.weight.grad is not None
    assert m.film_signal.weight.grad is not None
    assert m.l_head.weight.grad is not None
    assert torch.isfinite(m.l_head.weight.grad).any()


def test_autodecoder_3d_latent_jitter_off_at_eval():
    """When .eval() is set, get_latent must NOT inject noise."""
    m = _make_model(n_rooms=5, latent_dim=16, n_freq_bins=129).cuda()
    m.eval()
    z_a = m.get_latent(torch.tensor([0], device="cuda"))
    z_b = m.get_latent(torch.tensor([0], device="cuda"))
    assert torch.allclose(z_a, z_b)


def test_autodecoder_3d_latent_jitter_on_at_train():
    """When .train() is set, get_latent CAN inject noise (different draws)."""
    m = _make_model(n_rooms=5, latent_dim=16, n_freq_bins=129).cuda()
    m.train()
    torch.manual_seed(0)
    z_a = m.get_latent(torch.tensor([0], device="cuda"))
    torch.manual_seed(1)
    z_b = m.get_latent(torch.tensor([0], device="cuda"))
    assert not torch.allclose(z_a, z_b)
