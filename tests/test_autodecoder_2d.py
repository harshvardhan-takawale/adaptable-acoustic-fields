"""INR2D_AutoDecoder: latent-conditioned forward, gradient flow, latent table."""
import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="tcnn requires CUDA"
)


@cuda_required
def test_forward_shapes_with_z_s():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    n_freq, n_rooms, latent_dim = 257, 7, 32
    model = INR2D_AutoDecoder(n_rooms=n_rooms, latent_dim=latent_dim,
                              n_freq_bins=n_freq).cuda()
    B, N = 2, 32
    pts = torch.rand(B, N, 2, device="cuda") * 2 - 1
    view = torch.tensor([[1.0, 0.0]] * N, device="cuda").unsqueeze(0).expand(B, N, 2)
    tx = torch.zeros(B, N, 2, device="cuda")
    z_s = torch.randn(B, latent_dim, device="cuda")

    attn, signal = model(pts, view, tx, z_s=z_s)
    assert attn.shape == (B, N, n_freq)
    assert signal.shape == (B, N, n_freq)
    assert attn.is_complex() and signal.is_complex()
    assert torch.all(signal[..., 0].imag == 0), "DC bin imag must be zero"
    assert torch.all(signal[..., -1].imag == 0), "Nyquist imag must be zero"
    assert torch.all(attn.real >= 0)


@cuda_required
def test_forward_requires_z_s():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(n_rooms=3, latent_dim=8, n_freq_bins=129).cuda()
    pts = torch.rand(1, 4, 2, device="cuda")
    view = torch.zeros(1, 4, 2, device="cuda")
    tx = torch.zeros(1, 4, 2, device="cuda")
    with pytest.raises(ValueError, match="requires z_s"):
        model(pts, view, tx)


@cuda_required
def test_backward_flows_to_network_and_latents():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(n_rooms=3, latent_dim=8, n_freq_bins=129).cuda()
    pts = torch.rand(1, 8, 2, device="cuda")
    view = torch.zeros(1, 8, 2, device="cuda")
    tx = torch.zeros(1, 8, 2, device="cuda")
    z_s = model.get_latent(torch.tensor([0], device="cuda"))    # uses the embedding

    attn, signal = model(pts, view, tx, z_s=z_s)
    loss = (
        signal.real.pow(2).mean() + signal.imag.pow(2).mean()
        + attn.real.pow(2).mean() + attn.imag.pow(2).mean()
    )
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"params with no grad: {no_grad}"
    # Latent embedding should also receive a non-zero gradient.
    lat_grad = model.latents.weight.grad
    assert lat_grad is not None
    assert torch.any(lat_grad != 0), "latent embedding got zero gradient"


@cuda_required
def test_z_s_actually_changes_output():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(n_rooms=3, latent_dim=8, n_freq_bins=129).cuda().eval()
    pts = torch.rand(1, 16, 2, device="cuda")
    view = torch.zeros(1, 16, 2, device="cuda")
    tx = torch.zeros(1, 16, 2, device="cuda")
    z_a = torch.zeros(1, 8, device="cuda")
    z_b = torch.ones(1, 8, device="cuda") * 2.0
    a1, s1 = model(pts, view, tx, z_s=z_a)
    a2, s2 = model(pts, view, tx, z_s=z_b)
    delta_attn = (a1 - a2).abs().mean().item()
    delta_signal = (s1 - s2).abs().mean().item()
    assert delta_attn > 1e-3, f"attn unchanged across z_s: Δ={delta_attn}"
    assert delta_signal > 1e-3, f"signal unchanged across z_s: Δ={delta_signal}"


@cuda_required
def test_get_latent_matches_embedding_table():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(n_rooms=5, latent_dim=4, n_freq_bins=129).cuda()
    z = model.get_latent(2)
    assert z.shape == (4,)
    expected = model.latents.weight[2]
    assert torch.allclose(z, expected)
    z_batch = model.get_latent(torch.tensor([0, 2, 4], device="cuda"))
    assert z_batch.shape == (3, 4)
    assert torch.allclose(z_batch[1], expected)
