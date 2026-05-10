"""INR2D_Single: forward shape, gradient flow, RFFT symmetry.

tcnn requires CUDA at instantiation, so this test is CUDA-only. It will be
skipped on a CPU-only node.
"""
import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="tcnn requires CUDA"
)


@cuda_required
def test_forward_output_shapes_and_rfft_symmetry():
    from aaf.models.inr_2d import INR2D_Single

    n_freq = 257  # n_time_samples = 512 (even)
    model = INR2D_Single(n_freq_bins=n_freq).cuda()

    B, N = 2, 32
    pts = torch.rand(B, N, 2, device="cuda") * 2 - 1  # [-1, 1]
    view = torch.tensor([[1.0, 0.0]] * N, device="cuda").unsqueeze(0).expand(B, N, 2)
    tx = torch.zeros(B, N, 2, device="cuda")

    attn, signal = model(pts, view, tx)
    assert attn.shape == (B, N, n_freq), f"unexpected attn shape {attn.shape}"
    assert signal.shape == (B, N, n_freq), f"unexpected signal shape {signal.shape}"
    assert attn.is_complex() and signal.is_complex()

    # RFFT symmetry: imag of DC and (since 512 is even) Nyquist are 0.
    assert torch.all(signal[..., 0].imag == 0), "DC bin imag must be zero"
    assert torch.all(signal[..., -1].imag == 0), (
        "Nyquist bin imag must be zero for even n_time_samples"
    )

    # σ ≥ 0 enforced by softplus.
    assert torch.all(attn.real >= 0)


@cuda_required
def test_backward_flows_to_all_params():
    from aaf.models.inr_2d import INR2D_Single

    model = INR2D_Single(n_freq_bins=129).cuda()
    pts = torch.rand(1, 8, 2, device="cuda") * 2 - 1
    view = torch.zeros(1, 8, 2, device="cuda")
    tx = torch.zeros(1, 8, 2, device="cuda")

    attn, signal = model(pts, view, tx)
    # Touch both branches so every MLP gets a gradient.
    loss = (
        signal.real.pow(2).mean()
        + signal.imag.pow(2).mean()
        + attn.real.pow(2).mean()
        + attn.imag.pow(2).mean()
    )
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"params with no grad: {no_grad}"


@cuda_required
def test_z_s_argument_ignored_no_error():
    """z_s is accepted in the signature but ignored — passing it must not error."""
    from aaf.models.inr_2d import INR2D_Single

    model = INR2D_Single(n_freq_bins=129, latent_dim=32).cuda()
    pts = torch.rand(2, 8, 2, device="cuda") * 2 - 1
    view = torch.zeros(2, 8, 2, device="cuda")
    tx = torch.zeros(2, 8, 2, device="cuda")
    z_s = torch.randn(2, 32, device="cuda")
    _ = model(pts, view, tx, z_s=z_s)


@cuda_required
def test_tx_view_none_substitutes_zero():
    from aaf.models.inr_2d import INR2D_Single

    model = INR2D_Single(n_freq_bins=129).cuda().eval()
    pts = torch.rand(1, 4, 2, device="cuda") * 2 - 1
    view = torch.zeros(1, 4, 2, device="cuda")
    tx = torch.zeros(1, 4, 2, device="cuda")

    a1, s1 = model(pts, view, tx, tx_view=None)
    a2, s2 = model(pts, view, tx, tx_view=torch.zeros_like(tx))
    # Up to encoder normalization, tx_view=None and tx_view=0 should produce identical
    # outputs deterministically.
    assert torch.allclose(a1.real, a2.real, atol=1e-5)
    assert torch.allclose(s1.real, s2.real, atol=1e-5)
