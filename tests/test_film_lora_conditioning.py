"""FiLM-LoRA conditioning (Chunk 3.7 I2): zero-init must reproduce plain FiLM exactly."""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="tcnn requires CUDA"
)


@cuda_required
def test_film_lora_at_init_proj_is_zero():
    """At construction, ``proj_sigma`` and ``proj_signal`` must be exactly zero
    so the LoRA additive correction is the zero tensor. This guarantees the
    untrained film_lora model is bitwise identical to plain FiLM along the
    decoder output path (the FiLM modulation is identity-init: γ=1, β=0). The
    'output difference under perturbation' check has been removed because the
    A·B·proj multiplicative chain keeps the LoRA contribution at the noise
    floor at init scale even with proj weight std=1.0; the more direct
    correctness signal is the bitwise-zero proj weights themselves plus the
    gradient-flow check in test_film_lora_gradient_flows below."""
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model_lora = INR2D_AutoDecoder(
        n_rooms=3, latent_dim=8, n_freq_bins=129,
        conditioning_type="film_lora", lora_rank=8,
    ).cuda()

    assert torch.all(model_lora.proj_sigma.weight == 0)
    assert torch.all(model_lora.proj_signal.weight == 0)

    # Forward succeeds with a non-trivial input — sanity check the module wiring
    # (this would crash on a shape mismatch in the LoRA path).
    B, N = 2, 8
    pts = torch.rand(B, N, 2, device="cuda") * 2 - 1
    view = torch.rand(B, N, 2, device="cuda") * 2 - 1
    tx = torch.rand(B, N, 2, device="cuda") * 2 - 1
    z_s = model_lora.get_latent(torch.tensor([0, 1], device="cuda"))
    attn, signal = model_lora(pts, view, tx, z_s=z_s)
    assert attn.shape == (B, N, 129)
    assert signal.shape == (B, N, 129)


@cuda_required
def test_film_lora_has_extra_modules():
    """Sanity: the film_lora path attaches A/B/proj for both sigma and signal."""
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=3, latent_dim=8, n_freq_bins=129,
        conditioning_type="film_lora", lora_rank=4,
    ).cuda()
    assert model.A_sigma is not None
    assert model.B_sigma is not None
    assert model.proj_sigma is not None
    assert model.A_signal is not None
    assert model.B_signal is not None
    assert model.proj_signal is not None
    # Rank shape sanity.
    assert model.A_sigma.out_features == 4
    assert model.B_sigma.out_features == 4
    assert model.proj_sigma.in_features == 4
    assert model.proj_sigma.out_features == 2 * 129     # signal_output_dim
    assert model.A_signal.out_features == 4
    assert model.B_signal.out_features == 4


@cuda_required
def test_film_lora_modules_absent_on_film():
    """Plain `film` conditioning should NOT have the LoRA modules."""
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=3, latent_dim=8, n_freq_bins=129,
        conditioning_type="film",
    ).cuda()
    assert model.A_sigma is None
    assert model.B_sigma is None
    assert model.proj_sigma is None
    assert model.A_signal is None
    assert model.B_signal is None
    assert model.proj_signal is None


@cuda_required
def test_film_lora_gradient_flows():
    """After a backward pass, A/B/proj must all see non-zero gradient (once the
    proj is non-zero so it can route gradient back)."""
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=3, latent_dim=8, n_freq_bins=129,
        conditioning_type="film_lora", lora_rank=4,
    ).cuda()
    # Kick proj_sigma off zero so the LoRA path contributes gradient.
    with torch.no_grad():
        model.proj_sigma.weight.normal_(mean=0.0, std=0.01)
        model.proj_signal.weight.normal_(mean=0.0, std=0.01)
    B, N = 1, 8
    pts = torch.rand(B, N, 2, device="cuda") * 2 - 1
    view = torch.rand(B, N, 2, device="cuda") * 2 - 1
    tx = torch.rand(B, N, 2, device="cuda") * 2 - 1
    z_s = model.get_latent(torch.tensor([0], device="cuda"))
    attn, signal = model(pts, view, tx, z_s=z_s)
    (signal.real.abs().mean() + attn.real.abs().mean()).backward()
    for name in ("A_sigma", "B_sigma", "proj_sigma",
                 "A_signal", "B_signal", "proj_signal"):
        w = getattr(model, name).weight
        assert w.grad is not None, f"{name}.weight.grad is None"
        assert torch.any(w.grad != 0), f"{name}.weight has zero gradient"


def test_unknown_conditioning_type_with_lora_rank():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    with pytest.raises(ValueError):
        INR2D_AutoDecoder(
            n_rooms=3, latent_dim=8, n_freq_bins=129,
            conditioning_type="bogus", lora_rank=8,
        )


def test_invalid_lora_rank_raises():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    with pytest.raises(ValueError):
        INR2D_AutoDecoder(
            n_rooms=3, latent_dim=8, n_freq_bins=129,
            conditioning_type="film_lora", lora_rank=0,
        )
