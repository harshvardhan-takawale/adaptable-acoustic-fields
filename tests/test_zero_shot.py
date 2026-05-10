"""Zero-shot adapter: inner-loop loss decreases; z_star.pt round-trips.

The full zero_shot_adapt() pipeline depends on a real trained checkpoint and
HDF5 file, which we don't have at unit-test time. This test exercises only
the inner adaptation loop with a mock model.
"""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="renderer + tcnn-style ops use CUDA paths"
)


class _MockModel(nn.Module):
    """Toy model mapping (z_s, pts) → complex H[B, n_freq] in a way that admits
    a deterministic optimum z_star for a given target field.

    Output: H[b, f] = exp(-(z_s[b]·u_f)) · phase(f)
    where u_f is a fixed direction in latent space and phase(f) is a fixed
    phase ramp. This makes the loss landscape convex enough to test descent.
    """
    def __init__(self, latent_dim, n_freq):
        super().__init__()
        self.n_freq = n_freq
        torch.manual_seed(0)
        self.register_buffer("u", torch.randn(latent_dim) / math.sqrt(latent_dim))
        self.register_buffer(
            "phase", torch.exp(1j * torch.arange(n_freq, dtype=torch.float64) * 0.01)
        )
        self._dummy = nn.Parameter(torch.zeros(1))  # so model.parameters() is non-empty

    def forward(self, pts, view, tx, tx_view=None, z_s=None):
        # Ignore pts/view/tx; only z_s drives the output.
        if z_s.dim() == 1:
            z_s = z_s.unsqueeze(0)
        B = z_s.size(0)
        N = pts.size(1) if pts.dim() == 3 else 1
        amp = torch.exp(-(z_s @ self.u).real)            # [B], real
        # Make a complex H[B, N, n_freq] (renderer expects [B, N, n_freq] from model).
        H = amp.view(B, 1, 1) * self.phase.to(z_s.device).view(1, 1, -1).to(torch.complex64)
        H = H.expand(B, N, self.n_freq)
        # Return (attn, signal) with attn just zeros so renderer is bypassed in this mock.
        zero_attn = torch.zeros_like(H)
        return zero_attn, H


@cuda_required
def test_inner_loop_decreases_obs_loss(tmp_path):
    """Run a minimal version of the inner adaptation loop directly (no renderer)
    and confirm the observed loss decreases.
    """
    torch.manual_seed(0)
    latent_dim, n_freq = 16, 33
    device = "cuda"
    model = _MockModel(latent_dim=latent_dim, n_freq=n_freq).to(device)
    target_z = torch.randn(latent_dim, device=device) * 0.5  # the "true" latent
    target_H = model(torch.zeros(1, 1, 2, device=device),
                     torch.zeros(1, 1, 2, device=device),
                     torch.zeros(1, 1, 2, device=device), z_s=target_z)[1]  # [1, 1, n_freq]

    z_star = nn.Parameter(torch.randn(latent_dim, device=device) / math.sqrt(latent_dim))
    opt = torch.optim.Adam([z_star], lr=5e-2)
    losses = []
    for _ in range(80):
        z_s = z_star.unsqueeze(0)
        _, H_pred = model(torch.zeros(1, 1, 2, device=device),
                          torch.zeros(1, 1, 2, device=device),
                          torch.zeros(1, 1, 2, device=device), z_s=z_s)
        loss = (H_pred.real - target_H.real).abs().mean() + \
               (H_pred.imag - target_H.imag).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not decrease enough: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )

    # Round-trip z_star.pt
    out = tmp_path / "z_star.pt"
    torch.save(z_star.detach().cpu(), out)
    loaded = torch.load(out)
    assert loaded.shape == (latent_dim,)
    assert torch.allclose(loaded, z_star.detach().cpu(), atol=1e-6)
