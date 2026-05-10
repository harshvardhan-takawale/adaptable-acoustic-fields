"""FreqRenderer2D — 2D frequency-domain volume renderer.

Port of `aaf._inference_ref.inference_renderer.AVRRenderFD_FreqDep_PhaseCorrection_new`
to 2D shoebox geometry. Differences from the 3D parent:

  * Ray sampler: stochastic uniform-azimuth on `[0, 2π)`. No elevation, no
    extra zenith/nadir rays. Per-iteration jitter shifts the angle grid by a
    uniform offset in `[0, 2π / n_azi)` so the per-batch direction set rotates.
  * Ray-AABB intersection: 2D rectangle (4-wall slab), not 3D box.
  * Geometric attenuation: off permanently for Phase 1 (network learns spreading).

The volume rendering equation is identical to the parent: cumulative amplitude
transmittance from σ-only opacity, cumulative material phase factor from β,
plus per-sample geometric phase from the source-receiver delay.

Public API
----------
    renderer = FreqRenderer2D(n_azi=64, n_pts_per_ray=64, ...)
    H_pred = renderer(model, rx_pos, tx_pos, room_min, room_max, z_s=None)
    # rx_pos: [B, 2], tx_pos: [B, 2], room_min/room_max: [2]
    # H_pred: [B, n_freq_bins] complex64
"""
from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn


class FreqRenderer2D(nn.Module):
    def __init__(
        self,
        n_azi: int = 64,
        n_pts_per_ray: int = 64,
        near: float = 1e-3,
        fs: int = 4096,
        n_time_samples: int = 8192,
        c: float = 343.0,
        use_geometric_attn: bool = False,
    ):
        super().__init__()
        if n_azi <= 0 or n_pts_per_ray <= 0:
            raise ValueError(f"n_azi and n_pts_per_ray must be positive")
        self.n_azi = int(n_azi)
        self.n_pts_per_ray = int(n_pts_per_ray)
        self.near = float(near)
        self.fs = int(fs)
        self.n_time_samples = int(n_time_samples)
        self.c = float(c)
        self.use_geometric_attn = bool(use_geometric_attn)

        n_freq_bins = self.n_time_samples // 2 + 1
        f_axis = torch.arange(n_freq_bins, dtype=torch.float32) * (self.fs / self.n_time_samples)
        self.register_buffer("f_axis", f_axis, persistent=False)

        # Pre-compute the deterministic angle grid; per-iteration jitter is added at fwd.
        theta_grid = torch.arange(self.n_azi, dtype=torch.float32) * (2.0 * math.pi / self.n_azi)
        self.register_buffer("theta_grid", theta_grid, persistent=False)

    @property
    def n_freq_bins(self) -> int:
        return self.n_time_samples // 2 + 1

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _ray_directions_2d(self, device: torch.device) -> torch.Tensor:
        """Returns (cos θ, sin θ) for n_azi rays, jittered per call.

        Out shape: [n_azi, 2].
        """
        # In training mode, jitter; in eval mode, use the deterministic grid.
        if self.training:
            jitter = torch.empty(1, device=device).uniform_(0.0, 2.0 * math.pi / self.n_azi)
            theta = self.theta_grid + jitter
        else:
            theta = self.theta_grid
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    def _ray_aabb_intersect_2d(
        self,
        rx_pos: torch.Tensor,
        dirs: torch.Tensor,
        room_min: torch.Tensor,
        room_max: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Per-ray distance to far wall via 2D slab algorithm.

        Args:
            rx_pos: [B, 2] receiver positions.
            dirs:   [n_azi, 2] (cos θ, sin θ) ray directions.
            room_min, room_max: [2] AABB corners (m).
        Returns:
            t_far: [B, n_azi] distance to far wall along each ray, > 0.
        """
        # Broadcast: rx_pos [B,1,2], dirs [1,n_azi,2]
        rx = rx_pos.unsqueeze(1)
        dr = dirs.unsqueeze(0)
        # Avoid div-by-zero when a ray is exactly axis-aligned.
        safe_dr = torch.where(dr.abs() < eps, torch.full_like(dr, eps), dr)
        t_lo = (room_min - rx) / safe_dr  # [B, n_azi, 2]
        t_hi = (room_max - rx) / safe_dr
        t_min = torch.minimum(t_lo, t_hi)  # entry along each axis
        t_max = torch.maximum(t_lo, t_hi)  # exit  along each axis
        # Far intersection = minimum of the two axis-exits.
        t_far = t_max.min(dim=-1).values  # [B, n_azi]
        # Diagonal upper bound for safety.
        diag = torch.linalg.vector_norm(room_max - room_min) + 1.0
        t_far = t_far.clamp(min=self.near, max=float(diag))
        return t_far

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        model: nn.Module,
        rx_pos: torch.Tensor,
        tx_pos: torch.Tensor,
        room_min: torch.Tensor,
        room_max: torch.Tensor,
        z_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Render H(f) at each receiver via 2D volume integration.

        Args:
            model: an ``nn.Module`` whose forward signature is
                   ``forward(pts, view, tx, tx_view=None, z_s=None) -> (attn, signal)``
                   with ``attn``/``signal`` complex tensors of shape
                   ``[B, N, n_freq_bins]``.
            rx_pos:    [B, 2] receivers.
            tx_pos:    [B, 2] source positions (broadcast over rays/points).
            room_min:  [2] AABB lower corner.
            room_max:  [2] AABB upper corner.
            z_s:       [B, latent_dim] latent codes or None. Forwarded to the model.

        Returns:
            H_pred: [B, n_freq_bins] complex.
        """
        if rx_pos.dim() != 2 or rx_pos.shape[-1] != 2:
            raise ValueError(f"rx_pos must be [B, 2], got {tuple(rx_pos.shape)}")
        if tx_pos.shape != rx_pos.shape:
            raise ValueError(
                f"tx_pos shape {tuple(tx_pos.shape)} must match rx_pos {tuple(rx_pos.shape)}"
            )

        device = rx_pos.device
        B = rx_pos.size(0)
        N_a = self.n_azi
        N_p = self.n_pts_per_ray

        dirs = self._ray_directions_2d(device)  # [n_azi, 2]
        t_far = self._ray_aabb_intersect_2d(rx_pos, dirs, room_min, room_max)  # [B, n_azi]

        # Sample N_p points uniformly in [near, t_far] per ray.
        # u: [N_p] in [0, 1]; broadcast to [B, n_azi, N_p].
        u = torch.linspace(0.0, 1.0, N_p, device=device)
        d_vals = self.near + (t_far.unsqueeze(-1) - self.near) * u  # [B, n_azi, N_p]

        # pts[b, a, i] = rx_pos[b] + dirs[a] * d_vals[b, a, i]
        pts = rx_pos.view(B, 1, 1, 2) + dirs.view(1, N_a, 1, 2) * d_vals.unsqueeze(-1)
        # view direction at each sample = ray direction (broadcast)
        view = dirs.view(1, N_a, 1, 2).expand(B, N_a, N_p, 2)
        # tx broadcasts across rays/points
        tx = tx_pos.view(B, 1, 1, 2).expand(B, N_a, N_p, 2)
        # tx_view: omni source — pass None to the model (it maps None → zero vector)
        tx_view = None

        # Flatten to [B, N_a*N_p, 2] for the model.
        N = N_a * N_p
        pts_flat = pts.reshape(B, N, 2)
        view_flat = view.reshape(B, N, 2)
        tx_flat = tx.reshape(B, N, 2)
        attn, signal = model(pts_flat, view_flat, tx_flat, tx_view=tx_view, z_s=z_s)
        # attn, signal: [B, N, n_freq_bins] complex
        n_freq = attn.size(-1)
        attn = attn.view(B, N_a, N_p, n_freq)
        signal = signal.view(B, N_a, N_p, n_freq)

        # σ + jβ decomposition (verbatim from INFER ref).
        sigma = attn.real.clamp(min=0)            # [B, N_a, N_p, n_freq]
        beta = attn.imag

        # Δu between consecutive sample distances along each ray.
        # d_vals: [B, N_a, N_p] → diffs: [B, N_a, N_p-1]; pad tail with 1e10.
        d_diff = d_vals[..., 1:] - d_vals[..., :-1]
        d_diff = torch.cat(
            [d_diff, torch.full_like(d_diff[..., :1], 1e10)], dim=-1
        )  # [B, N_a, N_p]
        delta_u = d_diff.unsqueeze(-1)  # [B, N_a, N_p, 1]

        # Local opacity from absorption.
        alpha = 1.0 - torch.exp(-sigma * delta_u)  # [B, N_a, N_p, n_freq]

        # Cumulative amplitude transmittance: T_amp[i] = ∏_{j<i} (1 - α_j)
        ones_lead = torch.ones_like(alpha[..., :1, :])
        transmittance_amp = torch.cumprod(
            torch.cat([ones_lead, 1.0 - alpha[..., :-1, :] + 1e-10], dim=-2), dim=-2,
        )

        # Cumulative material phase: P[i] = exp(j ∑_{j<i} β_j Δu_j)
        phase_increments = beta[..., :-1, :] * delta_u[..., :-1, :]
        zeros_lead = torch.zeros_like(phase_increments[..., :1, :])
        cumulative_phase = torch.cumsum(
            torch.cat([zeros_lead, phase_increments], dim=-2), dim=-2,
        )
        transmittance_phase = torch.exp(1j * cumulative_phase)
        transmittance = transmittance_amp * transmittance_phase

        # Geometric phase: exp(-j 2π f d / c) per (sample, freq)
        geom_phase = torch.exp(
            -1j * 2.0 * math.pi
            * self.f_axis.view(1, 1, 1, -1)
            * (d_vals.unsqueeze(-1) / self.c)
        )

        if self.use_geometric_attn:
            # 1/r geometric amplitude attenuation (off by default in Phase 1).
            geom_amp = 1.0 / d_vals.clamp(min=1e-3).unsqueeze(-1)
            signal_with_delays = signal * geom_phase * geom_amp
        else:
            signal_with_delays = signal * geom_phase

        # Sum over points along each ray, then over rays.
        rendered_per_ray = (signal_with_delays * alpha * transmittance).sum(dim=-2)
        H_pred = rendered_per_ray.sum(dim=-2)  # [B, n_freq]
        return H_pred
