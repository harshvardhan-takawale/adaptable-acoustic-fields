"""FreqRenderer3D — 3D frequency-domain volume renderer.

3D port of `aaf.renderers.freq_2d.FreqRenderer2D`. The rendering equation is
identical (σ + jβ decomposition → amplitude transmittance via cumprod, material
phase via cumsum, geometric phase per (sample, freq)); only the geometry
helpers change:

  - Ray sampler: ``n_azi × n_ele`` solid-angle-uniform grid + 2 pole rays,
    yielding ``n_azi · n_ele + 2`` total rays. Azimuth is stratified uniform
    in [0, 2π) with per-iteration jitter (training only); elevation is
    ``arccos(2u - 1)`` for stratified ``u ∈ linspace(0, 1, n_ele)`` so the
    solid-angle weight is uniform. The 2 pole rays are explicitly added so
    zenith/nadir reflections (especially floor/ceiling first axial mode)
    aren't lost.
    Mirrors the vendored INFER pattern at ``aaf/_inference_ref/inference_renderer.py:40-57``.

  - Ray-AABB intersection: 3D slab algorithm — direct port of 2D over 3
    axes. Returns per-(receiver, ray) far distance so the ``n_pts_per_ray``
    sample budget is packed into actual in-room geometry (vs the vendored
    INFER renderer's fixed near/far which wastes ~40% outside the room).

Public API
----------
    renderer = FreqRenderer3D(n_azi=16, n_ele=16, n_pts_per_ray=32, ...)
    H_pred = renderer(model, rx_pos, tx_pos, room_min, room_max, z_s=None)
    # rx_pos: [B, 3], tx_pos: [B, 3], room_min/room_max: [3]
    # H_pred: [B, n_freq_bins] complex64
"""
from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn


class FreqRenderer3D(nn.Module):
    def __init__(
        self,
        n_azi: int = 16,
        n_ele: int = 16,
        n_pts_per_ray: int = 32,
        near: float = 1e-3,
        fs: int = 4096,
        n_time_samples: int = 8192,
        c: float = 343.0,
        use_geometric_attn: bool = False,
        include_pole_rays: bool = True,
    ):
        super().__init__()
        if n_azi <= 0 or n_ele <= 0 or n_pts_per_ray <= 0:
            raise ValueError(
                f"n_azi, n_ele, n_pts_per_ray must be positive; "
                f"got {n_azi}, {n_ele}, {n_pts_per_ray}"
            )
        self.n_azi = int(n_azi)
        self.n_ele = int(n_ele)
        self.n_pts_per_ray = int(n_pts_per_ray)
        self.include_pole_rays = bool(include_pole_rays)
        self.n_rays = self.n_azi * self.n_ele + (2 if self.include_pole_rays else 0)
        self.near = float(near)
        self.fs = int(fs)
        self.n_time_samples = int(n_time_samples)
        self.c = float(c)
        self.use_geometric_attn = bool(use_geometric_attn)

        n_freq_bins = self.n_time_samples // 2 + 1
        f_axis = torch.arange(n_freq_bins, dtype=torch.float32) * (self.fs / self.n_time_samples)
        self.register_buffer("f_axis", f_axis, persistent=False)

        # Deterministic azimuth / elevation grids (training-time jitter added at fwd).
        # Azimuth: stratified uniform on [0, 2π).
        azi_grid = torch.arange(self.n_azi, dtype=torch.float32) * (2.0 * math.pi / self.n_azi)
        self.register_buffer("azi_grid", azi_grid, persistent=False)
        # Elevation: u_i = (i + 0.5) / n_ele (cell centers) → θ = arccos(2u - 1).
        # Uniform-in-cosθ stratification, no per-cell jitter (keeps eval mode deterministic;
        # azi jitter alone is enough to break grid alignment per-iter).
        u_ele = (torch.arange(self.n_ele, dtype=torch.float32) + 0.5) / self.n_ele
        ele_grid = torch.acos(2.0 * u_ele - 1.0)
        self.register_buffer("ele_grid", ele_grid, persistent=False)

    @property
    def n_freq_bins(self) -> int:
        return self.n_time_samples // 2 + 1

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _ray_directions_3d(self, device: torch.device) -> torch.Tensor:
        """Returns 3D unit-vector ray directions, optionally jittered.

        Out shape: [n_rays, 3] where n_rays = n_azi · n_ele (+ 2 if poles).
        """
        if self.training:
            azi_jitter = torch.empty(1, device=device).uniform_(0.0, 2.0 * math.pi / self.n_azi)
            azi = self.azi_grid + azi_jitter
        else:
            azi = self.azi_grid
        ele = self.ele_grid                                                 # [n_ele]
        # Cartesian unit vectors from spherical:
        # x = sin(ele) cos(azi); y = sin(ele) sin(azi); z = cos(ele).
        sin_ele = torch.sin(ele).unsqueeze(0)                                # [1, n_ele]
        cos_ele = torch.cos(ele).unsqueeze(0)                                # [1, n_ele]
        cos_azi = torch.cos(azi).unsqueeze(-1)                               # [n_azi, 1]
        sin_azi = torch.sin(azi).unsqueeze(-1)                               # [n_azi, 1]
        x = (sin_ele * cos_azi).reshape(-1)                                  # [n_azi*n_ele]
        y = (sin_ele * sin_azi).reshape(-1)
        z = cos_ele.expand(self.n_azi, -1).reshape(-1)
        dirs = torch.stack([x, y, z], dim=-1)                                # [n_azi*n_ele, 3]
        if self.include_pole_rays:
            poles = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
                                 dtype=dirs.dtype, device=device)
            dirs = torch.cat([dirs, poles], dim=0)
        return dirs

    def _ray_aabb_intersect_3d(
        self,
        rx_pos: torch.Tensor,
        dirs: torch.Tensor,
        room_min: torch.Tensor,
        room_max: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Per-ray far-wall distance via 3D slab algorithm.

        Args:
            rx_pos: [B, 3] receiver positions.
            dirs:   [n_rays, 3] unit-vector ray directions.
            room_min, room_max: [3] AABB corners (m).
        Returns:
            t_far: [B, n_rays] distance to far wall along each ray, > self.near.
        """
        rx = rx_pos.unsqueeze(1)                                            # [B, 1, 3]
        dr = dirs.unsqueeze(0)                                              # [1, n_rays, 3]
        safe_dr = torch.where(dr.abs() < eps, torch.full_like(dr, eps), dr)
        t_lo = (room_min - rx) / safe_dr                                    # [B, n_rays, 3]
        t_hi = (room_max - rx) / safe_dr
        t_min = torch.minimum(t_lo, t_hi)
        t_max = torch.maximum(t_lo, t_hi)
        t_far = t_max.min(dim=-1).values                                    # [B, n_rays]
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
        """Render H(f) at each receiver via 3D volume integration.

        Args:
            model: an ``nn.Module`` whose forward signature is
                   ``forward(pts, view, tx, tx_view=None, z_s=None) -> (attn, signal)``
                   with ``attn``/``signal`` complex tensors of shape
                   ``[B, N, n_freq_bins]`` (N = n_rays · n_pts_per_ray).
            rx_pos:    [B, 3] receivers.
            tx_pos:    [B, 3] source positions (broadcast over rays/points).
            room_min:  [3] AABB lower corner.
            room_max:  [3] AABB upper corner.
            z_s:       [B, latent_dim] latent codes or None. Forwarded to model.

        Returns:
            H_pred: [B, n_freq_bins] complex.
        """
        if rx_pos.dim() != 2 or rx_pos.shape[-1] != 3:
            raise ValueError(f"rx_pos must be [B, 3], got {tuple(rx_pos.shape)}")
        if tx_pos.shape != rx_pos.shape:
            raise ValueError(
                f"tx_pos shape {tuple(tx_pos.shape)} must match rx_pos {tuple(rx_pos.shape)}"
            )

        device = rx_pos.device
        B = rx_pos.size(0)
        N_r = self.n_rays
        N_p = self.n_pts_per_ray

        dirs = self._ray_directions_3d(device)                              # [n_rays, 3]
        t_far = self._ray_aabb_intersect_3d(rx_pos, dirs, room_min, room_max)  # [B, n_rays]

        # Sample N_p points uniformly in [near, t_far] per ray.
        u = torch.linspace(0.0, 1.0, N_p, device=device)
        d_vals = self.near + (t_far.unsqueeze(-1) - self.near) * u           # [B, n_rays, N_p]

        # pts[b, r, i] = rx_pos[b] + dirs[r] * d_vals[b, r, i]
        pts = rx_pos.view(B, 1, 1, 3) + dirs.view(1, N_r, 1, 3) * d_vals.unsqueeze(-1)
        view = dirs.view(1, N_r, 1, 3).expand(B, N_r, N_p, 3)
        tx = tx_pos.view(B, 1, 1, 3).expand(B, N_r, N_p, 3)
        tx_view = None

        N = N_r * N_p
        pts_flat = pts.reshape(B, N, 3)
        view_flat = view.reshape(B, N, 3)
        tx_flat = tx.reshape(B, N, 3)
        attn, signal = model(pts_flat, view_flat, tx_flat, tx_view=tx_view, z_s=z_s)
        n_freq = attn.size(-1)
        attn = attn.view(B, N_r, N_p, n_freq)
        signal = signal.view(B, N_r, N_p, n_freq)

        # σ + jβ decomposition (verbatim from INFER ref / 2D port).
        sigma = attn.real.clamp(min=0)
        beta = attn.imag

        # Δu between consecutive sample distances along each ray.
        d_diff = d_vals[..., 1:] - d_vals[..., :-1]
        d_diff = torch.cat(
            [d_diff, torch.full_like(d_diff[..., :1], 1e10)], dim=-1
        )                                                                    # [B, N_r, N_p]
        delta_u = d_diff.unsqueeze(-1)                                       # [B, N_r, N_p, 1]

        # Local opacity from absorption.
        alpha = 1.0 - torch.exp(-sigma * delta_u)                            # [B, N_r, N_p, n_freq]

        # Cumulative amplitude transmittance.
        ones_lead = torch.ones_like(alpha[..., :1, :])
        transmittance_amp = torch.cumprod(
            torch.cat([ones_lead, 1.0 - alpha[..., :-1, :] + 1e-10], dim=-2), dim=-2,
        )

        # Cumulative material phase.
        phase_increments = beta[..., :-1, :] * delta_u[..., :-1, :]
        zeros_lead = torch.zeros_like(phase_increments[..., :1, :])
        cumulative_phase = torch.cumsum(
            torch.cat([zeros_lead, phase_increments], dim=-2), dim=-2,
        )
        transmittance_phase = torch.exp(1j * cumulative_phase)
        transmittance = transmittance_amp * transmittance_phase

        # Geometric phase: exp(-j 2π f d / c) per (sample, freq).
        geom_phase = torch.exp(
            -1j * 2.0 * math.pi
            * self.f_axis.view(1, 1, 1, -1)
            * (d_vals.unsqueeze(-1) / self.c)
        )

        if self.use_geometric_attn:
            geom_amp = 1.0 / d_vals.clamp(min=1e-3).unsqueeze(-1)
            signal_with_delays = signal * geom_phase * geom_amp
        else:
            signal_with_delays = signal * geom_phase

        # Sum over points along each ray, then over rays.
        rendered_per_ray = (signal_with_delays * alpha * transmittance).sum(dim=-2)
        H_pred = rendered_per_ray.sum(dim=-2)                                # [B, n_freq]
        return H_pred
