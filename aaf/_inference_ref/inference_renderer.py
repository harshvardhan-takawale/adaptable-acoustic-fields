"""Vendored INFER renderer — REFERENCE ONLY, DO NOT IMPORT IN PRODUCTION CODE.

Source: project_files/unified_renderers.py (INFER ICML'26 submission)

Vendored:
  - normalize_points              (source L42-44)
  - denormalize_points            (source L46-48)
  - ray_directions                (source L50-70)
  - class AVRRenderFD             (source L208-356)
  - class AVRRenderFD_FreqDep_PhaseCorrection_new (source L716-790, the chosen Phase-1 baseline)

Excluded from this vendor: every other renderer variant in the source file
(time-domain AVRRender, AbsAtt, FreqDep without _new, KK, DistInvar, NoAbsorp,
NAF*, INRAS*). They are reachable via the source path if needed.

The chosen renderer pairs with `AVRModel_complex_FD_FreqDep_PhaseCorrection`
in `inference_model.py`. See `tasks/CHUNK_0_RESULTS.md` for the architectural
walk-through and the rationale for picking this variant (non-KK, non-perceptual).
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers (source: unified_renderers.py L42-70)
# ---------------------------------------------------------------------------

def normalize_points(input_pts, xyz_min, xyz_max):
    """Normalize points to [-1, 1] range."""
    return 2 * (input_pts - xyz_min) / (xyz_max - xyz_min) - 1


def denormalize_points(input_pts, xyz_min, xyz_max):
    """Denormalize points from [-1, 1] to original range."""
    return (input_pts + 1) / 2 * (xyz_max - xyz_min) + xyz_min


def ray_directions(n_azi, n_ele, random_azi=True):
    """Get ray directions for spherical sampling (3D)."""
    azi_ray = torch.linspace(0, np.pi * 2, n_azi + 1)[:-1].cuda()
    azi_randadd = (np.pi * 2 / n_azi) * torch.rand(n_azi).cuda()
    azi_ray = azi_ray + azi_randadd if random_azi else azi_ray

    ele_ray = torch.linspace(0, 1, n_ele + 2)[1:-1].cuda() + (0.5 / n_ele) * torch.rand(n_ele).cuda() * 0
    ele_ray = torch.acos(2 * ele_ray - 1)

    azi_ray, ele_ray = torch.meshgrid(azi_ray, ele_ray, indexing='ij')
    pts_x = torch.mul(torch.cos(azi_ray.flatten()), torch.sin(ele_ray.flatten())).unsqueeze(1)
    pts_y = torch.mul(torch.sin(azi_ray.flatten()), torch.sin(ele_ray.flatten())).unsqueeze(1)
    pts_z = torch.cos(ele_ray.flatten()).unsqueeze(1)

    dir = torch.cat((pts_x, pts_y, pts_z), dim=1)
    dir = torch.cat((dir, torch.tensor([[0, 0, 1], [0, 0, -1]]).cuda()), dim=0)

    return dir, azi_ray, ele_ray


# ---------------------------------------------------------------------------
# Parent class (source: unified_renderers.py L208-356)
# ---------------------------------------------------------------------------

class AVRRenderFD(nn.Module):
    """Standard frequency-domain audio signal rendering (3D, INFER baseline)."""

    def __init__(self, networks_fn, **kwargs) -> None:
        super().__init__()
        self.network_fn = networks_fn
        self.n_samples = kwargs['n_samples']
        self.near = kwargs['near']
        self.far = kwargs['far']
        self.n_azi = kwargs['n_azi']
        self.n_ele = kwargs['n_ele']
        self.speed = kwargs['speed']
        self.fs = kwargs['fs']
        self.pathloss = kwargs['pathloss']
        self.xyz_min = kwargs['xyz_min']
        self.xyz_max = kwargs['xyz_max']

    def forward(self, rays_o, position_tx, direction_tx=None):
        bs = position_tx.size(0)

        dir, _, _ = ray_directions(n_azi=self.n_azi, n_ele=self.n_ele)

        d_vals = torch.linspace(0., 1., self.n_samples).cuda() * (self.far - self.near) + self.near
        ray_pts = rays_o.unsqueeze(1).unsqueeze(2) + (
            dir.unsqueeze(1) * (d_vals.unsqueeze(0).unsqueeze(2))
        ).unsqueeze(0)

        network_pts = normalize_points(ray_pts.reshape(bs, -1, 3), self.xyz_min, self.xyz_max)
        network_view = -1 * dir.unsqueeze(0).unsqueeze(2).expand(ray_pts.size()).reshape(bs, -1, 3)
        network_tx = normalize_points(
            position_tx.unsqueeze(1).expand(*network_pts.size()), self.xyz_min, self.xyz_max
        )

        if direction_tx is not None:
            network_dir_tx = direction_tx.unsqueeze(1).expand(*network_pts.size())
            attn, signal_fd = self.network_fn(network_pts, network_view, network_tx, network_dir_tx)
        else:
            attn, signal_fd = self.network_fn(network_pts, network_view, network_tx)

        if attn.size(-1) > 1:
            attn = attn.view(bs, -1, self.n_samples, attn.size(-1))
        else:
            attn = attn.view(bs, -1, self.n_samples)
        signal_fd_complex = signal_fd.view(bs, -1, self.n_samples, signal_fd.size(-1))

        rendered_signal_fd = self.acoustic_render_fd(
            attn, signal_fd_complex, d_vals, network_pts, network_tx
        )

        receive_sig_fd = torch.sum(rendered_signal_fd, dim=-2)
        receive_sig_fd = self.apply_causal_filter(receive_sig_fd, rays_o, position_tx)

        receive_sig = torch.cat(
            [torch.real(receive_sig_fd).unsqueeze(-1), torch.imag(receive_sig_fd).unsqueeze(-1)],
            dim=-1,
        )
        return receive_sig

    def acoustic_render_fd(self, attn, signal_fd, d_vals, network_pts, network_tx):
        """Frequency-domain volume-rendering integration. Overridden by subclasses."""
        bs, n_rays, n_samples, n_freq_bins = signal_fd.shape

        dists = d_vals[..., 1:] - d_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[..., :1].shape)], -1)
        dists = dists.unsqueeze(0).repeat(n_rays, 1)
        dists = dists.unsqueeze(0).repeat(bs, 1, 1)

        attn_a = torch.abs(attn)
        attn_p = torch.angle(attn)

        if len(attn.shape) == 4:
            raw2alpha = lambda raw, dists: 1. - torch.exp(-raw * dists.unsqueeze(-1))
            raw2phase = lambda raw, dists: raw * dists.unsqueeze(-1)
        else:
            raw2alpha = lambda raw, dists: 1. - torch.exp(-raw * dists)
            raw2phase = lambda raw, dists: raw * dists

        alpha = raw2alpha(attn_a, dists)
        phase = raw2phase(attn_p, dists)

        if len(attn.shape) == 4:
            ones_shape = [bs, n_rays, 1, n_freq_bins]
            att_i = torch.cumprod(
                torch.cat([torch.ones(ones_shape, device=alpha.device), 1. - alpha + 1e-6], dim=-2),
                dim=-2,
            )[..., :-1, :]
            zeros_shape = [bs, n_rays, 1, n_freq_bins]
            phase_i = torch.cumsum(
                torch.cat([torch.zeros(zeros_shape, device=phase.device), phase], dim=-2),
                dim=-2,
            )[..., :-1, :]
        else:
            att_i = torch.cumprod(
                torch.cat([torch.ones((alpha[..., :1].shape)).cuda(), 1. - alpha + 1e-6], -1), -1
            )[..., :-1]
            phase_i = torch.cumsum(
                torch.cat([torch.zeros((alpha[..., :1].shape)).cuda(), phase], -1), -1
            )[..., :-1]

        phase_i = torch.exp(1j * phase_i)

        freq_bins = torch.arange(n_freq_bins, device=signal_fd.device).float()
        pts2rx_delay = d_vals.unsqueeze(0).unsqueeze(0) / self.speed

        if len(attn.shape) == 4:
            pts2rx_phase = torch.exp(-1j * 2 * np.pi * freq_bins * pts2rx_delay.unsqueeze(-1))
        else:
            pts2rx_phase = torch.exp(
                -1j * 2 * np.pi
                * freq_bins.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                * pts2rx_delay.unsqueeze(-1)
            )

        signal_with_delays = signal_fd * pts2rx_phase

        if len(attn.shape) == 4:
            rendered_signal = torch.sum(signal_with_delays * alpha * att_i * phase_i, dim=-2)
        else:
            rendered_signal = torch.sum(
                signal_with_delays * alpha.unsqueeze(-1) * att_i.unsqueeze(-1) * phase_i.unsqueeze(-1),
                dim=-2,
            )

        return rendered_signal

    def apply_causal_filter(self, receive_sig_fd, rays_o, position_tx):
        """Zero out time-domain samples before time-of-flight; re-FFT to freq."""
        bs = receive_sig_fd.size(0)
        n_freq_bins = receive_sig_fd.size(1)

        n_time_samples = 2 * (n_freq_bins - 1)
        time_signal = torch.fft.irfft(receive_sig_fd, n=n_time_samples, dim=1)

        distance = torch.linalg.vector_norm(rays_o - position_tx, dim=1)
        time_of_flight = distance / self.speed
        samples_of_flight = torch.round(time_of_flight * self.fs).long()

        mask = torch.ones(bs, n_time_samples, device=receive_sig_fd.device)
        for i in range(bs):
            if samples_of_flight[i] < n_time_samples:
                mask[i, :samples_of_flight[i]] = 0.0

        masked_time_signal = time_signal * mask
        receive_sig_fd_filtered = torch.fft.rfft(masked_time_signal, dim=1)

        return receive_sig_fd_filtered


# ---------------------------------------------------------------------------
# Target class — Phase-1 starting point
# (source: unified_renderers.py L716-790)
# ---------------------------------------------------------------------------

class AVRRenderFD_FreqDep_PhaseCorrection_new(AVRRenderFD):
    """AVR renderer with frequency-dependent attenuation and configurable
    geometric attenuation. The complex attenuation `attn = σ + jβ` is
    decomposed into absorption (σ ≥ 0) and phase velocity change (β).
    """

    def __init__(self, networks_fn, **kwargs):
        super().__init__(networks_fn, **kwargs)
        self.geometric_attenuation = kwargs.get('geometric_attenuation', False)

    def acoustic_render_fd(self, attn, signal_fd, d_vals, network_pts, network_tx):
        """Frequency-domain rendering with correct complex attenuation handling.

        Implements Equation 7 from the AVR paper in the frequency domain, refined
        per the INFER paper to keep absorption and phase velocity decoupled.
        """
        bs, n_rays, n_samples, n_freq_bins = signal_fd.shape

        dists = d_vals[..., 1:] - d_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[..., :1].shape)], -1)
        dists = dists.unsqueeze(0).unsqueeze(0).expand(bs, n_rays, -1)
        delta_u = dists.unsqueeze(-1)  # [bs, n_rays, n_samples, 1]

        # σ + jβ decomposition: σ ≥ 0 = absorption, β signed = phase velocity change.
        sigma = attn.real.clamp(min=0)
        beta = attn.imag

        # Local opacity from absorption only.
        alpha = 1.0 - torch.exp(-sigma * delta_u)  # [bs, n_rays, n_samples, n_freq_bins]

        # T_amp = ∏(1 - α_i)
        transmittance_amp = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1, :]), 1.0 - alpha[..., :-1, :] + 1e-10], dim=-2),
            dim=-2,
        )

        # P = exp(j ∑ β_i Δu_i)
        phase_increments = beta[..., :-1, :] * delta_u[..., :-1, :]
        cumulative_phase = torch.cumsum(
            torch.cat(
                [torch.zeros_like(phase_increments[..., :1, :]), phase_increments],
                dim=-2,
            ),
            dim=-2,
        )
        transmittance_phase = torch.exp(1j * cumulative_phase)
        transmittance = transmittance_amp * transmittance_phase

        # RFFT frequency axis for geometric phase shift.
        n_time_samples = 2 * (n_freq_bins - 1)
        freq_hz = torch.arange(n_freq_bins, device=signal_fd.device).float() * (
            self.fs / n_time_samples
        )

        pts2rx_delay = d_vals.unsqueeze(0).unsqueeze(0) / self.speed
        geometric_phase = torch.exp(
            -1j * 2 * np.pi * freq_hz.view(1, 1, 1, -1) * pts2rx_delay.unsqueeze(-1)
        )

        if self.geometric_attenuation:
            u = d_vals.view(1, 1, -1, 1)
            geom_amp = 1.0 / torch.clamp(u, min=1e-3)
            signal_with_delays = signal_fd * geometric_phase * geom_amp
        else:
            signal_with_delays = signal_fd * geometric_phase

        rendered_signal = torch.sum(signal_with_delays * alpha * transmittance, dim=-2)
        return rendered_signal
