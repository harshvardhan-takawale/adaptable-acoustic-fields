"""P4-9 2c (D180 NMF-LB; D181 A10, E4): the neural modal field -- room modes predicted from the room's tokens and summed
by m1's modal formula. No ray renderer (Rule 10 untouched).

  room code c      R0 @120K's per-token encoder (`segment_encoder`, initialised from the checkpoint), pooled by the
                   EXTENT-WEIGHTED SUM over valid tokens (R0's position-independent base): one 64-d code per room.
  eigenfrequencies k_1 = softplus(a_1), k_n = k_{n-1} + softplus(a_n), a = MLP(c): sorted by construction (rad / m).
  mode shapes      phi(x; c) in R^24: a wavelength-matched hash grid on x / world_scale (MB1's low band: 6 levels, coarsest
                   cell 1 m, finest 12.4 cm -- A10) -> an MLP whose hidden layers are FiLM-modulated by c (zero-init
                   FiLM: identity at init). Each mode is normalised PER ROOM on that room's fixed quadrature points
                   (sum_q w_q phi_n(x_q)^2 = 1, m1's mu-weighted measure).
  damping          eta_n = sum_w w_w phi_n(x_w)^2 over the room's wall nodes (m1's first-order Robin rule, beta built into
                   the weights).
  field            H(x, f) = C [ sum_n phi_n(x) phi_n(x0) / (k_n^2 - k^2 + j k eta_n) ] S(f) e^{j w dt}
                            + psi_0(x) td0(f) C / c_pred        (the exact zero-mode term, from geometry, as m1)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MODES = 24
TOKEN_DIM = 26
N_TOK = 12
EMB = 64


def mb1_lowband_grid(world_scale: float = 10.0, base_cell_m: float = 1.0, f_top: float = 173.0,
                     cells_per_wavelength: float = 16.0, n_levels: int = 6, log2_hashmap_size: int = 18) -> dict:
    """MB1's band-1 grid (`multi_room_2d_mat.multiband_kwargs`, mode "matched"): finest cell c / (16 f_top) = 12.4 cm."""
    br = int(round(world_scale / base_cell_m + 1.0))
    cell = 343.0 / (cells_per_wavelength * f_top)
    s = ((world_scale / cell + 1.0) / br) ** (1.0 / (n_levels - 1))
    return dict(otype="HashGrid", n_levels=n_levels, n_features_per_level=2, log2_hashmap_size=log2_hashmap_size,
                base_resolution=br, per_level_scale=float(s))


class FiLMMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, d_cond, n_hidden=2):
        super().__init__()
        self.lin = nn.ModuleList([nn.Linear(d_in if i == 0 else d_hidden, d_hidden) for i in range(n_hidden)])
        self.film = nn.ModuleList([nn.Linear(d_cond, 2 * d_hidden) for _ in range(n_hidden)])
        for f in self.film:                                  # identity at init
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, x, c):
        h = x
        for lin, film in zip(self.lin, self.film):
            g, b = film(c).chunk(2, dim=-1)
            h = F.softplus(lin(h) * (1.0 + g) + b, beta=10.0)  # smooth activation (E4's alternative stays open)
        return self.out(h)


class NMF2D(nn.Module):
    def __init__(self, world_scale: float = 10.0, n_modes: int = N_MODES, hidden: int = 128):
        super().__init__()
        import tinycudann as tcnn
        self.world_scale = float(world_scale)
        self.n_modes = int(n_modes)
        self.segment_encoder = nn.Sequential(nn.Linear(TOKEN_DIM, EMB), nn.ReLU(), nn.Linear(EMB, EMB))
        self.k_head = nn.Sequential(nn.Linear(EMB, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, n_modes))
        # init: k_1 ~ 0.4 rad/m (~22 Hz) and increments ~ 0.12 rad/m (a ~30 m^2 room's mean mode spacing at 100 Hz)
        nn.init.zeros_(self.k_head[-1].weight)
        with torch.no_grad():
            self.k_head[-1].bias.fill_(math.log(math.expm1(0.12)))
            self.k_head[-1].bias[0] = math.log(math.expm1(0.40))
        self.grid_cfg = mb1_lowband_grid(world_scale)
        self.encoding = tcnn.Encoding(2, self.grid_cfg, dtype=torch.float32)
        self.decoder = FiLMMLP(self.encoding.n_output_dims, hidden, n_modes, EMB)
        from aaf.models.conditioning_2d import TOK_EXTENT_IDX
        self._tok_extent_idx = TOK_EXTENT_IDX

    # ---- room code
    def room_code(self, z_s: torch.Tensor) -> torch.Tensor:
        """z_s [R, 324] geom_token vectors -> [R, 64]: extent-weighted sum of the token embeddings (R0's base)."""
        t = z_s[:, :N_TOK * TOKEN_DIM].reshape(-1, N_TOK, TOKEN_DIM)
        m = z_s[:, N_TOK * TOKEN_DIM:N_TOK * TOKEN_DIM + N_TOK].reshape(-1, N_TOK, 1)
        w = t[..., self._tok_extent_idx:self._tok_extent_idx + 1] * m
        return (self.segment_encoder(t) * w).sum(dim=1)

    def k_modes(self, c: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(F.softplus(self.k_head(c)), dim=-1)                       # [R, 24] sorted

    def phi_raw(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x [N, 2] (m) of ONE room, c [64] -> [N, 24] (unnormalised)."""
        e = self.encoding((x / self.world_scale).clamp(0.0, 1.0)).float()
        return self.decoder(e, c.view(1, -1).expand(len(x), -1))

    def norms(self, xq: torch.Tensor, wq: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """per-mode normalisation sqrt(sum_q w_q phi^2) on the room's quadrature points -> [24]."""
        p = self.phi_raw(xq, c)
        return torch.sqrt((wq[:, None] * p ** 2).sum(0).clamp(min=1e-12))

    @staticmethod
    def field(k: torch.Tensor, eta: torch.Tensor, phi_x: torch.Tensor, phi_s: torch.Tensor, bins: torch.Tensor,
              S: torch.Tensor, C: complex, zero: torch.Tensor, df: float = 0.5, c0: float = 343.0,
              dt: float = 1.0 / 30720.0) -> torch.Tensor:
        """m1's formula. k, eta [24]; phi_x [N, 24]; phi_s [24]; bins [B] (int); S [B] complex (pulse spectrum);
        zero [N, B] complex (the zero-mode term, already scaled) -> [N, B] complex."""
        f = bins.to(torch.float64) * df
        kk = (2 * math.pi * f / c0).to(torch.float64)
        den = (k.double() ** 2)[:, None] - kk[None] ** 2 + 1j * kk[None] * eta.double()[:, None]      # [24, B]
        G = phi_x.double().to(torch.complex128) @ (phi_s.double()[:, None] / den)                    # [N, B]
        G = G * (S * torch.exp(1j * 2 * math.pi * f * dt))[None]
        return (C * G + zero).to(torch.complex64)


def soft_weights(k: torch.Tensor, k_a: float, k_b: float) -> torch.Tensor:
    """1 below k_a, cosine roll-off to 0 at k_b: the projector's soft cutoff (A10), continuous when modes cross."""
    t = ((k - k_a) / max(k_b - k_a, 1e-9)).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * t))


def soft_sqrt_weights(k: torch.Tensor, k_a: float, k_b: float) -> torch.Tensor:
    """sqrt of `soft_weights`, computed DIRECTLY: cos(pi t / 2), whose square is (1 + cos(pi t)) / 2 exactly. Taking
    torch.sqrt of the weights instead is a NaN trap (D186): just below k_b float32 rounds cos(pi t) to -1, the weight is
    exactly 0 while t < 1 (so the clamp does not mask the gradient), and sqrt's backward is g / 0 = inf."""
    t = ((k - k_a) / max(k_b - k_a, 1e-9)).clamp(0.0, 1.0)
    return torch.cos(0.5 * math.pi * t)


def subspace_loss(phi_hat: torch.Tensor, s_hat: torch.Tensor, psi: torch.Tensor, s_true: torch.Tensor,
                  wq: torch.Tensor) -> torch.Tensor:
    """||P_hat - P||_F^2 / ||P||_F^2 with P = sum_n w_n psi_n psi_n^T in the quadrature inner product:
    ||A A^T - B B^T||^2 = ||A^T A||^2 + ||B^T B||^2 - 2 ||A^T B||^2, A = sqrt(wq) phi_hat s_hat.
    s_hat, s_true are the SQRT weights (`soft_sqrt_weights`), never torch.sqrt of `soft_weights` (D186)."""
    s = torch.sqrt(wq)[:, None]
    A = s * phi_hat * s_hat[None]
    B = s * psi * s_true[None]
    AA, BB, AB = A.T @ A, B.T @ B, A.T @ B
    return ((AA ** 2).sum() + (BB ** 2).sum() - 2 * (AB ** 2).sum()) / (BB ** 2).sum().clamp(min=1e-12)


def orthonormality_loss(phi_hat: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    G = phi_hat.T @ (wq[:, None] * phi_hat)
    return ((G - torch.eye(G.shape[0], device=G.device, dtype=G.dtype)) ** 2).sum() / G.shape[0]


def hungarian_k_mae(k_hat: np.ndarray, k_true: np.ndarray, n: int = 10) -> float:
    """eigenfrequency MAE (Hz) of the first n modes, matched by the Hungarian algorithm on |f_hat - f|."""
    from scipy.optimize import linear_sum_assignment
    fh, ft = 343.0 * k_hat / (2 * np.pi), 343.0 * k_true[:n] / (2 * np.pi)
    cost = np.abs(fh[:, None] - ft[None])
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def subspace_overlap(phi_hat: np.ndarray, psi: np.ndarray, wq: np.ndarray, n: int = 10) -> float:
    """mean squared canonical correlation between the first n predicted and true modes (1 = same span)."""
    s = np.sqrt(wq)[:, None]
    qa, _ = np.linalg.qr(s * phi_hat[:, :n])
    qb, _ = np.linalg.qr(s * psi[:, :n])
    sv = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(sv ** 2))
