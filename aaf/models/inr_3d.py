"""INR3D_Single — 3D port of INFER's `AVRModel_complex_FD_FreqDep_PhaseCorrection`.

Mirrors `aaf.models.inr_2d.INR2D_Single` with the only architectural change
being all six tcnn HashGrid encoders going from ``tcnn.Encoding(2, …)`` to
``tcnn.Encoding(3, …)``. The σ + jβ output, softplus on σ, RFFT symmetry mask,
and ``z_s``-as-ignored-stub convention all carry over verbatim.

Chunk P2-1 HashGrid sizing (DECISIONS.md D10, user-approved):
    log2_hashmap_size = 18  (4× Phase-1's 14; restores collision rate parity
                             across the 3D volume)
    n_levels          = 16
    per_level_scale   = 1.38  (finest level ~1700/axis ≈ 3.5 mm in a 6 m room
                              → ≈ λ/2 at 2 kHz)

Forward signature
-----------------
    attn, signal = model(pts, view, tx, tx_view=None, z_s=None)

    pts:     [B, N, 3]
    view:    [B, N, 3]   3D unit-vector ray direction per sample
    tx:      [B, N, 3]   (broadcast — usually one source per batch)
    tx_view: [B, N, 3] or None (omni: model substitutes a zero vector)
    z_s:     IGNORED in INR3D_Single

    attn:    [B, N, n_freq_bins] complex64 (σ + jβ; σ = real, ≥ 0; β = imag)
    signal:  [B, N, n_freq_bins] complex64 (DC and Nyquist imag are zeroed)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn


def _default_hash_grid_config_3d() -> dict:
    """3D HashGrid defaults (DECISIONS.md D10, user-approved override of spec's 16/16/1.5)."""
    return {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 18,
        "base_resolution": 16,
        "per_level_scale": 1.38,
    }


def _default_mlp_config() -> dict:
    """Same MLP cfg as 2D — branch shapes are identical; only input dims change."""
    return {
        "sigma_encoder": {
            "otype": "FullyFusedMLP",
            "n_hidden_layers": 2,
            "n_neurons": 128,
            "activation": "ReLU",
            "output_activation": "None",
        },
        "sigma_decoder": {
            "otype": "CutlassMLP",
            "n_hidden_layers": 3,
            "n_neurons": 128,
            "activation": "ReLU",
            "output_activation": "None",
        },
        "signal": {
            "otype": "CutlassMLP",
            "n_hidden_layers": 3,
            "n_neurons": 512,
            "activation": "ReLU",
            "output_activation": "None",
        },
    }


class INR3D_Single(nn.Module):
    """Single-room 3D INR. ``z_s`` is accepted but ignored (P2-2 subclass uses it)."""

    def __init__(
        self,
        n_freq_bins: int,
        latent_dim: int = 0,                 # ignored in INR3D_Single; kept for API parity
        hash_grid_config: Optional[dict] = None,
        mlp_config: Optional[dict] = None,
        sigma_encoder_dim: int = 256,
    ):
        super().__init__()
        if n_freq_bins <= 1:
            raise ValueError(f"n_freq_bins must be > 1, got {n_freq_bins}")
        self.n_freq_bins = int(n_freq_bins)
        self.latent_dim = int(latent_dim)
        self.signal_output_dim = 2 * self.n_freq_bins
        self._n_time_samples = 2 * (self.n_freq_bins - 1)

        hg_cfg = hash_grid_config or _default_hash_grid_config_3d()
        mlp_cfg = mlp_config or _default_mlp_config()

        # Six 3D position/direction encoders (one per role; matched to INFER ref).
        self._pos_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)
        self._pos_signal_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)
        self._tx_pos_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)
        self._tx_pos_signal_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)
        self._dir_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)
        self._tx_dir_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)

        sigma_in_dims = (
            self._pos_encoding.n_output_dims + self._tx_pos_encoding.n_output_dims
        )
        # P2-2 INJECTION POINT (sigma): widen sigma_in_dims by latent_dim here.

        self._model_encoder_sigma = tcnn.Network(
            n_input_dims=sigma_in_dims,
            n_output_dims=sigma_encoder_dim,
            network_config=mlp_cfg["sigma_encoder"],
        )

        self._model_decoder_sigma = tcnn.Network(
            n_input_dims=sigma_encoder_dim,
            n_output_dims=self.signal_output_dim,
            network_config=mlp_cfg["sigma_decoder"],
        )

        n_signal_input = (
            sigma_encoder_dim
            + self._dir_encoding.n_output_dims
            + self._tx_dir_encoding.n_output_dims
            + self._pos_signal_encoding.n_output_dims
            + self._tx_pos_signal_encoding.n_output_dims
        )
        # P2-2 INJECTION POINT (signal): widen n_signal_input by latent_dim here.

        self._model_signal = tcnn.Network(
            n_input_dims=n_signal_input,
            n_output_dims=self.signal_output_dim,
            network_config=mlp_cfg["signal"],
        )

    @staticmethod
    def _normalize_unit(x: torch.Tensor) -> torch.Tensor:
        """Map values from [-1, 1] to [0, 1] for tcnn.HashGrid input."""
        return (x + 1.0) * 0.5

    def forward(
        self,
        pts: torch.Tensor,                # [B, N, 3]
        view: torch.Tensor,               # [B, N, 3]
        tx: torch.Tensor,                 # [B, N, 3]
        tx_view: Optional[torch.Tensor] = None,
        z_s: Optional[torch.Tensor] = None,  # IGNORED in INR3D_Single
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del z_s  # ignored stub for API parity with INR3D_AutoDecoder (P2-2)

        if pts.shape[-1] != 3 or view.shape[-1] != 3 or tx.shape[-1] != 3:
            raise ValueError(
                f"Expected last dim 3 for pts/view/tx; got "
                f"{pts.shape[-1]}, {view.shape[-1]}, {tx.shape[-1]}"
            )
        if tx_view is None:
            tx_view = torch.zeros_like(tx)
        elif tx_view.shape[-1] != 3:
            raise ValueError(f"Expected tx_view last dim 3, got {tx_view.shape[-1]}")

        B = pts.size(0)
        N = pts.size(1)

        # tcnn expects flat [B*N, 3] inputs in [0, 1].
        pts_flat = self._normalize_unit(pts.reshape(-1, 3))
        view_flat = self._normalize_unit(view.reshape(-1, 3))
        tx_flat = self._normalize_unit(tx.reshape(-1, 3))
        tx_view_flat = self._normalize_unit(tx_view.reshape(-1, 3))

        # Sigma branch.
        pos_emb = self._pos_encoding(pts_flat)
        tx_pos_emb = self._tx_pos_encoding(tx_flat)
        # P2-2 INJECTION POINT (sigma): concat z_s_broadcast with [pos_emb, tx_pos_emb] here.
        sigma_feature = self._model_encoder_sigma(torch.cat([pos_emb, tx_pos_emb], dim=-1))

        attn_raw = self._model_decoder_sigma(F.relu(sigma_feature))  # [BN, 2*n_freq_bins]

        one_sided = self.n_freq_bins
        attn_real = attn_raw[..., :one_sided]
        attn_imag = attn_raw[..., one_sided:]
        # Enforce σ ≥ 0 (physical absorption).
        attn_real = F.softplus(attn_real) + 1e-6
        attn_complex = torch.complex(attn_real, attn_imag)

        # Signal branch.
        view_emb = self._dir_encoding(view_flat)
        tx_view_emb = self._tx_dir_encoding(tx_view_flat)
        signal_pos_emb = self._pos_signal_encoding(pts_flat)
        tx_signal_pos_emb = self._tx_pos_signal_encoding(tx_flat)
        # P2-2 INJECTION POINT (signal): also concat z_s_broadcast here.
        feature_all = torch.cat(
            [F.relu(sigma_feature), view_emb, tx_view_emb, signal_pos_emb, tx_signal_pos_emb],
            dim=-1,
        )

        signal_raw = self._model_signal(feature_all)
        signal_re = signal_raw[..., :one_sided]
        signal_im = signal_raw[..., one_sided:]

        # RFFT symmetry: zero the imaginary part of DC (and Nyquist if even-length).
        signal_im = signal_im.clone()
        signal_im[..., 0] = 0
        if self._n_time_samples % 2 == 0:
            signal_im[..., -1] = 0

        signal_complex = torch.complex(signal_re, signal_im)

        attn_complex = attn_complex.view(B, N, one_sided)
        signal_complex = signal_complex.view(B, N, one_sided)
        return attn_complex, signal_complex
