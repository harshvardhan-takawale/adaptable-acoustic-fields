"""INR2D_Single — 2D port of INFER's `AVRModel_complex_FD_FreqDep_PhaseCorrection`.

Architecture
------------
Six tcnn HashGrid encoders (all 2D), two MLPs in the sigma branch, one MLP in
the signal branch. Returns complex per-frequency attenuation (σ + jβ) and
complex per-frequency emission spectrum.

For Chunk 2 (single-room overfit) the `z_s` argument is accepted but ignored.
The Chunk-3 subclass `INR2D_AutoDecoder` will inject the latent at the sigma
and signal branch concat points.

Forward signature
-----------------
    attn, signal = model(pts, view, tx, tx_view=None, z_s=None)

    pts:     [B, N, 2]
    view:    [B, N, 2]   (cos θ, sin θ) per sample
    tx:      [B, N, 2]   (broadcast — usually one source per batch)
    tx_view: [B, N, 2] or None (omni: model substitutes a zero vector)
    z_s:     IGNORED in INR2D_Single

    attn:    [B, N, n_freq_bins] complex64 (σ + jβ; σ = real, ≥ 0; β = imag)
    signal:  [B, N, n_freq_bins] complex64 (DC and Nyquist imag are zeroed)
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn


def _default_hash_grid_config() -> dict:
    return {
        "otype": "HashGrid",
        "n_levels": 20,
        "n_features_per_level": 2,
        "log2_hashmap_size": 18,
        "base_resolution": 16,
        "per_level_scale": 1.5,
    }


def _default_mlp_config() -> dict:
    return {
        "sigma_encoder": {
            "otype": "FullyFusedMLP",
            "n_hidden_layers": 2,
            "n_neurons": 128,
            "activation": "ReLU",
            "output_activation": "None",
        },
        # CutlassMLP for the wide outputs (2*n_freq_bins ≈ 8194 at fs=4096, n_time=8192).
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


class INR2D_Single(nn.Module):
    """Single-room 2D INR. ``z_s`` is accepted but ignored (Chunk-3 subclass uses it)."""

    def __init__(
        self,
        n_freq_bins: int,
        latent_dim: int = 0,                 # ignored in INR2D_Single; kept for API parity
        hash_grid_config: Optional[dict] = None,
        mlp_config: Optional[dict] = None,
        sigma_encoder_dim: int = 256,
    ):
        super().__init__()
        if n_freq_bins <= 1:
            raise ValueError(f"n_freq_bins must be > 1, got {n_freq_bins}")
        self.n_freq_bins = int(n_freq_bins)
        self.latent_dim = int(latent_dim)
        self.signal_output_dim = 2 * self.n_freq_bins  # split into [real, imag]
        # n_time_samples is implicit: 2*(n_freq_bins-1). Used for RFFT-symmetry mask parity.
        self._n_time_samples = 2 * (self.n_freq_bins - 1)

        hg_cfg = hash_grid_config or _default_hash_grid_config()
        mlp_cfg = mlp_config or _default_mlp_config()

        # 2D vs 3D: tcnn.Encoding(2, ...) instead of (3, ...). All six encoders are 2D.
        self._pos_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._pos_signal_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._tx_pos_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._tx_pos_signal_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._dir_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._tx_dir_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)

        sigma_in_dims = (
            self._pos_encoding.n_output_dims + self._tx_pos_encoding.n_output_dims
        )
        # Auto-decoder candidate A (Chunk 3): widen sigma_in_dims by latent_dim here.

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
        # Auto-decoder candidate A (Chunk 3): widen n_signal_input by latent_dim here.

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
        pts: torch.Tensor,                # [B, N, 2]
        view: torch.Tensor,               # [B, N, 2]
        tx: torch.Tensor,                 # [B, N, 2]
        tx_view: Optional[torch.Tensor] = None,
        z_s: Optional[torch.Tensor] = None,  # IGNORED in INR2D_Single
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # z_s ignored in INR2D_Single; subclass INR2D_AutoDecoder (Chunk 3) will use it.
        del z_s  # explicit no-op so pyflakes is happy

        if pts.shape[-1] != 2 or view.shape[-1] != 2 or tx.shape[-1] != 2:
            raise ValueError(
                f"Expected last dim 2 for pts/view/tx; got "
                f"{pts.shape[-1]}, {view.shape[-1]}, {tx.shape[-1]}"
            )
        if tx_view is None:
            tx_view = torch.zeros_like(tx)
        elif tx_view.shape[-1] != 2:
            raise ValueError(f"Expected tx_view last dim 2, got {tx_view.shape[-1]}")

        B = pts.size(0)
        N = pts.size(1)

        # tcnn expects flat [B*N, 2] inputs in [0, 1].
        pts_flat = self._normalize_unit(pts.reshape(-1, 2))
        view_flat = self._normalize_unit(view.reshape(-1, 2))
        tx_flat = self._normalize_unit(tx.reshape(-1, 2))
        tx_view_flat = self._normalize_unit(tx_view.reshape(-1, 2))

        # Sigma branch.
        pos_emb = self._pos_encoding(pts_flat)
        tx_pos_emb = self._tx_pos_encoding(tx_flat)
        # CHUNK-3 INJECTION POINT (sigma): concat z_s_broadcast with [pos_emb, tx_pos_emb] here.
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
        # CHUNK-3 INJECTION POINT (signal): also concat z_s_broadcast here.
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


class INR2D_AutoDecoder(nn.Module):
    """Multi-room shared INR with per-room learnable latent z_s (DeepSDF-style).

    Network weights are shared across rooms. Each training room has its own
    learnable latent in ``self.latents`` (an ``nn.Embedding(n_rooms, latent_dim)``).
    z_s is injected by concatenation at BOTH the sigma and signal branch concat
    points (candidate A from CHUNK_0_RESULTS.md §7).

    For zero-shot adaptation, the caller passes a separately-optimised z_s* (not
    looked up from the embedding table) to ``forward(... z_s=z_star)``.
    """

    def __init__(
        self,
        n_rooms: int,
        latent_dim: int = 32,
        n_freq_bins: int = 4097,
        hash_grid_config: Optional[dict] = None,
        mlp_config: Optional[dict] = None,
        sigma_encoder_dim: int = 256,
    ):
        super().__init__()
        if n_freq_bins <= 1:
            raise ValueError(f"n_freq_bins must be > 1, got {n_freq_bins}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0 in INR2D_AutoDecoder, got {latent_dim}")
        self.n_rooms = int(n_rooms)
        self.latent_dim = int(latent_dim)
        self.n_freq_bins = int(n_freq_bins)
        self.signal_output_dim = 2 * self.n_freq_bins
        self._n_time_samples = 2 * (self.n_freq_bins - 1)

        hg_cfg = hash_grid_config or _default_hash_grid_config()
        mlp_cfg = mlp_config or _default_mlp_config()

        # Six 2D position/direction encoders (identical to INR2D_Single).
        self._pos_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._pos_signal_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._tx_pos_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._tx_pos_signal_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._dir_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._tx_dir_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)

        # Sigma branch widened by latent_dim.
        sigma_in_dims = (
            self._pos_encoding.n_output_dims
            + self._tx_pos_encoding.n_output_dims
            + self.latent_dim
        )
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

        # Signal branch widened by latent_dim.
        n_signal_input = (
            sigma_encoder_dim
            + self._dir_encoding.n_output_dims
            + self._tx_dir_encoding.n_output_dims
            + self._pos_signal_encoding.n_output_dims
            + self._tx_pos_signal_encoding.n_output_dims
            + self.latent_dim
        )
        self._model_signal = tcnn.Network(
            n_input_dims=n_signal_input,
            n_output_dims=self.signal_output_dim,
            network_config=mlp_cfg["signal"],
        )

        # Per-room latent table (DeepSDF-style auto-decoder).
        self.latents = nn.Embedding(self.n_rooms, self.latent_dim)
        nn.init.normal_(self.latents.weight, mean=0.0, std=1.0 / math.sqrt(self.latent_dim))

    @staticmethod
    def _normalize_unit(x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5

    def get_latent(self, room_id: Union[int, torch.Tensor]) -> torch.Tensor:
        """Look up the learnable latent for one or more training rooms.

        ``room_id`` may be a Python int or a 1-D tensor of ints. Returns shape
        ``[latent_dim]`` for a scalar id or ``[B, latent_dim]`` for a 1-D batch.
        """
        if isinstance(room_id, int):
            t = torch.tensor(room_id, device=self.latents.weight.device)
            return self.latents(t)
        return self.latents(room_id.to(self.latents.weight.device))

    @staticmethod
    def _expand_z_s(z_s: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """Broadcast z_s of shape [B, latent_dim] (or [latent_dim] → assume B=1) to
        flat [B*N, latent_dim] matching the model's flattened (pts, view, tx) inputs.
        """
        if z_s.dim() == 1:
            z_s = z_s.unsqueeze(0)              # [1, latent_dim]
        if z_s.size(0) == 1 and B != 1:
            z_s = z_s.expand(B, -1)             # [B, latent_dim]
        if z_s.size(0) != B:
            raise ValueError(
                f"z_s batch dim {z_s.size(0)} doesn't match pts batch dim {B}"
            )
        # Repeat across N points within each batch element.
        return z_s.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

    def forward(
        self,
        pts: torch.Tensor,                  # [B, N, 2]
        view: torch.Tensor,                 # [B, N, 2]
        tx: torch.Tensor,                   # [B, N, 2]
        tx_view: Optional[torch.Tensor] = None,
        z_s: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z_s is None:
            raise ValueError(
                "INR2D_AutoDecoder.forward() requires z_s. Pass either "
                "model.get_latent(room_id) (training) or an externally-optimised "
                "z_star tensor (zero-shot)."
            )
        if pts.shape[-1] != 2 or view.shape[-1] != 2 or tx.shape[-1] != 2:
            raise ValueError(
                f"Expected last dim 2 for pts/view/tx; got "
                f"{pts.shape[-1]}, {view.shape[-1]}, {tx.shape[-1]}"
            )
        if tx_view is None:
            tx_view = torch.zeros_like(tx)

        B = pts.size(0)
        N = pts.size(1)

        pts_flat = self._normalize_unit(pts.reshape(-1, 2))
        view_flat = self._normalize_unit(view.reshape(-1, 2))
        tx_flat = self._normalize_unit(tx.reshape(-1, 2))
        tx_view_flat = self._normalize_unit(tx_view.reshape(-1, 2))
        z_s_flat = self._expand_z_s(z_s, B, N).to(pts_flat.dtype)  # [B*N, latent_dim]

        # Sigma branch — concat z_s with [pos_emb, tx_pos_emb].
        pos_emb = self._pos_encoding(pts_flat)
        tx_pos_emb = self._tx_pos_encoding(tx_flat)
        sigma_feature = self._model_encoder_sigma(
            torch.cat([pos_emb, tx_pos_emb, z_s_flat], dim=-1)
        )

        attn_raw = self._model_decoder_sigma(F.relu(sigma_feature))
        one_sided = self.n_freq_bins
        attn_real = F.softplus(attn_raw[..., :one_sided]) + 1e-6
        attn_imag = attn_raw[..., one_sided:]
        attn_complex = torch.complex(attn_real, attn_imag)

        # Signal branch — concat z_s with the existing five encodings.
        view_emb = self._dir_encoding(view_flat)
        tx_view_emb = self._tx_dir_encoding(tx_view_flat)
        signal_pos_emb = self._pos_signal_encoding(pts_flat)
        tx_signal_pos_emb = self._tx_pos_signal_encoding(tx_flat)
        feature_all = torch.cat(
            [
                F.relu(sigma_feature),
                view_emb,
                tx_view_emb,
                signal_pos_emb,
                tx_signal_pos_emb,
                z_s_flat,
            ],
            dim=-1,
        )

        signal_raw = self._model_signal(feature_all)
        signal_re = signal_raw[..., :one_sided]
        signal_im = signal_raw[..., one_sided:].clone()
        # RFFT symmetry: zero the imaginary part of DC (and Nyquist if even-length).
        signal_im[..., 0] = 0
        if self._n_time_samples % 2 == 0:
            signal_im[..., -1] = 0
        signal_complex = torch.complex(signal_re, signal_im)

        attn_complex = attn_complex.view(B, N, one_sided)
        signal_complex = signal_complex.view(B, N, one_sided)
        return attn_complex, signal_complex
