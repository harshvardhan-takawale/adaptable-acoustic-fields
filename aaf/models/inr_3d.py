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

import math
from typing import Union

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


class INR3D_AutoDecoder(nn.Module):
    """Multi-room shared 3D INR with per-room learnable latent z_s (DeepSDF-style).

    P2-2 analog of `aaf.models.inr_2d.INR2D_AutoDecoder`. Network weights are
    shared across rooms; each training room has its own learnable latent in
    `self.latents` (an `nn.Embedding(n_rooms, latent_dim)`).

    Conditioning (P2-2 D19): FiLM at both sigma and signal branches —
    `feature = γ(z) * feature + β(z)`. γ initialized to 1, β to 0 so the
    untrained FiLM is identity. LoRA-output adapter is supported via
    `conditioning_type='film_lora'` (mirrors Phase 1's Chunk 3.7 D2 design)
    but P2-2 default is `'film'`.

    Geometry head (P2-2 D22, D31): a linear `nn.Linear(latent_dim, 3)`
    predicting (L, W, H) in meters from z. Trained jointly via
    `l_head_weight · L1(predict_geometry(z), [L, W, H]_true)`. The geometry
    head is enabled by default (`l_head_enabled=True`).

    Latent jitter (P2-2 D21): if `latent_jitter_sigma > 0`, additive Gaussian
    noise is injected during training inside `get_latent`. Off at val / test /
    zero-shot.

    For zero-shot adaptation, the caller passes a separately-optimised z_star
    (not looked up from the embedding table) to `forward(..., z_s=z_star)`.

    Forward signature
    -----------------
        attn, signal = model(pts, view, tx, tx_view=None, z_s=z_s)

        pts:     [B, N, 3]
        view:    [B, N, 3]
        tx:      [B, N, 3]
        tx_view: [B, N, 3] or None
        z_s:     [B, latent_dim] (required)

        attn:    [B, N, n_freq_bins] complex64
        signal:  [B, N, n_freq_bins] complex64
    """

    def __init__(
        self,
        n_rooms: int,
        latent_dim: int = 16,                       # D20: 16 in P2-2 (was 8 in Phase 1)
        n_freq_bins: int = 4097,
        hash_grid_config=None,
        mlp_config=None,
        sigma_encoder_dim: int = 256,
        l_head_enabled: bool = True,                # D22/D31: default ON
        conditioning_type: str = "film",            # D19: FiLM
        latent_jitter_sigma: float = 0.1,           # D21
        lora_rank: int = 8,                         # only for film_lora
        cond_source: str = "latent",                # P3-1 D43: latent | geom_fourier | eigen
        cond_dim: Optional[int] = None,             # P3-1: FiLM input width (defaults to latent_dim)
    ):
        super().__init__()
        if n_freq_bins <= 1:
            raise ValueError(f"n_freq_bins must be > 1, got {n_freq_bins}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if conditioning_type not in ("concat", "film", "film_lora"):
            raise ValueError(
                f"conditioning_type must be 'concat', 'film', or 'film_lora', "
                f"got {conditioning_type!r}"
            )
        if float(latent_jitter_sigma) < 0:
            raise ValueError(
                f"latent_jitter_sigma must be >= 0, got {latent_jitter_sigma}"
            )
        if int(lora_rank) <= 0:
            raise ValueError(f"lora_rank must be > 0, got {lora_rank}")
        if cond_source not in ("latent", "geom_fourier", "eigen"):
            raise ValueError(
                f"cond_source must be 'latent', 'geom_fourier', or 'eigen', got {cond_source!r}"
            )

        self.n_rooms = int(n_rooms)
        self.latent_dim = int(latent_dim)
        self.n_freq_bins = int(n_freq_bins)
        self.signal_output_dim = 2 * self.n_freq_bins
        self._n_time_samples = 2 * (self.n_freq_bins - 1)
        self.l_head_enabled = bool(l_head_enabled)
        self.conditioning_type = str(conditioning_type)
        self.latent_jitter_sigma = float(latent_jitter_sigma)
        self.lora_rank = int(lora_rank)
        # P3-1: conditioning arm + FiLM input width (defaults to latent_dim → Phase-2 back-compat).
        self.cond_source = str(cond_source)
        self.cond_dim = int(cond_dim) if cond_dim is not None else self.latent_dim

        hg_cfg = hash_grid_config or _default_hash_grid_config_3d()
        mlp_cfg = mlp_config or _default_mlp_config()

        # Six 3D position/direction encoders (identical to INR3D_Single).
        self._pos_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)
        self._pos_signal_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)
        self._tx_pos_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)
        self._tx_pos_signal_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)
        self._dir_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)
        self._tx_dir_encoding = tcnn.Encoding(3, hg_cfg, dtype=torch.float32)

        # Sigma branch — FiLM modulates the (pos_emb, tx_pos_emb) concat.
        sigma_feat_dim = (
            self._pos_encoding.n_output_dims
            + self._tx_pos_encoding.n_output_dims
        )
        if self.conditioning_type in ("film", "film_lora"):
            sigma_in_dims = sigma_feat_dim
            self.film_sigma = nn.Linear(self.cond_dim, 2 * sigma_feat_dim)
            nn.init.zeros_(self.film_sigma.weight)
            with torch.no_grad():
                self.film_sigma.bias[:sigma_feat_dim].fill_(1.0)         # γ init=1
                self.film_sigma.bias[sigma_feat_dim:].zero_()             # β init=0
        else:
            sigma_in_dims = sigma_feat_dim + self.latent_dim
            self.film_sigma = None

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
        # film_lora: output-side rank-r adapter on the sigma decoder (zero-init).
        if self.conditioning_type == "film_lora":
            r = self.lora_rank
            self.A_sigma = nn.Linear(self.cond_dim, r)
            self.B_sigma = nn.Linear(sigma_encoder_dim, r, bias=False)
            self.proj_sigma = nn.Linear(r, self.signal_output_dim, bias=False)
            nn.init.zeros_(self.proj_sigma.weight)
        else:
            self.A_sigma = self.B_sigma = self.proj_sigma = None

        # Signal branch — FiLM modulates the 5-feature concat
        # [F.relu(sigma_feature), view_emb, tx_view_emb, signal_pos_emb, tx_signal_pos_emb].
        signal_feat_dim = (
            sigma_encoder_dim
            + self._dir_encoding.n_output_dims
            + self._tx_dir_encoding.n_output_dims
            + self._pos_signal_encoding.n_output_dims
            + self._tx_pos_signal_encoding.n_output_dims
        )
        if self.conditioning_type in ("film", "film_lora"):
            n_signal_input = signal_feat_dim
            self.film_signal = nn.Linear(self.cond_dim, 2 * signal_feat_dim)
            nn.init.zeros_(self.film_signal.weight)
            with torch.no_grad():
                self.film_signal.bias[:signal_feat_dim].fill_(1.0)
                self.film_signal.bias[signal_feat_dim:].zero_()
        else:
            n_signal_input = signal_feat_dim + self.latent_dim
            self.film_signal = None
        self._sigma_feat_dim = sigma_feat_dim
        self._signal_feat_dim = signal_feat_dim
        self._model_signal = tcnn.Network(
            n_input_dims=n_signal_input,
            n_output_dims=self.signal_output_dim,
            network_config=mlp_cfg["signal"],
        )
        if self.conditioning_type == "film_lora":
            r = self.lora_rank
            self.A_signal = nn.Linear(self.cond_dim, r)
            self.B_signal = nn.Linear(signal_feat_dim, r, bias=False)
            self.proj_signal = nn.Linear(r, self.signal_output_dim, bias=False)
            nn.init.zeros_(self.proj_signal.weight)
        else:
            self.A_signal = self.B_signal = self.proj_signal = None

        # Per-room latent table — ONLY for the latent arm; G/G+ derive conditioning
        # from geometry, so there is no learnable per-room table (P3-1 D43).
        if self.cond_source == "latent":
            self.latents = nn.Embedding(self.n_rooms, self.latent_dim)
            nn.init.normal_(self.latents.weight, mean=0.0, std=1.0 / math.sqrt(self.latent_dim))
        else:
            self.latents = None

        # Geometry head: linear (L, W, H) predictor (latent arm only; disabled via
        # l_head_enabled=False for G/G+ where geometry is the input).
        if self.l_head_enabled:
            self.l_head = nn.Linear(self.latent_dim, 3)
        else:
            self.l_head = None

        # P3-1 G+ eigenstructure: per-bin resonance modulation at the signal output.
        # w is a single learnable scalar, ZERO-INIT (so at t=0 G+ ≡ G structurally);
        # _R is the per-room resonance map [n_freq_bins], set per render via set_resonance
        # (a plain attribute — NOT a registered/persistent buffer; rebuilt analytically at eval).
        self.w = nn.Parameter(torch.zeros(())) if self.cond_source == "eigen" else None
        self._R = None

    def set_resonance(self, R: Optional[torch.Tensor]) -> None:
        """Set the per-room resonance map (Arm G+). ``R`` is a real tensor of length
        n_freq_bins (0 above the supervised band), or ``None`` to disable. Must be called
        immediately before each render of an eigen-arm room."""
        self._R = R

    @staticmethod
    def _normalize_unit(x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5

    def predict_geometry(self, z_s: torch.Tensor):
        """Predict (L, W, H) in meters from a [B, latent_dim] latent.

        Returns ``[B, 3]`` or ``None`` if no geometry head is wired.
        """
        if self.l_head is None:
            return None
        if z_s.dim() == 1:
            z_s = z_s.unsqueeze(0)
        return self.l_head(z_s)                                              # [B, 3]

    def get_latent(self, room_id: Union[int, torch.Tensor]) -> torch.Tensor:
        """Look up the learnable latent for one or more training rooms.

        Adds N(0, σ²) jitter at training time when ``latent_jitter_sigma > 0``
        (P2-2 D21).
        """
        if isinstance(room_id, int):
            t = torch.tensor(room_id, device=self.latents.weight.device)
            z = self.latents(t)
        else:
            z = self.latents(room_id.to(self.latents.weight.device))
        if self.training and self.latent_jitter_sigma > 0:
            z = z + torch.randn_like(z) * self.latent_jitter_sigma
        return z

    @staticmethod
    def _expand_z_s(z_s: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """Broadcast `[B, latent_dim]` (or `[latent_dim]` → assume B=1) to flat
        `[B*N, latent_dim]` matching the model's flattened (pts, view, tx) input.
        """
        if z_s.dim() == 1:
            z_s = z_s.unsqueeze(0)
        if z_s.size(0) == 1 and B != 1:
            z_s = z_s.expand(B, -1)
        if z_s.size(0) != B:
            raise ValueError(
                f"z_s batch dim {z_s.size(0)} doesn't match pts batch dim {B}"
            )
        return z_s.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

    def forward(
        self,
        pts: torch.Tensor,                                  # [B, N, 3]
        view: torch.Tensor,                                 # [B, N, 3]
        tx: torch.Tensor,                                   # [B, N, 3]
        tx_view=None,
        z_s: torch.Tensor = None,
    ):
        if z_s is None:
            raise ValueError(
                "INR3D_AutoDecoder.forward() requires z_s. Pass either "
                "model.get_latent(room_id) (training) or an externally-optimised "
                "z_star tensor (zero-shot)."
            )
        if pts.shape[-1] != 3 or view.shape[-1] != 3 or tx.shape[-1] != 3:
            raise ValueError(
                f"Expected last dim 3 for pts/view/tx; got "
                f"{pts.shape[-1]}, {view.shape[-1]}, {tx.shape[-1]}"
            )
        if tx_view is None:
            tx_view = torch.zeros_like(tx)

        B = pts.size(0)
        N = pts.size(1)

        pts_flat = self._normalize_unit(pts.reshape(-1, 3))
        view_flat = self._normalize_unit(view.reshape(-1, 3))
        tx_flat = self._normalize_unit(tx.reshape(-1, 3))
        tx_view_flat = self._normalize_unit(tx_view.reshape(-1, 3))
        z_s_flat = self._expand_z_s(z_s, B, N).to(pts_flat.dtype)            # [B*N, latent_dim]

        # ---------- Sigma branch ----------
        pos_emb = self._pos_encoding(pts_flat)
        tx_pos_emb = self._tx_pos_encoding(tx_flat)
        sigma_feat = torch.cat([pos_emb, tx_pos_emb], dim=-1)
        if self.conditioning_type in ("film", "film_lora"):
            gamma_beta = self.film_sigma(z_s_flat)
            gamma_s, beta_s = gamma_beta.chunk(2, dim=-1)
            sigma_input = gamma_s * sigma_feat + beta_s
        else:
            sigma_input = torch.cat([sigma_feat, z_s_flat], dim=-1)
        sigma_feature = self._model_encoder_sigma(sigma_input)

        attn_raw = self._model_decoder_sigma(F.relu(sigma_feature))
        if self.conditioning_type == "film_lora":
            attn_raw = attn_raw + self.proj_sigma(
                self.A_sigma(z_s_flat) * self.B_sigma(F.relu(sigma_feature))
            )
        one_sided = self.n_freq_bins
        attn_real = F.softplus(attn_raw[..., :one_sided]) + 1e-6
        attn_imag = attn_raw[..., one_sided:]
        attn_complex = torch.complex(attn_real, attn_imag)

        # ---------- Signal branch ----------
        view_emb = self._dir_encoding(view_flat)
        tx_view_emb = self._tx_dir_encoding(tx_view_flat)
        signal_pos_emb = self._pos_signal_encoding(pts_flat)
        tx_signal_pos_emb = self._tx_pos_signal_encoding(tx_flat)
        signal_feat = torch.cat(
            [
                F.relu(sigma_feature),
                view_emb,
                tx_view_emb,
                signal_pos_emb,
                tx_signal_pos_emb,
            ],
            dim=-1,
        )
        if self.conditioning_type in ("film", "film_lora"):
            gamma_beta = self.film_signal(z_s_flat)
            gamma_g, beta_g = gamma_beta.chunk(2, dim=-1)
            feature_all = gamma_g * signal_feat + beta_g
        else:
            feature_all = torch.cat([signal_feat, z_s_flat], dim=-1)

        signal_raw = self._model_signal(feature_all)
        if self.conditioning_type == "film_lora":
            signal_raw = signal_raw + self.proj_signal(
                self.A_signal(z_s_flat) * self.B_signal(feature_all)
            )
        signal_re = signal_raw[..., :one_sided]
        signal_im = signal_raw[..., one_sided:].clone()
        signal_im[..., 0] = 0
        if self._n_time_samples % 2 == 0:
            signal_im[..., -1] = 0
        signal_complex = torch.complex(signal_re, signal_im)

        # P3-1 Arm G+ eigenstructure: per-bin resonance modulation h_b·(1 + w·R[b]).
        # w zero-init ⇒ no-op at t=0 (G+ ≡ G). R is 0 above the supervised band, and a
        # real scale preserves the DC/Nyquist imag=0 enforced just above (RFFT symmetry).
        if self.w is not None and self._R is not None:
            scale = (1.0 + self.w * self._R.to(signal_re.dtype))     # real [one_sided]
            signal_complex = signal_complex * scale

        attn_complex = attn_complex.view(B, N, one_sided)
        signal_complex = signal_complex.view(B, N, one_sided)
        return attn_complex, signal_complex
