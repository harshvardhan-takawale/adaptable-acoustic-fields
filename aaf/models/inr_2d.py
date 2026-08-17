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
        l_head_enabled: bool = False,
        l_head_arch: str = "mlp_32",
        conditioning_type: str = "concat",
        latent_jitter_sigma: float = 0.0,
        lora_rank: int = 8,
        cond_source: str = "latent",
        cond_dim: Optional[int] = None,
        l_head_out_dim: int = 1,
    ):
        super().__init__()
        if n_freq_bins <= 1:
            raise ValueError(f"n_freq_bins must be > 1, got {n_freq_bins}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0 in INR2D_AutoDecoder, got {latent_dim}")
        if cond_source not in ("latent", "geom_alpha_fourier", "m_linear", "m_segment"):
            raise ValueError(
                f"cond_source must be 'latent', 'geom_alpha_fourier', 'm_linear' or "
                f"'m_segment', got {cond_source!r}"
            )
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
        self.n_rooms = int(n_rooms)
        self.latent_dim = int(latent_dim)
        # P3-2: conditioning may be an analytic feature vector instead of a latent.
        # cond_dim defaults to latent_dim so every pre-P3-2 config builds byte-identical
        # layers (same in_features -> same RNG draws -> same init).
        self.cond_source = str(cond_source)
        self.cond_dim = int(cond_dim) if cond_dim is not None else self.latent_dim
        self.l_head_out_dim = int(l_head_out_dim)
        self.n_freq_bins = int(n_freq_bins)
        self.signal_output_dim = 2 * self.n_freq_bins
        self._n_time_samples = 2 * (self.n_freq_bins - 1)
        self.l_head_enabled = bool(l_head_enabled)
        self.conditioning_type = str(conditioning_type)
        self.latent_jitter_sigma = float(latent_jitter_sigma)
        self.lora_rank = int(lora_rank)

        hg_cfg = hash_grid_config or _default_hash_grid_config()
        mlp_cfg = mlp_config or _default_mlp_config()

        # Six 2D position/direction encoders (identical to INR2D_Single).
        self._pos_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._pos_signal_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._tx_pos_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._tx_pos_signal_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._dir_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)
        self._tx_dir_encoding = tcnn.Encoding(2, hg_cfg, dtype=torch.float32)

        # Sigma branch — concat path widens by latent_dim; FiLM path drops z_s
        # from the cat and modulates the encoded feature instead.
        # Sigma encoded feature (without z_s): [pos_emb, tx_pos_emb].
        sigma_feat_dim = (
            self._pos_encoding.n_output_dims
            + self._tx_pos_encoding.n_output_dims
        )
        if self.conditioning_type in ("film", "film_lora"):
            sigma_in_dims = sigma_feat_dim
            # Linear(d -> 2*F) split into (gamma, beta), each F-dim. gamma is
            # initialised to 1 (no-op) and beta to 0 so untrained FiLM is identity.
            self.film_sigma = nn.Linear(self.cond_dim, 2 * sigma_feat_dim)
            nn.init.zeros_(self.film_sigma.weight)
            with torch.no_grad():
                self.film_sigma.bias[:sigma_feat_dim].fill_(1.0)         # gamma_init=1
                self.film_sigma.bias[sigma_feat_dim:].zero_()             # beta_init=0
        else:
            sigma_in_dims = sigma_feat_dim + self.cond_dim
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
        # film_lora: output-side rank-r additive adapter on the sigma decoder.
        # adapter(z, feat) = proj(A(z) * B(feat))  with zero-init on `proj` so
        # at construction the adapter contributes exactly 0 and behaviour matches
        # plain FiLM. The hidden rank `r = self.lora_rank` (default 8).
        if self.conditioning_type == "film_lora":
            r = self.lora_rank
            self.A_sigma = nn.Linear(self.cond_dim, r)
            self.B_sigma = nn.Linear(sigma_encoder_dim, r, bias=False)
            self.proj_sigma = nn.Linear(r, self.signal_output_dim, bias=False)
            nn.init.zeros_(self.proj_sigma.weight)
        else:
            self.A_sigma = self.B_sigma = self.proj_sigma = None

        # Signal branch — same scheme. Encoded feature (without z_s):
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
            n_signal_input = signal_feat_dim + self.cond_dim
            self.film_signal = None
        self._sigma_feat_dim = sigma_feat_dim
        self._signal_feat_dim = signal_feat_dim
        self._model_signal = tcnn.Network(
            n_input_dims=n_signal_input,
            n_output_dims=self.signal_output_dim,
            network_config=mlp_cfg["signal"],
        )
        # film_lora: output-side rank-r additive adapter on the signal MLP.
        if self.conditioning_type == "film_lora":
            r = self.lora_rank
            self.A_signal = nn.Linear(self.cond_dim, r)
            self.B_signal = nn.Linear(signal_feat_dim, r, bias=False)
            self.proj_signal = nn.Linear(r, self.signal_output_dim, bias=False)
            nn.init.zeros_(self.proj_signal.weight)
        else:
            self.A_signal = self.B_signal = self.proj_signal = None

        # Per-room latent table (DeepSDF-style auto-decoder). Absent for analytic
        # conditioning arms, where the conditioning vector IS the physical parameters.
        if self.cond_source == "latent":
            self.latents = nn.Embedding(self.n_rooms, self.latent_dim)
            nn.init.normal_(self.latents.weight, mean=0.0,
                            std=1.0 / math.sqrt(self.latent_dim))
        else:
            self.latents = None

        # Optional auxiliary L-prediction head (Chunk-3.5): adds an inductive bias
        # forcing z_s to encode room length L, mitigating Chunk-3's latent-collapse
        # failure mode. The head is consulted only when the trainer's
        # cfg.l_head_weight > 0 (loss term), or as a sanity check at zero-shot time.
        # Two architectures supported (Chunk-3.5+):
        #   "mlp_32": Linear(d→32) → ReLU → Linear(32→1) — expressive (R0-R5 default).
        #   "linear": Linear(d→1) — forces z_s to be linearly readable as L,
        #             which is the strongest inductive bias toward a 1-D manifold
        #             (R6-R8).
        self.l_head_arch = str(l_head_arch)
        if self.cond_source != "latent":
            # The geometry IS the input; a head predicting it back is a trivial identity.
            self.l_head = None
        elif self.l_head_enabled:
            if self.l_head_arch == "mlp_32":
                self.l_head = nn.Sequential(
                    nn.Linear(self.latent_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.l_head_out_dim),
                )
            elif self.l_head_arch == "linear":
                self.l_head = nn.Linear(self.latent_dim, self.l_head_out_dim)
            else:
                raise ValueError(
                    f"Unknown l_head_arch={self.l_head_arch!r}; "
                    "must be 'mlp_32' or 'linear'."
                )
        else:
            self.l_head = None

    @staticmethod
    def _normalize_unit(x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5

    def predict_L(self, z_s: torch.Tensor) -> Optional[torch.Tensor]:
        """Predict L (m) from a [B, latent_dim] latent.

        Returns ``[B]`` predicted L values, or ``None`` if no L-head is wired.
        Consumed by the trainer when ``cfg.l_head_weight > 0`` and (optionally)
        by zero-shot eval as a physical-meaning sanity check.
        """
        if self.l_head is None:
            return None
        if z_s.dim() == 1:
            z_s = z_s.unsqueeze(0)
        return self.l_head(z_s).squeeze(-1)

    def get_latent(self, room_id: Union[int, torch.Tensor]) -> torch.Tensor:
        """Look up the learnable latent for one or more training rooms.

        ``room_id`` may be a Python int or a 1-D tensor of ints. Returns shape
        ``[latent_dim]`` for a scalar id or ``[B, latent_dim]`` for a 1-D batch.

        When ``self.training`` AND ``self.latent_jitter_sigma > 0`` (Chunk 3.6
        Variant C2), additive Gaussian noise is injected — this smooths the
        loss landscape around each trained z_s so zero-shot adaptation has an
        easier time navigating the latent-to-spectrum response surface.
        """
        if self.latents is None:
            raise RuntimeError(
                "get_latent() requires cond_source='latent'; this model was built "
                f"with cond_source={self.cond_source!r} and has no latent table."
            )
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

        # Sigma branch.
        pos_emb = self._pos_encoding(pts_flat)
        tx_pos_emb = self._tx_pos_encoding(tx_flat)
        sigma_feat = torch.cat([pos_emb, tx_pos_emb], dim=-1)       # [B*N, sigma_feat_dim]
        if self.conditioning_type in ("film", "film_lora"):
            gamma_beta = self.film_sigma(z_s_flat)
            gamma_s, beta_s = gamma_beta.chunk(2, dim=-1)
            sigma_input = gamma_s * sigma_feat + beta_s
        else:
            sigma_input = torch.cat([sigma_feat, z_s_flat], dim=-1)
        sigma_feature = self._model_encoder_sigma(sigma_input)

        attn_raw = self._model_decoder_sigma(F.relu(sigma_feature))
        # film_lora: output-side rank-r additive z-gated correction. Zero-init
        # on `proj_sigma` means this is exactly 0 at construction (identical to
        # plain FiLM). After training, A_sigma(z) * B_sigma(feat) lives in R^r
        # and is projected back into the decoder's output space.
        if self.conditioning_type == "film_lora":
            attn_raw = attn_raw + self.proj_sigma(
                self.A_sigma(z_s_flat) * self.B_sigma(F.relu(sigma_feature))
            )
        one_sided = self.n_freq_bins
        attn_real = F.softplus(attn_raw[..., :one_sided]) + 1e-6
        attn_imag = attn_raw[..., one_sided:]
        attn_complex = torch.complex(attn_real, attn_imag)

        # Signal branch.
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
        )                                                           # [B*N, signal_feat_dim]
        if self.conditioning_type in ("film", "film_lora"):
            gamma_beta = self.film_signal(z_s_flat)
            gamma_g, beta_g = gamma_beta.chunk(2, dim=-1)
            feature_all = gamma_g * signal_feat + beta_g
        else:
            feature_all = torch.cat([signal_feat, z_s_flat], dim=-1)

        signal_raw = self._model_signal(feature_all)
        # film_lora: same additive rank-r correction on the signal MLP output.
        if self.conditioning_type == "film_lora":
            signal_raw = signal_raw + self.proj_signal(
                self.A_signal(z_s_flat) * self.B_signal(feature_all)
            )
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
