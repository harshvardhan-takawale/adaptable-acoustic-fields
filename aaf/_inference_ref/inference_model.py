"""Vendored INFER model — REFERENCE ONLY, DO NOT IMPORT IN PRODUCTION CODE.

Source: project_files/unified_models.py (INFER ICML'26 submission)

Vendored:
  - class AVRModel_complex_FD_FreqDep_PhaseCorrection (source L752-883, the chosen Phase-1 baseline)

Excluded from this vendor: every other model variant in the source file
(AVRModel*, NAFModel*, INRASModel*). They are reachable via the source path
if needed.

The chosen model pairs with `AVRRenderFD_FreqDep_PhaseCorrection_new` in
`inference_renderer.py`. Forward signature:

    forward(pts, view, tx, tx_view) -> (attn_complex, signal_complex)
    pts, view, tx, tx_view: [bs, n_pts, 3]   (3D positions/directions)
    attn_complex:   [bs, n_pts, n_freq_bins // 2]   complex64 (σ + jβ per freq)
    signal_complex: [bs, n_pts, n_freq_bins // 2]   complex64 (H per freq)

For our 2D port, every `tcnn.Encoding(3, ...)` becomes `(2, ...)` for positions
and `(2, ...)` over (cos θ, sin θ) for directions. See tasks/CHUNK_0_RESULTS.md
"§6 — 2D adaptation needs (exhaustive)".

For per-room latent `z_s` injection (DeepSDF-style), the candidate concat
points are L831 (sigma branch) and L854 (signal branch) of the source file —
in the vendored copy below, those become the lines just before the
`_model_encoder_sigma(...)` call and just before the `_model_signal(...)` call.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn


# ---------------------------------------------------------------------------
# Target class — Phase-1 starting point
# (source: unified_models.py L752-883)
# ---------------------------------------------------------------------------

class AVRModel_complex_FD_FreqDep_PhaseCorrection(nn.Module):
    """Complex AVR model with frequency-dependent processing."""

    def __init__(self, cfg):
        super().__init__()
        self.leaky_relu = cfg["leaky_relu"]
        pos_encoding_sigma = cfg["pos_encoding_sigma"]
        tx_pos_encoding_sigma = cfg["tx_pos_encoding_sigma"]
        pos_encoding_signal = cfg['pos_encoding_sig']
        tx_pos_encoding_signal = cfg['tx_pos_encoding_sig']
        dir_encoding_sig = cfg["dir_encoding_sig"]
        tx_dir_encoding_sig = cfg["tx_dir_encoding_sig"]
        sigma_encoder_network = cfg["sigma_encoder_network"]
        sigma_decoder_network = cfg["sigma_decoder_network"]
        signal_network = cfg['signal_network']
        self.signal_output_dim = cfg['signal_output_dim']

        # Six 3D position/direction encoders. For 2D, drop to Encoding(2, ...).
        self._pos_encoding = tcnn.Encoding(3, pos_encoding_sigma, dtype=torch.float32)
        self._pos_signal_encoding = tcnn.Encoding(3, pos_encoding_signal, dtype=torch.float32)
        self._tx_pos_encoding = tcnn.Encoding(3, tx_pos_encoding_sigma, dtype=torch.float32)
        self._tx_pos_signal_encoding = tcnn.Encoding(3, tx_pos_encoding_signal, dtype=torch.float32)
        self._dir_encoding = tcnn.Encoding(3, dir_encoding_sig, dtype=torch.float32)
        self._tx_dir_encoding = tcnn.Encoding(3, tx_dir_encoding_sig, dtype=torch.float32)

        network_in_dims = self._pos_encoding.n_output_dims + self._tx_pos_encoding.n_output_dims
        # Auto-decoder candidate A: widen `network_in_dims` by latent_dim here.

        self._model_encoder_sigma = tcnn.Network(
            n_input_dims=network_in_dims,
            n_output_dims=256,
            network_config=sigma_encoder_network,
        )

        self._model_decoder_sigma = tcnn.Network(
            n_input_dims=self._model_encoder_sigma.n_output_dims,
            n_output_dims=self.signal_output_dim,  # complex per-frequency attenuation
            network_config=sigma_decoder_network,
        )

        n_signal_input = (
            self._model_encoder_sigma.n_output_dims
            + self._dir_encoding.n_output_dims
            + self._tx_dir_encoding.n_output_dims
            + self._pos_signal_encoding.n_output_dims
            + self._tx_pos_signal_encoding.n_output_dims
        )
        # Auto-decoder candidate A: also widen `n_signal_input` by latent_dim here.

        self._model_signal = tcnn.Network(
            n_input_dims=n_signal_input,
            n_output_dims=self.signal_output_dim,
            network_config=signal_network,
        )

    def forward(self, pts, view, tx, tx_view):
        """
        Parameters
        ----------
        pts:     [bs, n_rays * n_samples, 3]  voxel positions
        view:    [bs, n_rays * n_samples, 3]  view direction
        tx:      [bs, n_rays * n_samples, 3]  emitter position
        tx_view: [bs, n_rays * n_samples, 3]  emitter view direction

        Returns
        -------
        attn:   [bs, n_rays * n_samples, N_freq_bins]  complex attenuation
        signal: [bs, n_rays * n_samples, N_freq_bins]  complex signal
        """
        bs = pts.size(0)
        n_ray_points = pts.size(1)

        pts = (pts.view(-1, 3) + 1) / 2
        view = (view.view(-1, 3) + 1) / 2
        tx = (tx.view(-1, 3) + 1) / 2
        tx_view = (tx_view.reshape(-1, 3) + 1) / 2

        pos_embedding = self._pos_encoding(pts)
        tx_pos_embedding = self._tx_pos_encoding(tx)

        # Sigma branch concat point — INJECT z_s HERE (auto-decoder candidate A).
        sigma_feature = self._model_encoder_sigma(
            torch.cat([pos_embedding, tx_pos_embedding], -1)
        )

        attn_raw = self._model_decoder_sigma(F.relu(sigma_feature))  # [N, signal_output_dim]

        # Split into real and imaginary parts per frequency.
        one_sided_length = self.signal_output_dim // 2
        attn_real = attn_raw[..., :one_sided_length]
        attn_imag = attn_raw[..., one_sided_length:]

        # Enforce σ ≥ 0 (physical absorption).
        attn_real = F.softplus(attn_real) + 1e-6
        attn_complex = torch.complex(attn_real, attn_imag)

        view_embedding = self._dir_encoding(view)
        tx_view_embedding = self._tx_dir_encoding(tx_view)
        signal_embedding = self._pos_signal_encoding(pts)
        tx_signal_embedding = self._tx_pos_signal_encoding(tx)

        # Signal branch concat point — INJECT z_s HERE (auto-decoder candidate A).
        feature_all = torch.cat(
            [
                F.relu(sigma_feature),
                view_embedding,
                tx_view_embedding,
                signal_embedding,
                tx_signal_embedding,
            ],
            -1,
        )

        signal = self._model_signal(feature_all)

        attn_complex = attn_complex.view(bs, n_ray_points, one_sided_length)

        signal_re = signal[..., :one_sided_length]
        signal_im = signal[..., one_sided_length:]

        # RFFT symmetry: zero the imaginary part of DC and (if even-length) Nyquist.
        mask_first = torch.ones_like(signal_im)
        mask_first[..., 0] = 0
        signal_im = signal_im * mask_first

        if self.signal_output_dim % 2 == 0:
            mask_last = torch.ones_like(signal_im)
            mask_last[..., -1] = 0
            signal_im = signal_im * mask_last

        signal_complex = torch.complex(signal_re, signal_im)
        signal_complex = signal_complex.reshape(bs, n_ray_points, one_sided_length)

        return attn_complex, signal_complex
