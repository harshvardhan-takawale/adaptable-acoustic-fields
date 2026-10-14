"""D144: signal-pathway parameter counts for B5, PB3 and candidate W3 widths (GPU; builds models)."""
import torch

from aaf.models.inr_2d import INR2D_AutoDecoder
from aaf.train.multi_room_2d_mat import signal_kwargs

HG = dict(otype="HashGrid", n_levels=20, n_features_per_level=2, log2_hashmap_size=18,
          base_resolution=80, per_level_scale=1.5)


def build(kw):
    return INR2D_AutoDecoder(n_rooms=1, latent_dim=16, n_freq_bins=4097, hash_grid_config=HG,
                             conditioning_type="film", cond_source="geom_token", cond_dim=324,
                             world_scale=10.0, token_pool="attn_residual_extent", dc_head_bins=41, **kw).cuda()


base = {"fs": 4096, "n_time_samples": 8192}
b5 = build({}).signal_param_counts()
pb3 = build(signal_kwargs(dict(base, signal_heads=3, signal_band_edges_hz="173.0,245.0", signal_xfade_hz=10.0), 601)).signal_param_counts()
print("B5  ", b5)
print("PB3 ", pb3)
best = None
for w in range(768, 1025, 16):
    c = build(signal_kwargs(dict(base, signal_band_only=True, signal_n_neurons=w), 601)).signal_param_counts()
    d = abs(c["raw"] - pb3["raw"]) / pb3["raw"]
    print("W3 width {:4d}: {}  (|diff| {:.2%})".format(w, c["raw"], d))
    if best is None or d < best[1]:
        best = (w, d, c["raw"])
    torch.cuda.empty_cache()
print("BEST W3 width", best)
