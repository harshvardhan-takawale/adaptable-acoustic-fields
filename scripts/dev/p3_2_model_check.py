"""P3-2 model-port check: bit-identical back-compat + the new conditioning arm. Needs CUDA."""
import torch
from aaf.models.inr_2d import INR2D_AutoDecoder
from aaf.models.conditioning_2d import FOURIER_DIM_2D, fourier_features_2d, COND_SOURCE
from aaf.walls import alphas_for

HG = dict(otype="HashGrid", n_levels=4, n_features_per_level=2, log2_hashmap_size=14,
          base_resolution=16, per_level_scale=1.5)


def build(**kw):
    torch.manual_seed(0)
    return INR2D_AutoDecoder(n_rooms=8, latent_dim=16, n_freq_bins=65,
                             hash_grid_config=HG, conditioning_type="film", **kw).cuda()


a = build()
b = build(cond_source="latent", cond_dim=None)
same = all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
print("1) legacy init bit-identical:", same, "| latents:", type(a.latents).__name__)
print("   film_sigma.in_features:", a.film_sigma.in_features, "(expect latent_dim=16)")

g = build(cond_source=COND_SOURCE, cond_dim=FOURIER_DIM_2D, l_head_enabled=False)
print("2) geom arm: latents is None:", g.latents is None, "| l_head is None:", g.l_head is None)
print("   film_sigma.in_features:", g.film_sigma.in_features, "(expect 64)")
try:
    g.get_latent(torch.zeros(1, dtype=torch.long, device="cuda"))
    print("   ERROR: get_latent guard missing")
except RuntimeError as e:
    print("   get_latent guard:", str(e)[:56])


def backbone_n(m):
    return sum(p.numel() for n, p in m.named_parameters()
               if not n.startswith(("film_", "A_", "proj_", "latents", "l_head")))


print("3) backbone params equal across arms:", backbone_n(a) == backbone_n(g), backbone_n(a))

z = fourier_features_2d(4.5, 4.0, alphas_for("west", "M3"), device="cuda").unsqueeze(0)
z0 = fourier_features_2d(4.5, 4.0, alphas_for(), device="cuda").unsqueeze(0)
pts = torch.rand(1, 32, 2, device="cuda")
view = torch.randn(1, 32, 2, device="cuda")
tx = torch.rand(1, 32, 2, device="cuda")
with torch.no_grad():
    s1 = g(pts, view, tx, z_s=z)[1]
    s0 = g(pts, view, tx, z_s=z0)[1]
print("4) forward ok:", tuple(s1.shape), "| identity-at-init (FiLM):", torch.allclose(s1, s0))
