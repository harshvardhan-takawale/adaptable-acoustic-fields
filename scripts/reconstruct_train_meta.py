"""Reconstruct <outdir>/train_meta.json for a (possibly cancelled / mid-training) run,
so the eval loader can build the model. Deterministic from the arm config + rooms yaml.
Usage: python scripts/reconstruct_train_meta.py <arm_config.yaml> <output_dir>
"""
import json, sys
import yaml

cfg_path, out_dir = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(cfg_path))
rooms = yaml.safe_load(open(d["rooms_yaml"]))["rooms"]

# cfg dict with every key the eval / model construction / renderer reads (yaml value or default)
cfg = dict(
    latent_dim=int(d.get("latent_dim", 16)),
    fs=int(d.get("fs", 4096)),
    n_time_samples=int(d.get("n_time_samples", 8192)),
    n_levels=int(d.get("n_levels", 16)),
    log2_hashmap_size=int(d.get("log2_hashmap_size", 18)),
    per_level_scale=float(d.get("per_level_scale", 1.38)),
    conditioning_type=str(d.get("conditioning_type", "film")),
    latent_jitter_sigma=float(d.get("latent_jitter_sigma", 0.1)),
    l_head_enabled=bool(d.get("l_head_enabled", True)),
    cond_source=str(d.get("cond_source", "latent")),
    cond_dim=(int(d["cond_dim"]) if d.get("cond_dim") is not None else None),
    band_max_hz=(float(d["band_max_hz"]) if d.get("band_max_hz") is not None else None),
    n_azi=int(d.get("n_azi", 16)),
    n_ele=int(d.get("n_ele", 16)),
    n_pts_per_ray=int(d.get("n_pts_per_ray", 32)),
    near=float(d.get("near", 1e-3)),
    c=float(d.get("c", 343.0)),
)
meta = dict(
    n_rooms=len(rooms),
    L_list=[float(r["L"]) for r in rooms],
    W_list=[float(r["W"]) for r in rooms],
    H_list=[float(r["H"]) for r in rooms],
    rooms_yaml=d["rooms_yaml"],
    cfg=cfg,
    reconstructed=True,
)
open(f"{out_dir}/train_meta.json", "w").write(json.dumps(meta, indent=2))
print(f"wrote {out_dir}/train_meta.json  (arm={cfg['cond_source']}, n_rooms={meta['n_rooms']})")
