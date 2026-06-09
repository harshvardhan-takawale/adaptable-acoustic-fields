import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, numpy as np, torch
from pathlib import Path
from aaf.eval.known_geometry import (_load_trained_model, build_lookup_maps, render_full,
                                     _build_renderer, _load_room)
L, W, H = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
dev = "cuda"
model, tm = _load_trained_model(Path("outputs/multi_room_3d/P3_45rooms_4gpu"), dev)
cfg = tm["cfg"]; fs = float(cfg["fs"]); n_time = int(cfg["n_time_samples"])
rnd = _build_renderer(cfg, fs, n_time, dev)
for p in model.parameters(): p.requires_grad_(False)
Z = model.latents.weight.detach().cpu().numpy()
LWH = np.stack([tm["L_list"], tm["W_list"], tm["H_list"]], 1)
i = int(np.argmin(np.linalg.norm(LWH - [L, W, H], axis=1)))   # leave THIS room out
keep = [j for j in range(len(LWH)) if j != i]
z = torch.as_tensor(build_lookup_maps(LWH[keep], Z[keep])["rbf"]((L, W, H)), device=dev)
room = _load_room(f"data/track_a_3d/L{L:.2f}_W{W:.2f}_H{H:.2f}.h5")
rmin = torch.tensor([0., 0., 0.], device=dev); rmax = torch.tensor([room["L"], room["W"], room["H"]], device=dev)
Hp = render_full(model, rnd, z, rmin, rmax, room["receiver_pos"], room["src"], dev)
np.savez("outputs/known_geometry/loo_median_spectrum.npz", H_pred=Hp, H_target=room["H_target"],
         fs=fs, n_time=n_time, L=L, W=W, H=H, receiver_pos=room["receiver_pos"])
print("saved loo_median_spectrum.npz  shapes", Hp.shape, room["H_target"].shape)
