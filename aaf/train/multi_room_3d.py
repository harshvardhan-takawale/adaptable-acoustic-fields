"""MultiRoom3DTrainer — joint training of `INR3D_AutoDecoder` on multiple rooms.

P2-2 analog of `aaf.train.multi_room.MultiRoomTrainer`. Trains one shared
network plus one learnable latent z_s per training room (DeepSDF-style) using
`Shoebox3DDataset`. Loss is the Phase-1 4-term spectral loss + latent L2 +
3-axis geometry-head L1.

Optimizer has two parameter groups:
  - network params (everything except `model.latents`) → lr=2e-4
  - latents                                            → lr=1e-3

Validation loops over every (room, receiver) pair, reports per-room LSD/MAE,
aggregate losses, latent-norm histogram, and per-axis geometry-head MAE on the
trained latents.

CLI
---
    python -m aaf.train.multi_room_3d --config configs/sweep_3d/M1_45rooms.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from aaf.data.loader import Shoebox3DDataset
from aaf.models.inr_3d import INR3D_AutoDecoder, _default_hash_grid_config_3d
from aaf.renderers.freq_3d import FreqRenderer3D


@dataclass
class MultiRoom3DTrainCfg:
    n_iters: int = 30_000                            # D26
    batch_size: int = 4                              # D12 cascade
    grad_accum_steps: int = 1
    lr_network: float = 2e-4
    lr_latent: float = 1e-3                          # D25
    eta_min: float = 5e-5
    grad_clip_max_norm: float = 1.0
    weights: tuple = (1.0, 1.0, 1.0, 0.1)            # real, imag, log_amp, phase
    lambda_latent: float = 1e-4                      # D24
    n_azi: int = 16
    n_ele: int = 16
    n_pts_per_ray: int = 16                          # D12 cascade
    near: float = 1e-3
    fs: int = 4096
    n_time_samples: int = 8192
    c: float = 343.0
    val_every: int = 1_000
    ckpt_every: int = 2_500
    log_every: int = 100
    early_stop_warmup: int = 2_000
    early_stop_patience: int = 2_000
    early_stop_min_rel_improvement: float = 0.01
    seed: int = 0
    latent_dim: int = 16                             # D20
    log2_hashmap_size: int = 18                      # D23 / P2-1 D10
    n_levels: int = 16
    per_level_scale: float = 1.38
    conditioning_type: str = "film"                  # D19
    latent_jitter_sigma: float = 0.1                 # D21
    l_head_enabled: bool = True                      # D22 / D31
    l_head_weight: float = 0.1                       # D22 / D24


def _losses(H_pred: torch.Tensor, H_target: torch.Tensor) -> dict:
    eps = 1e-6
    return {
        "L_spec_real": F.l1_loss(H_pred.real, H_target.real),
        "L_spec_imag": F.l1_loss(H_pred.imag, H_target.imag),
        "L_amp": F.l1_loss(
            torch.log10(H_pred.abs() + eps),
            torch.log10(H_target.abs() + eps),
        ),
        "L_phase": (1.0 - torch.cos(H_pred.angle() - H_target.angle())).mean(),
    }


def _full_band_metrics(H_pred: torch.Tensor, H_target: torch.Tensor) -> dict:
    eps = 1e-8
    lsd = (20.0 * (
        torch.log10(H_pred.abs().clamp(min=eps))
        - torch.log10(H_target.abs().clamp(min=eps))
    )).abs().mean().item()
    complex_l1 = (H_pred - H_target).abs().mean().item()
    mag_l1 = (H_pred.abs() - H_target.abs()).abs().mean().item()
    pd = H_pred.angle() - H_target.angle()
    pd_w = torch.minimum(pd.abs(), 2 * math.pi - pd.abs())
    return {
        "lsd_db": lsd,
        "complex_l1": complex_l1,
        "magnitude_l1": mag_l1,
        "phase_l1": pd_w.mean().item(),
    }


class MultiRoom3DTrainer:
    def __init__(
        self,
        rooms_yaml: str,
        output_dir: str,
        cfg: Optional[MultiRoom3DTrainCfg] = None,
    ):
        self.cfg = cfg or MultiRoom3DTrainCfg()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tb").mkdir(exist_ok=True)
        self.scalars_path = self.output_dir / "scalars.json"

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        if not torch.cuda.is_available():
            raise RuntimeError("MultiRoom3DTrainer requires CUDA.")
        self.device = torch.device("cuda")

        # Dataset: every training room (no room_filter).
        self.dataset = Shoebox3DDataset(rooms_yaml=rooms_yaml)
        self.n_rooms = len(self.dataset.rooms)
        self.n_freq_bins = self.dataset.n_freq_bins
        self.rooms_yaml = str(rooms_yaml)

        # Per-room AABB tensors (each room has different L, W, H).
        self.room_aabbs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for room_id, (L, W, H) in self.dataset.room_id_to_dims.items():
            self.room_aabbs[room_id] = (
                torch.tensor([0.0, 0.0, 0.0], device=self.device),
                torch.tensor([float(L), float(W), float(H)], device=self.device),
            )

        # Pin (room, rx) tensors to CPU; index + .to(device) per iter. The
        # H_target alone is 45 × 512 × 4097 × 8 B ≈ 720 MB; keeping it on GPU
        # leaves no headroom for the renderer's activation footprint on
        # 10-12 GB cards. Each iteration only needs `batch_size` receivers
        # (~130 KB), so the per-step transfer cost is negligible.
        items = [self.dataset[i] for i in range(len(self.dataset))]
        self.rx_pos_all = torch.stack([it["rx_pos"] for it in items])        # CPU
        self.tx_pos_all = torch.stack([it["tx_pos"] for it in items])
        self.H_target_all = torch.stack([it["H_complex"] for it in items])
        self.room_ids_all = torch.tensor(
            [it["room_id"] for it in items], dtype=torch.long
        )
        self.geom_per_sample = torch.tensor(
            [
                [float(it["L"]), float(it["W"]), float(it["H"])]
                for it in items
            ],
            dtype=torch.float32,
        )                                                                    # CPU [N_samples, 3]

        # HashGrid config (D23 / P2-1 D10).
        hg_cfg = {
            "otype": "HashGrid",
            "n_levels": self.cfg.n_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": self.cfg.log2_hashmap_size,
            "base_resolution": 16,
            "per_level_scale": self.cfg.per_level_scale,
        }
        self.model = INR3D_AutoDecoder(
            n_rooms=self.n_rooms,
            latent_dim=self.cfg.latent_dim,
            n_freq_bins=self.n_freq_bins,
            hash_grid_config=hg_cfg,
            l_head_enabled=self.cfg.l_head_enabled,
            conditioning_type=self.cfg.conditioning_type,
            latent_jitter_sigma=self.cfg.latent_jitter_sigma,
        ).to(self.device)

        self.renderer = FreqRenderer3D(
            n_azi=self.cfg.n_azi,
            n_ele=self.cfg.n_ele,
            n_pts_per_ray=self.cfg.n_pts_per_ray,
            near=self.cfg.near,
            fs=self.cfg.fs,
            n_time_samples=self.cfg.n_time_samples,
            c=self.cfg.c,
            use_geometric_attn=False,
        ).to(self.device)

        # Two-param-group Adam: network lr=2e-4, latents lr=1e-3.
        latent_params = list(self.model.latents.parameters())
        latent_param_ids = {id(p) for p in latent_params}
        network_params = [
            p for p in self.model.parameters() if id(p) not in latent_param_ids
        ]
        self.optimizer = torch.optim.Adam(
            [
                {"params": network_params, "lr": self.cfg.lr_network, "name": "network"},
                {"params": latent_params, "lr": self.cfg.lr_latent, "name": "latent"},
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.n_iters, eta_min=self.cfg.eta_min
        )

        try:
            self.writer = SummaryWriter(str(self.output_dir / "tb"))
        except Exception as e:
            print(f"[trainer] SummaryWriter init failed ({e!r}); disabling tb.")
            self.writer = None
        self.scalars: list[dict] = []
        self.start_iter = 0
        self._maybe_resume()

    # ------------------------- checkpoint I/O -------------------------------

    def _maybe_resume(self):
        ckpts = sorted(
            self.output_dir.glob("ckpt_iter*.pt"),
            key=lambda p: int(p.stem.split("ckpt_iter")[-1]),
            reverse=True,
        )
        for p in ckpts:
            try:
                state = torch.load(p, map_location=self.device)
                self.model.load_state_dict(state["model"])
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
                self.start_iter = int(state["iter"])
                print(f"[resume] loaded {p.name} → start_iter={self.start_iter}")
                if self.scalars_path.exists():
                    try:
                        self.scalars = json.loads(self.scalars_path.read_text())
                        self.scalars = [s for s in self.scalars if s["iter"] <= self.start_iter]
                    except Exception:
                        self.scalars = []
                return
            except Exception as e:
                print(f"[resume] skipping corrupted ckpt {p.name}: {e!r}")
        print("[resume] no usable checkpoint; starting from scratch")

    def save_ckpt(self, it: int):
        path = self.output_dir / f"ckpt_iter{it:07d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "iter": it,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, tmp)
        tmp.replace(path)
        self.scalars_path.write_text(json.dumps(self.scalars))
        ckpts = sorted(
            self.output_dir.glob("ckpt_iter*.pt"),
            key=lambda p: int(p.stem.split("ckpt_iter")[-1]),
        )
        for old in ckpts[:-3]:
            try:
                old.unlink()
            except Exception:
                pass

    # ------------------------- early-stop helper ----------------------------

    def _val_total_loss(self, val: dict) -> float:
        w_r, w_i, w_a, w_p = self.cfg.weights
        return (
            w_r * val["L_spec_real"]
            + w_i * val["L_spec_imag"]
            + w_a * val["L_amp"]
            + w_p * val["L_phase"]
        )

    def _check_early_stop(self, current_iter: int) -> tuple[bool, str]:
        cfg = self.cfg
        if current_iter < cfg.early_stop_warmup:
            return False, ""
        boundary = current_iter - cfg.early_stop_patience
        if boundary <= 0:
            return False, ""
        in_window: list[float] = []
        before: list[float] = []
        for r in self.scalars:
            if r.get("phase") != "val":
                continue
            it = int(r["iter"])
            v = self._val_total_loss(r)
            if it <= boundary:
                before.append(v)
            elif it <= current_iter:
                in_window.append(v)
        if not in_window or not before:
            return False, ""
        best_recent = min(in_window)
        best_prior = min(before)
        improvement = (best_prior - best_recent) / max(best_prior, 1e-12)
        if improvement < cfg.early_stop_min_rel_improvement:
            return True, (
                f"best_val_in_window={best_recent:.6f} (last {cfg.early_stop_patience} iters), "
                f"best_val_before={best_prior:.6f}, improvement={improvement*100:.2f}% "
                f"< {cfg.early_stop_min_rel_improvement*100:.0f}%"
            )
        return False, ""

    # ------------------------- training step --------------------------------

    def _render_grouped_by_room(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward all samples in `indices`, sub-batched by room (each room has
        its own AABB). `indices` is a CPU tensor of `room_ids_all`-indices.
        Returns (H_pred [B, n_freq], z_s [B, latent_dim]) on device.
        """
        indices_cpu = indices.cpu() if indices.device.type != "cpu" else indices
        room_ids = self.room_ids_all[indices_cpu]
        H_pred_list: list[tuple[int, torch.Tensor]] = []
        z_s_list: list[tuple[int, torch.Tensor]] = []
        unique_rooms = torch.unique(room_ids).tolist()
        for r in unique_rooms:
            mask = (room_ids == r)
            sub_idx = indices_cpu[mask]
            rx = self.rx_pos_all[sub_idx].to(self.device)
            tx = self.tx_pos_all[sub_idx].to(self.device)
            z_s = self.model.get_latent(
                torch.full(
                    (sub_idx.size(0),), r, dtype=torch.long, device=self.device,
                )
            )
            room_min, room_max = self.room_aabbs[r]
            H_pred = self.renderer(self.model, rx, tx, room_min, room_max, z_s=z_s)
            for k, pos in enumerate(mask.nonzero(as_tuple=True)[0].tolist()):
                H_pred_list.append((pos, H_pred[k]))
                z_s_list.append((pos, z_s[k]))
        H_pred_list.sort(key=lambda t: t[0])
        z_s_list.sort(key=lambda t: t[0])
        H_pred_batch = torch.stack([h for _, h in H_pred_list], dim=0)
        z_s_batch = torch.stack([z for _, z in z_s_list], dim=0)
        return H_pred_batch, z_s_batch

    def _step(self) -> dict:
        cfg = self.cfg
        bs_total = cfg.batch_size
        accum = max(1, cfg.grad_accum_steps)
        bs_micro = max(1, bs_total // accum)

        n_samples = self.rx_pos_all.size(0)
        accum_metrics: list[dict] = []
        self.optimizer.zero_grad(set_to_none=True)

        for _ in range(accum):
            idx = torch.randint(0, n_samples, (bs_micro,))                   # CPU
            H_target = self.H_target_all[idx].to(self.device)
            H_pred, z_s = self._render_grouped_by_room(idx)
            losses = _losses(H_pred, H_target)
            l_latent = (z_s ** 2).mean()
            w_r, w_i, w_a, w_p = cfg.weights
            loss = (
                w_r * losses["L_spec_real"]
                + w_i * losses["L_spec_imag"]
                + w_a * losses["L_amp"]
                + w_p * losses["L_phase"]
                + cfg.lambda_latent * l_latent
            )
            l_lhead_val = 0.0
            if cfg.l_head_enabled and cfg.l_head_weight > 0 and self.model.l_head is not None:
                geom_pred = self.model.predict_geometry(z_s)                # [B, 3]
                geom_true = self.geom_per_sample[idx].to(self.device)        # [B, 3]
                l_lhead = F.l1_loss(geom_pred, geom_true)
                loss = loss + cfg.l_head_weight * l_lhead
                l_lhead_val = float(l_lhead.detach())
            loss = loss / accum
            loss.backward()
            accum_metrics.append({
                "loss": float(loss.detach() * accum),
                "L_latent": float(l_latent.detach()),
                "L_lhead": l_lhead_val,
                **{k: float(v.detach()) for k, v in losses.items()},
            })

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_max_norm)
        self.optimizer.step()
        self.scheduler.step()

        keys = accum_metrics[0].keys()
        return {k: float(np.mean([m[k] for m in accum_metrics])) for k in keys}

    # ------------------------- validation -----------------------------------

    @torch.no_grad()
    def validate(self) -> dict:
        """Per-room validation. Renders a 64-receiver per-room sub-sample to
        keep val fast; aggregates losses + LSD across rooms.
        """
        self.model.eval()
        self.renderer.eval()
        try:
            # Build a deterministic 64-of-512 per-room subsample (each receiver
            # spaced 8 apart). Saves ~8× val time vs full 512.
            VAL_PER_ROOM = 64
            # Chunk at the MICRO-batch size, not cfg.batch_size. With grad
            # accumulation, cfg.batch_size is the (large) effective batch; a
            # val forward of that many receivers at n_pts=32 would OOM. The
            # micro-batch is the proven-fitting forward size used in training.
            chunk = max(1, self.cfg.batch_size // max(1, self.cfg.grad_accum_steps))
            all_pred = []
            all_target = []
            per_room_metrics: dict[str, float] = {}
            for r in range(self.n_rooms):
                room_idxs = (self.room_ids_all == r).nonzero(as_tuple=True)[0]   # CPU
                # Subsample VAL_PER_ROOM receivers evenly.
                step = max(1, room_idxs.size(0) // VAL_PER_ROOM)
                sub_idxs = room_idxs[::step][:VAL_PER_ROOM]                       # CPU
                room_min, room_max = self.room_aabbs[r]
                z_one = self.model.get_latent(
                    torch.full((1,), r, dtype=torch.long, device=self.device)
                )                                                            # [1, latent_dim]
                Hp_list = []
                Ht_list = []
                for s in range(0, sub_idxs.size(0), chunk):
                    sub = sub_idxs[s : s + chunk]
                    rx = self.rx_pos_all[sub].to(self.device)
                    tx = self.tx_pos_all[sub].to(self.device)
                    z_s = z_one.expand(sub.size(0), -1)
                    H_pred = self.renderer(
                        self.model, rx, tx, room_min, room_max, z_s=z_s
                    )
                    Hp_list.append(H_pred)
                    Ht_list.append(self.H_target_all[sub].to(self.device))
                if not Hp_list:
                    continue
                Hp = torch.cat(Hp_list, dim=0)
                Ht = torch.cat(Ht_list, dim=0)
                m = _full_band_metrics(Hp, Ht)
                L, W, H = self.dataset.room_id_to_dims[r]
                key_base = f"L{L:.2f}_W{W:.2f}_H{H:.2f}"
                for k, v in m.items():
                    per_room_metrics[f"{key_base}_{k}"] = float(v)
                all_pred.append(Hp)
                all_target.append(Ht)

            H_pred_all = torch.cat(all_pred, dim=0)
            H_target_all = torch.cat(all_target, dim=0)
            losses = _losses(H_pred_all, H_target_all)
            agg = _full_band_metrics(H_pred_all, H_target_all)
            latent_norms = self.model.latents.weight.norm(dim=-1).detach().cpu().numpy()
            extras = {}
            if (
                self.cfg.l_head_enabled
                and self.cfg.l_head_weight > 0
                and self.model.l_head is not None
            ):
                z_table = self.model.latents.weight                          # [n_rooms, latent_dim]
                geom_pred = (
                    self.model.predict_geometry(z_table).detach().cpu().numpy()
                )                                                            # [n_rooms, 3]
                geom_true = np.array(
                    [
                        list(self.dataset.room_id_to_dims[i])
                        for i in range(self.n_rooms)
                    ],
                    dtype=np.float32,
                )
                axis_err = np.abs(geom_pred - geom_true)                     # [n_rooms, 3]
                extras["geom_mae_L_m"] = float(axis_err[:, 0].mean())
                extras["geom_mae_W_m"] = float(axis_err[:, 1].mean())
                extras["geom_mae_H_m"] = float(axis_err[:, 2].mean())
                extras["geom_mae_overall_m"] = float(axis_err.mean())
                extras["geom_max_err_m"] = float(axis_err.max())
            return {
                **{k: float(v) for k, v in losses.items()},
                **agg,
                **per_room_metrics,
                "latent_norm_mean": float(np.mean(latent_norms)),
                "latent_norm_min": float(np.min(latent_norms)),
                "latent_norm_max": float(np.max(latent_norms)),
                **extras,
            }
        finally:
            self.model.train()
            self.renderer.train()

    # ------------------------- main loop ------------------------------------

    def train(self):
        cfg = self.cfg
        t0 = time.time()
        stopped_early = False
        stop_reason = ""
        stop_iter = None
        ran_to = self.start_iter

        for it in range(self.start_iter, cfg.n_iters):
            train_log = self._step()
            ran_to = it + 1
            if it % cfg.log_every == 0:
                row = {
                    "iter": it, "phase": "train", **train_log,
                    "lr_network": float(self.optimizer.param_groups[0]["lr"]),
                    "lr_latent": float(self.optimizer.param_groups[1]["lr"]),
                }
                self.scalars.append(row)
                if self.writer is not None:
                    try:
                        for k, v in train_log.items():
                            self.writer.add_scalar(f"train/{k}", v, it)
                    except Exception as e:
                        print(f"[trainer] tb add_scalar failed at iter {it}: {e!r}")
                        self.writer = None
            if (it + 1) % cfg.val_every == 0 or it == cfg.n_iters - 1:
                val = self.validate()
                row = {"iter": it + 1, "phase": "val", **val}
                self.scalars.append(row)
                if self.writer is not None:
                    try:
                        for k, v in val.items():
                            self.writer.add_scalar(f"val/{k}", v, it + 1)
                        self.writer.add_histogram(
                            "val/latent_norms",
                            self.model.latents.weight.norm(dim=-1).detach().cpu().numpy(),
                            it + 1,
                        )
                    except Exception as e:
                        print(f"[trainer] tb val write failed at iter {it+1}: {e!r}")
                        self.writer = None
                should_stop, why = self._check_early_stop(it + 1)
                if should_stop:
                    stopped_early = True
                    stop_reason = why
                    stop_iter = it + 1
                    print(f"[early-stop] iter={it+1}: {why}")
                    self.save_ckpt(it + 1)
                    break
            if (it + 1) % cfg.ckpt_every == 0:
                self.save_ckpt(it + 1)
        else:
            self.save_ckpt(cfg.n_iters)

        self.scalars_path.write_text(json.dumps(self.scalars))
        # Per-room dims for downstream eval.
        L_list = [self.dataset.room_id_to_dims[i][0] for i in range(self.n_rooms)]
        W_list = [self.dataset.room_id_to_dims[i][1] for i in range(self.n_rooms)]
        H_list = [self.dataset.room_id_to_dims[i][2] for i in range(self.n_rooms)]
        meta = {
            "n_rooms": self.n_rooms,
            "L_list": L_list, "W_list": W_list, "H_list": H_list,
            "rooms_yaml": self.rooms_yaml,
            "n_iters_target": cfg.n_iters,
            "n_iters_actual": ran_to,
            "stopped_early": stopped_early,
            "stop_iter": stop_iter,
            "stop_reason": stop_reason,
            "wall_clock_seconds": time.time() - t0,
            "cfg": asdict(cfg),
        }
        (self.output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
        print(
            f"[done] iters={ran_to}/{cfg.n_iters} early_stop={stopped_early} "
            f"wall={time.time()-t0:.1f}s output={self.output_dir}"
        )


def _load_sweep_yaml(path: str) -> dict:
    """Load a training-config YAML (configs/sweep_3d/M*.yaml). Required keys:
    ``run_id``, ``rooms_yaml``. All others override `MultiRoom3DTrainCfg` defaults.
    """
    import yaml as _yaml
    with open(path) as f:
        d = _yaml.safe_load(f)
    if "run_id" not in d or "rooms_yaml" not in d:
        raise ValueError(
            f"sweep YAML {path} must contain `run_id` and `rooms_yaml` keys"
        )
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=str, default=None,
        help="path to configs/sweep_3d/M*.yaml",
    )
    ap.add_argument("--rooms-yaml", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--n_iters", type=int, default=30_000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_pts_per_ray", type=int, default=16)
    ap.add_argument("--n_azi", type=int, default=16)
    ap.add_argument("--n_ele", type=int, default=16)
    ap.add_argument("--latent_dim", type=int, default=16)
    args = ap.parse_args()

    if args.config:
        d = _load_sweep_yaml(args.config)
        rooms_yaml = d["rooms_yaml"]
        out_dir = args.output_dir or f"outputs/multi_room_3d/{d['run_id']}"
        cfg = MultiRoom3DTrainCfg(
            n_iters=int(d.get("n_iters", 30_000)),
            batch_size=int(d.get("batch_size", 4)),
            grad_accum_steps=int(d.get("grad_accum_steps", 1)),
            n_azi=int(d.get("n_azi", 16)),
            n_ele=int(d.get("n_ele", 16)),
            n_pts_per_ray=int(d.get("n_pts_per_ray", 16)),
            lr_network=float(d.get("lr_network", 2e-4)),
            lr_latent=float(d.get("lr_latent", 1e-3)),
            val_every=int(d.get("val_every", 1_000)),
            ckpt_every=int(d.get("ckpt_every", 2_500)),
            latent_dim=int(d.get("latent_dim", 16)),
            log2_hashmap_size=int(d.get("log2_hashmap_size", 18)),
            n_levels=int(d.get("n_levels", 16)),
            per_level_scale=float(d.get("per_level_scale", 1.38)),
            lambda_latent=float(d.get("lambda_latent", 1e-4)),
            conditioning_type=str(d.get("conditioning_type", "film")),
            latent_jitter_sigma=float(d.get("latent_jitter_sigma", 0.1)),
            l_head_enabled=bool(d.get("l_head_enabled", True)),
            l_head_weight=float(d.get("l_head_weight", 0.1)),
            # P2-2.5 diagnostic: expose early-stop knobs to the YAML so we can
            # relax them for Run B without touching the trainer code.
            early_stop_warmup=int(d.get("early_stop_warmup", 2_000)),
            early_stop_patience=int(d.get("early_stop_patience", 2_000)),
            early_stop_min_rel_improvement=float(
                d.get("early_stop_min_rel_improvement", 0.01)
            ),
        )
    else:
        if not args.rooms_yaml or not args.output_dir:
            ap.error("either --config or both (--rooms-yaml + --output_dir) required")
        rooms_yaml = args.rooms_yaml
        out_dir = args.output_dir
        cfg = MultiRoom3DTrainCfg(
            n_iters=args.n_iters,
            batch_size=args.batch_size,
            n_pts_per_ray=args.n_pts_per_ray,
            n_azi=args.n_azi,
            n_ele=args.n_ele,
            latent_dim=args.latent_dim,
        )

    trainer = MultiRoom3DTrainer(
        rooms_yaml=rooms_yaml, output_dir=out_dir, cfg=cfg
    )
    trainer.train()


if __name__ == "__main__":
    main()
