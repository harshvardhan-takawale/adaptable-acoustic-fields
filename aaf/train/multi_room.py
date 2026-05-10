"""MultiRoomTrainer — joint training of an INR2D_AutoDecoder on multiple rooms.

Trains one shared network plus one learnable latent z_s per training room
(DeepSDF-style). Per-room latents live in ``model.latents`` (an
``nn.Embedding(n_rooms, latent_dim)``).

Loss (5 terms, weighted): the four from Chunk-2 (real, imag, log-amp, phase)
plus an L2 regulariser on z_s with weight ``λ_latent = 1e-4``.

Optimizer has two parameter groups:
  - network params (everything except ``model.latents``) → lr=2e-4
  - latents              → lr=1e-3 (latents benefit from higher LR per DeepSDF)

Validation loops over every (room, receiver) pair and reports per-room
LSD/MAE plus aggregate losses.

The helper functions ``_losses``, ``_full_band_metrics``, ``_check_early_stop``,
``_val_total_loss``, ``save_ckpt``, and ``_maybe_resume`` are intentionally
duplicated from ``aaf.train.single_room`` rather than imported, to avoid
coupling the two trainers (and to keep Chunk 2's code untouched).

CLI
---
    python -m aaf.train.multi_room --sweep configs/sweeps/dense.yaml \
                                    --output_dir outputs/multi_room/dense
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
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from aaf.data.loader import ShoeboxDataset
from aaf.models.inr_2d import INR2D_AutoDecoder
from aaf.renderers.freq_2d import FreqRenderer2D


@dataclass
class MultiRoomTrainCfg:
    n_iters: int = 30_000
    batch_size: int = 16
    grad_accum_steps: int = 1                    # >1 enables grad accumulation
    lr_network: float = 2e-4
    lr_latent: float = 1e-3
    eta_min: float = 5e-5
    grad_clip_max_norm: float = 1.0
    weights: tuple = (1.0, 1.0, 1.0, 0.1)        # real, imag, log_amp, phase
    lambda_latent: float = 1e-4                  # ||z_s||^2
    n_azi: int = 64
    n_pts_per_ray: int = 32                      # from Chunk-2 memory check
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
    latent_dim: int = 32


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


class MultiRoomTrainer:
    def __init__(self, sweep_yaml: str, output_dir: str,
                 cfg: Optional[MultiRoomTrainCfg] = None):
        self.cfg = cfg or MultiRoomTrainCfg()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tb").mkdir(exist_ok=True)
        self.scalars_path = self.output_dir / "scalars.json"

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        if not torch.cuda.is_available():
            raise RuntimeError("MultiRoomTrainer requires CUDA.")
        self.device = torch.device("cuda")

        # Dataset: every training room (no room_filter).
        self.dataset = ShoeboxDataset(sweep_yaml=sweep_yaml, split="train")
        self.n_rooms = len(self.dataset.L_list)
        self.n_freq_bins = self.dataset.n_freq_bins
        self.W = self.dataset.W

        # Pre-compute per-room AABB tensors once.
        self.room_aabbs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for room_id, L in self.dataset.room_id_to_L.items():
            self.room_aabbs[room_id] = (
                torch.tensor([0.0, 0.0], device=self.device),
                torch.tensor([float(L), self.W], device=self.device),
            )

        # Pre-load all (room, rx) tensors onto GPU for fast train-time batching.
        # The 7-room dense set is small enough (~14 MB total).
        items = [self.dataset[i] for i in range(len(self.dataset))]
        self.rx_pos_all = torch.stack([it["rx_pos"] for it in items]).to(self.device)
        self.tx_pos_all = torch.stack([it["tx_pos"] for it in items]).to(self.device)
        self.H_target_all = torch.stack([it["H_complex"] for it in items]).to(self.device)
        self.room_ids_all = torch.tensor(
            [it["room_id"] for it in items], dtype=torch.long, device=self.device
        )

        self.model = INR2D_AutoDecoder(
            n_rooms=self.n_rooms,
            latent_dim=self.cfg.latent_dim,
            n_freq_bins=self.n_freq_bins,
        ).to(self.device)
        self.renderer = FreqRenderer2D(
            n_azi=self.cfg.n_azi,
            n_pts_per_ray=self.cfg.n_pts_per_ray,
            near=self.cfg.near,
            fs=self.cfg.fs,
            n_time_samples=self.cfg.n_time_samples,
            c=self.cfg.c,
            use_geometric_attn=False,
        ).to(self.device)

        # Two-param-group optimizer: network (lr=2e-4) + latents (lr=1e-3).
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

        self.writer = SummaryWriter(str(self.output_dir / "tb"))
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
        # Keep the 3 newest checkpoints.
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

    def _render_grouped_by_room(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward all samples in `indices`, sub-batched by room (so each call to
        the renderer gets the correct AABB). Returns (H_pred [B, n_freq], z_s [B, latent_dim]).
        """
        room_ids = self.room_ids_all[indices]
        # Group indices by room.
        H_pred_list: list[tuple[int, torch.Tensor]] = []
        z_s_list: list[tuple[int, torch.Tensor]] = []
        unique_rooms = torch.unique(room_ids).tolist()
        for r in unique_rooms:
            mask = (room_ids == r)
            sub_idx = indices[mask]
            rx = self.rx_pos_all[sub_idx]
            tx = self.tx_pos_all[sub_idx]
            z_s = self.model.get_latent(torch.full((sub_idx.size(0),), r,
                                                   dtype=torch.long, device=self.device))
            room_min, room_max = self.room_aabbs[r]
            H_pred = self.renderer(self.model, rx, tx, room_min, room_max, z_s=z_s)
            # Stash with original-batch position so we can scatter back.
            for k, pos in enumerate(mask.nonzero(as_tuple=True)[0].tolist()):
                H_pred_list.append((pos, H_pred[k]))
                z_s_list.append((pos, z_s[k]))
        # Re-order to match the original `indices` order.
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
            idx = torch.randint(0, n_samples, (bs_micro,), device=self.device)
            H_target = self.H_target_all[idx]
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
            ) / accum
            loss.backward()
            accum_metrics.append({
                "loss": float(loss.detach() * accum),
                "L_latent": float(l_latent.detach()),
                **{k: float(v.detach()) for k, v in losses.items()},
            })

        # Mask non-finite gradients.
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_max_norm)
        self.optimizer.step()
        self.scheduler.step()

        # Average metrics across accum.
        keys = accum_metrics[0].keys()
        out = {k: float(np.mean([m[k] for m in accum_metrics])) for k in keys}
        return out

    # ------------------------- validation -----------------------------------

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        self.renderer.eval()
        try:
            n_samples = self.rx_pos_all.size(0)
            chunk = self.cfg.batch_size
            chunks_per_room: dict[int, list[torch.Tensor]] = {r: [] for r in range(self.n_rooms)}
            chunks_target: dict[int, list[torch.Tensor]] = {r: [] for r in range(self.n_rooms)}
            for r in range(self.n_rooms):
                idxs = (self.room_ids_all == r).nonzero(as_tuple=True)[0]
                room_min, room_max = self.room_aabbs[r]
                z_id = torch.full((1,), r, dtype=torch.long, device=self.device)
                z_s_one = self.model.get_latent(z_id)              # [1, latent_dim]
                for s in range(0, idxs.size(0), chunk):
                    sub = idxs[s : s + chunk]
                    rx = self.rx_pos_all[sub]
                    tx = self.tx_pos_all[sub]
                    z_s = z_s_one.expand(sub.size(0), -1)
                    H_pred = self.renderer(self.model, rx, tx, room_min, room_max, z_s=z_s)
                    chunks_per_room[r].append(H_pred)
                    chunks_target[r].append(self.H_target_all[sub])

            # Per-room metrics.
            per_room_metrics: dict[str, float] = {}
            all_pred = []
            all_target = []
            for r in range(self.n_rooms):
                if not chunks_per_room[r]:
                    continue
                Hp = torch.cat(chunks_per_room[r], dim=0)
                Ht = torch.cat(chunks_target[r], dim=0)
                m = _full_band_metrics(Hp, Ht)
                Lval = self.dataset.room_id_to_L[r]
                for k, v in m.items():
                    per_room_metrics[f"L_{Lval}_{k}"] = float(v)
                all_pred.append(Hp)
                all_target.append(Ht)

            H_pred_all = torch.cat(all_pred, dim=0)
            H_target_all = torch.cat(all_target, dim=0)
            losses = _losses(H_pred_all, H_target_all)
            agg = _full_band_metrics(H_pred_all, H_target_all)
            latent_norms = self.model.latents.weight.norm(dim=-1).detach().cpu().numpy()
            return {
                **{k: float(v) for k, v in losses.items()},
                **agg,
                **per_room_metrics,
                "latent_norm_mean": float(np.mean(latent_norms)),
                "latent_norm_min": float(np.min(latent_norms)),
                "latent_norm_max": float(np.max(latent_norms)),
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
                row = {"iter": it, "phase": "train", **train_log,
                       "lr_network": float(self.optimizer.param_groups[0]["lr"]),
                       "lr_latent": float(self.optimizer.param_groups[1]["lr"])}
                self.scalars.append(row)
                for k, v in train_log.items():
                    self.writer.add_scalar(f"train/{k}", v, it)
            if (it + 1) % cfg.val_every == 0 or it == cfg.n_iters - 1:
                val = self.validate()
                row = {"iter": it + 1, "phase": "val", **val}
                self.scalars.append(row)
                for k, v in val.items():
                    self.writer.add_scalar(f"val/{k}", v, it + 1)
                # Latent-norm histogram for visibility on collapse.
                self.writer.add_histogram(
                    "val/latent_norms",
                    self.model.latents.weight.norm(dim=-1).detach().cpu().numpy(),
                    it + 1,
                )
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

        # Final scalar dump and metadata.
        self.scalars_path.write_text(json.dumps(self.scalars))
        meta = {
            "n_rooms": self.n_rooms,
            "L_list": self.dataset.L_list,
            "n_iters_target": cfg.n_iters,
            "n_iters_actual": ran_to,
            "stopped_early": stopped_early,
            "stop_iter": stop_iter,
            "stop_reason": stop_reason,
            "wall_clock_seconds": time.time() - t0,
            "cfg": asdict(cfg),
        }
        (self.output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[done] iters={ran_to}/{cfg.n_iters} early_stop={stopped_early} "
              f"wall={time.time()-t0:.1f}s output={self.output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--n_iters", type=int, default=30_000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--n_pts_per_ray", type=int, default=32)
    ap.add_argument("--n_azi", type=int, default=64)
    args = ap.parse_args()

    cfg = MultiRoomTrainCfg(
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        n_pts_per_ray=args.n_pts_per_ray,
        n_azi=args.n_azi,
    )
    trainer = MultiRoomTrainer(sweep_yaml=args.sweep, output_dir=args.output_dir, cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
