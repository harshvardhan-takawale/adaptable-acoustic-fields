"""SingleRoomTrainer — overfit one INR2D_Single per room.

Establishes the upper bound on per-room reconstruction quality at our chosen
architecture. Used for Chunk 2's L ∈ {3.0, 4.5, 6.0} runs.

Loss (4 terms, weighted 1, 1, 1, 0.1):
    L_spec_real = L1(real(H_pred), real(H_target))
    L_spec_imag = L1(imag(H_pred), imag(H_target))
    L_amp       = L1(log10(|H_pred|+ε), log10(|H_target|+ε))
    L_phase     = mean(1 - cos(angle(H_pred) - angle(H_target)))

Drops AVR's time-domain and multi-STFT losses (we're frequency-native already).

Auto-resume on startup: glob output_dir for `ckpt_iter*.pt`, pick the highest
iter whose file loads cleanly (corrupted/partial preempted writes are skipped).

Run via SLURM (`scripts/slurm/single_room_train.sh L`).

CLI
---
    python -m aaf.train.single_room --L 3.0 --sweep configs/sweeps/dense.yaml
                                    --output_dir outputs/single_room/L3.0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from aaf.data.loader import ShoeboxDataset
from aaf.models.inr_2d import INR2D_Single
from aaf.renderers.freq_2d import FreqRenderer2D


@dataclass
class TrainCfg:
    n_iters: int = 10_000
    batch_size: int = 8
    lr: float = 2e-4
    eta_min: float = 5e-5
    grad_clip_max_norm: float = 1.0
    weights: tuple = (1.0, 1.0, 1.0, 0.1)        # (real, imag, log_amp, phase)
    n_azi: int = 64
    n_pts_per_ray: int = 64
    near: float = 1e-3
    fs: int = 4096
    n_time_samples: int = 8192
    c: float = 343.0
    val_every: int = 500
    ckpt_every: int = 2_500
    log_every: int = 100
    # Relative-improvement early-stop. After iter `early_stop_warmup`, compare the best
    # val loss within the last `early_stop_patience` iters against the best val loss
    # before that window. If the recent best is not at least
    # `early_stop_min_rel_improvement` lower (fractional), stop training.
    early_stop_patience: int = 2_000
    early_stop_warmup: int = 2_000
    early_stop_min_rel_improvement: float = 0.01
    seed: int = 0


def _losses(H_pred: torch.Tensor, H_target: torch.Tensor) -> dict:
    eps = 1e-6
    spec_real = F.l1_loss(H_pred.real, H_target.real)
    spec_imag = F.l1_loss(H_pred.imag, H_target.imag)
    log_amp = F.l1_loss(
        torch.log10(H_pred.abs() + eps),
        torch.log10(H_target.abs() + eps),
    )
    phase_diff = H_pred.angle() - H_target.angle()
    phase = (1.0 - torch.cos(phase_diff)).mean()
    return {
        "L_spec_real": spec_real,
        "L_spec_imag": spec_imag,
        "L_amp": log_amp,
        "L_phase": phase,
    }


def _full_band_metrics(H_pred: torch.Tensor, H_target: torch.Tensor) -> dict:
    """Cheap metrics computed on a torch tensor for in-loop logging."""
    eps = 1e-8
    lsd = (20.0 * (
        torch.log10(H_pred.abs().clamp(min=eps))
        - torch.log10(H_target.abs().clamp(min=eps))
    )).abs().mean().item()
    complex_l1 = (H_pred - H_target).abs().mean().item()
    mag_l1 = (H_pred.abs() - H_target.abs()).abs().mean().item()
    phase_diff = H_pred.angle() - H_target.angle()
    phase_wrapped = torch.minimum(phase_diff.abs(), 2 * math.pi - phase_diff.abs())
    phase_l1 = phase_wrapped.mean().item()
    return {"lsd_db": lsd, "complex_l1": complex_l1, "magnitude_l1": mag_l1, "phase_l1": phase_l1}


class SingleRoomTrainer:
    def __init__(self, L: float, sweep_yaml: str, output_dir: str, cfg: Optional[TrainCfg] = None):
        self.L = float(L)
        self.cfg = cfg or TrainCfg()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tb").mkdir(exist_ok=True)
        self.ckpt_dir = self.output_dir
        self.scalars_path = self.output_dir / "scalars.json"

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        if not torch.cuda.is_available():
            raise RuntimeError("SingleRoomTrainer requires CUDA.")
        self.device = torch.device("cuda")

        self.dataset = ShoeboxDataset(
            sweep_yaml=sweep_yaml, split="train", room_filter=[self.L]
        )
        self.n_freq_bins = self.dataset.n_freq_bins
        self.W = self.dataset.W

        # Pre-tensor the (room, all-receivers) view — overfit is cheap; load once.
        items = [self.dataset[i] for i in range(len(self.dataset))]
        self.rx_pos_all = torch.stack([it["rx_pos"] for it in items]).to(self.device)
        self.tx_pos_all = torch.stack([it["tx_pos"] for it in items]).to(self.device)
        self.H_target_all = torch.stack([it["H_complex"] for it in items]).to(self.device)
        # Room AABB.
        self.room_min = torch.tensor([0.0, 0.0], device=self.device)
        self.room_max = torch.tensor([self.L, self.W], device=self.device)

        self.model = INR2D_Single(n_freq_bins=self.n_freq_bins).to(self.device)
        self.renderer = FreqRenderer2D(
            n_azi=self.cfg.n_azi,
            n_pts_per_ray=self.cfg.n_pts_per_ray,
            near=self.cfg.near,
            fs=self.cfg.fs,
            n_time_samples=self.cfg.n_time_samples,
            c=self.cfg.c,
            use_geometric_attn=False,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.n_iters, eta_min=self.cfg.eta_min
        )
        self.writer = SummaryWriter(str(self.output_dir / "tb"))
        self.scalars: list[dict] = []
        self.start_iter = 0
        self._maybe_resume()

    def _maybe_resume(self):
        ckpts = sorted(self.ckpt_dir.glob("ckpt_iter*.pt"),
                       key=lambda p: int(p.stem.split("ckpt_iter")[-1]),
                       reverse=True)
        for p in ckpts:
            try:
                state = torch.load(p, map_location=self.device)
                self.model.load_state_dict(state["model"])
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
                self.start_iter = int(state["iter"])
                print(f"[resume] loaded {p.name} → start_iter={self.start_iter}")
                # Restore scalars history if present.
                if self.scalars_path.exists():
                    try:
                        self.scalars = json.loads(self.scalars_path.read_text())
                        # Drop entries past resume iter (stale).
                        self.scalars = [s for s in self.scalars if s["iter"] <= self.start_iter]
                    except Exception:
                        self.scalars = []
                return
            except Exception as e:
                print(f"[resume] skipping corrupted ckpt {p.name}: {e!r}")
                continue
        print("[resume] no usable checkpoint; starting from scratch")

    def save_ckpt(self, it: int):
        path = self.ckpt_dir / f"ckpt_iter{it:07d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "iter": it,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, tmp)
        tmp.replace(path)
        # Persist scalar history alongside the latest checkpoint.
        self.scalars_path.write_text(json.dumps(self.scalars))
        # Garbage-collect: keep the 3 newest checkpoints to absorb preemption + roll-back.
        ckpts = sorted(self.ckpt_dir.glob("ckpt_iter*.pt"),
                       key=lambda p: int(p.stem.split("ckpt_iter")[-1]))
        for old in ckpts[:-3]:
            try:
                old.unlink()
            except Exception:
                pass

    def _step(self) -> dict:
        n_rx = self.rx_pos_all.size(0)
        idx = torch.randperm(n_rx, device=self.device)[: self.cfg.batch_size]
        rx = self.rx_pos_all[idx]
        tx = self.tx_pos_all[idx]
        H_target = self.H_target_all[idx]

        H_pred = self.renderer(self.model, rx, tx, self.room_min, self.room_max)
        losses = _losses(H_pred, H_target)
        w_r, w_i, w_a, w_p = self.cfg.weights
        loss = (
            w_r * losses["L_spec_real"]
            + w_i * losses["L_spec_imag"]
            + w_a * losses["L_amp"]
            + w_p * losses["L_phase"]
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Mask non-finite gradients (per AVR convention).
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_max_norm)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": float(loss.detach()),
            **{k: float(v.detach()) for k, v in losses.items()},
        }

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        self.renderer.eval()
        try:
            # Validate on the same 64 receivers (overfit, train=test).
            chunks = []
            for s in range(0, self.rx_pos_all.size(0), self.cfg.batch_size):
                rx = self.rx_pos_all[s : s + self.cfg.batch_size]
                tx = self.tx_pos_all[s : s + self.cfg.batch_size]
                H_pred = self.renderer(self.model, rx, tx, self.room_min, self.room_max)
                chunks.append(H_pred)
            H_pred_all = torch.cat(chunks, dim=0)
            losses = _losses(H_pred_all, self.H_target_all)
            metrics = _full_band_metrics(H_pred_all, self.H_target_all)
        finally:
            self.model.train()
            self.renderer.train()
        return {**{k: float(v) for k, v in losses.items()}, **metrics}

    def _val_total_loss(self, val: dict) -> float:
        """Combine per-term val losses into a single scalar with the training weights."""
        w_r, w_i, w_a, w_p = self.cfg.weights
        return (
            w_r * val["L_spec_real"]
            + w_i * val["L_spec_imag"]
            + w_a * val["L_amp"]
            + w_p * val["L_phase"]
        )

    def _check_early_stop(self, current_iter: int) -> tuple[bool, str]:
        """Relative-improvement early stop. Returns (should_stop, reason).

        At ``current_iter`` (a val checkpoint iter), compare the best val total
        loss in the window ``(current_iter - patience, current_iter]`` against the
        best val total loss recorded *before* that window. If the recent best is
        not at least ``min_rel_improvement`` (fractional) below the prior best,
        stop. Inactive until ``current_iter > early_stop_warmup`` AND there's at
        least one val point on each side of the window boundary.
        """
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
                       "lr": float(self.optimizer.param_groups[0]["lr"])}
                self.scalars.append(row)
                for k, v in train_log.items():
                    self.writer.add_scalar(f"train/{k}", v, it)
            if (it + 1) % cfg.val_every == 0 or it == cfg.n_iters - 1:
                val = self.validate()
                row = {"iter": it + 1, "phase": "val", **val}
                self.scalars.append(row)
                for k, v in val.items():
                    self.writer.add_scalar(f"val/{k}", v, it + 1)
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
        meta_path = self.output_dir / "train_meta.json"
        meta_path.write_text(json.dumps({
            "L": self.L,
            "n_iters_target": cfg.n_iters,
            "n_iters_actual": ran_to,
            "stopped_early": stopped_early,
            "stop_iter": stop_iter,
            "stop_reason": stop_reason,
            "wall_clock_seconds": time.time() - t0,
            "cfg": asdict(cfg),
        }, indent=2))
        print(f"[done] L={self.L} iters={ran_to}/{cfg.n_iters} "
              f"early_stop={stopped_early} wall={time.time()-t0:.1f}s "
              f"output={self.output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--sweep", type=str, default="configs/sweeps/dense.yaml")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--n_iters", type=int, default=10_000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_azi", type=int, default=64)
    ap.add_argument("--n_pts_per_ray", type=int, default=64)
    args = ap.parse_args()

    cfg = TrainCfg(
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        n_azi=args.n_azi,
        n_pts_per_ray=args.n_pts_per_ray,
    )
    trainer = SingleRoomTrainer(L=args.L, sweep_yaml=args.sweep,
                                output_dir=args.output_dir, cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
