"""GPU memory smoke check for the 3D single-room training pipeline.

Cascade (DECISIONS.md D12, prioritise n_pts reduction over batch reduction
over n_rays reduction):
    (n_rays_grid=(16,16) → 258 rays, n_pts_per_ray=32, batch=8)
    (n_rays_grid=(16,16) → 258 rays, n_pts_per_ray=16, batch=8)
    (n_rays_grid=(16,16) → 258 rays, n_pts_per_ray=32, batch=4)
    (n_rays_grid=(16,16) → 258 rays, n_pts_per_ray=16, batch=4)

Writes ``outputs/memory_check_3d/REPORT.md`` and ``result.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aaf.models.inr_3d import INR3D_AutoDecoder, INR3D_Single
from aaf.renderers.freq_3d import FreqRenderer3D


def run_check(
    n_azi: int, n_ele: int, n_pts_per_ray: int, batch: int = 8,
    n_freq_bins: int = 4097, fs: int = 4096, n_time_samples: int = 8192,
    mode: str = "single",
    n_rooms: int = 45, latent_dim: int = 16,
) -> dict:
    """Memory smoke check for the 3D single-room or auto-decoder pipeline.

    ``mode='single'`` builds `INR3D_Single` (P2-1 default).
    ``mode='auto_decoder'`` builds `INR3D_AutoDecoder` with `n_rooms` and
    `latent_dim` set (P2-2 multi-room).
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        if mode == "auto_decoder":
            model = INR3D_AutoDecoder(
                n_rooms=n_rooms,
                latent_dim=latent_dim,
                n_freq_bins=n_freq_bins,
                conditioning_type="film",
                latent_jitter_sigma=0.1,
                l_head_enabled=True,
            ).cuda()
        else:
            model = INR3D_Single(n_freq_bins=n_freq_bins).cuda()
        renderer = FreqRenderer3D(
            n_azi=n_azi, n_ele=n_ele, n_pts_per_ray=n_pts_per_ray, near=1e-3,
            fs=fs, n_time_samples=n_time_samples,
            use_geometric_attn=False,
        ).cuda()
        # Plausible "centered" room: 5×4×3 m, receiver near centre, source corner.
        rx_pos = torch.tensor([[2.5, 2.0, 1.5]] * batch, device="cuda")
        tx_pos = torch.tensor([[0.5, 0.5, 0.5]] * batch, device="cuda")
        room_min = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        room_max = torch.tensor([5.0, 4.0, 3.0], device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        if mode == "auto_decoder":
            z_s = model.get_latent(
                torch.zeros(batch, dtype=torch.long, device="cuda")
            )
            H_pred = renderer(model, rx_pos, tx_pos, room_min, room_max, z_s=z_s)
        else:
            H_pred = renderer(model, rx_pos, tx_pos, room_min, room_max)
        loss = (H_pred.abs() ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        peak = torch.cuda.max_memory_allocated() / 1e9
        result = {
            "status": "pass",
            "n_azi": n_azi,
            "n_ele": n_ele,
            "n_rays_total": n_azi * n_ele + 2,
            "n_pts_per_ray": n_pts_per_ray,
            "batch": batch,
            "max_memory_gb": float(peak),
            "fwd_bwd_seconds": float(t1 - t0),
        }
    except torch.cuda.OutOfMemoryError as e:
        result = {
            "status": "oom",
            "n_azi": n_azi, "n_ele": n_ele,
            "n_rays_total": n_azi * n_ele + 2,
            "n_pts_per_ray": n_pts_per_ray,
            "batch": batch,
            "error": str(e)[:300],
        }
    except Exception as e:
        result = {
            "status": "error",
            "n_azi": n_azi, "n_ele": n_ele,
            "n_rays_total": n_azi * n_ele + 2,
            "n_pts_per_ray": n_pts_per_ray,
            "batch": batch,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1500:],
        }
    finally:
        try:
            del model, renderer, H_pred, loss  # noqa
        except Exception:
            pass
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/memory_check_3d"))
    ap.add_argument(
        "--mode", choices=("single", "auto_decoder"), default="single",
        help="single = INR3D_Single (P2-1); auto_decoder = INR3D_AutoDecoder (P2-2).",
    )
    ap.add_argument("--n_rooms", type=int, default=45)
    ap.add_argument("--latent_dim", type=int, default=16)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("memory check requires CUDA")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"# GPU: {gpu_name} ({gpu_total_gb:.1f} GB)")

    # Cascade per D12.
    candidates = [
        (16, 16, 32, 8),
        (16, 16, 16, 8),
        (16, 16, 32, 4),
        (16, 16, 16, 4),
    ]
    results = []
    chosen = None
    for n_azi, n_ele, n_pts, batch in candidates:
        print(f"# trying (mode={args.mode}, n_azi={n_azi}, n_ele={n_ele}, "
              f"n_pts={n_pts}, batch={batch}) → {n_azi*n_ele+2} rays")
        r = run_check(
            n_azi=n_azi, n_ele=n_ele, n_pts_per_ray=n_pts, batch=batch,
            mode=args.mode, n_rooms=args.n_rooms, latent_dim=args.latent_dim,
        )
        print(f"  → {r['status']}", end="")
        if r["status"] == "pass":
            print(f"  peak={r['max_memory_gb']:.2f} GB  "
                  f"fwd+bwd={r['fwd_bwd_seconds']:.2f}s")
        else:
            print(f"  ({r.get('error', 'unknown')[:80]})")
        results.append(r)
        if r["status"] == "pass":
            chosen = r
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# 3D GPU memory smoke check\n",
        f"\n**GPU**: {gpu_name}  \n**Total memory**: {gpu_total_gb:.1f} GB\n",
        "\n## Configurations tried\n",
        "| n_azi | n_ele | n_rays | n_pts | batch | status | peak GB | fwd+bwd s |\n",
        "|------:|------:|-------:|------:|------:|--------|--------:|----------:|\n",
    ]
    for r in results:
        peak = f"{r['max_memory_gb']:.2f}" if r.get("max_memory_gb") is not None else "—"
        secs = f"{r['fwd_bwd_seconds']:.2f}" if r.get("fwd_bwd_seconds") is not None else "—"
        md.append(
            f"| {r['n_azi']} | {r['n_ele']} | {r['n_rays_total']} | "
            f"{r['n_pts_per_ray']} | {r['batch']} | {r['status']} | "
            f"{peak} | {secs} |\n"
        )
    md.append("\n## Chosen configuration\n")
    if chosen is None:
        md.append("**FAIL** — no configuration fit. See OPEN_QUESTIONS.md.\n")
        oq = REPO_ROOT / "OPEN_QUESTIONS.md"
        existing = oq.read_text() if oq.exists() else ""
        oq.write_text(existing + (
            "\n### NEW (P2-1 3D memory check failed): no single-room "
            f"training config fits on this GPU.\nGPU: {gpu_name} "
            f"({gpu_total_gb:.1f} GB). Tried "
            f"{[(r['n_azi'], r['n_ele'], r['n_pts_per_ray'], r['batch']) for r in results]}. "
            "Need either a bigger GPU, a smaller architecture, or chunked "
            "frequency rendering.\n"
        ))
        (out_dir / "REPORT.md").write_text("".join(md))
        (out_dir / "result.json").write_text(json.dumps(
            {"status": "fail", "results": results,
             "gpu": gpu_name, "total_gb": gpu_total_gb}, indent=2))
        sys.exit(1)
    md.append(
        f"**PASS** — `n_azi={chosen['n_azi']}`, `n_ele={chosen['n_ele']}`, "
        f"`n_pts_per_ray={chosen['n_pts_per_ray']}`, `batch={chosen['batch']}` "
        f"({chosen['n_rays_total']} rays). Peak working set "
        f"**{chosen['max_memory_gb']:.2f} GB** "
        f"(fwd+bwd {chosen['fwd_bwd_seconds']:.2f}s).\n"
    )
    (out_dir / "REPORT.md").write_text("".join(md))
    (out_dir / "result.json").write_text(json.dumps(
        {"status": "pass", "chosen": chosen, "results": results,
         "gpu": gpu_name, "total_gb": gpu_total_gb}, indent=2))
    print(f"# wrote {out_dir / 'REPORT.md'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
