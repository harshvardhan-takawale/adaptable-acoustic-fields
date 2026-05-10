"""GPU memory smoke check for the Chunk-3 multi-room training pipeline.

Mirrors `scripts/single_room_memory_check.py` but instantiates
``INR2D_AutoDecoder(n_rooms=7)`` and tries (n_azi, n_pts_per_ray, batch) ∈
[(64, 32, 16), (64, 32, 8)]. The first config is the spec target; the second
is the Chunk-2 fallback (effective batch 16 via grad_accum=2). If even the
fallback OOMs, append to OPEN_QUESTIONS.md and exit 1.

Output: outputs/memory_check/multi_room_REPORT.md.
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

from aaf.models.inr_2d import INR2D_AutoDecoder
from aaf.renderers.freq_2d import FreqRenderer2D


def run_check(
    n_azi: int,
    n_pts_per_ray: int,
    batch: int,
    n_freq_bins: int = 4097,
    fs: int = 4096,
    n_time_samples: int = 8192,
    n_rooms: int = 7,
    latent_dim: int = 32,
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = INR2D_AutoDecoder(n_rooms=n_rooms, latent_dim=latent_dim,
                                  n_freq_bins=n_freq_bins).cuda()
        renderer = FreqRenderer2D(
            n_azi=n_azi, n_pts_per_ray=n_pts_per_ray, near=1e-3,
            fs=fs, n_time_samples=n_time_samples, use_geometric_attn=False,
        ).cuda()
        rx_pos = torch.tensor([[1.0, 1.0]] * batch, device="cuda")
        tx_pos = torch.tensor([[0.5, 0.5]] * batch, device="cuda")
        room_min = torch.tensor([0.0, 0.0], device="cuda")
        room_max = torch.tensor([6.5, 4.0], device="cuda")
        # Pull a real latent batch from the embedding (different per sample).
        room_ids = torch.arange(batch, device="cuda") % n_rooms
        z_s = model.get_latent(room_ids)
        torch.cuda.synchronize()
        t0 = time.time()
        H_pred = renderer(model, rx_pos, tx_pos, room_min, room_max, z_s=z_s)
        loss = (H_pred.abs() ** 2).mean() + (z_s ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        peak = torch.cuda.max_memory_allocated() / 1e9
        result = {
            "status": "pass",
            "n_azi": n_azi, "n_pts_per_ray": n_pts_per_ray, "batch": batch,
            "max_memory_gb": float(peak),
            "fwd_bwd_seconds": float(t1 - t0),
        }
    except torch.cuda.OutOfMemoryError as e:
        result = {
            "status": "oom",
            "n_azi": n_azi, "n_pts_per_ray": n_pts_per_ray, "batch": batch,
            "error": str(e)[:300],
        }
    except Exception as e:
        result = {
            "status": "error",
            "n_azi": n_azi, "n_pts_per_ray": n_pts_per_ray, "batch": batch,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1500:],
        }
    finally:
        try:
            del model, renderer, H_pred, loss, z_s  # noqa
        except Exception:
            pass
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/memory_check"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("memory check requires CUDA")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"# GPU: {gpu_name} ({gpu_total_gb:.1f} GB)")

    candidates = [(64, 32, 16), (64, 32, 8)]
    results = []
    chosen = None
    chosen_grad_accum = 1
    for n_azi, n_pts, bs in candidates:
        print(f"# trying (n_azi={n_azi}, n_pts={n_pts}, batch={bs})")
        r = run_check(n_azi=n_azi, n_pts_per_ray=n_pts, batch=bs)
        if r["status"] == "pass":
            print(f"  → pass  peak={r['max_memory_gb']:.2f} GB  fwd+bwd={r['fwd_bwd_seconds']:.2f}s")
        else:
            print(f"  → {r['status']}  ({r.get('error', '')[:80]})")
        results.append(r)
        if r["status"] == "pass":
            chosen = r
            chosen_grad_accum = 16 // bs  # effective batch 16
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# GPU memory smoke check — multi-room (Chunk 3)\n",
          f"\n**GPU**: {gpu_name}  \n**Total memory**: {gpu_total_gb:.1f} GB\n",
          "\n## Configurations tried\n",
          "| n_azi | n_pts | batch | status | peak GB | fwd+bwd s |\n",
          "|------:|------:|------:|--------|--------:|----------:|\n"]
    for r in results:
        peak = f"{r['max_memory_gb']:.2f}" if r.get("max_memory_gb") is not None else "—"
        secs = f"{r['fwd_bwd_seconds']:.2f}" if r.get("fwd_bwd_seconds") is not None else "—"
        md.append(f"| {r['n_azi']} | {r['n_pts_per_ray']} | {r['batch']} | "
                  f"{r['status']} | {peak} | {secs} |\n")
    md.append("\n## Chosen configuration\n")
    if chosen is None:
        md.append("**FAIL** — no configuration fit. See OPEN_QUESTIONS.md.\n")
        oq = REPO_ROOT / "OPEN_QUESTIONS.md"
        existing = oq.read_text() if oq.exists() else ""
        oq.write_text(existing + (
            "\n### NEW (Chunk-3 memory check failed): multi-room training does not fit on this GPU.\n"
            f"GPU: {gpu_name} ({gpu_total_gb:.1f} GB). Tried "
            f"{[(r['n_azi'], r['n_pts_per_ray'], r['batch']) for r in results]}. "
            "Need either a bigger GPU (tron) or smaller architecture.\n"
        ))
        (out_dir / "multi_room_REPORT.md").write_text("".join(md))
        (out_dir / "multi_room_result.json").write_text(json.dumps(
            {"status": "fail", "results": results,
             "gpu": gpu_name, "total_gb": gpu_total_gb}, indent=2))
        sys.exit(1)
    md.append(
        f"**PASS** — `n_azi={chosen['n_azi']}`, `n_pts_per_ray={chosen['n_pts_per_ray']}`, "
        f"`batch={chosen['batch']}` (effective batch 16 via "
        f"`grad_accum_steps={chosen_grad_accum}`). Peak working set "
        f"**{chosen['max_memory_gb']:.2f} GB** (fwd+bwd "
        f"{chosen['fwd_bwd_seconds']:.2f}s).\n"
    )
    (out_dir / "multi_room_REPORT.md").write_text("".join(md))
    (out_dir / "multi_room_result.json").write_text(json.dumps(
        {"status": "pass", "chosen": chosen, "grad_accum_steps": chosen_grad_accum,
         "results": results, "gpu": gpu_name, "total_gb": gpu_total_gb}, indent=2))
    print(f"# wrote {out_dir/'multi_room_REPORT.md'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
