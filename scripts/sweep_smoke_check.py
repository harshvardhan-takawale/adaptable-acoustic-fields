"""Chunk-3.5 smoke check: short R0 training run + 5 assertions.

Runs the production trainer with R0's config but n_iters=500, val_every=200,
ckpt_every=500. Then asserts:

  1. Training completed (n_iters_actual == 500).
  2. No NaN/Inf in any scalar.
  3. Validation ran at iter 500.
  4. L_lhead is logged and finite (R0 has l_head_weight > 0).
  5. The iter-500 checkpoint exists, loads cleanly, and contains an l_head
     submodule + latent embedding.
  6. Total loss decreased monotonically from the first to the last train log
     (allowing some batch noise — we just check end < start).

If any assertion fails, write outputs/multi_room/sweep/smoke_FAILED.txt and
exit 1 so the orchestrator's afterok dependency aborts the rest of the sweep.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = REPO_ROOT / "outputs/multi_room/sweep/_smoke_R0"


def _run_training():
    """Spawn the trainer with R0 cfg overridden for a short run."""
    # Clean slate.
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    # Read R0 YAML, override the budget fields, write a temporary smoke YAML.
    import yaml
    r0_path = REPO_ROOT / "configs/sweep/R0_central.yaml"
    cfg = yaml.safe_load(open(r0_path))
    cfg["run_id"] = "_smoke_R0"
    cfg["n_iters"] = 500
    cfg["val_every"] = 200
    cfg["ckpt_every"] = 500
    smoke_yaml = SMOKE_DIR / "smoke_cfg.yaml"
    with open(smoke_yaml, "w") as f:
        yaml.safe_dump(cfg, f)

    cmd = [
        sys.executable, "-m", "aaf.train.multi_room",
        "--config", str(smoke_yaml),
        "--output_dir", str(SMOKE_DIR),
    ]
    print(f"# running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"smoke training subprocess failed (rc={proc.returncode})")
    return proc


def _assert(cond: bool, msg: str, failures: list):
    if not cond:
        print(f"  FAIL: {msg}")
        failures.append(msg)
    else:
        print(f"  ok:   {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/multi_room/sweep"))
    args = ap.parse_args()
    out_root = Path(args.out_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("smoke check requires CUDA")

    _run_training()

    failures: list[str] = []

    # 1. Training completed.
    train_meta_path = SMOKE_DIR / "train_meta.json"
    _assert(train_meta_path.exists(), "train_meta.json present", failures)
    if not train_meta_path.exists():
        sys.exit(_finalize(failures, out_root))
    meta = json.loads(train_meta_path.read_text())
    _assert(int(meta["n_iters_actual"]) == 500,
            f"n_iters_actual == 500 (got {meta['n_iters_actual']})", failures)

    # 2. No NaN/Inf in scalars.
    scalars_path = SMOKE_DIR / "scalars.json"
    _assert(scalars_path.exists(), "scalars.json present", failures)
    if not scalars_path.exists():
        sys.exit(_finalize(failures, out_root))
    scalars = json.loads(scalars_path.read_text())
    bad = []
    for r in scalars:
        for k, v in r.items():
            if isinstance(v, float) and not math.isfinite(v):
                bad.append((r.get("phase"), r.get("iter"), k, v))
    _assert(not bad, f"all scalar values finite (offenders: {bad[:3]})", failures)

    # 3. Validation ran at iter 500.
    vals = [r for r in scalars if r.get("phase") == "val"]
    val_iters = sorted({r["iter"] for r in vals})
    _assert(500 in val_iters, f"val ran at iter 500 (val_iters={val_iters})", failures)

    # 4. L_lhead logged + finite at the last train log (R0 has l_head_weight > 0).
    trains = [r for r in scalars if r.get("phase") == "train"]
    last_train = trains[-1] if trains else {}
    _assert("L_lhead" in last_train,
            f"train scalar contains L_lhead key (keys={list(last_train.keys())[:8]}...)",
            failures)
    if "L_lhead" in last_train:
        _assert(math.isfinite(last_train["L_lhead"]),
                f"L_lhead is finite (got {last_train['L_lhead']})", failures)

    # Also check it's nontrivial (not stuck at 0 immediately).
    l_lhead_present = any(r.get("L_lhead", 0.0) > 0 for r in trains)
    _assert(l_lhead_present, "L_lhead is positive in at least one train log", failures)

    # 5. Checkpoint exists + loads + contains l_head + latents.
    ckpt = SMOKE_DIR / "ckpt_iter0000500.pt"
    _assert(ckpt.exists(), f"checkpoint {ckpt.name} exists", failures)
    if ckpt.exists():
        try:
            state = torch.load(ckpt, map_location="cpu")
            _assert("model" in state, "checkpoint has 'model' key", failures)
            keys = list(state["model"].keys())
            has_lhead = any(k.startswith("l_head.") for k in keys)
            _assert(has_lhead,
                    f"checkpoint contains l_head.* params (sample keys: {keys[:5]})",
                    failures)
            has_latents = any(k.startswith("latents.") for k in keys)
            _assert(has_latents,
                    f"checkpoint contains latents.* params", failures)
        except Exception as e:
            _assert(False, f"ckpt load failed: {type(e).__name__}: {e}", failures)

    # 6. Total loss decreased.
    if len(trains) >= 2:
        first = trains[0]["loss"]
        last = trains[-1]["loss"]
        _assert(last < first,
                f"train loss decreased: {first:.4f} → {last:.4f}", failures)

    sys.exit(_finalize(failures, out_root))


def _finalize(failures: list, out_root: Path) -> int:
    if failures:
        flag = out_root / "smoke_FAILED.txt"
        out_root.mkdir(parents=True, exist_ok=True)
        flag.write_text("Chunk-3.5 smoke check failed:\n" + "\n".join(f"  - {m}" for m in failures))
        print(f"\nSMOKE CHECK FAILED. {len(failures)} failure(s). See {flag}.")
        return 1
    flag = out_root / "smoke_PASSED.txt"
    out_root.mkdir(parents=True, exist_ok=True)
    flag.write_text("ok")
    print(f"\nSMOKE CHECK PASSED. {flag} written.")
    return 0


if __name__ == "__main__":
    main()
