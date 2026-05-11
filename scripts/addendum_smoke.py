"""Chunk-3.5+ addendum smoke: short R6 (linear L-head) training run + assertions.

Same shape as scripts/sweep_smoke_check.py but for R6 with n_iters=200, just
to catch any bug in the new ``l_head_arch="linear"`` code path before
launching R6-R8.
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
SMOKE_DIR = REPO_ROOT / "outputs/multi_room/sweep/_smoke_addendum_R6"


def _run_training():
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    import yaml
    src = REPO_ROOT / "configs/sweep/R6_tiny_lhead.yaml"
    cfg = yaml.safe_load(open(src))
    cfg["run_id"] = "_smoke_addendum_R6"
    cfg["n_iters"] = 200
    cfg["val_every"] = 100
    cfg["ckpt_every"] = 200
    smoke_yaml = SMOKE_DIR / "smoke_cfg.yaml"
    with open(smoke_yaml, "w") as f:
        yaml.safe_dump(cfg, f)

    cmd = [
        sys.executable, "-m", "aaf.train.multi_room",
        "--config", str(smoke_yaml), "--output_dir", str(SMOKE_DIR),
    ]
    print(f"# running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"smoke training subprocess failed (rc={proc.returncode})")


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
        raise RuntimeError("addendum smoke requires CUDA")

    _run_training()

    failures: list = []

    # Train completed.
    train_meta_path = SMOKE_DIR / "train_meta.json"
    _assert(train_meta_path.exists(), "train_meta.json present", failures)
    if not train_meta_path.exists():
        sys.exit(_finalize(failures, out_root))
    meta = json.loads(train_meta_path.read_text())
    _assert(int(meta["n_iters_actual"]) == 200,
            f"n_iters_actual == 200 (got {meta['n_iters_actual']})", failures)
    _assert(meta["cfg"].get("l_head_arch") == "linear",
            f"train_meta records l_head_arch=='linear' (got {meta['cfg'].get('l_head_arch')})",
            failures)

    # Scalars finite.
    sc = json.loads((SMOKE_DIR / "scalars.json").read_text())
    bad = []
    for r in sc:
        for k, v in r.items():
            if isinstance(v, float) and not math.isfinite(v):
                bad.append((r.get("phase"), r.get("iter"), k, v))
    _assert(not bad, f"all scalar values finite (offenders: {bad[:3]})", failures)

    # L_lhead present in train logs and finite + reasonable magnitude.
    # NOTE: at iter 200 with weight 0.1 and the linear head's modest learning
    # rate, L_lhead can hover near its random-init L1 ≈ |mean(L_train)| ≈ 4.5 m.
    # We don't assert a downtrend here — the 30K-iter run is what validates
    # convergence. Smoke just checks the code path runs without crashing.
    trains = [r for r in sc if r.get("phase") == "train" and "L_lhead" in r]
    _assert(len(trains) >= 2, "≥ 2 train logs with L_lhead", failures)
    if len(trains) >= 2:
        last = trains[-1]["L_lhead"]
        _assert(math.isfinite(last) and 0.0 <= last < 10.0,
                f"L_lhead is finite and in plausible range [0, 10] m: {last:.3f}",
                failures)

    # Checkpoint loadable + has linear l_head (single weight tensor + bias).
    ckpt = SMOKE_DIR / "ckpt_iter0000200.pt"
    _assert(ckpt.exists(), f"checkpoint {ckpt.name} exists", failures)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        keys = list(state["model"].keys())
        # Linear head has only `l_head.weight` and `l_head.bias`.
        lhead_keys = [k for k in keys if k.startswith("l_head.")]
        _assert(set(lhead_keys) == {"l_head.weight", "l_head.bias"},
                f"linear L-head saves exactly weight+bias (got {lhead_keys})",
                failures)
        # Weight shape [1, latent_dim=8]
        w = state["model"]["l_head.weight"]
        _assert(tuple(w.shape) == (1, 8),
                f"l_head.weight shape (1, 8) (got {tuple(w.shape)})", failures)

    sys.exit(_finalize(failures, out_root))


def _finalize(failures: list, out_root: Path) -> int:
    if failures:
        flag = out_root / "addendum_smoke_FAILED.txt"
        out_root.mkdir(parents=True, exist_ok=True)
        flag.write_text("Chunk-3.5+ addendum smoke failed:\n" + "\n".join(f"  - {m}" for m in failures))
        print(f"\nADDENDUM SMOKE FAILED. {len(failures)} failure(s). See {flag}.")
        return 1
    flag = out_root / "addendum_smoke_PASSED.txt"
    out_root.mkdir(parents=True, exist_ok=True)
    flag.write_text("ok")
    print(f"\nADDENDUM SMOKE PASSED. {flag} written.")
    return 0


if __name__ == "__main__":
    main()
