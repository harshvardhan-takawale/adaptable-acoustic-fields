"""Run zero-shot adaptation on (run, L) using the Track B winner's kwargs.

Reads ``outputs/inner_loop_experiments/best_variant.txt`` for the variant ID +
kwargs to apply, and runs ``zero_shot_adapt`` on the requested (run, L). Output
goes to ``outputs/multi_room/sweep/<run>/zero_shot_<variant>/L<L>/`` so it
doesn't collide with the B1-baseline run sitting at the standard
``zero_shot/L<L>/`` path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aaf.data.dataset_builder import room_filename
from aaf.eval.zero_shot import zero_shot_adapt


REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True,
                    help="run-id (e.g. C1_film, C2_latent_jitter)")
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--sweep_root", type=str,
                    default=str(REPO_ROOT / "outputs/multi_room/sweep"))
    ap.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "data/track_a"))
    ap.add_argument("--best_variant_file", type=str,
                    default=str(REPO_ROOT / "outputs/inner_loop_experiments/best_variant.txt"))
    args = ap.parse_args()

    payload = json.loads(Path(args.best_variant_file).read_text())
    variant = payload["variant"]
    kwargs = payload["kwargs"]
    print(f"# best variant: {variant}")
    print(f"# kwargs: {json.dumps(kwargs)}")

    train_output_dir = Path(args.sweep_root) / args.run
    target_h5 = Path(args.data_dir) / room_filename(L=float(args.L), W=4.0, alpha=0.15)
    if not target_h5.exists():
        raise FileNotFoundError(f"missing target room HDF5: {target_h5}")
    output_dir = Path(args.sweep_root) / args.run / f"zero_shot_{variant}" / f"L{args.L}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"# run={args.run} L={args.L} -> {output_dir}")

    out = zero_shot_adapt(
        train_output_dir=train_output_dir,
        target_h5=target_h5,
        output_dir=output_dir,
        **kwargs,
    )
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
