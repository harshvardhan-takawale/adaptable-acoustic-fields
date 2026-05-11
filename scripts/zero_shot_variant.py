"""Run a single Track-B inner-loop variant for one (variant, L, run).

Wraps ``aaf.eval.zero_shot.zero_shot_adapt`` with the per-variant kwargs from
``aaf.eval.zero_shot_variants.variant_kwargs``. Output goes under
``outputs/inner_loop_experiments/<variant>/<run>/L<L>/`` (no overwrite of the
Chunk-3.5 sweep zero-shot dirs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aaf.data.dataset_builder import room_filename
from aaf.eval.zero_shot import zero_shot_adapt
from aaf.eval.zero_shot_variants import VARIANT_DESCRIPTIONS, variant_kwargs


REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, required=True,
                    help="B1..B6")
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--run", type=str, default="R6_tiny_lhead",
                    help="run-id of the trained model under outputs/multi_room/sweep/")
    ap.add_argument("--sweep_root", type=str,
                    default=str(REPO_ROOT / "outputs/multi_room/sweep"))
    ap.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "data/track_a"))
    ap.add_argument("--out_root", type=str,
                    default=str(REPO_ROOT / "outputs/inner_loop_experiments"))
    args = ap.parse_args()

    kwargs = variant_kwargs(args.variant)
    train_output_dir = Path(args.sweep_root) / args.run
    target_h5 = Path(args.data_dir) / room_filename(L=float(args.L), W=4.0, alpha=0.15)
    if not target_h5.exists():
        raise FileNotFoundError(f"missing target room HDF5: {target_h5}")
    output_dir = Path(args.out_root) / args.variant / args.run / f"L{args.L}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"# variant={args.variant} ({VARIANT_DESCRIPTIONS[args.variant]})")
    print(f"# run={args.run} L={args.L}")
    print(f"# kwargs={json.dumps(kwargs)}")
    print(f"# output={output_dir}")

    out = zero_shot_adapt(
        train_output_dir=train_output_dir,
        target_h5=target_h5,
        output_dir=output_dir,
        **kwargs,
    )
    print("# done. summary:")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
