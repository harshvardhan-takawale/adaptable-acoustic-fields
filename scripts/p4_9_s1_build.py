"""P4-9 S-1 (D181 E8; D185): the single-room 200-300 Hz ceiling -- is R0's fit to its own training rooms at 200-300 Hz
(off-lattice r 0.58-0.73, `outputs/p4_5/train_offlat_fit/R0_120K.json`) capped by the renderer, or by sharing one model
across 1000 rooms?

Writes, per S-1 room, a ONE-ROW manifest in GX's schema (the row copied verbatim from the GX manifest) and a config
GENERATED from P4_8_R0.yaml that differs ONLY in run_id, configs_manifest, n_iters (10K: ~83x R0's per-room exposure),
early_stop_warmup (= n_iters: no early stop), val_max_configs (1) and ckpt_keep_iters ("4000": the trajectory read).
Rooms (D185): R0's LOWEST and MEDIAN 200-300 Hz fit among the 23 D125 fit rooms.
  python scripts/p4_9_s1_build.py   -> configs/sweeps_2d_mat/p4_9_s1_<tag>_manifest.json, configs/sweep_2d_mat/P4_9_S1<tag>.yaml
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GX = "configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"
R0_YAML = "configs/sweep_2d_mat/P4_8_R0.yaml"
ROOMS = {"lo": "L6.16_W4.56_d0.34_w0.48.h5", "med": "L5.74_W4.78_d0.62_w2.32.h5"}
N_ITERS = 10000


def main():
    gx = json.load(open(GX))
    base = open(R0_YAML).read().splitlines()
    for tag, fn in ROOMS.items():
        rows = [r for r in gx["configs"] if r["filename"] == fn]
        assert len(rows) == 1 and rows[0]["split"] == "train", fn
        man = {k: v for k, v in gx.items() if k not in ("configs", "rows_sha256", "n_train")}
        man.update({"corpus": "S-1 single room {} (row copied from {}, D185)".format(fn, GX), "n_train": 1,
                    "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                    "configs": rows})
        mp = Path("configs/sweeps_2d_mat/p4_9_s1_{}_manifest.json".format(tag))
        mp.write_text(json.dumps(man, indent=1))
        over = {"run_id": "p4_9_S1{}".format(tag), "configs_manifest": str(mp), "n_iters": N_ITERS,
                "early_stop_warmup": N_ITERS, "val_max_configs": 1, "ckpt_keep_iters": '"4000"'}
        out, seen = ["# P4-9 S-1 (D181 E8, D185): R0's recipe on ONE training room ({}), from scratch, {} steps.".format(
            fn, N_ITERS), "# GENERATED from P4_8_R0.yaml by scripts/p4_9_s1_build.py -- differs ONLY in run_id, "
            "configs_manifest, n_iters,", "# early_stop_warmup, val_max_configs and ckpt_keep_iters. Output: "
            "outputs/p4_9/arms/S1{}.".format(tag)], set()
        for line in base:
            if line.startswith("#"):
                continue
            k = line.split(":", 1)[0].strip()
            if k in over:
                line = "{}: {}".format(k, over[k])
                seen.add(k)
            out.append(line)
        for k in over:
            if k not in seen:
                out.append("{}: {}".format(k, over[k]))
        yp = Path("configs/sweep_2d_mat/P4_9_S1{}.yaml".format(tag))
        yp.write_text("\n".join(out) + "\n")
        print("wrote", mp, man["rows_sha256"][:16], yp)


if __name__ == "__main__":
    main()
