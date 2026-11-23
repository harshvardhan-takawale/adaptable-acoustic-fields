"""P4-9 2a's one-time confirmation read (D181 c6, D190 context): score a checkpoint on the FROZEN 16-room confirmation set
(configs/sweeps_2d_mat/p4_9_confirm_manifest.json; data/track_p4_9_confirm), by exactly the p4_8_unseen_eval protocol.
  python scripts/p4_9_confirm_eval.py --items S2X_240K=outputs/p4_9/arms/S2X/ckpt_iter0240000.pt+R0_120K=outputs/p4_8/arms/R0/ckpt_iter0120000.pt
"""
import sys
from pathlib import Path

import scripts.p4_8_unseen_eval as u

u.MANIFEST = "configs/sweeps_2d_mat/p4_9_confirm_manifest.json"
u.DATA = Path("data/track_p4_9_confirm")
u.OUT = Path("outputs/p4_9/confirm")
u.CACHE = Path("outputs/p4_9/confirm/cache")


if __name__ == "__main__":
    u.main()
