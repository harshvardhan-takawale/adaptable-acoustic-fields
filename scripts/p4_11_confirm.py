"""P4-11 confirmation set (D201 Q32): a fresh 16-room set by the P4-9 rule (8 U + 8 C3), new seed, frozen before any P4-11 number.

Same construction as `scripts/p4_9_build_extra_rooms.py` (D182 (a)), seed 20261124, and additionally excluding every FAM,
frozen-unseen and P4-9 confirmation filename. The manifest sha is recorded in a D-entry before the set is scored.
  python scripts/p4_11_confirm.py -> configs/sweeps_2d_mat/p4_11_confirm_manifest.json (manifest only; the dense FDTD is built later)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEED = 20261124
MAN = Path("configs/sweeps_2d_mat/p4_11_confirm_manifest.json")
OLD = "configs/sweeps_2d_mat/p4_9_confirm_manifest.json"
UNSEEN = "configs/sweeps_2d_mat/p4_8_unseen_manifest.json"


def main():
    from aaf.data.multi_notch import sample_family_shapes
    from scripts.build_p4_4_family_manifest import N_FAM_TEST_IN_SLAB
    from scripts.build_p4_8_unseen import FAM_MANIFEST, U_PICKS, draw_c3, row
    fam, frozen, p49 = (json.load(open(p))["configs"] for p in (FAM_MANIFEST, UNSEEN, OLD))
    used = {r["filename"] for r in fam} | {r["filename"] for r in frozen} | {r["filename"] for r in p49}
    test = sample_family_shapes(10, split="test", seed=SEED, n_in_slab=N_FAM_TEST_IN_SLAB, exclude=sorted(used))
    us = sorted([c for c in test if c.kind == "U"], key=lambda c: (c.d_hat, c.filename))
    assert len(us) == 10, len(us)
    u = [(i, us[i]) for i in U_PICKS]
    c3, tried = draw_c3(8, seed=SEED)
    assert not {c.filename for _, c in u} & used and not {c.filename for c in c3} & used
    rows = [row(c, "U", {"u_sorted_index": i, "origin": "p4_11_confirm"}) for i, c in u]
    rows += [row(c, "C3", {"draw": j}) for j, c in enumerate(c3)]
    assert len({r["filename"] for r in rows}) == 16
    man = {"schema": "p4_1.poly/1", "set": "P4-11 confirmation set (D201 Q32)", "built": "2026-11-23",
           "u_rule": "P4-9's rule (D182 (a)) with seed {}, also excluding the P4-9 confirmation set; picks (4, 5, 3, 6, 2, 7, 1, 8)".format(SEED),
           "c3_rule": "NW+NE+SE, first 8 valid draws of default_rng([{}, k]) ({} tried)".format(SEED, tried),
           "use": "for P4-11; not to be scored before a P4-11 spec says so", "n_excluded": len(used),
           "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), "configs": rows}
    MAN.write_text(json.dumps(man, indent=1))
    print("wrote", MAN, man["rows_sha256"][:16], "excluded", len(used), "c3 tried", tried)
    for r in rows:
        print("  ", r["family"], r["label"], "d_hat {:.3f}".format(r["d_hat"]))


if __name__ == "__main__":
    main()
