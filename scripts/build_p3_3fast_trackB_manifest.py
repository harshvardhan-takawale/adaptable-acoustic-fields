"""Freeze the Track 2b (doorway aperture) manifest."""
import hashlib
import json
from pathlib import Path

from aaf.data.aperture_configs import (A_HOLDOUT, A_HOLDOUT_TEST_VALUES, A_NORM, A_RANGE,
                                       A_TEST_VALUES, DIVIDER_ALPHA, L_RANGE, TEST_DOMAINS,
                                       W_RANGE, X0_FRAC_RANGE, enumerate_test_configs,
                                       in_holdout, manifest_rows, sample_train_configs,
                                       sample_train_domains)

doms = sample_train_domains()
tr, te = sample_train_configs(doms), enumerate_test_configs(TEST_DOMAINS)
rows = manifest_rows(tr, te)

# Hard invariants, asserted at FREEZE time so a broken manifest never reaches the cluster.
assert len({r["filename"] for r in rows}) == len(rows), "duplicate filenames in manifest"
bad = [r for r in rows if r["split"] == "train" and not r["sealed"] and in_holdout(r["a"])]
assert not bad, "training apertures inside the hold-out band: {}".format(bad[:3])
n_hold_test = sum(1 for r in rows if r["split"] == "test" and in_holdout(r["a"]))
assert n_hold_test >= 3, "expected >= 3 test apertures in the hold-out band"

man = {
    "schema": "p3_3fast.trackB/1",
    "edit_axis": "doorway aperture width a (metres) of one interior divider at x = x0",
    "cond_coordinate": "sqrt(a) / sqrt({})".format(A_NORM),
    "cond_coordinate_evidence": ("FT-B: pooled r^2 = 0.9870 for the inter-room level "
                                 "difference against sqrt(a); raw a gives 0.905, a^2 0.704. "
                                 "outputs/p3_3fast/trackB/aperture_sweep.json"),
    "sealed_note": ("a = 0 is a TOPOLOGICAL discontinuity, not the small-aperture limit: a "
                    "sealed one-node divider disconnects room B exactly, so H_B == 0 and the "
                    "inter-room level difference is -inf. Rows with sealed = true are kept in "
                    "the dataset but MUST be excluded from every continuous-coordinate fit "
                    "and from training (the trainer filters them via config_kinds)."),
    "L_range": list(L_RANGE), "W_range": list(W_RANGE), "x0_frac_range": list(X0_FRAC_RANGE),
    "a_range": list(A_RANGE), "a_holdout": list(A_HOLDOUT), "a_norm": A_NORM,
    "a_holdout_test_values": list(A_HOLDOUT_TEST_VALUES),
    "a_test_values": list(A_TEST_VALUES),
    "divider_alpha": DIVIDER_ALPHA,
    "train_domains": [list(d) for d in doms],
    "test_domains": [list(d) for d in TEST_DOMAINS],
    "n_train": len(tr), "n_test": len(te),
    "n_train_in_holdout": 0,
    "n_test_in_holdout": n_hold_test,
    "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest(),
    "configs": rows,
}
Path("configs/sweeps_2d_mat").mkdir(parents=True, exist_ok=True)
Path("configs/sweeps_2d_mat/p3_3fast_trackB_manifest.json").write_text(json.dumps(man, indent=1))
print("train {} test {} unique rooms {} test-in-band {} sha {}".format(
    len(tr), len(te), len({r["filename"] for r in rows}), n_hold_test,
    man["rows_sha256"][:12]))
