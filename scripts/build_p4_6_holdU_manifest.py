"""P4-6 g2 (D160, user decision): FAM's family corpus with the U family HELD OUT as a whole.

FAM (P4_4_FAMILY) trained on all five families (rect, L, U, T, DN; 60 rooms each) and held out only
10 instances per family, so it had no held-out family. This manifest is FAM's, unchanged except that
every U row -- its 60 training rooms and its 10 test rooms -- gets split "test_family". The trainer
reads split == "train", so U is never seen; the other four families keep FAM's own train / test
split (240 train, 40 in-family test). No data is rebuilt.
"""
import json

from aaf.data.shape_configs import rows_sha256

SRC = "configs/sweeps_2d_mat/p4_4_family_manifest.json"
OUT = "configs/sweeps_2d_mat/p4_6_family_holdU_manifest.json"

m = json.load(open(SRC))
rows = [dict(r) for r in m["configs"]]
for r in rows:
    if r["kind"] == "U":
        r["split_fam"] = r["split"]          # FAM's own split, kept for reference
        r["split"] = "test_family"
n = {s: sum(r["split"] == s for r in rows) for s in ("train", "test", "test_family")}
assert n == {"train": 240, "test": 40, "test_family": 70}, n
m.update({"schema": m["schema"], "configs": rows, "rows_sha256": rows_sha256(rows),
          "derived_from": SRC, "derived_from_sha256": m["rows_sha256"], "holdout_family": "U",
          "n_train": n["train"], "n_test": n["test"], "n_test_family": n["test_family"],
          "why_p4_6": "g2: does the fixed recipe (PB3) transfer to a family it never saw? U is the "
                      "roadmap blocker (D111 dense U room, r 0.06 at 100 Hz)."})
json.dump(m, open(OUT, "w"), indent=1)
print(OUT, n)
