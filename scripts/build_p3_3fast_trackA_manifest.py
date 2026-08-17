"""Freeze the Track 1 manifest."""
import json
from pathlib import Path
import yaml
from aaf.data.seg_configs import (HOLDOUT_SEGMENT, M_NORM_SEG, M_RANGE_SEG, N_SEG,
                                  SEGMENT_NAMES, enumerate_test_configs, manifest_rows,
                                  sample_train_configs, segment_index)
import hashlib
gt = [(g["L"], g["W"]) for g in yaml.safe_load(
    open("configs/sweeps_2d_mat/p3_2_train.yaml"))["geometries"]][:20]
gte = [(g["L"], g["W"]) for g in yaml.safe_load(
    open("configs/sweeps_2d_mat/p3_2_test_frozen.yaml"))["geometries"]]
tr, te = sample_train_configs(gt), enumerate_test_configs(gte)
rows = manifest_rows(tr, te)
man = {"schema": "p3_3fast.trackA/1", "n_segments": N_SEG,
       "segment_names": list(SEGMENT_NAMES),
       "holdout_segment": "{}_{}".format(*HOLDOUT_SEGMENT),
       "holdout_index": segment_index(*HOLDOUT_SEGMENT),
       "m_range": list(M_RANGE_SEG), "m_norm": M_NORM_SEG,
       "window_note": ("alpha=0.95 is a matched-impedance segment: the classical open-window "
                       "model, equivalent to a first-order absorbing boundary. It carries NO "
                       "radiation reactance and NO edge diffraction."),
       "n_train": len(tr), "n_test": len(te),
       "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest(),
       "configs": rows}
Path("configs/sweeps_2d_mat").mkdir(parents=True, exist_ok=True)
Path("configs/sweeps_2d_mat/p3_3fast_trackA_manifest.json").write_text(json.dumps(man, indent=1))
print("train {} test {} unique rooms {} sha {}".format(
    len(tr), len(te), len({r["filename"] for r in rows}), man["rows_sha256"][:12]))
