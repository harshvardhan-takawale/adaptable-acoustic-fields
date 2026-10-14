"""P4-5 density sweep (manager item 7, D126): the 2x receiver corpus, by SLICING the 4x one.

`data/track_p4_5_rx4x` holds 3200 receivers per training room: indices 0-799 are the baseline's
800 verbatim and 800-3199 were appended from the same 0.06 m candidate pool (D93). The first 1600
are therefore a 2x set nested between the two -- 1x subset 2x subset 4x -- so the sweep varies
density alone. No new simulation: H_complex, receiver_pos and line_of_sight are sliced, n_rx and
nlos_fraction recomputed, every other attribute copied. The first 800 receivers of each output
file are checked against the 1x corpus: positions exactly, H to 1e-6 relative (the 4x corpus
re-simulated them; max recorded difference 1.3e-8).
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

SRC, ONE, OUT, N = Path("data/track_p4_5_rx4x"), Path("data/track_p4_2_shapes"), Path("data/track_p4_5_rx2x"), 1600


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.h5"))
    for f in files:
        with h5py.File(f) as a, h5py.File(OUT / f.name, "w") as b:
            H = np.asarray(a["ism/H_complex"])
            rx = json.loads(a.attrs["receiver_pos"]); los = json.loads(a.attrs["line_of_sight"])
            assert H.shape[0] == len(rx) == len(los) == 3200, (f.name, H.shape)
            b.create_dataset("ism/H_complex", data=H[:N])
            for k, v in a.attrs.items():
                b.attrs[k] = v
            b.attrs["receiver_pos"] = json.dumps(rx[:N])
            b.attrs["line_of_sight"] = json.dumps(los[:N])
            b.attrs["n_rx"] = N
            b.attrs["nlos_fraction"] = float(1.0 - np.mean(los[:N]))
            b.attrs["rx_construction"] = "first {} receivers of {} (indices 0-799 = baseline verbatim)".format(N, SRC / f.name)
        with h5py.File(ONE / f.name) as c, h5py.File(OUT / f.name) as b:
            assert np.array_equal(np.asarray(json.loads(c.attrs["receiver_pos"])),
                                  np.asarray(json.loads(b.attrs["receiver_pos"]))[:800]), f.name
            # the 4x corpus RE-SIMULATED the first 800 (D93); its own record of the difference is
            # h_first800_rel_maxdiff, <= 1.3e-8 (float32 rounding) -- checked here, not assumed
            h1, h2 = np.asarray(c["ism/H_complex"]), np.asarray(b["ism/H_complex"])[:800]
            rel = float(np.max(np.abs(h1 - h2)) / np.max(np.abs(h1)))
            assert rel <= 1e-6, (f.name, rel)
    print("wrote {} files to {} ({} receivers each); first 800 verified against {} (positions exact, H <= 1e-6 rel)".format(len(files), OUT, N, ONE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
