"""P4-8 S2 (D174(g)): CX1000 -- 1000 more rooms under CX500's rules (aaf.data.cx_rooms), the continuation of the
prefix-stable draw: sample_cx_rooms(1500)[500:] (shape_ids 9500-10499). U is excluded by the generator's redraw and
the held-out C3 topology (three corner notches) cannot occur (CX draws at most 2 notches). Same FDTD, receiver and
source construction as CX500 (scripts/build_p4_7_cx500.py, whose `build` is reused with this corpus's paths).
  python scripts/build_p4_8_cx1000.py --manifest
  python scripts/build_p4_8_cx1000.py --idx 0 --chunk 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import scripts.build_p4_7_cx500 as bx
from aaf.data.cx_rooms import CX_SEED, CxNotch, CxRoom, sample_cx_rooms

MANIFEST = Path("configs/sweeps_2d_mat/p4_8_cx1000_manifest.json")
OUT = Path("data/track_p4_8_cx1000")


def write_manifest():
    allr = sample_cx_rooms(1500, seed=CX_SEED)
    cx500 = json.loads(bx.MANIFEST.read_text())["configs"]
    assert [r.row(i)["filename"] for i, r in enumerate(allr[:500])] == [r["filename"] for r in cx500], "prefix moved"
    rows = [r.row(i) for i, r in enumerate(allr[500:])]
    assert all(len(r["notches"]) <= 2 and r["kind"] != "cx_U" for r in rows)
    man = {"schema": "p4_1.poly/1", "corpus": "CX1000 (P4-8 S2, D174(g))", "seed": CX_SEED, "n_train": len(rows),
           "rx_seed": bx.RX_SEED_CX, "per_room_source": True, "continuation_of": str(bx.MANIFEST),
           "kinds": dict(Counter(r["kind"] for r in rows)),
           "positions": dict(Counter(n["pos"] for r in rows for n in r["notches"])),
           "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), "configs": rows}
    MANIFEST.write_text(json.dumps(man, indent=1))
    print("wrote", MANIFEST, json.dumps({k: man[k] for k in ("kinds", "positions")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    if a.manifest:
        write_manifest()
        return
    rows = json.loads(MANIFEST.read_text())["configs"]
    bx.OUT = OUT                                                  # build() writes under the module's OUT
    rooms = [CxRoom(r["L"], r["W"], tuple(CxNotch(n["pos"], n["d"], n["w"], n["off"]) for n in r["notches"]),
                    tuple(r["src"]), r["shape_id"]) for r in rows]
    if a.idx is None:
        print("{} rooms -> --array=0-{} at chunk {}".format(len(rooms), (len(rooms) - 1) // a.chunk, a.chunk))
        return
    for r, row in list(zip(rooms, rows))[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
        print(bx.build(r, row), r.filename, flush=True)


if __name__ == "__main__":
    main()
