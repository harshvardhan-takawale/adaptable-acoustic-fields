"""P4-7 m3 (D170(g)): precompute the modal prior H_modal and the modal-kernel features per (room, receiver).

One file per room per data set: data/track_p4_7_modal/<data dir name>/<filename>.npz with
  H [n_rx, 601] complex64 (aaf.eval.modal_prior.compute), s [n_rx, 48] / kc [48] (features),
  rx [n_rx, 2] and src [2] (read from the corpus file, so the lookup matches what evaluators pass),
  cond [324] (geom_token, the trainer's / evaluators' own expression) and cond_key, meta (json).
Sets (D170(g) order: M3-fam first, M3-500 second):
  famtrain   G2's 240 training rooms (track_p4_4_family, holdU manifest)
  testdense  the 15 frozen test shapes, dense (Gate-2, per-band r, speckle)
  shapestest the same 15 at their 800 corpus receivers (track_p4_2_shapes; p4_5_eval's sparse set)
  famdense   D111's 10 family rooms (out-of-family report)
  famoffl    the 20 family training rooms with off-lattice receivers (training-room fit, g2-style)
  c500train  G1's 500 training rooms (track_p4_2_shapes)               -- M3-500
  c250offl   the C250 training rooms with off-lattice receivers (G1-style training-room fit) -- M3-500
  python scripts/p4_7_m3_precompute.py --sets famtrain,testdense,shapestest,famdense,famoffl --plan
  python scripts/p4_7_m3_precompute.py --sets ... --idx 0 --chunk 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from aaf.eval.modal_prior import ROOT, compute, cond_key

HOLDU = "configs/sweeps_2d_mat/p4_6_family_holdU_manifest.json"
TEST_MAN = "configs/sweeps_2d_mat/p4_2_shapes_manifest.json"
C500 = "configs/sweeps_2d_mat/p4_4_shapes_500.json"


def _family(man, split):
    from aaf.data.multi_notch import configs_from_rows
    return configs_from_rows(json.load(open(man))["configs"], split=split)


def _shapes(man, split):
    from aaf.data.shape_configs import configs_from_rows
    return configs_from_rows(json.load(open(man))["configs"], split=split)


def work(sets):
    from scripts.build_p4_5_rooms import family_test_rooms
    from scripts.build_p4_6_famtrain_offlat import rooms as famoffl_rooms
    out = []
    for s in sets:
        if s == "famtrain":
            out += [(s, c, "data/track_p4_4_family") for c in _family(HOLDU, "train")]
        elif s == "testdense":
            out += [(s, c, "data/track_p4_5_testdense") for c in _shapes(TEST_MAN, "test")]
        elif s == "shapestest":
            out += [(s, c, "data/track_p4_2_shapes") for c in _shapes(TEST_MAN, "test")]
        elif s == "famdense":
            out += [(s, c, "data/track_p4_5_famdense") for _f, c in family_test_rooms() + family_test_rooms(second=True)]
        elif s == "famoffl":
            out += [(s, c, "data/track_p4_6_famtrain_offlat") for c in famoffl_rooms()]
        elif s == "c500train":
            out += [(s, c, "data/track_p4_2_shapes") for c in _shapes(C500, "train")]
        elif s == "c250offl":
            from scripts.p4_5_train_offlat_fit import rooms as offl_rooms
            out += [(s, c, "data/track_p4_5_offlat") for c in offl_rooms()]
        else:
            raise ValueError(s)
    return out


def build(c, data_dir, force=False):
    from aaf.models.conditioning_2d import build_cond_vector_2d
    out = ROOT / Path(data_dir).name / (c.filename + ".npz")
    if out.exists() and not force:
        return "skip", out
    with h5py.File(Path(data_dir) / c.filename) as f:
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]), dtype=np.float64)
        src = np.asarray(json.loads(f.attrs["source_pos"]), dtype=np.float64)
        n_air = int(f.attrs["n_air_nodes"]) if "n_air_nodes" in f.attrs else None
    d = compute(c.verts, c.L, c.W, rx, src, n_air_expected=n_air)
    cond = build_cond_vector_2d("geom_token", c.L, c.W, c.edge_alphas, x0=getattr(c, "x0", None),
                                a=getattr(c, "a", None), verts=c.verts)
    cond = np.asarray(cond.detach().cpu() if hasattr(cond, "detach") else cond, dtype=np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, H=d["H"], s=d["s"], kc=d["kc"], rx=rx, src=src, cond=cond,
                        cond_key=np.asarray(cond_key(cond)), meta=np.asarray(json.dumps(dict(d["meta"], label=c.label))))
    tmp.replace(out)
    return "built", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    w = work([x for x in a.sets.replace("+", ",").split(",") if x])   # "+" survives sbatch --export
    if a.plan or a.idx is None:
        print("{} rooms -> --array=0-{} at chunk {}".format(len(w), (len(w) - 1) // a.chunk, a.chunk))
        return
    for s, c, dd in w[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
        st, out = build(c, dd)
        m = json.loads(str(np.load(str(out))["meta"])) if st == "built" else {}
        print(st, s, c.label, "modes {} weyl {:.0f} eig {}s".format(m.get("n_modes"), m.get("weyl", 0), m.get("eig_s"))
              if m else "", flush=True)


if __name__ == "__main__":
    main()
