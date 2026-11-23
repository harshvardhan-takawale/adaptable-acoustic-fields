"""P4-10: integrity check of an assembled variant corpus (STR / MAT): re-simulate a few (variant, slot) at 50 receivers and compare,
band-normalised (bins >= 50), against the memmap; also confirm the index polygon, size and receivers match the manifest.
  PYTHONPATH=$PWD python scripts/p4_10_verify_variants.py --kind str|mat [--n 6]"""
import argparse
import json

import numpy as np

from scripts.p4_10_build_ms import MAN as MS_MAN, room, rows, sim
from scripts.p4_10_build_variants import MAN, MM, geometry_of, str_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    d = MM[a.kind]
    idx = json.loads((d / "index.json").read_text())
    M, MS, R = json.loads(MAN[a.kind].read_text())["configs"], json.loads(MS_MAN.read_text())["configs"], rows()
    mm = np.load(d / "H.npy", mmap_mode="r")
    rx_all, src_all = np.load(d / "rx.npy"), np.load(d / "src.npy")
    n = len(M)
    worst, bad = 0.0, 0
    for v in np.linspace(0, n - 1, a.n).round().astype(int):
        c = M[v]
        i = c["i"]
        verts = [tuple(x) for x in c["verts"]]
        if a.kind == "str":
            rx, src, extra, L, W = str_points(verts, [MS[i]["slot0"]] + MS[i]["new_sources"], int(v))
        else:
            rx, src = room(R[i])["rx"], np.array([MS[i]["slot0"]] + MS[i]["new_sources"])
            L, W = c["L"], c["W"]
            extra, _ = geometry_of(verts, L, W, c["edge_alphas"])
        assert np.allclose(idx["variants"][v]["verts"], c["verts"]) and np.allclose([idx["variants"][v]["L"], idx["variants"][v]["W"]], [L, W])
        assert np.allclose(rx_all[v], rx) and np.allclose(src_all[v], src), v
        sel = np.linspace(0, 799, 50).round().astype(int)
        for s in (0, 3):
            f = sim({"L": L, "W": W}, extra, src[s], rx[sel])
            t = np.array(mm[v, s][sel])
            e = float(np.linalg.norm((f - t)[:, 50:]) / np.linalg.norm(t[:, 50:]))
            worst = max(worst, e)
            bad += e > 1e-4
            print("variant", int(v), "slot", s, "rel", "{:.2e}".format(e), "alpha" if a.kind == "mat" else "scale",
                  c["edge_alphas"][:3] if a.kind == "mat" else (c["sx"], c["sy"]), flush=True)
    print("VERIFY", "PASS" if bad == 0 else "FAIL", "worst", worst)


if __name__ == "__main__":
    main()
