"""P4-5 (manager item 3, D146): is the near-DC term spatially uniform ACROSS a doorway?

The DC head (B5, D101) predicts ONE complex value per room per bin over 0-20 Hz, which assumes the
near-DC compliance term is uniform in space. Across a narrow doorway, sub-rooms A and B may relax
differently. D101's statistic, |spatial mean|^2 / mean |H|^2, over ALL receivers of both sub-rooms,
per bin 0-20 Hz, on the doorway training corpus (`data/track_p3_3fast_B`), stratified by aperture.
1.0 = perfectly uniform. The per-sub-room ratio and the A/B level gap are reported beside it.
"""
import json
from pathlib import Path

import h5py
import numpy as np

MAN = "configs/sweeps_2d_mat/p3_3fast_trackB_manifest.json"
DATA = Path("data/track_p3_3fast_B")
HZ = (0.0, 0.5, 1, 2, 5, 10, 15, 20)
STRATA = (("sealed (a = 0)", lambda r: r["kind"] == "sealed"),
          ("a <= 0.5 m", lambda r: r["kind"] == "aperture" and r["a"] <= 0.5),
          ("0.5 < a <= 1.5 m", lambda r: r["kind"] == "aperture" and 0.5 < r["a"] <= 1.5),
          ("a > 1.5 m", lambda r: r["kind"] == "aperture" and r["a"] > 1.5),
          ("fully open", lambda r: r["kind"] == "open"))


def main():
    rows = [r for r in json.load(open(MAN))["configs"] if r["split"] == "train"]
    out = {}
    print("{:18s} {:>4s} | ".format("stratum", "n") + " ".join("{:>7s}".format("%gHz" % h) for h in HZ) +
          " | A/B level gap dB @2 / 10 / 20 Hz")
    for name, sel in STRATA:
        rs = [r for r in rows if sel(r)]
        U, gap = [], []
        for r in rs:
            fn = DATA / (r.get("filename") or "")
            if not fn.exists():
                continue
            with h5py.File(fn) as f:
                H = np.asarray(f["ism/H_complex"])[:, :41]
                rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
            U.append(np.abs(H.mean(0)) ** 2 / np.mean(np.abs(H) ** 2, axis=0))
            a, b = rx[:, 0] < r["x0"], rx[:, 0] > r["x0"]
            gap.append(20 * np.log10(np.abs(H[b]).mean(0) / np.abs(H[a]).mean(0)))
        U, gap = np.array(U), np.array(gap)
        u = {str(h): float(U[:, int(round(h / 0.5))].mean()) for h in HZ}
        g = {str(h): float(np.abs(gap[:, int(round(h / 0.5))]).mean()) for h in (2, 10, 20)}
        out[name] = {"n": len(U), "uniformity": u, "abs_level_gap_B_minus_A_db": g}
        print("{:18s} {:4d} | ".format(name, len(U)) + " ".join("{:7.3f}".format(u[str(h)]) for h in HZ) +
              " | {:.2f} / {:.2f} / {:.2f}".format(g["2"], g["10"], g["20"]))
    json.dump(out, open("outputs/p4_5/diagnostics/doorway_dc_uniformity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
