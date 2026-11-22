"""P4-9 S-4 (D181 E8, c5): parity of the sm_86/89 tinycudann build (env aaf_sm89) against aaf's (compute capability 52).

aaf's wheel reports compute capability 52, so tinycudann's FullyFusedMLP (R0 configures one 2x128 MLP so) falls back;
the sm_86/89 build runs it natively. No P4-9 arm switches environments; this decides adoption from P4-10 (E8).
  --render ENV      R0 @120K on the 15 in-family dense test rooms (p4_5_gate2_dense.render_all, verbatim) in THIS
                    env -> outputs/p4_9/s4/render_<ENV>.npz (the env is named by the caller; both run on one node)
  --compare         per band (25-100 / 100-200 / 200-300 Hz), off-lattice r (the frozen scorer), both envs, the
                    difference (parity bar: |delta| <= 0.003 in every band, c5) and the max relative field difference;
                    throughput from the two trainer logs (s/it of R0's recipe, 300 steps each, same node)
                    -> outputs/p4_9/s4_parity.{json,md}
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

OUT = Path("outputs/p4_9/s4")
CK = Path("outputs/p4_8/arms/R0/ckpt_iter0120000.pt")
BANDS = ("25-100", "100-200", "200-300")


def render(env):
    import torch

    import tinycudann as tcnn
    from scripts.p4_5_gate2_dense import render_all
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    cache = OUT / "render_{}.npz".format(env)
    rooms = render_all(CK, "data/track_p4_5_testdense", cache, dev)
    meta = {"env": env, "gpu": torch.cuda.get_device_name(0), "cc": list(torch.cuda.get_device_capability()),
            "tcnn_file": tcnn.__file__, "n_rooms": len(rooms)}
    (OUT / "render_{}.json".format(env)).write_text(json.dumps(meta, indent=1))
    print(meta, flush=True)


def _sps(log):
    v = [float(m) for m in re.findall(r"\(([0-9.]+)s/it\)", Path(log).read_text())]
    return v[-1] if v else float("nan")


def compare(log_a, log_b):
    from scripts.p4_5_gate2_dense import on_lattice
    from scripts.p4_7_m1_modal import scores
    za, zb = np.load(OUT / "render_aaf.npz"), np.load(OUT / "render_sm89.npz")
    n = sum(1 for k in za.files if k.startswith("P"))
    rows = []
    for i in range(n):
        Pa, Pb, T, rx = za["P%d" % i], zb["P%d" % i], za["T%d" % i], za["rx%d" % i]
        assert np.array_equal(T, zb["T%d" % i]) and np.array_equal(rx, zb["rx%d" % i])
        off = ~on_lattice(rx)
        sa, sb = scores(Pa[off], T[off]), scores(Pb[off], T[off])
        rel = float(np.max(np.abs(Pb[:, :601] - Pa[:, :601])) / np.max(np.abs(Pa[:, :601])))
        rows.append({"room": i, "aaf": {b: sa[b] for b in BANDS}, "sm89": {b: sb[b] for b in BANDS}, "max_rel_diff": rel})
    res = {"bands": {}, "rooms": rows, "meta": {e: json.load(open(OUT / "render_{}.json".format(e)))
                                                for e in ("aaf", "sm89")}}
    for b in BANDS:
        a = float(np.mean([r["aaf"][b] for r in rows]))
        s = float(np.mean([r["sm89"][b] for r in rows]))
        res["bands"][b] = {"aaf": a, "sm89": s, "delta": s - a, "parity": abs(s - a) <= 0.003}
    res["max_rel_field_diff"] = max(r["max_rel_diff"] for r in rows)
    res["throughput_s_per_it"] = {"aaf": _sps(log_a), "sm89": _sps(log_b)}
    res["parity"] = all(v["parity"] for v in res["bands"].values())
    Path("outputs/p4_9/s4_parity.json").write_text(json.dumps(res, indent=1))
    t = res["throughput_s_per_it"]
    md = ["# P4-9 S-4: sm_86/89 tinycudann build -- parity and throughput (D181 E8, c5)", "",
          "R0 @120K, 15 in-family dense test rooms, off-lattice r (frozen scorer); both envs rendered on one node ({}).".format(
              res["meta"]["aaf"]["gpu"]), "",
          "| band | aaf (cc 52 wheel) | aaf_sm89 | delta | parity (abs <= 0.003) |", "|---|---|---|---|---|"]
    for b in BANDS:
        v = res["bands"][b]
        md.append("| {} Hz | {:.4f} | {:.4f} | {:+.4f} | {} |".format(b, v["aaf"], v["sm89"], v["delta"],
                                                                    "yes" if v["parity"] else "NO"))
    md += ["", "Max relative field difference (bins 0-600, any receiver): {:.2e}.".format(res["max_rel_field_diff"]),
           "Training throughput (R0's recipe, 300 steps, same node): aaf {:.3f} s/it, aaf_sm89 {:.3f} s/it ({:.2f}x).".format(
               t["aaf"], t["sm89"], t["aaf"] / t["sm89"]), "",
           "**Parity {}.** Adoption is a P4-10 decision (E8); no P4-9 arm switches environments.".format(
               "HOLDS" if res["parity"] else "FAILS")]
    Path("outputs/p4_9/s4_parity.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render")
    ap.add_argument("--compare", nargs=2, metavar=("LOG_AAF", "LOG_SM89"))
    a = ap.parse_args()
    if a.render:
        render(a.render)
    if a.compare:
        compare(*a.compare)


if __name__ == "__main__":
    main()
