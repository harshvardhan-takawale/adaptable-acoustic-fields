"""P4-5: test D80's reversal mechanism instead of asserting it.

D102 found the attention arms' off-lattice accuracy FALLS with corpus size (C92 0.7545, C250
0.6109, C500 0.5641) and offered, untested: "more rooms make each lattice point a receiver in more
of them". Across the three corpora that count is collinear with corpus size, so a 3-point plot
cannot distinguish it from any other corpus-size effect (e.g. D80's own draws-per-shape).

THE WITHIN-CORPUS TEST. Lattice points near room edges lie inside fewer training rooms than
central ones, so the per-point receiver count varies WITHIN a corpus. Location is a confound --
central points may simply be easier -- so each on-lattice test point is compared with its own
off-lattice neighbours (the dense test lattice is 0.12 + 0.08 k, so an on-lattice point's four axis
neighbours lie 8 cm away, all off the 0.06 lattice; 8 cm is lambda/14 at 300 Hz -- same location,
not memorised): ADVANTAGE = neighbours' |dB error| minus the point's. If the count drives memorisation, the advantage rises with the count,
and -- the sharper form -- points with the SAME count carry the same advantage whichever corpus
they come from. The matched-count comparison (counts 10-20, where all three corpora overlap) is
bootstrapped over the 15 test shapes, since points within a room are not independent.
Error = mean |dB(P) - dB(T)| over the six Gate-2 mode bins of each room (D80's metric), and
separately over 100-200 Hz. The count is TRAINED receivers only: `val_rx_indices` are excluded.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from aaf.data.shape_configs import configs_from_rows
from aaf.eval.modal_projection import enumerate_modes
from scripts.p4_5_gate2_dense import MANIFEST, on_lattice

CORPORA = {"C92": ("configs/sweeps_2d_mat/p4_3_shapes_manifest.json", "P4-4_C92"),
           "C250": ("configs/sweeps_2d_mat/p4_4_shapes_250.json", "P4-4_C250"),
           "C500": ("configs/sweeps_2d_mat/p4_4_shapes_500.json", "P4-4_C500")}
MATCH = (10, 20)                                     # counts present in all three corpora
SUMMARY = {r["name"]: r for r in json.load(open("outputs/p4_5/gate2_dense/SUMMARY.json"))}
NBR = (0.079, 0.081)                                 # the 4 axis neighbours, 8 cm, off-lattice


def val_rx_indices(n_rx):
    """`aaf.train.multi_room_2d_mat.val_rx_indices` for the 800-receiver corpora, inlined so this
    CPU script does not import tinycudann; asserted against nothing else here, so kept literal."""
    assert n_rx == 800, n_rx
    return tuple(range(3, n_rx, 8))


def key(p):
    return (int(round(p[0] * 100)), int(round(p[1] * 100)))      # cm grid, exact for 0.02 nodes


def receiver_counts(manifest):
    cnt = {}
    for c in configs_from_rows(json.load(open(manifest))["configs"], split="train"):
        with h5py.File(Path("data/track_p4_2_shapes") / c.filename) as f:
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
            held = set(val_rx_indices(len(rx)))
            for i, p in enumerate(rx):
                if i in held:
                    continue
                k = key(p)
                cnt[k] = cnt.get(k, 0) + 1
    return cnt


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-30))


def main() -> int:
    test = configs_from_rows(json.load(open(MANIFEST))["configs"], split="test")
    rep = {}
    for name, (man, cache) in CORPORA.items():
        cnt = receiver_counts(man)
        z = np.load("outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz".format(cache))
        rows = []
        for i, c in enumerate(test):
            P, T, rx = z["P%d" % i][:, :601], z["T%d" % i][:, :601], z["rx%d" % i]
            bins = [int(round(m.f / 0.5)) for m in enumerate_modes(c.L, c.W, f_max=200.0)[:6]]
            e_modes = np.abs(db(P[:, bins]) - db(T[:, bins])).mean(1)
            e_hf = np.abs(db(P[:, 200:400]) - db(T[:, 200:400])).mean(1)
            on = on_lattice(rx)
            off_idx = np.flatnonzero(~on)
            for j in np.flatnonzero(on):
                d = np.hypot(*(rx[off_idx] - rx[j]).T)
                nb = off_idx[(d >= NBR[0]) & (d <= NBR[1])]
                if len(nb) < 2:
                    continue
                rows.append((cnt.get(key(rx[j]), 0), e_modes[nb].mean() - e_modes[j],
                             e_hf[nb].mean() - e_hf[j], i))
        a = np.array(rows)
        q = np.quantile(a[:, 0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        tab = []
        for lo, hi in zip(q[:-1], q[1:]):
            s = (a[:, 0] >= lo) & (a[:, 0] <= hi)
            tab.append({"count_range": [float(lo), float(hi)], "n_points": int(s.sum()),
                        "advantage_db_modes": float(a[s, 1].mean()), "advantage_db_100_200": float(a[s, 2].mean())})
        rho_m = float(np.corrcoef(np.argsort(np.argsort(a[:, 0])), np.argsort(np.argsort(a[:, 1])))[0, 1])
        rho_h = float(np.corrcoef(np.argsort(np.argsort(a[:, 0])), np.argsort(np.argsort(a[:, 2])))[0, 1])
        mw = (a[:, 0] >= MATCH[0]) & (a[:, 0] <= MATCH[1])
        matched = {}
        for col, lab in ((1, "modes"), (2, "100_200")):
            rooms = np.unique(a[mw, 3])
            per = {r: a[mw & (a[:, 3] == r), col] for r in rooms}
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(10000):
                pick = rng.choice(rooms, len(rooms))
                bs.append(np.concatenate([per[r] for r in pick]).mean())
            matched[lab] = [float(a[mw, col].mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]
        pen = SUMMARY[cache]
        rep[name] = {"n_on_lattice_points": len(a), "mean_count": float(a[:, 0].mean()),
                     "gate2_dense_in_slab_on_minus_off": pen["on_in"] - pen["off_in"],
                     "gate2_dense_out_slab_on_minus_off": pen["on_out"] - pen["off_out"],
                     "matched_count_window": list(MATCH), "n_matched": int(mw.sum()),
                     "matched_count_advantage_db_ci": matched,
                     "points": a.tolist(),
                     "mean_advantage_db_modes": float(a[:, 1].mean()), "mean_advantage_db_100_200": float(a[:, 2].mean()),
                     "spearman_count_vs_advantage_modes": rho_m, "spearman_count_vs_advantage_100_200": rho_h,
                     "by_count_quintile": tab}
        print("== {}: {} on-lattice test points, mean training-receiver count {:.1f}".format(name, len(a), a[:, 0].mean()))
        print("   memorisation advantage (neighbours' |dB err| - point's): Gate-2 modes {:+.3f} dB, 100-200 Hz {:+.3f} dB".format(
            a[:, 1].mean(), a[:, 2].mean()))
        for t in tab:
            print("   count {:5.1f}-{:5.1f}  n {:4d}  advantage modes {:+.3f} dB   100-200 Hz {:+.3f} dB".format(
                *t["count_range"], t["n_points"], t["advantage_db_modes"], t["advantage_db_100_200"]))
        print("   Spearman(count, advantage): modes {:+.3f}   100-200 Hz {:+.3f}".format(rho_m, rho_h))
        print("   MATCHED count {}-{} (n {}): advantage modes {:+.3f} [{:+.3f}, {:+.3f}]  100-200 Hz {:+.3f} [{:+.3f}, {:+.3f}] dB".format(
            *MATCH, int(mw.sum()), *matched["modes"], *matched["100_200"]))
        print("   Gate-2 dense off-lattice penalty (on - off): in-slab {:+.3f}, out-slab {:+.3f}".format(
            rep[name]["gate2_dense_in_slab_on_minus_off"], rep[name]["gate2_dense_out_slab_on_minus_off"]))
    figure(rep)
    for r in rep.values():
        r.pop("points")
    json.dump(rep, open("outputs/p4_5/gate2_dense/D80_MECHANISM.json", "w"), indent=1)
    return 0


def figure(rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"C92": "0.6", "C250": "0.25", "C500": "#D55E00"}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), dpi=160)
    mc = [rep[n]["mean_count"] for n in CORPORA]
    ax[0].plot(mc, [rep[n]["gate2_dense_in_slab_on_minus_off"] for n in CORPORA], "o-", color="k", label="in-slab")
    ax[0].plot(mc, [rep[n]["gate2_dense_out_slab_on_minus_off"] for n in CORPORA], "s--", color="0.5", label="out-of-slab")
    for n, x in zip(CORPORA, mc):
        ax[0].annotate(n, (x, rep[n]["gate2_dense_in_slab_on_minus_off"]), textcoords="offset points", xytext=(5, 5))
    ax[0].set_xlabel("mean # training rooms in which a test lattice\npoint is a TRAINED receiver")
    ax[0].set_ylabel("Gate-2 dense: on-lattice minus off-lattice Pearson")
    ax[0].set_title("ACROSS corpora\n(3 points; collinear with corpus size)", loc="left", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
    for k, (c, lab) in enumerate(((1, "Gate-2 mode bins"), (2, "100-200 Hz"))):
        a_ = ax[1 + k]
        for n in CORPORA:
            p = np.asarray(rep[n]["points"])
            edges = np.unique(np.quantile(p[:, 0], np.linspace(0, 1, 9)))
            mid, val = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = (p[:, 0] >= lo) & (p[:, 0] <= hi)
                mid.append(p[s, 0].mean()); val.append(p[s, c].mean())
            a_.plot(mid, val, "o-", color=col[n], lw=2, label=n)
        a_.axvspan(*MATCH, color="0.92", zorder=0)
        a_.set_xlabel("# training rooms in which THIS lattice\npoint is a trained receiver")
        a_.set_ylabel("memorisation advantage (dB)\n= 8-cm neighbours' |err| - point's |err|")
        a_.set_title("WITHIN corpora, {}\n(octile bins; grey = matched-count window)".format(lab), loc="left", fontsize=10)
        a_.legend(fontsize=8); a_.grid(alpha=0.25)
    fig.suptitle("D80 mechanism test: if per-point receiver count drove memorisation, the three corpora "
                 "would lie on ONE curve in the right-hand panels", fontsize=11)
    fig.tight_layout()
    fig.savefig("outputs/p4_5/gate2_dense/D80_MECHANISM.png", facecolor="white")


if __name__ == "__main__":
    raise SystemExit(main())
