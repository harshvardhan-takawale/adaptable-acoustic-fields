"""P4-6 Step 0 (D160): the checkpoint KEEP-LIST for finished P4-5 arms, written to the repo BEFORE any
deletion, then verified loadable through `load_model`, then (separately) applied.

Rule (manager, P4-6): for every finished arm under outputs/p4_5/arms/, keep
  * the FINAL checkpoint (highest iteration on disk),
  * the 30K checkpoint,
  * every checkpoint CITED in a D-entry (DECISIONS.md) or a figure manifest
    (outputs/**/FIGURE_MANIFEST*.md), found as
      - an explicit path `outputs/p4_5/arms/<ARM>/ckpt_iter<NNN>.pt`,
      - `<ARM> @<N>K` / `<ARM>@<N>K` / `<ARM>_<N>K`, and
      - every arm named in a list that precedes an `@<N>K` or `at <N>K` on the same line within
        60 characters (e.g. "B1/B1b @30K", "PB3 / W3 interim at 20K") -- deliberately generous: keeping too much costs
        disk, deleting a cited checkpoint costs the record.
Only checkpoints that exist are listed. Corpora, dense test sets and D111's dense family rooms are
never touched by this script (it only reads outputs/p4_5/arms).

  python scripts/p4_6_ckpt_keeplist.py              # write the keep-list (JSON + MD)
  python scripts/p4_6_ckpt_keeplist.py --verify     # GPU: load every kept ckpt through load_model
  python scripts/p4_6_ckpt_keeplist.py --apply      # delete what is NOT kept (requires --verify ok)
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

ARMS = Path("outputs/p4_5/arms")
OUT = Path("outputs/p4_6/housekeeping")
SOURCES = ["DECISIONS.md"] + sorted(glob.glob("outputs/**/FIGURE_MANIFEST*.md", recursive=True))


def inventory():
    inv = {}
    for d in sorted(ARMS.iterdir()):
        its = sorted(int(p.stem.replace("ckpt_iter", "")) for p in d.glob("ckpt_iter*.pt"))
        if its:
            inv[d.name] = its
    return inv


def citations(arms):
    names = sorted(arms, key=len, reverse=True)           # B10s1 before B10 before B1
    tok = r"(?<![A-Za-z0-9])({})(?![A-Za-z0-9])".format("|".join(map(re.escape, names)))
    cited = {a: {} for a in arms}

    def add(a, k, where):
        cited[a].setdefault(k * 1000, set()).add(where)

    for src in SOURCES:
        text = Path(src).read_text(errors="replace")
        for m in re.finditer(r"outputs/p4_5/arms/([A-Za-z0-9]+)/ckpt_iter0*(\d+)\.pt", text):
            if m.group(1) in cited:
                cited[m.group(1)].setdefault(int(m.group(2)), set()).add(src + ":path")
        for m in re.finditer(tok + r"\s*(?:@\s*|_)(\d+)\s*K(?![A-Za-z0-9])", text):
            add(m.group(1), int(m.group(2)), src + ":direct")
        for line in text.splitlines():
            for m in re.finditer(r"(?:@|(?<![A-Za-z])at)\s*(\d+)\s*K(?![A-Za-z0-9])", line):
                pre = line[max(0, m.start() - 60):m.start()]
                for a in re.findall(tok, pre):
                    add(a, int(m.group(1)), src + ":list")
    return cited


def build():
    inv = inventory()
    cited = citations(list(inv))
    rows, keep_bytes, del_bytes = {}, 0, 0
    for arm, its in inv.items():
        keep = {its[-1]: ["final"]}
        if 30000 in its:
            keep.setdefault(30000, []).append("30K")
        for it, why in cited[arm].items():
            if it in its:
                keep.setdefault(it, []).append("cited: " + ", ".join(sorted(why)))
        delete = [i for i in its if i not in keep]
        for i in its:
            sz = (ARMS / arm / "ckpt_iter{:07d}.pt".format(i)).stat().st_size
            if i in keep:
                keep_bytes += sz
            else:
                del_bytes += sz
        rows[arm] = {"on_disk": its, "keep": {str(k): v for k, v in sorted(keep.items())},
                     "delete": delete}
    rep = {"rule": __doc__.split("\n\n")[1], "sources": SOURCES, "arms": rows,
           "keep_gb": round(keep_bytes / 1e9, 2), "delete_gb": round(del_bytes / 1e9, 2)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ckpt_keeplist.json").write_text(json.dumps(rep, indent=1))
    md = ["# P4-6 checkpoint keep-list (written before any deletion)", "",
          "Rule: final + 30K + every checkpoint cited in DECISIONS.md or a figure manifest. "
          "Keep {} GB, delete {} GB.".format(rep["keep_gb"], rep["delete_gb"]), "",
          "| arm | kept (iteration: reason) | deleted |", "|---|---|---|"]
    for arm, r in rows.items():
        kept = "; ".join("{}K: {}".format(int(k) // 1000, " / ".join(
            v if not v.startswith("cited") else "cited" for v in why)) for k, why in r["keep"].items())
        md.append("| {} | {} | {} |".format(arm, kept, " ".join("{}K".format(i // 1000) for i in r["delete"]) or "--"))
    (OUT / "ckpt_keeplist.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def verify():
    import torch

    from aaf.eval.p3_2_eval import load_model
    rep = json.loads((OUT / "ckpt_keeplist.json").read_text())
    dev = torch.device("cuda")
    res, bad = {}, 0
    for arm, r in rep["arms"].items():
        for k in r["keep"]:
            p = ARMS / arm / "ckpt_iter{:07d}.pt".format(int(k))
            try:
                model, renderer, cfg, _meta, it = load_model(p, dev)
                ok = int(it) == int(k)
                res[str(p)] = "ok" if ok else "iter mismatch {}".format(it)
                del model, renderer
                torch.cuda.empty_cache()
            except Exception as e:                       # noqa: BLE001 -- record, do not stop
                res[str(p)] = "FAIL: {}: {}".format(type(e).__name__, e)
            bad += res[str(p)] != "ok"
            print(res[str(p)], p, flush=True)
    (OUT / "ckpt_keeplist_verify.json").write_text(json.dumps({"n": len(res), "n_bad": bad,
                                                               "results": res}, indent=1))
    print("verified {} kept checkpoints, {} bad".format(len(res), bad))


def apply():
    rep = json.loads((OUT / "ckpt_keeplist.json").read_text())
    ver = json.loads((OUT / "ckpt_keeplist_verify.json").read_text())
    assert ver["n_bad"] == 0, "refusing to delete: {} kept checkpoints failed to load".format(ver["n_bad"])
    freed = 0
    for arm, r in rep["arms"].items():
        for i in r["delete"]:
            p = ARMS / arm / "ckpt_iter{:07d}.pt".format(i)
            assert str(i) not in r["keep"]
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
    print("deleted; freed {:.1f} GB".format(freed / 1e9))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    verify() if a.verify else apply() if a.apply else build()
