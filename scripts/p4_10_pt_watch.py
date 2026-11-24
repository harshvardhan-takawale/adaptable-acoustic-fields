"""P4-10 P-1 read watcher (D218). Runs INSIDE a 1-CPU Slurm job (scripts/p4_10_pt_watch.sbatch, --requeue): no session or login-node process is needed. Restart-safe:
every stage is gated on its own output file, so a requeue or a second copy skips what exists; jobs in flight are found by name in the queue and not resubmitted.
Per step (152.5K 155K 157.5K 160K), once BOTH arms' checkpoints exist and have stopped growing:
  per arm: dense (gate2_dense) -> speckle and extra (0d residual, bypass); unseen; 15 direction rooms (each gated on its own output) -> all outputs for both arms ->
  score (inline, p4_10_pt_read.py --score) -> read_<step>.json -> report.{json,md} rebuilt atomically -> stop rule (scancel the R1PT training job, write STOPPED).
  python scripts/p4_10_pt_watch.py [--dry-run] [--once] [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--once", action="store_true")
ap.add_argument("--max-hours", type=float, default=9.0)
A = ap.parse_args()
HERE = Path("/fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields")
os.chdir(HERE)
SCR = Path("/fs/nexus-scratch/htakawal/aaf_p410/pt")
OUT = Path("outputs/p4_10/pt")
ARMS = ("R1PT", "R1CTL")
STEPS = {"152K": 152500, "155K": 155000, "157K": 157500, "160K": 160000}
MAX_TRIES = 4


def sh(*cmd):
    r = subprocess.run(list(cmd), capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def excluded():
    _, out, _ = sh("sinfo", "-h", "-N", "-o", "%N")
    bad = re.compile(r"^(legacy|cml23$|gammagpu0[2-6]$|gammagpu09$|quics00$|tron36$)")
    return ",".join(sorted({n for n in out.split() if bad.match(n)}))


EXC = excluded()
GPU = ["--partition=scavenger", "--account=scavenger", "--qos=scavenger", "--requeue", "--exclude=" + EXC, "--cpus-per-task=2"]
TRIES = {}


def log(*a):
    print(time.strftime("%H:%M"), *a, flush=True)


def queued(name):
    _, out, _ = sh("squeue", "-u", os.environ.get("USER", "htakawal"), "-h", "-o", "%j %T")
    return sum(1 for ln in out.splitlines() if ln.split() and ln.split()[0] == name)


def submit(name, args, cmd, script, array=None):
    """one stage; a name already in the queue is never resubmitted; failures are retried up to MAX_TRIES"""
    if queued(name):
        return False
    k = TRIES.get(name, 0)
    if k >= MAX_TRIES:
        log("FAILED", name, "gave up after", k, "submissions")
        return False
    TRIES[name] = k + 1
    c = ["sbatch", "--parsable", "--job-name=" + name] + args + (["--array=" + array] if array else []) + ["--export=ALL,CMD=" + cmd, script]
    if A.dry_run:
        log("DRY", " ".join(c))
        return True
    rc, out, err = sh(*c)
    if rc:
        log("FAILED submit", name, err.strip()[:200])
        return False
    log("submitted", name, out.strip())
    return True


def ckpt(arm, it):
    if os.environ.get("P410_PT_FAKE"):                                              # dry-run self-test: R1's own checkpoint stands in for both arms
        return Path("outputs/p4_10/arms/R1/ckpt_iter0150000.pt")
    return Path("outputs/p4_10/arms_pt/{}/ckpt_iter{:07d}.pt".format(arm, it))


SIZES = {}


def settled(it):
    """both checkpoints exist and kept their size over two passes of the loop (save_ckpt writes the penalty state in a second pass)"""
    if not all(ckpt(a, it).exists() for a in ARMS):
        return False
    cur = tuple(ckpt(a, it).stat().st_size for a in ARMS)
    ok = os.environ.get("P410_PT_FAKE") or SIZES.get(it) == cur and all(time.time() - ckpt(a, it).stat().st_mtime > 60 for a in ARMS)
    SIZES[it] = cur
    return ok


def stage_files(n):
    return {"dense": Path("outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz".format(n)), "unseen": Path("outputs/p4_8/unseen/{}.json".format(n)),
            "speckle": Path("outputs/p4_5/speckle/{}.json".format(n)), "extra": SCR / "extra" / (n + ".npz")}


def missing_dirs(n):
    return [i for i in range(15) if not (SCR / "dir" / "{}__{:02d}.npz".format(n, i)).exists()]


def read_step(k, it):
    """advance one step's reads by one pass; True when scored (or stopped)"""
    if (OUT / "read_{}.json".format(k)).exists():
        return True
    if not settled(it):
        return False
    ready = True
    for a in ARMS:
        n, ck = "{}_{}".format(a, k), str(ckpt(a, it))
        f = stage_files(n)
        tag = "aaf_pt_{}_{}".format(a.lower(), k)
        if not f["dense"].exists():
            ready = False
            submit(tag + "_dense", GPU + ["--gres=gpu:rtxa5000:1", "--time=02:00:00", "--mem=32G"], "scripts/p4_5_gate2_dense.py --checkpoint {} --name {}".format(ck, n), "scripts/slurm/p4_5_gpu.sh")
        if not f["unseen"].exists():
            ready = False
            submit(tag + "_unseen", GPU + ["--gres=gpu:rtxa5000:1", "--time=02:00:00", "--mem=32G"], "scripts/p4_8_unseen_eval.py --items {}={}".format(n, ck), "scripts/slurm/p4_5_gpu.sh")
        if f["dense"].exists() and not f["speckle"].exists():
            ready = False
            submit(tag + "_speckle", GPU + ["--gres=gpu:rtxa5000:1", "--time=02:00:00", "--mem=64G"], "scripts/p4_5_speckle.py --names {} --project".format(n), "scripts/slurm/p4_5_gpu.sh")
        if f["dense"].exists() and not f["extra"].exists():
            ready = False
            submit(tag + "_extra", GPU + ["--gres=gpu:rtxa5000:1", "--time=02:00:00", "--mem=32G"], "scripts/p4_10_pt_read.py --extra {}={}".format(n, ck), "scripts/slurm/p4_10_gpu.sh")
        if not f["speckle"].exists() or not f["extra"].exists():
            ready = False
        miss = missing_dirs(n)
        if miss:
            ready = False
            if not queued(tag + "_dir"):
                submit(tag + "_dir", GPU + ["--gres=gpu:rtxa4000:1", "--time=01:00:00", "--mem=14G"], "scripts/p4_10_pt_read.py --dir {}={} --idx @".format(n, ck), "scripts/slurm/p4_10_gpu.sh", ",".join(map(str, miss)))
    if not ready:
        return False
    log("all outputs of", k, "exist; scoring")
    if A.dry_run:
        return True
    rc, out, err = subprocess.run([sys.executable, "scripts/p4_10_pt_read.py", "--score", k], capture_output=True, text=True).returncode, None, None
    if rc:
        log("FAILED scoring", k)
        return False
    return True


def report():
    """outputs/p4_10/pt/report.{json,md} from the read files, atomically"""
    rd = {k: json.loads((OUT / "read_{}.json".format(k)).read_text()) for k in STEPS if (OUT / "read_{}.json".format(k)).exists()}
    stopped = (OUT / "STOPPED").read_text() if (OUT / "STOPPED").exists() else ""
    rep = {"updated": time.strftime("%Y-%m-%d %H:%M ET"), "reads_done": list(rd), "stopped": stopped, "reads": rd}
    L = ["# P4-10 P-1: R1-P(t) (R1 @150K + mean-value penalty, 10K steps) against R1-ctl (D217 / D218)", "", "Updated {}. Reads done: {}.{}".format(
        rep["updated"], ", ".join("+{}".format(STEPS[k] - 150000) for k in rd) or "none", "  **STOPPED: {}**".format(stopped) if stopped else ""), ""]
    for k, v in rd.items():
        f = OUT / "read_{}.md".format(k)
        L += [f.read_text().rstrip(), ""] if f.exists() else []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, txt in (("report.json", json.dumps(rep, indent=1)), ("report.md", "\n".join(L) + "\n")):
        t = OUT / (name + ".tmp")
        t.write_text(txt)
        os.replace(t, OUT / name)


def stop_rule():
    """the stop rule on the newest read of R1-P(t): scancel its training job, write STOPPED"""
    for k in reversed(list(STEPS)):
        p = OUT / "read_{}.json".format(k)
        if not p.exists():
            continue
        st = json.loads(p.read_text()).get("stop_rule_triggered") or []
        if st and not (OUT / "STOPPED").exists():
            msg = "stop rule triggered at {} (step {}): {} more than 0.02 below R1-ctl with the CI entirely below zero; R1-P(t) cancelled at {}".format(
                k, STEPS[k], ", ".join(st), time.strftime("%H:%M ET"))
            _, out, _ = sh("squeue", "-u", os.environ.get("USER", "htakawal"), "-h", "-n", "aaf_r1pt", "-o", "%i")
            for j in out.split():
                log("scancel", j)
                if not A.dry_run:
                    sh("scancel", j)
            (OUT / "STOPPED").write_text(msg + "\n")
            log("STOPPED", msg)
        break


def main():
    t0 = time.time()
    log("watcher on", os.uname().nodename, "dry" if A.dry_run else "", "exclude", len(EXC.split(",")), "nodes")
    while time.time() - t0 < A.max_hours * 3600:
        done = 0
        for k, it in STEPS.items():
            if (OUT / "STOPPED").exists() and not (OUT / "read_{}.json".format(k)).exists():
                done += 1                                                           # the arm was stopped: later reads have no checkpoint pair
                continue
            n0 = (OUT / "read_{}.json".format(k)).exists()
            if read_step(k, it):
                done += 1
                if not n0:
                    report()
                    log("read", k, "posted")
        report() if any((OUT / "read_{}.json".format(k)).exists() for k in STEPS) else None
        stop_rule()
        if done == len(STEPS) and not A.dry_run:
            report()
            log("all reads done")
            return
        if A.once:
            return
        time.sleep(90)
    log("time limit reached")


if __name__ == "__main__":
    main()
