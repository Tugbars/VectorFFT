#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ab_fp64.py -- the FP64 regression A/B: two builds of the canonical bench on one store, paired
rounds (2026-11-10: the check that the FP32 work left FP64 where it was).

USAGE
  python gauntlet/ab_fp64.py --old OLD.exe --new NEW.exe --store DIR [--rounds 15] [--cells SPEC]
                             [--control 4096] [--threads 8] [--pace-ms 200] [--out OUTDIR]
  python gauntlet/ab_fp64.py --old A.exe --new A.exe --store DIR --cells oop:4096,lm:1024:8 --control 0 --rounds 1
                             (a smoke test of the parser: one exe as both arms)
  python gauntlet/ab_fp64.py --list             (the default cells; nothing runs)

  OLD, NEW   two bench_1d_vs_mkl executables built from two trees by the same compiler: in each tree
             `python gauntlet/build.py --compile --mkl --vfft` (VFFT_ISA=avx2, the same CC), the two
             exes copied out under two names
  DIR        ONE wisdom store with every cell banked (a scratch copy of src/wisdom/<this CPU>): both
             arms get DIR/wisdom2_oop.txt and VFFT_WISDOM_DIR=DIR. Nothing may race: the store's files
             are checked after every process and the run stops at the first content change
  SPEC       the cells, comma-separated (default: see --list):
               oop:N       --k1noop       K=1 natural out of place, one thread
               ip:N        --k1nat        K=1 natural in place, one thread
               lm:N:K      --lmc2c        the lane-major batch of K, out of place, one thread
               tc:N:K      --ilmt         the contiguous batch of K: its one-thread arm (st) and its
                                          threaded arm at --threads (mt) -- NOISIER, threaded protocol
               mt:N        --k1noop --mt  K=1 natural out of place at --threads -- NOISIER
  --control  the control cell oop:N, first in every round beside the list (default 4096; 0 = none)
  --threads  VFFT_MT of the threaded cells (default 8)
  --pace-ms  the idle before every process (default 200; at least 200)
  OUTDIR     default gauntlet/results/ab_fp64_<date>_<time>: rounds.csv (every process) and summary.md

PROTOCOL. A round runs every cell once per arm, the two arms of a cell back to back. The arm order
alternates cell by cell and round by round; the bench's flip argument (which engine runs first inside
the process) alternates on its own period of two rounds, so each arm meets both orders and both flips.
Every process is the bench's one-cell protocol with gauntlet.py's arguments (pace 300, cool 300, core
2; --ilmt keeps its own core): the pin, the sibling guard, HIGH, the machine's measurement lock, the
cachebust and the cool between the engines, two timing windows. The threaded cells use the bench's
--mt protocol: no pacing inside the process (VFFT_TRIAL_PACE_MS is removed from their environment).
The environment is gauntlet.py's Run.env (the store, VFFT_MT, the MKL and MinGW bin folders on PATH).
PARSING. The numbers are the bench's stdout line. A K=1 cell's line has no route, so its route is
read from the one row of a fresh per-process csv (never a shared csv: its rows are replaced); the
lane-major line carries its route; the contiguous-batch cell has none ("n/a").
REPORT. Per cell (the contiguous batch: per arm st and mt): the paired new/old ratio of our ns, median
and min..max over the rounds; MKL/ours of each arm, median -- the in-process control (MKL is the same
library in both arms); whether the route matched in every round.
"""
import argparse
import csv
import datetime
import hashlib
import io
import os
import re
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gauntlet as G   # noqa: E402  Run.env: the bench's environment, gauntlet.py's own

BENCH_PACE = str(G.PACE_MS)   # the bench's pace and cool arguments, gauntlet.py's (300 ms)
TIMEOUT_S = 1200

# the bench's stdout lines (bench_1d_vs_mkl.c): the K=1 cell, the lane-major batch, the contiguous batch
K1_LINE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(nat-oop|nat-ip)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)x\s+(\S+)")
LM_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+lm-oop\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)x\s+([\d.]+)x"
                     r"\s+([\d.]+)x\s+(\S+)\s+(\S+)\s+(\S+)")
TC_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\|\s+([\d.]+)x\s+([\d.]+)x"
                     r"\s+([\d.]+)x\s+([\d.]+)x\s+([\d.]+)%\s+(\S+)\s*(.*)$")
ENGAGED = re.compile(r"k1noop-mt N=\s*(\d+)\s+T=(\d+): engaged (-?\d+)")


class Cell:
    def __init__(self, tok, control=False):
        p = tok.strip().split(":")
        self.kind = p[0]
        if self.kind in ("oop", "ip", "mt") and len(p) == 2:
            self.N, self.K = int(p[1]), 1
        elif self.kind in ("lm", "tc") and len(p) == 3:
            self.N, self.K = int(p[1]), int(p[2])
        else:
            raise SystemExit("ab_fp64: a cell is oop:N, ip:N, mt:N, lm:N:K or tc:N:K, not %r" % tok)
        if self.N < 2 or self.K < 1 or (self.kind in ("lm", "tc") and self.K < 2):
            raise SystemExit("ab_fp64: bad cell %r" % tok)
        self.control = control
        self.label = ("ctl " if control else "") + tok.strip()
        self.noisy = self.kind in ("mt", "tc")
        self.series = ("st", "mt") if self.kind == "tc" else ("",)

    def argv(self, exe, store, csv_path, flip):
        flags, k, core = {"oop": (["--k1noop"], "1", "2"), "ip": (["--k1nat"], "1", "2"),
                          "mt": (["--k1noop", "--mt"], "1", "2"), "lm": (["--lmc2c"], str(self.K), "2"),
                          "tc": (["--ilmt"], str(self.K), "-1")}[self.kind]
        return ([exe] + flags + [os.path.join(store, "wisdom2_oop.txt"), csv_path, BENCH_PACE, str(self.N), k,
                                 BENCH_PACE, str(flip), core])


def default_cells():
    cells = ["oop:%d" % (1 << e) for e in range(6, 21)]       # 64 .. 2^20
    cells += ["ip:%d" % (1 << e) for e in range(8, 19)]       # 256 .. 2^18
    cells += ["tc:4096:4", "lm:1024:8", "mt:65536"]
    return cells


def env_of(store, outdir, threads):
    e = G.Run.env(SimpleNamespace(store=store, cmp="mkl", threads=threads, dir=outdir))
    if threads > 1:
        e.pop("VFFT_TRIAL_PACE_MS", None)   # the threaded protocol: no pacing inside the process
    return e


def parse(cell, text, csv_path):
    """-> {series: (ours_ns, mkl_ns)}, route, note; raises ValueError when the line is not there"""
    lines = text.splitlines()
    if cell.kind in ("oop", "ip", "mt"):
        m = next((K1_LINE.match(l) for l in lines if K1_LINE.match(l)), None)
        if not m or int(m.group(1)) != cell.N:
            raise ValueError("no K=1 cell line")
        route = "?"
        try:
            rows = list(csv.DictReader(io.open(csv_path, encoding="utf-8", errors="replace")))
            if rows:
                route = rows[-1].get("route", "?") or "?"
        except OSError:
            pass
        note = ""
        if cell.kind == "mt":
            e = next((ENGAGED.search(l) for l in lines if ENGAGED.search(l)), None)
            note = "engaged %s" % (e.group(3) if e else "?")
        return {"": (float(m.group(4)), float(m.group(5)))}, route, note
    if cell.kind == "lm":
        m = next((LM_LINE.match(l) for l in lines if LM_LINE.match(l)), None)
        if not m or int(m.group(1)) != cell.N or int(m.group(2)) != cell.K:
            raise ValueError("no lane-major line")
        return {"": (float(m.group(3)), float(m.group(5)))}, m.group(12), ""
    m = next((TC_LINE.match(l) for l in lines if TC_LINE.match(l)), None)
    if not m or int(m.group(1)) != cell.N or int(m.group(2)) != cell.K:
        raise ValueError("no contiguous-batch line")
    return ({"st": (float(m.group(4)), float(m.group(6))), "mt": (float(m.group(3)), float(m.group(5)))},
            "n/a", (m.group(13) or "").strip())


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def store_digest(store):
    return {f: sha1(os.path.join(store, f)) for f in sorted(os.listdir(store))
            if os.path.isfile(os.path.join(store, f))}


class StoreGuard:
    """nothing may race: every file of the store, by content, against the start (hashed after every
    process: a size or a time can miss a rewrite); a changed file, or one that appeared or went, is a change"""
    def __init__(self, store):
        self.store = store
        self.digest = store_digest(store)

    def changed(self):
        now = store_digest(self.store)
        return [f for f in sorted(set(now) | set(self.digest)) if now.get(f) != self.digest.get(f)]


def med(v):
    return statistics.median(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old")
    ap.add_argument("--new")
    ap.add_argument("--store")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--cells")
    ap.add_argument("--control", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pace-ms", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    toks = [t for t in (a.cells.split(",") if a.cells else default_cells()) if t.strip()]
    cells = ([Cell("oop:%d" % a.control, control=True)] if a.control else []) + [Cell(t) for t in toks]
    if a.list:
        print("cells (%d per round per arm): %s" % (len(cells), ", ".join(c.label for c in cells)))
        return 0
    if not (a.old and a.new and a.store):
        ap.error("--old, --new and --store are required")
    old, new, store = os.path.abspath(a.old), os.path.abspath(a.new), os.path.abspath(a.store)
    for p in (old, new):
        if not os.path.isfile(p):
            raise SystemExit("ab_fp64: no such executable: %s" % p)
    if not os.path.isfile(os.path.join(store, "wisdom2_oop.txt")):
        raise SystemExit("ab_fp64: %s holds no wisdom2_oop.txt (the store must have every cell banked)" % store)
    if a.pace_ms < 200:
        raise SystemExit("ab_fp64: --pace-ms is at least 200 (the protocol's idle between processes)")
    if a.rounds < 1 or a.threads < 2:
        raise SystemExit("ab_fp64: --rounds >= 1, --threads >= 2")
    outdir = os.path.abspath(a.out or os.path.join(G.RESULTS, "ab_fp64_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")))
    tmp = os.path.join(outdir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    guard = StoreGuard(store)

    head = ["# FP64 A/B: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "old   %s (%d bytes, sha1 %s)" % (old, os.path.getsize(old), sha1(old)[:12]),
            "new   %s (%d bytes, sha1 %s)" % (new, os.path.getsize(new), sha1(new)[:12]),
            "store %s" % store,
            "%d rounds, %d cells per round per arm, pace %d ms, threaded cells at T=%d" % (a.rounds, len(cells), a.pace_ms, a.threads)]
    if old == new or sha1(old) == sha1(new):
        head.append("NOTE: old and new are the same executable -- a null A/B (a smoke test)")
    if a.rounds < 15:
        head.append("NOTE: fewer than 15 rounds -- a smoke test, not a verdict")
    print("\n".join(head), flush=True)

    fcsv = io.open(os.path.join(outdir, "rounds.csv"), "w", encoding="utf-8", newline="")
    w = csv.writer(fcsv)
    w.writerow(["round", "cell", "series", "arm", "first", "flip", "ours_ns", "mkl_ns", "mkl_over_ours", "route", "note"])
    res = {}        # (cell label, series) -> list of per-round dicts
    errors = []
    stopped = None
    t0 = time.time()
    try:
        for r in range(a.rounds):
            for i, c in enumerate(cells):
                old_first = (r + i) % 2 == 0
                flip = ((r // 2) + i) % 2
                arms = [("old", old), ("new", new)] if old_first else [("new", new), ("old", old)]
                got = {}
                for arm, exe in arms:
                    time.sleep(a.pace_ms / 1000.0)
                    cp = os.path.join(tmp, "r%d_c%d_%s.csv" % (r, i, arm))
                    if os.path.exists(cp):
                        os.remove(cp)
                    argv = c.argv(exe, store, cp, flip)
                    text = ""
                    try:
                        p = subprocess.run(argv, capture_output=True, text=True, errors="replace", cwd=tmp,
                                           env=env_of(store, outdir, a.threads if c.noisy else 1), timeout=TIMEOUT_S)
                        text = p.stdout + p.stderr
                        if p.returncode != 0:
                            raise ValueError("exit %d" % p.returncode)
                        got[arm] = parse(c, text, cp)
                    except (ValueError, subprocess.TimeoutExpired) as ex:
                        tail = [l.strip() for l in text.splitlines() if l.strip()][-3:]
                        errors.append("round %d %s %s: %s | %s" % (r + 1, c.label, arm, ex, " / ".join(tail)))
                        print("  ERROR " + errors[-1], flush=True)
                    finally:
                        if os.path.exists(cp):
                            os.remove(cp)
                    bad = guard.changed()
                    if bad:
                        stopped = ("the store changed after round %d, %s, arm %s: %s -- a cell raced (it was not "
                                   "banked) or something else wrote the store; the run stops" % (r + 1, c.label, arm, ", ".join(bad)))
                        raise KeyboardInterrupt
                if "old" not in got or "new" not in got:
                    continue
                (so, ro, no), (sn, rn, nn) = got["old"], got["new"]
                for s in c.series:
                    (oo, mo), (on, mn) = so[s], sn[s]
                    rec = dict(r=r, flip=flip, first="old" if old_first else "new", oo=oo, mo=mo, on=on, mn=mn, ro=ro, rn=rn)
                    res.setdefault((c.label, s), []).append(rec)
                    for arm, ours, mkl, route, note in (("old", oo, mo, ro, no), ("new", on, mn, rn, nn)):
                        w.writerow([r + 1, c.label, s, arm, rec["first"], flip, "%.0f" % ours, "%.0f" % mkl,
                                    "%.4f" % (mkl / ours if ours > 0 and mkl > 0 else 0), route, note])
                    print("  r%-2d %-16s %-2s first=%-3s flip=%d  new/old %.3f  mkl/ours old %.3f new %.3f  route %s%s%s"
                          % (r + 1, c.label, s, rec["first"], flip, on / oo if oo > 0 else 0,
                             mo / oo if oo > 0 and mo > 0 else 0, mn / on if on > 0 and mn > 0 else 0,
                             ro, "" if ro == rn else " != " + rn, ("  [" + no + " | " + nn + "]") if (no or nn) else ""),
                          flush=True)
                fcsv.flush()
    except KeyboardInterrupt:
        if not stopped:
            stopped = "interrupted"
    fcsv.close()

    out = head + ["", "| cell | arm | rounds | new/old median | min..max | MKL/ours old | MKL/ours new | route |",
                  "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for s in c.series:
            v = res.get((c.label, s), [])
            ratio = [x["on"] / x["oo"] for x in v if x["oo"] > 0 and x["on"] > 0]
            if not ratio:
                out.append("| %s | %s | %d | - | - | - | - | - |" % (c.label, s or "-", len(v)))
                continue
            cto = [x["mo"] / x["oo"] for x in v if x["oo"] > 0 and x["mo"] > 0]
            ctn = [x["mn"] / x["on"] for x in v if x["on"] > 0 and x["mn"] > 0]
            diff = [x for x in v if x["ro"] != x["rn"]]
            route = ("matched (%s)" % v[0]["ro"]) if not diff else \
                    ("DIFFERS in %d of %d (old %s, new %s)" % (len(diff), len(v), diff[0]["ro"], diff[0]["rn"]))
            out.append("| %s%s | %s | %d | %.3f | %.3f..%.3f | %.3f | %.3f | %s |"
                       % (c.label, " (noisier)" if c.noisy else "", s or "-", len(v), med(ratio), min(ratio), max(ratio),
                          med(cto), med(ctn), route))
    out.append("")
    out.append("new/old = our ns, new arm over old arm, paired within a round (< 1: the new build is faster); "
               "MKL/ours = the in-process control of each arm (MKL is the same library in both).")
    if errors:
        out.append("")
        out.append("%d process(es) failed:" % len(errors))
        out += ["- " + e for e in errors]
    if stopped:
        out.append("")
        out.append("STOPPED: " + stopped)
    out.append("")
    out.append("%.0f s; rounds.csv beside this file" % (time.time() - t0))
    text = "\n".join(out) + "\n"
    io.open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8", newline="\n").write(text)
    print("\n" + text)
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    return 3 if stopped and stopped != "interrupted" else (1 if errors else 0)


if __name__ == "__main__":
    sys.exit(main())
