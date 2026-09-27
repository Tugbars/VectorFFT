#!/usr/bin/env python3
"""api_sweep.py - drive api_sweep.c: the public-API sweep, the wisdom replay and
the store round trip, as byte-diffable artifacts.

THE THREE ARTIFACTS
-------------------
api_sweep.txt     every cell of api_sweep.c's matrix, ONE PROCESS PER CELL,
                  replayed against the SWEEP STORE (below), REPEAT times; a
                  cell whose output differs between repeats is written as
                  NONDETERMINISTIC, a stable fact rather than a lucky sample.
wisdom_replay.txt a deterministic, stratified sample of the rows of the store
                  itself: each @cell key becomes a config, created against a
                  fresh copy of the store. Per row: accepted or refused, how
                  many creates raced, and - only when nothing raced - the plan
                  fingerprint and the output bits. A row served from the store
                  must build the same plan after every step; a row that races
                  must still race. This is the codec proof for the wisdom
                  readers the separation splits.
roundtrip.txt     vfft_wisdom_load(store) -> vfft_wisdom_save(scratch), and
                  the sha256 of every saved file.

THE SWEEP STORE
---------------
Cells the shipped store does not cover would race at create and put the clock
inside the artifact. So a reference capture first BANKS: every cell once,
sequentially, with wisdom_write=1, into a scratch copy of src/wisdom. That
banked store is saved beside the reference (sweep_store/) and every later
capture replays from a copy of THAT, never from src/wisdom. The same code and
the same store then give the same plans; a race on replay is reported per cell.

USAGE
  python api_sweep.py bank     --exe BIN --store-out DIR
  python api_sweep.py capture  --exe BIN --store DIR --out FILE [--repeat N] [--jobs J]
  python api_sweep.py replay   --exe BIN --store DIR --out FILE [--per-group G] [--jobs J]
  python api_sweep.py roundtrip --exe BIN --store DIR --out FILE
"""
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toolchain  # noqa: E402

ROOT = toolchain.ROOT


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def copy_store(src, dst):
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p) and f.endswith(".txt"):
            shutil.copy2(p, dst)


def run_exe(exe, args, store, timeout=900):
    env = dict(os.environ, VFFT_WISDOM_DIR=store)
    r = subprocess.run([exe] + args, cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=timeout)
    rows = [l for l in r.stdout.replace("\r\n", "\n").splitlines()
            if l.startswith(("cell ", "races ", "fp ", "bits ", "roundtrip"))]
    if r.returncode != 0:
        rows.append("EXIT %d" % r.returncode)
    return rows


def cells(exe):
    r = subprocess.run([exe, "--list"], cwd=ROOT, capture_output=True, text=True)
    out = [l.split(None, 1)[1] for l in r.stdout.splitlines() if l.strip()]
    if r.returncode != 0 or not out:
        raise SystemExit("%s --list failed" % exe)
    return out


def _pool_map(fn, items, jobs):
    """Map fn over items on `jobs` workers, each with its own scratch dir."""
    scratch = tempfile.mkdtemp(prefix="vfft_sweep_")
    slots = [os.path.join(scratch, "w%d" % k) for k in range(jobs)]
    free = list(slots)
    import threading
    lock = threading.Lock()

    def task(item):
        with lock:
            slot = free.pop()
        try:
            return fn(item, slot)
        finally:
            with lock:
                free.append(slot)
    try:
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(task, items))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def cmd_bank():
    exe, out = opt("--exe"), opt("--store-out")
    src = opt("--store", os.path.join(ROOT, "src", "wisdom"))
    copy_store(src, out)
    names = cells(exe)
    for i, n in enumerate(names):
        run_exe(exe, ["--cell", str(i), "--write"], out)
        if i % 25 == 0:
            print("  banked %d/%d %s" % (i, len(names), n), flush=True)
    # a second pass: a cell whose create consults a child banked LATER in the
    # first pass would otherwise race on replay
    for i in range(len(names)):
        run_exe(exe, ["--cell", str(i), "--write"], out)
    print("bank: %d cells -> %s" % (len(names), out))
    return 0


def cmd_capture():
    exe, store, out = opt("--exe"), opt("--store"), opt("--out")
    repeat, jobs = int(opt("--repeat", "3")), int(opt("--jobs", "4"))
    names = cells(exe)

    def one(i, slot):
        seen = []
        for _ in range(repeat):
            copy_store(store, slot)
            seen.append(run_exe(exe, ["--cell", str(i)], slot))
        if all(s == seen[0] for s in seen[1:]):
            return seen[0]
        return ["NONDETERMINISTIC %s differed across %d repeats" % (names[i], repeat)]

    results = _pool_map(one, range(len(names)), jobs)
    return _write(out, "# api_sweep: one process per cell, replayed from the sweep store\n",
                  [r for rows in results for r in rows])


_CELL = re.compile(r"^@cell (.*?)\s*\|")


def store_specs(store, per_group):
    """A deterministic stratified sample of the store's @cell keys: up to
    `per_group` rows per (shard, key-without-n) group, evenly spaced in file
    order, so every family and every layout/placement/order combination the
    store holds is exercised without replaying all ~15k rows."""
    groups = {}
    for f in sorted(os.listdir(store)):
        if not (f.startswith("wisdom2_") and f.endswith(".txt")):
            continue
        for line in open(os.path.join(store, f), encoding="utf-8", errors="replace"):
            m = _CELL.match(line)
            if not m:
                continue
            key = m.group(1)
            fam = (f, re.sub(r"\bn=\S+", "", key))
            groups.setdefault(fam, [])
            if key not in groups[fam]:
                groups[fam].append(key)
    specs = []
    for fam in sorted(groups):
        rows = groups[fam]
        step = max(1, len(rows) // per_group)
        specs.extend(rows[::step][:per_group])
    return specs


def cmd_replay():
    exe, store, out = opt("--exe"), opt("--store"), opt("--out")
    per, jobs = int(opt("--per-group", "4")), int(opt("--jobs", "4"))
    specs = store_specs(store, per)

    def one(spec, slot):
        copy_store(store, slot)
        rows = run_exe(exe, ["--spec", spec], slot)
        return ["%s :: %s" % (spec, r.split(" ", 2)[-1] if r.startswith(("cell", "races", "fp", "bits")) else r)
                for r in rows]

    results = _pool_map(one, specs, jobs)
    return _write(out, "# wisdom_replay: %d store rows (<= %d per family), fresh store copy each\n"
                  % (len(specs), per), [r for rows in results for r in rows])


def cmd_roundtrip():
    exe, store, out = opt("--exe"), opt("--store"), opt("--out")
    d = tempfile.mkdtemp(prefix="vfft_rt_")
    try:
        src = os.path.join(d, "src")
        dst = os.path.join(d, "dst")
        copy_store(store, src)
        os.makedirs(dst)
        rows = run_exe(exe, ["--roundtrip", src, dst], src)
        for f in sorted(os.listdir(dst)):
            p = os.path.join(dst, f)
            if os.path.isfile(p):
                same = os.path.exists(os.path.join(src, f)) and \
                    toolchain.sha256(p) == toolchain.sha256(os.path.join(src, f))
                rows.append("saved %s sha256=%s %s" % (f, toolchain.sha256(p)[:16],
                                                        "IDENTICAL_TO_LOADED" if same else "DIFFERS_FROM_LOADED"))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return _write(out, "# wisdom store load -> save round trip\n", rows)


def _write(out, header, rows):
    with open(out, "w", newline="\n") as f:
        f.write(header)
        for r in rows:
            f.write(r + "\n")
    bad = sum(1 for r in rows if r.startswith(("NONDETERMINISTIC", "EXIT")))
    print("%s: %d rows, %d nondeterministic/crashed" % (out, len(rows), bad))
    return 0


def main():
    cmds = {"bank": cmd_bank, "capture": cmd_capture, "replay": cmd_replay,
            "roundtrip": cmd_roundtrip}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        return 2
    return cmds[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
