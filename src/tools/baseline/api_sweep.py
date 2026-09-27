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
                               [--ref-variants REF_OUT.variants.json]
  python api_sweep.py replay   --exe BIN --store DIR --out FILE [--per-group G] [--jobs J]
                               [--ref-variants REF_OUT.variants.json]
  Without --ref-variants a capture is a REFERENCE (every variant recorded);
  with it, a STEP capture (see _variants).
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


def _variants(keys, run_once, out, header, repeat, jobs, ref_json):
    """The VARIANT SET protocol.

    Some creates pick between arms by a clock the race counter cannot see (the
    split out-of-place tuner reads __rdtsc directly and banks nothing: measured
    2026-09-27, c2c split oop N=45 K=4 gives one fingerprint and three
    different output bit patterns across eight processes). Recording such a
    cell as NONDETERMINISTIC after R repeats made the artifact itself flap:
    with R=3 the cell was sometimes seen deterministic, and the unchanged tree
    failed its own gate.

    So a REFERENCE capture runs every key `repeat` times and keeps EVERY
    distinct output (each one complete and bit-exact) in `out`.variants.json.
    A STEP capture (ref_json given) runs a key once; if its output is not one
    of the reference's variants it reruns, up to `repeat` more times, until it
    lands on one. A key that matches writes the reference's canonical rows, so
    the two text artifacts are byte-identical exactly when every key matched a
    known variant. A regression changes the bits of every arm, so it never
    matches: zero tolerance is kept, the coin flip is not a failure."""
    import hashlib
    import json

    ref = None
    if ref_json:
        with open(ref_json) as f:
            ref = json.load(f)

    def h(rows):
        return hashlib.sha1("\n".join(rows).encode()).hexdigest()[:16]

    def one(k, slot):
        if ref is None:
            seen = {}
            for _ in range(repeat):
                rows = run_once(k, slot)
                seen.setdefault(h(rows), rows)
            return k, seen, None
        known = ref.get(str(k))
        tries = []
        for _ in range(1 + repeat):
            rows = run_once(k, slot)
            if known and h(rows) in known["hashes"]:
                return k, None, known
            tries.append(rows)
            if not known:
                break
        return k, {h(tries[0]): tries[0]}, None

    results = _pool_map(lambda k, slot: one(k, slot), keys, jobs)
    rows_out, var = [], {}
    for k, seen, known in results:
        if known is not None:
            rows_out.extend(known["rows"])
            if len(known["hashes"]) > 1:
                rows_out.append("VARIANTS %s %d" % (k, len(known["hashes"])))
            continue
        first = next(iter(seen.values()))
        rows_out.extend(first)
        if ref is None:
            var[str(k)] = dict(hashes=sorted(seen), rows=first)
            if len(seen) > 1:
                rows_out.append("VARIANTS %s %d" % (k, len(seen)))
        else:
            rows_out.append("UNMATCHED %s (no reference variant reproduced)" % k)
    if ref is None:
        with open(out + ".variants.json", "w") as f:
            json.dump(var, f)
    return _write(out, header, rows_out)


def cmd_capture():
    exe, store, out = opt("--exe"), opt("--store"), opt("--out")
    repeat, jobs = int(opt("--repeat", "6")), int(opt("--jobs", "4"))
    names = cells(exe)

    def run_once(i, slot):
        copy_store(store, slot)
        return run_exe(exe, ["--cell", str(i)], slot)

    return _variants(list(range(len(names))), run_once, out,
                     "# api_sweep: one process per cell, replayed from the sweep store;\n"
                     "# VARIANTS = the cell picks between arms by an uncounted clock (see api_sweep.py)\n",
                     repeat, jobs, opt("--ref-variants"))


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
    repeat = int(opt("--repeat", "4"))
    specs = store_specs(store, per)

    def run_once(spec, slot):
        copy_store(store, slot)
        rows = run_exe(exe, ["--spec", spec], slot)
        out = []
        for r in rows:
            p = r.split(None, 2)            # "races spec 0" -> kind, "spec", rest
            if len(p) >= 2 and p[0] in ("cell", "races", "fp", "bits") and p[1] == "spec":
                r = "%s %s" % (p[0], p[2] if len(p) > 2 else "")
            out.append("%s :: %s" % (spec, r))
        return out

    return _variants(specs, run_once, out,
                     "# wisdom_replay: %d store rows (<= %d per family), fresh store copy each\n"
                     % (len(specs), per), repeat, jobs, opt("--ref-variants"))


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
