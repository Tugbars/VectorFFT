#!/usr/bin/env python3
"""trig_capture.py — a STABLE trig output artifact, for migration step 27.

WHY THIS EXISTS
---------------
The migration plan calls step 27 (the trig create tier) its least-protected
step. It is right: the wisdom store holds 539 cells and ZERO of them are trig,
so the fingerprint replay has almost nothing to replay and harness_golden
covers trig only as two REFUSE decisions -- no output bits at all. Moving that
tier with nothing watching the output would be moving it on hope.

THE RACE, AND WHY A PLAIN RUN CANNOT BE A BASELINE
--------------------------------------------------
Every rigor level in this library MEASURES (VFFT_ESTIMATE is a planned fourth
tier that does not exist yet). With no banked trig verdict there is nothing to
replay, so a trig create RACES, the clock picks the plan, and the plan decides
the last bits of the output. Measured on 2026-08-31: three of fourteen cells
disagreed across four runs. Diffing one sample of that measures thermal noise,
not the refactor.

THE FIX: SEED, WARM, THEN MEASURE
---------------------------------
  1. seed   a fresh SCRATCH copy of the store -- never the shipped tree;
  2. warm   one --bank run against it, banking the trig verdicts. Its output
            is discarded: it is the run that races;
  3. repeat N runs against that warmed store. Each create now REPLAYS a banked
            verdict, so it is deterministic and fast.

The repeat still has to be ACROSS PROCESSES. Running a cell twice inside one
process proves nothing: banking is in-memory first even at wisdom_write=0, so
the second create replays the first one's verdict and agrees with itself by
construction. capture_baseline.py documents the same trap and takes the same
way out -- one process per observation.

Banking into the scratch dir is safe by the library's own construction: with
VFFT_WISDOM_DIR unset the store opens READ-ONLY and refuses to bank. The
shipped tree is never a legal target for --bank.

WHAT IT EMITS
  digest                    every run agreed;
  RACED_NONDETERMINISTIC    they did not, with the number of distinct values.

Recording the FACT of a race beats recording one sample of it: the line is
stable across captures, and a cell that starts or stops racing still shows as
a diff -- which is itself worth knowing.

WHAT THIS IS NOT
----------------
A REGRESSION check, not a correctness check. It asserts only that the output
did not change. The naive O(N^2) reference that would prove trig CORRECT is
still deliberately absent: harness_golden.c states the reason and it holds
here too -- the plane-role contract is not stated plainly enough in
include/vfft.h to encode without guessing, and a wrong expectation baked into
a baseline is worse than a missing one, because every later step then passes.

USAGE
  python build_tuned/trig_capture.py --out FILE [--repeat 3]
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "benches", "trig_digest_probe.exe")
STORE = os.path.normpath(os.path.join(
    HERE, "..", "src", "dag-fft-compiler", "generator", "generated"))
SCRATCH = os.path.join(os.environ.get("TEMP", "/tmp"), "vfft_trig_capture")


def seeded_dir(path):
    """A fresh scratch copy of the store. Never the shipped tree."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    for f in os.listdir(STORE):
        if f.endswith(".txt"):
            shutil.copy2(os.path.join(STORE, f), path)
    return path


def one_run(env, tmp, bank=False):
    cmd = [PROBE, "--out", tmp] + (["--bank"] if bank else [])
    subprocess.run(cmd, cwd=HERE, env=env, capture_output=True,
                   text=True, check=True, timeout=3600)
    rows = {}
    for line in open(tmp, encoding="utf-8", errors="replace"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) >= 3:
            rows[p[1]] = p[2]
    return rows


def main():
    def opt(n, d=None):
        return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d

    out = opt("--out")
    repeat = int(opt("--repeat", "3"))
    if not out:
        print(__doc__)
        return 2
    if not os.path.exists(PROBE):
        print("probe not built: python build.py --src benches/"
              "trig_digest_probe.c --vfft --compile")
        return 2

    # THE WARMED STORE IS A FIXTURE, and it has to be the SAME fixture on both
    # sides of the change. Re-seeding and re-warming per capture does not work:
    # the warm run is the run that races, so two independent captures can bank
    # two different verdicts and the digests then differ for a reason that has
    # nothing to do with the refactor. Measured -- dct1.N64 disagreed across two
    # independent captures while the other thirteen cells matched.
    #
    # So: --store names the fixture. It is built (seed + warm) the first time
    # and REUSED afterwards, which is what makes the before/after comparison an
    # answer about the code rather than about which arm won a race that day.
    d = opt("--store", SCRATCH)
    if os.path.isdir(d) and any(f.endswith(".txt") for f in os.listdir(d)):
        print("reusing warmed fixture: %s" % d)
        env = dict(os.environ, VFFT_WISDOM_DIR=d)
    else:
        print("building warmed fixture: %s" % d)
        env = dict(os.environ, VFFT_WISDOM_DIR=seeded_dir(d))
        one_run(env, out + ".run", bank=True)   # the run that races
    tmp = out + ".run"

    runs = [one_run(env, tmp) for _ in range(repeat)]
    try:
        os.remove(tmp)
    except OSError:
        pass

    names = []
    for r in runs:
        for k in r:
            if k not in names:
                names.append(k)

    with open(out, "wb") as f:
        f.write(b"# trig output digests - REGRESSION, not correctness.\n")
        f.write(b"# Seeded SCRATCH store, one --bank warm run, then %d runs\n"
                % repeat)
        f.write(b"# of ONE PROCESS EACH replaying the banked verdicts.\n")
        f.write(b"# A cell that still disagrees is recorded AS raced, never\n"
                b"# sampled. See the file header for why.\n#\n")
        for n in names:
            vals = {r.get(n, "MISSING") for r in runs}
            if len(vals) == 1:
                f.write(("trig %-16s %s\n" % (n, vals.pop())).encode())
            else:
                f.write(("trig %-16s RACED_NONDETERMINISTIC distinct=%d\n"
                         % (n, len(vals))).encode())

    stable = sum(1 for n in names if len({r.get(n) for r in runs}) == 1)
    print("%s: %d cells, %d stable, %d raced"
          % (out, len(names), stable, len(names) - stable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
