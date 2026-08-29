#!/usr/bin/env python3
"""capture_baseline.py - produce the two byte-diffable migration artifacts.

WHY THIS IS A SCRIPT AND NOT A SHELL LOOP
-----------------------------------------
These files are compared byte-for-byte after every migration step, roughly two
dozen times. Captured by hand they drifted on the very first comparison: the
committed baseline was LF in its header and CRLF in its data rows, so a
byte-identical result reported all 36 rows as changed. A check that manufactures
reds trains you to wave the real one through. The capture is fixed in one place,
with the four properties the artifacts depend on spelled out below.

1. ONE PROCESS PER CELL
   Several things in the library are process-lifetime, not per-plan - above all
   the K=1 pair-order memo, keyed by N and consulted BEFORE the race. Share a
   process and a later cell inherits what an earlier one memoized, so the
   artifact becomes order- and history-dependent. Not theory: it produced a
   false "step 4 changed output bits" and cost a revert scare.

2. A FRESH SEEDED STORE PER CELL, IN SCRATCH
   Seeded so cells REPLAY instead of racing - a raced plan makes the output a
   coin flip. Scratch because a gate must never write the shipped tree; the
   store under generator/generated is copied FROM, never pointed AT.

3. LF, ALWAYS
   Both binaries put stdout in binary mode; newline="\n" below keeps Python from
   undoing that on Windows.

4. REPEATED, AND NONDETERMINISM RECORDED RATHER THAN SAMPLED
   The race counter is a fast positive signal, not a proof. It only fires where
   someone added it, and the sites live in vfft.c - a racer DEFINED in another
   header is invisible to it. One is now known: c2c.split.ip.nat has no banked
   nat entry, so it races in natorder_calibrate.h every time, and it picked
   nat=5/natcyc=96 in 8 of 10 runs and nat=4/natcyc=34 in the other 2 while
   reporting races=0. Nineteen headers under src/core call a clock, so assume
   more exist. Each cell is therefore captured REPEAT times and a cell whose
   output is not identical every time is written as NONDETERMINISTIC instead of
   as one lucky sample. That trades a flapping artifact for a stable fact, and
   the fact is still a check: a cell that becomes nondeterministic, or stops
   being, changes this line and the diff catches it.

USAGE
  python build_tuned/capture_baseline.py --out <dir> [--repeat N]
    --out build_tuned/baseline   re-stamp the reference (use a high --repeat)
    --out <scratch>              capture for comparison against the reference
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(HERE, "benches")
STORE = os.path.normpath(os.path.join(
    HERE, "..", "src", "dag-fft-compiler", "generator", "generated"))
SCRATCH = os.path.join(os.environ.get("TEMP", "/tmp"), "vfft_capture")


def seeded_dir(tag):
    """A fresh scratch copy of the store. Never the shipped tree."""
    d = os.path.join(SCRATCH, tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(STORE):
        if f.endswith(".txt"):
            shutil.copy2(os.path.join(STORE, f), d)
    return d


def cells(exe):
    """Ask the binary. Never infer from output length: the first version of
    this script did that, read a count off the all-cells path, then ran --cell
    for indices past the end - which fell through to the all-cells path again,
    once per index, and turned 6 cells into 1268 rows."""
    r = subprocess.run([exe, "--list"], cwd=ROOT, capture_output=True, text=True)
    rows = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    if r.returncode != 0 or not rows:
        raise SystemExit("%s does not support --list; add it rather than "
                         "guessing the cell count" % exe)
    return [l.split(None, 1)[1] for l in rows]


def run_cell(exe, i, tag):
    env = dict(os.environ, VFFT_WISDOM_DIR=seeded_dir(tag))
    r = subprocess.run([exe, "--cell", str(i)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    return r.stdout.replace("\r\n", "\n").rstrip("\n").splitlines()


def build(exe_name):
    """Build with VFFT_FINGERPRINT=1, always.

    Not a convenience. Both binaries read the create-race counter, which only
    exists under that flag, and harness_golden USED to fall back to "cannot
    tell" without it - so a plain build.py invocation produced a harness that
    ran green while checking nothing, and the golden baseline was captured that
    way. harness_golden now #errors instead, and the flag is set here so the
    capture cannot be run against a stale binary built the old way."""
    env = dict(os.environ, VFFT_FINGERPRINT="1")
    r = subprocess.run(
        [sys.executable, "build.py", "--src", "benches/%s.c" % exe_name,
         "--vfft", "--compile"],
        cwd=HERE, env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise SystemExit("build of %s FAILED:\n%s"
                         % (exe_name, r.stdout[-2000:] + r.stderr[-2000:]))


def capture(exe_name, header, out_path, repeat):
    build(exe_name)
    exe = os.path.join(BENCH, exe_name + ".exe")
    if not os.path.exists(exe):
        raise SystemExit("missing %s - build it first" % exe)
    names = cells(exe)
    rows, flaky = [], []
    for i, name in enumerate(names):
        seen = [run_cell(exe, i, "%s%d" % (exe_name, i)) for _ in range(repeat)]
        if all(s == seen[0] for s in seen[1:]):
            rows.extend(seen[0])
        else:
            rows.append("NONDETERMINISTIC %-28s differed across %d repeats"
                        % (name, repeat))
            flaky.append(name)
    with open(out_path, "w", newline="\n") as f:
        f.write(header)
        for row in rows:
            f.write(row + "\n")
    return len(rows), flaky


GOLD_HDR = (
    "# golden artifacts - refusal decisions and output-bit digests.\n"
    "# ONE PROCESS PER CELL, each against a fresh seeded copy of the store.\n"
    "# Built with -DVFFT_FINGERPRINT so the purity assert is live: a cell whose\n"
    "# create RACED emits NOT_BANKED_RACED instead of a digest, because a raced\n"
    "# plan makes the digest a coin flip. Zero violations here is the contract.\n"
    "#\n")
FP_HDR = (
    "# create-time plan fingerprints, one process per cell, seeded store.\n"
    "# A raced cell emits NOT_BANKED_RACED; a cell that differed across repeat\n"
    "# captures emits NONDETERMINISTIC. Both are stable FACTS about the cell -\n"
    "# recording them beats recording one lucky sample of a coin flip.\n"
    "#\n")


def main():
    out = "build_tuned/baseline"
    repeat = 3
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    if "--repeat" in sys.argv:
        repeat = int(sys.argv[sys.argv.index("--repeat") + 1])
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    print("repeat=%d" % repeat)
    for exe, hdr, name in (("harness_golden", GOLD_HDR, "golden_bits.txt"),
                           ("fp_sweep", FP_HDR, "fp_replay.txt")):
        n, flaky = capture(exe, hdr, os.path.join(out, name), repeat)
        print("%-16s %3d rows -> %s" % (exe, n, os.path.join(out, name)))
        for c in flaky:
            print("   NONDETERMINISTIC: %s" % c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
