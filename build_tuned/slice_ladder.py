#!/usr/bin/env python3
"""slice_ladder.py - run the whole migration ladder for ONE slice step.

WHY THIS EXISTS
---------------
Steps 22-27 each cut a create tier out of `_vfft_create_inner`. They are the
same five checks every time, and a ladder run by hand is a ladder with a rung
quietly skipped: the harness's own history is a list of checks that stopped
checking (a compiled-out assert, a CRLF diff, a cell count inferred from output
length). Driving all five from one command makes skipping a rung impossible
rather than merely unlikely.

THE RUNGS, and what each one is the only witness for

  1. identity object   the DOCUMENTED flags -- `-O2 -mavx2 -mfma` plus
                       baseline/include_flags.txt, from the REPO ROOT, and NO
                       `-w`. Not build.py's flag set: build.py compiles at a
                       different optimisation level and comparing across the
                       two reports ~780 changed bodies that mean nothing.
  2. obj_equiv --slice the SHAPE of the cut (see obj_equiv.py's SLICE MODE).
  3. sym censuses      `undefined` catches a new cross-TU edge; `mutable`
                       catches a new file-scope object -- a `static` in a
                       header is one copy per includer, which is the failure
                       this whole migration is shaped to avoid. Both must be
                       IDENTICAL. `defined` is EXPECTED to move on a slice, so
                       it is reported, not gated.
  4. race census       the protocol constants obj_equiv is measured blind to.
  5. golden bits       THE SEMANTIC GATE. golden_bits.txt and fp_replay.txt
                       must be byte-identical. For a slice this is the rung
                       that carries the weight; everything above it is shape.

Exit 0 only when every gated rung passes.

USAGE
  python build_tuned/slice_ladder.py --parent _vfft_create_inner \\
      --helper _vfft_create_rank34 --scratch <dir> [--repeat 3]
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "build_tuned", "baseline")
CC = r"C:\mingw152\mingw64\bin\gcc.exe"


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def same(a, b):
    """Byte comparison with the LINE ENDING normalized on BOTH sides.

    The line ending is not part of any artifact's meaning, and it has produced
    a false verdict in both directions already: a CRLF baseline against an LF
    capture reported all 36 golden rows changed, and an LF-normalizing writer
    against the CRLF census files reported all 881 undefined symbols changed.
    Normalizing both sides retires the whole class rather than the last
    instance of it.
    """
    try:
        with open(a, "rb") as f, open(b, "rb") as g:
            return (f.read().replace(b"\r\n", b"\n")
                    == g.read().replace(b"\r\n", b"\n"))
    except OSError:
        return False


def main():
    def opt(name, default=None):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    parent = opt("--parent")
    helper = opt("--helper")
    scratch = opt("--scratch")
    repeat = opt("--repeat", "3")
    # A step is either a SLICE (--helper: a tier cut out of a parent, gated on
    # shape) or a MOVE (--move: whole functions relocated, where the stronger
    # classic EQUIVALENT verdict is reachable and is therefore what we demand).
    # Step 28 is both, in two parts, so the ladder has to speak both.
    is_move = "--move" in sys.argv
    if not scratch or (not is_move and not (parent and helper)):
        print(__doc__)
        return 2
    os.makedirs(scratch, exist_ok=True)

    results = []          # (gated, name, ok, detail)
    obj = os.path.join(scratch, "vfft_new.o")

    # ---- 1. identity object, documented flags -----------------------------
    incs = open(os.path.join(BASE, "include_flags.txt")).read().split()
    r = run([CC, "-c", "-O2", "-mavx2", "-mfma"] + incs +
            [os.path.join("src", "core", "vfft.c"), "-o", obj])
    if r.returncode != 0:
        print("BUILD FAILED\n" + r.stderr[-4000:])
        return 1
    warns = [l for l in r.stderr.splitlines() if "warning:" in l]
    results.append((False, "identity build", True,
                    "%d warning(s)" % len(warns)))
    for w in warns:
        print("   warn: " + w.strip())

    # ---- 2. slice shape, or full equivalence for a move -------------------
    cmd = [sys.executable, "build_tuned/obj_equiv.py",
           os.path.join(BASE, "vfft_baseline.o"), obj]
    if not is_move:
        cmd += ["--slice", "%s:%s" % (parent, helper)]
    r = run(cmd)
    verdict = [l for l in r.stdout.splitlines()
               if l.startswith("SLICE") or l.startswith("  - ")
               or l.startswith("NOT EQUIV") or l.startswith("EQUIVALENT")
               or l.startswith("  DISAPPEARED") or l.startswith("  APPEARED")
               or l.startswith("  BODY CHANGED")]
    results.append((True, "obj_equiv " + ("(move)" if is_move else "--slice"),
                    r.returncode == 0, verdict[0] if verdict else "?"))
    for l in verdict[1:]:
        print("   " + l.strip())

    # ---- 3. symbol censuses ----------------------------------------------
    for mode, ref in (("undefined", "nm_undefined.txt"),
                      ("mutable", "mutable_objects.txt"),
                      ("defined", "nm_defined.txt")):
        out = os.path.join(scratch, "nm_%s.txt" % mode)
        r = run([sys.executable, "build_tuned/sym_census.py", obj, "--" + mode])
        with open(out, "wb") as f:
            f.write(r.stdout.replace("\r\n", "\n").encode())
        ok = same(os.path.join(BASE, ref), out)
        gated = mode != "defined" or is_move
        detail = "identical" if ok else "DIFFERS (expected on a slice)" \
            if not gated else "DIFFERS"
        results.append((gated, "sym_census --" + mode, ok, detail))

    # ---- 4. race protocol census -----------------------------------------
    out = os.path.join(scratch, "race_census.txt")
    r = run([sys.executable, "build_tuned/race_census.py"])
    with open(out, "wb") as f:
        f.write(r.stdout.replace("\r\n", "\n").encode())
    # The census's own contract: "fn is an ATTRIBUTE, so a MOVE shows as an fn
    # change on a stable key, not a delete plus an add." A slice moves races
    # from _vfft_create_inner into the new helper, so the fn attribute is
    # EXPECTED to change and the protocol KEY is what must not. Gate the keys;
    # report the fn drift.
    def race_keys(path):
        keys = []
        try:
            for line in open(path, encoding="utf-8", errors="replace"):
                if line.startswith("#") or not line.strip():
                    continue
                keys.append(re.sub(r"fn=\S+", "fn=<moved>", line.rstrip("\n")))
        except OSError:
            return None
        return sorted(keys)

    kb = race_keys(os.path.join(BASE, "race_census.txt"))
    kn = race_keys(out)
    ok = kb is not None and kb == kn
    detail = "protocol constants"
    if not ok and kb is not None and kn is not None:
        lost = [k for k in kb if k not in kn]
        gained = [k for k in kn if k not in kb]
        detail = "%d row(s) lost, %d gained" % (len(lost), len(gained))
        for r in lost[:4]:
            print("   census LOST  : " + r[:110])
        for r in gained[:4]:
            print("   census GAINED: " + r[:110])
    results.append((True, "race_census (keys)", ok, detail))

    # ---- 5. golden bits (THE semantic gate) -------------------------------
    cap = os.path.join(scratch, "cap")
    if os.path.isdir(cap):
        for f in os.listdir(cap):
            os.remove(os.path.join(cap, f))
    r = run([sys.executable, "build_tuned/capture_baseline.py",
             "--out", cap, "--repeat", repeat])
    if r.returncode != 0:
        print("CAPTURE FAILED\n" + (r.stderr or r.stdout)[-3000:])
        return 1
    for ref in ("golden_bits.txt", "fp_replay.txt"):
        results.append((True, "golden " + ref,
                        same(os.path.join(BASE, ref), os.path.join(cap, ref)),
                        "byte-identical"))

    # ---- verdict ----------------------------------------------------------
    print("\n%-26s %-8s %s" % ("RUNG", "RESULT", "DETAIL"))
    failed = 0
    for gated, name, ok, detail in results:
        tag = ("PASS" if ok else "FAIL") if gated else ("ok" if ok else "info")
        if gated and not ok:
            failed += 1
        print("%-26s %-8s %s" % (name, tag, detail))
    if failed:
        print("\nLADDER FAILED - %d gated rung(s). Stop rule: revert, do not "
              "triage in place." % failed)
        return 1
    print("\nLADDER GREEN - %s is clean."
          % ("move" if is_move else "slice %s" % helper))

    if "--restamp" in sys.argv:
        # The reference must advance, or step N+1 is diffed against step N-1
        # and reports N's expected symbol move as its own finding. Only the two
        # artifacts a slice legitimately moves are re-stamped: the identity
        # object and the DEFINED census. The gated artifacts (undefined,
        # mutable, race census, golden bits) were just proven identical, so
        # re-stamping them would be writing back what is already there -- and
        # a re-stamp that can rewrite a gated artifact is a laundering path.
        import shutil
        shutil.copyfile(obj, os.path.join(BASE, "vfft_baseline.o"))
        shutil.copyfile(os.path.join(scratch, "nm_defined.txt"),
                        os.path.join(BASE, "nm_defined.txt"))
        sha = run(["git", "rev-parse", "HEAD"]).stdout.strip()
        note = opt("--note", helper)
        with open(os.path.join(BASE, "baseline_sha.txt"), "w",
                  newline="\n") as f:
            f.write("%s\n# short: %s  (re-stamped at %s)\n"
                    % (sha, sha[:8], note))
        print("RE-STAMPED: vfft_baseline.o, nm_defined.txt, baseline_sha.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
