#!/usr/bin/env python3
"""run_gates.py - build and run every gate, and record PASS/FAIL with no timings.

WHY A RUNNER AND NOT A SHELL LOOP
---------------------------------
Three things about this gate suite defeat a naive loop, and each one has already
produced a wrong answer at least once:

1. THE PREBUILT .exe FILES ARE STALE. Measured at the baseline: zr2c_fd_gate.exe
   dated 2026-08-23 against a vfft.c dated 2026-08-28. Running them tests old code
   and reports green. Gates are ALWAYS rebuilt here.

2. THE BUILD MODE CANNOT BE GUESSED. Grepping for vfft.h / vfft_create
   misclassifies gates that reach the library only through a module header -
   wisdom2_2d_gate calls vfft_wisdom2_2d_gate_run and matches neither pattern, so a
   grep-based classifier reported two false build failures. The mode is DISCOVERED:
   try standalone, fall back to --vfft. sp_ccol_decode_gate is the one hard
   exception - it #includes vfft.c textually, so compiling vfft.c beside it is a
   duplicate-symbol error.

3. THE WISDOM-DIR CONVENTION VARIES, AND GETTING IT WRONG INVERTS THE RESULT.
   Three spellings are in use: `--wisdir <dir>`, a BARE positional dir, and no
   argument. Passing `--wisdir` to a bare-positional gate makes the flag string
   itself become the wisdom directory and every save fails.

   Worse, the dir CONTENT matters. wisdom2_real_gate is ALL PASS against a scratch
   dir and fails 7 against the populated store - not a defect, a fixture collision:
   it stages an eng=route cell at (t=r2c, n=4096, q=1, ord=nat, place=oop) where the
   real store already holds an eng=zr2c cell, so the engine-ownership guard
   correctly declines. The gate fails BECAUSE the feature works.

   And vfft_natural_front_gate needs a COLD dir: seed it and every measure cell
   reports NO RACE, because it correctly replays instead of racing.

THE SCRATCH-WISDIR LAW
----------------------
No gate is ever pointed at src/dag-fft-compiler/generator/generated/. Every gate
gets a FRESH scratch directory. The few that need real data to do their job get a
scratch COPY of the store. A gate must never be able to write the shipped tree.

USAGE
  python build_tuned/run_gates.py [--out FILE] [--only SUBSTR] [--keep]
Exit 0 when every gate passes.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "benches")
STORE = os.path.normpath(os.path.join(
    HERE, "..", "src", "dag-fft-compiler", "generator", "generated"))

# How each gate wants its wisdom directory.
#   "none"        - takes no argument
#   "flag"        - --wisdir <dir>
#   "bare"        - <dir> as a positional
# and whether that dir must be SEEDED with a copy of the store.
#   seeded=False  - a cold, empty scratch dir (the default, and the safe one)
#   seeded=True   - a scratch COPY of the store, for gates that decode real wisdom
ARGSTYLE = {
    "k1z_inplace_gate":        ("flag", False),
    "mt_c2c_gate":             ("flag", False),
    "vfft_ilp_front_gate":     ("flag", False),
    "vfft_k1scr_gate":         ("flag", False),
    # COLD on purpose: seeding makes every measure cell report NO RACE, because it
    # correctly replays the banked verdict instead of racing.
    "vfft_natural_front_gate": ("flag", False),
    "vfft_tcbatch_gate":       ("flag", False),
    "zturn_wisdom_width_gate": ("flag", False),
    # wants a dir containing oop_wisdom.txt -> needs real data, but a COPY
    "zturn_tcut_gate":         ("flag", True),

    "il2d_m1_gate":            ("bare", False),
    "il2d_real_gate":          ("bare", False),
    "odd_partner_cells_gate":  ("bare", False),
    "wisdom2_g0_gate":         ("bare", False),
    # SCRATCH ONLY - see the fixture-collision note in the header
    "wisdom2_real_gate":       ("bare", False),
    "wisdom2_2d_gate":         ("bare", False),
    "zr2c_store_decode_gate":  ("bare", False),
    # decode real wisdom -> seeded copy
    "sp_ccol_decode_gate":     ("bare", True),
    "zr2c_fd_gate":            ("bare", True),
}

TEXTUAL = {"sp_ccol_decode_gate"}       # #includes vfft.c; must NOT add --vfft


def build(src, name):
    """Discover the build mode. -> mode string, or None if unbuildable."""
    modes = [("textual", ["--compile"])] if name in TEXTUAL else [
        ("standalone", ["--compile"]), ("vfft", ["--vfft", "--compile"])]
    for label, flags in modes:
        r = subprocess.run([sys.executable, "build.py", "--src", src] + flags,
                           cwd=HERE, capture_output=True, text=True)
        if r.returncode == 0:
            return label
    return None


def run(name, workdir):
    """Run one gate in its own scratch dir. -> (ok, tail_of_output)."""
    style, seeded = ARGSTYLE.get(name, ("none", False))
    scratch = os.path.join(workdir, name)
    os.makedirs(scratch, exist_ok=True)
    if seeded:
        for f in os.listdir(STORE):
            if f.endswith(".txt"):
                shutil.copy2(os.path.join(STORE, f), scratch)

    exe = os.path.join(BENCH, name + ".exe")
    argv = [exe]
    if style == "flag":
        argv += ["--wisdir", scratch]
    elif style == "bare":
        argv += [scratch]

    try:
        r = subprocess.run(argv, cwd=BENCH, capture_output=True, text=True,
                           timeout=900)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT after 900s"
    out = (r.stdout or "") + (r.stderr or "")
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    # prefer the gate's own verdict line over whatever happened to print last
    # (the wisdom store banner is written to stderr AFTER the verdict)
    verdict = [l for l in lines
               if any(k in l for k in ("ALL PASS", "ALL CORRECT", "FAIL",
                                       "PASS", "CORRECT", "fail"))]
    tail = verdict[-1] if verdict else (lines[-1] if lines else "<no output>")
    return r.returncode == 0, tail[:110]


def main():
    out_path = None
    only = None
    keep = "--keep" in sys.argv
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    gates = sorted(f[:-2] for f in os.listdir(BENCH)
                   if f.endswith(".c") and "gate" in f)
    if only:
        gates = [g for g in gates if only in g]

    workdir = tempfile.mkdtemp(prefix="vfft_gates_")
    lines, npass, nfail, nbuild = [], 0, 0, 0
    for g in gates:
        src = os.path.join("benches", g + ".c")
        mode = build(src, g)
        if mode is None:
            lines.append("BUILD_FAIL %-28s" % g)
            nbuild += 1
            continue
        ok, tail = run(g, workdir)
        style = ARGSTYLE.get(g, ("none", False))
        lines.append("%-4s %-10s %-6s %-28s %s"
                     % ("PASS" if ok else "FAIL", mode,
                        style[0] + ("+seed" if style[1] else ""), g, tail))
        npass += ok
        nfail += (not ok)

    hdr = ["# gate results - PASS/FAIL only, no timings, every gate REBUILT first.",
           "# Every gate runs in its OWN scratch dir; the shipped wisdom store is",
           "# never passed to a gate and never written. See run_gates.py header.",
           "#"]
    body = hdr + lines + ["#",
                          "# pass=%d fail=%d unbuildable=%d total=%d"
                          % (npass, nfail, nbuild, len(gates))]
    text = "\n".join(body) + "\n"
    if out_path:
        open(out_path, "w", encoding="utf-8", newline="\n").write(text)
    print(text, end="")

    if not keep:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print("# scratch kept at %s" % workdir)
    return 1 if (nfail or nbuild) else 0


if __name__ == "__main__":
    sys.exit(main())
