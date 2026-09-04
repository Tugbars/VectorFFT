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


def _kill_children_when_i_die():
    """Put this process in a Windows Job Object that kills its children on close.

    WHY: the per-gate timeout below lives in the PARENT. If the parent is killed
    - an outer timeout, a Ctrl-C, the harness reaping it - the child gate is
    ORPHANED and then has no timeout at all. That is not hypothetical: a single
    orphaned mt_c2c_gate ran unsupervised for 1,674 seconds of CPU, and killing
    it by name only started the next one, because the parent was still alive
    spawning them.

    A Job Object with KILL_ON_JOB_CLOSE is the actual guarantee: the handle dies
    with the process by any means, including SIGKILL, and every child in the job
    dies with it. atexit and signal handlers do NOT cover the killed case.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _LIMIT(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _IO(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_uint64),
                        ("WriteOperationCount", ctypes.c_uint64),
                        ("OtherOperationCount", ctypes.c_uint64),
                        ("ReadTransferCount", ctypes.c_uint64),
                        ("WriteTransferCount", ctypes.c_uint64),
                        ("OtherTransferCount", ctypes.c_uint64)]

        class _EXT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _LIMIT),
                        ("IoInfo", _IO),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return
        info = _EXT()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info),
                                           ctypes.sizeof(info)):
            return
        k32.AssignProcessToJobObject(job, k32.GetCurrentProcess())
        # keep the handle alive for the life of the process, deliberately leaked
        globals()["_VFFT_JOB"] = job
    except Exception:
        pass                      # best effort; the per-gate timeout still applies

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "benches")
# Gates run from the REPO ROOT, not from benches/. fftw_bind_gate writes its
# wisdom to the repo-root-relative path "build_tuned/benches/_fftw_bind_gate.wis";
# from benches/ that resolves to benches/build_tuned/benches/... which does not
# exist, so export and import both return 0 and the gate reports a FALSE RED
# (verified: VERDICT FAIL from benches/, VERDICT PASS from the root). A harness
# that manufactures reds is worse than none - it teaches you to ignore it.
ROOT = os.path.dirname(HERE)
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
    "k1z_inplace_gate":        ("flag", True),
    "mt_c2c_gate":             ("flag", True),
    # COLD on purpose, same class as vfft_natural_front_gate: it asserts a RACE
    # OCCURRED. Seeded, it correctly replays the banked verdict and reports
    # "NO RACE" with accuracy fine (7.8e-16) - a green engine, a red gate.
    # These two are the whole class: grep "NO RACE" over benches/*gate*.c.
    "vfft_ilp_front_gate":     ("flag", False),
    "vfft_k1scr_gate":         ("flag", True),
    # COLD: the MONO tier (solo kernels) races once per cell and must land route=mono
    "il_solo_gate":            ("flag", False),
    # COLD on purpose: seeding makes every measure cell report NO RACE, because it
    # correctly replays the banked verdict instead of racing.
    "vfft_natural_front_gate": ("flag", False),
    "nat_bankloss_gate": ("flag", False),      # cold: the banked-loss law races on pass 1 by design
    "vfft_tcbatch_gate":       ("flag", True),
    "zturn_wisdom_width_gate": ("flag", True),
    # wants a dir containing oop_wisdom.txt -> needs real data, but a COPY
    "zturn_tcut_gate":         ("flag", True),

    "il2d_m1_gate":            ("bare", True),
    "il2d_real_gate":          ("bare", True),
    "odd_partner_cells_gate":  ("bare", True),
    "wisdom2_g0_gate":         ("bare", False),
    # SCRATCH ONLY - see the fixture-collision note in the header
    "wisdom2_real_gate":       ("bare", False),
    "wisdom2_2d_gate":         ("bare", True),  # SEEDED: create-twice coherence needs the 2D shard; VFFT_WISDOM2_OFF is RETIRED (no legacy arm, no dual fixture)
    "zr2c_store_decode_gate":  ("bare", True),
    # decode real wisdom -> seeded copy
    "sp_ccol_decode_gate":     ("bare", True),
    "zr2c_fd_gate":            ("bare", True),
    "pool_preserve_gate":      ("bare", True),   # SEEDED: the OOP natural race needs a banked K=1 cascade cell to replay into zct
    "natorder_scratch_gate":   ("bare", True),   # SEEDED: must REPLAY mode=pcyc (cycle reorder is the scratch-using path)
}

TEXTUAL = {"sp_ccol_decode_gate"}       # #includes vfft.c; must NOT add --vfft

# Wall-clock budget overrides, seconds (see run()). Only gates whose honest
# runtime does not fit the flat seeded/cold split belong here.
BUDGET_OVERRIDE = {
    "vfft_natural_front_gate": 1800,   # cold races at 5 N x 4 passes + reload: 12-18 min on the i9
    "zturn_tcut_gate":         900,    # 4 cells x (arms + naive-DFT reference per tiled arm): 576 s measured uncapped on a store that already serves its cells (2026-09-02); the time is the correctness work, not recalibration
    "odd_partner_cells_gate":  900,    # 20 cells x (correctness + A/B build pair) and wisdom_write=0: it cannot seed itself, so it recalibrates every run (464 s measured uncapped, 2026-09-02)
}


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

    # 300s, not 900: with a SEEDED dir a gate replays banked wisdom and finishes
    # in seconds. A gate that needs minutes is recalibrating, which means either
    # its dir is wrong or it genuinely races - either way, worth failing loudly
    # rather than burning half an hour.
    # A SEEDED gate replays and finishes in seconds; 300s means something is
    # wrong. A COLD gate races by design - that is its whole purpose - so it is
    # allowed to take real time. Budget by which one it is, not one flat number.
    budget = 300 if seeded else 900
    # per-gate override: a COLD gate whose honest runtime straddles the flat
    # budget on this host (vfft_natural_front: 12-18 min of cold races). The
    # organic replacement (progress watchdog / host-scaled cap) is deferred
    # to the next version; until then the constant is per gate, not global.
    budget = BUDGET_OVERRIDE.get(name, budget)
    try:
        r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                           timeout=budget)
    except subprocess.TimeoutExpired:
        # subprocess.run kills only the direct child; a gate that spawned its own
        # children would leave them behind. Kill the whole tree.
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/IM", name + ".exe"],
                           capture_output=True)
        return False, "TIMEOUT after %ds (tree killed)" % budget
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

    _kill_children_when_i_die()
    workdir = tempfile.mkdtemp(prefix="vfft_gates_")
    lines, npass, nfail, nbuild = [], 0, 0, 0

    # Emit each result AS IT COMPLETES, flushed. A run over 32 gates takes ~20
    # minutes; buffering it all to the end makes a long run indistinguishable
    # from a hung one, and leaves nothing at all behind if the run is killed -
    # which is exactly how the first attempt at this baseline was lost.
    def emit(line):
        lines.append(line)
        print(line, flush=True)

    for g in gates:
        src = os.path.join("benches", g + ".c")
        mode = build(src, g)
        if mode is None:
            emit("BUILD_FAIL %-28s" % g)
            nbuild += 1
            continue
        ok, tail = run(g, workdir)
        style = ARGSTYLE.get(g, ("none", False))
        emit("%-4s %-10s %-6s %-28s %s"
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
    # The per-gate lines were already printed live by emit(); re-printing the
    # whole body here would double every row on stdout. Only the summary.
    print("\n".join(body[-2:]))

    if not keep:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print("# scratch kept at %s" % workdir)
    return 1 if (nfail or nbuild) else 0


if __name__ == "__main__":
    sys.exit(main())
