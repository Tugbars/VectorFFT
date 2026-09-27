#!/usr/bin/env python3
"""capture_ref.py - capture every artifact of the restructuring gate into one
directory, for one ISA, on this host.

A REFERENCE is a capture of the tree BEFORE a series of steps; step_gate.py
captures the tree again after each step and compares the two, rung by rung
(docs/roadmap/layout_separation_plan.md, section 7). Both use this file, so the
reference and the step can never be captured two different ways.

WHAT IS CAPTURED (file -> rung)
  meta.txt              flags key, git sha, compiler, CPU (never compared)
  vfft_O2.o             R1  the portable identity object (toolchain.identity_flags)
  vfft_O3native.o       R1  the shipped object (gauntlet/build.py's driver flags)
  objects.sha           R1  sha256 of the two objects
  codelets.sha          R1  sha256 of every codelet .o and libdagcodelets.a
  bins.sha              R1  sha256 of the harness executables (fingerprint build)
  cmake.sha             R1  (--cmake-dir) the CMake build's libraries and tools
  vfft.i                R2  vfft.c preprocessed (-E -P), identity flags
  vfft.i.sorted         R2  its top-level declarations, whitespace-normalized, sorted
  vfft.macros           R2  -dM, sorted
  rodata_strings.txt    R2b every string in the O2 object's read-only data, sorted
  sym_defined.txt ...   R4  sym_census defined / undefined / mutable (O2 object)
  race_census.txt       R4  the protocol census (all of src/core)
  race_census_files.txt R0  the same census, site counts per file (basename)
  layout.txt            R4  every struct's size and member offsets (DWARF)
  dup_basenames.txt     R0  headers under src/core sharing a basename (must be none)
  includes_unresolved   R0  quoted #includes that resolve to no file
  warnings.txt          R0  -Wimplicit-function-declaration -Wunused-function,
                            path- and line-free, sorted
  golden_bits.txt       R5  capture_baseline.py (harness_golden)
  fp_replay.txt         R5  capture_baseline.py (fp_sweep)
  api_sweep.txt         R5  the public-API sweep, replayed from sweep_store/
  wisdom_replay.txt     R5  a stratified sample of src/wisdom's rows, replayed
  roundtrip.txt         R5  the store's load -> save round trip
  gates.txt             R5  (--gates) build_tuned/run_gates.py

USAGE
  python capture_ref.py --isa avx2|avx512 --out DIR
        [--no-semantic] [--sweep-store DIR] [--repeat N] [--jobs J]
  --repeat       reference: runs per API-sweep cell / replay row, every
                 distinct output kept as a variant (default 6); step: the
                 retries allowed to land on a reference variant
        [--cmake-dir DIR] [--gates]
  --sweep-store  replay the API sweep from this banked store (step captures
                 pass the reference's); without it the sweep is BANKED first
                 into DIR/sweep_store (reference captures).
"""
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toolchain  # noqa: E402
import sym_census  # noqa: E402

ROOT = toolchain.ROOT
VFFT_C = os.path.join(toolchain.CORE, "vfft.c")
WARN_FLAGS = ["-Wimplicit-function-declaration", "-Wunused-function"]


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def log(msg):
    print("[capture %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def write(path, text):
    with open(path, "w", newline="\n") as f:
        f.write(text)


# ------------------------------------------------------------------ objects

def compile_objects(isa, out):
    """The two identity objects, the preprocessed TU and the macros, in parallel."""
    inc = toolchain.include_flags()
    idf = toolchain.identity_flags(isa)
    shipped = toolchain.shipped_flags(isa)
    jobs = {
        "O2": [toolchain.cc(), "-c"] + idf + WARN_FLAGS + inc + [VFFT_C, "-o",
               os.path.join(out, "vfft_O2.o")],
        "O3": [toolchain.cc(), "-c"] + shipped + inc + [VFFT_C, "-o",
               os.path.join(out, "vfft_O3native.o")],
        "E": [toolchain.cc(), "-E", "-P"] + idf + inc + [VFFT_C, "-o",
              os.path.join(out, "vfft.i")],
        "dM": [toolchain.cc(), "-E", "-dM"] + idf + inc + [VFFT_C],
    }
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        res = dict(zip(jobs, ex.map(lambda c: toolchain.run(c), jobs.values())))
    for k in ("O2", "O3", "E"):
        if res[k].returncode != 0:
            raise SystemExit("compile %s failed:\n%s" % (k, res[k].stderr[-4000:]))
    write(os.path.join(out, "vfft.macros"),
          "".join(l + "\n" for l in sorted(set(res["dM"].stdout.splitlines()))))
    write(os.path.join(out, "warnings.txt"), normalize_warnings(res["O2"].stderr))
    write(os.path.join(out, "objects.sha"), "".join(
        "%s  %s\n" % (toolchain.sha256(os.path.join(out, f)), f)
        for f in ("vfft_O2.o", "vfft_O3native.o")))
    write(os.path.join(out, "vfft.i.sorted"),
          "".join(d + "\n" for d in sorted(top_level_decls(
              open(os.path.join(out, "vfft.i"), encoding="utf-8", errors="replace").read()))))
    write(os.path.join(out, "shipped_flags.txt"), " ".join(shipped) + "\n")


_WARN = re.compile(r"^(.*?):\d+:\d+: warning: (.*)$")


def normalize_warnings(stderr):
    """File BASENAME plus the message: no directory (a move changes it), no
    line or column (any edit above changes them)."""
    rows = []
    for line in stderr.splitlines():
        m = _WARN.match(line)
        if m:
            rows.append("%s: %s" % (os.path.basename(m.group(1)), m.group(2)))
    return "".join(r + "\n" for r in sorted(rows))


def top_level_decls(text):
    """Split preprocessed C into top-level items (a declaration ending in ';'
    at depth 0, or a definition closing its '}' at depth 0), normalize the
    whitespace inside each, and return them. String and character literals are
    skipped so a brace inside one does not count."""
    items, buf, depth, i, n = [], [], 0, 0, len(text)
    start = 0
    while i < n:
        ch = text[i]
        if ch in "\"'":
            q = ch
            i += 1
            while i < n and text[i] != q:
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and _is_function(text[start:i]):
                # a function definition ends at its closing brace; a
                # struct/union/enum/initializer runs on to its ';'
                items.append(text[start:i + 1])
                start = i + 1
        elif ch == ";" and depth == 0:
            items.append(text[start:i + 1])
            start = i + 1
        i += 1
    if text[start:].strip():
        items.append(text[start:])
    return [re.sub(r"\s+", " ", it).strip() for it in items if it.strip()]


def _is_function(item):
    head = item[:item.find("{")] if "{" in item else item
    h = head.strip()
    return ("(" in h and "=" not in h
            and not re.match(r"(typedef|struct|union|enum)\b", h))


def rodata_strings(obj, out):
    """Every printable string in the object's read-only sections, sorted: a
    moved file must not change one (non-ASan builds embed no paths)."""
    import elfmini
    elf = elfmini.Elf(obj)
    strs = set()
    for sec in elf.sections:
        if sec.name.startswith(".rodata") and sec.type != elfmini.SHT_NOBITS:
            strs |= set(m.decode("latin-1")
                        for m in re.findall(rb"[\x20-\x7e]{6,}", sec.data))
    write(os.path.join(out, "rodata_strings.txt"), "".join(s + "\n" for s in sorted(strs)))


def hygiene_artifacts(out):
    import hygiene
    write(os.path.join(out, "dup_basenames.txt"), "".join(
        "%s: %s\n" % (b, " ".join(ps)) for b, ps in sorted(hygiene.dup_basenames().items())))
    write(os.path.join(out, "includes_unresolved.txt"),
          "".join(r + "\n" for r in hygiene.unresolved_includes()))


def censuses(out):
    obj = os.path.join(out, "vfft_O2.o")
    for mode in ("defined", "undefined", "mutable"):
        write(os.path.join(out, "sym_%s.txt" % mode),
              "".join(r + "\n" for r in sym_census.census(obj, mode, toolchain.nm())))
    for args, name in (([], "race_census.txt"), (["--files"], "race_census_files.txt")):
        r = toolchain.run([sys.executable, os.path.join(HERE, "race_census.py")] + args)
        if r.returncode != 0:
            raise SystemExit("race_census failed: " + r.stderr[-2000:])
        write(os.path.join(out, name), r.stdout)


def layout(isa, out):
    r = toolchain.run([sys.executable, os.path.join(HERE, "layout_dump.py"),
                       "--isa", isa, "--out", os.path.join(out, "layout.txt")])
    if r.returncode != 0:
        raise SystemExit("layout_dump failed: " + r.stderr[-2000:])


def codelets(isa, out):
    """The codelet library build.py links (built by the harness builds; built
    here if absent). Codelets include no core header, so every one of these
    must stay byte-identical through the whole separation."""
    bp = toolchain.load_build_py(isa)
    tc = bp.detect_toolchain()
    lib = bp.dag_codelet_lib(tc)
    objdir = os.path.dirname(lib)
    rows = []
    for f in sorted(os.listdir(objdir)):
        if f.endswith((".o", ".a")):
            rows.append("%s  %s" % (toolchain.sha256(os.path.join(objdir, f)), f))
    write(os.path.join(out, "codelets.sha"), "".join(r + "\n" for r in rows))
    log("codelets: %d files hashed" % len(rows))


# ---------------------------------------------------------------- semantics

def build_harness(isa, name):
    env = dict(os.environ, VFFT_FINGERPRINT="1", VFFT_ISA=isa)
    env.pop("VFFT_WARN", None)
    env.pop("VFFT_ASAN", None)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "gauntlet", "build.py"),
                        "--src", os.path.join(HERE, name + ".c"), "--vfft", "--compile"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise SystemExit("build of %s failed:\n%s" % (name, (r.stdout + r.stderr)[-4000:]))
    return os.path.join(HERE, name + toolchain.EXE)


def semantics(isa, out, sweep_store, repeat, jobs):
    os.environ["VFFT_ISA"] = isa
    bins = {n: build_harness(isa, n) for n in ("harness_golden", "fp_sweep", "api_sweep")}
    write(os.path.join(out, "bins.sha"), "".join(
        "%s  %s\n" % (toolchain.sha256(p), n) for n, p in sorted(bins.items())))
    py = sys.executable
    r = toolchain.run([py, os.path.join(HERE, "capture_baseline.py"), "--out", out,
                       "--repeat", str(repeat)], env=dict(os.environ, VFFT_ISA=isa))
    if r.returncode != 0:
        raise SystemExit("capture_baseline failed:\n" + (r.stdout + r.stderr)[-3000:])
    log("golden_bits / fp_replay captured")
    exe = bins["api_sweep"]
    sw = os.path.join(HERE, "api_sweep.py")
    ref_dir = None
    if not sweep_store:
        sweep_store = os.path.join(out, "sweep_store")
        log("banking the sweep store (reference capture)")
        _ok(toolchain.run([py, sw, "bank", "--exe", exe, "--store-out", sweep_store]), "bank")
    else:
        ref_dir = os.path.dirname(os.path.abspath(sweep_store))

    def refv(name):
        # a STEP capture matches against the reference's variant sets
        return (["--ref-variants", os.path.join(ref_dir, name + ".variants.json")]
                if ref_dir else [])
    for args, what in (
            (["capture", "--store", sweep_store, "--out", os.path.join(out, "api_sweep.txt"),
              "--repeat", str(repeat), "--jobs", str(jobs)] + refv("api_sweep.txt"), "api_sweep"),
            (["replay", "--store", os.path.join(ROOT, "src", "wisdom"),
              "--out", os.path.join(out, "wisdom_replay.txt"), "--jobs", str(jobs),
              "--repeat", str(max(repeat, 4))] + refv("wisdom_replay.txt"), "wisdom_replay"),
            (["roundtrip", "--store", os.path.join(ROOT, "src", "wisdom"),
              "--out", os.path.join(out, "roundtrip.txt")], "roundtrip")):
        _ok(toolchain.run([py, sw, args[0], "--exe", exe] + args[1:]), what)
        log("%s captured" % what)


def _ok(r, what):
    if r.returncode != 0:
        raise SystemExit("%s failed:\n%s" % (what, (r.stdout + r.stderr)[-3000:]))


def cmake_leg(isa, out, bdir):
    """Re-configure (new directories are picked up at configure time only) and
    build the CMake tree, then hash its libraries and tools. Compared only
    against a reference captured from the SAME build dir: CMake's flags are
    not build.py's."""
    r = toolchain.run(["cmake", "-S", ROOT, "-B", bdir, "-DVFFT_ISA=%s" % isa])
    _ok(r, "cmake configure")
    r = toolchain.run(["cmake", "--build", bdir, "-j", str(os.cpu_count() or 4)], timeout=7200)
    _ok(r, "cmake build")
    rows = []
    for dp, _, fs in os.walk(bdir):
        if "CMakeFiles" in dp:
            continue
        for f in sorted(fs):
            p = os.path.join(dp, f)
            if f.endswith(".a") or (os.access(p, os.X_OK) and os.path.isfile(p)
                                    and "." not in f):
                rows.append("%s  %s" % (toolchain.sha256(p), os.path.relpath(p, bdir)))
    write(os.path.join(out, "cmake.sha"), "".join(r + "\n" for r in sorted(rows, key=lambda x: x[66:])))
    log("cmake: %d artifacts hashed" % len(rows))


def gates(isa, out):
    r = toolchain.run([sys.executable, os.path.join(ROOT, "build_tuned", "run_gates.py"),
                       "--out", os.path.join(out, "gates.txt")],
                      env=dict(os.environ, VFFT_ISA=isa), timeout=6 * 3600)
    log("gates: exit %d" % r.returncode)


# --------------------------------------------------------------------- main

def capture(isa, out, semantic=True, sweep_store=None, repeat=3, jobs=4,
            cmake_dir=None, run_gates=False):
    os.makedirs(out, exist_ok=True)
    os.environ["VFFT_ISA"] = isa
    t0 = time.time()
    write(os.path.join(out, "meta.txt"),
          "key=%s\nsha=%s\ncc=%s %s\ncpu=%s\nisa=%s\n"
          % (toolchain.flags_key(isa), toolchain.git_sha(), toolchain.cc(),
             toolchain.cc_version(), toolchain.cpu_model(), isa))
    log("objects (%s)" % isa)
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        f_lay = ex.submit(layout, isa, out)
        compile_objects(isa, out)
        f_lay.result()
    rodata_strings(os.path.join(out, "vfft_O2.o"), out)
    censuses(out)
    hygiene_artifacts(out)
    log("objects, preprocessed TU, censuses, layout done (%.0fs)" % (time.time() - t0))
    if semantic:
        semantics(isa, out, sweep_store, repeat, jobs)
    codelets(isa, out)
    if cmake_dir:
        cmake_leg(isa, out, cmake_dir)
    if run_gates:
        gates(isa, out)
    log("capture complete in %.0fs -> %s" % (time.time() - t0, out))


def main():
    isa = opt("--isa")
    out = opt("--out")
    if isa not in toolchain.ISAS or not out:
        print(__doc__)
        return 2
    capture(isa, os.path.abspath(out), semantic="--no-semantic" not in sys.argv,
            sweep_store=opt("--sweep-store"), repeat=int(opt("--repeat", "6")),
            jobs=int(opt("--jobs", "4")), cmake_dir=opt("--cmake-dir"),
            run_gates="--gates" in sys.argv)
    return 0



if __name__ == "__main__":
    sys.exit(main())
