#!/usr/bin/env python3
"""The listings harness: build every listing against an installed VectorFFT, run each one on a
fresh copy of the store, and keep what it printed for the chapters to quote. The four worked
examples (how-to-use/examples/*.c, named poisson3d, ...) are built and run the same way.

  1. cmake configures listings/ (find_package(VectorFFT), -DCMAKE_PREFIX_PATH=<prefix>) in a
     work folder outside the repository, and builds the listings asked for;
  2. each program runs in its own empty folder <work>/run/<name>/, which holds a fresh copy of
     src/wisdom/<CPU folder>/ (its wisdom2_*.txt shards) named wisdom, with VFFT_WISDOM_DIR=wisdom:
     a relative name, so the store line it prints carries no path. Every other VFFT_* variable
     of the caller's environment is removed;
  3. stdout goes to listings/out/<name>.out and stderr to listings/out/<name>.err, with
     listings/out/<name>.json recording the source's SHA-256, the exit status and the run's
     settings. Nothing is rewritten: an output that carries an absolute path, the user's name
     or the machine's name is refused (the run fails and the old files stay), and so is an
     exit status other than the one the listing declares.

A listing declares its run in comment lines the chapters never show:
  /* harness: env NAME=VALUE */     an environment variable (repeatable; never VFFT_WISDOM_DIR)
  /* harness: exit N */             the exit status it must end with (default 0)
  /* harness: args A B */           command-line arguments
  /* harness: timeout S */          seconds before it is stopped (default 900)

usage (from how-to-use/source, Python 3.13, the MinGW-w64 bin folder first on PATH):
  py -3.13 tools/listings.py --prefix <installed VectorFFT>               every listing
  py -3.13 tools/listings.py --prefix <prefix> --only ch07                the listings ch07_*
  py -3.13 tools/listings.py --status                                     which outputs are current
options: --work DIR (default: <temp>/vectorfft-guide/<only or all>), --cpu-folder 14900KF,
         --compiler gcc, --no-run (build only), --keep-going
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
LISTINGS = SOURCE / "listings"
EXAMPLES = SOURCE.parent / "examples"
OUTS = LISTINGS / "out"
REPO = SOURCE.parent.parent
sys.dont_write_bytecode = True    # no __pycache__ beside the sources
sys.path.insert(0, str(HERE))
import privacy  # noqa: E402

RE_HARNESS = re.compile(r"^\s*/\*\s*harness:\s*(\w+)\s*(.*?)\s*\*/\s*$")
RE_STORE_LINE = re.compile(r"^\[wisdom2\] wisdom: (\d+) record\(s\) loaded")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings(path):
    s = {"env": {}, "exit": 0, "args": [], "timeout": 900}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = RE_HARNESS.match(line)
        if not m:
            if "harness:" in line:
                raise SystemExit("%s:%d: a harness line is /* harness: KEY VALUE */ on a line of its own"
                                 % (path.name, n))
            continue
        key, val = m.group(1), m.group(2)
        if key == "env":
            name, eq, value = val.partition("=")
            if not eq or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise SystemExit("%s:%d: harness env takes NAME=VALUE" % (path.name, n))
            if name == "VFFT_WISDOM_DIR":
                raise SystemExit("%s:%d: every listing runs on a fresh copy of the store; "
                                 "VFFT_WISDOM_DIR is not a listing's to set" % (path.name, n))
            s["env"][name] = value
        elif key == "exit":
            s["exit"] = int(val)
        elif key == "args":
            s["args"] = val.split()
        elif key == "timeout":
            s["timeout"] = int(val)
        else:
            raise SystemExit("%s:%d: unknown harness key %r (env, exit, args, timeout)" % (path.name, n, key))
    return s


def listing_files(only):
    """The listings, then the four worked examples (how-to-use/examples/*.c, run as they are)."""
    files = sorted(list(LISTINGS.glob("*.c")) + list(LISTINGS.glob("*.cpp")))
    files += sorted(EXAMPLES.glob("*.c"))
    if only:
        files = [f for f in files if any(f.stem == o or f.stem.startswith(o + "_") for o in only)]
    return files


def status():
    rows = []
    for f in listing_files(None):
        rec = OUTS / (f.stem + ".json")
        if not rec.exists():
            st = "not run"
        else:
            st = "current" if json.loads(rec.read_text(encoding="utf-8")).get("source_sha256") == sha256(f) else "STALE"
        rows.append((f.name, st))
    for name, st in rows:
        print("%-40s %s" % (name, st))
    orphans = sorted({p.stem for p in OUTS.glob("*.json")} - {f.stem for f in listing_files(None)})
    for o in orphans:
        print("%-40s %s" % (o, "output kept for a listing that no longer exists"))
    return 0


def inside(child, parent):
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def run_cmd(cmd, env, log, cwd=None):
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        r = subprocess.run(cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", help="the installed VectorFFT (CMAKE_PREFIX_PATH)")
    ap.add_argument("--only", nargs="*", default=[], help="listing names or chapter prefixes (ch07)")
    ap.add_argument("--work", help="the build and run folder, outside the repository")
    ap.add_argument("--cpu-folder", default="14900KF", help="the store copied for each run: src/wisdom/<this>")
    ap.add_argument("--compiler", default="gcc")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="run the rest after a failure")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        return status()
    if not a.prefix:
        ap.error("--prefix <installed VectorFFT> is required")
    prefix = Path(a.prefix).resolve()
    if not (prefix / "lib" / "cmake" / "VectorFFT" / "VectorFFTConfig.cmake").exists():
        raise SystemExit("listings: %s is not an installed VectorFFT (no lib/cmake/VectorFFT)" % prefix)
    store_src = REPO / "src" / "wisdom" / a.cpu_folder
    shards = sorted(store_src.glob("wisdom2_*.txt"))
    if not shards:
        raise SystemExit("listings: no store shards in src/wisdom/%s" % a.cpu_folder)
    files = listing_files(a.only)
    if not files:
        raise SystemExit("listings: no listing matches %s" % (" ".join(a.only) or "listings/*.c"))
    tag = "_".join(a.only) if a.only else "all"
    work = Path(a.work).resolve() if a.work else Path(tempfile.gettempdir()) / "vectorfft-guide" / tag
    if inside(work, REPO):
        raise SystemExit("listings: the work folder must be outside the repository (binaries never go there)")
    cc = shutil.which(a.compiler)
    if not cc:
        raise SystemExit("listings: no compiler %r on PATH (put the MinGW-w64 bin folder first)" % a.compiler)
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("VFFT_")}
    env["PATH"] = str(Path(cc).parent) + os.pathsep + env.get("PATH", "")   # and its runtime DLLs
    (work / "logs").mkdir(parents=True, exist_ok=True)
    build = work / "build"
    print("listings: %d program(s); work folder %s" % (len(files), work))

    t0 = time.time()
    rc = run_cmd(["cmake", "-S", str(LISTINGS), "-B", str(build), "-G", "Ninja",
                  "-DCMAKE_C_COMPILER=" + a.compiler, "-DCMAKE_PREFIX_PATH=" + prefix.as_posix()],
                 env, work / "logs" / "configure.log")
    if rc != 0:
        print((work / "logs" / "configure.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        raise SystemExit("listings: configure failed (log: %s)" % (work / "logs" / "configure.log"))
    rc = run_cmd(["cmake", "--build", str(build), "--target"] + [f.stem for f in files],
                 env, work / "logs" / "build.log")
    blog = (work / "logs" / "build.log").read_text(encoding="utf-8", errors="replace")
    if rc != 0:
        print(blog[-6000:])
        raise SystemExit("listings: build failed (log: %s)" % (work / "logs" / "build.log"))
    print("listings: configured and built in %.1f s" % (time.time() - t0))
    if a.no_run:
        return 0

    OUTS.mkdir(parents=True, exist_ok=True)
    failed = []
    for f in files:
        s = settings(f)
        exe = build / (f.stem + (".exe" if os.name == "nt" else ""))
        rundir = work / "run" / f.stem
        shutil.rmtree(rundir, ignore_errors=True)
        (rundir / "wisdom").mkdir(parents=True)
        for sh in shards:
            shutil.copyfile(sh, rundir / "wisdom" / sh.name)
        before = {sh.name: sha256(sh) for sh in shards}
        renv = dict(env)
        renv.update(s["env"])
        renv["VFFT_WISDOM_DIR"] = "wisdom"
        t1 = time.time()
        try:
            r = subprocess.run([str(exe)] + s["args"], cwd=str(rundir), env=renv, capture_output=True,
                               timeout=s["timeout"])
            code, out_b, err_b = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            code, out_b, err_b = "timeout", e.stdout or b"", e.stderr or b""
        secs = time.time() - t1
        out = out_b.decode("utf-8", "replace").replace("\r\n", "\n")
        err = err_b.decode("utf-8", "replace").replace("\r\n", "\n")
        after = {p.name: sha256(p) for p in (rundir / "wisdom").glob("wisdom2_*.txt")}
        saved = after != before
        problems = []
        if code != s["exit"]:
            problems.append("exit status %s, the listing declares %s" % (code, s["exit"]))
        for stream, text in (("stdout", out), ("stderr", err)):
            for p in privacy.findings(text):
                problems.append("%s carries %s" % (stream, p))
        m = next((RE_STORE_LINE.match(l) for l in err.split("\n") if RE_STORE_LINE.match(l)), None)
        note = "served from the store" if not saved else "measured, and saved the winner"
        print("  %-36s exit %-4s %7.2f s  %s%s" % (f.stem, code, secs, note,
                                                  "" if not problems else "  REFUSED"))
        if problems:
            rej = work / "rejected"
            rej.mkdir(exist_ok=True)
            (rej / (f.stem + ".out")).write_text(out, encoding="utf-8")
            (rej / (f.stem + ".err")).write_text(err, encoding="utf-8")
            for p in problems:
                print("      " + p)
            print("      what it printed: %s" % (rej / (f.stem + ".out")))
            failed.append(f.stem)
            if not a.keep_going:
                break
            continue
        for name, text in ((f.stem + ".out", out), (f.stem + ".err", err)):
            with open(OUTS / name, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        rec = {"source": f.name, "source_sha256": sha256(f), "exit": code, "args": s["args"],
               "env": s["env"], "store": "a fresh copy of src/wisdom/%s, VFFT_WISDOM_DIR=wisdom" % a.cpu_folder,
               "store_records_loaded": int(m.group(1)) if m else None, "measured_and_saved": saved}
        with open(OUTS / (f.stem + ".json"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, indent=1) + "\n")
    if failed:
        print("listings: FAILED: %s" % ", ".join(failed))
        return 1
    print("listings: %d output(s) written to listings/out/" % len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
