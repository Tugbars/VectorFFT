#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gauntlet.py -- the VectorFFT gauntlet driver (2026-09-22).

    python gauntlet/gauntlet.py run       --group pow2 [--max 8388608] [--threads T] [--inplace] [--name X] [--calibrate] [--merge]
    python gauntlet/gauntlet.py run       --cells 4096 | 2..4096 | 1000,1024,4096 | @file
    python gauntlet/gauntlet.py run       --group primes --max 16384
    python gauntlet/gauntlet.py run       --group mixed --max 4000000 --primes 2,3,5
    python gauntlet/gauntlet.py cells     --group mixed --max 4000000          (list + duration estimate, nothing runs)
    python gauntlet/gauntlet.py calibrate / bench / report / merge / verify / gflops ... (the run's stages, one at a time)

A run = calibrate (one front-door create per cell on a scratch copy of the
shipped wisdom; a miss races and banks, --calibrate re-races every cell) ->
bench (one process per cell, both engine orders, a control cell every 100
cells) -> report (report.md). Contracts: the default is 1D c2c, K=1, natural
order, out of place, one thread; --threads T and --inplace are the others,
each with its own csv/log/report suffix. A stopped run resumes on the same
--name. Windows: the machine is kept awake for the run's duration.
"""
import argparse, csv, ctypes, datetime, io, math, os, re, shutil, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHIPPED = os.path.join(ROOT, "src", "dag-fft-compiler", "generator", "generated")
RESULTS = os.path.join(HERE, "results")
WISDOM_FILES = ("wisdom2_oop.txt", "wisdom2_prime.txt", "wisdom2_scr.txt", "wisdom2_2d.txt",
                "wisdom2_3d.txt", "wisdom2_real.txt", "spike_wisdom.txt")
EXE = ".exe" if os.name == "nt" else ""
CONTROL_N = 4096
CONTROL_EVERY = 100
PACE_MS = 300


# ── cells ──────────────────────────────────────────────────────────────────

def fac(n):
    f, d = [], 2
    while d * d <= n:
        while n % d == 0:
            f.append(d); n //= d
        d += 1
    if n > 1:
        f.append(n)
    return f


def isprime(n):
    return n > 1 and len(fac(n)) == 1


def smooth(limit, primes):
    out = {1}
    for p in primes:
        new = set()
        for x in out:
            y = x
            while y * p <= limit:
                y *= p; new.add(y)
        out |= new
    return sorted(x for x in out if x >= 2)


def parse_cells(spec):
    cells = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("@"):
            for line in io.open(part[1:], encoding="utf-8"):
                line = line.split("#")[0].strip()
                if line:
                    cells.add(int(line.split()[0]))
        elif ".." in part:
            a, b = part.split("..")
            cells.update(range(int(a), int(b) + 1))
        else:
            cells.add(int(part))
    return sorted(cells)


def group_cells(group, maxn, primes):
    if group == "pow2":
        maxn = maxn or 1 << 23
        return [1 << e for e in range(1, 24) if (1 << e) <= maxn]
    if group == "primes":
        maxn = maxn or 16384
        sieve = bytearray([1]) * (maxn + 1); sieve[0] = sieve[1] = 0
        for i in range(2, int(maxn ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
        return [i for i in range(2, maxn + 1) if sieve[i]]
    if group == "mixed":
        maxn = maxn or 4_000_000
        return smooth(maxn, primes or (2, 3, 5))
    if group == "all":
        maxn = maxn or 4096
        return list(range(2, maxn + 1))
    raise SystemExit("unknown group %r (pow2 | primes | mixed | all)" % group)


def estimate_seconds(cells, threads, calibrate):
    """a rough pace model from the runs of 2026-09-20..22 on the i9-14900KF"""
    cal = 0.0
    for n in cells:
        if n & (n - 1) == 0:
            cal += 3 if n <= 2048 else 20 if n <= 262144 else 60
        elif isprime(n):
            cal += 1 if n <= 2048 else 2 + n / 40000.0
        else:
            cal += 4 if n <= 2048 else 8 if n <= 4096 else 20 if n <= 65536 else 60
    bench = sum(1.5 if n <= 4096 else 5 if n <= 262144 else 30 for n in cells)
    if not calibrate:
        cal *= 0.15   # replay: the create still runs, without the race
    return cal, bench


# ── the run directory and its store ────────────────────────────────────────

class Run:
    def __init__(self, args):
        self.args = args
        self.threads = int(args.threads)
        self.ip = 1 if args.inplace else 0
        self.sfx = ("_ip" if self.ip else "") + ("_mt%d" % self.threads if self.threads > 1 else "")
        name = args.name or self.default_name()
        self.dir = os.path.join(RESULTS, name)
        self.store = args.store or os.path.join(self.dir, "store")
        self.cells_file = os.path.join(self.dir, "cells.txt")
        self.cal_log = os.path.join(self.dir, "calibrate%s.log" % self.sfx)
        self.csv = os.path.join(self.dir, "gauntlet%s.csv" % self.sfx)
        self.ctl = os.path.join(self.dir, "control%s.csv" % self.sfx)
        self.log = os.path.join(self.dir, "run.log")
        self.bin = args.bin_dir or self.find_bin_dir()

    def default_name(self):
        a = self.args
        base = a.group or ("cells_" + re.sub(r"[^0-9a-zA-Z]+", "_", a.cells or "x").strip("_"))
        return "%s_%s" % (base, datetime.date.today().strftime("%Y-%m-%d"))

    @staticmethod
    def find_bin_dir():
        cands = [HERE] + sorted(os.path.join(ROOT, d, "gauntlet") for d in os.listdir(ROOT) if d.startswith("build") and os.path.isdir(os.path.join(ROOT, d, "gauntlet")))
        for c in cands:
            if os.path.isfile(os.path.join(c, "bench_1d_vs_mkl" + EXE)):
                return c
        return HERE

    def exe(self, name):
        p = os.path.join(self.bin, name + EXE)
        if not os.path.isfile(p):
            raise SystemExit("missing %s -- build the gauntlet first (see gauntlet/README.md), or pass --bin-dir" % p)
        return p

    def prepare(self, cells):
        os.makedirs(self.dir, exist_ok=True)
        if not os.path.isdir(self.store) or not os.listdir(self.store):
            os.makedirs(self.store, exist_ok=True)
            for f in WISDOM_FILES:
                src = os.path.join(SHIPPED, f)
                if os.path.isfile(src):
                    shutil.copyfile(src, os.path.join(self.store, f))
            self.note("store: fresh copy of the shipped wisdom (%s)" % SHIPPED)
        if cells:
            io.open(self.cells_file, "w", encoding="utf-8", newline="\n").write("\n".join(map(str, cells)) + "\n")

    def note(self, msg):
        line = "%s %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line, flush=True)
        with io.open(self.log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def env(self):
        e = os.environ.copy()
        e["VFFT_WISDOM_DIR"] = self.store
        if self.threads > 1:
            e["VFFT_MT"] = str(self.threads)
        if os.name == "nt":
            # the comparator's runtime and the toolchain's runtime, when present
            extra = [os.path.join(os.environ.get("MKLROOT", ""), "bin"),
                     r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
                     r"C:\Program Files\Intel\oneAPI\mkl\latest\bin"]
            e["PATH"] = os.pathsep.join([d for d in extra if d and os.path.isdir(d)] + [e.get("PATH", "")])
        return e


# ── keep the machine awake (Windows) ───────────────────────────────────────

def keep_awake(on):
    if os.name != "nt":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if on else ES_CONTINUOUS)
    except Exception:
        pass


# ── the store's rows for one cell (raced / replayed / refused) ─────────────

def cell_rows(store, n, ip):
    """the wisdom rows that decide this cell: the K=1 rows at its placement (both
    order classes) and the prime row. Returns a dict key -> payload."""
    out = {}
    pl = "ip" if ip else "oop"
    for f, pat in (("wisdom2_oop.txt", r"@cell t=c2c n=%d q=1 ord=\w+ place=%s [^|\n]*\| ([^\n]*)" % (n, pl)),
                   ("wisdom2_prime.txt", r"@cell t=c2c n=%d q=1 [^|\n]*\| ([^\n]*)" % n)):
        p = os.path.join(store, f)
        if not os.path.isfile(p):
            continue
        t = io.open(p, encoding="utf-8", errors="ignore").read()
        for m in re.finditer(pat, t):
            key = m.group(0).split(" | ")[0]
            out[key] = re.sub(r" date=\S+", "", m.group(1))
    return out


def route_of(rows):
    for k, v in rows.items():
        if "ord=nat" in k and "il_route=" in v:
            return re.search(r"il_route=(\w+)", v).group(1)
    for k, v in rows.items():
        if "eng=rader" in v or "eng=bluestein" in v:
            return "prime"
    return "-"


# ── the stages ─────────────────────────────────────────────────────────────

def stage_calibrate(run, cells, recal):
    done = set()
    if os.path.isfile(run.cal_log):
        for line in io.open(run.cal_log, encoding="utf-8", errors="ignore"):
            f = line.split()
            if len(f) >= 2 and f[0].isdigit() and f[1] in ("banked", "REFUSED") and not recal:
                done.add(int(f[0]))
    todo = [n for n in cells if n not in done]
    run.note("calibrate: %d cells (%d already done)%s" % (len(todo), len(done), ", RECALIBRATE (every cell re-raced)" if recal else ""))
    probe = run.exe("recal_1d_probe")
    t0 = time.time()
    for i, n in enumerate(todo, 1):
        before = cell_rows(run.store, n, run.ip)
        s0 = time.time()
        r = subprocess.run([probe, run.store, str(n), "0", str(run.ip), str(run.threads), "1" if recal else "0"],
                           capture_output=True, text=True, errors="replace", env=run.env())
        ms = int((time.time() - s0) * 1000)
        text = r.stdout + r.stderr
        status = "REFUSED" if "REFUSED" in text else ("banked" if "banked" in text else "ERROR")
        after = cell_rows(run.store, n, run.ip)
        if status != "banked":
            served = "refused"
        elif after != before:
            served = "raced"
        else:
            served = "replayed"
        route = route_of(after)
        # replace an older line for this cell (rerun) so the log has one line per cell
        lines = []
        if os.path.isfile(run.cal_log):
            lines = [l for l in io.open(run.cal_log, encoding="utf-8", errors="ignore") if not (l.split() and l.split()[0] == str(n))]
        lines.append("%-10d %-8s %7dms %-9s %s\n" % (n, status, ms, served, route))
        io.open(run.cal_log, "w", encoding="utf-8", newline="\n").write("".join(lines))
        if i % 25 == 0 or i == len(todo):
            run.note("  calibrated %d/%d (%s), %.0f s elapsed" % (i, len(todo), n, time.time() - t0))
    return todo


def bench_cell(run, n, csv_path):
    bench = run.exe("bench_1d_vs_mkl")
    flag = ["--k1nat" if run.ip else "--k1noop"] + (["--mt"] if run.threads > 1 else [])
    ok = True
    for flip in ("0", "1"):
        r = subprocess.run([bench] + flag + [os.path.join(run.store, "spike_wisdom.txt"), csv_path,
                           str(PACE_MS), str(n), "1", str(PACE_MS), flip, "2"],
                           capture_output=True, text=True, errors="replace", env=run.env())
        if r.returncode != 0:
            ok = False
            run.note("  bench %d flip %s: exit %d: %s" % (n, flip, r.returncode, (r.stderr or r.stdout).strip().splitlines()[-1:] ))
    return ok


def control_cell(run):
    tmp = os.path.join(run.dir, ".ctl.tmp.csv")
    if os.path.isfile(tmp):
        os.remove(tmp)
    bench_cell(run, CONTROL_N, tmp)
    if os.path.isfile(tmp):
        lines = io.open(tmp, encoding="utf-8", errors="ignore").read().splitlines(True)
        if lines:
            if not os.path.isfile(run.ctl):
                io.open(run.ctl, "w", encoding="utf-8", newline="").write(lines[0])
            io.open(run.ctl, "a", encoding="utf-8", newline="").write("".join(lines[1:]))
        os.remove(tmp)


def stage_bench(run, cells):
    banked = set()
    if os.path.isfile(run.cal_log):
        for line in io.open(run.cal_log, encoding="utf-8", errors="ignore"):
            f = line.split()
            if len(f) >= 2 and f[0].isdigit() and f[1] == "banked":
                banked.add(int(f[0]))
    have = {}
    if os.path.isfile(run.csv):
        for r in csv.DictReader(open(run.csv, encoding="utf-8", errors="ignore")):
            try:
                have[int(r["N"])] = have.get(int(r["N"]), 0) + 1
            except (KeyError, ValueError):
                pass
    todo = [n for n in cells if n in banked and have.get(n, 0) < 2]
    skipped = [n for n in cells if n not in banked]
    run.note("bench: %d cells (%d already benched, %d not banked -> not benched)" % (len(todo), sum(1 for n in cells if have.get(n, 0) >= 2), len(skipped)))
    if not todo:
        return
    control_cell(run)
    t0 = time.time()
    for i, n in enumerate(todo, 1):
        bench_cell(run, n, run.csv)
        if i % CONTROL_EVERY == 0:
            control_cell(run)
        if i % 25 == 0 or i == len(todo):
            run.note("  benched %d/%d (%s), %.0f s elapsed" % (i, len(todo), n, time.time() - t0))
    control_cell(run)


def stage_report(run):
    sys.path.insert(0, HERE)
    import gauntlet_report
    text = gauntlet_report.build(run.dir, run.sfx)
    path = os.path.join(run.dir, "report%s.md" % run.sfx)
    io.open(path, "w", encoding="utf-8").write(text)
    print(text)
    run.note("report: %s" % path)


def stage_verify(run, cells):
    probe = run.exe("k1_fwd_ref_probe")
    args = [probe] + (["--ip"] if run.ip else []) + [run.store] + [str(n) for n in cells]
    r = subprocess.run(args, capture_output=True, text=True, errors="replace", env=run.env())
    out = [l for l in (r.stdout + r.stderr).splitlines() if not l.startswith("[")]
    print("\n".join(out[-min(len(out), len(cells) + 3):]))
    run.note("verify: %s" % (out[-1] if out else "no output"))


def stage_merge(run):
    """the run's store rows into the shipped wisdom: a row keyed the same REPLACES
    the shipped one, a new row is ADDED, shipped rows the run lacks are KEPT;
    backups beside the shipped files."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    for f in WISDOM_FILES:
        src, dst = os.path.join(run.store, f), os.path.join(SHIPPED, f)
        if not os.path.isfile(src) or not os.path.isfile(dst) or f == "spike_wisdom.txt":
            continue
        def parse(p):
            head, rows, order = [], {}, []
            for line in io.open(p, encoding="utf-8", errors="ignore").read().split("\n"):
                if line.startswith("@cell ") and " | " in line:
                    key = line.split(" | ")[0]
                    if key not in rows:
                        order.append(key)
                    rows[key] = line
                else:
                    head.append(line)
            return head, rows, order
        h_d, r_d, o_d = parse(dst)
        h_s, r_s, o_s = parse(src)
        replaced = sum(1 for k in o_s if k in r_d and r_d[k] != r_s[k])
        added = sum(1 for k in o_s if k not in r_d)
        if not replaced and not added:
            print("%-18s unchanged" % f)
            continue
        shutil.copyfile(dst, dst + ".bak_%s" % stamp)
        for k in o_s:
            if k not in r_d:
                o_d.append(k)
            r_d[k] = r_s[k]
        body = [l for l in h_d if l.strip()] + [r_d[k] for k in o_d]
        io.open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(body) + "\n")
        print("%-18s %d replaced, %d added -> %s (backup .bak_%s)" % (f, replaced, added, dst, stamp))
    run.note("merged into %s" % SHIPPED)


def stage_gflops(run):
    """the run as a GFLOPS list, ours vs the comparator, one line per cell: the
    bench's convention (5 N log2 N K per transform; the best of the two engine
    orders for each engine), written to gflops<sfx>.csv beside the run's csv."""
    if not os.path.isfile(run.csv):
        raise SystemExit("no %s yet -- bench first" % run.csv)
    rows = {}
    for r in csv.DictReader(open(run.csv, encoding="utf-8", errors="ignore")):
        try:
            rows.setdefault(int(r["N"]), []).append(r)
        except (KeyError, ValueError):
            pass
    out = os.path.join(run.dir, "gflops%s.csv" % run.sfx)
    has_cmp = any(float(r.get("mkl_ns", 0) or 0) > 0 for rs in rows.values() for r in rs)
    lines = ["N,K,route,vfft_ns,vfft_gflops,mkl_ns,mkl_gflops,mkl_over_vfft"]
    table = []
    for n in sorted(rows):
        rs = rows[n]
        k = int(rs[0].get("K", 1) or 1)
        flops = 5.0 * n * math.log2(n) * k
        v_ns = min(int(r["vfft_ns"]) for r in rs)
        v_gf = flops / v_ns if v_ns > 0 else 0.0
        m_ns = min(int(r["mkl_ns"]) for r in rs) if has_cmp else 0
        m_gf = flops / m_ns if m_ns > 0 else 0.0
        lines.append("%d,%d,%s,%d,%.3f,%d,%.3f,%.3f" % (n, k, rs[0]["route"], v_ns, v_gf, m_ns, m_gf, (m_ns / v_ns) if (m_ns and v_ns) else 0.0))
        table.append((n, rs[0]["route"], v_gf, m_gf))
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("GFLOPS, %s (5 N log2 N per transform, best of the two engine orders)" % ("VectorFFT vs MKL" if has_cmp else "VectorFFT"))
    print(" %9s  %-7s %10s %10s %7s" % ("N", "route", "VectorFFT", "MKL" if has_cmp else "", "x" if has_cmp else ""))
    for n, rt, v, m in table:
        print(" %9d  %-7s %10.2f %10s %7s" % (n, rt, v, ("%.2f" % m) if has_cmp else "", ("%.2f" % (m and v / m)) if has_cmp and m else ""))
    if table:
        vs = [t[2] for t in table]
        print(" VectorFFT: median %.2f, peak %.2f GFLOPS at N=%d" % (statistics.median(vs), max(vs), max(table, key=lambda t: t[2])[0]))
        if has_cmp:
            ms = [t[3] for t in table]
            print(" MKL:       median %.2f, peak %.2f GFLOPS at N=%d" % (statistics.median(ms), max(ms), max(table, key=lambda t: t[3])[0]))
    run.note("gflops: %s" % out)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=["run", "calibrate", "bench", "report", "merge", "cells", "verify", "gflops"])
    ap.add_argument("--cells", help="4096 | 2..4096 | 1000,1024,4096 | @file")
    ap.add_argument("--group", choices=["pow2", "primes", "mixed", "all"])
    ap.add_argument("--max", type=int, help="ceiling for a group (pow2 2^23, primes 16384, mixed 4000000)")
    ap.add_argument("--primes", help="the prime set of the mixed group, e.g. 2,3,5,7 (default 2,3,5)")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--name", help="run directory name under gauntlet/results/ (default: group_date)")
    ap.add_argument("--store", help="use this wisdom store instead of a fresh copy of the shipped one")
    ap.add_argument("--bin-dir", help="where the gauntlet binaries are (default: beside the sources, else <build>/gauntlet)")
    ap.add_argument("--calibrate", action="store_true", help="re-race every cell (recalibrate) instead of replaying the shipped verdicts")
    ap.add_argument("--merge", action="store_true", help="after the run, merge the store's verdicts into the shipped wisdom")
    ap.add_argument("--yes", action="store_true", help="do not stop for the duration estimate")
    args = ap.parse_args()

    cells = []
    if args.cells:
        cells = parse_cells(args.cells)
    elif args.group:
        cells = group_cells(args.group, args.max, tuple(int(x) for x in args.primes.split(",")) if args.primes else None)
    run = Run(args)
    if args.verb in ("run", "calibrate", "bench", "cells", "verify") and not cells:
        if os.path.isfile(run.cells_file):
            cells = parse_cells("@" + run.cells_file)
        else:
            raise SystemExit("give --cells or --group")
    cal_s, bench_s = estimate_seconds(cells, run.threads, args.calibrate)
    if args.verb == "cells":
        print("%d cells: %s%s" % (len(cells), " ".join(map(str, cells[:12])), " ..." if len(cells) > 12 else ""))
        print("estimate on this class of machine: calibrate ~%.0f min%s, bench ~%.0f min" % (cal_s / 60, "" if args.calibrate else " (replay)", bench_s / 60))
        return 0

    run.prepare(cells if args.verb in ("run", "calibrate") else None)
    run.note("%s: %d cells, contract %s%s, bin %s" % (args.verb, len(cells), "in place" if run.ip else "out of place",
                                                     ", T=%d" % run.threads if run.threads > 1 else "", run.bin))
    if args.verb in ("run", "calibrate") and not args.yes and (cal_s + bench_s) > 1800:
        print("estimated %.0f min (calibrate %.0f + bench %.0f). Pass --yes to skip this prompt." % ((cal_s + bench_s) / 60, cal_s / 60, bench_s / 60))
        if input("continue? [y/N] ").strip().lower() != "y":
            return 1
    keep_awake(True)
    try:
        if args.verb in ("run", "calibrate"):
            stage_calibrate(run, cells, args.calibrate)
        if args.verb in ("run", "bench"):
            stage_bench(run, cells)
        if args.verb == "verify":
            stage_verify(run, cells)
        if args.verb in ("run", "bench", "report"):
            stage_report(run)
        if args.verb == "gflops":
            stage_gflops(run)
        if args.verb == "merge" or (args.verb == "run" and args.merge):
            stage_merge(run)
    finally:
        keep_awake(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
