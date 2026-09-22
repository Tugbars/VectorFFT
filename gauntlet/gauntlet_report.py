# -*- coding: utf-8 -*-
"""gauntlet_report.py -- the tables of a gauntlet run (2026-09-22).

Reads a run directory (gauntlet.csv + control.csv + calibrate.log + store/) and
writes report.md: the per-cell table (how each cell was SERVED: route, raced /
replayed / refused), the route table, the size bands, the family table, the
control series, flip agreement. Every ratio is comparator time / our time, the
WORSE of the two engine orders; without a comparator the columns are ns and
GFLOPS only. Used by gauntlet.py's `report` verb; runnable alone:

    python gauntlet_report.py <run-dir> [--contract oop|ip] [--threads T]
"""
import csv, collections, io, math, os, re, statistics, sys


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


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def read_calibrate(path):
    """N -> (status, ms, served, route)"""
    out = {}
    if not os.path.isfile(path):
        return out
    for line in io.open(path, encoding="utf-8", errors="ignore"):
        f = line.split()
        if len(f) >= 3 and f[0].isdigit():
            ms = int(f[2].rstrip("ms")) if f[2].endswith("ms") else 0
            out[int(f[0])] = (f[1], ms, f[3] if len(f) > 3 else "?", f[4] if len(f) > 4 else "-")
    return out


def read_csv(path):
    rows = collections.defaultdict(list)
    if not os.path.isfile(path):
        return rows
    for r in csv.DictReader(open(path, encoding="utf-8", errors="ignore")):
        try:
            rows[int(r["N"])].append(r)
        except (KeyError, ValueError):
            pass
    return rows


def read_prime_engines(store):
    p = os.path.join(store, "wisdom2_prime.txt")
    if not os.path.isfile(p):
        return {}
    t = io.open(p, encoding="utf-8", errors="ignore").read()
    return {int(m.group(1)): m.group(2) for m in re.finditer(r"@cell t=c2c n=(\d+) q=1 [^|\n]*\|[^\n]*?eng=(\w+)", t)}


def build(run_dir, sfx=""):
    rows = read_csv(os.path.join(run_dir, "gauntlet%s.csv" % sfx))
    cal = read_calibrate(os.path.join(run_dir, "calibrate%s.log" % sfx))
    ctl = [float(r["ratio_vs_mkl"]) for r in csv.DictReader(open(os.path.join(run_dir, "control%s.csv" % sfx), encoding="utf-8", errors="ignore"))] \
        if os.path.isfile(os.path.join(run_dir, "control%s.csv" % sfx)) else []
    eng = read_prime_engines(os.path.join(run_dir, "store"))
    has_cmp = any(float(r.get("mkl_ns", 0) or 0) > 0 for rs in rows.values() for r in rs)
    cells = {}
    for n, rs in rows.items():
        v = [int(r["vfft_ns"]) for r in rs]
        rat = [float(r["ratio_vs_mkl"]) for r in rs] if has_cmp else []
        cells[n] = dict(route=rs[0]["route"], best=min(v), worst=max(v), nflips=len(rs),
                        lo=min(rat) if rat else 0.0, hi=max(rat) if rat else 0.0,
                        mkl=statistics.median(int(r["mkl_ns"]) for r in rs) if has_cmp else 0,
                        rt=max(float(r.get("rt_err", 0) or 0) for r in rs),
                        gflops=max(float(r.get("vfft_gflops", 0) or 0) for r in rs),
                        engaged=rs[0].get("engaged", ""))
    out = []
    W = out.append
    ns = sorted(set(cells) | set(cal))
    W("# gauntlet report\n")
    W("run: `%s`  contract file suffix: `%s`  cells: %d listed, %d benched, comparator: %s\n" % (
        os.path.basename(os.path.abspath(run_dir)), sfx or "(oop, T=1)", len(ns), len(cells), "MKL" if has_cmp else "none (absolute numbers)"))
    if ctl:
        W("control cell: %d readings, %.3f..%.3f (a run is internally comparable when the first and the last agree)\n" % (len(ctl), min(ctl), max(ctl)))
    # the per-cell table
    W("\n## every cell\n")
    W("```")
    W(" %9s  %-16s %-8s %-9s %10s %10s %7s %8s  %s" % ("N", "factors", "route", "served", "ours ns", "cmp ns", "x", "rt err", "note"))
    for n in ns:
        c = cells.get(n)
        st = cal.get(n, ("-", 0, "-", "-"))
        served = st[2] if st[0] == "banked" else st[0].lower()
        route = c["route"] if c else st[3]
        note = ""
        if c and c["nflips"] == 2 and c["worst"] / max(1, c["best"]) > 1.25:
            note = "flips differ %.2fx" % (c["worst"] / c["best"])
        if route == "prime" and n in eng:
            note = (note + " " if note else "") + eng[n]
        if c:
            W(" %9d  %-16s %-8s %-9s %10d %10s %7s %8.1e  %s" % (
                n, ".".join(map(str, fac(n)))[:16], route, served, c["best"],
                ("%d" % c["mkl"]) if has_cmp else "-", ("%.2f" % c["lo"]) if has_cmp else "-", c["rt"], note))
        else:
            W(" %9d  %-16s %-8s %-9s %10s %10s %7s %8s  %s" % (n, ".".join(map(str, fac(n)))[:16], route, served, "-", "-", "-", "-", "not benched"))
    W("```\n")
    if not has_cmp or not cells:
        W("\n(no comparator: the ratio tables need MKL; see the ns and GFLOPS columns)\n")
        return "\n".join(out)
    # routes
    W("\n## by route (worse of the two flips)\n```")
    W(" %-8s %5s %6s %6s %6s %6s %6s %7s" % ("route", "cells", "<0.8", "<1.0", "p10", "med", "p90", "gmean"))
    for rt in sorted(set(c["route"] for c in cells.values()), key=lambda r: -sum(1 for c in cells.values() if c["route"] == r)):
        xs = [c["lo"] for c in cells.values() if c["route"] == rt]
        W(" %-8s %5d %6d %6d %6.2f %6.2f %6.2f %7.2f" % (rt, len(xs), sum(1 for x in xs if x < 0.8), sum(1 for x in xs if x < 1), q(xs, .1), statistics.median(xs), q(xs, .9), gm(xs)))
    xs = [c["lo"] for c in cells.values()]
    W(" %-8s %5d %6d %6d %6.2f %6.2f %6.2f %7.2f" % ("ALL", len(xs), sum(1 for x in xs if x < 0.8), sum(1 for x in xs if x < 1), q(xs, .1), statistics.median(xs), q(xs, .9), gm(xs)))
    W("```\n")
    # size bands (log2 decades)
    W("\n## by size\n```")
    W(" %-18s %5s %6s %6s %6s" % ("band", "cells", "median", "<1.0", "<0.8"))
    lo = 2
    while lo <= max(cells):
        hi = lo * 4 - 1
        xs = [c["lo"] for n, c in cells.items() if lo <= n <= hi]
        if xs:
            W(" %-18s %5d %6.2f %6d %6d" % ("%d..%d" % (lo, min(hi, max(cells))), len(xs), statistics.median(xs), sum(1 for x in xs if x < 1), sum(1 for x in xs if x < 0.8)))
        lo *= 4
    W("```\n")
    # families
    def fam(n):
        c = cells[n]; f = fac(n); p = max(f)
        if n & (n - 1) == 0:
            return "pow2"
        if c["route"] == "prime":
            if isprime(n):
                return "prime N, %s" % eng.get(n, "prime cell")
            return "composite with a prime >= 53 (prime cell)" if p >= 53 else "composite, prime cell by race"
        return c["route"]
    g = collections.defaultdict(list)
    for n in cells:
        g[fam(n)].append(cells[n]["lo"])
    W("\n## by family\n```")
    W(" %-44s %5s %6s %6s %6s" % ("family", "cells", "median", "<1.0", "gmean"))
    for k, xs in sorted(g.items(), key=lambda kv: -len(kv[1])):
        W(" %-44s %5d %6.2f %6d %6.2f" % (k, len(xs), statistics.median(xs), sum(1 for x in xs if x < 1), gm(xs)))
    W("```\n")
    dis = sum(1 for c in cells.values() if c["nflips"] == 2 and c["worst"] / max(1, c["best"]) > 1.25)
    W("\nflip agreement: our two readings more than 25%% apart at %d of %d cells." % (dis, len(cells)))
    worst = sorted(cells, key=lambda n: cells[n]["lo"])[:10]
    W("\nworst 10: " + ", ".join("%d (%s %.2f)" % (n, cells[n]["route"], cells[n]["lo"]) for n in worst))
    best = sorted(cells, key=lambda n: -cells[n]["lo"])[:5]
    W("best 5: " + ", ".join("%d (%s %.2f)" % (n, cells[n]["route"], cells[n]["lo"]) for n in best) + "\n")
    return "\n".join(out)


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    run_dir = argv[1]
    sfx = ""
    if "--contract" in argv and argv[argv.index("--contract") + 1] == "ip":
        sfx += "_ip"
    if "--threads" in argv and int(argv[argv.index("--threads") + 1]) > 1:
        sfx += "_mt%s" % argv[argv.index("--threads") + 1]
    text = build(run_dir, sfx)
    io.open(os.path.join(run_dir, "report%s.md" % sfx), "w", encoding="utf-8").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
