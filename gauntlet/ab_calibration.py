#!/usr/bin/env python3
"""ab_calibration.py -- what does host calibration actually buy?

The experiment (2026-10-02, Zen 4): run the same dense cell range twice.

  phase A   store seeded from src/wisdom (the i9-14900KF verdicts), NO --calibrate
            -> every covered cell REPLAYS the Intel plan on this host
  phase B   same range, --calibrate
            -> every cell re-raced at VFFT_PATIENT on this host

The A->B delta on our own time is the value of calibration. FFTW is identical in
both runs, so its column is the drift control: if fftw_B/fftw_A strays from 1.0
the machine moved between the runs and the comparison is weakened.

Only cells phase A's calibrate log marks `replayed` are admissible -- a cell the
Intel store did not cover was RACED in phase A too (store miss races, by law),
so for those A and B are both Zen 4 plans and the delta is noise.

  python gauntlet/ab_calibration.py --a <runA> --b <runB> [--sfx _fftw]
"""
import argparse, collections, csv, io, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def served_map(run, sfx):
    """cell -> served ('replayed' | 'raced' | 'refused') from the calibrate log"""
    out = {}
    for name in ("calibrate%s.log" % sfx, "calibrate.log"):
        p = os.path.join(RESULTS, run, name)
        if not os.path.isfile(p):
            continue
        for line in io.open(p, encoding="utf-8", errors="ignore"):
            f = line.split()
            if len(f) >= 4 and f[0].isdigit():
                out[int(f[0])] = f[3]
        break
    return out


def best(run, sfx):
    """cell -> (ours_ns, cmp_ns, route) taking each engine's faster order"""
    p = os.path.join(RESULTS, run, "gauntlet%s.csv" % sfx)
    if not os.path.isfile(p):
        sys.exit("no csv at %s" % p)
    acc = collections.defaultdict(lambda: [[], [], None])
    for r in csv.DictReader(open(p, encoding="utf-8", errors="ignore")):
        try:
            n = int(r["N"])
            v, c = float(r["vfft_ns"]), float(r["mkl_ns"])
        except (KeyError, ValueError):
            continue
        if v > 0:
            acc[n][0].append(v)
        if c > 0:
            acc[n][1].append(c)
        acc[n][2] = r.get("route") or acc[n][2]
    return {n: (min(a), min(b) if b else 0.0, rt)
            for n, (a, b, rt) in acc.items() if a}


def stats(xs):
    if not xs:
        return None
    g = 1.0
    for x in xs:
        g *= x
    return dict(n=len(xs), med=statistics.median(xs), gm=g ** (1.0 / len(xs)),
                lo=min(xs), hi=max(xs))


def line(label, s):
    if not s:
        print(" %-26s      (no cells)" % label)
        return
    print(" %-26s %5d   %6.3f  %6.3f   %6.2f  %6.2f"
          % (label, s["n"], s["med"], s["gm"], s["lo"], s["hi"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="phase A run dir (Intel wisdom, replayed)")
    ap.add_argument("--b", required=True, help="phase B run dir (recalibrated on this host)")
    ap.add_argument("--sfx", default="_fftw", help="contract csv suffix (default _fftw)")
    ap.add_argument("--csv", help="write the per-cell join here")
    g = ap.parse_args()

    A, B = best(g.a, g.sfx), best(g.b, g.sfx)
    srv = served_map(g.a, g.sfx)
    common = sorted(set(A) & set(B))
    if not common:
        sys.exit("no cells in common")

    rows = []
    for n in common:
        va, ca, ra = A[n]
        vb, cb, rb = B[n]
        rows.append(dict(N=n, served=srv.get(n, "?"),
                         ours_a=va, ours_b=vb, fftw_a=ca, fftw_b=cb,
                         route_a=ra, route_b=rb,
                         gain=va / vb if vb else 0.0,          # >1 = calibration made us faster
                         drift=cb / ca if ca else 0.0,         # FFTW control, want ~1.0
                         x_a=ca / va if va and ca else 0.0,
                         x_b=cb / vb if vb and cb else 0.0))

    rep = [r for r in rows if r["served"] == "replayed"]
    raced_in_a = [r for r in rows if r["served"] == "raced"]

    print("A = %s   (Intel wisdom, replayed)" % g.a)
    print("B = %s   (recalibrated on this host)" % g.b)
    print("cells in both runs: %d   | phase A replayed %d, raced %d, other %d\n"
          % (len(rows), len(rep), len(raced_in_a), len(rows) - len(rep) - len(raced_in_a)))

    print(" %-26s %5s   %6s  %6s   %6s  %6s" % ("", "cells", "median", "gmean", "min", "max"))
    print(" " + "-" * 69)
    print(" THE RESULT -- ours_A / ours_B  (>1 means calibration made us faster)")
    line("replayed cells", stats([r["gain"] for r in rep]))
    line("  N <= 1024", stats([r["gain"] for r in rep if r["N"] <= 1024]))
    line("  N > 1024", stats([r["gain"] for r in rep if r["N"] > 1024]))
    print()
    print(" CONTROL -- fftw_B / fftw_A  (want ~1.00; far from it = machine drifted)")
    line("replayed cells", stats([r["drift"] for r in rep if r["drift"]]))
    print()
    print(" vs FFTW, same cells, each phase")
    line("phase A  (Intel plans)", stats([r["x_a"] for r in rep if r["x_a"]]))
    line("phase B  (Zen 4 plans)", stats([r["x_b"] for r in rep if r["x_b"]]))
    print()
    print(" SANITY -- cells phase A also raced (expect gain ~1.0, it is noise)")
    line("raced in both", stats([r["gain"] for r in raced_in_a]))

    if rep:
        faster = sum(1 for r in rep if r["gain"] > 1.02)
        same = sum(1 for r in rep if 0.98 <= r["gain"] <= 1.02)
        slower = sum(1 for r in rep if r["gain"] < 0.98)
        print("\n calibration verdict over %d replayed cells:" % len(rep))
        print("   faster by >2%%   %5d  (%4.1f%%)" % (faster, 100.0 * faster / len(rep)))
        print("   within +-2%%     %5d  (%4.1f%%)" % (same, 100.0 * same / len(rep)))
        print("   slower by >2%%   %5d  (%4.1f%%)" % (slower, 100.0 * slower / len(rep)))

        chg = [r for r in rep if r["route_a"] and r["route_b"] and r["route_a"] != r["route_b"]]
        print("\n route CHANGED by calibration on %d of %d replayed cells (%.1f%%)"
              % (len(chg), len(rep), 100.0 * len(chg) / len(rep)))
        if chg:
            byc = collections.Counter("%s -> %s" % (r["route_a"], r["route_b"]) for r in chg)
            print("   %-18s %6s  %8s" % ("transition", "cells", "med gain"))
            for k, c in byc.most_common(12):
                gains = [r["gain"] for r in chg if "%s -> %s" % (r["route_a"], r["route_b"]) == k]
                print("   %-18s %6d  %7.3f" % (k, c, statistics.median(gains)))

        print("\n biggest 12 wins from calibration")
        print("   %-8s %-7s %-7s %10s %10s %8s" % ("N", "A route", "B route", "ours A", "ours B", "gain"))
        for r in sorted(rep, key=lambda r: -r["gain"])[:12]:
            print("   %-8d %-7s %-7s %10.0f %10.0f %7.2fx"
                  % (r["N"], r["route_a"], r["route_b"], r["ours_a"], r["ours_b"], r["gain"]))

        print("\n worst 12 (calibration lost ground -- re-race noise or a bad verdict)")
        for r in sorted(rep, key=lambda r: r["gain"])[:12]:
            print("   %-8d %-7s %-7s %10.0f %10.0f %7.2fx"
                  % (r["N"], r["route_a"], r["route_b"], r["ours_a"], r["ours_b"], r["gain"]))

    if g.csv:
        with io.open(g.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nper-cell join -> %s" % g.csv)


if __name__ == "__main__":
    main()
