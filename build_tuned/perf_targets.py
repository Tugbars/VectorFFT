#!/usr/bin/env python3
"""perf_targets.py - derive the performance corpus for the refactor from BANKED data.

THE IDEA
--------
A refactor's perf check should not use hand-picked cells or invented targets. Every
cell the library has calibrated is already recorded, WITH the plan that produced it
and the latency it produced:

  wisdom store  @cell t=c2c n=1024 q=4 ord=scr place=ip | chain=64.16 ... ns=3955.33
  results CSV   N=1024,K=4,plan=64x16/DIT,vfft_ns=3318

So the corpus is a JOIN of the two, keyed by (N, K). The store supplies the plan and
a banked latency; the CSV supplies an independently measured latency for the same
cell. The plans agree exactly across the join - 64x16/DIT is chain=64.16 dif=0.

WHY THE JOIN AND NOT EITHER SOURCE ALONE
----------------------------------------
Because the two saved latencies frequently DISAGREE. Measured over the 190 cells
present in both:

    132 agree within 10%          <- usable as a target
     58 disagree by more than 10% <- NOT usable, up to 2.94x apart

and in every disagreeing case THE PLAN IS IDENTICAL. Same cell, same plan, two
recorded latencies up to 3x apart. That is not drift in what the library builds; it
is measurement regime - the project's own rule that cross-session numbers are not
comparable, quantified.

The consequence is the whole point of this tool: a "within 10% of the saved time"
check is only meaningful where two independent sources CORROBORATE the target. For
the other 58 cells the band is undefined, because the two saved times differ from
each other by more than the band.

WHAT A TARGET HERE IS AND IS NOT
--------------------------------
IS:     a regression detector. Thermal noise is one-sided - nothing makes a
        deterministic computation faster than its best case - so the MINIMUM over N
        paced runs is a robust estimator, and a 2x regression cannot hide under a
        10% band.
IS NOT: a proof of no regression. It cannot see 3-5% drift, and it is not a re-race:
        no verdict is banked from it.

STRONGER STILL: prefer A/B against the ARCHIVED pre-refactor binary re-run in the
same session. That removes machine-state drift entirely instead of absorbing it in
the tolerance, and it needs no corroboration filter at all. These targets are the
fallback for when no archived binary exists, and the sanity anchor.

USAGE
  python build_tuned/perf_targets.py [--band 0.10] [--out FILE]
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.normpath(os.path.join(
    HERE, "..", "src", "dag-fft-compiler", "generator", "generated",
    "wisdom2_stride.txt"))
CSVP = os.path.join(HERE, "results", "vfft_perf_tuned_1d.csv")

CELL = re.compile(r"^@cell t=c2c .*?\bn=(\d+)\b.*?\bq=(\d+)\b")


def load_store():
    """(N,K) -> (ns, chain, dif) for scrambled in-place c2c cells."""
    out = {}
    if not os.path.exists(STORE):
        return out
    for line in open(STORE, encoding="utf-8", errors="replace"):
        m = CELL.match(line)
        if not m:
            continue
        if "ord=scr" not in line or "place=ip" not in line:
            continue
        ns = re.search(r"\bns=([\d.]+)", line)
        ch = re.search(r"\bchain=([\d.]+)", line)
        di = re.search(r"\bdif=(\d)", line)
        if not ns:
            continue
        out[(int(m.group(1)), int(m.group(2)))] = (
            float(ns.group(1)), ch.group(1) if ch else "?",
            di.group(1) if di else "?")
    return out


def plans_match(csv_plan, chain, dif):
    """'4x4x8x32/DIF' vs chain='4.4.8.32' dif='1'."""
    if "/" not in csv_plan:
        return csv_plan.replace("x", ".") == chain
    shape, order = csv_plan.rsplit("/", 1)
    want = "1" if order.upper() == "DIF" else "0"
    return shape.replace("x", ".") == chain and dif == want


def main():
    band = 0.10
    out_path = None
    if "--band" in sys.argv:
        band = float(sys.argv[sys.argv.index("--band") + 1])
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]

    store = load_store()
    if not os.path.exists(CSVP):
        print("missing %s" % CSVP, file=sys.stderr)
        return 2
    rows = list(csv.DictReader(open(CSVP)))

    good, bad, plan_drift, only_csv = [], [], [], 0
    for r in rows:
        k = (int(r["N"]), int(r["K"]))
        if k not in store:
            only_csv += 1
            continue
        st_ns, chain, dif = store[k]
        csv_ns = float(r["vfft_ns"])
        ratio = max(csv_ns, st_ns) / max(1e-9, min(csv_ns, st_ns))
        rec = (k[0], k[1], csv_ns, st_ns, ratio, r["plan"], chain, dif,
               float(r["rt_err"]))
        if not plans_match(r["plan"], chain, dif):
            plan_drift.append(rec)
        elif ratio <= 1.0 + band:
            good.append(rec)
        else:
            bad.append(rec)

    lines = []
    add = lines.append
    add("# performance corpus for the refactor, derived from BANKED data.")
    add("# target_ns = min(csv, store); a cell qualifies only when the two")
    add("# independent saved latencies corroborate within %.0f%%." % (band * 100))
    add("# Method: N>=10 paced runs, take the MINIMUM, pass within %.0f%%."
        % (band * 100))
    add("# Noise is one-sided, so the minimum is the robust estimator.")
    add("#")
    add("# N        K     target_ns    chain                dif  rt_err")
    for (n, kk, c, s, rt, cp, ch, di, err) in sorted(good):
        add("%-9d %-5d %-12.0f %-20s %-4s %.3e" % (n, kk, min(c, s), ch, di, err))
    add("#")
    add("# EXCLUDED - the two saved latencies disagree, so no trustworthy target.")
    add("# Same cell, same plan, different measurement regime.")
    for (n, kk, c, s, rt, cp, ch, di, err) in sorted(bad, key=lambda x: -x[4]):
        add("# excl N=%-8d K=%-5d csv=%-11.0f store=%-11.0f %.2fx  %s"
            % (n, kk, c, s, rt, ch))
    if plan_drift:
        add("#")
        add("# PLAN DRIFT - the CSV plan no longer matches the banked chain.")
        for (n, kk, c, s, rt, cp, ch, di, err) in sorted(plan_drift):
            add("# drift N=%-8d K=%-5d csv_plan=%-18s store=%s dif=%s"
                % (n, kk, cp, ch, di))
    add("#")
    add("# usable=%d excluded=%d plan_drift=%d csv_only=%d store_cells=%d"
        % (len(good), len(bad), len(plan_drift), only_csv, len(store)))

    text = "\n".join(lines) + "\n"
    if out_path:
        open(out_path, "w", encoding="utf-8", newline="\n").write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
