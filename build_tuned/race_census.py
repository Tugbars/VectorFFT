#!/usr/bin/env python3
"""race_census.py - a STATIC census of every measurement protocol in the source.

WHY THIS EXISTS
---------------
build_tuned/obj_equiv.py proves a code move changed no emitted instruction. It is
MEASURED BLIND to data: floating-point constants live in .rdata, and an injected
0.97 -> 0.96 hysteresis change passed it as EQUIVALENT. A race's hysteresis, round
count, reps ladder and tie-holder are exactly what select the winner that gets
BANKED, so that blindness sits on top of the most expensive failure this refactor
can produce: a verdict that changes silently and then persists forever.

This tool covers that gap by reading the SOURCE, not the object.

WHY STATIC AND NOT A RUNTIME TRACE
----------------------------------
A trace only records branches that execute. _calibrate_zsplit_t2q contains

    if (inc == 0) win = (n1 < n0 * 0.97); else win = (n0 < n1 * 0.97);

and on the only path that races nothing sets `inc` to 1, so the else arm never runs
and a trace can never see its constant. A source census covers branches that never
execute, and needs no wisdom state, no thread count and no clock.

ENUMERATION - TWO PASSES, BOTH REQUIRED
---------------------------------------
Pass 1: functions containing a clock call. Three spellings are in use and all three
        must be recognised - vfft_proto_now_ns (QPC on Windows), _il_ab_now
        (clock_gettime) and clock_gettime directly.
Pass 2: functions calling a timing HELPER but holding no clock of their own. Not
        optional: _calibrate_pad is a 125-line racer with ZERO clock calls - it
        delegates to _pad_burst - so a clock-only enumerator silently drops it.

REGIONS, NOT FUNCTIONS
----------------------
Features are extracted per RACE REGION, not per function. Ten of the race sites are
anonymous inline blocks inside _vfft_create_inner (measured: its clock calls cluster
at 10 distinct places across 4,006 lines). Keying by function would collapse all ten
onto one key and destroy exactly the per-site independence this census exists to
check. Clock calls are clustered with a GAP threshold; each cluster extends forward
to pick up the verdict comparison that follows it.

KEYING
------
A site is keyed by a hash of its normalized features - never by file/line (which
churn on every move) and never by function name. The enclosing function travels as
an ATTRIBUTE, so a MOVE shows up as an fn change on a stable key rather than as a
delete plus an add.

WHAT IT DOES NOT DO
-------------------
It reports the protocol, not the verdict. It cannot tell you whether a changed
protocol still picks the same arm - nothing can, on this machine, without re-racing.
That is why protocol unification is out of scope for the migration.

USAGE
  python build_tuned/race_census.py [src/core/vfft.c ...]
Writes a sorted, diffable census to stdout.
"""
import hashlib
import re
import os
import sys

CLOCKS = ("vfft_proto_now_ns(", "_il_ab_now(", "clock_gettime(")
HELPERS = ("_pad_burst(", "_pad_med(", "_il_ab_med9(")
# timer and aggregation DEFINITIONS are not races; _bank_zr2c only records a number
NOT_A_RACE = {"_il_ab_now", "vfft_proto_now_ns", "_pad_med", "_il_ab_med9",
              "_bank_zr2c"}
GAP = 40          # clock calls further apart than this start a new region
LOOKAHEAD = 30    # lines past the last clock call to search for the verdict

FN_DEF = re.compile(r"^[A-Za-z_]")
FN_NAME = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*\(")
# a verdict with hysteresis: n1 < n0 * 0.97
HYST = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*(<=|<|>=|>)\s*"
                  r"([A-Za-z_][A-Za-z_0-9]*)\s*\*\s*(\d*\.\d+)")
# a bare verdict with no hysteresis at all: h->pq_mt = (tq < tl);
BARE = re.compile(r"=\s*\(?([A-Za-z_][A-Za-z_0-9]*)\s*(<=|<|>=|>)\s*"
                  r"([A-Za-z_][A-Za-z_0-9]*)\)?\s*;")
ROUND = re.compile(r"for\s*\(\s*(?:int\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*0\s*;\s*"
                   r"\1\s*<\s*(\w+)\s*;")
REPS = re.compile(r"\breps\s*=\s*(.+?);")
MINMAX = re.compile(r"if\s*\(\s*\w+\s*<\s*\w+\s*\)")


def functions(lines):
    """-> [(name, start, end)] for top-level definitions (column 0, not a decl)."""
    # A function ENDS AT ITS CLOSING BRACE, not at the next function's start.
    #
    # The original heuristic ran each extent up to the line before the next
    # definition, which silently attributed every comment, forward declaration
    # and #include BETWEEN two functions to the earlier one. That was invisible
    # while the file was densely packed, and became visible the moment the
    # migration opened gaps: _calibrate_pad started reporting an aggregator
    # (_il_ab_med9) it does not use, picked up from a forward declaration two
    # blocks below it. A census that attributes features to the wrong function
    # is worse than a coarse one - it reads as a real finding.
    #
    # This tree closes every top-level definition with a brace in column 0, so
    # that is the terminator. A function whose brace is never found falls back
    # to end-of-file, which is the old behaviour and only reachable on
    # malformed input.
    out = []
    for i, line in enumerate(lines):
        if FN_DEF.match(line) and "(" in line:
            m = FN_NAME.search(line)
            if not m:
                continue
            # DEFINITION vs DECLARATION: a definition reaches '{' before ';'.
            # Testing only whether the FIRST line ends in ';' is not enough -
            # this tree wraps long parameter lists, so a two-line declaration
            # looked like a definition, and the brace scan below then ran
            # forward into the NEXT real function and attributed its timing to
            # the declaration. Scan until one of the two characters decides it.
            end, kind = None, None
            for j in range(i, len(lines)):
                if kind is None:
                    if ";" in lines[j] and "{" not in lines[j]:
                        kind = "decl"
                        break
                    if "{" in lines[j]:
                        kind = "def"
                if kind == "def" and lines[j] == "}":
                    end = j
                    break
            if kind != "def":
                continue
            out.append((m.group(1), i, end if end is not None else len(lines) - 1))
    return out


def regions(body):
    """Split one function body into race regions. See REGIONS above."""
    hits = [i for i, line in enumerate(body) if any(c in line for c in CLOCKS)]
    if not hits:
        joined = "\n".join(body)
        if any(h in joined for h in HELPERS):
            return [(0, len(body) - 1)]
        return []
    out, start, prev = [], hits[0], hits[0]
    for i in hits[1:]:
        if i - prev > GAP:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return [(a, min(len(body) - 1, b + LOOKAHEAD)) for a, b in out]


def features(name, body):
    """Protocol features of every race region inside one function."""
    sites = []
    if name in NOT_A_RACE:
        return sites
    for (a, b) in regions(body):
        seg = body[a:b + 1]
        text = "\n".join(seg)
        base = dict(
            fn=name,
            clocks=sorted(c.rstrip("(") for c in CLOCKS if c in text),
            rounds=sorted({m.group(2) for m in (ROUND.search(x) for x in seg) if m}),
            reps=sorted({m.group(1).strip()
                         for m in (REPS.search(x) for x in seg) if m}),
            aggs=sorted(h.rstrip("(") for h in HELPERS if h in text),
            alt=("& 1)" in text or "r & 1" in text),
            minmax=bool(MINMAX.search(text)),
        )
        found = False
        for line in seg:
            for m in HYST.finditer(line):
                sites.append(dict(base, kind="hyst", op=m.group(2), lit=m.group(4)))
                found = True
            m = BARE.search(line)
            if m and not HYST.search(line):
                sites.append(dict(base, kind="bare", op=m.group(2), lit="NONE"))
                found = True
        if not found:
            sites.append(dict(base, kind="UNMATCHED", op="?", lit="?"))
    return sites


def key_of(s):
    """Stable key: the normalized protocol, WITHOUT the function name."""
    blob = "|".join([s["kind"], s["op"], s["lit"],
                     ",".join(s["clocks"]), ",".join(s["rounds"]),
                     ",".join(s["reps"]), ",".join(s["aggs"]),
                     str(s["alt"]), str(s["minmax"])])
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def verdicts(lines):
    """Every verdict comparison in the file, with its enclosing function.

    A SEPARATE pass, because timing and verdict are frequently in DIFFERENT
    functions: _r2c_race_arms only returns the two medians and the comparison
    lives in _r2c_route_decide. Associating them would need a call graph; not
    associating them costs nothing, because a hysteresis change shows up as a
    verdict-row change wherever it lives. This pass is what actually closes
    obj_equiv's .rdata blindness - the timing rows are context.
    """
    out, cur = [], "<file scope>"
    for line in lines:
        if FN_DEF.match(line) and "(" in line and not line.rstrip().endswith(";"):
            m = FN_NAME.search(line)
            if m:
                cur = m.group(1)
        for m in HYST.finditer(line):
            out.append(dict(fn=cur, op=m.group(2), lit=m.group(4)))
        m = BARE.search(line)
        if m and not HYST.search(line) and "0." not in line:
            # a bare verdict is only interesting when both operands look like
            # timings; require short numeric-ish names to avoid assignment noise
            a, b = m.group(1), m.group(3)
            if len(a) <= 4 and len(b) <= 4:
                out.append(dict(fn=cur, op=m.group(2), lit="NONE"))
    return out


# The census must FOLLOW THE CODE, not one file.
#
# It originally scanned vfft.c alone, which was correct while every racer lived
# there. The migration moves racers into module headers, and a file-scoped
# scanner would then report a shrinking census while every racer still existed -
# drifting to zero and reading as "nothing races here" exactly when the opposite
# is true. That is the same failure mode as an assert compiled out: it does not
# fail, it stops testing.
#
# So the default set grows with the migration. Each entry is a header a
# migration step moved timing INTO; adding one is part of that step, and the
# census is expected to come back UNCHANGED afterwards - same racers, found in
# their new home. A step that moves a racer and does NOT extend this list will
# show up as a census shrink, which is the intended alarm.
_MIGRATED = [
    "src/core/support/race_timing.h",             # step 5  - the primitives
    "src/core/transforms/real/real_route_race.h", # step 11 - r2c/c2r racers
    "src/core/planning/cascade_calibrate.h",      # step 12 - t2q calibrators
    "src/core/planning/pad_calibrate.h",          # step 13 - pad-vs-tail
    "src/core/transforms/fft2d/il2d_tier.h",      # step 17 - the four il2d racers
    "src/core/transforms/real/zr2c_build.h",      # step 18 - kind-5 route race
    "src/core/oop/k1_commit.h",                   # step 19 - K=1 race-and-bank
    "src/core/oop/zturn_mt.h",                    # step 20 - zt_mt_race
    "src/core/transforms/fft2d/plane_queue.h",    # step 20 - pq_mt_race
    "src/core/transforms/fftnd/fftnd_create.h",   # step 22 - rank-3/4 create (no racer today)
    "src/core/transforms/fft2d/fft2d_create.h",   # step 23 - 2D tier: 2 races + a verdict
    "src/core/oop/c2c_ip_create.h",               # step 24 - c2c in-place tier
    "src/core/oop/c2c_oop_create.h",              # step 25 - c2c out-of-place tier
    "src/core/transforms/real/real_create.h",     # step 26 - r2c/c2r tier
    "src/core/transforms/trig/trig_create.h",     # step 27 - trig tier + builders
]


def main():
    paths = sys.argv[1:] or (["src/core/vfft.c"] +
                             [q for q in _MIGRATED if os.path.exists(q)])
    rows, vrows = [], []
    for p in paths:
        lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
        for name, s, e in functions(lines):
            rows.extend(features(name, lines[s:e + 1]))
        vrows.extend(verdicts(lines))

    print("# race protocol census - the constants obj_equiv.py cannot see")
    print("# key = hash of the normalized protocol; fn is an ATTRIBUTE, so a MOVE")
    print("# shows as an fn change on a stable key, not a delete plus an add.")
    print("# UNMATCHED = a timing region whose verdict shape is not recognised;")
    print("# it is still tracked, and its count must not grow silently.")
    print("#")
    seen = {}
    for s in rows:
        seen.setdefault(key_of(s), []).append(s)
    for k in sorted(seen):
        g = seen[k]
        s = g[0]
        fns = ",".join(sorted({x["fn"] for x in g}))
        print("%s n=%d kind=%s op=%s hyst=%s rounds=%s reps=%s agg=%s "
              "alt=%d minmax=%d clock=%s fn=%s"
              % (k, len(g), s["kind"], s["op"], s["lit"],
                 "/".join(s["rounds"]) or "-",
                 "/".join(s["reps"]) or "-",
                 "/".join(s["aggs"]) or "inline",
                 int(s["alt"]), int(s["minmax"]),
                 "/".join(s["clocks"]) or "-", fns))
    print("#")
    print("# --- VERDICTS: every comparison that picks an arm, wherever it lives ---")
    vseen = {}
    for v in vrows:
        vseen.setdefault((v["op"], v["lit"]), []).append(v["fn"])
    for (op, lit) in sorted(vseen):
        fns = sorted(set(vseen[(op, lit)]))
        print("verdict op=%-2s hyst=%-6s n=%-2d fn=%s"
              % (op, lit, len(vseen[(op, lit)]), ",".join(fns)))

    unmatched = sum(1 for r in rows if r["kind"] == "UNMATCHED")
    print("#")
    print("# timing_regions=%d protocols=%d functions=%d unmatched=%d"
          % (len(rows), len(seen), len({r["fn"] for r in rows}), unmatched))
    print("# verdicts=%d distinct=%d  (hysteresis literals: %s)"
          % (len(vrows), len(vseen),
             ",".join(sorted({v["lit"] for v in vrows}))))


if __name__ == "__main__":
    sys.exit(main())
