# -*- coding: utf-8 -*-
"""gauntlet_check.py -- the merge-time audit of a gauntlet store (2026-09-20).

Every cell the calibration log says was BANKED must have its verdict on disk:
a natural/OOP K=1 row in the interleaved shard, or a row in the prime shard.
A cell that is missing was raced and then lost -- the 2026-09-20 run found
one (515: a silent persist failure) -- and must be re-raced before the store
is merged into the shipped one. Also reports the route split and the
prime-cell classification (prime / no-kernel composite / SUSPICIOUS).

  python gauntlet_check.py <out-dir>        # e.g. results/gauntlet_2026-09-20
  python gauntlet_check.py <out-dir> --missing-list missing.txt   # cells to re-race
  python gauntlet_check.py <out-dir> --place ip      # the in-place run (calibrate_ip.log, place=ip rows)
"""
import io, re, sys, os

out = sys.argv[1]
place = sys.argv[sys.argv.index("--place") + 1] if "--place" in sys.argv else "oop"
store = os.path.join(out, "store")
cal = os.path.join(out, "calibrate%s.log" % ("_ip" if place == "ip" else ""))


def rd(p):
    return io.open(p, encoding="utf-8", errors="ignore").read()


banked = []
for line in io.open(cal, encoding="utf-8"):
    f = line.split()
    if len(f) >= 2 and f[1] == "banked":
        banked.append(int(f[0]))
oop = rd(os.path.join(store, "wisdom2_oop.txt"))
prime = rd(os.path.join(store, "wisdom2_prime.txt"))


def cells(txt, pat):
    return set(int(m) for m in re.findall(r"@cell t=c2c n=(\d+) " + pat, txt))


# the in-place cell's verdict is its OWN kind-3 row, place=ip (raced executed in place, 2026-09-21)
nat = cells(oop, r"q=1 ord=nat place=%s role=comp lay=il \|" % place)
pr = cells(prime, r"q=1 ord=scr place=ip role=comp lay=il \|")
missing = sorted(n for n in banked if n not in nat and n not in pr)
routes = {}
for m in re.finditer(r"@cell t=c2c n=(\d+) q=1 ord=nat place=%s role=comp lay=il \|.*?il_route=(\w+)" % place, oop):
    n = int(m.group(1))
    if n in banked:
        routes[n] = m.group(2)
for n in banked:
    if n in pr and n not in routes:
        routes[n] = "prime"


def fac(n):
    f, d = [], 2
    while d * d <= n:
        while n % d == 0:
            f.append(d); n //= d
        d += 1
    if n > 1:
        f.append(n)
    return f


susp = [n for n in banked if routes.get(n) == "prime" and len(fac(n)) > 1 and max(fac(n)) <= 19]
print("banked cells: %d   with a row: %d   MISSING: %d" % (len(banked), len(banked) - len(missing), len(missing)))
if missing:
    print("  missing (re-race before merging):", " ".join(map(str, missing[:60])), "..." if len(missing) > 60 else "")
hist = {}
for n in banked:
    hist[routes.get(n, "?")] = hist.get(routes.get(n, "?"), 0) + 1
print("route split:", ", ".join("%s=%d" % kv for kv in sorted(hist.items(), key=lambda kv: -kv[1])))
print("prime-cell composites whose every factor is a kernel radix (should have had a route): %d" % len(susp))
if susp:
    print("  ", " ".join("%d(%s)" % (n, ".".join(map(str, fac(n)))) for n in susp[:40]))
if "--missing-list" in sys.argv:
    mp = sys.argv[sys.argv.index("--missing-list") + 1]
    io.open(mp, "w", encoding="utf-8", newline="\n").write("\n".join(map(str, missing)) + "\n")
    print("wrote", mp)
sys.exit(1 if missing else 0)
