#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_cells.py -- writes the regression test's three cell files beside this script (2026-11-20):

    octaves.txt      the 1D lengths
    2d-octaves.txt   the 2D shapes N1xN2 (N1 = the column length)
    3d-octaves.txt   the 3D shapes N1xN2xN3

    python tests/regression/make_cells.py

The cells come from arithmetic alone. No cache size, planner choice or measurement goes in, so every platform
runs the same cells and compares them as ratios on itself (README.md).

The classes of a length:
  - a power of two;
  - 2^a x odd, every prime factor at most 47;
  - odd, every prime factor at most 47;
  - a prime above 47;
  - a composite with a prime factor above 47.

1D: two cells per class per octave [2^k, 2^(k+1)), 2^3 to 2^24 points. A power of two has one cell, 2^k.
  - The beginning: the first member from 2^k whose factors are all at most 7, within the first eighth of the
    octave. Failing that, the first member in that eighth, then the first member of the octave. For the last two
    classes, simply the first member.
  - The end: the last member below 2^(k+1).
  - No odd part twice. A cell skips any length whose odd part a lower cell of its class already took, so
    doubling a length never repeats it.
  - In even octaves, the 2^a x odd beginning prefers 2^4 or more with an odd part of 3, 5 and 7 only (ZTURN-T
    odd's band). It searches the first eighth of the octave, then the first quarter.
2D: N1 is the beginning cell of its class, and N2 is the end cell of its class (2^b for a power of two).
  - Every pair of classes, at planes of about 2^8, 2^12, 2^16, 2^20 and 2^23 points.
  - Each class with itself, also tall (16:1) and wide (1:16).
3D: axis 0 is the beginning cell of its class. The plane is the beginning cell by the end cell of its class,
  which is axis 0's class or a power of two.
  - Volumes of about 2^9, 2^12, 2^15, 2^18, 2^21 and 2^23 points.
  - Each class with itself, also with axis 0 about three octaves shorter and longer than the plane's axes.
No cell has more than 2^24 points.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = 1 << 24
CLASSES = ("P", "E", "O", "R", "B")
NAME = {"P": "power of two", "E": "2^a x odd", "O": "odd", "R": "prime > 47", "B": "prime factor > 47"}
SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)


def is_prime(n):
    """Miller-Rabin, deterministic below 2^32 with these bases"""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def smooth(n, b):
    """every prime factor of n is at most b"""
    for p in SMALL:
        if p > b:
            break
        while n % p == 0:
            n //= p
    return n == 1


def factors(n):
    """'2^5*3*7^2'"""
    f, p = [], 2
    while p * p <= n:
        while n % p == 0:
            f.append(p)
            n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        f.append(n)
    return "*".join(("%d" % p) if f.count(p) == 1 else "%d^%d" % (p, f.count(p)) for p in sorted(set(f)))


def cls(n):
    if n & (n - 1) == 0:
        return "P"
    if is_prime(n):
        return "R" if n > 47 else ("O" if n % 2 else "E")
    if smooth(n, 47):
        return "E" if n % 2 == 0 else "O"
    return "B"


def odd_part(n):
    while n % 2 == 0:
        n //= 2
    return n


def scan(lo, hi, step, test):
    n = lo
    while (n <= hi) if step > 0 else (n >= hi):
        if test(n):
            return n
        n += step
    return None


def pick(c, k, end, used):
    """the cell of class c at the beginning or the end of octave k; used: the odd parts its lower cells took"""
    a, b = 1 << k, (1 << (k + 1)) - 1
    fresh = lambda n: n >= 2 and cls(n) == c and odd_part(n) not in used
    if end == "end":
        return scan(b, a, -1, fresh)
    if c in ("R", "B"):
        return scan(a, b, 1, fresh)
    eighth, quarter = min(a + max(a // 8, 1), b), min(a + max(a // 4, 1), b)
    tests = [(eighth, lambda n: fresh(n) and smooth(n, 7)), (eighth, fresh)]
    if c == "E" and k % 2 == 0:   # ZTURN-T odd's band: 2^4 or more and an odd part of 3, 5 and 7 only
        band = lambda n: fresh(n) and smooth(n, 7) and n % 16 == 0
        tests = [(eighth, band), (quarter, band)] + tests
    for hi, test in tests:
        n = scan(a, hi, 1, test)
        if n:
            return n
    return scan(a, b, 1, fresh)


def ladder():
    """{class: {(k, 'beginning' | 'end'): length}}, octaves 2^1 .. 2^23 (and 2^24 itself)"""
    lad = {c: {} for c in CLASSES}
    used = {c: set() for c in CLASSES}
    for k in range(1, 24):
        for c in CLASSES:
            if c == "P":
                lad[c][(k, "beginning")] = 1 << k
                continue
            for end in ("beginning", "end"):
                n = pick(c, k, end, used[c])
                if n and n < TOP:
                    lad[c][(k, end)] = n
                    used[c].add(odd_part(n))
    lad["P"][(24, "beginning")] = TOP
    return lad


def cells_1d(lad):
    return [(n, c, k, end) for c in CLASSES for k in range(3, 25) for end in ("beginning", "end")
            for n in [lad[c].get((k, end))] if n]


def cells_2d(lad):
    out = []

    def add(K, c1, c2, skew, shape):
        s = K - (0 if c2 == "P" else 1)    # an end cell sits near 2^(b+1): one octave fewer to reach 2^K
        a = (s + skew) // 2
        b = s - a
        n1 = lad[c1].get((a, "beginning"))
        n2 = lad[c2].get((b, "end")) or lad[c2].get((b, "beginning"))
        if n1 and n2 and n1 * n2 <= TOP:
            out.append(((n1, n2), c1, c2, (n1 * n2).bit_length() - 1, shape))

    for K in (8, 12, 16, 20, 23):
        for c1 in CLASSES:
            for c2 in CLASSES:
                add(K, c1, c2, 0, "square")
        for c in CLASSES:
            add(K, c, c, 4, "tall 16:1")
            add(K, c, c, -4, "wide 1:16")
    return sorted(out, key=lambda x: (x[3], CLASSES.index(x[1]), CLASSES.index(x[2]), x[4]))


def cells_3d(lad):
    out = []

    def add(K, c0, cp, skew, shape):
        s = K - (0 if cp == "P" else 1)
        b = (s - skew) // 3
        a = s - 2 * b
        n1 = lad[c0].get((a, "beginning"))
        n2 = lad[cp].get((b, "beginning"))
        n3 = lad[cp].get((b, "end")) or lad[cp].get((b, "beginning"))
        if n1 and n2 and n3 and n1 * n2 * n3 <= TOP:
            out.append(((n1, n2, n3), c0, cp, (n1 * n2 * n3).bit_length() - 1, shape))

    for K in (9, 12, 15, 18, 21, 23):
        for c0 in CLASSES:
            for cp in ((c0,) if c0 == "P" else (c0, "P")):
                add(K, c0, cp, 0, "cube")
            add(K, c0, c0, -3, "short axis 0")
            add(K, c0, c0, 3, "long axis 0")
    return sorted(out, key=lambda x: (x[3], CLASSES.index(x[1]), CLASSES.index(x[2]), x[4]))


def write(name, what, cols, rows):
    head = ("# %s -- %s, the regression test's cells (README.md beside this file)\n"
            "# Written by make_cells.py from arithmetic alone; every platform runs the same cells.\n"
            "# The gauntlet reads it: python gauntlet/gauntlet.py run --cells @tests/regression/%s\n"
            "%s\n") % (name, what, name, cols)
    io.open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n").write(head + "".join(rows))
    print("%-15s %3d cells" % (name, len(rows)))


def main():
    lad = ladder()
    c1, c2, c3 = cells_1d(lad), cells_2d(lad), cells_3d(lad)
    for cells in (c1, c2, c3):
        keys = [c[0] for c in cells]
        assert len(keys) == len(set(keys)), "a cell twice"
    write("octaves.txt", "%d 1D lengths" % len(c1),
          "%-11s # %-18s %-7s %-10s %s" % ("# length", "class", "octave", "end", "factors"),
          ["%-11d # %-18s 2^%-5d %-10s %s\n" % (n, NAME[c], k, end, factors(n)) for n, c, k, end in c1])
    write("2d-octaves.txt", "%d 2D shapes N1xN2" % len(c2),
          "%-17s # %-18s %-18s %-6s %-10s %s" % ("# N1xN2", "N1 class", "N2 class", "plane", "shape", "factors"),
          ["%-17s # %-18s %-18s 2^%-4d %-10s %s x %s\n" % ("%dx%d" % n, NAME[a], NAME[b], K, shape, factors(n[0]), factors(n[1]))
           for n, a, b, K, shape in c2])
    write("3d-octaves.txt", "%d 3D shapes N1xN2xN3" % len(c3),
          "%-21s # %-18s %-18s %-6s %-12s %s" % ("# N1xN2xN3", "axis-0 class", "plane class", "volume", "shape", "factors"),
          ["%-21s # %-18s %-18s 2^%-4d %-12s %s x %s x %s\n" % ("%dx%dx%d" % n, NAME[a], NAME[b], K, shape, factors(n[0]), factors(n[1]), factors(n[2]))
           for n, a, b, K, shape in c3])


if __name__ == "__main__":
    main()
