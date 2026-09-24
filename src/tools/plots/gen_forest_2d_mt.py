"""VectorFFT 2D c2c pow2 vs MKL: one thread against eight threads, as a compact forest plot.

One row per plane-size class (the README's classes) and the whole grid; in each row the
one-thread record (open marker) and the eight-thread record (filled marker), the marker at
the class median and the whisker from the 10th to the 90th percentile of the cells' speedups
(worse of the two engine orders per cell, the convention of the 2D matrix figure). The
comparator runs at the same thread count as the transform in both records.

usage:  python3 gen_forest_2d_mt.py <one-thread gauntlet_2d.csv> <eight-thread gauntlet_2d_mt8.csv> [output_prefix]
        -> <output_prefix>.svg and <output_prefix>.png  (default prefix: vectorfft-2d-mt-forest)

Needs only matplotlib, which ships the cmr10 fonts."""
import csv, math, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    sys.exit(__doc__)
SRC1, SRC8 = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "vectorfft-2d-mt-forest"

CLASSES = [("up to 256 points", 1, 256), ("257 to 4,096", 257, 4096),
           ("4,097 to 65,536", 4097, 65536), ("65,537 to 4M", 65537, 1 << 22),
           ("all 159 planes", 1, 1 << 22)]


def worse_of_flips(path):
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[(int(r["N1"]), int(r["N2"]))].append(float(r["ratio_vs_mkl"]))
    return {k: min(v) for k, v in by.items()}


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def stats(cells, lo, hi):
    xs = [x for (a, b), x in cells.items() if lo <= a * b <= hi]
    return len(xs), pct(xs, .10), pct(xs, .50), pct(xs, .90)


one, eight = worse_of_flips(SRC1), worse_of_flips(SRC8)
rows = [(label, stats(one, lo, hi), stats(eight, lo, hi)) for label, lo, hi in CLASSES]
for label, s1, s8 in rows:
    print("%-18s  1 thread: n=%d p10 %.2f med %.2f p90 %.2f   8 threads: n=%d p10 %.2f med %.2f p90 %.2f"
          % (label, *s1, *s8))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "svg.fonttype": "path",
})
INK, ONE, EIGHT = "#000000", "#000000", "#1f5fa8"
fig, ax = plt.subplots(figsize=(8.6, 3.5), dpi=150)
ax.set_xscale("log")
ax.set_xlim(0.7, 12)
ticks = [0.8, 1, 1.25, 1.5, 2, 3, 4, 6, 8]
ax.set_xticks(ticks)
ax.set_xticklabels([("%g" % t) for t in ticks], fontsize=9)
ax.tick_params(axis="x", length=3)
ax.axvline(1.0, color=INK, lw=0.8, zorder=1)
nrow = len(rows)
ax.set_ylim(-0.7, nrow - 0.3)
ax.set_yticks(range(nrow))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10)
ax.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="x", color="#d8d8d8", lw=0.5, zorder=0)
for i, (label, (n1, l1, m1, h1), (n8, l8, m8, h8)) in enumerate(rows):
    y = nrow - 1 - i
    ax.plot([l1, h1], [y + 0.17, y + 0.17], color=ONE, lw=1.0, zorder=2)
    ax.plot(m1, y + 0.17, marker="o", ms=6, mfc="white", mec=ONE, mew=1.1, zorder=3)
    ax.plot([l8, h8], [y - 0.17, y - 0.17], color=EIGHT, lw=1.0, zorder=2)
    ax.plot(m8, y - 0.17, marker="o", ms=6, mfc=EIGHT, mec=EIGHT, zorder=3)
    ax.text(12.6, y + 0.17, "%.2f" % m1, va="center", ha="left", fontsize=9, color=ONE, clip_on=False)
    ax.text(12.6, y - 0.17, "%.2f" % m8, va="center", ha="left", fontsize=9, color=EIGHT, clip_on=False)
ax.text(12.6, nrow - 0.55, "median", va="bottom", ha="left", fontsize=9, fontstyle="italic", clip_on=False)
ax.set_xlabel("speedup over MKL, worse of two orders per plane; marker median, whisker 10th to 90th percentile", fontsize=9)
ax.plot([], [], marker="o", ms=6, mfc="white", mec=ONE, mew=1.1, color=ONE, lw=1.0, label="one thread, MKL at one thread")
ax.plot([], [], marker="o", ms=6, mfc=EIGHT, mec=EIGHT, color=EIGHT, lw=1.0, label="eight threads, MKL at eight threads")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(OUT + ".svg", bbox_inches="tight")
fig.savefig(OUT + ".png", bbox_inches="tight")
print(f"wrote {OUT}.svg and {OUT}.png")
