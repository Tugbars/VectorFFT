"""VectorFFT 2D c2c pow2 gauntlet vs MKL, drawn as an N1 x N2 speedup matrix.

Diverging blue (faster) / red (slower), five steps per arm binned symmetrically in ratio;
Computer Modern throughout, text outlined to paths in the SVG.

usage:  python3 gen_2d_pow2.py <gauntlet_report.txt> [output_prefix]
        -> <output_prefix>.svg and <output_prefix>.png  (default prefix: vectorfft-2d-pow2)

Reads the "every shape" table of the gauntlet report and plots the x column (the report's
worse-of-two-flips speedup). Needs only matplotlib, which ships the cmr10 fonts."""
import re, math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

if len(sys.argv) < 2:
    sys.exit(__doc__)
SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "vectorfft-2d-pow2"
rows = {}
for l in open(SRC):
    m = re.match(r"\s+(\d+)x(\d+)\s+\S+\s+(\S+)\s+raced\s+(\d+)\s+(\d+)\s+([\d.]+)", l)
    if m:
        a, b, r, o, c, x = m.groups()
        rows[(int(math.log2(int(a))), int(math.log2(int(b))))] = (r, float(x))
if not rows:
    sys.exit(f"no cells parsed from {SRC}")
print(f"{len(rows)} cells parsed from {SRC}")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "svg.fonttype": "path",
    "axes.formatter.use_mathtext": True,
})

K = range(1, 14)                       # log2 N, 2 .. 8192
CELL = 1.0
# Five steps per arm, matched in OKLCH lightness step for step, so a cell 1.25x slower is exactly
# as deep a red as a cell 1.25x faster is blue. Step 1 is the palest; step 5 takes white numerals.
EDGES = [1.1, 1.25, 1.5, 2.0]          # ratio edges; the red arm uses the reciprocals
BLUE = ["#c5dffe", "#9ec5f4", "#6da7ec", "#2a78d6", "#184f95"]
RED  = ["#fed0cb", "#f3ada7", "#e7837c", "#ca4442", "#8c2828"]

def step(x):
    f = x if x >= 1 else 1 / x
    return sum(f >= e for e in EDGES)          # 0..4

def fill(x):
    return (BLUE if x >= 1 else RED)[step(x)]

fig = plt.figure(figsize=(8.6, 9.4))
ax = fig.add_axes([0.13, 0.17, 0.80, 0.70])
ax.set_xlim(0.5, 13.5); ax.set_ylim(13.5, 0.5); ax.set_aspect("equal"); ax.axis("off")

# cells
for (i, j), (r, x) in rows.items():
    loss = x < 1.0
    ax.add_patch(Rectangle((j - .5, i - .5), CELL, CELL, facecolor=fill(x), edgecolor="white", lw=1.2))
    s = f"{x:.2f}" if x < 10 else f"{x:.1f}"
    ax.text(j, i + .02, s, ha="center", va="center", fontsize=8.6,
            color="white" if step(x) == 4 else "black",
            fontstyle="italic" if loss else "normal")      # italic = slower: sign never rides on hue alone

# route boundaries: heavy rule wherever adjacent cells were served by different routes
for (i, j), (r, _) in rows.items():
    if (i, j + 1) in rows and rows[(i, j + 1)][0] != r:
        ax.plot([j + .5, j + .5], [i - .5, i + .5], color="black", lw=2.0, solid_capstyle="projecting")
    if (i + 1, j) in rows and rows[(i + 1, j)][0] != r:
        ax.plot([j - .5, j + .5], [i + .5, i + .5], color="black", lw=2.0, solid_capstyle="projecting")

# outer staircase of the measured domain (N1 N2 <= 2^22)
for (i, j) in rows:
    for (di, dj, seg) in [(0, 1, ([j + .5, j + .5], [i - .5, i + .5])), (1, 0, ([j - .5, j + .5], [i + .5, i + .5])),
                          (0, -1, ([j - .5, j - .5], [i - .5, i + .5])), (-1, 0, ([j - .5, j + .5], [i - .5, i - .5]))]:
        if (i + di, j + dj) not in rows:
            ax.plot(*seg, color="black", lw=0.8)
ax.text(12.35, 12.35, "$N_1N_2 > 2^{22}$\nnot run", ha="center", va="center", fontsize=9, color="0.35")

# axes labels, matrix style: N2 across the top, N1 down the left
for k in K:
    ax.text(k, 0.28, f"{2**k}", ha="center", va="bottom", fontsize=9.5)
    ax.text(0.28, k, f"{2**k}", ha="right", va="center", fontsize=9.5)
ax.text(7, -0.35, "$N_2$", ha="center", va="bottom", fontsize=13)
ax.text(-0.95, 3.5, "$N_1$", ha="right", va="center", fontsize=13)

# predominant-route brackets (exact regions are the heavy rules)
def hbracket(j0, j1, y, label):
    ax.plot([j0 - .42, j0 - .42, j1 + .42, j1 + .42], [y + .12, y, y, y + .12], color="black", lw=0.7, clip_on=False)
    ax.text((j0 + j1) / 2, y - .1, label, ha="center", va="bottom", fontsize=9.5, fontstyle="italic")
hbracket(1, 3, -0.95, "chain + rb"); hbracket(4, 6, -0.95, "chain + rb2"); hbracket(7, 13, -0.95, "chain")
ax.plot([-1.55, -1.67, -1.67, -1.55], [6.58, 6.58, 13.42, 13.42], color="black", lw=0.7, clip_on=False)
ax.text(-1.85, 10, "turn", ha="center", va="center", rotation=90, fontsize=9.5, fontstyle="italic")

# legend: one symmetric strip, red arm left, blue arm right, edges labelled in the printed ratio
lg = fig.add_axes([0.13, 0.095, 0.80, 0.03]); lg.axis("off"); lg.set_xlim(0, 100); lg.set_ylim(0, 1)
W = 4.6
sw = list(reversed(RED)) + BLUE
for n, c in enumerate(sw):
    lg.add_patch(Rectangle((n * W, 0), W, 1, facecolor=c, edgecolor="white", lw=0.8))
labels = ["0.5", "0.67", "0.8", "0.91", "1", "1.1", "1.25", "1.5", "2"]
for n, t in enumerate(labels, 1):
    lg.plot([n * W, n * W], [0, -0.18], color="black", lw=0.5)
    lg.text(n * W, -0.3, t, ha="center", va="top", fontsize=8.2)
lg.text(-0.8, 0.5, "slower", ha="right", va="center", fontsize=8.8, fontstyle="italic")
lg.text(10 * W + 0.8, 0.5, "faster", ha="left", va="center", fontsize=8.8)
lg.text(0, 1.45, "speedup over MKL (worse of two flips)", fontsize=9, va="bottom")
lg.plot([62, 66], [0.5, 0.5], color="black", lw=2); lg.text(67, 0.5, "route change", va="center", fontsize=9)

gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
bands = [("$\\leq 256$", 0, 256), ("$257$ - $4096$", 257, 4096), ("$> 4096$", 4097, 1 << 22)]
parts = []
for lab, lo, hi in bands:
    v = [x for (i, j), (r, x) in rows.items() if lo <= 2 ** (i + j) <= hi]
    parts.append(f"{lab} pts: {sum(t >= 1 for t in v)}/{len(v)} won, gmean {gm(v):.2f}")
fig.text(0.13, 0.035, "     ".join(parts), fontsize=9.2)
fig.text(0.13, 0.008, "2D c2c, interleaved, natural order, out of place, K = 1, against MKL DFTI 2D (out of place)", fontsize=9.5)

fig.savefig(f"{OUT}.svg")
fig.savefig(f"{OUT}.png", dpi=170)
print(f"wrote {OUT}.svg and {OUT}.png")
