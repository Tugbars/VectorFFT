"""VectorFFT 3D c2c pow2 gauntlet vs MKL, drawn as small multiples: one N2 x N3 matrix per N1.

Same encoding as the 2D figure (gen_2d_pow2.py): diverging blue (faster) / red (slower),
five steps per arm, binned symmetrically in ratio. With 1288 cells there is no room for
per-cell numbers, so cells slower than MKL also get a black outline - the sign of a cell
never rides on hue alone. Computer Modern throughout, text outlined to paths in the SVG.

usage:  python3 gen_3d_pow2.py <gauntlet_3d.csv> [output_prefix]
        -> <output_prefix>.svg and <output_prefix>.png  (default prefix: vectorfft-3d-pow2)

Reads the per-flip CSV (N1,N2,N3,...,ratio_vs_mkl,...,flip) and scores each cell by the
worse of its flips, the same convention as the 2D report. Needs only matplotlib."""
import csv, math, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

if len(sys.argv) < 2:
    sys.exit(__doc__)
SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "vectorfft-3d-pow2"

lg2 = lambda n: int(round(math.log2(int(n))))
x = {}
for r in csv.DictReader(open(SRC)):
    k = (lg2(r["N1"]), lg2(r["N2"]), lg2(r["N3"]))
    x[k] = min(x.get(k, math.inf), float(r["ratio_vs_mkl"]))
if not x:
    sys.exit(f"no cells parsed from {SRC}")
CAP = max(i + j + k for i, j, k in x)                  # log2 of the largest volume run
print(f"{len(x)} cells parsed from {SRC}; volume cap 2^{CAP}")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "svg.fonttype": "path", "axes.formatter.use_mathtext": True,
})

# colour scale: identical to the 2D figure
EDGES = [1.1, 1.25, 1.5, 2.0]
BLUE = ["#c5dffe", "#9ec5f4", "#6da7ec", "#2a78d6", "#184f95"]
RED  = ["#fed0cb", "#f3ada7", "#e7837c", "#ca4442", "#8c2828"]
def step(v):
    f = v if v >= 1 else 1 / v
    return sum(f >= e for e in EDGES)
def fill(v):
    return (BLUE if v >= 1 else RED)[step(v)]

# panel layout, in cell units: three rows, panels shrink once the volume cap bites
N = 13                                                  # log2 N from 1 (2) to 13 (8192)
side = lambda i: min(N, CAP - i - 1)                    # rows/cols present in the N1 = 2^i panel
ROWS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13]]
GX, GY = 3.2, 4.2                                       # gaps between panels (room for ticks, titles)
TICKS = {1: "2", 5: "32", 9: "512", 13: "8192"}

fig = plt.figure(figsize=(9.4, 9.0))
ax = fig.add_axes([0.02, 0.17, 0.96, 0.77])
ax.set_aspect("equal"); ax.axis("off")

origins = {}
y0 = 0.0
for row in ROWS:
    x0 = 0.0
    for i in row:
        origins[i] = (x0, y0)
        x0 += side(i) + GX
    y0 += N + GY

for i, (ox, oy) in origins.items():
    cells = [(j, k) for (a, j, k) in x if a == i]
    for j, k in cells:
        v = x[(i, j, k)]
        px, py = ox + k - 1, oy + j - 1
        ax.add_patch(Rectangle((px, py), 1, 1, facecolor=fill(v), edgecolor="white", lw=0.35))
    for j, k in cells:                                  # outlines last, so no neighbour paints over them
        v = x[(i, j, k)]
        if v < 1:
            ax.add_patch(Rectangle((ox + k - 1 + .06, oy + j - 1 + .06), .88, .88,
                                   fill=False, edgecolor="black", lw=0.9))
    # staircase edge of the measured domain
    for j, k in cells:
        px, py = ox + k - 1, oy + j - 1
        if (i, j, k + 1) not in x: ax.plot([px + 1, px + 1], [py, py + 1], color="black", lw=0.5)
        if (i, j + 1, k) not in x: ax.plot([px, px + 1], [py + 1, py + 1], color="black", lw=0.5)
        if k == 1: ax.plot([px, px], [py, py + 1], color="black", lw=0.5)
        if j == 1: ax.plot([px, px + 1], [py, py], color="black", lw=0.5)
    s = side(i)
    ax.text(ox + s / 2, oy - 1.55, f"$N_1 = {2**i}$", ha="center", va="bottom", fontsize=10)
    for t, lab in TICKS.items():
        if t <= s:
            ax.text(ox + t - .5, oy - .25, lab, ha="center", va="bottom", fontsize=6.3, color="0.25")
            ax.text(ox - .3, oy + t - .5, lab, ha="right", va="center", fontsize=6.3, color="0.25")

# axis names once, on the first panel
ox, oy = origins[1]
ax.annotate("", xy=(ox + 9.5, oy - 3.35), xytext=(ox + 3.5, oy - 3.35),
            arrowprops=dict(arrowstyle="->", lw=0.6, color="0.25"))
ax.text(ox + 2.9, oy - 3.35, "$N_3$", ha="right", va="center", fontsize=9, color="0.25")

W = max(o[0] + side(i) for i, o in origins.items())
H = max(o[1] + side(i) for i, o in origins.items())
ax.set_xlim(-2.6, W + .3); ax.set_ylim(H + .5, -4.0)
ax.text(-2.4, origins[1][1] + 9.5, "$N_2$", ha="center", va="center", fontsize=9, color="0.25")
ax.annotate("", xy=(-2.4, origins[1][1] + 12.8), xytext=(-2.4, origins[1][1] + 10.4),
            arrowprops=dict(arrowstyle="->", lw=0.6, color="0.25"))

# legend: the 2D strip, plus the outline mark
lg = fig.add_axes([0.13, 0.110, 0.80, 0.028]); lg.axis("off"); lg.set_xlim(0, 100); lg.set_ylim(0, 1)
SW = 4.6
for n, c in enumerate(list(reversed(RED)) + BLUE):
    lg.add_patch(Rectangle((n * SW, 0), SW, 1, facecolor=c, edgecolor="white", lw=0.8))
for n, t in enumerate(["0.5", "0.67", "0.8", "0.91", "1", "1.1", "1.25", "1.5", "2"], 1):
    lg.text(n * SW, -0.3, t, ha="center", va="top", fontsize=8.2)
lg.text(-0.8, 0.5, "slower", ha="right", va="center", fontsize=8.8, fontstyle="italic")
lg.text(10 * SW + 0.8, 0.5, "faster", ha="left", va="center", fontsize=8.8)
lg.text(0, 1.45, "speedup over MKL (worse of two flips)", fontsize=9, va="bottom")
lg.add_patch(Rectangle((62, 0.05), 3.6, 0.9, facecolor=RED[1], edgecolor="black", lw=0.9))
lg.text(66.8, 0.5, "slower than MKL", va="center", fontsize=9)

# band summary by transform volume, as a small table under the legend
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
bands = [(3, 8), (9, 12), (13, 16), (17, 19), (20, CAP)]
tx0, tdx = 0.29, 0.135
fig.text(0.13, 0.052, "points", fontsize=8.8, color="0.3")
fig.text(0.13, 0.030, "won", fontsize=8.8, color="0.3")
fig.text(0.13, 0.008, "gmean", fontsize=8.8, color="0.3")
for n, (lo, hi) in enumerate(bands):
    v = [t for (i, j, k), t in x.items() if lo <= i + j + k <= hi]
    cx = tx0 + n * tdx
    fig.text(cx, 0.052, f"$2^{{{lo}}}$ - $2^{{{hi}}}$", fontsize=8.8, ha="center")
    fig.text(cx, 0.030, f"{sum(t >= 1 for t in v)}/{len(v)}", fontsize=8.8, ha="center")
    fig.text(cx, 0.008, f"{gm(v):.2f}", fontsize=8.8, ha="center")
fig.text(0.13, 0.965, f"3D c2c, interleaved, natural order, out of place, K = 1, against MKL DFTI 3D:  "
                      f"{sum(t >= 1 for t in x.values())}/{len(x)} cells won, all volumes up to $2^{{{CAP}}}$ points",
         fontsize=9.2)

fig.savefig(f"{OUT}.svg")
fig.savefig(f"{OUT}.png", dpi=170)
print(f"wrote {OUT}.svg and {OUT}.png")
