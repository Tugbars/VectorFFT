"""VectorFFT 3D pow2 gauntlet (c2c, r2c or c2r) vs MKL (or another comparator), drawn as small multiples: one N2 x N3 matrix per N1.

Same encoding as the 2D figure (gen_2d_pow2.py): diverging blue (faster) / red (slower),
five steps per arm, binned symmetrically in ratio. With 1288 cells there is no room for
per-cell numbers, so cells slower than MKL also get a black outline - the sign of a cell
never rides on hue alone. Computer Modern throughout, text outlined to paths in the SVG.

usage:  python3 gen_3d_pow2.py <gauntlet_3d.csv> [output_prefix] [--cmp NAME]
        -> <output_prefix>.svg and <output_prefix>.png  (default prefix: vectorfft-3d-pow2)
        --cmp: the comparator's name on the legend and the caption (default MKL)

Reads the per-flip CSV (N1,N2,N3,...,ratio_vs_mkl,...,flip) and scores each cell by the
worse of its flips, the same convention as the 2D report. The grid is the data's: the axis
range, the panels (one per N1 present) and the ticks follow the power-of-two volumes in the
CSV; a volume with a side that is not a power of two is not drawn (counted in the caption).
The transform comes from the plan column. Needs only matplotlib."""
import csv, math, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

argv = sys.argv[1:]
CMP = "MKL"
if "--cmp" in argv:
    i = argv.index("--cmp")
    CMP = argv[i + 1]
    del argv[i:i + 2]
if not argv:
    sys.exit(__doc__)
SRC = argv[0]
OUT = argv[1] if len(argv) > 1 else "vectorfft-3d-pow2"

lg2 = lambda n: int(round(math.log2(int(n))))
pow2 = lambda n: int(n) > 0 and int(n) & (int(n) - 1) == 0
x = {}
odd = set()
XF = "c2c"
for r in csv.DictReader(open(SRC)):
    XF = "r2c" if "r2c" in r["plan"] else "c2r" if "c2r" in r["plan"] else "c2c"
    if not (pow2(r["N1"]) and pow2(r["N2"]) and pow2(r["N3"])):
        odd.add((r["N1"], r["N2"], r["N3"]))   # a side not a power of two: no cell on the log2 grid
        continue
    k = (lg2(r["N1"]), lg2(r["N2"]), lg2(r["N3"]))
    x[k] = min(x.get(k, math.inf), float(r["ratio_vs_mkl"]))
if not x:
    sys.exit(f"no cells parsed from {SRC}")
CAP = max(i + j + k for i, j, k in x)                  # log2 of the largest volume run
LO = min(min(c) for c in x)                            # the grid's smallest and largest side (log2)
HI = max(max(c) for c in x)
print(f"{len(x)} cells parsed from {SRC}; volume cap 2^{CAP}; sides 2^{LO}..2^{HI}"
      + (f"; {len(odd)} non-power-of-two volume(s) not drawn" if odd else ""))

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

# panel layout, in cell units: rows of four panels, panels shrink once the volume cap bites
N = HI - LO + 1                                         # sides per axis (13 on the 2..8192 grid)
side = lambda i: min(N, CAP - i - 2 * LO + 1)           # rows/cols present in the N1 = 2^i panel
PAN = sorted({i for i, j, k in x})
ROWS = [PAN[q:q + 4] for q in range(0, len(PAN), 4)]
if len(ROWS) > 1 and len(ROWS[-1]) == 1:                # a lone last panel joins the row above
    last = ROWS.pop()
    ROWS[-1] += last
GX, GY = 3.2, 4.2                                       # gaps between panels (room for ticks, titles)
TSTEP = 4 if N > 8 else 2
TICKS = {t: f"{2**t}" for t in range(LO, HI + 1, TSTEP)}
F = N / 13.0                                            # the axis-name marks scale with the panel

fig = plt.figure(figsize=(9.4, 9.0))
ax = fig.add_axes([0.02, 0.17, 0.96, 0.77])
ax.set_aspect("equal"); ax.axis("off")
ax.set_anchor("N")                                      # a grid shorter than the box sits under the title

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
        px, py = ox + k - LO, oy + j - LO
        ax.add_patch(Rectangle((px, py), 1, 1, facecolor=fill(v), edgecolor="white", lw=0.35))
    for j, k in cells:                                  # outlines last, so no neighbour paints over them
        v = x[(i, j, k)]
        if v < 1:
            ax.add_patch(Rectangle((ox + k - LO + .06, oy + j - LO + .06), .88, .88,
                                   fill=False, edgecolor="black", lw=0.9))
    # staircase edge of the measured domain
    for j, k in cells:
        px, py = ox + k - LO, oy + j - LO
        if (i, j, k + 1) not in x: ax.plot([px + 1, px + 1], [py, py + 1], color="black", lw=0.5)
        if (i, j + 1, k) not in x: ax.plot([px, px + 1], [py + 1, py + 1], color="black", lw=0.5)
        if k == LO: ax.plot([px, px], [py, py + 1], color="black", lw=0.5)
        if j == LO: ax.plot([px, px + 1], [py, py], color="black", lw=0.5)
    s = side(i)
    ax.text(ox + s / 2, oy - 1.55, f"$N_1 = {2**i}$", ha="center", va="bottom", fontsize=10)
    for t, lab in TICKS.items():
        if t - LO + 1 <= s:
            ax.text(ox + t - LO + .5, oy - .25, lab, ha="center", va="bottom", fontsize=6.3, color="0.25")
            ax.text(ox - .3, oy + t - LO + .5, lab, ha="right", va="center", fontsize=6.3, color="0.25")

# axis names once, on the first panel
ox, oy = origins[PAN[0]]
ax.annotate("", xy=(ox + 9.5 * F, oy - 3.35), xytext=(ox + 3.5 * F, oy - 3.35),
            arrowprops=dict(arrowstyle="->", lw=0.6, color="0.25"))
ax.text(ox + 2.9 * F, oy - 3.35, "$N_3$", ha="right", va="center", fontsize=9, color="0.25")

W = max(o[0] + side(i) for i, o in origins.items())
H = max(o[1] + side(i) for i, o in origins.items())
ax.set_xlim(-2.6, W + .3); ax.set_ylim(H + .5, -4.0)
ax.text(-2.4, origins[PAN[0]][1] + 9.5 * F, "$N_2$", ha="center", va="center", fontsize=9, color="0.25")
ax.annotate("", xy=(-2.4, origins[PAN[0]][1] + 12.8 * F), xytext=(-2.4, origins[PAN[0]][1] + 10.4 * F),
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
lg.text(0, 1.45, f"speedup over {CMP} (worse of two flips)", fontsize=9, va="bottom")
lg.add_patch(Rectangle((62, 0.05), 3.6, 0.9, facecolor=RED[1], edgecolor="black", lw=0.9))
lg.text(66.8, 0.5, f"slower than {CMP}", va="center", fontsize=9)

# band summary by transform volume, as a small table under the legend
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
MINP = min(i + j + k for i, j, k in x)
bands = [(max(lo, MINP), hi) for lo, hi in [(3, 8), (9, 12), (13, 16), (17, 19), (20, CAP)]
         if any(max(lo, MINP) <= i + j + k <= hi for i, j, k in x)]   # the bands the data reaches
tx0, tdx = 0.29, 0.135
fig.text(0.13, 0.052, "points", fontsize=8.8, color="0.3")
fig.text(0.13, 0.030, "won", fontsize=8.8, color="0.3")
fig.text(0.13, 0.008, "gmean", fontsize=8.8, color="0.3")
for n, (lo, hi) in enumerate(bands):
    v = [t for (i, j, k), t in x.items() if lo <= i + j + k <= hi]
    cx = tx0 + n * tdx
    fig.text(cx, 0.052, f"$2^{{{lo}}}$ - $2^{{{hi}}}$" if hi > lo else f"$2^{{{lo}}}$", fontsize=8.8, ha="center")
    fig.text(cx, 0.030, f"{sum(t >= 1 for t in v)}/{len(v)}", fontsize=8.8, ha="center")
    fig.text(cx, 0.008, f"{gm(v):.2f}", fontsize=8.8, ha="center")
fig.text(0.13, 0.965, f"3D {XF}, interleaved, natural order, out of place, K = 1, against {'MKL DFTI' if CMP == 'MKL' else CMP} 3D:  "
                      f"{sum(t >= 1 for t in x.values())}/{len(x)} cells won, all volumes up to $2^{{{CAP}}}$ points",
         fontsize=9.2)
if odd:
    fig.text(0.13, 0.945, f"{len(odd)} volume(s) with a side that is not a power of two: not drawn", fontsize=8.6, color="0.3")

fig.savefig(f"{OUT}.svg")
fig.savefig(f"{OUT}.png", dpi=170)
print(f"wrote {OUT}.svg and {OUT}.png")
