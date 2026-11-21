"""The regression test's single transforms (tests/regression/), one thread, as a class x octave grid per machine.

One panel per run, stacked: a row per class of the test's cell file (the length's class in 1D, N1's in 2D, axis 0's
in 3D), a column per octave (of the length, the plane, the volume), and in each square one band per cell of that class
and octave, coloured by the cell's own speedup over the run's competitor (the worse of its two engine orders, the
gauntlet's convention) in gen_2d_pow2.py's blue / red steps, the cell and its speedup printed on it. A cell of the
file with no row in the run (not created in that build) is a grey band. Right of a row: the class's geometric mean;
under a column: the octave's.

usage:  python3 gen_regression_grid.py <rank 1|2|3> <cell file> <out prefix>
                 --run <gauntlet csv> <competitor> <heading> [--run ...]
        -> <out prefix>.svg and <out prefix>.png

Needs only matplotlib, which ships the cmr10 fonts."""
import csv, math, re, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

argv = sys.argv[1:]
runs = []
while "--run" in argv:
    i = argv.index("--run")
    runs.append(tuple(argv[i + 1:i + 4]))
    del argv[i:i + 4]
if len(argv) < 3 or not runs:
    sys.exit(__doc__)
RANK, CELLS, OUT = int(argv[0]), argv[1], argv[2]
KEYS = {1: ["N"], 2: ["N1", "N2"], 3: ["N1", "N2", "N3"]}[RANK]

CL = r"(power of two|2\^a x odd|odd|prime > 47|prime factor > 47)"
PAT = (r"^(\d+)\s+#\s+%s\s+2\^(\d+)\s" % CL) if RANK == 1 else (r"^([\dx]+)\s+#\s+%s\s+%s\s+2\^(\d+)\s" % (CL, CL))
CLASSES = ["power of two", "2^a x odd", "odd", "prime > 47", "prime factor > 47"]
SHOW = {"power of two": "power of two", "2^a x odd": "$2^a\\times$odd", "odd": "odd",
        "prime > 47": "prime $>47$", "prime factor > 47": "prime factor $>47$"}
WHOSE = {1: "class of the length", 2: "class of $N_1$", 3: "class of $N_1$ (axis 0)"}[RANK]
OCTNAME = {1: "octave of the length", 2: "octave of the plane ($N_1N_2$)", 3: "octave of the volume ($N_1N_2N_3$)"}[RANK]
UNIT = {1: "lengths", 2: "planes", 3: "volumes"}[RANK]

cells = []                      # (key, class, octave) in the file's order
for l in open(CELLS):
    m = re.match(PAT, l)
    if m:
        g = m.groups()
        cells.append((tuple(int(t) for t in g[0].split("x")), g[1], int(g[-1])))
OCT = sorted({o for _, _, o in cells})


def pts(k):
    p = 1
    for t in k:
        p *= t
    return p


def worse_of_flips(path):
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[tuple(int(r[k]) for k in KEYS)].append(float(r["ratio_vs_mkl"]))   # the exact ratio (the ns columns are rounded)
    return {k: min(v) for k, v in by.items()}


gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
EDGES = [1.1, 1.25, 1.5, 2.0]
BLUE = ["#c5dffe", "#9ec5f4", "#6da7ec", "#2a78d6", "#184f95"]
RED = ["#fed0cb", "#f3ada7", "#e7837c", "#ca4442", "#8c2828"]
NONE = "#ececec"


def step(x):
    f = x if x >= 1 else 1 / x
    return sum(f >= e for e in EDGES)


def fill(x):
    return (BLUE if x >= 1 else RED)[step(x)]


def label(k):
    return "x".join(str(t) for t in k)


# ---------------- geometry (data units; one unit = S inches, so the type sizes below hold) ----------------
S = {1: 0.56, 2: 0.56, 3: 0.56}[RANK]
CW = {1: 1.0, 2: 2.1, 3: 2.6}[RANK]          # a square's width: 1D prints a band in two lines, 2D / 3D in one
TWO_LINES = RANK == 1
BH = 0.66 if TWO_LINES else 0.40               # one band's height
LM, RS = 3.6, 1.5                              # the row labels' margin; the class column on the right
W = LM + CW * len(OCT) + RS + 0.3
members = defaultdict(list)                    # (class, octave) -> keys, smallest first
for k, c, o in cells:
    members[(c, o)].append(k)
for v in members.values():
    v.sort(key=lambda k: (pts(k), k))
rowh = {c: BH * max([len(members[(c, o)]) for o in OCT] + [1]) for c in CLASSES}
PANEL_H = 1.0 + 0.75 + sum(rowh[c] for c in CLASSES) + 0.75
H = 1.3 + PANEL_H * len(runs) + 0.4 * (len(runs) - 1) + 1.5

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.unicode_minus": False, "svg.fonttype": "path", "axes.formatter.use_mathtext": True})
fig = plt.figure(figsize=(W * S, H * S))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
x0 = LM
ax.text(x0, 0.75, {1: "VectorFFT's speedup, 1D: every length of the test at one thread",
                   2: "VectorFFT's speedup, 2D: every plane of the test at one thread",
                   3: "VectorFFT's speedup, 3D: every volume of the test at one thread"}[RANK], fontsize=13.5,
        va="center")
# the key, top right: one sample block and what its two numbers are
CMPS = {c for _, c, _ in runs}
OVER = ("speedup over %s" % next(iter(CMPS))) if len(CMPS) == 1 else "speedup over the competitor"
SIZE = {1: "the FFT size $N$", 2: "the FFT size $N_1\\times N_2$", 3: "the FFT size $N_1\\times N_2\\times N_3$"}[RANK]
KEYT = ["top: the " + OVER, "bottom: " + SIZE] if TWO_LINES else ["left: " + SIZE, "right: the " + OVER]
SAMPLE = {1: ((4096,), 1.69), 2: ((64, 64), 2.06), 3: ((64, 64, 64), 2.29)}[RANK]
KH = max(BH, 0.66) + 0.4
kr, ky = x0 + CW * len(OCT) + RS, 0.22                 # the box's right edge: the class column's
ts = [ax.text(0, ky + KH / 2 + dy, t, ha="left", va="center", fontsize=9) for t, dy in zip(KEYT, (-0.17, 0.17))]
rend = fig.canvas.get_renderer()
tw = max(t.get_window_extent(rend).width for t in ts) / (fig.dpi * S)   # the widest label in data units
KW = 0.2 + CW + 0.3 + tw + 0.3
kx = kr - KW
bx, by = kx + 0.2, ky + (KH - BH) / 2
for t in ts:
    t.set_x(bx + CW + 0.3)
ax.add_patch(Rectangle((kx, ky), KW, KH, facecolor="white", edgecolor="black", lw=0.8))
ax.add_patch(Rectangle((bx, by), CW, BH, facecolor=fill(SAMPLE[1]), edgecolor="white", lw=0.7))
kink = "white" if step(SAMPLE[1]) == 4 else "black"
if TWO_LINES:
    ax.text(bx + CW / 2, by + BH / 2 - 0.13, "%.2f" % SAMPLE[1], ha="center", va="center", fontsize=8.6, color=kink)
    ax.text(bx + CW / 2, by + BH / 2 + 0.18, label(SAMPLE[0]), ha="center", va="center", fontsize=6.9, color=kink)
else:
    ax.text(bx + 0.12, by + BH / 2, label(SAMPLE[0]), ha="left", va="center", fontsize=6.8, color=kink)
    ax.text(bx + CW - 0.12, by + BH / 2, "%.2f" % SAMPLE[1], ha="right", va="center", fontsize=8.0, color=kink)
y = 1.3
for path, cmp_name, heading in runs:
    sp = worse_of_flips(path)
    got = [x for k, x in sp.items()]
    ax.text(x0, y + 0.45, heading, fontsize=11.5, va="center", fontweight="bold")
    ax.text(x0, y + 0.95, "%d of the file's %d %s measured, geometric mean %.2fx, %d at or above 1"
            % (sum(1 for k, _, _ in cells if k in sp), len(cells), UNIT, gm(got), sum(x >= 1 for x in got)),
            fontsize=9.5, va="center", color="#333333")
    yh = y + 1.0 + 0.45
    for j, o in enumerate(OCT):
        ax.text(x0 + CW * (j + 0.5), yh, f"$2^{{{o}}}$", ha="center", va="center", fontsize=10.5)
    ax.text(x0 + CW * len(OCT) + RS / 2, yh, "class", ha="center", va="center", fontsize=9, fontstyle="italic")
    ax.text(x0 - 0.25, yh, WHOSE, ha="right", va="center", fontsize=9, fontstyle="italic")
    yr = y + 1.75
    for c in CLASSES:
        h = rowh[c]
        ax.text(x0 - 0.25, yr + h / 2, SHOW[c], ha="right", va="center", fontsize=10.5)
        for j, o in enumerate(OCT):
            xs = x0 + CW * j
            ks = members[(c, o)]
            if not ks:
                ax.add_patch(Rectangle((xs, yr), CW, h, facecolor="#f7f7f7", edgecolor="white", lw=1.4))
                continue
            bh = h / len(ks)
            for b, k in enumerate(ks):
                yb = yr + b * bh
                x = sp.get(k)
                ax.add_patch(Rectangle((xs, yb), CW, bh, facecolor=fill(x) if x else NONE, edgecolor="white",
                                       lw=0.7))
                ink = "white" if x and step(x) == 4 else "black"
                val = ("%.2f" % x if x < 10 else "%.1f" % x) if x else "not run"
                sty = "italic" if (x and x < 1) or not x else "normal"
                if TWO_LINES:
                    ax.text(xs + CW / 2, yb + bh / 2 - 0.13, val, ha="center", va="center", fontsize=8.6, color=ink,
                            fontstyle=sty)
                    ax.text(xs + CW / 2, yb + bh / 2 + 0.18, label(k), ha="center", va="center", fontsize=6.9,
                            color=ink)
                else:
                    ax.text(xs + 0.12, yb + bh / 2, label(k), ha="left", va="center", fontsize=6.8, color=ink)
                    ax.text(xs + CW - 0.12, yb + bh / 2, val, ha="right", va="center", fontsize=8.0, color=ink,
                            fontstyle=sty)
            ax.add_patch(Rectangle((xs, yr), CW, h, facecolor="none", edgecolor="white", lw=1.8))
        v = [sp[k] for k, cc, _ in cells if cc == c and k in sp]
        if v:
            ax.text(x0 + CW * len(OCT) + RS / 2, yr + h / 2, "%.2f" % gm(v), ha="center", va="center", fontsize=10)
        yr += h
    for j, o in enumerate(OCT):
        v = [sp[k] for k, _, oo in cells if oo == o and k in sp]
        if v:
            ax.text(x0 + CW * (j + 0.5), yr + 0.38, "%.2f" % gm(v), ha="center", va="center", fontsize=8.6)
    ax.text(x0 - 0.25, yr + 0.38, "octave", ha="right", va="center", fontsize=9, fontstyle="italic")
    ax.text(x0 + CW * len(OCT) + RS / 2, yr + 0.38, "%.2f" % gm(got), ha="center", va="center", fontsize=10,
            fontweight="bold")
    y += PANEL_H + 0.4

# the legend: one strip, red arm left, blue arm right, the edges in the printed ratio
yl = y + 0.1
SW = 0.95
for n, col in enumerate(list(reversed(RED)) + BLUE):
    ax.add_patch(Rectangle((x0 + n * SW, yl), SW, 0.42, facecolor=col, edgecolor="white", lw=0.8))
for n, t in enumerate(["0.5", "0.67", "0.8", "0.91", "1", "1.1", "1.25", "1.5", "2"], 1):
    ax.text(x0 + n * SW, yl + 0.72, t, ha="center", va="center", fontsize=8.4)
ax.text(x0 - 0.2, yl + 0.21, ("%s faster" % next(iter(CMPS))) if len(CMPS) == 1 else "the competitor faster",
        ha="right", va="center", fontsize=9, fontstyle="italic")
ax.text(x0 + 10 * SW + 0.2, yl + 0.21, "VectorFFT faster", ha="left", va="center", fontsize=9)
ax.add_patch(Rectangle((x0 + 10 * SW + 3.6, yl), 0.8, 0.42, facecolor=NONE, edgecolor="white"))
ax.text(x0 + 10 * SW + 4.6, yl + 0.21, "not run: the build did not create the cell", ha="left", va="center",
        fontsize=9)
ax.text(x0, yl + 1.2, "speedup = %s time over VectorFFT's, the worse of the cell's two engine orders; "
        % (("%s's" % next(iter(CMPS))) if len(CMPS) == 1 else "the competitor's") +
        "right: the class's geometric mean over all octaves; under a panel: the octave's over all classes",
        fontsize=8.6, va="center", color="#333333")

fig.savefig(OUT + ".svg")
fig.savefig(OUT + ".png", dpi=110)
print("wrote", OUT + ".svg", OUT + ".png", "size %.1f x %.1f in" % (W * S, H * S))
