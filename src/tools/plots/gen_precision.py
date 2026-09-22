#!/usr/bin/env python3
"""FFT precision comparison, paper style (Computer Modern, black on white).

    python3 gen_precision.py                 # demo data, four libraries
    python3 gen_precision.py errors.csv      # your data

CSV columns: library,N,l2_error   (one row per measured transform).
Series are told apart by marker shape, not color, so the figure survives
grayscale printing. Points are thinned to at most MAX_PTS per series and a
running median per series is drawn on top; the error axis is in units of
1e-16 with a dashed line at machine epsilon for reference."""
import sys, csv, math, random
from collections import defaultdict
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

W, H = 1400, 760
INK, PAPER = "#000000", "#FFFFFF"
FD = matplotlib.get_data_path() + "/fonts/ttf/"
CMR, CMB = FD + "cmr10.ttf", FD + "cmb10.ttf"
MAX_PTS = 2500
EPS = 2.220446049250313e-16

# ---------------- data ----------------
def demo():
    random.seed(7)
    shapes = {"VectorFFT": (0.36, 0.035), "MKL": (0.40, 0.050)}
    out = defaultdict(list)
    for name, (a, b) in shapes.items():
        for n in range(16, 100001, 3):
            lg = math.log2(n)
            base = a * lg ** 0.5 * EPS * 0.9
            ripple = b * EPS * (1 + math.sin(lg * 6.0) ** 2) * (n & (n - 1) != 0)
            out[name].append((n, base + ripple + random.gauss(0, 0.05 * EPS)))
    return out

def load(path):
    out = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["library"]].append((int(row["N"]), float(row["l2_error"])))
    return out

data = load(sys.argv[1]) if len(sys.argv) > 1 else demo()
names = list(data.keys())
XMIN, XMAX = 10, 1e5
YMAX = math.ceil(max(e for s in data.values() for _, e in s) / 1e-16 / 2) * 2  # units of 1e-16

# ---------------- geometry ----------------
X0, X1, Y0, Y1 = 150, 1330, 90, 600
def xr(n):  return X0 + (math.log10(n) - 1) / (math.log10(XMAX) - 1) * (X1 - X0)
def yr(e):  return Y1 - (e / 1e-16) / YMAX * (Y1 - Y0)

def mpath(s, size, fname=None):
    tp = TextPath((0, 0), s, size=size, prop=FontProperties(fname=fname) if fname else None)
    v, c = tp.vertices, tp.codes
    d, i = [], 0
    while i < len(c):
        k = c[i]
        if k == 1:   d.append(f"M{v[i][0]:.2f} {-v[i][1]:.2f}"); i += 1
        elif k == 2: d.append(f"L{v[i][0]:.2f} {-v[i][1]:.2f}"); i += 1
        elif k == 3: d.append(f"Q{v[i][0]:.2f} {-v[i][1]:.2f} {v[i+1][0]:.2f} {-v[i+1][1]:.2f}"); i += 2
        elif k == 4: d.append(f"C{v[i][0]:.2f} {-v[i][1]:.2f} {v[i+1][0]:.2f} {-v[i+1][1]:.2f} {v[i+2][0]:.2f} {-v[i+2][1]:.2f}"); i += 3
        else:        d.append("Z"); i += 1
    return (" ".join(d), v[:, 0].min(), v[:, 0].max()) if len(v) else ("", 0, 0)

E, L = [], []
def put(s, size, cx=None, x=None, baseline=0, anchor="l", fname=None):
    d, a, b = mpath(s, size, fname)
    w = b - a
    tx = cx - a - w/2 if cx is not None else (x - b if anchor == "r" else x - a)
    L.append(f'<g transform="translate({tx:.1f} {baseline:.1f})" fill="{INK}"><path d="{d}"/></g>')
    return w
def line(x1, y1, x2, y2, wd=1.4, dash=None, col=INK):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    E.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="{wd}" stroke-linecap="round"{d}/>')

MARKERS = [("circle", True), ("square", True), ("diamond", False), ("tri", True), ("circle", False)]
COLORS = {"VectorFFT": "#B4261E", "MKL": "#1F5FBF"}          # vermilion / blue
FALLBACK = ["#B4261E", "#1F5FBF", "#2E7D32", "#6A1B9A", "#000000"]
def color_of(i, name): return COLORS.get(name, FALLBACK[i % len(FALLBACK)])
def marker(kind, filled, x, y, r, col=INK):
    fill = col if filled else PAPER
    INKc = col
    if kind == "circle":
        E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{INKc}" stroke-width="0.9"/>')
    elif kind == "square":
        E.append(f'<rect x="{x-r:.1f}" y="{y-r:.1f}" width="{2*r}" height="{2*r}" fill="{fill}" stroke="{INKc}" stroke-width="0.9"/>')
    elif kind == "diamond":
        E.append(f'<path d="M{x:.1f} {y-r*1.3:.1f} L{x+r*1.3:.1f} {y:.1f} L{x:.1f} {y+r*1.3:.1f} L{x-r*1.3:.1f} {y:.1f} Z" fill="{fill}" stroke="{INKc}" stroke-width="0.9"/>')
    else:
        E.append(f'<path d="M{x:.1f} {y-r*1.3:.1f} L{x+r*1.2:.1f} {y+r:.1f} L{x-r*1.2:.1f} {y+r:.1f} Z" fill="{fill}" stroke="{INKc}" stroke-width="0.9"/>')

# ---------------- axes ----------------
line(X0, Y0 - 10, X0, Y1, 1.8); line(X0, Y1, X1, Y1, 1.8)
for d in range(1, 6):
    n = 10 ** d
    if n <= XMAX:
        line(xr(n), Y1, xr(n), Y1 + 9, 1.6)
        put(rf"$10^{{{d}}}$", 15, cx=xr(n), baseline=Y1 + 34)
    for m in range(2, 10):
        if n * m <= XMAX: line(xr(n * m), Y1, xr(n * m), Y1 + 5, 1.0)
step = 2 if YMAX <= 12 else 4
for t in range(0, YMAX + 1, step):
    line(X0 - 9, yr(t * 1e-16), X0, yr(t * 1e-16), 1.6)
    put(f"{t}", 15, x=X0 - 16, baseline=yr(t * 1e-16) + 5, anchor="r", fname=CMR)
    if t: line(X0, yr(t * 1e-16), X1, yr(t * 1e-16), 0.7, dash="1 6")
put(r"$N\ \ \mathrm{(log\ scale)}$", 16, cx=(X0 + X1) / 2, baseline=Y1 + 70)
L.append(f'<g transform="translate({X0 - 100} {(Y0 + Y1) / 2}) rotate(-90)">')
put(r"$L_2\ \mathrm{error}\ \ (\times 10^{-16})$", 16, cx=0, baseline=0)
L.append('</g>')
line(X0, yr(EPS), X1, yr(EPS), 1.2, dash="8 5")
put(r"$\varepsilon = 2.2\times10^{-16}$", 12.5, x=X1 - 4, baseline=yr(EPS) - 6, anchor="r")

# ---------------- series ----------------
def thin(pts):
    pts = sorted(pts)
    if len(pts) <= MAX_PTS: return pts
    st = len(pts) / MAX_PTS
    return [pts[int(i * st)] for i in range(MAX_PTS)]

def running_median(pts, bins=48):
    pts = sorted(pts); out = []
    lo, hi = math.log10(pts[0][0]), math.log10(pts[-1][0])
    for b in range(bins):
        a0, a1 = lo + (hi - lo) * b / bins, lo + (hi - lo) * (b + 1) / bins
        seg = [e for n, e in pts if a0 <= math.log10(n) < a1 or (b == bins - 1 and math.log10(n) == a1)]
        if seg:
            seg.sort(); out.append((10 ** ((a0 + a1) / 2), seg[len(seg) // 2]))
    return out

for i, name in enumerate(names):
    kind, filled = MARKERS[i % len(MARKERS)]
    E.append('<g opacity="0.55">')
    for n, e in thin(data[name]):
        if XMIN <= n <= XMAX: marker(kind, filled, xr(n), yr(e), 2.1, color_of(i, name))
    E.append('</g>')
for i, name in enumerate(names):
    med = running_median(data[name])
    for (na, ea), (nb, eb) in zip(med, med[1:]):
        line(xr(na), yr(ea), xr(nb), yr(eb), 2.6, col=color_of(i, name))

# ---------------- legend + title ----------------
lx, ly = X0 + 22, Y0 + 14
E.append(f'<rect x="{lx-12}" y="{ly-14}" width="190" height="{len(names)*26+16}" fill="{PAPER}" stroke="{INK}" stroke-width="1.2"/>')
for i, name in enumerate(names):
    kind, filled = MARKERS[i % len(MARKERS)]
    line(lx - 4, ly + i * 26 + 4, lx + 30, ly + i * 26 + 4, 2.6, col=color_of(i, name))
    marker(kind, filled, lx + 13, ly + i * 26 + 4, 4.2, color_of(i, name))
    put(name, 15, x=lx + 42, baseline=ly + i * 26 + 9, fname=CMR)
tw = put("FP64, forward transform", 16, cx=(X0 + X1) / 2 + 120, baseline=Y0 + 26, fname=CMB)
E.append(f'<rect x="{(X0+X1)/2+120-tw/2-12}" y="{Y0+4}" width="{tw+24}" height="32" fill="none" stroke="{INK}" stroke-width="1.2"/>')

cap = ("Relative L2 error of the forward complex transform against a long-double reference, every length in the sweep; "
       "markers are individual lengths (thinned), heavy lines the running median per library.")
put(cap, 13.5, cx=W / 2, baseline=Y1 + 106, fname=CMR)
if len(sys.argv) < 2:
    put("Demo data - replace with errors.csv (library,N,l2_error).", 12.5, cx=W/2, baseline=Y1 + 128, fname=CMR)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="FFT precision comparison: L2 error versus transform length, one marker shape per library, running medians">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
{"".join(E)}
{"".join(L)}
</svg>'''
open("vectorfft-precision.svg", "w").write(svg)
print("ok", len(svg))
