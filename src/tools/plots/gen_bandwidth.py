#!/usr/bin/env python3
"""FFT bandwidth comparison, paper style (Computer Modern, black on white).

    python3 gen_bandwidth.py                       # demo data
    python3 gen_bandwidth.py bandwidth.csv [PEAK]  # your data; PEAK in GB/s
                                                   # draws the peak-DRAM reference

CSV columns: library,N,gbps   (achieved bandwidth = bytes moved / wall time;
one row per measured length). Same marker vocabulary and running-median
overlay as gen_precision.py, so the two figures sit together. Vertical
dotted markers show where the FP64 complex working set (16 N bytes,
in-place) leaves L1, L2 and L3 - edit CACHES for your host."""
import sys, csv, math, random
from collections import defaultdict
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

W, H = 1400, 780
INK, PAPER = "#000000", "#FFFFFF"
FD = matplotlib.get_data_path() + "/fonts/ttf/"
CMR, CMB = FD + "cmr10.ttf", FD + "cmb10.ttf"
MAX_PTS = 2500
CACHES = [("L1", 48 * 1024), ("L2", 2 * 1024 * 1024), ("L3", 36 * 1024 * 1024)]   # i9-14900KF P-core
BYTES_PER_N = 16                                                                  # FP64 complex, in place
PEAK = float(sys.argv[2]) if len(sys.argv) > 2 else 85.0                          # GB/s, DDR5-5600 dual channel ~ 85

# ---------------- data ----------------
def demo():
    random.seed(11)
    prof = {"VectorFFT": (1.00, 0.06), "MKL": (0.88, 0.08)}
    out = defaultdict(list)
    for name, (a, jit) in prof.items():
        for n in range(16, 4_000_001, 37):
            ws = n * BYTES_PER_N
            if ws < CACHES[0][1]:     lvl = 380.0
            elif ws < CACHES[1][1]:   lvl = 240.0
            elif ws < CACHES[2][1]:   lvl = 150.0
            else:                     lvl = 70.0
            ramp = min(1.0, math.log2(n) / 9.0)
            v = a * lvl * ramp * (1 + jit * random.gauss(0, 1)) * (0.9 if n & (n - 1) else 1.0)
            out[name].append((n, max(1.0, v)))
    return out

def load(path):
    out = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["library"]].append((int(row["N"]), float(row["gbps"])))
    return out

data = load(sys.argv[1]) if len(sys.argv) > 1 else demo()
names = list(data.keys())
XMIN = 10 ** math.floor(math.log10(min(n for s in data.values() for n, _ in s)))
XMAX = 10 ** math.ceil(math.log10(max(n for s in data.values() for n, _ in s)))
YMAX = math.ceil(max(v for s in data.values() for _, v in s) / 50) * 50

# ---------------- geometry ----------------
X0, X1, Y0, Y1 = 150, 1330, 90, 600
LX0, LX1 = math.log10(XMIN), math.log10(XMAX)
def xr(n):  return X0 + (math.log10(n) - LX0) / (LX1 - LX0) * (X1 - X0)
def yr(v):  return Y1 - v / YMAX * (Y1 - Y0)

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
for d in range(int(LX0), int(LX1) + 1):
    n = 10 ** d
    line(xr(n), Y1, xr(n), Y1 + 9, 1.6)
    put(rf"$10^{{{d}}}$", 15, cx=xr(n), baseline=Y1 + 34)
    for m in range(2, 10):
        if n * m <= XMAX: line(xr(n * m), Y1, xr(n * m), Y1 + 5, 1.0)
step = 50 if YMAX <= 500 else 100
for t in range(0, YMAX + 1, step):
    line(X0 - 9, yr(t), X0, yr(t), 1.6)
    put(f"{t}", 15, x=X0 - 16, baseline=yr(t) + 5, anchor="r", fname=CMR)
    if t: line(X0, yr(t), X1, yr(t), 0.7, dash="1 6")
put(r"$N\ \ \mathrm{(log\ scale)}$", 16, cx=(X0 + X1) / 2, baseline=Y1 + 70)
L.append(f'<g transform="translate({X0 - 100} {(Y0 + Y1) / 2}) rotate(-90)">')
put(r"$\mathrm{achieved\ bandwidth\ \ (GB/s)}$", 16, cx=0, baseline=0)
L.append('</g>')

# cache boundaries: where 16 N bytes leaves each level
for lab, bytes_ in CACHES:
    n_edge = bytes_ / BYTES_PER_N
    if XMIN < n_edge < XMAX:
        line(xr(n_edge), Y0 - 4, xr(n_edge), Y1, 1.0, dash="3 5")
        put(lab, 12.5, cx=xr(n_edge), baseline=Y0 - 12, fname=CMR)
# peak DRAM reference
if 0 < PEAK < YMAX:
    line(X0, yr(PEAK), X1, yr(PEAK), 1.2, dash="8 5")
    put(f"DRAM peak, {PEAK:g} GB/s", 12.5, x=X0 + 8, baseline=yr(PEAK) - 6, fname=CMR)

# ---------------- series ----------------
def thin(pts):
    pts = sorted(pts)
    if len(pts) <= MAX_PTS: return pts
    st = len(pts) / MAX_PTS
    return [pts[int(i * st)] for i in range(MAX_PTS)]

def running_median(pts, bins=56):
    pts = sorted(pts); out = []
    lo, hi = math.log10(pts[0][0]), math.log10(pts[-1][0])
    for b in range(bins):
        a0, a1 = lo + (hi - lo) * b / bins, lo + (hi - lo) * (b + 1) / bins
        seg = [v for n, v in pts if a0 <= math.log10(n) < a1 or (b == bins - 1 and math.log10(n) == a1)]
        if seg:
            seg.sort(); out.append((10 ** ((a0 + a1) / 2), seg[len(seg) // 2]))
    return out

for i, name in enumerate(names):
    kind, filled = MARKERS[i % len(MARKERS)]
    E.append('<g opacity="0.55">')
    for n, v in thin(data[name]):
        marker(kind, filled, xr(n), yr(min(v, YMAX)), 2.1, color_of(i, name))
    E.append('</g>')
for i, name in enumerate(names):
    med = running_median(data[name])
    for (na, va), (nb, vb) in zip(med, med[1:]):
        line(xr(na), yr(va), xr(nb), yr(vb), 2.6, col=color_of(i, name))

# ---------------- legend + title ----------------
lx, ly = X1 - 200, Y0 + 14
E.append(f'<rect x="{lx-12}" y="{ly-14}" width="190" height="{len(names)*26+16}" fill="{PAPER}" stroke="{INK}" stroke-width="1.2"/>')
for i, name in enumerate(names):
    kind, filled = MARKERS[i % len(MARKERS)]
    line(lx - 4, ly + i * 26 + 4, lx + 30, ly + i * 26 + 4, 2.6, col=color_of(i, name))
    marker(kind, filled, lx + 13, ly + i * 26 + 4, 4.2, color_of(i, name))
    put(name, 15, x=lx + 42, baseline=ly + i * 26 + 9, fname=CMR)
tw = put("FP64 complex, single thread, in place", 16, cx=(X0 + X1) / 2 - 40, baseline=Y0 + 26, fname=CMB)
E.append(f'<rect x="{(X0+X1)/2-40-tw/2-12}" y="{Y0+4}" width="{tw+24}" height="32" fill="none" stroke="{INK}" stroke-width="1.2"/>')

cap1 = "Achieved bandwidth (bytes read and written per transform, over wall time) against transform length; markers are individual"
cap2 = "lengths (thinned), heavy lines the running median. Dotted verticals: where the 16 N-byte working set leaves each cache level."
put(cap1, 14, cx=W / 2, baseline=Y1 + 106, fname=CMR)
put(cap2, 14, cx=W / 2, baseline=Y1 + 126, fname=CMR)
if len(sys.argv) < 2:
    put("Demo data - replace with bandwidth.csv (library,N,gbps) and pass the host's peak as the second argument.", 12.5, cx=W/2, baseline=Y1 + 148, fname=CMR)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="FFT bandwidth comparison: achieved GB/s versus transform length, one marker shape per library, running medians, cache-boundary markers">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
{"".join(E)}
{"".join(L)}
</svg>'''
open("vectorfft-bandwidth.svg", "w").write(svg)
print("ok", len(svg))
