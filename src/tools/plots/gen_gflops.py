#!/usr/bin/env python3
"""Throughput in GFLOPS, VectorFFT vs MKL, from the gauntlet CSV.

    python3 gen_gflops.py sweep.csv [more.csv ...] [--out name.svg] [--logy]

Input rows:  N,K,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err,route,flip

Several CSVs concatenate into one series: the 2..2048 and the 2049..4096
gauntlets together make the full 2..4096 graph. vfft_gflops is read directly;
MKL's follows from the same operation count: mkl_gflops = vfft_gflops *
vfft_ns / mkl_ns. Each size is the median over the order-flipped arms; the
sizes are joined in N, one connected line per engine, with a dot per size
while the series is sparse enough to show them. --logy puts the y axis on a
log scale, where equal speed ratios are equal vertical gaps."""
import sys, math, statistics
from collections import defaultdict
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

INK, PAPER = "#000000", "#FFFFFF"
VF, MK = "#B4261E", "#1F5FBF"
FD = matplotlib.get_data_path() + "/fonts/ttf/"
CMR, CMB = FD + "cmr10.ttf", FD + "cmb10.ttf"

args = sys.argv[1:]
out = args[args.index("--out") + 1] if "--out" in args else "vectorfft-gflops.svg"
LOGY = "--logy" in args
files = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--out")]
if not files: sys.exit(__doc__)

rows = []
for path in files:
    with open(path) as f:
        for raw in f:
            p = raw.strip().split(",")
            if len(p) < 11 or p[0] == "N": continue
            try:
                rows.append((int(p[0]), int(p[1]), p[2], p[3], float(p[4]), float(p[5]), float(p[6]), float(p[7])))
            except ValueError:
                continue

byN = defaultdict(lambda: ([], []))
byR = defaultdict(list)                            # ratio_vs_mkl per arm
for n, k, plan, path, vns, mns, gf, rt in rows:
    byN[n][0].append(gf)                           # VectorFFT GFLOPS as reported
    byN[n][1].append(gf * vns / mns)               # same flops, MKL's time
    byR[n].append(rt)
pts = [(n, statistics.median(v), statistics.median(m)) for n, (v, m) in sorted(byN.items())]

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
def put(s, size, cx=None, x=None, baseline=0, anchor="l", fname=None, col=INK):
    d, a, b = mpath(s, size, fname)
    w = b - a
    tx = cx - a - w/2 if cx is not None else (x - b if anchor == "r" else x - a)
    L.append(f'<g transform="translate({tx:.1f} {baseline:.1f})" fill="{col}"><path d="{d}"/></g>')
    return w
def line(x1, y1, x2, y2, wd=1.4, dash=None, col=INK):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    E.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="{wd}" stroke-linecap="round"{d}/>')
def dot(x, y, r, col):
    E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}"/>')
def poly(xy, col, wd, op=1.0):
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in xy)
    E.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{wd}" stroke-opacity="{op}" stroke-linejoin="round" stroke-linecap="round"/>')

W, H = 1400, 680
X0, X1, Y0, Y1 = 140, 1350, 60, 550
nmin, nmax = pts[0][0], pts[-1][0]
LOGX = nmax / nmin > 8
def xr(n):
    if LOGX: return X0 + (math.log10(n) - math.log10(nmin)) / (math.log10(nmax) - math.log10(nmin)) * (X1 - X0)
    return X0 + (n - nmin) / max(1, nmax - nmin) * (X1 - X0)
allv = [v for _, v, m in pts] + [m for _, v, m in pts]
top = max(allv)
if LOGY:
    def nice_below(v):                     # the largest 1/2/5 x 10^k not above v
        k = 10 ** math.floor(math.log10(v))
        return max(c * k for c in (1, 2, 5) if c * k <= v * (1 + 1e-9))
    def nice_above(v):                     # the smallest 1/2/5 x 10^k not below v
        k = 10 ** math.floor(math.log10(v))
        return min([c * k for c in (1, 2, 5) if c * k >= v * (1 - 1e-9)] or [10 * k])
    ylo, yhi = nice_below(min(allv)), nice_above(top)
    def yr(v): return Y1 - (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo)) * (Y1 - Y0)
else:
    step = 10 ** math.floor(math.log10(top / 5))
    if top / step > 10: step *= 2
    if top / step > 10: step *= 2.5
    ymax = math.ceil(top / step) * step
    def yr(v): return Y1 - v / ymax * (Y1 - Y0)

E.append(f'<rect x="{X0}" y="{Y0}" width="{X1-X0}" height="{Y1-Y0}" fill="none" stroke="{INK}" stroke-width="3"/>')
if LOGY:
    d = int(math.floor(math.log10(ylo) + 1e-9))
    while 10 ** d <= yhi * (1 + 1e-9):
        for m in range(1, 10):
            t = m * 10 ** d
            if t < ylo * (1 - 1e-9) or t > yhi * (1 + 1e-9): continue
            yy = yr(t); major = m in (1, 2, 5)
            if Y0 < yy < Y1:
                line(X0, yy, X0 + (12 if major else 6), yy, 2.2 if major else 1.4)
                line(X1 - (12 if major else 6), yy, X1, yy, 2.2 if major else 1.4)
            if major: put(f"{t:g}", 17, x=X0 - 14, baseline=yy + 5.5, anchor="r", fname=CMR)
        d += 1
else:
    t = 0
    while t <= ymax + 1e-9:
        yy = yr(t)
        if Y0 < yy < Y1:
            line(X0, yy, X0 + 12, yy, 2.2); line(X1 - 12, yy, X1, yy, 2.2)
        put(f"{t:g}", 17, x=X0 - 14, baseline=yy + 5.5, anchor="r", fname=CMR)
        t += step
if LOGX:
    d0, d1 = int(math.floor(math.log10(nmin))), int(math.ceil(math.log10(nmax)))
    for d in range(d0, d1 + 1):
        for m in range(1, 10):
            n = m * 10 ** d
            if nmin <= n <= nmax:
                major = m == 1 or (m in (2, 5) and d1 - d0 <= 4)
                if X0 < xr(n) < X1:
                    line(xr(n), Y1, xr(n), Y1 - (12 if major else 6), 2.2 if major else 1.4)
                    line(xr(n), Y0, xr(n), Y0 + (12 if major else 6), 2.2 if major else 1.4)
                if major:
                    put(f"{n:,}".replace(",", " "), 16, cx=xr(n), baseline=Y1 + 32, fname=CMR)
    # the ends of the range get a label when no decade label sits near them
    labeled = [m * 10 ** d for d in range(d0, d1 + 1) for m in (1, 2, 5)
               if nmin <= m * 10 ** d <= nmax and (m == 1 or d1 - d0 <= 4)]
    for n in (nmin, nmax):
        if n not in labeled and all(abs(xr(n) - xr(q)) > 48 for q in labeled):
            put(f"{n:,}".replace(",", " "), 16, cx=xr(n), baseline=Y1 + 32, fname=CMR)
else:
    span = max(1, nmax - nmin)
    xs = 10 ** math.floor(math.log10(span / 6))
    if span / xs > 12: xs *= 2
    if span / xs > 12: xs *= 2.5
    tt = math.ceil(nmin / xs) * xs
    while tt <= nmax:
        if X0 < xr(tt) < X1:
            line(xr(tt), Y1, xr(tt), Y1 - 12, 2.2); line(xr(tt), Y0, xr(tt), Y0 + 12, 2.2)
        put(f"{int(tt)}", 16, cx=xr(tt), baseline=Y1 + 32, fname=CMR)
        tt += xs
put(r"$N$" + (r"$\ \ \mathrm{(log\ scale)}$" if LOGX else ""), 18, cx=(X0 + X1) / 2, baseline=Y1 + 66)
L.append(f'<g transform="translate({X0 - 88} {(Y0 + Y1) / 2}) rotate(-90)">')
put(r"$\mathrm{throughput}\ \ (\mathrm{GFLOPS}" + (r",\ \mathrm{log\ scale}" if LOGY else "") + r")$", 17, cx=0, baseline=0)
L.append('</g>')

# the series: one connected line per engine, MKL under VectorFFT; dots while
# there are few enough sizes for a dot to mean one size
dense = len(pts) > 256
wd, op = (1.0, 0.8) if dense else (1.8, 1.0)
rad = 0 if dense else (4.6 if len(pts) <= 64 else 2.6)
poly([(xr(n), yr(m)) for n, v, m in pts], MK, wd, op)
poly([(xr(n), yr(v)) for n, v, m in pts], VF, wd, op)
if rad:
    for n, v, m in pts: dot(xr(n), yr(m), rad, MK)
    for n, v, m in pts: dot(xr(n), yr(v), rad, VF)

r0 = rows[0]
lay = "interleaved" if r0[2].startswith("z") else "split"
ordr = "natural" if r0[3].startswith("nat") else "scrambled"
plc = "out-of-place" if r0[3].endswith("oop") else "in-place"
tw = put(f"FP64 1D c2c, {lay}, {ordr} order, {plc}, K = {r0[1]}", 18, x=X0 + 16, baseline=Y0 + 30, fname=CMB)
E.append(f'<rect x="{X0+4}" y="{Y0+2}" width="{tw+24}" height="40" fill="{PAPER}" stroke="{INK}" stroke-width="2"/>')

lx, ly = X0 + 30, Y0 + 60
E.append(f'<rect x="{lx-12}" y="{ly-6}" width="186" height="64" fill="{PAPER}" stroke="{INK}" stroke-width="2"/>')
line(lx - 4, ly + 12, lx + 16, ly + 12, 2.4, col=VF); dot(lx + 6, ly + 12, 4.6, VF); put("VectorFFT", 16, x=lx + 22, baseline=ly + 17, fname=CMR)
line(lx - 4, ly + 38, lx + 16, ly + 38, 2.4, col=MK); dot(lx + 6, ly + 38, 4.6, MK); put("MKL", 16, x=lx + 22, baseline=ly + 43, fname=CMR)

# the speedup per cell is the gauntlet's: MKL time over ours, the WORSE of the
# two engine orders; the GFLOPS summary is over the POWERS OF TWO in the
# series, since a median over every size would be a median over the prime cells
worst = {n: min(r) for n, r in byR.items()}
med_all = statistics.median(worst.values())
p2 = [p for p in pts if p[0] & (p[0] - 1) == 0]
cap1 = (f"GFLOPS = 5 N K log2(N) flops over wall time, the same count for both engines; {len(pts):,} sizes, N = {nmin:,}..{nmax:,}, "
        f"each the median of two order-flipped arms, joined in N.")
cap2 = f"Median speedup over all sizes x{med_all:.2f} (MKL time over VectorFFT time, the worse of the two orders)."
if p2:
    mv = statistics.median(v for _, v, _ in p2); mm = statistics.median(m for _, _, m in p2)
    mp = statistics.median(worst[n] for n, _, _ in p2)
    pk = max(p2, key=lambda t: t[1])
    cap2 += (f" Pure powers of two ({len(p2)} cells, {p2[0][0]:,}..{p2[-1][0]:,}): VectorFFT {mv:.1f} vs MKL {mm:.1f} GFLOPS median, "
             f"x{mp:.2f}; peak {pk[1]:.0f} GFLOPS at N = {pk[0]:,}.")
else:
    mv = mm = mp = float("nan")
for cap, base in ((cap1, H - 42), (cap2, H - 20)):
    cs = 14.0
    while cs > 9 and mpath(cap, cs, CMR)[2] - mpath(cap, cs, CMR)[1] > W - 60: cs -= 0.5
    put(cap, cs, cx=W / 2, baseline=base, fname=CMR)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="Throughput in GFLOPS of VectorFFT and MKL per transform length">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
{"".join(E)}
{"".join(L)}
</svg>'''
open(out, "w").write(svg)
print("ok", out, len(pts), "sizes", f"N = {nmin}..{nmax}", f"median speedup x{med_all:.2f}; pow2 ({len(p2)} cells) VF {mv:.1f}, MKL {mm:.1f} GFLOPS, x{mp:.2f}")
