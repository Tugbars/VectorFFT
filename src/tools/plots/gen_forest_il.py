#!/usr/bin/env python3
"""Fig. 4 - interleaved layout against Intel MKL, forest panels.
Data from docs/performance V1_0_results.md (state of 2026-09-22).
Each row: (label, (n, lo, med, hi), open_marker, whisker_kind) where
whisker_kind is "mm" (min-max) or "p" (p10-p90, used for the thousand-cell
sweeps whose extremes the doc reports only as counts)."""
import math
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

W = 1400
INK, PAPER = "#000000", "#FFFFFF"
FD = matplotlib.get_data_path() + "/fonts/ttf/"
CMR, CMB = FD + "cmr10.ttf", FD + "cmb10.ttf"
X0, X1, NUMX = 332, 1178, 1210
ROWSP = 33

# ---------------- data ----------------
P1 = [  # 1D c2c, K=1, natural order, one thread, vs MKL DFTI
 ("every N, 2 to 2048  (out-of-place)",       (2047, 1.00, 1.27, 2.10), False, "p"),
 ("every N, 2049 to 4096  (out-of-place)",    (2048, 0.96, 1.16, 1.76), False, "p"),
 ("every N, 2 to 512  (in-place)",            (511,  0.85, 1.33, 2.35), False, "p"),
 ("pow2ladder",                                (22,   0.94, 1.14, 2.89), False, "mm"),
 ("oddband",                                   (6,    1.73, 1.88, 1.94), False, "mm"),
 ("prime N, raced Rader / Bluestein",          (13,   1.07, 1.66, 4.23), False, "mm"),
]
P2 = [  # T = 8
 ("batch, K = 4",                              (6, 1.11, 2.56,  4.47), False, "mm"),
 ("batch, K = 8",                              (6, 1.25, 4.46,  9.87), False, "mm"),
 ("batch, K = 32",                             (6, 2.98, 5.55, 10.21), False, "mm"),
 ("pow2t8",                                    (22, 0.98, 1.29, 2.84), False, "mm"),
]
P2_ALL = ("all batch cells", (18, 1.11, 4.10, 10.21), False, "mm")
P3 = [  # 2D native tier vs MKL CCE in-place, one thread
 ("sq",     (4, 1.53, 1.83, 2.18), False, "mm"),
 ("tall",   (4, 1.25, 1.39, 1.66), False, "mm"),
 ("aspect", (3, 1.49, 1.74, 1.80), True,  "mm"),
]
P3_ALL = ("all cells", (11, 1.25, 1.59, 2.18), False, "mm")
P4 = [  # 3D native tier vs MKL CCE out-of-place, 14 cells
 ("scrambled order, T = 1",  (14, 1.05, 1.35, 1.80), False, "mm"),
 ("natural order, T = 1",    (14, 0.87, 1.15, 1.72), True,  "mm"),
 ("scrambled order, T = 8",  (14, 0.65, 1.06, 1.46), True,  "mm"),
 ("natural order, T = 8",    (14, 0.78, 1.17, 1.89), True,  "mm"),
]
P5 = [  # 1D real, in-place, placement-matched
 ("r2c, in-place", (7, 1.04, 1.24, 1.50), False, "mm"),
 ("c2r, in-place", (7, 0.94, 1.05, 1.49), False, "mm"),
]
MATH_LABELS = {
 "pow2ladder": r"$\mathrm{every\ power\ of\ two,}\ 2\ \mathrm{to}\ 2^{22}$",
 "oddband":    r"$\mathrm{odd\ band,}\ N = 2^{a}{\cdot}\mathrm{odd}$",
 "pow2t8":     r"$\mathrm{single\ transform,\ powers\ of\ two\ to}\ 2^{22}$",
 "sq":         r"$\mathrm{Squares,}\ 128^2\ \mathrm{to}\ 1024^2$",
 "tall":       r"$\mathrm{Tall,}\ N_1{\times}64\ (4096\ \mathrm{to}\ 32768)$",
 "aspect":     r"$\mathrm{Aspect,}\ 16{\times}4096,\ 32{\times}1024,\ 64{\times}256$",
}

# ---------------- helpers ----------------
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
def line(x1, y1, x2, y2, wd=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    E.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{INK}" stroke-width="{wd}" stroke-linecap="round"{d}/>')

VMIN, VMAX = 0.8, 5.0
def xr(v): return X0 + (math.log(v) - math.log(VMIN)) / (math.log(VMAX) - math.log(VMIN)) * (X1 - X0)

def row(y, label, dat, open_marker=False, wk="mm", bold=False):
    n, lo, med, hi = dat
    if label in MATH_LABELS:
        put(MATH_LABELS[label] + f"  ({n})", 15.5, x=X0 - 22, baseline=y + 4.8, anchor="r")
    else:
        put(f"{label}  ({n})", 15.5, x=X0 - 22, baseline=y + 4.8, anchor="r", fname=(CMB if bold else CMR))
    clipped = hi > VMAX
    xlo, xhi = xr(lo), xr(min(hi, VMAX))
    line(xlo, y, xhi, y, 2.0 if wk == "mm" else 3.4)
    line(xlo, y - 5, xlo, y + 5, 2.0)
    if clipped:
        for dx in (-4, 3): line(xhi + dx - 3, y + 6, xhi + dx + 3, y - 6, 1.6)
    else:
        line(xhi, y - 5, xhi, y + 5, 2.0)
    if open_marker:
        E.append(f'<circle cx="{xr(med):.1f}" cy="{y}" r="4.7" fill="{PAPER}" stroke="{INK}" stroke-width="2"/>')
    else:
        E.append(f'<circle cx="{xr(med):.1f}" cy="{y}" r="4.7" fill="{INK}"/>')
    rng = f"{lo:.2f} - {hi:.2f}" if wk == "mm" else f"p10 {lo:.2f}, p90 {hi:.2f}"
    put(f"{med:.2f}  ({rng})", 13.5, x=NUMX, baseline=y + 4.2, fname=(CMB if bold else CMR))

def panel(ytop, title, note, rows, all_row, vmin, vmax, ticks, descs=()):
    global VMIN, VMAX
    VMIN, VMAX = vmin, vmax
    put(title, 18, x=96, baseline=ytop, fname=CMB)
    put(note, 15.5, x=NUMX, baseline=ytop, fname=CMR)
    y0 = ytop + 42
    y = y0
    for label, dat, op, wk in rows:
        row(y, label, dat, open_marker=op, wk=wk); y += ROWSP
    if all_row:
        y += 10
        row(y, all_row[0], all_row[1], open_marker=all_row[2], wk=all_row[3], bold=True)
    else:
        y -= ROWSP
    y1 = y
    for t in ticks:
        xt = xr(t)
        if t == 1: line(xt, y0 - 16, xt, y1 + 18, 1.4, dash="7 5")
        else:      line(xt, y0 - 12, xt, y1 + 14, 0.8, dash="1 6")
    ax = y1 + 32
    line(X0 - 4, ax, X1 + 8, ax, 1.8)
    for t in ticks:
        xt = xr(t)
        line(xt, ax, xt, ax + 7, 1.6)
        put("%g" % t, 14, cx=xt, baseline=ax + 25, fname=CMR)
    put("1 = parity", 13, cx=xr(1), baseline=y0 - 24, fname=CMR)
    dy = ax + 48
    for dl in descs:
        put(dl, 16.5, x=96, baseline=dy, fname=CMR); dy += 23
    return dy - 23 if descs else ax

# ---------------- panels ----------------
b1 = panel(54, "1D c2c, single transform (K = 1), natural order, one thread", "vs MKL DFTI, same order",
           P1, None, 0.72, 4.6, [0.8, 1, 1.25, 1.5, 2, 3, 4], descs=(
 "The gauntlets: every length from 2 to 4096 created through the front door on a scratch store (the library's own race banks each cell), then",
 "timed against MKL in its own process, core 2 with the SMT sibling held, both engine orders, best-of-5 in two windows - the WORSE flip kept.",
 "2 to 2048: 2047 cells, 17 under 0.8x; 2049 to 4096: 2048 cells, three quarters served by whole-N Bluestein at M = 8192, which carries the",
 "losers. The power-of-two ladder runs to 4 194 304 (the four-step's ceiling), 4 of 22 under parity, none under 0.8x. Thick whiskers span p10 to p90.",
))
b2 = panel(b1 + 58, "T = 8 (8 P-cores)", "18/18 batch cells won",
           P2, P2_ALL, 0.9, 11.0, [1, 1.5, 2, 3, 5, 7, 10], descs=(
 "Batch: transform-contiguous, N = 256 to 65536, byte-identical layout on both engines, scored per cell against whichever MKL arm is faster -",
 "its serial arm at every cell measured; our own T1-to-T8 scaling reaches 8.9x. Single transform: every power of two re-raced at T = 8, the",
 "door's per-thread-count race of each engine's threaded arm against the serial verdict; below 2048 the serial plan keeps the cell.",
))
b3 = panel(b2 + 58, "2D c2c, native tier, one thread, vs MKL's best 2D arm (CCE, in-place)", "11/11 cells won",
           P3, P3_ALL, 0.9, 2.4, [1, 1.25, 1.5, 1.75, 2], descs=(
 "Front-door creates with banked chains; same-run 5-arm race, 9 rounds with per-round order flip, medians. Rows served by the K=1 pairs below",
 "1024 and by ZTURN-T from 1024 up. The x64 ladder is the L2 band-threshold race: at 32768x64 the width race picked wl = 1024 and MKL's CCE",
 "arm loses even to its own real-real configuration. Aspect cells ran with wide arm spreads (open dot): sign-reliable, not two-decimal.",
))
b4 = panel(b3 + 58, "3D c2c, native tier, vs MKL's best 3D arm (CCE, out-of-place)", "14 cells, four regimes",
           P4, None, 0.6, 2.0, [0.67, 0.8, 1, 1.25, 1.5, 1.75], descs=(
 "Cubes from 16 to 128 a side, anisotropic and odd shapes, and the long-N3 cells; one race at create over structure x band width (x form x strip width",
 "for natural). Scrambled at one thread wins 14/14, 10 outside the control spread. Natural at one thread: 4 win, 8 tie, 2 lose (small cubes,",
 "where MKL's natural output is cheap). At T = 8 MKL's arm spreads reach 50-770%: natural wins or ties 14/14; scrambled loses the four small",
 "cells to the two-phase fork-join floor. The natural strip form won the threaded race at 10 of 14 cells. Ties are drawn open.",
))
b5 = panel(b4 + 58, "1D real (r2c / c2r), K = 1, in-place, vs MKL DFTI real (CCE, in-place)", "placement-matched",
           P5, None, 0.85, 1.62, [0.9, 1, 1.1, 1.25, 1.5], descs=(
 "The like-for-like real cells: matched on layout, order, direction and placement, N = 512 to 65536. Forward spectra gated elementwise",
 "cross-engine, each backward against N*x; medians of 5. r2c wins all 7; c2r wins or matches 6 of 7 (the dip is 0.94x at 4096). Out-of-place",
 "cells are withheld until a placement-matched MKL out-of-place arm is benched.",
))
put("wall-time speedup over Intel MKL (log scale; note the panels use different scales)", 15, cx=(X0 + X1) / 2, baseline=b5 + 40, fname=CMR)

caps = (
 "Fig. 4.  Interleaved layout against Intel MKL, state of 2026-09-22. Dots mark medians, whiskers span each row's cells (counts in parentheses);",
 "thin whiskers are min to max, thick ones p10 to p90; filled dots are cells outside the control spread, open dots ties or directional. i9-14900KF.",
)
for i, c in enumerate(caps):
    put(c, 16.5, cx=W/2, baseline=b5 + 74 + i * 24, fname=CMR)

H = int(b5 + 74 + len(caps) * 24 + 8)
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="Forest plots: VectorFFT interleaved layout against Intel MKL - 1D gauntlets, T=8, 2D and 3D native tiers, real transforms">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
{"".join(E)}
{"".join(L)}
</svg>'''
open("/home/claude/vectorfft-forest-il.svg", "w").write(svg)
print("ok", W, H)
