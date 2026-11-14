#!/usr/bin/env python3
"""The two ways a create plans, side by side, in the style of the planner's and the estimate's figures
(vectorfft-planner.svg, gen_estimate.py): the request splits into measuring, the default (wisdom: found, or
race the candidates and keep the winner) and the estimate, VFFT_FLAG_ESTIMATE (this machine, the rules, the
row written in memory); both branches end in the same build of the plan.

    python3 gen_planning.py           # -> vectorfft-planning.svg beside this script
"""
import os
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

INK, PAPER, GREY = "#000000", "#FFFFFF", "#6E6E6E"
FD = matplotlib.get_data_path() + "/fonts/ttf/"
CMR, CMTT, CMBX = FD + "cmr10.ttf", FD + "cmtt10.ttf", FD + "cmb10.ttf"

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
def put(s, size, cx=None, x=None, baseline=0, anchor="l", fname=CMR, color=INK):
    d, a, b = mpath(s, size * 1.28, fname)
    w = b - a
    tx = cx - a - w / 2 if cx is not None else (x - b if anchor == "r" else x - a)
    L.append(f'<g transform="translate({tx:.1f} {baseline:.1f})" fill="{color}"><path d="{d}"/></g>')
    return w
def width(s, size, fname=CMR):
    _, a, b = mpath(s, size * 1.28, fname)
    return b - a
def line(x1, y1, x2, y2, wd=2.0, color=INK):
    E.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{wd}" stroke-linecap="round"/>')
def head(x, y, dx, dy, s=9.0):
    """an arrowhead with its tip at (x, y), pointing along one axis (dx, dy)"""
    pts = ([(x, y), (x - dx * s, y - s * 0.55), (x - dx * s, y + s * 0.55)] if dx else
           [(x, y), (x - s * 0.55, y - dy * s), (x + s * 0.55, y - dy * s)])
    E.append('<polygon points="%s" fill="%s"/>' % (" ".join(f"{a:.1f},{b:.1f}" for a, b in pts), INK))
def dot(x, y, r=3.6):
    E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{INK}"/>')
def box(cx, cy, lines, w, size=16, fname=CMR):
    h = 22 + 22 * len(lines)
    E.append(f'<rect x="{cx - w / 2:.1f}" y="{cy - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{PAPER}" '
             f'stroke="{INK}" stroke-width="2.4"/>')
    y0 = cy + 6 - 11 * (len(lines) - 1)
    for k, s in enumerate(lines):
        put(s, size, cx=cx, baseline=y0 + 22 * k, fname=fname)
    return h

# ---------------- layout ----------------
W, CX = 1600, 800
XL, XR = 430, 1170                       # the two branches' centre lines
WL, WR = 250, 400                        # their boxes' widths

# the request, split in two
put("the request:  transform, lengths, placement, order, batch, threads", 18, cx=CX, baseline=58)
line(CX, 74, CX, 120)
dot(CX, 120, 4.4)
line(XL, 120, XR, 120)
line(XL, 120, XL, 160)
line(XR, 120, XR, 160)
put("measuring, the default", 21, cx=XL, baseline=186, fname=CMBX)
wl = width("the estimate, ", 21, CMBX)
wt = width("VFFT_FLAG_ESTIMATE", 19, CMTT)
x0 = XR - (wl + 8 + wt) / 2
put("the estimate, ", 21, x=x0, baseline=186, fname=CMBX)
put("VFFT_FLAG_ESTIMATE", 19, x=x0 + wl + 8, baseline=186, fname=CMTT)
line(XL, 200, XL, 222)
head(XL, 230, 0, 1)
line(XR, 200, XR, 222)
head(XR, 230, 0, 1)

# ---------- measuring: wisdom, then found, or race and keep the winner ----------
Y1, Y2, Y3 = 262, 372, 472
h = box(XL, Y1, ["wisdom store"], WL)
XF = XL + WL / 2 + 120                   # the found line's x
line(XL + WL / 2, Y1, XF, Y1)
put("found", 15, cx=(XL + WL / 2 + XF) / 2, baseline=Y1 - 10)
line(XL, Y1 + h / 2, XL, Y2 - h / 2 - 8)
head(XL, Y2 - h / 2, 0, 1)
put("missing", 15, x=XL + 10, baseline=(Y1 + Y2) / 2 + 5)
box(XL, Y2, ["race the candidates"], WL)
line(XL, Y2 + h / 2, XL, Y3 - h / 2 - 8)
head(XL, Y3 - h / 2, 0, 1)
box(XL, Y3, ["keep the winner"], WL)
put("in wisdom, for", 14, x=XL - WL / 2 - 16, baseline=Y3 - 2, anchor="r", color=GREY)
put("every later create", 14, x=XL - WL / 2 - 16, baseline=Y3 + 20, anchor="r", color=GREY)

# ---------- the estimate: this machine, the rules, the row in memory ----------
R1, R2, R3 = 268, 378, 488
h2 = box(XR, R1, ["this machine:", "L1d, L2, L3, vector width"], WR)
line(XR, R1 + h2 / 2, XR, R2 - h2 / 2 - 8)
head(XR, R2 - h2 / 2, 0, 1)
box(XR, R2, ["rules: the first that holds picks", "the engine and its settings"], WR)
line(XR, R2 + h2 / 2, XR, R3 - h2 / 2 - 8)
head(XR, R3 - h2 / 2, 0, 1)
box(XR, R3, ["written as the row a race would keep,", "in memory only"], WR)
put("nothing measured,", 14, x=XR + WR / 2 + 16, baseline=R3 - 2, color=GREY)
put("no store read or written", 14, x=XR + WR / 2 + 16, baseline=R3 + 20, color=GREY)

# ---------- both build the plan the same way ----------
YB, BW = 640, 230
hb = box(CX, YB, ["build the plan"], BW)
yj = YB                                   # the joins enter the box's sides
line(XL, Y3 + h / 2, XL, yj)
line(XF, Y1, XF, yj)
line(XL, yj, CX - BW / 2 - 8, yj)
head(CX - BW / 2, yj, 1, 0)
line(XR, R3 + h2 / 2, XR, yj)
line(XR, yj, CX + BW / 2 + 8, yj)
head(CX + BW / 2, yj, -1, 0)
line(CX, YB + hb / 2, CX, YB + hb / 2 + 30)
head(CX, YB + hb / 2 + 38, 0, 1)
put("plan", 17, cx=CX, baseline=YB + hb / 2 + 64, fname=CMTT)

# ---------- the caption ----------
put("Two ways to plan. Measuring, the default, keeps the winner of a race in wisdom and serves it to every later "
    "create;", 15, cx=CX, baseline=772)
put("the estimate computes the plan from this machine's caches by fixed rules, measuring nothing. Both build the "
    "plan the same way.", 15, cx=CX, baseline=796)
H = 822

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "vectorfft-planning.svg")
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="VectorFFT: the two ways a create plans, measuring and the estimate">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
{"".join(E)}
{"".join(L)}
</svg>'''
open(out, "w").write(svg)
print("ok", out, W, H)
