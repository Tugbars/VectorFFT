#!/usr/bin/env python3
"""VectorFFT transform coverage tree.
layout -> dims -> transform -> placement, one glyph per leaf.
Order is not a branch: it is a live axis only for c2c.

    python3 gen_coverage.py           # docs version (wordmark header)
    python3 gen_coverage.py --site    # site version (Fig. 1 caption, no wordmark)
"""
import sys
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

SITE = "--site" in sys.argv
INK, PAPER = "#000000", "#FFFFFF"
FD = matplotlib.get_data_path() + "/fonts/ttf/"
CMR, CMTT = FD + "cmr10.ttf", FD + "cmtt10.ttf"

# ---------------- data ----------------
# N native engine | R refused by contract      (P planned / Q unconfirmed no longer used)
# each string: in-place, out-of-place
COV = {
 ("interleaved", "1D"): {"c2c": "NN", "r2c": "NN", "c2r": "NN", "r2r": "NN"},
 ("interleaved", "2D"): {"c2c": "NN", "r2c": "RN", "c2r": "RN", "r2r": "NN"},
 ("interleaved", "3D"): {"c2c": "NN", "r2c": "RN", "c2r": "RN", "r2r": "NN"},
 ("split",       "1D"): {"c2c": "NN", "r2c": "RN", "c2r": "RN", "r2r": "NN"},
 ("split",       "2D"): {"c2c": "NN", "r2c": "RN", "c2r": "RN", "r2r": "NN"},
 ("split",       "3D"): {"c2c": "NN", "r2c": "RN", "c2r": "RN", "r2r": "NN"},
}
TNOTE = {"c2c": "order is live:  scrambled / natural",
         "r2c": "natural by construction",
         "c2r": "natural by construction",
         "r2r": "wrapper over r2c  (DCT / DST / DHT)"}

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
    size = size * 1.28
    d, a, b = mpath(s, size, fname)
    w = b - a
    tx = cx - a - w/2 if cx is not None else (x - b if anchor == "r" else x - a)
    L.append(f'<g transform="translate({tx:.1f} {baseline:.1f})" fill="{INK}"><path d="{d}"/></g>')
    return w
def line(x1, y1, x2, y2, wd=1.5):
    E.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{INK}" stroke-width="{wd}" stroke-linecap="round"/>')
def dot(x, y, r=3.0):
    E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{INK}"/>')
def elbow(xp, yp, xc, yc, xm):
    line(xp, yp, xm, yp); line(xm, yp, xm, yc); line(xm, yc, xc, yc)
def glyph(kind, x, y):
    if kind == "N":
        E.append(f'<circle cx="{x}" cy="{y}" r="5.6" fill="{INK}"/>')
    elif kind == "R":
        E.append(f'<line x1="{x-6.4}" y1="{y}" x2="{x+6.4}" y2="{y}" stroke="{INK}" stroke-width="2.4" stroke-linecap="round"/>')

# ---------------- layout ----------------
W = 1150
X_LAY, X_DIM, X_TR, X_PL, X_GL, X_NOTE = 96, 250, 404, 596, 722, 760
ROW = 27.5
TOP = 96 if SITE else 132
y = TOP

lay_y = []
for lay in ("interleaved", "split"):
    dys = []
    for dm in ("1D", "2D", "3D"):
        tys = []
        for tr in ("c2c", "r2c", "c2r", "r2r"):
            codes = COV[(lay, dm)][tr]
            ys = []
            for code, pl in zip(codes, ("in-place", "out-of-place")):
                put(pl, 14, x=X_PL, baseline=y + 4.5, fname=CMR)
                glyph(code, X_GL, y)
                dot(X_PL - 12, y, 2.6)
                ys.append(y)
                y += ROW
            cy = (ys[0] + ys[-1]) / 2
            tys.append(cy)
            wtr = put(tr, 15.5, x=X_TR, baseline=cy + 5, fname=CMTT)
            put(TNOTE[tr], 12.5, x=X_NOTE, baseline=cy + 4, fname=CMR)
            dot(X_TR - 12, cy, 2.8)
            xo = X_TR + wtr + 12
            for yy in ys:
                elbow(xo, cy, X_PL - 12, yy, (xo + X_PL) / 2)
            y += 8
        cy = (tys[0] + tys[-1]) / 2
        dys.append(cy)
        wdm = put(dm, 16, x=X_DIM, baseline=cy + 5.5, fname=CMR)
        dot(X_DIM - 12, cy, 3.0)
        xo = X_DIM + wdm + 12
        for yy in tys:
            elbow(xo, cy, X_TR - 12, yy, (xo + X_TR) / 2)
        y += 12
    cy = (dys[0] + dys[-1]) / 2
    lay_y.append(cy)
    wl_ = put(lay, 17, x=X_LAY, baseline=cy + 6, fname=CMR)
    dot(X_LAY - 14, cy, 3.4)
    xo = X_LAY + wl_ + 12
    for yy in dys:
        elbow(xo, cy, X_DIM - 12, yy, (xo + X_DIM) / 2)
    y += 22

ROOT_X = 46
root_y = (lay_y[0] + lay_y[-1]) / 2
dot(ROOT_X, root_y, 4.0)
for yy in lay_y:
    elbow(ROOT_X, root_y, X_LAY - 14, yy, ROOT_X + 20)

H = int(y + 96)

# ---------------- header / legend / footnotes ----------------
if not SITE:
    put(r"$\mathbf{VectorFFT}$", 30, x=X_LAY - 50, baseline=54)
    put("transform coverage", 20, x=X_LAY + 178, baseline=54, fname=CMR)
    line(X_LAY - 50, 68, W - 60, 68, 1.2)

LG_Y = TOP - 42
lx = X_LAY - 50
for code, txt in (("N", "native engine"), ("R", "refused by contract")):
    glyph(code, lx, LG_Y)
    put(txt, 12.5, x=lx + 13, baseline=LG_Y + 4, fname=CMR)
    lx += 40 + len(txt) * 8.6

foot = "Order does not branch: it is a measured axis only for c2c.  Real transforms are natural by construction, so the order axis collapses there."
foot2 = "In-place real is accepted only for 1D interleaved; every other real in-place cell is a contract refusal, not missing work."
if SITE:
    put("Fig. 1.  Transform coverage.", 16, cx=W/2, baseline=H - 62, fname=CMR)
    put(foot, 13.5, cx=W/2, baseline=H - 40, fname=CMR)
    put(foot2, 13.5, cx=W/2, baseline=H - 20, fname=CMR)
    out = "/home/claude/site-coverage.svg"
else:
    put(foot, 14, x=X_LAY - 50, baseline=H - 40, fname=CMR)
    put(foot2, 14, x=X_LAY - 50, baseline=H - 18, fname=CMR)
    out = "/home/claude/vectorfft-coverage.svg"

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="VectorFFT transform coverage tree">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
{"".join(E)}
{"".join(L)}
</svg>'''
open(out, "w").write(svg)
print("ok", out, W, H)
