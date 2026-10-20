#!/usr/bin/env python3
"""VectorFFT transform coverage tree, the README's figure.

layout -> dims -> transform -> request, one glyph per leaf. A transform forks into the six
requests a configuration can make of it: in place, out of place, scrambled order, a contiguous
batch, a lane-major batch, threads. A dot is a request vfft_create() serves, a dash one it
refuses by contract; a letter beside a dot names a condition written under the tree. The
split layout is a single line, unreleased, with no leaves. The data table (LEAVES, NOTES) is
written in the same form as the guide's figure (how-to-use/source/figures/ch03_coverage.py),
so a change is made once in each.

    python3 gen_coverage.py           # docs version (wordmark header) -> vectorfft-coverage.svg beside this script
    python3 gen_coverage.py --site    # site version (Fig. 1 caption, no wordmark) -> site-coverage.svg
"""
import os
import sys
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

SITE = "--site" in sys.argv
INK, PAPER = "#000000", "#FFFFFF"
FD = matplotlib.get_data_path() + "/fonts/ttf/"
CMR, CMTT, CMMI = FD + "cmr10.ttf", FD + "cmtt10.ttf", FD + "cmmi10.ttf"

# ---------------- data ----------------
# S served | R refused by contract. One row per transform leaf: (rank, transform, [in place,
# out of place, scrambled order, batch contiguous, batch lane-major, threads], {request:
# footnote letter}). The same table as the guide's figure.
S, R = 1, 0
REQUESTS = ["in place", "out of place", "scrambled order", "batch, contiguous", "batch, lane-major", "threads"]
LEAVES = [
    ("1D", "c2c", [S, S, S, S, S, S], {}),
    ("1D", "r2c", [S, S, R, S, S, S], {4: "a"}),
    ("1D", "c2r", [S, S, R, S, S, S], {4: "a"}),
    ("2D", "c2c", [S, S, S, S, S, S], {}),
    ("2D", "r2c", [S, S, R, S, S, S], {4: "b"}),
    ("2D", "c2r", [S, S, R, S, S, S], {4: "b"}),
    ("3D", "c2c", [S, S, S, S, S, S], {}),
    ("3D", "r2c", [S, S, R, S, S, S], {4: "c"}),
    ("3D", "c2r", [S, S, R, S, S, S], {4: "c"}),
]
TNOTE = {"c2c": "order is live:  scrambled / natural",
         "r2c": "natural by construction",
         "c2r": "natural by construction"}
NOTES = [   # (letter, the note; $...$ is Computer Modern mathtext)
    ("a", "a lane-major batch of 1D real transforms needs an even $N \\geq 4$"),
    ("b", "a lane-major batch of 2D real arrays needs $N_1 \\geq 2$ and an even $N_2 \\geq 4$"),
    ("c", "a lane-major batch of 3D real arrays needs $N_1, N_2 \\geq 2$ and an even $N_3 \\geq 4$"),
]
SPLIT_NOTE = "unreleased in this version"

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
def put_mixed(note, size, x, baseline):
    """the words in Computer Modern roman, the $...$ pieces as mathtext"""
    for i, piece in enumerate(note.split("$")):
        if not piece:
            continue
        lead = 4.0 if piece[0] == " " else 0.0
        trail = 4.0 if piece[-1] == " " else 0.0
        x += lead + put(piece.strip() if i % 2 == 0 else "$" + piece + "$", size, x=x + lead, baseline=baseline,
                        fname=CMR if i % 2 == 0 else None) + trail
def line(x1, y1, x2, y2, wd=1.5):
    E.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{INK}" stroke-width="{wd}" stroke-linecap="round"/>')
def dot(x, y, r=3.0):
    E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{INK}"/>')
def elbow(xp, yp, xc, yc, xm):
    line(xp, yp, xm, yp); line(xm, yp, xm, yc); line(xm, yc, xc, yc)
def glyph(kind, x, y):
    if kind == S:
        E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.4" fill="{INK}"/>')
    else:
        E.append(f'<line x1="{x-7.2:.1f}" y1="{y:.1f}" x2="{x+7.2:.1f}" y2="{y:.1f}" stroke="{INK}" stroke-width="2.8" stroke-linecap="round"/>')

# ---------------- layout ----------------
W = 1300
X_LAY, X_DIM, X_TR, X_RQ, X_GL, X_NOTE = 110, 300, 480, 690, 905, 950
ROW = 33
TOP = 108 if SITE else 150
y = TOP

# the interleaved layout: ranks, transforms, requests
dys = []
for rank in ("1D", "2D", "3D"):
    tys = []
    for rk, tr, glyphs, notes in LEAVES:
        if rk != rank:
            continue
        ys = []
        for c, (rq, g) in enumerate(zip(REQUESTS, glyphs)):
            put(rq, 18, x=X_RQ, baseline=y + 6, fname=CMR)
            glyph(g, X_GL, y)
            if c in notes:
                put(notes[c], 14, x=X_GL + 10, baseline=y - 6, fname=CMMI)
            dot(X_RQ - 14, y, 3.0)
            ys.append(y)
            y += ROW
        cy = (ys[0] + ys[-1]) / 2
        tys.append(cy)
        wtr = put(tr, 20, x=X_TR, baseline=cy + 6.5, fname=CMTT)
        put(TNOTE[tr], 16, x=X_NOTE, baseline=cy + 5.5, fname=CMR)
        dot(X_TR - 14, cy, 3.2)
        xo = X_TR + wtr + 14
        for yy in ys:
            elbow(xo, cy, X_RQ - 14, yy, (xo + X_RQ) / 2)
        y += 8
    cy = (tys[0] + tys[-1]) / 2
    dys.append(cy)
    wdm = put(rank, 21, x=X_DIM, baseline=cy + 7, fname=CMR)
    dot(X_DIM - 14, cy, 3.4)
    xo = X_DIM + wdm + 14
    for yy in tys:
        elbow(xo, cy, X_TR - 14, yy, (xo + X_TR) / 2)
    y += 12
cy = (dys[0] + dys[-1]) / 2
lay_y = [cy]
wl_ = put("interleaved", 22, x=X_LAY, baseline=cy + 7.5, fname=CMR)
dot(X_LAY - 16, cy, 3.8)
xo = X_LAY + wl_ + 14
for yy in dys:
    elbow(xo, cy, X_DIM - 14, yy, (xo + X_DIM) / 2)

# the split layout: one line, no leaves
y += 26
lay_y.append(y)
wl_ = put("split", 22, x=X_LAY, baseline=y + 7.5, fname=CMR)
dot(X_LAY - 16, y, 3.8)
put(SPLIT_NOTE, 16, x=X_LAY + wl_ + 18, baseline=y + 5.5, fname=CMR)
y += ROW

ROOT_X = 50
root_y = (lay_y[0] + lay_y[-1]) / 2
dot(ROOT_X, root_y, 4.4)
for yy in lay_y:
    elbow(ROOT_X, root_y, X_LAY - 16, yy, ROOT_X + 22)

# ---------------- header / legend / footnotes ----------------
if not SITE:
    put(r"$\mathbf{VectorFFT}$", 36, x=X_LAY - 60, baseline=62)
    put("transform coverage", 24, x=X_LAY + 220, baseline=62, fname=CMR)
    line(X_LAY - 60, 78, W - 70, 78, 1.2)

LG_Y = TOP - 48
lx = X_LAY - 60
for code, txt in ((S, "served"), (R, "refused by contract")):
    glyph(code, lx, LG_Y)
    put(txt, 16, x=lx + 14, baseline=LG_Y + 5, fname=CMR)
    lx += 48 + len(txt) * 11

fy = y + 46
if SITE:
    put("Fig. 1.  Transform coverage.", 20, cx=W/2, baseline=fy, fname=CMR)
    fy += 32
for letter, note in NOTES:
    put(letter, 14, x=X_LAY - 60, baseline=fy - 5, fname=CMMI)
    put_mixed(note, 18, X_LAY - 42, fy)
    fy += 28
H = int(fy + 24)

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "site-coverage.svg" if SITE else "vectorfft-coverage.svg")
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="VectorFFT transform coverage tree">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
{"".join(E)}
{"".join(L)}
</svg>'''
open(out, "w").write(svg)
print("ok", out, W, H)
