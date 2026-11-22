#!/usr/bin/env python3
"""VectorFFT transform coverage tree.

layout -> ranks -> transform, then one row of six small glyphs per transform, one per request a
configuration can make of it: in place, out of place, scrambled order, a contiguous batch, a
lane-major batch, threads. The tree carries the figure; a glyph only has to be seen: a dot is a
request vfft_create() serves, a dash one it refuses by contract; a letter beside a dot or a
transform names a condition written under the tree. Ranks that serve the same requests share one
branch. The data table (LEAVES, NOTES) is written in the same form as the guide's figure
(how-to-use/source/figures/ch03_coverage.py), so a change is made once in each. Every glyph was
checked with vfft_create() on small sizes on 2026-11-14, the rows of rank 4 to 7 on 2026-11-22.

The README shows vectorfft-coverage-0.7.svg; the earlier figure, vectorfft-coverage.svg, is no longer written.

    python3 gen_coverage.py           # docs version (wordmark header) -> vectorfft-coverage-0.7.svg beside this script
    python3 gen_coverage.py --site    # site version (Fig. 1 caption, no wordmark) -> site-coverage-0.7.svg
"""
import os
import sys
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

SITE = "--site" in sys.argv
INK, PAPER, LEAD = "#000000", "#FFFFFF", "#B0B0B0"
FD = matplotlib.get_data_path() + "/fonts/ttf/"
CMR, CMTT, CMMI = FD + "cmr10.ttf", FD + "cmtt10.ttf", FD + "cmmi10.ttf"

# ---------------- data ----------------
# S served | R refused by contract. One row per leaf: (layout, ranks, the transform's lines, code
# font?, [in place, out of place, scrambled order, batch contiguous, batch lane-major, threads],
# {request: footnote letter}, footnote letter of the transform or None). The same table as the
# guide's figure.
S, R = 1, 0
REQUESTS = [("in", "place"), ("out of", "place"), ("scrambled", "order"),
            ("batch,", "contiguous"), ("batch,", "lane-major"), ("threads", None)]
LEAVES = [
    ("interleaved", "1D, 2D, 3D", ["c2c"], True, [S, S, S, S, S, S], {}, None),
    ("interleaved", "1D, 2D, 3D", ["r2c"], True, [S, S, R, S, S, S], {4: "a"}, None),
    ("interleaved", "1D, 2D, 3D", ["c2r"], True, [S, S, R, S, S, S], {4: "a"}, None),
    ("interleaved", "4D to 7D", ["c2c"], True, [S, S, R, S, S, S], {}, None),
    ("interleaved", "4D to 7D", ["r2c"], True, [R, S, R, S, S, S], {4: "a"}, None),
    ("interleaved", "4D to 7D", ["c2r"], True, [R, S, R, S, S, S], {4: "a"}, None),
    ("split", "1D", ["c2c"], True, [S, S, R, S, S, S], {}, "b"),
    ("split", "2D, 3D", ["c2c"], True, [S, S, R, R, R, S], {}, "b"),
    ("real-to-real", "1D", ["DCT-I, DST-I,", "DST-II, DST-III"], False, [S, S, S, R, R, S], {2: "c"}, None),
]
NOTES = [   # (letter, the note's lines; $...$ is Computer Modern mathtext)
    ("a", ["a lane-major batch of real arrays needs an even last extent of at least 4, every other extent at least 2"]),
    ("b", ["the split layout serves, in 1D, the powers of two from 16 to 262144 and the lengths $2^a m$ up to 262144,",
           "$a \\geq 4$ and $m$ one of 3, 5, 7, 9 and 15; in 2D and 3D, such a last extent up to 131072, every other",
           "extent a product of two to seven of the factors 3, 4, 5, 7, 8, 9 and 15"]),
    ("c", ["a real-to-real transform asked in scrambled order runs in natural order"]),
]

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
        lead = 6.5 if piece[0] == " " else 0.0
        trail = 6.5 if piece[-1] == " " else 0.0
        x += lead + put(piece.strip() if i % 2 == 0 else "$" + piece + "$", size, x=x + lead, baseline=baseline,
                        fname=CMR if i % 2 == 0 else None) + trail
def line(x1, y1, x2, y2, wd=2.0, color=INK, dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    E.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{wd}" stroke-linecap="round"{da}/>')
def dot(x, y, r=4.0):
    E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{INK}"/>')
def elbow(xp, yp, xc, yc, xm):
    line(xp, yp, xm, yp); line(xm, yp, xm, yc); line(xm, yc, xc, yc)
def glyph(kind, x, y, small=True):
    r, half, sw = (4.6, 5.4, 2.4) if small else (6.4, 7.2, 2.8)
    if kind == S:
        E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{INK}"/>')
    else:
        E.append(f'<line x1="{x-half:.1f}" y1="{y:.1f}" x2="{x+half:.1f}" y2="{y:.1f}" stroke="{INK}" stroke-width="{sw}" stroke-linecap="round"/>')

# ---------------- layout ----------------
W = 1420
ROOT_X, X_LAY, X_DIM, X_TR = 50, 115, 370, 640   # the root's dot; each level's lettering
F_LAY, F_DIM, F_TR = 28, 26, 25                  # each level's lettering size
X_COL0, PITCH = 925, 84                          # the glyph columns: small, close together
COLS = [X_COL0 + i * PITCH for i in range(len(REQUESTS))]
ROW, GAP, LINE2 = 76, 44, 17                     # a leaf row's pitch; the gap between layouts; half the
                                                 # distance between the two lines of one label
HEAD = 108 if SITE else 150                      # the column headings' first line
y = HEAD + 80                                    # the first leaf row

# the column headings, two lines, over the glyphs
for x, (top, bottom) in zip(COLS, REQUESTS):
    if bottom is None:
        put(top, 13, cx=x, baseline=HEAD + 19, fname=CMR)
    else:
        put(top, 13, cx=x, baseline=HEAD, fname=CMR)
        put(bottom, 13, cx=x, baseline=HEAD + 19, fname=CMR)

# the leaves, layout by layout and rank group by rank group
lay_y = []
layouts = list(dict.fromkeys(leaf[0] for leaf in LEAVES))
for li, lay in enumerate(layouts):
    if li:
        y += GAP
    dim_y = []
    groups = list(dict.fromkeys(leaf[1] for leaf in LEAVES if leaf[0] == lay))
    for dims in groups:
        leaf_y = []
        for _, _, lines, code, glyphs, notes, tnote in (lf for lf in LEAVES if lf[0] == lay and lf[1] == dims):
            two = len(lines) > 1
            if two:
                y += LINE2
            xe = X_TR
            for k, txt in enumerate(lines):
                off = (2 * k - (len(lines) - 1)) * LINE2
                xe = max(xe, X_TR + put(txt, F_TR, x=X_TR, baseline=y + 8 + off, fname=CMTT if code else CMR))
            if tnote:
                put(tnote, 16, x=xe + 5, baseline=y - 8, fname=CMMI)
                xe += 14
            dot(X_TR - 16, y)
            line(xe + 16, y, COLS[0] - 20, y, wd=1.2, color=LEAD, dash="1.5 6")
            for c, (x, g) in enumerate(zip(COLS, glyphs)):
                glyph(g, x, y)
                if c in notes:
                    put(notes[c], 12, x=x + 8, baseline=y - 5, fname=CMMI)
            leaf_y.append(y)
            y += ROW + (LINE2 if two else 0)
        cy = (leaf_y[0] + leaf_y[-1]) / 2
        dim_y.append(cy)
        wdm = put(dims, F_DIM, x=X_DIM, baseline=cy + 9, fname=CMR)
        dot(X_DIM - 16, cy)
        xo = X_DIM + wdm + 16
        for yy in leaf_y:
            elbow(xo, cy, X_TR - 16, yy, (xo + X_TR) / 2)
    cy = (dim_y[0] + dim_y[-1]) / 2
    lay_y.append(cy)
    wl_ = put(lay, F_LAY, x=X_LAY, baseline=cy + 9.5, fname=CMR)
    dot(X_LAY - 16, cy, 4.4)
    xo = X_LAY + wl_ + 16
    for yy in dim_y:
        elbow(xo, cy, X_DIM - 16, yy, (xo + X_DIM) / 2)

root_y = (lay_y[0] + lay_y[-1]) / 2
dot(ROOT_X, root_y, 5.5)
for yy in lay_y:
    elbow(ROOT_X, root_y, X_LAY - 16, yy, ROOT_X + 24)
y -= ROW

# ---------------- header / legend / footnotes ----------------
if not SITE:
    put(r"$\mathbf{VectorFFT}$", 36, x=X_LAY - 65, baseline=62)
    put("transform coverage", 24, x=X_LAY + 215, baseline=62, fname=CMR)
    line(X_LAY - 65, 78, W - 50, 78, 1.2)

LG_Y = HEAD + 4
lx = X_LAY - 60
for code, txt in ((S, "served"), (R, "refused by contract")):
    glyph(code, lx, LG_Y)
    put(txt, 15, x=lx + 12, baseline=LG_Y + 5, fname=CMR)
    lx += 44 + len(txt) * 10.5

fy = y + 72
if SITE:
    put("Fig. 1.  Transform coverage.", 20, cx=W/2, baseline=fy, fname=CMR)
    fy += 34
for letter, lines in NOTES:
    put(letter, 14, x=X_LAY - 60, baseline=fy - 5, fname=CMMI)
    for k, note in enumerate(lines):
        put_mixed(note, 18, X_LAY - 42, fy)
        fy += 26 if k < len(lines) - 1 else 30
H = int(fy + 20)

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "site-coverage-0.7.svg" if SITE else "vectorfft-coverage-0.7.svg")
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="VectorFFT transform coverage tree">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
{"".join(E)}
{"".join(L)}
</svg>'''
open(out, "w").write(svg)
print("ok", out, W, H)
