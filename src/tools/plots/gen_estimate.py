#!/usr/bin/env python3
"""The estimate (VFFT_FLAG_ESTIMATE): how a plan is computed instead of measured, drawn like the wisdom planner's
figure (vectorfft-planner.svg): the stages in boxes on one line, a small picture of each above it, a caption below.

    (N, K) -> machine -> rules -> recipe -> replay -> check -> plan, the recipe's rows held in a store in memory

The example is the estimate's own result on an Intel Core i9-14900KF, AVX2 build, for one 1D complex transform of
65536 points, natural order, out of place, one thread (est_plan_1d, src/core/il/estimate/estimate_il.h, computed
2026-11-14): 4 lanes, L1d 48 KiB, L2 2 MiB, L3 36 MiB; the tiled engine (route ztt) with the stage chain
8.4.8.4.8.8 and a tile of 1024 points (16 KiB). The rules shown are the first steps of est_c2c's list.

    python3 gen_estimate.py           # -> vectorfft-estimate.svg beside this script
"""
import math
import os
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

INK, PAPER, GREY, FAINT = "#000000", "#FFFFFF", "#8C8C8C", "#C8C8C8"
FD = matplotlib.get_data_path() + "/fonts/ttf/"
CMR, CMTT = FD + "cmr10.ttf", FD + "cmtt10.ttf"

# ---------------- the example (the estimate's own numbers) ----------------
REQUEST = ["t=c2c n=65536", "ord=nat place=oop", "T=1"]
CACHES = [("L1d", "48 KiB", 48 << 10), ("L2", "2 MiB", 2 << 20), ("L3", "36 MiB", 36 << 20)]
LANES = 4
RULES = [   # (the rule, 0 fails / 1 holds / 2 not reached)
    ("one kernel", 0),
    ("$N$ prime", 0),
    ("$2^k \\geq 2^{19}$: four-step", 0),
    ("$2^k \\leq 512$: two factors", 0),
    ("$2^k$: tiled", 1),
    ("factors, chains, ...", 2),
]
CHAIN = [8, 4, 8, 4, 8, 8]
TILE, L1D = 1024 * 16, 48 << 10
ROWS = ["il_route=ztt", "il_ztt=8.4.8.4.8.8", "il_tw=1024"]

# ---------------- drawing helpers ----------------
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
def put(s, size, cx=None, x=None, baseline=0, anchor="l", fname=None, color=INK):
    d, a, b = mpath(s, size * 1.28, fname)
    w = b - a
    tx = cx - a - w / 2 if cx is not None else (x - b if anchor == "r" else x - a)
    L.append(f'<g transform="translate({tx:.1f} {baseline:.1f})" fill="{color}"><path d="{d}"/></g>')
    return w
def put_mixed(note, size, x, baseline, color=INK):
    """words in Computer Modern roman, $...$ pieces as mathtext"""
    for i, piece in enumerate(note.split("$")):
        if not piece:
            continue
        lead = 5.0 if piece[0] == " " else 0.0
        trail = 5.0 if piece[-1] == " " else 0.0
        x += lead + put(piece.strip() if i % 2 == 0 else "$" + piece + "$", size, x=x + lead, baseline=baseline,
                        fname=CMR if i % 2 == 0 else None, color=color) + trail
    return x
def line(x1, y1, x2, y2, wd=2.0, color=INK, dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    E.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{wd}" stroke-linecap="round"{da}/>')
def rect(x, y, w, h, fill="none", stroke=INK, wd=2.0, dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    E.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{wd}"{da}/>')
def dot(x, y, r=3.6, color=INK):
    E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>')
def head(x, y, dx, dy, s=8.0):
    """an arrowhead with its tip at (x, y), pointing along (dx, dy), one of the four axes"""
    if dx:
        pts = [(x, y), (x - dx * s, y - s * 0.55), (x - dx * s, y + s * 0.55)]
    else:
        pts = [(x, y), (x - s * 0.55, y - dy * s), (x + s * 0.55, y - dy * s)]
    E.append('<polygon points="%s" fill="%s"/>' % (" ".join(f"{a:.1f},{b:.1f}" for a, b in pts), INK))
def box(cx, cy, label, w=150, h=46):
    rect(cx - w / 2, cy - h / 2, w, h, fill=PAPER, wd=2.4)
    put(label, 17, cx=cx, baseline=cy + 6.5, fname=CMTT)
    dot(cx - w / 2, cy, 2.4)
    dot(cx + w / 2, cy, 2.4)
def tick(cx, y0, y1):
    line(cx, y0, cx, y1, wd=1.2, color=GREY)
def caption(cx, lines, y):
    for k, s in enumerate(lines):
        put(s, 14, cx=cx, baseline=y + 22 * k, fname=CMR)

# ---------------- layout ----------------
W = 1600
Y = 300                                   # the main line
X = {"machine": 300, "rules": 570, "recipe": 840, "replay": 1110, "check": 1370}
BOX_W, BOX_H = 150, 46
TOP = Y - BOX_H / 2

# the line, the input and the output
line(92, Y, 1478, Y)
dot(92, Y)
put("$(N, K)$", 17, x=8, baseline=Y + 6)
for k, s in enumerate(REQUEST):
    put(s, 10, x=8, baseline=Y - 96 + 20 * k, fname=CMTT)
tick(44, Y - 44, Y - 22)
head(1478, Y, 1, 0)
put("plan", 16, x=1490, baseline=Y + 6, fname=CMTT)

# ---------- machine: the caches as bars (log scale) and the vector's lanes ----------
cx = X["machine"]
x0 = cx - 72
for k, (name, size, b) in enumerate(CACHES):
    w = 22 + 9 * (math.log2(b) - 15)          # log scale: each doubling of the cache adds 9 px
    y = 150 + 26 * k
    rect(x0, y, w, 12, fill=INK, wd=0)
    put(f"{name} {size}", 10.5, x=x0 + w + 8, baseline=y + 10, fname=CMTT)
for i in range(LANES):
    rect(x0 + 17 * i, 228, 14, 14, wd=1.6)
put(f"{LANES} lanes", 10.5, x=x0 + 17 * LANES + 6, baseline=240, fname=CMTT)
tick(cx, 252, TOP)
box(cx, Y, "machine")
caption(cx, ["caches and vector width,", "read once"], 357)

# ---------- rules: the list, read top down; the first that holds decides ----------
cx = X["rules"]
x0 = cx - 92
for k, (rule, state) in enumerate(RULES):
    y = 128 + 21 * k
    if state == 0:                                              # an empty box: the rule does not hold
        rect(x0, y - 6, 11, 11, wd=1.4)
        put_mixed(rule, 11.5, x0 + 19, y + 4.5)
    elif state == 1:
        rect(x0, y - 6, 11, 11, fill=INK, wd=1.4)
        put_mixed(rule, 11.5, x0 + 19, y + 4.5)
        line(x0 - 18, y, x0 - 6, y, wd=1.6)
        head(x0 - 3, y, 1, 0, 6.0)
    else:
        rect(x0, y - 6, 11, 11, stroke=FAINT, wd=1.4)
        put_mixed(rule, 11.5, x0 + 19, y + 4.5, color=GREY)
tick(cx, 252, TOP)
box(cx, Y, "rules")
caption(cx, ["the first rule that holds", "decides"], 357)

# ---------- recipe: the stage chain, and the tile against the L1d ----------
cx = X["recipe"]
n = len(CHAIN)
x0 = cx - (n * 24 - 4) / 2
for i, r in enumerate(CHAIN):
    rect(x0 + 24 * i, 146, 20, 20, wd=1.6)
    put(str(r), 11, cx=x0 + 24 * i + 10, baseline=161, fname=CMTT)
put("stages", 10.5, cx=cx, baseline=186, fname=CMR)
bw = 140
bx = cx - bw / 2
rect(bx, 200, bw, 16, wd=1.6)
rect(bx, 200, bw * TILE / L1D, 16, fill=INK, wd=0)
put("tile 1024 points, 16 KiB", 10.5, cx=cx, baseline=234, fname=CMR)
put("in the L1d", 10.5, x=bx + bw + 6, baseline=213, fname=CMR)
tick(cx, 252, TOP)
box(cx, Y, "recipe")
caption(cx - 12, ["the engine and", "every setting"], 357)

# ---------- replay: the row read back, the plan's passes bound ----------
cx = X["replay"]
for k in range(3):
    line(cx - 70, 158 + 14 * k, cx - 22 - (12 if k == 1 else 0), 158 + 14 * k, wd=2.2)
line(cx - 14, 172, cx + 4, 172, wd=1.8)
head(cx + 10, 172, 1, 0, 7.0)
for i in range(n):
    rect(cx + 14 + 9 * i, 160, 9, 24, fill=INK if i % 2 == 0 else "#5A5A5A", wd=0)
put("kernels, twiddles, scratch", 10.5, cx=cx, baseline=216, fname=CMR)
tick(cx, 252, TOP)
box(cx, Y, "replay")
caption(cx + 14, ["the library's own", "create builds it,", "as from wisdom"], 357)

# ---------- check: a clock that did not run ----------
cx = X["check"]
sx, sy, rr = cx - 44, 186, 25
E.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{rr}" fill="none" stroke="{INK}" stroke-width="2.2"/>')
rect(sx - 5, sy - rr - 9, 10, 7, fill=INK, wd=0)
line(sx, sy, sx, sy - rr + 7, wd=2.2)
dot(sx, sy, 2.6)
put("races   0", 10.5, x=cx - 10, baseline=176, fname=CMTT)
put("store   closed", 10.5, x=cx - 10, baseline=198, fname=CMTT)
tick(cx, 252, TOP)
box(cx, Y, "check")
caption(cx, ["nothing measured, no store", "opened; else refused"], 357)

# ---------- the store in memory, under recipe and replay ----------
SX, SY, SW, SH = 975 - 95, 420, 190, 96
rect(SX, SY, SW, SH, wd=2.0, dash="7 5")
put("store, in memory", 12, x=SX + 12, baseline=SY + 22, fname=CMTT)
line(SX + 10, SY + 31, SX + SW - 10, SY + 31, wd=1.0)
for k, s in enumerate(ROWS):
    put(s, 10.5, x=SX + 12, baseline=SY + 52 + 18 * k, fname=CMTT)
# the recipe writes its rows; the replay reads them (both lines clear of the captions)
xr, xp = X["recipe"] + 62, X["replay"] - 64
line(xr, Y + BOX_H / 2, xr, SY - 8)
head(xr, SY - 2, 0, 1)
line(xp, SY, xp, Y + BOX_H / 2 + 8)
head(xp, Y + BOX_H / 2 + 2, 0, -1)
put("write", 11, x=xr + 7, baseline=SY - 16, fname=CMR)
put("read", 11, x=xp - 7, baseline=SY - 16, anchor="r", fname=CMR)
put("the rows a race would bank; discarded after the create", 13, cx=975, baseline=SY + SH + 26, fname=CMR)

# ---------- the figure's caption ----------
put("The estimate: the plan computed from the request and this machine's caches by fixed rules, written as the "
    "verdict a race would bank", 15, cx=W / 2, baseline=598, fname=CMR)
put("and built by the same replay. Nothing is measured; no wisdom is read or written.", 15, cx=W / 2,
    baseline=622, fname=CMR)
H = 650

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "vectorfft-estimate.svg")
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="VectorFFT estimate: machine, rules, recipe, replay, check">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
{"".join(E)}
{"".join(L)}
</svg>'''
open(out, "w").write(svg)
print("ok", out, W, H)
