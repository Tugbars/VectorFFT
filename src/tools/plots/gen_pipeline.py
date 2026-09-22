#!/usr/bin/env python3
"""The OCaml codelet compiler, drawn as a signal-flow pipeline.

    python3 gen_pipeline.py                # writes vectorfft-pipeline.svg
    python3 gen_pipeline.py --fig 1        # renumber the caption

Six stages on one rail - expr, algsimp, dft, schedule, regalloc, emit_c -
each with a small vignette of the program at that point: a shared DAG, the
same DAG with a branch eliminated, a 4-point butterfly lattice, a linearized
instruction list, a 16-slot register file with 12 live (the R=20 zero-spill
peak), and a page of straight-line C. W_20 enters on the left; r20_dit.c
leaves on the right. Computer Modern throughout, outlined to paths."""
import sys
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

FIG = sys.argv[sys.argv.index("--fig") + 1] if "--fig" in sys.argv else "2"
W, H = 1600, 400
INK, PAPER = "#000000", "#FFFFFF"
FD = matplotlib.get_data_path() + "/fonts/ttf/"
CMR, CMTT = FD + "cmr10.ttf", FD + "cmtt10.ttf"

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
    size = size * 1.12
    d, a, b = mpath(s, size, fname)
    w = b - a
    tx = cx - a - w/2 if cx is not None else (x - b if anchor == "r" else x - a)
    L.append(f'<g transform="translate({tx:.1f} {baseline:.1f})" fill="{INK}"><path d="{d}"/></g>')
    return w
def line(x1, y1, x2, y2, wd=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    E.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{INK}" stroke-width="{wd}" stroke-linecap="round"{d}/>')
def dot(x, y, r=3.3):
    E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{INK}"/>')
def node(x, y, r=4.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    E.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{PAPER}" stroke="{INK}" stroke-width="1.7"{d}/>')

# ---------------- rail + stage boxes ----------------
RY = 330
BW, BH = 150, 46
centers = [240 + i * 224 for i in range(6)]
names = ["expr", "algsimp", "dft", "schedule", "regalloc", "emit_c"]
descs = ["arithmetic as a shared DAG", "simplification and CSE",
         "factor into butterflies", "dependency-aware ordering",
         "Sethi-Ullman, zero spills", "straight-line C codelets"]

TAP0, TAP1 = 70, 1470
line(TAP0, RY, TAP1, RY, 2.0)
dot(TAP0, RY, 3.6); dot(TAP1, RY, 3.6)

boxes, labels = [], []
for cx, nm, ds in zip(centers, names, descs):
    xl, xr = cx - BW / 2, cx + BW / 2
    boxes.append(f'<rect x="{xl:.1f}" y="{RY - BH / 2}" width="{BW}" height="{BH}" fill="{PAPER}" stroke="{INK}" stroke-width="2.2"/>')
    dot(xl, RY); dot(xr, RY)
    put(nm, 19, cx=cx, baseline=RY + 6.5, fname=CMTT)
    put(ds, 15, cx=cx, baseline=RY + 60, fname=CMR)

put(r"$\mathbf{W}_{20}$", 21, x=TAP0 - 14, baseline=RY + 7, anchor="r")
put("r20_dit.c", 16, x=TAP1 + 14, baseline=RY + 5.5, fname=CMTT)

# ---------------- vignettes ----------------
VB = 278
def leader(cx): line(cx, VB + 6, cx, RY - BH / 2 - 4, 1.0)

# 1 expr - shared DAG
cx = centers[0]
P = {"r": (cx, 168), "a": (cx - 34, 208), "b": (cx + 34, 208),
     "l1": (cx - 54, 252), "s": (cx, 252), "l2": (cx + 54, 252)}
for u, v in (("r", "a"), ("r", "b"), ("a", "l1"), ("a", "s"), ("b", "s"), ("b", "l2")):
    line(*P[u], *P[v], 1.5)
for p in P.values(): node(*p)
leader(cx)

# 2 algsimp - same DAG, one branch eliminated (dashed)
cx = centers[1]
P = {"r": (cx, 168), "a": (cx - 34, 208), "b": (cx + 34, 208),
     "l1": (cx - 54, 252), "s": (cx, 252), "l2": (cx + 54, 252)}
for u, v in (("r", "a"), ("a", "l1"), ("a", "s")):
    line(*P[u], *P[v], 1.5)
for u, v in (("r", "b"), ("b", "s"), ("b", "l2")):
    line(*P[u], *P[v], 1.3, dash="3 4")
for k in ("r", "a", "l1", "s"): node(*P[k])
for k in ("b", "l2"): node(*P[k], dash="3 4")
leader(cx)

# 3 dft - 4-point butterfly lattice
cx = centers[2]
xs = [cx - 46, cx, cx + 46]
ys = [176, 204, 232, 260]
for s, prs in ((0, [(0, 2), (1, 3)]), (1, [(0, 1), (2, 3)])):
    for a, b in prs:
        for p, q in ((a, a), (b, b), (a, b), (b, a)):
            line(xs[s], ys[p], xs[s + 1], ys[q], 1.3)
for x in xs:
    for y in ys: dot(x, y, 2.4)
leader(cx)

# 4 schedule - linearized instruction list
cx = centers[3]
for i, w in enumerate((52, 40, 56, 34, 48, 42)):
    line(cx - 42, 178 + i * 17, cx - 42 + w, 178 + i * 17, 2.6)
ax = cx + 40
line(ax, 176, ax, 258, 1.4)
E.append(f'<path d="M{ax - 4.5} 256 L{ax + 4.5} 256 L{ax} 268 Z" fill="{INK}"/>')
leader(cx)

# 5 regalloc - 16-register file, 12 live (the R=20 zero-spill peak)
cx = centers[4]
g, cell = 6, 15
x0, y0 = cx - 2 * cell - 1.5 * g, 174
for i in range(16):
    r, c = divmod(i, 4)
    x, y = x0 + c * (cell + g), y0 + r * (cell + g)
    E.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell}" height="{cell}" fill="{INK if i < 12 else PAPER}" stroke="{INK}" stroke-width="1.5"/>')
put("ymm", 11, cx=cx, baseline=270, fname=CMTT)
leader(cx)

# 6 emit_c - a page of straight-line C
cx = centers[5]
pw, ph = 118, 104
E.append(f'<rect x="{cx - pw / 2:.1f}" y="{VB - ph}" width="{pw}" height="{ph}" fill="{PAPER}" stroke="{INK}" stroke-width="1.7"/>')
for i, ln in enumerate(("t1 = x0 + x5;", "t2 = x0 - x5;", "y0 = t1 + t3;", "y5 = w5 * t2;")):
    put(ln, 9.5, x=cx - pw / 2 + 10, baseline=VB - ph + 24 + i * 21, fname=CMTT)
leader(cx)

# ---------------- caption ----------------
put(f"Fig. {FIG}.  The OCaml codelet compiler: a DFT specification enters as an expression DAG and leaves as straight-line, register-allocated C.",
    17, cx=W / 2, baseline=468, fname=CMR)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="VectorFFT OCaml codelet compiler pipeline">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
<g transform="translate(0 -75)">
{"".join(E)}
{"".join(boxes)}
{"".join(L)}
</g>
</svg>'''
open("/home/claude/vectorfft-pipeline.svg", "w").write(svg)
print("ok", W, H)
