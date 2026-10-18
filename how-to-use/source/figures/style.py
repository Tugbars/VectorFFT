"""The guide's figure helper: usage figures (array layouts, which bin is which frequency, padded
rows, batch layouts) as self-contained SVG in the site's palette.

  ink #101010, muted #6a6a6a, rule #dcdcdc, on a white plate; thin rules; no shadows, no rounded
  corners. Two accents, each with one meaning through the whole book (owner, 2026-10-18):
    BLUE  #2457c5 (tint BLUE_T)  = yours / input / before: the samples you supply, the input
          buffer, the real part, the forward direction, a wisdom hit;
    RED   #c8322b (tint RED_T)   = the library's / output / after: the bins it writes, the output
          buffer, the imaginary part, the backward direction, a miss that measures.
  Grey stays for not read / padding / not stored. A tint fills a cell, the full colour strokes
  and letters it. Colour never carries meaning alone: a coloured cell also carries a word, a
  glyph or a hatch, so greyscale print and colour-blind readers lose nothing.

Drawn at print size: 1 unit = 1 pt. The full width W = 396.85 pt is the 140 mm text frame of the
A4 edition, so lettering is 8 pt in print and about 13 px on a screen. Every glyph is outlined
in Computer Modern through matplotlib (cmr10 for words, cmtt10 for code names, mathtext in the
cm fontset for $...$), so a figure carries no font and no <text> element.

A figure script is figures/<chapter key>_<name>.py; build.py runs it when its SVG is missing or
older than the script or this file, and the script writes figures/<chapter key>_<name>.svg:

    import style
    f = style.Figure(height=90)
    xs = f.cells(40, 20, 8, w=36, h=18, labels=["$x[%d]$" % n for n in range(8)])
    f.save()

Coordinates: x to the right, y downwards, both in pt from the top-left corner of the plate.
The saved figure keeps the full width W and is cropped to the drawing: PAD pt above its topmost
and below its lowest ink, so the space between a figure and the text is the same everywhere;
height only sizes the plate of a figure that draws nothing.
Text: s is words in cmr10 with $...$ for math (font="rm"), or code in cmtt10 (font="tt");
y is the baseline; anchor "l", "c" or "r" places x at the left, centre or right of the ink.
Run with Python 3.13 and matplotlib (py -3.13 -E figures/<name>.py).
"""
import inspect
import math
import os
from pathlib import Path

import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

matplotlib.rcParams.update({"mathtext.fontset": "cm", "svg.hashsalt": "vectorfft-guide"})

W = 396.85                                   # pt: the 140 mm text frame
INK, MUTED, RULE, PAPER = "#101010", "#6a6a6a", "#dcdcdc", "#ffffff"
BLUE, BLUE_T = "#2457c5", "#dbe4f7"           # yours / input / before; the tint fills a cell
RED, RED_T = "#c8322b", "#f6dcda"             # the library's / output / after
TEXT = 8.0                                   # pt: lettering
LINE = 0.6                                   # pt: cell borders, leaders
THIN = 0.4                                   # pt: hatching, ticks
DART_L, DART_W = 5.0, 3.0                    # pt: the filled dart of an arrow
PAD = 6.0                                    # pt: the plate above and below the drawing
_TTF = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
_FONTS = {"rm": FontProperties(fname=os.path.join(_TTF, "cmr10.ttf")),
          "tt": FontProperties(fname=os.path.join(_TTF, "cmtt10.ttf"))}
_RM_BAD = set('<>{}_|\\"')                    # cmr10 sets these as other glyphs: use math or tt


def _fmt(v):
    s = "%.2f" % v
    return s.rstrip("0").rstrip(".") if "." in s else s


class Figure:
    def __init__(self, height, width=W):
        self.w, self.h = width, height
        self.items = []
        self.defs = {}
        self.top = self.bottom = None           # the drawing's extent in y, for the crop

    def _ink(self, y0, y1):
        lo, hi = min(y0, y1), max(y0, y1)
        self.top = lo if self.top is None else min(self.top, lo)
        self.bottom = hi if self.bottom is None else max(self.bottom, hi)

    # ── primitives ──────────────────────────────────────────────────────────────────────────
    def text(self, s, x, y, size=TEXT, anchor="c", color=INK, font="rm"):
        """Outlined text with its baseline at y. Returns (left, right) of the ink."""
        if font == "rm":
            outside = "".join(part for i, part in enumerate(s.split("$")) if i % 2 == 0)
            bad = sorted(set(outside) & _RM_BAD)
            if bad:
                raise ValueError("%r: cmr10 has no %s; put it in $...$ or use font='tt'" % (s, "".join(bad)))
        tp = TextPath((0, 0), s, size=size, prop=_FONTS[font])
        v, codes = tp.vertices, tp.codes
        if not len(v):
            return x, x
        x0, x1 = v[:, 0].min(), v[:, 0].max()
        self._ink(y - v[:, 1].max(), y - v[:, 1].min())
        tx = {"c": x - (x0 + x1) / 2, "l": x - x0, "r": x - x1}[anchor]
        d, i = [], 0
        while i < len(codes):
            k = codes[i]
            if k == 1:
                d.append("M%s %s" % (_fmt(v[i][0]), _fmt(-v[i][1])))
                i += 1
            elif k == 2:
                d.append("L%s %s" % (_fmt(v[i][0]), _fmt(-v[i][1])))
                i += 1
            elif k == 3:
                d.append("Q%s %s %s %s" % (_fmt(v[i][0]), _fmt(-v[i][1]), _fmt(v[i + 1][0]), _fmt(-v[i + 1][1])))
                i += 2
            elif k == 4:
                d.append("C%s %s %s %s %s %s" % (_fmt(v[i][0]), _fmt(-v[i][1]), _fmt(v[i + 1][0]),
                                                 _fmt(-v[i + 1][1]), _fmt(v[i + 2][0]), _fmt(-v[i + 2][1])))
                i += 3
            else:
                d.append("Z")
                i += 1
        self.items.append('<path transform="translate(%s %s)" fill="%s" d="%s"/>'
                          % (_fmt(tx), _fmt(y), color, " ".join(d)))
        return tx + x0, tx + x1

    def line(self, x1, y1, x2, y2, lw=LINE, color=INK, dash=None):
        da = ' stroke-dasharray="%s"' % " ".join(_fmt(t) for t in dash) if dash else ""
        self._ink(min(y1, y2) - lw / 2, max(y1, y2) + lw / 2)
        self.items.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s"%s/>'
                          % (_fmt(x1), _fmt(y1), _fmt(x2), _fmt(y2), color, _fmt(lw), da))

    def rect(self, x, y, w, h, fill=None, stroke=INK, lw=LINE, hatch=False, dash=None):
        """fill: None (paper), "rule" (the light grey) or a colour. hatch: thin diagonal lines."""
        f = {None: PAPER, "rule": RULE, "muted": MUTED}.get(fill, fill)
        self._ink(y - (lw / 2 if stroke else 0), y + h + (lw / 2 if stroke else 0))
        self.items.append('<rect x="%s" y="%s" width="%s" height="%s" fill="%s"/>'
                          % (_fmt(x), _fmt(y), _fmt(w), _fmt(h), f))
        if hatch:
            self.defs["hatch"] = ('<pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" '
                                  'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="4" '
                                  'stroke="%s" stroke-width="%s"/></pattern>' % (MUTED, _fmt(THIN)))
            self.items.append('<rect x="%s" y="%s" width="%s" height="%s" fill="url(#hatch)"/>'
                              % (_fmt(x), _fmt(y), _fmt(w), _fmt(h)))
        if stroke:
            da = ' stroke-dasharray="%s"' % " ".join(_fmt(t) for t in dash) if dash else ""
            self.items.append('<rect x="%s" y="%s" width="%s" height="%s" fill="none" stroke="%s" '
                              'stroke-width="%s"%s/>' % (_fmt(x), _fmt(y), _fmt(w), _fmt(h), stroke, _fmt(lw), da))

    def arrow(self, x1, y1, x2, y2, lw=LINE, color=INK):
        """A line from (x1, y1) with a filled dart at (x2, y2)."""
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - DART_L * 0.72 * math.cos(ang), y2 - DART_L * 0.72 * math.sin(ang)
        self._ink(y2 - DART_L, y2 + DART_L)
        self.line(x1, y1, bx, by, lw, color)
        self.items.append('<path transform="translate(%s %s) rotate(%s)" fill="%s" d="M0 0 L%s %s L%s 0 L%s %s Z"/>'
                          % (_fmt(x2), _fmt(y2), _fmt(math.degrees(ang)), color, _fmt(-DART_L),
                             _fmt(-DART_W / 2), _fmt(-DART_L * 0.72), _fmt(-DART_L), _fmt(DART_W / 2)))

    def brace(self, x1, x2, y, label=None, below=True, depth=5.0, color=INK, font="rm"):
        """A curly brace spanning x1..x2 whose back is at y, opening away from the label;
        the label sits beyond its tip."""
        s = 1 if below else -1
        xm = (x1 + x2) / 2
        q = depth / 2
        d = ("M{a} {y0} Q{a} {y1} {b} {y1} L{c} {y1} Q{m} {y1} {m} {y2} Q{m} {y1} {e} {y1} "
             "L{f} {y1} Q{g} {y1} {g} {y0}").format(
            a=_fmt(x1), b=_fmt(x1 + q), c=_fmt(xm - q), m=_fmt(xm), e=_fmt(xm + q), f=_fmt(x2 - q),
            g=_fmt(x2), y0=_fmt(y), y1=_fmt(y + s * q), y2=_fmt(y + s * depth))
        self.items.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s"/>' % (d, color, _fmt(LINE)))
        self._ink(y - LINE / 2, y + s * depth + s * LINE / 2)
        if label:
            ly = y + s * depth + (TEXT + 2.5 if below else -3.0)
            self.text(label, xm, ly, color=color, font=font)

    # ── composites ──────────────────────────────────────────────────────────────────────────
    def cells(self, x, y, n, w=30.0, h=18.0, labels=None, fills=None, hatches=None, index=None,
              font="rm", label_color=None, dash=None, strokes=None):
        """A row of n cells sharing their borders, top-left at (x, y). labels: the text in each
        cell; fills: None, "rule", BLUE_T, ... per cell; strokes: the border colour, one or one
        per cell (a tinted cell is stroked in its colour); hatches: True per cell; index: the text
        under each cell (an index). Returns the cells' centre x positions."""
        xs = []
        for i in range(n):
            cx = x + i * w
            fill = fills[i] if fills else None
            stroke = (strokes[i] if isinstance(strokes, (list, tuple)) else strokes) or INK
            self.rect(cx, y, w, h, fill=fill, stroke=stroke, hatch=bool(hatches and hatches[i]), dash=dash)
            xs.append(cx + w / 2)
            if labels and labels[i]:
                col = (label_color[i] if isinstance(label_color, (list, tuple)) else label_color) or INK
                self.text(labels[i], cx + w / 2, y + h / 2 + TEXT * 0.33, color=col, font=font)
            if index and index[i] is not None:
                self.text(index[i], cx + w / 2, y + h + TEXT + 2.5, color=MUTED, font="rm")
        return xs

    def grid(self, x, y, rows, cols, w=24.0, h=16.0, label=None, fill=None, font="rm"):
        """rows x cols cells, top-left at (x, y); label(r, c) and fill(r, c) give each cell's
        text and fill. Returns (x0, y0, x1, y1) of the grid."""
        for r in range(rows):
            for c in range(cols):
                f = fill(r, c) if fill else None
                self.rect(x + c * w, y + r * h, w, h, fill=f)
                if label:
                    t = label(r, c)
                    if t:
                        self.text(t, x + c * w + w / 2, y + r * h + h / 2 + TEXT * 0.33, font=font)
        return x, y, x + cols * w, y + rows * h

    # ── output ──────────────────────────────────────────────────────────────────────────────
    def save(self, path=None):
        """Write the SVG; by default next to the calling script, with the script's name."""
        if path is None:
            caller = Path(inspect.stack()[1].filename).resolve()
            path = caller.with_suffix(".svg")
        defs = "<defs>%s</defs>\n" % "".join(self.defs.values()) if self.defs else ""
        top, h = (0.0, self.h) if self.top is None else (self.top - PAD, self.bottom - self.top + 2 * PAD)
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 %s %s %s" width="%spt" height="%spt">\n'
               '%s<rect y="%s" width="%s" height="%s" fill="%s"/>\n%s\n</svg>\n'
               % (_fmt(top), _fmt(self.w), _fmt(h), _fmt(self.w), _fmt(h), defs, _fmt(top), _fmt(self.w),
                  _fmt(h), PAPER, "\n".join(self.items)))
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg)
        return path
