"""build.py -- the machinery tiling sheets, docs/architecture/machinery/tiling_{1d,2d,3d}.html, from the sources beside
this script: tiling_{1d,2d,3d}.html (each page, with /*@FONTS@*/ and /*@CSS@*/ where the faces and the stylesheet go)
and sheet.css (the user guide's look, how-to-use/source/tools/guide.css).

The faces are the guide's vendored Computer Modern Unicode (how-to-use/source/vendor/cmu-0.7.0, SIL Open Font
License), subset to the characters the three pages use and embedded as data: URIs, so each page is one file that opens
anywhere. CMU has no <= or >= glyph: those two fall back to the reader's serif.

    py -3 docs/architecture/machinery/source/build.py        (needs fontTools: pip install fonttools)

The sheets are written by hand from the code and from measurements; src/tools/archgraph.py does not generate them.
"""
import base64
import html
import io
import os
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

SRC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(SRC)
ROOT = os.path.normpath(os.path.join(SRC, '..', '..', '..', '..'))
CMU = os.path.join(ROOT, 'how-to-use', 'source', 'vendor', 'cmu-0.7.0')
PAGES = ('tiling_1d.html', 'tiling_2d.html', 'tiling_3d.html')
FACES = (('CMU Serif', 400, 'normal', 'cmunrm.woff'),
         ('CMU Serif', 700, 'normal', 'cmunbx.woff'),
         ('CMU Serif', 400, 'italic', 'cmunti.woff'),
         ('CMU Serif', 700, 'italic', 'cmunbi.woff'),
         ('CMU Typewriter Text', 400, 'normal', 'cmuntt.woff'))


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def main():
    if not os.path.isdir(CMU):
        sys.exit('build.py: the Computer Modern faces are not at %s' % CMU)
    css = read(os.path.join(SRC, 'sheet.css'))
    pages = {name: read(os.path.join(SRC, name)) for name in PAGES}
    chars = set(chr(c) for c in range(0x20, 0x7f)) | set(' –—’“”')
    for text in pages.values():
        chars |= set(html.unescape(text))
    cps = sorted(ord(c) for c in chars if ord(c) >= 0x20)

    rules = []
    for family, weight, style, file in FACES:
        font = TTFont(os.path.join(CMU, file))
        opts = subset.Options()
        opts.flavor = 'woff'
        opts.layout_features = ['*']
        opts.notdef_outline = True
        sub = subset.Subsetter(options=opts)
        sub.populate(unicodes=cps)
        sub.subset(font)
        buf = io.BytesIO()
        font.flavor = 'woff'
        font.save(buf)
        rules.append("@font-face{font-family:'%s';font-weight:%d;font-style:%s;font-display:swap;"
                     "src:url(data:font/woff;base64,%s) format('woff')}"
                     % (family, weight, style, base64.b64encode(buf.getvalue()).decode('ascii')))
    fonts = '\n'.join(rules)

    for name in PAGES:
        text = pages[name]
        if '/*@FONTS@*/' not in text or '/*@CSS@*/' not in text:
            sys.exit('build.py: %s lacks the /*@FONTS@*/ or /*@CSS@*/ mark' % name)
        out = os.path.join(OUT, name)
        with open(out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text.replace('/*@FONTS@*/', fonts).replace('/*@CSS@*/', css))
        print('%s  %d bytes' % (os.path.relpath(out, ROOT), os.path.getsize(out)))


if __name__ == '__main__':
    main()
