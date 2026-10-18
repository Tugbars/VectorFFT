#!/usr/bin/env python3
"""Build the VectorFFT User Guide.

Markdown chapters (the dialect of STYLE.md) become one self-contained HTML book (fonts, math,
figures inlined; no JavaScript) with a contents sidebar, and an A4 PDF that headless Chrome
prints from the print edition of the same pages. Every number (sections, equations, figures,
tables, cross-references, the contents' page numbers) is assigned here and written into the
HTML as text, so the two editions cannot disagree.

usage (from how-to-use/source, Python 3.13):
  py -3.13 tools/build.py                   the book: ../VectorFFT-Guide.html and ../VectorFFT-Guide.pdf
  py -3.13 tools/build.py --chapter ch07    one chapter: _build/ch07.html (add --pdf for _build/ch07.pdf)
options:
  --book FILE              the manifest (default book.yml)
  --draft                  chapters whose file does not exist yet become placeholders
  --allow-missing-output   a listing the harness has not run shows a placeholder for its output
  --no-pdf                 the book: write the HTML only
  --pdf                    a chapter: print it to PDF as well
  --out-dir DIR            write everything to DIR instead of the defaults
  --chrome PATH            the Chrome executable (default: found in the usual places)
"""
import argparse
import base64
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True    # no __pycache__ beside the sources
import yaml
from markdown_it import MarkdownIt
from markdown_it.renderer import RendererHTML
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Comment
from pygments.util import ClassNotFound

HERE = Path(__file__).resolve().parent            # how-to-use/source/tools
SOURCE = HERE.parent                              # how-to-use/source
GUIDE = SOURCE.parent                             # how-to-use
EXAMPLES = GUIDE / "examples"
LISTINGS = SOURCE / "listings"
OUTS = LISTINGS / "out"
FIGURES = SOURCE / "figures"
VENDOR = SOURCE / "vendor"
KATEX = VENDOR / "katex-0.16.21" / "katex.js"
FONTS = VENDOR / "cmu-0.7.0"

sys.path.insert(0, str(HERE))
import mdext            # noqa: E402
import privacy          # noqa: E402

ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}
PH_O, PH_C = "\ue000", "\ue001"                    # private-use code points: never in a source
RE_PH = re.compile("\ue000(\\w):([^\ue001]*)\ue001")
RE_FILE = re.compile(r"^(fm\d|ch\d\d|app[A-Z])_[a-z0-9-]+\.md$")
RE_REGION_BEGIN = re.compile(r"^\s*/\*\s*region\s+([a-z0-9_-]+)\s*\*/\s*$")
RE_REGION_END = re.compile(r"^\s*/\*\s*end\s+region\s+([a-z0-9_-]+)\s*\*/\s*$")
RE_HARNESS = re.compile(r"^\s*/\*\s*harness:.*\*/\s*$")
DIRECTIVE_OPTIONS = {"figure": {"name", "alt"}, "table": {"name"}, "note": set(), "warning": set(),
                     "steps": set(), "listing": {"region"}, "output": {"stream"}, "api": set()}
CODE_LANGS = {"sh", "cmake", "powershell", "text", "ini", "json"}   # shown as written
MAX_COLUMNS = 80
SHORT_LINES = 15            # a code block of this many lines or fewer is never split across pages
LONG_TABLE_ROWS = 8         # a table of this many body rows, or of more text, may break between rows
LONG_TABLE_CHARS = 1000
RE_TABLE_ID = re.compile(r'<code class="nw">([A-Za-z0-9_]{20,})</code>')   # may break after "_"
CONT = (",", "(", "\\", "&&", "||", "+", "=", "?")   # a code line ending so goes on in the next


class BuildError(Exception):
    pass


def ph(kind, arg):
    return "{}{}:{}{}".format(PH_O, kind, arg, PH_C)


# ── the manifest ─────────────────────────────────────────────────────────────────────────────

class Part:
    def __init__(self, n, title):
        self.key, self.n, self.roman, self.title = "part%d" % n, n, ROMAN[n], title


class Entry:
    """One chapter file: front matter (unnumbered), a chapter (1..19) or an appendix (A, B)."""

    def __init__(self, kind, path, key, planned, number, prefix, part):
        self.kind, self.path, self.key, self.planned = kind, path, key, planned
        self.number, self.prefix, self.part = number, prefix, part
        self.exists = path.exists()
        self.html = None
        self.title_text = planned
        self.title_html = html.escape(planned)

    @property
    def rel(self):
        try:
            return self.path.relative_to(SOURCE).as_posix()
        except ValueError:
            return self.path.as_posix()

    @property
    def word(self):
        return {"chapter": "Chapter", "appendix": "Appendix"}.get(self.kind, "")

    def running_head(self):
        if self.kind == "front":
            return self.title_text
        return "{} {}\u2003{}".format(self.word, self.number, self.title_text)


def load_manifest(path):
    y = yaml.safe_load(path.read_text(encoding="utf-8"))
    chapters_dir = (path.parent / y.get("chapters", "chapters")).resolve()
    entries, parts = [], []

    def entry(kind, item, number, prefix, part):
        if not isinstance(item, dict) or set(item) - {"file", "title", "key"} or not item.get("file") \
                or not isinstance(item.get("title"), str):
            raise BuildError("%s: an entry is {file: ..., title: \"...\"} (quote a title that holds "
                             "a comma or a colon): %r" % (path.name, item))
        f = item["file"]
        key = item.get("key")
        if key is None:
            m = RE_FILE.match(f)
            if not m:
                raise BuildError("%s: the file name %r is not fmN_name.md, chNN_name.md or appX_name.md"
                                 % (path.name, f))
            key = m.group(1).lower()
            if kind == "chapter" and int(key[2:]) != int(number):
                raise BuildError("%s: %s is chapter %s by its place in the manifest" % (path.name, f, number))
            if kind == "appendix" and key[3:].upper() != number:
                raise BuildError("%s: %s is appendix %s by its place in the manifest" % (path.name, f, number))
            if kind == "front" and not key.startswith("fm"):
                raise BuildError("%s: front matter files are named fmN_name.md: %s" % (path.name, f))
        e = Entry(kind, chapters_dir / f, key, item["title"], number, prefix, part)
        entries.append(e)
        return e

    for item in y.get("front", []) or []:
        entry("front", item, "", "0", None)
    chap, app = 0, 0
    for i, p in enumerate(y.get("parts", []) or [], 1):
        part = Part(i, p["title"])
        parts.append(part)
        for item in p.get("chapters", []) or []:
            chap += 1
            entry("chapter", item, str(chap), str(chap), part)
        for item in p.get("appendices", []) or []:
            letter = "ABCDEFGH"[app]
            app += 1
            entry("appendix", item, letter, letter, part)
    keys = [e.key for e in entries]
    if len(set(keys)) != len(keys):
        raise BuildError("%s: two entries have the same key" % path.name)
    meta = {k: y[k] for k in ("title", "wordmark", "subtitle", "edition", "date", "repository", "output") if k in y}
    return meta, parts, entries


# ── rendering ────────────────────────────────────────────────────────────────────────────────

class Label:
    def __init__(self, kind, number, anchor, owner, text=""):
        self.kind, self.number, self.anchor, self.owner, self.text = kind, number, anchor, owner, text


class Book:
    def __init__(self, opts):
        self.opts = opts
        self.labels = {}            # label -> Label
        self.math = []              # (tex, display, where)
        self.toc = {}               # entry key -> [(level, number, title_html, anchor)]
        self.listings, self.outputs, self.excerpts, self.figures = [], [], [], []
        self.warnings = []

    def register(self, label, kind, number, anchor, owner, text="", where=None):
        if label in self.labels:
            raise BuildError("%sthe label %r is defined twice" % (where + ": " if where else "", label))
        self.labels[label] = Label(kind, number, anchor, owner, text)

    def warn(self, msg):
        self.warnings.append(msg)


class Ctx:
    """Per-file render state."""

    def __init__(self, book, entry):
        self.book, self.entry = book, entry
        self.sec = self.sub = 0
        self.n = {"eq": 0, "fig": 0, "tbl": 0}
        self.group = None
        self.pending = None
        self.line = 1
        self.nested = 0
        self.seen_title = False
        book.toc[entry.key] = []

    def where(self):
        return "%s:%d" % (self.entry.rel, self.line)

    def fail(self, msg):
        raise BuildError("%s: %s" % (self.where(), msg))

    def next(self, kind):
        self.n[kind] += 1
        return "{}.{}".format(self.entry.prefix, self.n[kind])

    def check_label(self, label):
        if not re.fullmatch(mdext.LABEL, label):
            self.fail("a label is lower-case letters, digits and hyphens: %r" % label)
        if not label.startswith(self.entry.key + "-"):
            self.fail("a label in this file starts with %r: %r" % (self.entry.key + "-", label))

    def math_ph(self, tex, display):
        self.book.math.append((tex.strip(), display, self.where()))
        return ph("M", str(len(self.book.math) - 1))


class Renderer(RendererHTML):
    """Keeps the source line of the block being rendered, for error messages."""

    def render(self, tokens, options, env):
        ctx = env["ctx"]
        out = []
        for i, tok in enumerate(tokens):
            if tok.map and not ctx.nested:
                ctx.line = tok.map[0] + 1
            if tok.type == "inline":
                out.append(self.renderInline(tok.children or [], options, env))
            elif tok.type in self.rules:
                out.append(self.rules[tok.type](tokens, i, options, env))
            else:
                out.append(self.renderToken(tokens, i, options, env))
        return "".join(out)


def highlight(src, lang):
    """Monochrome, as on the site: comments muted and upright, nothing else coloured."""
    try:
        lexer = get_lexer_by_name(lang)
    except ClassNotFound:
        return html.escape(src, quote=False).rstrip("\n")
    out = []
    for ttype, value in lex(src, lexer):
        esc = html.escape(value, quote=False)
        out.append('<span class="c">{}</span>'.format(esc) if ttype in Comment and ttype not in
                   (Comment.Preproc, Comment.PreprocFile) else esc)
    return "".join(out).rstrip("\n")


def split_condition(tex):
    """'formula, \\qquad condition' at the top level -> (formula, condition), else (tex, None)."""
    depth, i = 0, 0
    while i < len(tex):
        if tex[i] == "\\" and tex[i + 1:i + 2] in ("{", "}"):
            i += 2
            continue
        if tex[i] == "{" or tex.startswith("\\begin", i):
            depth += 1
        elif tex[i] == "}" or tex.startswith("\\end", i):
            depth -= 1
        elif depth == 0 and tex.startswith("\\qquad", i):
            return tex[:i].rstrip(), tex[i + len("\\qquad"):].strip()
        i += 1
    return tex, None


def parse_directive(ctx, tok):
    m = re.match(r"\{([a-z]+)\}\s*(.*)$", tok.info.strip())
    if not m:
        ctx.fail("a directive starts with :::{name}")
    name, arg = m.group(1), m.group(2).strip()
    if name not in DIRECTIVE_OPTIONS:
        ctx.fail("unknown directive {%s}; the directives are %s" % (name, ", ".join(sorted(DIRECTIVE_OPTIONS))))
    opts, body, head = {}, [], True
    for line in tok.content.splitlines():
        om = re.match(r"^:([a-z]+):\s*(.*)$", line) if head else None
        if om:
            if om.group(1) not in DIRECTIVE_OPTIONS[name]:
                ctx.fail("{%s} takes no option :%s:" % (name, om.group(1)))
            opts[om.group(1)] = om.group(2).strip()
            continue
        head = False
        body.append(line)
    return name, arg, opts, "\n".join(body).strip("\n")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def listing_lines(path):
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def shown_listing(ctx, path, region):
    """The text a chapter shows: marker and harness lines removed; a region if one is named."""
    lines = listing_lines(path)
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if region:
        begins = [i for i, l in enumerate(lines) if (RE_REGION_BEGIN.match(l) or [None, None])[1] == region]
        ends = [i for i, l in enumerate(lines) if (RE_REGION_END.match(l) or [None, None])[1] == region]
        if len(begins) != 1 or len(ends) != 1 or ends[0] < begins[0]:
            ctx.fail("%s: the region %r needs exactly one /* region %s */ and one later "
                     "/* end region %s */" % (path.name, region, region, region))
        lines = lines[begins[0] + 1:ends[0]]
    keep = [l for l in lines if not (RE_REGION_BEGIN.match(l) or RE_REGION_END.match(l) or RE_HARNESS.match(l))]
    while keep and not keep[0].strip():
        keep.pop(0)
    while keep and not keep[-1].strip():
        keep.pop()
    return "\n".join(keep)


def run_record(ctx, stem, need_output):
    """The harness's record of a listing; checks that the output was made from this source."""
    o = ctx.book.opts
    src = None
    for cand in (LISTINGS / (stem + ".c"), LISTINGS / (stem + ".cpp"), EXAMPLES / (stem + ".c")):
        if cand.exists():
            src = cand
            break
    if src is None:
        ctx.fail("no listing listings/%s.c and no example examples/%s.c" % (stem, stem))
    rec_path = OUTS / (stem + ".json")
    if not rec_path.exists():
        if o.allow_missing_output:
            ctx.book.warn("%s: %s has not been run by the harness" % (ctx.where(), src.name))
            return None
        ctx.fail("%s has not been run by the harness (tools/listings.py); --allow-missing-output "
                 "builds without its output" % src.name)
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if rec.get("source_sha256") != sha256_file(src):
        msg = "%s changed since the harness last ran it: run tools/listings.py again" % src.name
        if o.allow_missing_output:
            ctx.book.warn("%s: %s" % (ctx.where(), msg))
        else:
            ctx.fail(msg)
    return rec


def read_output(ctx, stem, stream):
    rec = run_record(ctx, stem, True)
    path = OUTS / (stem + (".out" if stream == "stdout" else ".err"))
    if rec is None or not path.exists():
        if ctx.book.opts.allow_missing_output:
            return None
        ctx.fail("no captured %s for %s" % (stream, stem))
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
    bad = privacy.findings(text)
    if bad:
        ctx.fail("%s carries %s: the harness should have refused it" % (path.name, bad[0]))
    return text


def verify_excerpt(ctx, fname, text):
    """An excerpt of how-to-use/examples/<fname>: each piece between '...' lines must occur in
    the file verbatim (trailing blanks aside), in order. Returns the 1-based line ranges."""
    path = EXAMPLES / fname
    if not path.exists() or "/" in fname or "\\" in fname:
        ctx.fail("excerpt=%s: no such file in how-to-use/examples/" % fname)
    file_lines = [l.rstrip() for l in listing_lines(path)]
    pieces, cur = [], []
    for l in text.rstrip("\n").split("\n"):
        if l.strip() == "...":
            pieces.append(cur)
            cur = []
        else:
            cur.append(l.rstrip())
    pieces.append(cur)
    if any(not p for p in pieces):
        ctx.fail("excerpt=%s: an empty piece around '...'" % fname)
    at, ranges = 0, []
    for p in pieces:
        hit = next((i for i in range(at, len(file_lines) - len(p) + 1) if file_lines[i:i + len(p)] == p), None)
        if hit is None:
            first = p[0]
            near = next((i + 1 for i, l in enumerate(file_lines) if l == first), None)
            ctx.fail("excerpt=%s no longer matches the file%s: quote it again from the file"
                     % (fname, " (its first line is line %d)" % near if near else ""))
        ranges.append((hit + 1, hit + len(p)))
        at = hit + len(p)
    return ranges


def line_spans(body):
    """One <span class="ln"> per line of a code or output block, the newline kept inside it, so the
    text is unchanged: --i is the line's indentation, under which a wrapped line hangs, and "cont"
    marks a code line whose statement goes on in the next (the print keeps the two together). A
    comment that runs over several lines is closed and reopened at each newline."""
    body = re.sub(r'<span class="c">(.*?)</span>',
                  lambda m: '<span class="c">' + m.group(1).replace("\n", '</span>\n<span class="c">') + "</span>",
                  body, flags=re.S)
    lines = body.split("\n")
    out = []
    for k, ln in enumerate(lines):
        text = html.unescape(re.sub(r"<[^>]+>", "", ln))
        ind = len(text) - len(text.lstrip(" "))
        plain = text.rstrip()
        cont = plain.endswith(CONT) and not plain.lstrip().startswith(("/*", "*", "//"))
        out.append('<span class="ln{}" style="--i:{}">{}{}</span>'.format(
            " cont" if cont else "", ind, ln, "\n" if k < len(lines) - 1 else ""))
    return "".join(out)


def code_block(label_html, text, lang, elide=False):
    body = highlight(text + "\n", lang) if lang else html.escape(text, quote=False)
    if elide:
        body = re.sub(r"(?m)^(\s*)\.\.\.$", r'\1<span class="elide">...</span>', body)
    head = '<p class="file">{}</p>'.format(label_html) if label_html else ""
    cls = "listing short" if text.count("\n") < SHORT_LINES else "listing"
    return '<div class="{}">{}<pre class="code"><code>{}</code></pre></div>\n'.format(cls, head, line_spans(body))


def table_cell_breaks(tab):
    """An identifier of 20 characters or more in a table cell may break after each "_" (on screen)."""
    def cell(m):
        inner = RE_TABLE_ID.sub(lambda c: '<code class="nw">' + c.group(1).replace("_", "_<wbr>") + "</code>",
                                m.group(3))
        return "<{0}{1}>{2}</{0}>".format(m.group(1), m.group(2), inner)
    return re.sub(r"<(t[hd])((?:\s[^>]*)?)>(.*?)</\1>", cell, tab, flags=re.S)


def make_md(book):
    md = mdext.dialect(MarkdownIt("commonmark", {"html": False, "typographer": True},
                                  renderer_cls=Renderer)
                       .enable("table").enable("smartquotes").disable("replacements"))

    def C(env):
        return env["ctx"]

    def nested_render(env, text, inline=False):
        c = C(env)
        c.nested += 1
        try:
            return md.renderInline(text, env) if inline else md.render(text, env)
        finally:
            c.nested -= 1

    def r_target(self, tokens, idx, options, env):
        c = C(env)
        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
        if nxt is None or nxt.type != "heading_open" or nxt.tag == "h1":
            c.fail("a target line (label)= stands directly before a ## or ### heading")
        c.pending = tokens[idx].content
        return ""

    def plain(inline_tok):
        return "".join(ch.content for ch in (inline_tok.children or [])
                       if ch.type in ("text", "code_inline", "softbreak")).strip()

    def r_heading_open(self, tokens, idx, options, env):
        c, b = C(env), env["book"]
        e = c.entry
        level = int(tokens[idx].tag[1])
        inline = tokens[idx + 1]
        if c.nested:
            c.fail("headings cannot stand inside a directive")
        if level == 1:
            if c.seen_title:
                c.fail("a file has one # title")
            c.seen_title = True
            e.title_text = plain(inline)
            e.title_html = nested_render(env, inline.content, inline=True)
            if e.title_text != e.planned:
                b.warn("%s: the title %r differs from the manifest's %r" % (c.where(), e.title_text, e.planned))
            b.labels[e.key].text = e.title_html          # a front matter reference shows the title
            if e.kind == "front":
                return '<h2 class="front-title">'
            return '<h2 class="chapter-title"><span class="num">{}</span> '.format(e.number)
        if not c.seen_title:
            c.fail("a file starts with its # title")
        if level == 2:
            c.sec += 1
            c.sub = 0
            num = "" if e.kind == "front" else "{}.{}".format(e.prefix, c.sec)
            auto = "{}-s{}".format(e.key, c.sec)
        elif level == 3:
            if not c.sec:
                c.fail("a ### heading comes after a ## heading")
            c.sub += 1
            num = "" if e.kind == "front" else "{}.{}.{}".format(e.prefix, c.sec, c.sub)
            auto = "{}-s{}-{}".format(e.key, c.sec, c.sub)
        else:
            c.fail("the guide uses #, ## and ### headings only")
        anchor = c.pending or auto
        if c.pending:
            c.check_label(c.pending)
        c.pending = None
        title_html = nested_render(env, inline.content, inline=True)
        b.register(anchor, "sec", num, anchor, e.key, title_html, where=c.where())
        b.toc[e.key].append((level, num, title_html, anchor))
        numspan = '<span class="num">{}</span> '.format(num) if num else ""
        return '<h{0} id="{1}">{2}'.format(level + 1, anchor, numspan)

    def r_heading_close(self, tokens, idx, options, env):
        return "</h{}>\n".format(int(tokens[idx].tag[1]) + 1)

    def r_math_inline(self, tokens, idx, options, env):
        return C(env).math_ph(tokens[idx].content, False)

    def r_math_block(self, tokens, idx, options, env):
        return '<div class="eq"><span class="math">{}</span></div>\n'.format(
            C(env).math_ph(tokens[idx].content, True))

    def r_math_block_eqno(self, tokens, idx, options, env):
        c, b = C(env), env["book"]
        label = tokens[idx].info.strip()
        c.check_label(label)
        lead, group, tex, narrow = "", None, [], None
        for line in tokens[idx].content.splitlines():
            m = re.match(r"\s*%(lead|group|narrow):\s*(.*)$", line)
            if m and m.group(1) == "lead":
                lead = m.group(2).strip()
            elif m and m.group(1) == "narrow":
                narrow = []
            elif m:
                group = m.group(2).strip()
                c.check_label(group)
            elif narrow is not None:
                narrow.append(line)
            else:
                tex.append(line)
        if group and c.group and c.group[0] == group:
            name, base, letter = c.group
            num = base + letter
            c.group = (name, base, chr(ord(letter) + 1))
        elif group:
            base = c.next("eq")
            b.register(group, "eq", base, label, c.entry.key, where=c.where())
            num, c.group = base + "a", (group, base, "b")
        else:
            num, c.group = c.next("eq"), None
        b.register(label, "eq", num, label, c.entry.key, where=c.where())
        lead_html = '<span class="lead">{}</span>'.format(html.escape(lead)) if lead else ""
        formula, cond = split_condition("\n".join(tex))
        math = (c.math_ph(formula, True) if cond is None else
                '<span class="parts"><span class="f">{}</span><span class="cond">{}</span></span>'.format(
                    c.math_ph(formula, True), c.math_ph(cond, True)))
        if narrow is not None:
            math = '<span class="wide">{}</span><span class="narrow">{}</span>'.format(
                math, c.math_ph("\n".join(narrow), True))
        return ('<div class="eq{}" id="{}">{}<span class="math">{}</span>'
                '<span class="no">({})</span></div>\n'.format(
                    " has-lead" if lead else "", label, lead_html, math, num))

    def r_code_inline(self, tokens, idx, options, env):
        """Inline code never breaks inside a word; a long one (a command) breaks at spaces only."""
        s = tokens[idx].content
        if " " in s and len(s) > 32:
            return "<code>{}</code>".format(" ".join(
                '<span class="nw">{}</span>'.format(html.escape(w, quote=False)) for w in s.split(" ")))
        return '<code class="nw">{}</code>'.format(html.escape(s, quote=False))

    def r_role(self, tokens, idx, options, env):
        c = C(env)
        name = tokens[idx].meta.get("name")
        arg = tokens[idx].content.strip()
        if name in ("eq", "numref", "ref"):
            return ph("R", "|".join((name, arg, c.entry.key, c.where())))
        if name == "output":
            return ph("O", "|".join((arg, c.entry.key, c.where())))
        if name == "status":
            out = mdext.status_html(arg)
            if out is None:
                c.fail("{status} takes served, difference or refused, not %r" % arg)
            return out
        c.fail("unknown role {%s}; the roles are {ref}, {numref}, {eq}, {output} and {status}" % name)

    def r_fence(self, tokens, idx, options, env):
        c, b = C(env), env["book"]
        tok = tokens[idx]
        words = (tok.info or "text").split()
        lang = words[0]
        attrs = {}
        for w in words[1:]:
            if "=" not in w:
                c.fail("a code fence's info is a language and key=value attributes: %r" % tok.info)
            k, v = w.split("=", 1)
            attrs[k] = v
        text = tok.content.rstrip("\n")
        if lang in ("c", "cpp", "c++"):
            if set(attrs) != {"excerpt"}:
                c.fail("C code is shown from a listing (:::{listing}) or as an excerpt of an example "
                       "(```c excerpt=poisson3d.c); a C block typed into a chapter is not compiled")
            ranges = verify_excerpt(c, attrs["excerpt"], text)
            for n, l in enumerate(text.split("\n")):
                if len(l) > MAX_COLUMNS:
                    b.warn("%s: excerpt line %d is %d columns (the PDF wraps it)" % (c.where(), n + 1, len(l)))
            b.excerpts.append({"file": attrs["excerpt"], "lines": ranges, "where": c.where(), "text": text})
            span = ", ".join("%d" % a if a == z else "%d\u2013%d" % (a, z) for a, z in ranges)
            label = 'From <code class="nw">examples/{}</code>, line{} {}'.format(
                html.escape(attrs["excerpt"]), "" if ranges == [(ranges[0][0], ranges[0][0])] else "s", span)
            return code_block(label, text, "c", elide=True)
        if attrs:
            c.fail("only C blocks take attributes (excerpt=...)")
        if lang not in CODE_LANGS:
            c.fail("code blocks are %s, or C from a listing; got %r" % (", ".join(sorted(CODE_LANGS)), lang))
        for n, l in enumerate(text.split("\n")):
            if len(l) > MAX_COLUMNS:
                b.warn("%s: a %s line is %d columns (the PDF wraps it)" % (c.where(), lang, len(l)))
        if lang == "text":
            # what a program printed, quoted by hand: shown as an output, not as code to type
            return '<pre class="quote"><code>{}</code></pre>\n'.format(line_spans(html.escape(text, quote=False)))
        return code_block("", text, None if lang in ("json", "ini") else
                          {"sh": "bash"}.get(lang, lang))

    def r_colon_fence(self, tokens, idx, options, env):
        c, b = C(env), env["book"]
        name, arg, opts, body = parse_directive(c, tokens[idx])
        if name == "figure":
            label, alt = opts.get("name"), opts.get("alt")
            if not arg or not label or not alt or not body:
                c.fail("{figure} needs the SVG file, :name:, :alt: and a caption")
            c.check_label(label)
            svg = FIGURES / arg
            ensure_figure(b, svg, c)
            num = c.next("fig")
            b.register(label, "fig", num, label, c.entry.key, where=c.where())
            b.figures.append(arg)
            uri = "data:image/svg+xml;base64," + base64.b64encode(svg.read_bytes()).decode("ascii")
            return ('<figure class="figure" id="{0}"><div class="frame"><img src="{1}" alt="{2}"></div>'
                    '<figcaption><span class="label">Figure {3}</span> {4}</figcaption></figure>\n'
                    .format(label, uri, html.escape(alt), num, nested_render(env, body, inline=True)))
        if name == "table":
            label = opts.get("name")
            if not arg or not label:
                c.fail("{table} needs a caption after {table} and :name:")
            c.check_label(label)
            num = c.next("tbl")
            b.register(label, "tbl", num, label, c.entry.key, where=c.where())
            tab = nested_render(env, body)
            if tab.count("<table>") != 1 or not tab.lstrip().startswith("<table>"):
                c.fail("a {table} holds one pipe table and nothing else")
            rows = len(re.findall(r"<tr>", tab.split("<tbody>", 1)[1])) if "<tbody>" in tab else 0
            long_table = rows >= LONG_TABLE_ROWS or len(re.sub(r"<[^>]+>", "", tab)) > LONG_TABLE_CHARS
            tab = table_cell_breaks(tab)
            tab = tab.replace("<table>", '<table><caption><span class="label">Table {}</span> {}</caption>'
                              .format(num, nested_render(env, arg, inline=True)), 1)
            return '<div class="{}" id="{}">{}</div>\n'.format("table long" if long_table else "table", label, tab)
        if name in ("note", "warning"):
            if arg or not body:
                c.fail("{%s} takes its text in the body" % name)
            inner = nested_render(env, body)
            inner = inner.replace("<p>", '<p><span class="label">{}</span> '.format(
                "Note." if name == "note" else "Warning."), 1)
            return '<div class="{}">\n{}</div>\n'.format(name, inner)
        if name == "steps":
            inner = nested_render(env, body)
            if not inner.lstrip().startswith("<ol>"):
                c.fail("{steps} holds one numbered list (1. 2. 3.)")
            count = [0]

            def step(_m):
                count[0] += 1
                return '<li><span class="label">Step {}.</span> '.format(count[0])
            return re.sub(r"<li>", step, inner.replace("<ol>", '<ol class="steps">', 1))
        if name == "listing":
            path = LISTINGS / arg
            if not arg or path.suffix not in (".c", ".cpp") or not path.exists():
                c.fail("{listing} names a file in listings/: %r" % arg)
            if body:
                c.fail("{listing} has no body: the code comes from the file")
            rec = run_record(c, path.stem, False)
            text = shown_listing(c, path, opts.get("region"))
            for n, l in enumerate(text.split("\n")):
                if len(l) > MAX_COLUMNS:
                    c.fail("%s: a line is %d columns; listings keep to %d" % (path.name, len(l), MAX_COLUMNS))
                if "\t" in l:
                    c.fail("%s: a tab; listings indent with spaces" % path.name)
            b.listings.append({"file": arg, "region": opts.get("region"), "where": c.where(), "text": text,
                               "ran": rec is not None})
            label = ("From " if opts.get("region") else "") + '<code class="nw">{}</code>'.format(html.escape(arg))
            return code_block(label, text, "cpp" if path.suffix == ".cpp" else "c")
        if name == "output":
            stream = opts.get("stream", "stdout")
            if stream not in ("stdout", "stderr"):
                c.fail(":stream: is stdout or stderr")
            if not arg or body:
                c.fail("{output} names a listing (without .c) and has no body")
            text = read_output(c, arg, stream)
            if text is None:
                return '<pre class="output"><code>[the {} of {} has not been captured]</code></pre>\n'.format(
                    stream, html.escape(arg))
            for n, l in enumerate(text.split("\n")):
                if len(l) > MAX_COLUMNS:
                    b.warn("%s: %s.%s line %d is %d columns (the PDF wraps it)"
                           % (c.where(), arg, "out" if stream == "stdout" else "err", n + 1, len(l)))
            b.outputs.append({"name": arg, "stream": stream, "where": c.where(), "text": text})
            return '<pre class="output"><code>{}</code></pre>\n'.format(line_spans(html.escape(text, quote=False)))
        if name == "api":
            return api_block(c, env, body, nested_render)
        c.fail("unhandled directive %s" % name)

    md.add_render_rule("myst_target", r_target)
    md.add_render_rule("heading_open", r_heading_open)
    md.add_render_rule("heading_close", r_heading_close)
    md.add_render_rule("math_inline", r_math_inline)
    md.add_render_rule("math_block", r_math_block)
    md.add_render_rule("math_block_eqno", r_math_block_eqno)
    md.add_render_rule("myst_role", r_role)
    md.add_render_rule("code_inline", r_code_inline)
    md.add_render_rule("fence", r_fence)
    md.add_render_rule("code_block", lambda self, t, i, o, env: C(env).fail(
        "an indented code block: use a fenced block (```sh) or a {listing}"))
    md.add_render_rule("colon_fence", r_colon_fence)
    return md


def api_block(c, env, body, nested_render):
    """Signature lines, a blank line, then terms (a line in backticks) each followed by ': ...'
    definition lines; any other paragraph is ordinary text."""
    sig, _, rest = body.partition("\n\n")
    if not sig.strip():
        c.fail("{api} starts with the signature lines")
    lines = []
    for line in sig.strip().splitlines():
        esc = html.escape(line.strip(), quote=False)
        lines.append(re.sub(r"^(.*?)(\w+)(\()", r"\1<b>\2</b>\3", esc, count=1))
    out = ['<div class="api">', "".join('<p class="sig"><code>{}</code></p>'.format(l) for l in lines)]
    paras = [p for p in re.split(r"\n\s*\n", rest.strip()) if p.strip()]
    dl = []
    for p in paras:
        ls = p.split("\n")
        if len(ls) >= 2 and ls[1].startswith(": "):
            term = nested_render(env, ls[0].strip(), inline=True)
            defn = " ".join(l[2:].strip() if l.startswith(": ") else l.strip() for l in ls[1:])
            dl.append("<dt>{}</dt><dd>{}</dd>".format(term, nested_render(env, defn, inline=True)))
            continue
        if dl:
            out.append("<dl>{}</dl>".format("".join(dl)))
            dl = []
        out.append(nested_render(env, p))
    if dl:
        out.append("<dl>{}</dl>".format("".join(dl)))
    out.append("</div>\n")
    return "\n".join(out)


def ensure_figure(book, svg, c):
    """A figure's SVG is made by figures/<stem>.py; run it when the SVG is missing or older than
    the script or figures/style.py."""
    script = svg.with_suffix(".py")
    if script.exists():
        newest = max(script.stat().st_mtime, (FIGURES / "style.py").stat().st_mtime)
        if not svg.exists() or svg.stat().st_mtime < newest:
            r = subprocess.run([sys.executable, "-E", "-B", str(script)], cwd=str(FIGURES),
                               capture_output=True, text=True)
            if r.returncode != 0:
                c.fail("figures/%s failed:\n%s" % (script.name, r.stderr.strip()))
    if not svg.exists():
        c.fail("no figure figures/%s (and no figures/%s to make it)" % (svg.name, script.name))


def render_entry(md, book, e):
    src = e.path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if src.startswith("\ufeff"):
        src = src[1:]
    ctx = Ctx(book, e)
    if src.lstrip().startswith("---"):
        ctx.fail("chapter files carry no front matter: book.yml numbers them")
    env = {"book": book, "ctx": ctx}
    try:
        tokens = md.parse(src, env)
    except mdext.DialectError as err:
        raise BuildError("%s: %s" % (e.rel, err))
    first = next((t for t in tokens if t.type != "myst_target"), None)
    if first is None or first.type != "heading_open" or first.tag != "h1":
        ctx.line = (first.map[0] + 1) if first is not None and first.map else 1
        ctx.fail("a file starts with its # title")
    body = md.renderer.render(tokens, md.options, env)
    if ctx.pending:
        ctx.fail("a target line (label)= with no heading after it")
    if e.kind == "front":
        e.html = '<section class="front" id="{}">\n{}</section>\n'.format(e.key, body)
    else:
        e.html = '<section class="chapter" id="{}">\n{}</section>\n'.format(e.key, body)


def placeholder_html(e):
    if e.kind == "front":
        head = '<h2 class="front-title">{}</h2>'.format(html.escape(e.planned))
    else:
        head = '<h2 class="chapter-title"><span class="num">{}</span> {}</h2>'.format(e.number, html.escape(e.planned))
    cls = "front" if e.kind == "front" else "chapter"
    return '<section class="{}" id="{}">\n{}\n<p class="placeholder">Not written yet.</p>\n</section>\n'.format(
        cls, e.key, head)


def part_html(p):
    return ('<section class="part" id="{}"><h1 class="part-title"><span class="num">Part {}</span> {}</h1>'
            '</section>\n'.format(p.key, p.roman, html.escape(p.title)))


# ── math, references, outputs ────────────────────────────────────────────────────────────────

def render_math(book, indices):
    req = [{"tex": book.math[i][0], "display": book.math[i][1]} for i in indices]
    proc = subprocess.run(["node", str(HERE / "katex_render.js"), str(KATEX)],
                          input=json.dumps(req).encode("utf-8"), capture_output=True)
    out = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0:
        try:
            errs = json.loads(out)["errors"]
        except (ValueError, KeyError):
            raise BuildError("KaTeX failed:\n" + proc.stderr.decode("utf-8", "replace"))
        raise BuildError("TeX errors:\n" + "\n".join(
            "  %s: %s\n    in: %s" % (book.math[indices[x["index"]]][2], x["message"],
                                      book.math[indices[x["index"]]][0]) for x in errs))
    res = json.loads(out)
    return res["version"], dict(zip(indices, res["html"]))


def ref_text(role, lab):
    if role == "eq":
        return "({})".format(lab.number)
    if lab.kind == "fig":
        return "Figure&nbsp;" + lab.number
    if lab.kind == "tbl":
        return "Table&nbsp;" + lab.number
    if lab.kind == "sec":
        return "Section&nbsp;" + lab.number if lab.number else lab.text
    if lab.kind == "chapter":
        return "Chapter&nbsp;" + lab.number
    if lab.kind == "appendix":
        return "Appendix&nbsp;" + lab.number
    if lab.kind == "part":
        return "Part&nbsp;" + lab.number
    return lab.text                                     # front matter: its title


ROLE_KINDS = {"eq": {"eq"}, "numref": {"fig", "tbl"}, "ref": {"sec", "chapter", "appendix", "part", "front"}}


def resolve(text, book, rendered, linked, current, print_alt, strict):
    """Placeholders -> HTML. linked: the entry keys on this page (their labels are links)."""
    def sub(m):
        kind, arg = m.group(1), m.group(2)
        if kind == "M":
            h = rendered[int(arg)]
            if print_alt:   # the PDF's tags: one /Figure per formula, its TeX as /Alt
                label = html.escape(" ".join(book.math[int(arg)][0].split()), quote=True)
                h = '<span role="img" aria-label="{}">{}</span>'.format(
                    label, h.replace('class="katex-html" aria-hidden="true"', 'class="katex-html"'))
            return h
        if kind == "O":
            name, key, where = arg.split("|", 2)
            path = OUTS / (name + ".out")
            if not path.exists():
                if book.opts.allow_missing_output:
                    return "<code>[output of {}]</code>".format(html.escape(name))
                raise BuildError("%s: no captured output for {output}`%s`" % (where, name))
            out = path.read_text(encoding="utf-8").strip()
            if "\n" in out:
                raise BuildError("%s: {output}`%s` is for a one-line output; use :::{output}" % (where, name))
            if privacy.findings(out):
                raise BuildError("%s: %s.out carries %s" % (where, name, privacy.findings(out)[0]))
            if not any(o.get("inline") and o["where"] == where for o in book.outputs):
                book.outputs.append({"name": name, "stream": "stdout", "where": where, "text": out, "inline": True})
            return "<code>{}</code>".format(" ".join(
                '<span class="nw">{}</span>'.format(html.escape(w, quote=False)) for w in out.split(" ")))
        role, label, key, where = arg.split("|", 3)
        lab = book.labels.get(label)
        if lab is None:
            owner = label.split("-", 1)[0]
            if strict or owner == current:
                raise BuildError("%s: {%s}`%s` names no label" % (where, role, label))
            book.warn("%s: {%s}`%s` is not defined yet (its chapter is not written or does not build)"
                      % (where, role, label))
            return '<span class="ref-missing">[{}]</span>'.format(html.escape(label))
        if lab.kind not in ROLE_KINDS[role]:
            raise BuildError("%s: {%s}`%s` points at a %s; {eq} is for equations, {numref} for figures "
                             "and tables, {ref} for parts, chapters and sections" % (where, role, label, lab.kind))
        shown = ref_text(role, lab)
        if lab.owner in linked:
            return '<a class="ref" href="#{}">{}</a>'.format(lab.anchor, shown)
        return '<span class="ref-ext">{}</span>'.format(shown)
    for _ in range(3):          # a reference may show a title that holds math or code
        text = RE_PH.sub(sub, text)
        if PH_O not in text:
            return text
    raise BuildError("a placeholder is left in the page")


# ── the page ─────────────────────────────────────────────────────────────────────────────────

def data_uri(path, mime):
    return "data:{};base64,{}".format(mime, base64.b64encode(Path(path).read_bytes()).decode("ascii"))


def katex_css():
    css = ("/* KaTeX 0.16.21 CSS: MIT licence, (c) 2013-2020 Khan Academy and other contributors; "
           "the KaTeX fonts: SIL Open Font License 1.1, Reserved Font Names KaTeX_* */\n"
           + (KATEX.parent / "katex.min.css").read_text(encoding="utf-8"))

    def one(m):
        woff2 = re.search(r"url\((fonts/[^)]+\.woff2)\)", m.group(0)).group(1)
        return "src:url({}) format(\"woff2\")".format(data_uri(KATEX.parent / woff2, "font/woff2"))
    return re.sub(r"src:url\(fonts/[^;}]*", one, css)


CMU_FACES = [("CMU Serif", 400, "normal", "cmunrm"), ("CMU Serif", 700, "normal", "cmunbx"),
             ("CMU Serif", 400, "italic", "cmunti"), ("CMU Serif", 700, "italic", "cmunbi"),
             ("CMU Typewriter Text", 400, "normal", "cmuntt"), ("CMU Typewriter Text", 700, "normal", "cmuntb")]


def font_faces():
    rules = ["/* Computer Modern Unicode fonts 0.7.0 by Andrey V. Panov and the authors named in "
             "OFL.txt, SIL Open Font License 1.1 (https://openfontlicense.org): the upstream "
             "TrueType files wrapped in WOFF 1.0, every table unchanged */"]
    for fam, wt, st, stem in CMU_FACES:
        rules.append("@font-face{{font-family:'{}';font-weight:{};font-style:{};font-display:block;"
                     "src:url({}) format('woff')}}".format(fam, wt, st, data_uri(FONTS / (stem + ".woff"), "font/woff")))
    return "\n".join(rules)


def css_string(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\u2003", "\\2003 ") + '"'


def page_rules(entries):
    rules = ['@page contents{@top-left{content:"Contents"}}']
    for e in entries:
        rules.append("@page pg-{0}{{@top-left{{content:{1}}}}} #{0}{{page:pg-{0}}}".format(
            e.key, css_string(e.running_head())))
    return "\n".join(rules)


COLOPHON = ("Set in CMU Serif and CMU Typewriter Text (Computer Modern Unicode 0.7.0, SIL Open Font "
            "License 1.1). Mathematics set by KaTeX 0.16.21 (MIT licence) in the KaTeX fonts (SIL Open "
            "Font License 1.1).")

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{title}, {edition}">
<meta name="generator" content="how-to-use/source/tools/build.py (markdown-it-py {mdv}); KaTeX {kv} at build time">
<style>
{faces}
</style>
<style>
{katex}
</style>
<style>
{css}
{pages}
</style>
</head>
<body class="guide">
{nav}<main>
{titleblock}{contents}{body}{footer}</main>
</body>
</html>
"""


def toc_items(book, parts, entries, mode, pages=None):
    """mode 'nav': the screen sidebar; 'print': the contents pages with page-number slots."""
    pages = pages or {}
    items = []
    seen_parts = set()

    def pg(anchor):
        return '<span class="pg">{}</span>'.format(pages.get(anchor, "")) if mode == "print" else ""

    for e in entries:
        if e.part is not None and e.part.key not in seen_parts:
            seen_parts.add(e.part.key)
            p = e.part
            if mode == "nav":
                items.append('<li class="part-label"><a href="#{}"><span class="num">{}</span> {}</a></li>'
                             .format(p.key, p.roman, html.escape(p.title)))
            else:
                items.append('<li class="c-part"><a href="#{}"><span class="n">Part {}</span>'
                             '<span class="t">{}</span>{}</a></li>'.format(p.key, p.roman, html.escape(p.title), pg(p.key)))
        num = e.number
        subs = [t for t in book.toc.get(e.key, []) if t[0] == 2]
        if mode == "nav":
            inner = "".join('<li><a href="#{}"><span class="num">{}</span> {}</a></li>'.format(a, n, t)
                            for _, n, t, a in subs)
            items.append('<li><a href="#{}"><span class="num">{}</span> {}</a>{}</li>'.format(
                e.key, num, e.title_html, "<ol>{}</ol>".format(inner) if inner else ""))
        else:
            cls = "c-front" if e.kind == "front" else "c-chap"
            label = num                       # 7, or A for an appendix; front matter has none
            items.append('<li class="{}"><a href="#{}"><span class="n">{}</span><span class="t">{}</span>{}</a></li>'
                         .format(cls, e.key, label, e.title_html, pg(e.key)))
            for _, n, t, a in subs:
                items.append('<li class="c-sec"><a href="#{}"><span class="n">{}</span><span class="t">{}</span>{}</a></li>'
                             .format(a, n, t, pg(a)))
    return "\n".join(items)


def assemble(meta, book, parts, entries, rendered, kv, mode, whole, current, pages=None):
    """mode 'screen' or 'print'. whole: the book (title page, parts); else one chapter."""
    linked = {e.key for e in entries} | ({p.key for p in parts} if whole else set())
    print_alt = mode == "print"
    strict = whole and not book.opts.draft
    body, seen_parts = [], set()
    for e in entries:
        if whole and e.part is not None and e.part.key not in seen_parts:
            seen_parts.add(e.part.key)
            body.append(part_html(e.part))
        body.append(resolve(e.html, book, rendered, linked, current, print_alt, strict))
    body = "".join(body)
    title = meta.get("title", "VectorFFT User Guide")
    if whole:
        titleblock = ('<header class="titleblock">\n<p class="wordmark">{}</p>\n<p class="booktitle">{}</p>\n'
                      '<p class="edition">{}&ensp;&middot;&ensp;{}</p>\n<p class="colophon">{}</p>\n</header>\n'
                      .format(html.escape(meta.get("wordmark", "VectorFFT")), html.escape(meta.get("subtitle", "User Guide")),
                              html.escape(meta.get("edition", "")), html.escape(meta.get("date", "")), COLOPHON))
    else:
        e = entries[0]
        titleblock = ('<header class="titleblock draft">\n<p class="booktitle">{}</p>\n'
                      '<p class="edition">{}&ensp;&middot;&ensp;a build of {} alone</p>\n</header>\n'
                      .format(html.escape(title), html.escape(meta.get("edition", "")),
                              html.escape(e.running_head().replace("\u2003", " "))))
    toc_nav = resolve(toc_items(book, parts if whole else [], entries, "nav"), book, rendered, linked, current, False, strict)
    nav = ""
    contents = ""
    footer = ""
    if mode == "screen":
        nav = ('<nav class="toc" aria-label="Contents"><details><summary>Contents</summary>\n<ol>\n{}\n</ol>\n'
               '</details></nav>\n'.format(toc_nav))
        repo = meta.get("repository", "")
        footer = ('<footer>{} &middot; {}<p class="colophon">{}</p></footer>\n'.format(
            html.escape(title), '<a href="{0}">{1}</a>'.format(html.escape(repo), html.escape(repo.split("//")[-1]))
            if repo else "", COLOPHON))
    elif whole:
        items = resolve(toc_items(book, parts, entries, "print", pages), book, rendered, linked, current, False, strict)
        contents = ('<div class="contents">\n<p class="contents-head">Contents</p>\n<ol>\n{}\n</ol>\n</div>\n'
                    .format(items))
    import markdown_it
    return PAGE.format(title=html.escape(title if whole else "{} - {}".format(
                           title, entries[0].running_head().replace("\u2003", " "))),
                       edition=html.escape(meta.get("edition", "")), mdv=markdown_it.__version__, kv=kv,
                       faces=font_faces(), katex=katex_css(), css=(HERE / "guide.css").read_text(encoding="utf-8"),
                       pages=page_rules(entries), nav=nav, titleblock=titleblock, contents=contents,
                       body=body, footer=footer)


# ── printing ─────────────────────────────────────────────────────────────────────────────────

def find_chrome(arg):
    cands = [arg, os.environ.get("VFFT_GUIDE_CHROME"),
             r"C:\Program Files\Google\Chrome\Application\chrome.exe",
             r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
             shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chrome")]
    for c in cands:
        if c and Path(c).exists():
            return c
    raise BuildError("no Chrome found: pass --chrome PATH (Edge is not used: it signs fresh profiles in)")


def chrome_print(chrome, html_path, pdf_path):
    """Headless Chrome in a fresh throwaway profile, checked for a signed-in account, deleted."""
    prof = Path(tempfile.mkdtemp(prefix="vfft-guide-chrome-"))
    try:
        cmd = [chrome, "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               "--disable-sync", "--disable-extensions", "--user-data-dir=" + str(prof),
               "--no-pdf-header-footer", "--generate-pdf-document-outline", "--virtual-time-budget=30000",
               "--print-to-pdf=" + str(pdf_path), Path(html_path).resolve().as_uri()]
        if pdf_path.exists():
            pdf_path.unlink()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        prefs = prof / "Default" / "Preferences"
        if prefs.exists() and '"account_info"' in prefs.read_text(encoding="utf-8", errors="replace"):
            print("build: warning: the throwaway Chrome profile signed in to an account", file=sys.stderr)
        if not pdf_path.exists():
            raise BuildError("Chrome wrote no PDF:\n" + (r.stderr or r.stdout))
    finally:
        shutil.rmtree(prof, ignore_errors=True)


def dest_pages(pdf_path):
    from pypdf import PdfReader
    r = PdfReader(str(pdf_path))
    out = {}
    for name, d in r.named_destinations.items():
        try:
            out[str(name).lstrip("/")] = r.get_destination_page_number(d) + 1
        except Exception:
            pass
    return out, len(r.pages)


# ── main ─────────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default=str(SOURCE / "book.yml"))
    ap.add_argument("--chapter", help="build one entry by its key (ch07, appa, fm2)")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--allow-missing-output", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--out-dir")
    ap.add_argument("--chrome")
    a = ap.parse_args()
    try:
        return build(a)
    except BuildError as err:
        print("build: error: %s" % err, file=sys.stderr)
        return 1


def build(a):
    book_path = Path(a.book).resolve()
    meta, parts, entries = load_manifest(book_path)
    book = Book(a)
    md = make_md(book)
    whole = a.chapter is None
    if whole:
        targets = entries
        missing = [e for e in targets if not e.exists]
        if missing and not a.draft:
            raise BuildError("not written yet: %s (--draft builds placeholders)" % ", ".join(e.rel for e in missing))
    else:
        targets = [e for e in entries if e.key == a.chapter.lower()]
        if not targets:
            raise BuildError("no entry %r in %s; the keys are %s" % (a.chapter, book_path.name,
                                                                   ", ".join(e.key for e in entries)))
        if not targets[0].exists:
            raise BuildError("%s does not exist yet" % targets[0].rel)
    # every existing file is rendered so that its labels and numbers are known; in a chapter
    # build, a broken other chapter only costs its labels
    for p in parts:
        book.register(p.key, "part", p.roman, p.key, p.key)
    for e in entries:   # the manifest numbers every entry, written or not: {ref}`ch10` always resolves
        book.register(e.key, e.kind, e.number, e.key, e.key, html.escape(e.planned))
    for e in entries:
        if not e.exists:
            continue
        if whole or e in targets:
            render_entry(md, book, e)
            continue
        snapshot = (len(book.listings), len(book.outputs), len(book.excerpts), len(book.figures), len(book.warnings))
        failure = None
        try:
            render_entry(md, book, e)
        except BuildError as err:
            failure = err
        del book.listings[snapshot[0]:], book.outputs[snapshot[1]:], book.excerpts[snapshot[2]:]
        del book.figures[snapshot[3]:], book.warnings[snapshot[4]:]
        if failure is not None:
            book.warn("%s does not build, so its labels are unknown here: %s" % (e.rel, failure))
    for e in targets:
        if not e.exists:
            e.html = placeholder_html(e)
    # the math of the pages written here
    shown = "".join([e.html for e in targets] + [e.title_html for e in targets]
                    + [t for e in targets for (_, _, t, _) in book.toc.get(e.key, [])]
                    + [lab.text for lab in book.labels.values()])
    used = sorted({int(i) for i in re.findall("\ue000M:(\\d+)\ue001", shown)})
    kv, rendered = render_math(book, used) if used else ("0.16.21", {})
    current = None if whole else targets[0].key
    default_out = GUIDE if (whole and book_path == (SOURCE / "book.yml").resolve()) else SOURCE / "_build"
    out_dir = Path(a.out_dir).resolve() if a.out_dir else default_out
    work = Path(a.out_dir).resolve() if a.out_dir else SOURCE / "_build"
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    stem = meta.get("output", "VectorFFT-Guide") if whole else targets[0].key
    screen = assemble(meta, book, parts, targets, rendered, kv, "screen", whole, current)
    screen_path = out_dir / (stem + ".html")
    write_lf(screen_path, screen)
    report = {"html": rel(screen_path), "bytes": len(screen.encode("utf-8")), "katex": kv, "math": len(used),
              "entries": [e.key for e in targets], "labels": {k: v.number for k, v in book.labels.items()
                                                              if v.owner in {e.key for e in targets}},
              "listings": book.listings, "outputs": book.outputs, "excerpts": book.excerpts,
              "figures": book.figures}
    want_pdf = (whole and not a.no_pdf) or (not whole and a.pdf)
    if want_pdf:
        chrome = find_chrome(a.chrome)
        print_path = work / (stem + ".print.html")
        pdf_path = out_dir / (stem + ".pdf")
        pages = {}
        passes = 0
        while True:   # the contents' page numbers: print, read the pages, write them in, print again
            passes += 1
            write_lf(print_path, assemble(meta, book, parts, targets, rendered, kv, "print", whole, current, pages))
            chrome_print(chrome, print_path, pdf_path)
            found, npages = dest_pages(pdf_path)
            if not whole:
                break
            wanted = {k: v for k, v in found.items() if k in book.labels or k in {e.key for e in targets}}
            if wanted == pages:
                break
            if passes == 3:
                raise BuildError("the contents' page numbers do not settle after 3 prints")
            pages = wanted
        report.update({"pdf": rel(pdf_path), "pages": npages, "prints": passes, "print_html": rel(print_path)})
        import check
        heads = [e.running_head() for e in targets]
        report["running_heads"] = heads
        result = check.run(print_path, pdf_path, work / (stem + ".txt"), report, heads)
        (work / (stem + ".check.json")).write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                                   encoding="utf-8")
        report["check"] = result["verdict"]
        report["check_json"] = rel(work / (stem + ".check.json"))
    report["warnings"] = book.warnings
    write_lf(work / (stem + ".build.json"), json.dumps(report, indent=1, ensure_ascii=False))
    summary = {k: report[k] for k in ("html", "bytes", "math", "pdf", "pages", "prints", "check", "check_json")
               if k in report}
    summary["listings"] = len(book.listings)
    summary["outputs"] = len(book.outputs)
    summary["excerpts"] = len(book.excerpts)
    summary["figures"] = len(book.figures)
    print(json.dumps(summary, indent=1))
    for w in book.warnings:
        print("build: warning: " + w, file=sys.stderr)
    if report.get("check") == "FAIL":
        print("build: the checks failed: see " + report["check_json"], file=sys.stderr)
        return 1
    return 0


def rel(p):
    try:
        return Path(p).resolve().relative_to(GUIDE.parent).as_posix()
    except ValueError:
        return Path(p).as_posix()


def write_lf(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")


if __name__ == "__main__":
    sys.exit(main())
