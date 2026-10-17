#!/usr/bin/env python3
"""Check the print edition against the PDF Chrome made of it (pypdf, and pdftotext when present).

  pages, size     page count and media box (A4)
  fonts           every font on every page: an embedded program, no Type 3, and only the guide's
                  faces (CMU Serif, CMU Typewriter Text, KaTeX_*): any other face means a glyph
                  the CMU fonts lack fell back to a system font
  outline         the PDF bookmarks equal the HTML headings (parts, chapters, sections), in order
  links           every cross-reference is a PDF link whose named destination exists
  numbering       the numbered labels and references (Part, Chapter, Appendix, Section, Figure,
                  Table n.m, equation numbers (n.m)) read from the HTML equal those read from
                  the PDF's text, running heads removed
  listings        each listing, excerpt and output shown equals the text the build took it from
  privacy         no shown output carries an absolute path, the user's name or the machine's
  tags            every formula's glyphs lie in tagged content

Called by build.py after printing; also usable alone:
  py -3.13 tools/check.py PRINT.html FILE.pdf BUILD.json
"""
import collections
import html.parser
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.generic import ContentStream, IndirectObject

sys.path.insert(0, str(Path(__file__).resolve().parent))
import privacy  # noqa: E402

NUM = r"(?:\d+|[A-Z])(?:\.\d+)*"
TOKEN = re.compile(r"(?:Part\s+[IVX]+\b|(?:Chapter|Appendix)\s+(?:\d+|[A-Z])\b|(?:Section|Figure|Table)\s+"
                   + NUM + r"|\((?:\d+|[A-Z])\.\d+[a-z]?\)|Step\s+\d+\.)")
ALLOWED = ("CMUSerif", "CMUTypewriter", "KaTeX_")
PDFTOTEXT = [shutil.which("pdftotext"), r"C:\Program Files\Git\mingw64\bin\pdftotext.exe"]


class Text(html.parser.HTMLParser):
    """The printed text: skips <style>, <title>, KaTeX's MathML twin (hidden in print) and the
    phone-only equation variants. Collects h1-h4 texts, reference links, ids, and the text of
    each pre block."""
    VOID = {"img", "br", "meta", "link", "hr", "input", "col", "area", "base", "wbr", "source"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.skip, self.out = [], 0, []
        self.heads, self.cur_head, self.refs, self.ids = [], None, [], set()
        self.pres, self.cur_pre = [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag in self.VOID:
            return
        cls = (a.get("class") or "").split()
        hide = tag in ("style", "title", "script") or "katex-mathml" in cls or "narrow" in cls
        self.stack.append(hide)
        self.skip += hide
        if tag in ("h1", "h2", "h3", "h4"):
            self.cur_head = []
        if tag == "a" and "ref" in cls:
            self.refs.append(a.get("href", "")[1:])
        if tag == "pre":
            self.cur_pre = ("output" if "output" in cls else "code", [])
        if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "tr", "pre", "figcaption", "caption", "dt", "dd"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if tag in ("h1", "h2", "h3", "h4") and self.cur_head is not None:
            self.heads.append(re.sub(r"\s+", " ", "".join(self.cur_head)).strip())
            self.cur_head = None
        if tag == "pre" and self.cur_pre is not None:
            self.pres.append((self.cur_pre[0], "".join(self.cur_pre[1])))
            self.cur_pre = None
        hide = self.stack.pop() if self.stack else False
        self.skip -= hide
        self.out.append(" ")

    def handle_data(self, data):
        if self.cur_pre is not None:
            self.cur_pre[1].append(data)
        if self.skip:
            return
        self.out.append(data)
        if self.cur_head is not None:
            self.cur_head.append(data)


def tokens(text):
    return [re.sub(r"\s+", " ", t) for t in TOKEN.findall(re.sub(r"\s+", " ", text))]


def resolve(o):
    return o.get_object() if isinstance(o, IndirectObject) else o


def pdf_text(pdf, txt_out):
    """The PDF's text, one string per page: pdftotext when present, else pypdf."""
    exe = next((p for p in PDFTOTEXT if p and Path(p).exists()), None)
    if exe:
        subprocess.run([exe, "-enc", "UTF-8", str(pdf), str(txt_out)], check=True, capture_output=True)
        return Path(txt_out).read_text(encoding="utf-8", errors="replace").split("\f"), "pdftotext"
    r = PdfReader(str(pdf))
    pages = [p.extract_text() or "" for p in r.pages]
    Path(txt_out).write_text("\f".join(pages), encoding="utf-8")
    return pages, "pypdf"


def strip_heads(pages, heads):
    """Remove the running heads and folios (Chromium's margin boxes) from each page's text."""
    norm = {re.sub(r"\s+", "", h) for h in heads} | {"Contents"}
    out = []
    for pg in pages:
        lines = pg.split("\n")
        keep = []
        for i, l in enumerate(lines):
            s = re.sub(r"\s+", "", l)       # pdftotext drops the em space of a running head
            if i < 4 and (s in norm or re.fullmatch(r"\d+", s)):
                continue
            keep.append(l)
        out.append("\n".join(keep))
    return out


def tag_audit(r):
    katex_untagged, runs = 0, 0
    for page in r.pages:
        fonts = {k: str(resolve(v).get("/BaseFont")).split("+")[-1]
                 for k, v in (resolve(resolve(page["/Resources"]).get("/Font")) or {}).items()}
        stack, cur = [], "?"
        for operands, op in ContentStream(page.get_contents(), r).operations:
            if op in (b"BDC", b"BMC"):
                props = operands[1] if op == b"BDC" and len(operands) > 1 else None
                stack.append((str(operands[0]), props.get("/MCID") if hasattr(props, "get") else None))
            elif op == b"EMC":
                if stack:
                    stack.pop()
            elif op == b"Tf":
                cur = fonts.get(operands[0], "?")
            elif op in (b"Tj", b"TJ"):
                runs += 1
                if any(m is not None for _, m in stack) or any(t == "/Artifact" for t, _ in stack):
                    continue
                if cur.startswith("KaTeX"):
                    katex_untagged += 1
    figs = collections.Counter()

    def walk(n):
        n = resolve(n)
        if isinstance(n, list):
            for k in n:
                walk(k)
        elif hasattr(n, "get") and "/K" in n:
            if n.get("/S") == "/Figure":
                figs["with_alt" if n.get("/Alt") else "without_alt"] += 1
            walk(n["/K"])
    st = resolve(r.trailer["/Root"]).get("/StructTreeRoot")
    if st is not None:
        walk(resolve(st)["/K"])
    return {"text_runs": runs, "katex_runs_untagged": katex_untagged, "figure_elements": dict(figs)}


def run(print_html, pdf, txt_out, report, heads):
    rep, ok = {}, True
    p = Text()
    p.feed(Path(print_html).read_text(encoding="utf-8"))
    html_text = "".join(p.out)
    r = PdfReader(str(pdf))
    rep["pages"] = len(r.pages)
    mb = r.pages[0].mediabox
    rep["page_size_pt"] = [round(float(mb.width), 2), round(float(mb.height), 2)]
    rep["a4"] = abs(float(mb.width) - 595.3) < 2 and abs(float(mb.height) - 841.9) < 2
    ok &= rep["a4"]

    fonts, bad = {}, []
    for i, pg in enumerate(r.pages, 1):
        res = resolve(pg.get("/Resources", {}))
        for _, fref in (resolve(res.get("/Font", {})) or {}).items():
            f = resolve(fref)
            base = str(f.get("/BaseFont", "?")).lstrip("/")
            sub = str(f.get("/Subtype", "?"))
            prog, desc = "?", None
            if sub == "/Type0":
                d = resolve(resolve(f["/DescendantFonts"])[0])
                desc = resolve(d.get("/FontDescriptor"))
            elif "/FontDescriptor" in f:
                desc = resolve(f.get("/FontDescriptor"))
            if desc is not None:
                prog = ("FontFile2" if "/FontFile2" in desc else "FontFile3" if "/FontFile3" in desc
                        else "FontFile" if "/FontFile" in desc else "NOT EMBEDDED")
            elif sub == "/Type3":
                prog = "Type3"
            name = base.split("+", 1)[-1]
            fonts.setdefault(name, {"program": prog, "pages": set()})["pages"].add(i)
            if prog in ("NOT EMBEDDED", "Type3") or not name.startswith(ALLOWED):
                bad.append([i, base, prog])
    rep["fonts"] = {k: "%s embedded, on %d page(s)" % (v["program"], len(v["pages"])) if v["program"].startswith("FontFile")
                    else v["program"] for k, v in sorted(fonts.items())}
    rep["fonts_all_embedded"] = all(v["program"].startswith("FontFile") for v in fonts.values())
    rep["font_problems"] = bad[:20]
    ok &= not bad

    outline = []

    def walk(items):
        for it in items:
            if isinstance(it, list):
                walk(it)
            else:
                outline.append(re.sub(r"\s+", " ", it.title).strip())
    walk(r.outline)
    rep["outline_entries"] = len(outline)
    rep["outline_equals_headings"] = outline == p.heads
    if outline != p.heads:
        ok = False
        for i, (x, y) in enumerate(zip(outline + [""] * len(p.heads), p.heads + [""] * len(outline))):
            if x != y:
                rep["outline_first_difference"] = {"index": i, "pdf": x, "html": y}
                break

    dests = {str(k).lstrip("/") for k in r.named_destinations.keys()}
    links = []
    for pg in r.pages:
        for an in pg.get("/Annots", []) or []:
            an = resolve(an)
            if an.get("/Subtype") == "/Link":
                d = an.get("/Dest")
                act = resolve(an.get("/A")) if "/A" in an else None
                if d is None and act is not None:
                    d = act.get("/D")
                links.append(str(d).lstrip("/") if d is not None else
                             ("URI:" + str(act.get("/URI")) if act else "?"))
    internal = [l for l in links if not l.startswith("URI:")]
    missing = sorted({l for l in internal if l not in dests})
    rep["pdf_links"] = len(internal)
    rep["html_refs"] = len(p.refs)
    rep["links_without_destination"] = missing
    rep["every_ref_linked"] = all(internal.count(t) >= p.refs.count(t) for t in set(p.refs))
    ok &= not missing and rep["every_ref_linked"]

    pages, how = pdf_text(pdf, txt_out)
    t_html = tokens(html_text)
    t_pdf = tokens("\n".join(strip_heads(pages, heads)))
    rep["text_from"] = how
    rep["numbered_tokens"] = [len(t_html), len(t_pdf)]
    # the same numbers, each as often: the text layer's order differs from the page's where an
    # equation number sits in its own column, so the comparison is of counts, not sequences
    ch, cp = collections.Counter(t_html), collections.Counter(t_pdf)
    rep["numbering_identical"] = ch == cp
    if ch != cp:
        ok = False
        rep["numbering_only_in_html"] = sorted((ch - cp).elements())[:20]
        rep["numbering_only_in_pdf"] = sorted((cp - ch).elements())[:20]

    # every listing, excerpt and output shown equals the text the build took it from (the
    # parser has already turned character references back into characters)
    shown_code = [t.rstrip("\n") for k, t in p.pres if k == "code"]
    shown_out = [t.rstrip("\n") for k, t in p.pres if k == "output"]
    want_code = [x["text"] for x in report.get("listings", [])] + [x["text"] for x in report.get("excerpts", [])]
    want_out = [x["text"] for x in report.get("outputs", []) if not x.get("inline")]
    rep["listings_shown_equal_sources"] = sorted(want_code) == sorted(c for c in shown_code if c in want_code) \
        and all(c in shown_code for c in want_code)
    rep["outputs_shown_equal_captures"] = sorted(want_out) == sorted(o for o in shown_out if o in want_out) \
        and len(shown_out) == len(want_out)
    ok &= rep["listings_shown_equal_sources"] and rep["outputs_shown_equal_captures"]
    leaks = [f for o in shown_out for f in privacy.findings(o)]
    rep["outputs_private"] = not leaks
    if leaks:
        rep["privacy_problems"] = leaks[:10]
    ok &= not leaks

    tags = tag_audit(r)
    rep["tags"] = tags
    rep["formulas_tagged"] = tags["katex_runs_untagged"] == 0
    ok &= rep["formulas_tagged"]
    rep["verdict"] = "PASS" if ok else "FAIL"
    return rep


def main():
    print_html, pdf, build_json = sys.argv[1:4]
    report = json.loads(Path(build_json).read_text(encoding="utf-8"))
    heads = report.get("running_heads", [])
    rep = run(Path(print_html), Path(pdf), Path(pdf).with_suffix(".txt"), report, heads)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
