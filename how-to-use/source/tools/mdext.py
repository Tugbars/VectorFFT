"""The guide's Markdown dialect on markdown-it-py 4: the block and inline rules that
CommonMark lacks. Each rule only produces tokens; build.py renders them.

  (label)=              a target line: labels the heading that follows
  :::{name} argument    a directive, closed by a line of at least as many colons;
  :key: value           option lines directly after the opening line
  $$ ... $$ (label)     display math, numbered when a label follows the closing $$
  $...$                 inline math (a closing $ followed by a digit is not math)
  {role}`content`       a role: {ref}, {numref}, {eq}, {output}

Only the standard library and markdown-it-py are used.
"""
import re


class DialectError(ValueError):
    """A source error the parser can place: build.py adds the file name."""

LABEL = r"[a-z][a-z0-9-]*"
RE_TARGET = re.compile(r"^\((" + LABEL + r")\)=\s*$")
RE_EQ_LABEL = re.compile(r"^\((" + LABEL + r")\)$")
RE_ROLE = re.compile(r"\{([a-z]+)\}`([^`\n]+)`")


def _target(state, start, end, silent):
    if state.is_code_block(start):
        return False
    pos = state.bMarks[start] + state.tShift[start]
    m = RE_TARGET.match(state.src[pos:state.eMarks[start]])
    if not m:
        return False
    if silent:
        return True
    state.line = start + 1
    tok = state.push("myst_target", "", 0)
    tok.content = m.group(1)
    tok.map = [start, state.line]
    return True


def _colon_fence(state, start, end, silent):
    if state.is_code_block(start):
        return False
    pos = state.bMarks[start] + state.tShift[start]
    maximum = state.eMarks[start]
    if pos + 3 > maximum or state.src[pos] != ":":
        return False
    mem = pos
    pos = state.skipCharsStr(pos, ":")
    length = pos - mem
    if length < 3:
        return False
    params = state.src[pos:maximum].strip()
    if not params.startswith("{"):
        return False
    if silent:
        return True
    line, closed = start, False
    while True:
        line += 1
        if line >= end:
            break
        p = state.bMarks[line] + state.tShift[line]
        mx = state.eMarks[line]
        if p < mx and state.sCount[line] < state.blkIndent:
            break
        if state.src[p:p + 1] != ":":
            continue
        q = state.skipCharsStr(p, ":")
        if q - p < length or state.skipSpaces(q) < mx:
            continue
        closed = True
        break
    if not closed:
        raise DialectError("line %d: the directive '%s' is never closed (a line of %d colons)"
                           % (start + 1, state.src[mem:maximum].strip(), length))
    state.line = line + 1
    tok = state.push("colon_fence", "div", 0)
    tok.info = params
    tok.content = state.getLines(start + 1, line, state.sCount[start], True)
    tok.markup = state.src[mem:mem + length]
    tok.map = [start, state.line]
    return True


def _math_block(state, start, end, silent):
    if state.is_code_block(start):
        return False
    begin = state.bMarks[start] + state.tShift[start]
    if not state.src.startswith("$$", begin):
        return False
    line, at = start, None
    while line < end:
        lo = begin + 2 if line == start else state.bMarks[line]
        i = state.src.find("$$", lo, state.eMarks[line])
        if i != -1:
            at = i
            break
        line += 1
    if at is None:
        raise DialectError("line %d: display math opened with $$ is never closed" % (start + 1))
    after = state.src[at + 2:state.eMarks[line]].strip()
    label = ""
    if after:
        m = RE_EQ_LABEL.match(after)
        if not m:
            raise DialectError("line %d: after the closing $$ only an equation label "
                               "'(label)' may follow, found %r" % (line + 1, after))
        label = m.group(1)
    if silent:
        return True
    state.line = line + 1
    tok = state.push("math_block_eqno" if label else "math_block", "math", 0)
    tok.block = True
    tok.content = state.src[begin + 2:at]
    tok.info = label
    tok.markup = "$$"
    tok.map = [start, state.line]
    return True


def _math_inline(state, silent):
    src, pos = state.src, state.pos
    if src[pos] != "$" or src.startswith("$$", pos):
        return False
    close = pos + 1
    while True:
        close = src.find("$", close, state.posMax)
        if close == -1:
            return False
        if src[close - 1] == "\\":
            close += 1
            continue
        break
    if close == pos + 1:
        return False
    if close + 1 < state.posMax and src[close + 1].isdigit():
        return False
    if not silent:
        tok = state.push("math_inline", "math", 0)
        tok.content = src[pos + 1:close]
        tok.markup = "$"
    state.pos = close + 1
    return True


def _role(state, silent):
    if state.src[state.pos] != "{":
        return False
    m = RE_ROLE.match(state.src, state.pos, state.posMax)
    if not m:
        return False
    if not silent:
        tok = state.push("myst_role", "", 0)
        tok.meta = {"name": m.group(1)}
        tok.content = m.group(2)
    state.pos = m.end()
    return True


# The {status} role: {status}`served`, {status}`difference`, {status}`refused` show a glyph (a full,
# a half and an empty circle, so greyscale print and colour-blind readers lose nothing) and the
# word beside it. The glyph is inline SVG, not a font glyph: the CMU faces lack the characters, and
# the PDF check refuses a fallback font. build.py renders it; tools/guide.css (its last block)
# colours it.
STATUS = {
    "served": ("served", '<circle cx="5" cy="5" r="4" fill="currentColor"/>'),
    "difference": ("served with a difference",
                   '<circle cx="5" cy="5" r="3.7" fill="none" stroke="currentColor" stroke-width=".6"/>'
                   '<path d="M5 1.3A3.7 3.7 0 0 0 5 8.7Z" fill="currentColor"/>'),
    "refused": ("refused",
                '<circle cx="5" cy="5" r="3.7" fill="none" stroke="currentColor" stroke-width=".6"/>'),
}


def status_html(name):
    """The HTML of {status}`name`, or None when the role has no such name."""
    if name not in STATUS:
        return None
    word, shape = STATUS[name]
    return ('<span class="status status-{0}"><svg class="glyph" viewBox="0 0 10 10" aria-hidden="true" '
            'focusable="false">{1}</svg>{2}</span>'.format(name, shape, word))


def dialect(md):
    """Install the rules on a MarkdownIt instance."""
    alt = {"alt": ["paragraph", "reference", "blockquote", "list"]}
    md.block.ruler.before("fence", "myst_target", _target, alt)
    md.block.ruler.before("fence", "colon_fence", _colon_fence, alt)
    md.block.ruler.before("fence", "math_block", _math_block, alt)
    md.inline.ruler.before("escape", "math_inline", _math_inline)
    md.inline.ruler.before("backticks", "myst_role", _role)
    return md
