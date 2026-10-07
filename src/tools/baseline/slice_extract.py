#!/usr/bin/env python3
"""slice_extract.py - cut one create tier out of `_vfft_create_inner` into a
module header, mechanically.

WHY A SCRIPT AND NOT AN EDITOR
------------------------------
Steps 22-27 are the same edit six times: lift a `if (<cond>) { ... }` block
that returns on every path, put it behind a helper taking the block's free
variables, and replace it with a guarded call. Done by hand the risk is not
the logic, it is the transcription -- a body retyped instead of moved is a body
that can differ from the original by a character nobody will ever see. Here the
body is passed through byte-for-byte by slicing the source, so "the code did
not change" is a property of the method rather than a claim to be checked.

WHAT IT ENFORCES BEFORE TOUCHING ANYTHING
  * the start line really is the `if (<cond>)` the caller named;
  * the block's braces balance, and the matched end is the line given;
  * the LAST top-level statement in the block is a `return` -- otherwise the
    block can fall through, the guarded call would swallow that path, and the
    extraction is NOT behaviour-preserving. This is the one property that makes
    the whole transformation legal, so it is checked, not assumed.

USAGE
  python build_tuned/slice_extract.py --start N --end M --cond "cfg->dims == 2"
      --helper _vfft_create_2d --header src/core/transforms/fft2d/fft2d_create.h
      --guard VFFT_TRANSFORMS_FFT2D_CREATE_H --doc <preamble.txt>
      --after-include "oop/k1_commit.h" --note "2D create tier (step 23)"
"""
import os
import re
import sys

SRC = os.path.join('src', 'core', 'vfft.c')

# The free-variable set differs per tier, so the signature is built from the
# names the caller derived rather than fixed here. These are the only names a
# tier can legitimately take from _vfft_create_inner's scope: its two
# parameters, and the four locals computed at the top of it.
PARAM_TYPE = {
    'cfg': 'const vfft_config_t *cfg',
    'ob': 'vfft_batch ob',
    'reg': 'const vfft_proto_registry_t *reg',
    'N': 'int N',
    'K': 'size_t K',
    'W': 'struct vfft_wisdom_s *W',
}

TAIL = """    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* %s */
"""


def opt(name, default=None, required=False):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    if required:
        raise SystemExit('missing %s' % name)
    return default


def top_level_ends_in_return(blk):
    """Depth-1 statements of the block; True when the last one is a return."""
    d, in_comment, top = 0, False, []
    for line in blk:
        s = line
        if in_comment:
            if '*/' in s:
                s = s.split('*/', 1)[1]
                in_comment = False
            else:
                continue
        s = re.sub(r'//.*', '', s)
        while '/*' in s:
            head, rest = s.split('/*', 1)
            if '*/' in rest:
                s = head + rest.split('*/', 1)[1]
            else:
                s, in_comment = head, True
                break
        s = re.sub(r'"(?:\\.|[^"\\])*"', '""', s)
        s = re.sub(r"'(?:\\.|[^'\\])*'", "''", s)
        start = d
        for ch in s:
            if ch == '{':
                d += 1
            elif ch == '}':
                d -= 1
        t = s.strip()
        if start == 1 and t and t not in ('{', '}'):
            top.append(t)
    return bool(top) and bool(re.match(r'return\b', top[-1])), top


def main():
    start = int(opt('--start', required=True))
    end = int(opt('--end', required=True))
    cond = opt('--cond', required=True)
    helper = opt('--helper', required=True)
    header = opt('--header', required=True)
    guard = opt('--guard', required=True)
    doc = opt('--doc', required=True)
    after_inc = opt('--after-include', required=True)
    note = opt('--note', helper)

    raw = open(SRC, 'rb').read().split(b'\n')
    s0, e0 = start - 1, end - 1

    if cond.encode() not in raw[s0]:
        raise SystemExit('line %d is not the %r block: %r'
                         % (start, cond, raw[s0].decode(errors='replace')))
    blk = raw[s0:e0 + 1]
    text = b'\n'.join(blk).decode('utf-8', 'replace')
    if text.count('{') != text.count('}'):
        raise SystemExit('braces do not balance in %d-%d' % (start, end))
    ok, top = top_level_ends_in_return(text.split('\n'))
    if not ok:
        raise SystemExit('block does NOT end in a return at block scope; a '
                         'guarded call would swallow the fall-through path. '
                         'last top-level statement: %r'
                         % (top[-1][:90] if top else None))

    params = [p.strip() for p in opt('--params', 'cfg,W,reg,K').split(',')]
    unknown = [p for p in params if p not in PARAM_TYPE]
    if unknown:
        raise SystemExit('unknown parameter(s): %s' % ', '.join(unknown))
    pad = ' ' * (len('static vfft_plan ') + len(helper) + 1)
    sig = ('static vfft_plan %s(%s)\n{\n'
           % (helper, (',\n' + pad).join(PARAM_TYPE[p] for p in params)))

    body = b'\n'.join(blk)
    out = (open(doc, 'rb').read().rstrip(b'\n') + b'\n'
           + b'#ifndef ' + guard.encode() + b'\n'
           + b'#define ' + guard.encode() + b'\n\n'
           + sig.encode()
           + body + b'\n'
           + (TAIL % guard).encode())
    os.makedirs(os.path.dirname(header), exist_ok=True)
    open(header, 'wb').write(out)

    call = ('    /* %s */\n    if (%s)\n        return %s(%s);'
            % (note, cond, helper, ', '.join(params))).encode().split(b'\n')
    raw[s0:e0 + 1] = call

    inc = ('#include "%s" /* %s */'
           % (header.replace('src/core/', '').replace(os.sep, '/'), note))
    for i, ln in enumerate(raw):
        if after_inc.encode() in ln and ln.lstrip().startswith(b'#include'):
            raw.insert(i + 1, inc.encode())
            break
    else:
        raise SystemExit('anchor include %r not found' % after_inc)

    open(SRC, 'wb').write(b'\n'.join(raw))
    print('extracted %d lines -> %s' % (len(blk), header))
    print('  %d top-level statements, last is a return' % len(top))
    print('  vfft.c now %d lines' % (len(raw) - 1))


if __name__ == '__main__':
    main()
