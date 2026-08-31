#!/usr/bin/env python3
"""obj_equiv.py — prove that two builds of the same translation unit are
SEMANTICALLY IDENTICAL, with no clock involved.

WHY THIS EXISTS
---------------
The refactor moves code out of src/core/vfft.c into module headers. A pure move
must not change what the compiler emits. Comparing the .o files byte-for-byte
does NOT work: moving a function changes its position in the translation unit,
so the linker/assembler reorders .text and picks different alignment padding.
Measured on a real extraction: same .o size, 9303 differing bytes, .text alone
8484 differing bytes -- yet every symbol body was identical.

So the comparable artifact is the SET OF SYMBOL BODIES, not the file. Three
normalizations are required, each one derived from an observed false positive:

  1. ADDRESSES        instruction offsets shift when code moves.
  2. OBJDUMP COMMENTS objdump annotates a RIP-relative load with the NEAREST
                      symbol, which changes when neighbours move. The
                      instruction is identical; only the comment differs.
                      (Observed on _vfft_tname.)
  3. ALIGNMENT NOPS   the assembler picks a different NOP encoding
                      (nopl / data16 cs nopw / xchg %ax,%ax) depending on where
                      a function lands. (Observed on _dht_worker_post.)

With all three applied, a verified-pure extraction compared 1007/1007 symbol
bodies identical, 0 differing.

WHAT IT PROVES / DOES NOT PROVE
-------------------------------
PROVES: the emitted code for every symbol is unchanged -- a true pure move.
A symbol that appears or disappears is reported and is always a finding: it
means a function was inlined, dropped, or duplicated.

DOES NOT PROVE anything about a MERGE (two racers collapsed into one) or any
change that deliberately alters code. For those the object WILL differ by
design; use the golden decision trace instead.

!! BLIND SPOT, MEASURED, DO NOT FORGET !!
This tool is blind to DATA. Floating-point constants live in .rdata, not in
the instruction stream: the instruction is `vcomisd HEX(%rip),%xmm0` and only
the constant's ADDRESS appears in the disassembly. Verified by injecting a
one-token change of a race hysteresis constant, 0.97 -> 0.96 -- exactly the
silent-verdict-drift bug this refactor most fears -- and this tool reported
EQUIVALENT.

Comparing .rdata directly does not rescue it. As a byte sequence it differs on
any pure move (the constant pool reorders). As a sorted multiset of 8-byte
words, a pure move perturbs 92 of 4072 words while the 0.97->0.96 bug perturbs
4: the signal sits under the noise, so no threshold separates them.

CONSEQUENCE: this tool covers CODE. It cannot be the protocol detector. Pair
it with the golden decision trace, which must print the protocol constants
(reps, median rank, arm order, hysteresis) as text. Neither check alone is
sufficient; a change that is EQUIVALENT here and byte-identical in the trace
is what "nothing moved" actually means.

USAGE
  python build_tuned/obj_equiv.py before.o after.o [--objdump PATH]
Exit 0 when equivalent, 1 when not.
"""
import re
import subprocess
import sys

DEFAULT_OBJDUMP = r"C:\mingw152\mingw64\bin\objdump.exe"

_ADDR_PREFIX = re.compile(r"^\s*[0-9a-f]+:")
_COMMENT = re.compile(r"#.*$")
_SYMREF = re.compile(r"<[^>]*\+[^>]*>")
# A DISPLACEMENT's SIGN IS PART OF THE ADDRESS.
#
# `_HEX0X` used to be r"0x[0-9a-f]+", which turns -0x8(%rip) into -HEX(%rip)
# and 0x8(%rip) into HEX(%rip) - reporting a difference for what is the SAME
# instruction reaching the same object from the other side of the instruction
# pointer. Address layout is precisely what this normalizer exists to absorb,
# so leaking the sign made it a false-positive generator: giving two counters
# external linkage shifted the BSS layout and "changed" six function bodies
# whose instruction streams were otherwise identical, line for line.
#
# The lookahead confines this to DISPLACEMENTS (a hex immediately followed by
# an open paren). An IMMEDIATE's sign is NOT normalized - $-0x1 and $0x1 are
# genuinely different instructions, and collapsing those would blind the gate
# to a real change.
#
# SCOPE NOTE, measured while fixing the above. Because _HEX0X collapses EVERY
# hex to HEX, this comparison is also blind to IMMEDIATE VALUES: `mov $0x1` and
# `mov $0x2` normalize equal. That is the same blindness as the .rdata one
# documented at the top of this file, and it is exactly why the race protocol
# census exists alongside this check - obj_equiv covers the SHAPE of the code,
# the census covers the CONSTANTS, and neither alone is sufficient. Immediate
# SIGNS do still differ, because the minus falls outside the hex match.
_DISP = re.compile(r"-?0x[0-9a-f]+(?=\()")
_HEX0X = re.compile(r"0x[0-9a-f]+")
_HEXBARE = re.compile(r"\b[0-9a-f]{4,}\b")
_HEADER = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")
_NOP = re.compile(r"^\s*(nop|nopl|nopw|nopq|data16|cs nopw|xchg\s+%ax,%ax)")


def symbol_bodies(path, objdump):
    """-> {symbol: normalized body}. Raises on objdump failure."""
    out = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", path],
        capture_output=True, text=True, check=True).stdout

    bodies, cur, buf = {}, None, []
    for line in out.splitlines():
        if "file format" in line:
            continue
        head = _HEADER.match(line.strip())
        if head:
            if cur is not None:
                bodies[cur] = "\n".join(buf)
            cur, buf = head.group(2), []
            continue
        if cur is None:
            continue
        s = _COMMENT.sub("", line)    # 2. objdump's nearest-symbol annotation
        s = _SYMREF.sub("", s)
        s = _ADDR_PREFIX.sub("", s)   # 1. addresses
        # NOP check must run AFTER the address prefix is stripped, or the
        # leading "  4a1b:" prevents the anchor from ever matching.
        if _NOP.match(s):             # 3. alignment padding
            continue
        s = _DISP.sub("HEX", s)      # displacement: sign is address, not value
        s = _HEX0X.sub("HEX", s)
        s = _HEXBARE.sub("ADDR", s)
        s = s.rstrip()
        if s:
            buf.append(s)
    if cur is not None:
        bodies[cur] = "\n".join(buf)
    return bodies


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    objdump = DEFAULT_OBJDUMP
    if "--objdump" in sys.argv:
        objdump = sys.argv[sys.argv.index("--objdump") + 1]
    if len(args) != 2:
        print(__doc__)
        return 2

    a = symbol_bodies(args[0], objdump)
    b = symbol_bodies(args[1], objdump)

    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])

    print("symbols: %d -> %d" % (len(a), len(b)))
    for k in only_a[:20]:
        print("  DISAPPEARED: %s" % k)
    for k in only_b[:20]:
        print("  APPEARED   : %s" % k)
    for k in changed[:20]:
        print("  BODY CHANGED: %s" % k)
    if len(changed) > 20:
        print("  ... and %d more changed bodies" % (len(changed) - 20))

    if not (only_a or only_b or changed):
        print("\nEQUIVALENT - all %d symbol bodies identical." % len(a))
        return 0
    print("\nNOT EQUIVALENT - %d changed, %d gone, %d new."
          % (len(changed), len(only_a), len(only_b)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
