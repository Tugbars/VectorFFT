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

SLICE MODE -- WHOLE-FUNCTION MOVE vs CUTTING A FUNCTION IN TWO
--------------------------------------------------------------
The default verdict (EQUIVALENT) is only reachable for a MOVE: the relocated
body is byte-identical and only its position changes. Cutting a tier OUT of a
large function can never reach it, and that is not a defect. Splitting changes
the object two ways at once:

  1. the parent's body changes -- it IS the thing being split;
  2. the parent shrinks, so GCC's per-function inlining budget frees up and
     small statics elsewhere in it get inlined and vanish from the symbol
     table. Measured on the rank-3/4 extraction: 6 statics unrelated to the
     slice disappeared, and TU total instructions ROSE (273557 -> 273784),
     which is what inlining looks like and what dropping code does not.

So for a slice the gate becomes SHAPE, not equality. --slice PARENT:HELPER
requires all of:

  * PARENT's body changed -- it is the thing being split, so if it did not
    change, nothing was extracted;
  * HELPER was emitted (a .constprop.N / .part.N suffix is the same function
    and is accepted). If it never appears it was inlined straight back, which
    defeats the extraction;
  * no EXTERNALLY VISIBLE symbol appeared or disappeared. A local can vanish
    by being inlined and can appear by ceasing to be; a global can do neither,
    and a new global is new API.

LOCAL churn is reported, NOT gated -- measured on steps 22-24, it takes three
benign shapes: a static inlined away into the budget the parent freed; a
static that had been inlined everywhere needing an out-of-line copy once one
call site moves; and a static whose only call site travelled with the slice
being re-optimized in its new context. An earlier draft gated on "no body
changed but the parent" and had to be withdrawn: it failed on a correct
extraction, and a gate that cries wolf on correct work is one that gets
argued past when it finally matters.

What this does NOT prove is unchanged: slice mode is code-shape only, and
inherits the .rdata and immediate-value blindness documented above. The golden
decision trace is the semantic gate for a slice and carries the weight.

USAGE
  python build_tuned/obj_equiv.py before.o after.o [--objdump PATH]
  python build_tuned/obj_equiv.py before.o after.o --slice PARENT:HELPER
Exit 0 when equivalent (or when the slice shape holds), 1 when not.
"""
import re
import os
import subprocess
# env OBJDUMP / NM override the historical mingw152 paths (2026-09-03): the
# ceremony must run on any host whose binutils live somewhere else.
DEFAULT_OBJDUMP = os.environ.get("OBJDUMP", "C:/mingw152/mingw64/bin/objdump.exe")
DEFAULT_NM = os.environ.get("NM", "C:/mingw152/mingw64/bin/nm.exe")
import sys


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


def local_symbols(path, nm=DEFAULT_NM):
    """-> {name} of symbols with nm class 't' (function, LOCAL linkage).

    Slice mode needs the distinction because the two ways a symbol can leave an
    object are not equally benign: a local can vanish by being inlined into its
    only caller, an exported one cannot.
    """
    out = subprocess.run([nm, "--defined-only", path],
                         capture_output=True, text=True, check=True).stdout
    names = set()
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and p[-2] == "t":
            names.add(p[-1])
    return names


def _same_fn(sym, base):
    """True when `sym` is `base`, possibly with a GCC clone suffix.

    GCC renames a specialized clone -- _vfft_create_rank34.constprop.0,
    foo.part.0 -- and the suffix is an arbitrary per-TU counter, so the bare
    name and any clone of it are the same function for shape purposes.
    """
    return sym == base or sym.startswith(base + ".")


def _base_name(sym):
    """Strip a GCC clone suffix: foo.constprop.0 / foo.part.1 -> foo."""
    return re.sub(r"\.(constprop|part|isra|cold|lto_priv)\.[0-9]+$", "", sym)


def check_slice(only_a, only_b, changed, parent, helper, before_obj, after_obj):
    """The SHAPE gate for cutting a function in two. -> (ok, [reasons], churn).

    WHERE THE LINE FALLS, and why it is not "nothing but the parent moved".

    Cutting a tier out perturbs the inliner across the whole translation unit,
    and the churn lands on LOCAL symbols in three shapes -- all three measured
    on steps 22-24, all three benign:

      gone      a static was fully inlined into the budget the parent freed;
      appeared  the mirror image -- a static that HAD been inlined at every
                call site needs an out-of-line copy once one call site moves
                (stride_dct2_plan, stride_dct4_plan at step 24);
      changed   a static whose only call site moved travels with the slice and
                is re-optimized in its new context (_il_me_decide at step 24).

    None of those is reachable for an EXTERNALLY VISIBLE symbol: a global
    cannot be inlined away, and a new global is new API, which a move does not
    produce. So the gate is global-symbol identity plus the parent/helper
    shape. Local churn is REPORTED rather than gated, and the golden decision
    trace is what proves the semantics did not move.

    Gating on "no body changed but the parent" was tried first and is wrong: it
    fails on a correct extraction, and a gate that cries wolf on correct work
    is a gate that gets argued past.
    """
    bad = []
    if parent not in changed:
        bad.append("parent %s did not change - was anything extracted?" % parent)

    loc_before = local_symbols(before_obj)
    loc_after = local_symbols(after_obj)
    globals_gone = [k for k in only_a if k not in loc_before]
    globals_new = [k for k in only_b
                   if k not in loc_after and not _same_fn(k, helper)]
    if globals_gone:
        bad.append("GLOBAL symbols disappeared (a local can vanish by being "
                   "inlined, an exported one cannot): %s"
                   % ", ".join(globals_gone[:10]))
    if globals_new:
        bad.append("new GLOBAL symbols (a move does not create API): %s"
                   % ", ".join(globals_new[:10]))

    churn = {"gone": [k for k in only_a if k in loc_before],
             "appeared": [k for k in only_b
                          if k in loc_after and not _same_fn(k, helper)],
             "changed": [k for k in changed if k != parent]}
    churn["inlined_back"] = not any(_same_fn(k, helper) for k in only_b)

    # A HELPER THAT NEVER APPEARS is two different situations, and only one is
    # a problem.
    #
    # Benign: the tier was small enough that GCC chose to inline it straight
    # back, and NOTHING ELSE in the object moved. The extraction was then
    # purely a source reorganization -- which is the point of this migration --
    # and the object is the same object. Step 27 (the 51-line trig tier) landed
    # exactly here: 985 -> 985 symbols, nothing gone, nothing new, the parent
    # the only changed body. That is the cleanest outcome available, not a
    # failure.
    #
    # A problem: the helper vanished AND the object churned, which means the
    # inliner did something we are not accounting for. Step 22 is the warning
    # case -- forcing always_inline to chase a clean verdict spilled `cfg` to
    # the stack and grew the parent by 393 instructions. Never chase the symbol.
    if churn["inlined_back"] and (churn["gone"] or churn["appeared"]
                                  or churn["changed"]):
        bad.append("helper %s never appeared AND the object churned (%d gone, "
                   "%d new, %d re-optimized) - the inliner did something this "
                   "gate cannot account for"
                   % (helper, len(churn["gone"]), len(churn["appeared"]),
                      len(churn["changed"])))
    return (not bad), bad, churn


def main():
    # Both options take a VALUE, so the value must be consumed as well as the
    # flag. Filtering only on a leading "--" left the value in the positional
    # list and made every option-bearing invocation print the usage text.
    _TAKES_VALUE = ("--objdump", "--slice")
    argv, args, opts = sys.argv[1:], [], {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in _TAKES_VALUE:
            if i + 1 >= len(argv):
                print("%s wants a value" % a)
                return 2
            opts[a] = argv[i + 1]
            i += 2
            continue
        if not a.startswith("--"):
            args.append(a)
        i += 1

    objdump = opts.get("--objdump", DEFAULT_OBJDUMP)
    slice_spec = opts.get("--slice")
    if slice_spec is not None and ":" not in slice_spec:
        print("--slice wants PARENT:HELPER")
        return 2
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

    if slice_spec:
        parent, helper = slice_spec.split(":", 1)
        ok, why, churn = check_slice(only_a, only_b, changed, parent, helper,
                                     args[0], args[1])
        if ok:
            print("\nSLICE OK - %s split out of %s." % (helper, parent))
            if churn["inlined_back"]:
                print("  helper inlined straight back and the object is "
                      "OTHERWISE UNCHANGED -")
                print("  the extraction was purely a source reorganization.")
            else:
                print("  parent changed, helper emitted, no global symbol moved.")
            print("  local inliner churn: %d gone, %d appeared, %d re-optimized"
                  % (len(churn["gone"]), len(churn["appeared"]),
                     len(churn["changed"])))
            for k in churn["changed"][:8]:
                print("    re-optimized: %s" % k)
            print("  NOT a semantic proof: the golden decision trace is the "
                  "gate that carries the weight.")
            return 0
        print("\nSLICE SHAPE VIOLATED:")
        for r in why:
            print("  - %s" % r)
        return 1

    print("\nNOT EQUIVALENT - %d changed, %d gone, %d new."
          % (len(changed), len(only_a), len(only_b)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
