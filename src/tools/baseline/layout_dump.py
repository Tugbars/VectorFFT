#!/usr/bin/env python3
"""layout_dump.py - every struct's size and member offsets in vfft.c's TU.

WHY THIS EXISTS
---------------
Splitting `struct vfft_plan_s` (and moving the structs that live in the mixed
headers) can reorder fields. A reorder is a real change - every access moves -
yet obj_equiv's default mode normalizes displacements away, so it would pass.
This artifact states the layout outright: one block per struct type, its
byte size, and each member's offset, read from the compiler's own DWARF.

How: compile vfft.c at -O0 -g (types only; the code is irrelevant) with
-fno-eliminate-unused-debug-types, so a struct that nothing uses yet is still
listed, then walk `readelf --debug-dump=info`. Anonymous structs and unions are
named by their first member, `<anon:first>`, so a regrouping into anonymous
sub-structs is visible too. Types from system headers are listed as well; they
never change and cost nothing.

USAGE
  python src/tools/baseline/layout_dump.py --isa avx2|avx512 [--out FILE]
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toolchain  # noqa: E402

_DIE = re.compile(r"^\s*<(\d+)><([0-9a-f]+)>: Abbrev Number: \d+ \((DW_TAG_\w+)\)")
_ATTR = re.compile(r"^\s*<[0-9a-f]+>\s+(DW_AT_\w+)\s*:\s*(.*)$")


def _name(v):
    # "(indirect string, offset: 0x1234): foo"  or plain "foo"
    return v.split("): ", 1)[1].strip() if "): " in v else re.sub(r"^\([^)]*\)\s*", "", v.strip())


def _num(v):
    # "(data1) 16" (the form in parentheses first, on binutils 2.42)
    v = re.sub(r"^\([^)]*\)\s*", "", v.strip())
    return int(v.split()[0], 0)


def dump(obj):
    out = subprocess.run([toolchain.readelf(), "--debug-dump=info", "-W", obj],
                         capture_output=True, text=True, check=True).stdout
    structs = []       # [kind, name, size, [(member, offset)], depth, die offset]
    stack = []         # open struct/union per depth
    typedefs = {}      # target DIE offset -> typedef name (typedef struct {..} x_t)
    cur_die = None
    for line in out.splitlines():
        m = _DIE.match(line)
        if m:
            depth, tag = int(m.group(1)), m.group(3)
            die = int(m.group(2), 16)
            while stack and stack[-1][4] >= depth:
                stack.pop()
            cur_die = None
            if tag in ("DW_TAG_structure_type", "DW_TAG_union_type"):
                s = ["struct" if tag == "DW_TAG_structure_type" else "union",
                     None, None, [], depth, die]
                structs.append(s)
                stack.append(s)
                cur_die = ("agg", s)
            elif tag == "DW_TAG_typedef":
                cur_die = ("td", [None, None])
            elif tag == "DW_TAG_member" and stack and stack[-1][4] == depth - 1:
                mem = [None, None]
                stack[-1][3].append(mem)
                cur_die = ("mem", mem)
            continue
        a = _ATTR.match(line)
        if not a or cur_die is None:
            continue
        at, val = a.group(1), a.group(2)
        kind, obj_ = cur_die
        if kind == "td":
            if at == "DW_AT_name":
                obj_[0] = _name(val)
            elif at == "DW_AT_type":
                tm = re.search(r"<0x([0-9a-f]+)>", val)
                obj_[1] = int(tm.group(1), 16) if tm else -1
            if obj_[0] and obj_[1] is not None:
                typedefs.setdefault(obj_[1], obj_[0])
            continue
        if kind == "agg":
            if at == "DW_AT_name":
                obj_[1] = _name(val)
            elif at == "DW_AT_byte_size":
                obj_[2] = _num(val)
            elif at == "DW_AT_declaration":
                obj_[2] = "decl"
        else:
            if at == "DW_AT_name":
                obj_[0] = _name(val)
            elif at == "DW_AT_data_member_location":
                obj_[1] = _num(val)
            elif at == "DW_AT_data_bit_offset":
                obj_[1] = "bit%d" % _num(val)
    rows = set()
    for kind, name, size, mems, _, die in structs:
        if size == "decl" or size is None:
            continue
        if not name and die in typedefs:
            name = typedefs[die]
        if not name:
            first = next((m[0] for m in mems if m[0]), "?")
            name = "<anon:%s>" % first
        body = "\n".join("  %-40s %s" % (m[0] or "<anon>", m[1]) for m in mems)
        rows.add("%s %s size=%d\n%s" % (kind, name, size, body))
    return sorted(rows)


def main():
    isa = sys.argv[sys.argv.index("--isa") + 1] if "--isa" in sys.argv else "avx2"
    outp = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    with tempfile.TemporaryDirectory() as d:
        obj = os.path.join(d, "vfft_types.o")
        flags = ["-O0", "-g", "-fno-eliminate-unused-debug-types", "-w", "-c"]
        flags += toolchain.identity_flags(isa)[1:]      # the ISA, not the -O
        r = toolchain.run([toolchain.cc()] + flags + toolchain.include_flags()
                          + [os.path.join(toolchain.CORE, "vfft.c"), "-o", obj])
        if r.returncode != 0:
            raise SystemExit("layout_dump: compile failed\n" + r.stderr[-3000:])
        rows = dump(obj)
    text = "".join(r + "\n" for r in rows)
    if outp:
        with open(outp, "w", newline="\n") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
