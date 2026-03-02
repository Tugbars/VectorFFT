#!/usr/bin/env python3
"""
spill_analysis.py — Register spill analysis for x86 SIMD assembly

Usage:
    # From C source (compiles automatically):
    python3 spill_analysis.py source.c -I include/dir -O2

    # From pre-compiled assembly:
    python3 spill_analysis.py output.s

    # Filter to specific functions:
    python3 spill_analysis.py output.s -f radix8_dif -f radix4_dit

    # Show per-block detail in hot functions:
    python3 spill_analysis.py output.s --blocks

    # Compare two optimization levels:
    python3 spill_analysis.py source.c -I avx2 --compare -O2 -O3
"""

import argparse
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# YMM/ZMM spill: register → stack
RE_SPILL = re.compile(
    r'vmov[au]p[ds]\s+%(ymm|zmm)\d+\s*,\s*-?\d*\(%rsp'
)
# YMM/ZMM reload: stack → register
RE_RELOAD = re.compile(
    r'vmov[au]p[ds]\s+-?\d*\(%rsp[^,]*,\s*%(ymm|zmm)\d+'
)
# Any SIMD instruction (v-prefixed)
RE_SIMD = re.compile(r'^\tv[a-z]')
# FMA
RE_FMA = re.compile(r'^\tv(fmadd|fmsub|fnmadd|fnmsub)')
# Arithmetic SIMD (add/sub/mul/div, packed double/single)
RE_ARITH = re.compile(r'^\tv(add|sub|mul|div)[ps][ds]')
# Function label
RE_FUNC = re.compile(r'^([a-zA-Z_]\w*):')
# Local label
RE_LOCAL_LABEL = re.compile(r'^(\.L\d+):')
# Jump (any conditional or unconditional)
RE_JUMP = re.compile(r'^\tj[a-z]+\s+(\.L\d+)')
# Stack frame allocation
RE_FRAME = re.compile(r'subq\s+\$(\d+),\s*%rsp')
# Spill offset extraction
RE_SPILL_OFFSET = re.compile(r'(-?\d+)\(%rsp\)')
# Register in spill/reload
RE_SPILL_REG = re.compile(r'%(ymm|zmm)(\d+)')
# Any instruction line
RE_INSTR = re.compile(r'^\t[a-z]')


@dataclass
class Block:
    label: str = ""
    instr_count: int = 0
    simd_count: int = 0
    fma_count: int = 0
    arith_count: int = 0
    spill_count: int = 0
    reload_count: int = 0
    spill_offsets: set = field(default_factory=set)
    spill_regs: defaultdict = field(default_factory=lambda: defaultdict(int))
    reload_regs: defaultdict = field(default_factory=lambda: defaultdict(int))


@dataclass
class Function:
    name: str = ""
    blocks: list = field(default_factory=list)
    frame_size: int = 0
    total_instr: int = 0
    total_simd: int = 0
    total_fma: int = 0
    total_arith: int = 0
    total_spill: int = 0
    total_reload: int = 0
    spill_offsets: set = field(default_factory=set)
    spill_regs: defaultdict = field(default_factory=lambda: defaultdict(int))
    reload_regs: defaultdict = field(default_factory=lambda: defaultdict(int))
    back_edges: list = field(default_factory=list)

    @property
    def total_traffic(self):
        return self.total_spill + self.total_reload

    @property
    def spill_slots(self):
        return len(self.spill_offsets)

    @property
    def frame_ymm_slots(self):
        return self.frame_size // 32 if self.frame_size else 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_assembly(lines, func_filter=None):
    """Parse assembly lines into Function objects."""
    functions = []
    cur_func = None
    cur_block = None
    label_nums = {}  # label -> first-seen order

    def finish_block():
        nonlocal cur_block
        if cur_block and cur_func and cur_block.instr_count > 0:
            cur_func.blocks.append(cur_block)
        cur_block = None

    def finish_func():
        nonlocal cur_func
        finish_block()
        if cur_func and cur_func.blocks:
            # Aggregate
            for b in cur_func.blocks:
                cur_func.total_instr += b.instr_count
                cur_func.total_simd += b.simd_count
                cur_func.total_fma += b.fma_count
                cur_func.total_arith += b.arith_count
                cur_func.total_spill += b.spill_count
                cur_func.total_reload += b.reload_count
                cur_func.spill_offsets |= b.spill_offsets
                for r, c in b.spill_regs.items():
                    cur_func.spill_regs[r] += c
                for r, c in b.reload_regs.items():
                    cur_func.reload_regs[r] += c
            if func_filter is None or any(f in cur_func.name for f in func_filter):
                if cur_func.total_instr > 0:
                    functions.append(cur_func)
        cur_func = None

    for line in lines:
        line = line.rstrip('\n')

        # Function start
        m = RE_FUNC.match(line)
        if m:
            finish_func()
            cur_func = Function(name=m.group(1))
            cur_block = Block(label="entry")
            label_nums.clear()
            continue

        if cur_func is None:
            continue

        # .size directive → end of function
        if line.startswith('\t.size\t' + cur_func.name):
            finish_func()
            continue

        # Stack frame
        m = RE_FRAME.search(line)
        if m:
            sz = int(m.group(1))
            if sz > cur_func.frame_size:
                cur_func.frame_size = sz

        # Local label
        m = RE_LOCAL_LABEL.match(line)
        if m:
            finish_block()
            lbl = m.group(1)
            cur_block = Block(label=lbl)
            if lbl not in label_nums:
                label_nums[lbl] = len(label_nums)
            continue

        # Instructions
        if not RE_INSTR.match(line):
            continue

        if cur_block is None:
            cur_block = Block(label="???")

        cur_block.instr_count += 1

        if RE_SIMD.match(line):
            cur_block.simd_count += 1
        if RE_FMA.match(line):
            cur_block.fma_count += 1
        if RE_ARITH.match(line):
            cur_block.arith_count += 1

        if RE_SPILL.search(line):
            cur_block.spill_count += 1
            m2 = RE_SPILL_OFFSET.search(line)
            if m2:
                cur_block.spill_offsets.add(int(m2.group(1)))
            m2 = RE_SPILL_REG.search(line)
            if m2:
                cur_block.spill_regs[f"%{m2.group(1)}{m2.group(2)}"] += 1

        if RE_RELOAD.search(line):
            cur_block.reload_count += 1
            m2 = RE_SPILL_OFFSET.search(line)
            if m2:
                cur_block.spill_offsets.add(int(m2.group(1)))
            m2 = re.search(r'%(ymm|zmm)(\d+)\s*$', line)
            if m2:
                cur_block.reload_regs[f"%{m2.group(1)}{m2.group(2)}"] += 1

        # Back-edge detection
        m = RE_JUMP.match(line)
        if m and cur_block:
            target = m.group(1)
            if target in label_nums:
                cur_label = cur_block.label
                if cur_label in label_nums and label_nums[target] <= label_nums[cur_label]:
                    cur_func.back_edges.append((cur_label, target))

    finish_func()
    return functions


# ---------------------------------------------------------------------------
# Compilation helper
# ---------------------------------------------------------------------------

def compile_to_asm(source, extra_flags, opt_level):
    """Compile a C source file to assembly, return list of lines."""
    with tempfile.NamedTemporaryFile(suffix='.s', delete=False) as f:
        asm_path = f.name

    cmd = ['gcc', f'-{opt_level}', '-mavx2', '-mfma', '-S',
           '-o', asm_path, source] + extra_flags
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try with warnings only
        stderr_lines = result.stderr.strip().split('\n')
        errors = [l for l in stderr_lines if 'error:' in l]
        if errors:
            print(f"Compilation failed:\n" + '\n'.join(errors[:5]), file=sys.stderr)
            sys.exit(1)

    with open(asm_path) as f:
        lines = f.readlines()
    Path(asm_path).unlink(missing_ok=True)
    return lines


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_function(func, show_blocks=False, indent=""):
    """Print spill report for one function."""
    p = indent
    hdr = f"{func.name}"
    print(f"{p}{hdr}")
    print(f"{p}{'─' * len(hdr)}")
    print(f"{p}  Stack frame:       {func.frame_size} bytes "
          f"({func.frame_ymm_slots} YMM slots)")
    print(f"{p}  Unique spill slots: {func.spill_slots}")
    print(f"{p}  Instructions:      {func.total_instr} total, "
          f"{func.total_simd} SIMD, {func.total_fma} FMA, "
          f"{func.total_arith} arith")
    print(f"{p}  Spill stores:      {func.total_spill}")
    print(f"{p}  Spill reloads:     {func.total_reload}")
    total = func.total_traffic
    if func.total_simd > 0:
        pct = total * 100 / func.total_simd
        print(f"{p}  Spill traffic:     {total}  "
              f"({pct:.1f}% of SIMD ops)")
    else:
        print(f"{p}  Spill traffic:     {total}")

    if func.back_edges:
        print(f"{p}  Loop back-edges:   {len(func.back_edges)}")

    # Top spilled registers
    if func.spill_regs:
        top = sorted(func.spill_regs.items(), key=lambda x: -x[1])[:5]
        regs_str = ", ".join(f"{r}×{c}" for r, c in top)
        print(f"{p}  Top spilled regs:  {regs_str}")

    if show_blocks:
        # Show blocks with spills, sorted by traffic
        hot = [b for b in func.blocks if b.spill_count + b.reload_count > 0]
        hot.sort(key=lambda b: -(b.spill_count + b.reload_count))

        if hot:
            print(f"{p}")
            print(f"{p}  {'Block':<10} {'Insts':>5} {'SIMD':>5} "
                  f"{'Spill':>5} {'Reload':>6} {'Total':>5} {'%':>5}")
            print(f"{p}  {'─'*10} {'─'*5} {'─'*5} {'─'*5} {'─'*6} {'─'*5} {'─'*5}")
            for b in hot[:15]:
                traffic = b.spill_count + b.reload_count
                pct = (traffic * 100 / b.simd_count) if b.simd_count else 0
                is_loop = any(b.label == be[1] or b.label == be[0]
                              for be in func.back_edges)
                marker = " ◄loop" if is_loop else ""
                print(f"{p}  {b.label:<10} {b.instr_count:>5} "
                      f"{b.simd_count:>5} {b.spill_count:>5} "
                      f"{b.reload_count:>6} {traffic:>5} "
                      f"{pct:>4.0f}%{marker}")

    print()


def report(functions, show_blocks=False):
    """Print full report."""
    if not functions:
        print("No functions found (check -f filter?)")
        return

    total_spill = sum(f.total_spill for f in functions)
    total_reload = sum(f.total_reload for f in functions)

    print()
    print("═" * 72)
    print(" REGISTER SPILL ANALYSIS")
    print("═" * 72)
    print()
    print(f"  Functions analyzed: {len(functions)}")
    print(f"  Total spill stores:  {total_spill}")
    print(f"  Total spill reloads: {total_reload}")
    print(f"  Total spill traffic: {total_spill + total_reload}")
    print()

    # Sort by spill traffic descending
    for func in sorted(functions, key=lambda f: -f.total_traffic):
        if func.total_traffic > 0 or show_blocks:
            report_function(func, show_blocks)

    # Functions with zero spills
    clean = [f for f in functions if f.total_traffic == 0]
    if clean:
        names = ", ".join(f.name for f in clean)
        print(f"  Zero spills: {names}")
        print()


def compare_report(funcs_a, funcs_b, label_a, label_b):
    """Side-by-side comparison of two optimization levels."""
    print()
    print("═" * 72)
    print(f" COMPARISON: {label_a} vs {label_b}")
    print("═" * 72)
    print()

    names_a = {f.name: f for f in funcs_a}
    names_b = {f.name: f for f in funcs_b}
    all_names = sorted(set(names_a) | set(names_b))

    print(f"  {'Function':<45} {label_a:>8} {label_b:>8} {'Delta':>8}")
    print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*8}")

    for name in all_names:
        fa = names_a.get(name)
        fb = names_b.get(name)
        ta = fa.total_traffic if fa else 0
        tb = fb.total_traffic if fb else 0
        if ta == 0 and tb == 0:
            continue
        delta = tb - ta
        sign = "+" if delta > 0 else ""
        # Truncate name
        short = name[:44] if len(name) > 44 else name
        print(f"  {short:<45} {ta:>8} {tb:>8} {sign}{delta:>7}")

    ta = sum(f.total_traffic for f in funcs_a)
    tb = sum(f.total_traffic for f in funcs_b)
    delta = tb - ta
    sign = "+" if delta > 0 else ""
    print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'TOTAL':<45} {ta:>8} {tb:>8} {sign}{delta:>7}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze register spills in x86 SIMD assembly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input', help='Assembly (.s) or C source (.c/.h) file')
    parser.add_argument('-f', '--filter', action='append', default=None,
                        help='Only show functions containing this substring '
                             '(can repeat)')
    parser.add_argument('-I', '--include', action='append', default=[],
                        help='Include directory (for C compilation)')
    parser.add_argument('-O', '--opt', default='O2',
                        help='Optimization level (default: O2)')
    parser.add_argument('--blocks', action='store_true',
                        help='Show per-basic-block spill detail')
    parser.add_argument('--compare', nargs=2, metavar='OPT',
                        help='Compare two optimization levels, e.g. --compare O2 O3')
    parser.add_argument('--extra', action='append', default=[],
                        help='Extra compiler flags')

    args = parser.parse_args()

    extra = []
    for inc in args.include:
        extra.extend(['-I', inc])
    extra.extend(args.extra)

    is_source = args.input.endswith(('.c', '.h', '.cpp'))

    if args.compare:
        if not is_source:
            print("--compare requires a C source file, not assembly",
                  file=sys.stderr)
            sys.exit(1)
        lines_a = compile_to_asm(args.input, extra, args.compare[0])
        lines_b = compile_to_asm(args.input, extra, args.compare[1])
        funcs_a = parse_assembly(lines_a, args.filter)
        funcs_b = parse_assembly(lines_b, args.filter)
        compare_report(funcs_a, funcs_b, args.compare[0], args.compare[1])
        # Also show detail for the worse one
        worse = funcs_b if sum(f.total_traffic for f in funcs_b) >= \
                           sum(f.total_traffic for f in funcs_a) else funcs_a
        report(worse, args.blocks)
    else:
        if is_source:
            lines = compile_to_asm(args.input, extra, args.opt)
        else:
            with open(args.input) as f:
                lines = f.readlines()

        functions = parse_assembly(lines, args.filter)
        report(functions, args.blocks)


if __name__ == '__main__':
    main()