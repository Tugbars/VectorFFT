#!/usr/bin/env python3
"""
gen_radix64_n1.py — Generate DFT-64 N1 kernels for scalar and AVX2

8×8 decomposition: 8 radix-8 sub-FFTs + 8 radix-8 column combines.
Native forward + backward. Hoisted twiddle broadcasts (AVX2).

Usage:
  python3 gen_radix64_n1.py scalar > fft_radix64_scalar_n1_gen.h
  python3 gen_radix64_n1.py avx2   > fft_radix64_avx2_n1_gen.h
"""

import math, sys

# ──────────────────────────────────────────────────────────────────
# ISA abstraction
# ──────────────────────────────────────────────────────────────────

class ISA_Scalar:
    name = 'scalar'
    vtype = 'double'
    width = 1
    k_step = 1
    target_attr = ''
    macro_prefix = 'R64S'
    func_suffix = 'scalar'
    sign_flip_decl = None
    sqrt2_decl = 'const double sqrt2_inv = 0.70710678118654752440;'

class ISA_AVX2:
    name = 'avx2'
    vtype = '__m256d'
    width = 4
    k_step = 4
    target_attr = '__attribute__((target("avx2,fma")))'
    macro_prefix = 'R64A'
    func_suffix = 'avx2'
    sign_flip_decl = 'const __m256d sign_flip = _mm256_set1_pd(-0.0);'
    sqrt2_decl = 'const __m256d sqrt2_inv = _mm256_set1_pd(0.70710678118654752440);'
    p = '_mm256'
    align = 32

class ISA_AVX512:
    name = 'avx512'
    vtype = '__m512d'
    width = 8
    k_step = 8
    target_attr = '__attribute__((target("avx512f,avx512dq,fma")))'
    macro_prefix = 'R64Z'
    func_suffix = 'avx512'
    sign_flip_decl = 'const __m512d sign_flip = _mm512_set1_pd(-0.0);'
    sqrt2_decl = 'const __m512d sqrt2_inv = _mm512_set1_pd(0.70710678118654752440);'
    p = '_mm512'
    align = 64

ISAS = {'scalar': ISA_Scalar, 'avx2': ISA_AVX2, 'avx512': ISA_AVX512}

# ──────────────────────────────────────────────────────────────────
# Twiddle helpers
# ──────────────────────────────────────────────────────────────────

def wN(e, N):
    e = e % N
    angle = 2.0 * math.pi * e / N
    return (math.cos(angle), -math.sin(angle))

def wN_label(e, N):
    return f"W{N}_{e % N}"

def twiddle_is_trivial(e, N):
    e = e % N
    if e == 0:
        return True, 'one'
    if (8 * e) % N == 0:
        octant = (8 * e) // N
        types = ['one', 'w8_1', 'neg_j', 'w8_3',
                 'neg_one', 'neg_w8_1', 'pos_j', 'neg_w8_3']
        return True, types[octant % 8]
    return False, 'cmul'

# ──────────────────────────────────────────────────────────────────
# Emitter
# ──────────────────────────────────────────────────────────────────

class Emitter:
    def __init__(self, isa):
        self.isa = isa
        self.lines = []
        self.indent = 1
        self.spill_count = 0
        self.reload_count = 0
        self.twiddles_needed = set()

    def emit(self, line=""):
        self.lines.append("    " * self.indent + line)

    def comment(self, text):
        self.emit(f"/* {text} */")

    def blank(self):
        self.lines.append("")

    def emit_spill(self, var, slot):
        I = self.isa
        if I.name == 'scalar':
            self.emit(f"spill_re[{slot}] = {var}_re;")
            self.emit(f"spill_im[{slot}] = {var}_im;")
        else:
            self.emit(f"{I.p}_store_pd(&spill_re[{slot} * {I.width}], {var}_re);")
            self.emit(f"{I.p}_store_pd(&spill_im[{slot} * {I.width}], {var}_im);")
        self.spill_count += 1

    def emit_reload(self, var, slot):
        I = self.isa
        if I.name == 'scalar':
            self.emit(f"{var}_re = spill_re[{slot}];")
            self.emit(f"{var}_im = spill_im[{slot}];")
        else:
            self.emit(f"{var}_re = {I.p}_load_pd(&spill_re[{slot} * {I.width}]);")
            self.emit(f"{var}_im = {I.p}_load_pd(&spill_im[{slot} * {I.width}]);")
        self.reload_count += 1

    def emit_load_input(self, var, n):
        mp = self.isa.macro_prefix
        self.emit(f"{var}_re = {mp}_LD(&in_re[{n} * K + k]);")
        self.emit(f"{var}_im = {mp}_LD(&in_im[{n} * K + k]);")

    def emit_store_output(self, var, m):
        mp = self.isa.macro_prefix
        self.emit(f"{mp}_ST(&out_re[{m} * K + k], {var}_re);")
        self.emit(f"{mp}_ST(&out_im[{m} * K + k], {var}_im);")

    # ── ISA-polymorphic arithmetic ──

    def _add(self, a, b):
        if self.isa.name == 'scalar': return f"({a} + {b})"
        return f"{self.isa.p}_add_pd({a}, {b})"

    def _sub(self, a, b):
        if self.isa.name == 'scalar': return f"({a} - {b})"
        return f"{self.isa.p}_sub_pd({a}, {b})"

    def _mul(self, a, b):
        if self.isa.name == 'scalar': return f"({a} * {b})"
        return f"{self.isa.p}_mul_pd({a}, {b})"

    def _neg(self, a):
        if self.isa.name == 'scalar': return f"(-{a})"
        return f"{self.isa.p}_xor_pd({a}, sign_flip)"

    def _fmadd(self, a, b, c):
        if self.isa.name == 'scalar': return f"({a} * {b} + {c})"
        return f"{self.isa.p}_fmadd_pd({a}, {b}, {c})"

    def _fmsub(self, a, b, c):
        if self.isa.name == 'scalar': return f"({a} * {b} - {c})"
        return f"{self.isa.p}_fmsub_pd({a}, {b}, {c})"

    # ── Radix-8 butterfly ──

    def emit_radix8(self, v, direction, comment_str=""):
        assert len(v) == 8
        fwd = (direction == 'fwd')
        T = self.isa.vtype
        if comment_str:
            self.comment(f"{comment_str} [{direction}]")

        self.emit(f"{{ {T} e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;")
        self.emit(f"  {T} t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i;")
        # Even radix-4: v[0], v[2], v[4], v[6]
        self.emit(f"  t0r = {self._add(f'{v[0]}_re', f'{v[4]}_re')}; t0i = {self._add(f'{v[0]}_im', f'{v[4]}_im')};")
        self.emit(f"  t1r = {self._sub(f'{v[0]}_re', f'{v[4]}_re')}; t1i = {self._sub(f'{v[0]}_im', f'{v[4]}_im')};")
        self.emit(f"  t2r = {self._add(f'{v[2]}_re', f'{v[6]}_re')}; t2i = {self._add(f'{v[2]}_im', f'{v[6]}_im')};")
        self.emit(f"  t3r = {self._sub(f'{v[2]}_re', f'{v[6]}_re')}; t3i = {self._sub(f'{v[2]}_im', f'{v[6]}_im')};")
        self.emit(f"  e0r = {self._add('t0r', 't2r')}; e0i = {self._add('t0i', 't2i')};")
        self.emit(f"  e2r = {self._sub('t0r', 't2r')}; e2i = {self._sub('t0i', 't2i')};")
        if fwd:
            self.emit(f"  e1r = {self._add('t1r', 't3i')}; e1i = {self._sub('t1i', 't3r')};")
            self.emit(f"  e3r = {self._sub('t1r', 't3i')}; e3i = {self._add('t1i', 't3r')};")
        else:
            self.emit(f"  e1r = {self._sub('t1r', 't3i')}; e1i = {self._add('t1i', 't3r')};")
            self.emit(f"  e3r = {self._add('t1r', 't3i')}; e3i = {self._sub('t1i', 't3r')};")

        # Odd radix-4: v[1], v[3], v[5], v[7]
        self.emit(f"  {T} o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;")
        self.emit(f"  t0r = {self._add(f'{v[1]}_re', f'{v[5]}_re')}; t0i = {self._add(f'{v[1]}_im', f'{v[5]}_im')};")
        self.emit(f"  t1r = {self._sub(f'{v[1]}_re', f'{v[5]}_re')}; t1i = {self._sub(f'{v[1]}_im', f'{v[5]}_im')};")
        self.emit(f"  t2r = {self._add(f'{v[3]}_re', f'{v[7]}_re')}; t2i = {self._add(f'{v[3]}_im', f'{v[7]}_im')};")
        self.emit(f"  t3r = {self._sub(f'{v[3]}_re', f'{v[7]}_re')}; t3i = {self._sub(f'{v[3]}_im', f'{v[7]}_im')};")
        self.emit(f"  o0r = {self._add('t0r', 't2r')}; o0i = {self._add('t0i', 't2i')};")
        self.emit(f"  o2r = {self._sub('t0r', 't2r')}; o2i = {self._sub('t0i', 't2i')};")
        if fwd:
            self.emit(f"  o1r = {self._add('t1r', 't3i')}; o1i = {self._sub('t1i', 't3r')};")
            self.emit(f"  o3r = {self._sub('t1r', 't3i')}; o3i = {self._add('t1i', 't3r')};")
        else:
            self.emit(f"  o1r = {self._sub('t1r', 't3i')}; o1i = {self._add('t1i', 't3r')};")
            self.emit(f"  o3r = {self._add('t1r', 't3i')}; o3i = {self._sub('t1i', 't3r')};")

        # W₈ twiddle
        if fwd:
            self.emit(f"  t0r = {self._mul(self._add('o1r', 'o1i'), 'sqrt2_inv')};")
            self.emit(f"  t0i = {self._mul(self._sub('o1i', 'o1r'), 'sqrt2_inv')};")
            self.emit(f"  o1r = t0r; o1i = t0i;")
            self.emit(f"  t0r = o2i; t0i = {self._neg('o2r')};")
            self.emit(f"  o2r = t0r; o2i = t0i;")
            self.emit(f"  t0r = {self._mul(self._sub('o3i', 'o3r'), 'sqrt2_inv')};")
            self.emit(f"  t0i = {self._neg(self._mul(self._add('o3r', 'o3i'), 'sqrt2_inv'))};")
            self.emit(f"  o3r = t0r; o3i = t0i;")
        else:
            self.emit(f"  t0r = {self._mul(self._sub('o1r', 'o1i'), 'sqrt2_inv')};")
            self.emit(f"  t0i = {self._mul(self._add('o1r', 'o1i'), 'sqrt2_inv')};")
            self.emit(f"  o1r = t0r; o1i = t0i;")
            self.emit(f"  t0r = {self._neg('o2i')}; t0i = o2r;")
            self.emit(f"  o2r = t0r; o2i = t0i;")
            self.emit(f"  t0r = {self._neg(self._mul(self._add('o3r', 'o3i'), 'sqrt2_inv'))};")
            self.emit(f"  t0i = {self._mul(self._sub('o3r', 'o3i'), 'sqrt2_inv')};")
            self.emit(f"  o3r = t0r; o3i = t0i;")

        # Combine
        self.emit(f"  {v[0]}_re = {self._add('e0r', 'o0r')}; {v[0]}_im = {self._add('e0i', 'o0i')};")
        self.emit(f"  {v[4]}_re = {self._sub('e0r', 'o0r')}; {v[4]}_im = {self._sub('e0i', 'o0i')};")
        self.emit(f"  {v[1]}_re = {self._add('e1r', 'o1r')}; {v[1]}_im = {self._add('e1i', 'o1i')};")
        self.emit(f"  {v[5]}_re = {self._sub('e1r', 'o1r')}; {v[5]}_im = {self._sub('e1i', 'o1i')};")
        self.emit(f"  {v[2]}_re = {self._add('e2r', 'o2r')}; {v[2]}_im = {self._add('e2i', 'o2i')};")
        self.emit(f"  {v[6]}_re = {self._sub('e2r', 'o2r')}; {v[6]}_im = {self._sub('e2i', 'o2i')};")
        self.emit(f"  {v[3]}_re = {self._add('e3r', 'o3r')}; {v[3]}_im = {self._add('e3i', 'o3i')};")
        self.emit(f"  {v[7]}_re = {self._sub('e3r', 'o3r')}; {v[7]}_im = {self._sub('e3i', 'o3i')};")
        self.emit(f"}}")

    def emit_radix8_split(self, v, direction, comment_str=""):
        """Split radix-8 for AVX2 (16 YMM). Even/odd/combine with explicit bfr/bfi spill."""
        assert len(v) == 8
        fwd = (direction == 'fwd')
        T = self.isa.vtype
        W = self.isa.width
        P = self.isa.p
        if comment_str:
            self.comment(f"{comment_str} [{direction}] (split)")

        # Phase 1: Even DFT-4 → spill to bfr/bfi
        self.comment('Phase 1: Even DFT-4 → spill A0..A3')
        self.emit(f"{{ {T} t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        self.emit(f"  t0r = {self._add(f'{v[0]}_re', f'{v[4]}_re')}; t0i = {self._add(f'{v[0]}_im', f'{v[4]}_im')};")
        self.emit(f"  t1r = {self._sub(f'{v[0]}_re', f'{v[4]}_re')}; t1i = {self._sub(f'{v[0]}_im', f'{v[4]}_im')};")
        self.emit(f"  t2r = {self._add(f'{v[2]}_re', f'{v[6]}_re')}; t2i = {self._add(f'{v[2]}_im', f'{v[6]}_im')};")
        self.emit(f"  t3r = {self._sub(f'{v[2]}_re', f'{v[6]}_re')}; t3i = {self._sub(f'{v[2]}_im', f'{v[6]}_im')};")
        self.emit(f"  {P}_store_pd(&bfr[0*{W}], {self._add('t0r','t2r')}); {P}_store_pd(&bfi[0*{W}], {self._add('t0i','t2i')});")
        self.emit(f"  {P}_store_pd(&bfr[2*{W}], {self._sub('t0r','t2r')}); {P}_store_pd(&bfi[2*{W}], {self._sub('t0i','t2i')});")
        if fwd:
            self.emit(f"  {P}_store_pd(&bfr[1*{W}], {self._add('t1r','t3i')}); {P}_store_pd(&bfi[1*{W}], {self._sub('t1i','t3r')});")
            self.emit(f"  {P}_store_pd(&bfr[3*{W}], {self._sub('t1r','t3i')}); {P}_store_pd(&bfi[3*{W}], {self._add('t1i','t3r')});")
        else:
            self.emit(f"  {P}_store_pd(&bfr[1*{W}], {self._sub('t1r','t3i')}); {P}_store_pd(&bfi[1*{W}], {self._add('t1i','t3r')});")
            self.emit(f"  {P}_store_pd(&bfr[3*{W}], {self._add('t1r','t3i')}); {P}_store_pd(&bfi[3*{W}], {self._sub('t1i','t3r')});")
        self.emit(f"}}")

        # Phase 2: Odd DFT-4 + W8 twiddles
        self.comment('Phase 2: Odd DFT-4 + W8 twiddles')
        self.emit(f"{{ {T} o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i, t0r,t0i;")
        self.emit(f"  t0r = {self._add(f'{v[1]}_re', f'{v[5]}_re')}; t0i = {self._add(f'{v[1]}_im', f'{v[5]}_im')};")
        self.emit(f"  {{ {T} t1r = {self._sub(f'{v[1]}_re', f'{v[5]}_re')}, t1i = {self._sub(f'{v[1]}_im', f'{v[5]}_im')};")
        self.emit(f"    {T} t2r = {self._add(f'{v[3]}_re', f'{v[7]}_re')}, t2i = {self._add(f'{v[3]}_im', f'{v[7]}_im')};")
        self.emit(f"    {T} t3r = {self._sub(f'{v[3]}_re', f'{v[7]}_re')}, t3i = {self._sub(f'{v[3]}_im', f'{v[7]}_im')};")
        self.emit(f"    o0r = {self._add('t0r', 't2r')}; o0i = {self._add('t0i', 't2i')};")
        self.emit(f"    o2r = {self._sub('t0r', 't2r')}; o2i = {self._sub('t0i', 't2i')};")
        if fwd:
            self.emit(f"    o1r = {self._add('t1r', 't3i')}; o1i = {self._sub('t1i', 't3r')};")
            self.emit(f"    o3r = {self._sub('t1r', 't3i')}; o3i = {self._add('t1i', 't3r')};")
        else:
            self.emit(f"    o1r = {self._sub('t1r', 't3i')}; o1i = {self._add('t1i', 't3r')};")
            self.emit(f"    o3r = {self._add('t1r', 't3i')}; o3i = {self._sub('t1i', 't3r')};")
        self.emit(f"  }}")

        # W8 twiddles on o1, o2, o3
        if fwd:
            self.emit(f"  t0r = {self._mul(self._add('o1r', 'o1i'), 'sqrt2_inv')};")
            self.emit(f"  t0i = {self._mul(self._sub('o1i', 'o1r'), 'sqrt2_inv')};")
            self.emit(f"  o1r = t0r; o1i = t0i;")
            self.emit(f"  t0r = o2i; t0i = {self._neg('o2r')};")
            self.emit(f"  o2r = t0r; o2i = t0i;")
            self.emit(f"  t0r = {self._mul(self._sub('o3i', 'o3r'), 'sqrt2_inv')};")
            self.emit(f"  t0i = {self._neg(self._mul(self._add('o3r', 'o3i'), 'sqrt2_inv'))};")
            self.emit(f"  o3r = t0r; o3i = t0i;")
        else:
            self.emit(f"  t0r = {self._mul(self._sub('o1r', 'o1i'), 'sqrt2_inv')};")
            self.emit(f"  t0i = {self._mul(self._add('o1r', 'o1i'), 'sqrt2_inv')};")
            self.emit(f"  o1r = t0r; o1i = t0i;")
            self.emit(f"  t0r = {self._neg('o2i')}; t0i = o2r;")
            self.emit(f"  o2r = t0r; o2i = t0i;")
            self.emit(f"  t0r = {self._neg(self._mul(self._add('o3r', 'o3i'), 'sqrt2_inv'))};")
            self.emit(f"  t0i = {self._mul(self._sub('o3r', 'o3i'), 'sqrt2_inv')};")
            self.emit(f"  o3r = t0r; o3i = t0i;")

        # Phase 3: Reload A, combine incrementally
        self.comment('Phase 3: Reload A, combine A ± B (incremental)')
        self.emit(f"  {{ {T} Ar = {P}_load_pd(&bfr[0*{W}]), Ai = {P}_load_pd(&bfi[0*{W}]);")
        self.emit(f"    {v[0]}_re = {self._add('Ar', 'o0r')}; {v[0]}_im = {self._add('Ai', 'o0i')};")
        self.emit(f"    {v[4]}_re = {self._sub('Ar', 'o0r')}; {v[4]}_im = {self._sub('Ai', 'o0i')}; }}")
        self.emit(f"  {{ {T} Ar = {P}_load_pd(&bfr[1*{W}]), Ai = {P}_load_pd(&bfi[1*{W}]);")
        self.emit(f"    {v[1]}_re = {self._add('Ar', 'o1r')}; {v[1]}_im = {self._add('Ai', 'o1i')};")
        self.emit(f"    {v[5]}_re = {self._sub('Ar', 'o1r')}; {v[5]}_im = {self._sub('Ai', 'o1i')}; }}")
        self.emit(f"  {{ {T} Ar = {P}_load_pd(&bfr[2*{W}]), Ai = {P}_load_pd(&bfi[2*{W}]);")
        self.emit(f"    {v[2]}_re = {self._add('Ar', 'o2r')}; {v[2]}_im = {self._add('Ai', 'o2i')};")
        self.emit(f"    {v[6]}_re = {self._sub('Ar', 'o2r')}; {v[6]}_im = {self._sub('Ai', 'o2i')}; }}")
        self.emit(f"  {{ {T} Ar = {P}_load_pd(&bfr[3*{W}]), Ai = {P}_load_pd(&bfi[3*{W}]);")
        self.emit(f"    {v[3]}_re = {self._add('Ar', 'o3r')}; {v[3]}_im = {self._add('Ai', 'o3i')};")
        self.emit(f"    {v[7]}_re = {self._sub('Ar', 'o3r')}; {v[7]}_im = {self._sub('Ai', 'o3i')}; }}")
        self.emit(f"}}")

    def emit_r8(self, v, direction, comment_str=""):
        """ISA dispatch: split for AVX2 (16 regs), monolithic for AVX-512/scalar."""
        if self.isa.name == 'avx2':
            self.emit_radix8_split(v, direction, comment_str)
        else:
            self.emit_radix8(v, direction, comment_str)

    # ── Twiddle emission ──

    def emit_twiddle(self, dst, src, e, tN, direction):
        _, typ = twiddle_is_trivial(e, tN)
        fwd = (direction == 'fwd')
        T = self.isa.vtype

        if typ == 'one':
            if dst != src:
                self.emit(f"{dst}_re = {src}_re; {dst}_im = {src}_im;")
        elif typ == 'neg_one':
            self.emit(f"{dst}_re = {self._neg(f'{src}_re')}; {dst}_im = {self._neg(f'{src}_im')};")
        elif typ == 'neg_j':
            if fwd:
                self.emit(f"{{ const {T} t = {src}_re; {dst}_re = {src}_im; {dst}_im = {self._neg('t')}; }}")
            else:
                self.emit(f"{{ const {T} t = {src}_re; {dst}_re = {self._neg(f'{src}_im')}; {dst}_im = t; }}")
        elif typ == 'pos_j':
            if fwd:
                self.emit(f"{{ const {T} t = {src}_re; {dst}_re = {self._neg(f'{src}_im')}; {dst}_im = t; }}")
            else:
                self.emit(f"{{ const {T} t = {src}_re; {dst}_re = {src}_im; {dst}_im = {self._neg('t')}; }}")
        elif typ == 'w8_1':
            self.emit(f"{{ const {T} tr = {src}_re, ti = {src}_im;")
            if fwd:
                self.emit(f"  {dst}_re = {self._mul(self._add('tr', 'ti'), 'sqrt2_inv')};")
                self.emit(f"  {dst}_im = {self._mul(self._sub('ti', 'tr'), 'sqrt2_inv')}; }}")
            else:
                self.emit(f"  {dst}_re = {self._mul(self._sub('tr', 'ti'), 'sqrt2_inv')};")
                self.emit(f"  {dst}_im = {self._mul(self._add('tr', 'ti'), 'sqrt2_inv')}; }}")
        elif typ == 'w8_3':
            self.emit(f"{{ const {T} tr = {src}_re, ti = {src}_im;")
            if fwd:
                self.emit(f"  {dst}_re = {self._mul(self._sub('ti', 'tr'), 'sqrt2_inv')};")
                self.emit(f"  {dst}_im = {self._neg(self._mul(self._add('tr', 'ti'), 'sqrt2_inv'))}; }}")
            else:
                self.emit(f"  {dst}_re = {self._neg(self._mul(self._add('tr', 'ti'), 'sqrt2_inv'))};")
                self.emit(f"  {dst}_im = {self._mul(self._sub('tr', 'ti'), 'sqrt2_inv')}; }}")
        elif typ == 'neg_w8_1':
            self.emit(f"{{ const {T} tr = {src}_re, ti = {src}_im;")
            if fwd:
                self.emit(f"  {dst}_re = {self._neg(self._mul(self._add('tr', 'ti'), 'sqrt2_inv'))};")
                self.emit(f"  {dst}_im = {self._mul(self._sub('tr', 'ti'), 'sqrt2_inv')}; }}")
            else:
                self.emit(f"  {dst}_re = {self._mul(self._sub('ti', 'tr'), 'sqrt2_inv')};")
                self.emit(f"  {dst}_im = {self._neg(self._mul(self._add('tr', 'ti'), 'sqrt2_inv'))}; }}")
        elif typ == 'neg_w8_3':
            self.emit(f"{{ const {T} tr = {src}_re, ti = {src}_im;")
            if fwd:
                self.emit(f"  {dst}_re = {self._mul(self._sub('tr', 'ti'), 'sqrt2_inv')};")
                self.emit(f"  {dst}_im = {self._mul(self._add('tr', 'ti'), 'sqrt2_inv')}; }}")
            else:
                self.emit(f"  {dst}_re = {self._mul(self._add('tr', 'ti'), 'sqrt2_inv')};")
                self.emit(f"  {dst}_im = {self._mul(self._sub('ti', 'tr'), 'sqrt2_inv')}; }}")
        else:
            # General twiddle: inline broadcast from static array (zero hoisted regs)
            e_idx = e % tN
            if self.isa.name == 'scalar':
                wr = f'iw_re[{e_idx}]'
                wi = f'iw_im[{e_idx}]'
            else:
                wr = f'{self.isa.p}_set1_pd(iw_re[{e_idx}])'
                wi = f'{self.isa.p}_set1_pd(iw_im[{e_idx}])'
            self.emit(f"{{ const {T} tr = {src}_re;")
            if fwd:
                self.emit(f"  {dst}_re = {self._fmsub(f'{src}_re', wr, self._mul(f'{src}_im', wi))};")
                self.emit(f"  {dst}_im = {self._fmadd('tr', wi, self._mul(f'{src}_im', wr))}; }}")
            else:
                self.emit(f"  {dst}_re = {self._fmadd(f'{src}_re', wr, self._mul(f'{src}_im', wi))};")
                self.emit(f"  {dst}_im = {self._fmsub(f'{src}_im', wr, self._mul('tr', wi))}; }}")

# ──────────────────────────────────────────────────────────────────
# Twiddle collection for 8×8 DFT-64
# ──────────────────────────────────────────────────────────────────

def collect_twiddles():
    tw_set = set()
    N, N1, N2 = 64, 8, 8
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2 * k1) % N
            is_special, _ = twiddle_is_trivial(e, N)
            if not is_special:
                tw_set.add((e, N))
    return tw_set

# ──────────────────────────────────────────────────────────────────
# Kernel generation
# ──────────────────────────────────────────────────────────────────

def emit_kernel(em, direction, tw_set):
    N, N1, N2 = 64, 8, 8
    I = em.isa
    T = I.vtype

    if I.target_attr:
        em.lines.append(f"static {I.target_attr} void")
    else:
        em.lines.append(f"static void")
    em.lines.append(f"radix64_n1_dit_kernel_{direction}_{I.func_suffix}(")
    em.lines.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.lines.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.lines.append(f"    size_t K)")
    em.lines.append(f"{{")

    em.indent = 1
    em.spill_count = 0
    em.reload_count = 0

    if I.sign_flip_decl:
        em.emit(I.sign_flip_decl)
    em.emit(I.sqrt2_decl)
    em.blank()

    if I.name == 'scalar':
        em.emit(f"double spill_re[{N}];")
        em.emit(f"double spill_im[{N}];")
    else:
        em.emit(f"__attribute__((aligned({I.align}))) double spill_re[{N} * {I.width}];")
        em.emit(f"__attribute__((aligned({I.align}))) double spill_im[{N} * {I.width}];")
    if I.name == 'avx2':
        em.emit(f"__attribute__((aligned({I.align}))) double bfr[4 * {I.width}], bfi[4 * {I.width}];")
    em.blank()

    for i in range(0, N1, 4):
        chunk = min(4, N1 - i)
        parts = [f"x{i+j}_re, x{i+j}_im" for j in range(chunk)]
        em.emit(f"{T} {', '.join(parts)};")
    em.blank()

    # No hoisted twiddle broadcasts — inline from static arrays
    # This keeps all 16 YMM registers free for data + butterfly temps

    em.emit(f"for (size_t k = 0; k < K; k += {I.k_step}) {{")
    em.indent += 1

    xvars = [f"x{i}" for i in range(N1)]

    # PASS 1: 8 radix-8 sub-FFTs
    em.comment(f"PASS 1: {N2} radix-{N1} sub-FFTs [{direction}]")
    em.blank()

    for n2 in range(N2):
        em.comment(f"sub-FFT n₂={n2}")
        for n1 in range(N1):
            em.emit_load_input(f"x{n1}", N2 * n1 + n2)
        em.blank()
        em.emit_r8(xvars, direction, f"radix-8 sub-FFT n₂={n2}")
        em.blank()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.blank()

    # PASS 2: 8 radix-8 column combines
    em.comment(f"PASS 2: {N1} radix-{N2} combines with W_{N} twiddles [{direction}]")
    em.blank()

    for k1 in range(N1):
        em.comment(f"column k₁={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.blank()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, N, direction)
            em.blank()
        em.emit_r8(xvars, direction, f"radix-8 combine k₁={k1}")
        em.blank()
        for k2 in range(N2):
            em.emit_store_output(f"x{k2}", k1 + N1 * k2)
        em.blank()

    em.indent -= 1
    em.emit("} /* end k-loop */")
    em.lines.append("}")
    em.lines.append("")

    return em.spill_count, em.reload_count

# ──────────────────────────────────────────────────────────────────
# File generation
# ──────────────────────────────────────────────────────────────────

def generate(isa_name):
    isa = ISAS[isa_name]
    em = Emitter(isa)
    tw_set = collect_twiddles()
    mp = isa.macro_prefix
    guard = f"FFT_RADIX64_{isa.name.upper()}_N1_GEN_H"

    em.lines.append(f"/**")
    em.lines.append(f" * @file fft_radix64_{isa.name}_n1_gen.h")
    em.lines.append(f" * @brief GENERATED DFT-64 {isa.name.upper()} N1 kernels (fwd + bwd)")
    em.lines.append(f" *")
    em.lines.append(f" * 8×8: 8 radix-8 sub-FFTs + 8 radix-8 combines")
    em.lines.append(f" * Vector width: {isa.width} doubles, k-step: {isa.k_step}")
    em.lines.append(f" * Generated by gen_radix64_n1.py")
    em.lines.append(f" */")
    em.lines.append(f"")
    em.lines.append(f"#ifndef {guard}")
    em.lines.append(f"#define {guard}")
    em.lines.append(f"")
    if isa.name in ('avx2', 'avx512'):
        em.lines.append(f"#include <immintrin.h>")
        em.lines.append(f"")

    # Twiddle constants — static arrays for inline broadcast (zero hoisted registers)
    em.lines.append(f"/* Internal W64 twiddle constants — static arrays for inline broadcast */")
    em.lines.append(f"#ifndef FFT_W64_STATIC_ARRAYS_DEFINED")
    em.lines.append(f"#define FFT_W64_STATIC_ARRAYS_DEFINED")
    em.lines.append(f"static const double __attribute__((aligned(8))) iw_re[64] = {{")
    for i in range(0, 64, 4):
        vals = [f'{wN(j, 64)[0]:.20e}' for j in range(i, i+4)]
        em.lines.append(f"    {', '.join(vals)},")
    em.lines.append(f"}};")
    em.lines.append(f"static const double __attribute__((aligned(8))) iw_im[64] = {{")
    for i in range(0, 64, 4):
        vals = [f'{wN(j, 64)[1]:.20e}' for j in range(i, i+4)]
        em.lines.append(f"    {', '.join(vals)},")
    em.lines.append(f"}};")
    em.lines.append(f"#endif /* FFT_W64_STATIC_ARRAYS_DEFINED */")
    em.lines.append(f"")

    # Load/store macros
    if isa.name == 'scalar':
        em.lines.append(f"#ifndef {mp}_LD")
        em.lines.append(f"#define {mp}_LD(p) (*(p))")
        em.lines.append(f"#endif")
        em.lines.append(f"#ifndef {mp}_ST")
        em.lines.append(f"#define {mp}_ST(p,v) (*(p) = (v))")
        em.lines.append(f"#endif")
    elif isa.name == 'avx2':
        em.lines.append(f"#ifndef {mp}_LD")
        em.lines.append(f"#define {mp}_LD(p) _mm256_loadu_pd(p)")
        em.lines.append(f"#endif")
        em.lines.append(f"#ifndef {mp}_ST")
        em.lines.append(f"#define {mp}_ST(p,v) _mm256_storeu_pd((p),(v))")
        em.lines.append(f"#endif")
    else:  # avx512
        em.lines.append(f"#ifndef {mp}_LD")
        em.lines.append(f"#define {mp}_LD(p) _mm512_load_pd(p)")
        em.lines.append(f"#endif")
        em.lines.append(f"#ifndef {mp}_ST")
        em.lines.append(f"#define {mp}_ST(p,v) _mm512_store_pd((p),(v))")
        em.lines.append(f"#endif")
    em.lines.append(f"")

    fwd_s, fwd_r = emit_kernel(em, 'fwd', tw_set)
    bwd_s, bwd_r = emit_kernel(em, 'bwd', tw_set)

    em.lines.append(f"#endif /* {guard} */")

    return em, tw_set, fwd_s, fwd_r, bwd_s, bwd_r

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ISAS:
        print(f"Usage: {sys.argv[0]} <scalar|avx2>", file=sys.stderr)
        sys.exit(1)

    isa_name = sys.argv[1]
    em, tw_set, fs, fr, bs, br = generate(isa_name)

    print("\n".join(em.lines))

    print(f"\n=== DFT-64 {isa_name.upper()} ===", file=sys.stderr)
    print(f"  Forward:  {fs} spills + {fr} reloads = {fs+fr} L1 ops", file=sys.stderr)
    print(f"  Backward: {bs} spills + {br} reloads = {bs+br} L1 ops", file=sys.stderr)
    print(f"  Twiddles: {len(tw_set)} generic W₆₄ constants", file=sys.stderr)
    print(f"  Lines:    {len(em.lines)}", file=sys.stderr)

if __name__ == '__main__':
    main()
