#!/usr/bin/env python3
"""
gen_radix128_n1.py — Generate DFT-128 N1 kernels for scalar and AVX2

16×8 decomposition: 8 radix-16 sub-FFTs + 16 radix-8 column combines.
Native forward + backward. Hoisted twiddle broadcasts (AVX2).

Usage:
  python3 gen_radix128_n1.py scalar > fft_radix128_scalar_n1_gen.h
  python3 gen_radix128_n1.py avx2   > fft_radix128_avx2_n1_gen.h
"""

import math, sys

# ──────────────────────────────────────────────────────────────────
# ISA abstraction
# ──────────────────────────────────────────────────────────────────

class ISA_Scalar:
    name = 'scalar'
    vtype = 'double'
    width = 1          # doubles per "vector"
    k_step = 1
    target_attr = ''
    macro_prefix = 'R128S'
    func_suffix = 'scalar'
    sign_flip_decl = None  # not needed
    sqrt2_decl = 'const double sqrt2_inv = 0.70710678118654752440;'
    
    @staticmethod
    def load(var, expr):
        return f"{var} = {expr};"
    @staticmethod
    def store(ptr, var):
        return f"{ptr} = {var};"
    @staticmethod
    def add(a, b):  return f"({a} + {b})"
    @staticmethod
    def sub(a, b):  return f"({a} - {b})"
    @staticmethod
    def mul(a, b):  return f"({a} * {b})"
    @staticmethod
    def neg(a):     return f"(-{a})"
    @staticmethod
    def fmadd(a, b, c):  return f"({a} * {b} + {c})"
    @staticmethod
    def fmsub(a, b, c):  return f"({a} * {b} - {c})"
    @staticmethod
    def xor_sign(a):     return f"(-{a})"
    @staticmethod
    def broadcast(val):  return val
    @staticmethod
    def spill_store(ptr, var): return f"*({ptr}) = {var};"
    @staticmethod
    def spill_load(ptr):       return f"*({ptr})"

class ISA_AVX2:
    name = 'avx2'
    vtype = '__m256d'
    width = 4
    k_step = 4
    target_attr = '__attribute__((target("avx2,fma")))'
    macro_prefix = 'R128A'
    func_suffix = 'avx2'
    sign_flip_decl = 'const __m256d sign_flip = _mm256_set1_pd(-0.0);'
    sqrt2_decl = 'const __m256d sqrt2_inv = _mm256_set1_pd(0.70710678118654752440);'
    
    @staticmethod
    def load(var, expr):
        return f"{var} = {expr};"
    @staticmethod
    def store(ptr, var):
        return f"{ptr} = {var};"  # caller wraps with macro
    @staticmethod
    def add(a, b):  return f"_mm256_add_pd({a}, {b})"
    @staticmethod
    def sub(a, b):  return f"_mm256_sub_pd({a}, {b})"
    @staticmethod
    def mul(a, b):  return f"_mm256_mul_pd({a}, {b})"
    @staticmethod
    def neg(a):     return f"_mm256_xor_pd({a}, sign_flip)"
    @staticmethod
    def fmadd(a, b, c):  return f"_mm256_fmadd_pd({a}, {b}, {c})"
    @staticmethod
    def fmsub(a, b, c):  return f"_mm256_fmsub_pd({a}, {b}, {c})"
    @staticmethod
    def xor_sign(a):     return f"_mm256_xor_pd({a}, sign_flip)"
    @staticmethod
    def broadcast(val):  return f"_mm256_set1_pd({val})"
    @staticmethod
    def spill_store(ptr, var): return f"_mm256_store_pd({ptr}, {var});"
    @staticmethod
    def spill_load(ptr):       return f"_mm256_load_pd({ptr})"

ISAS = {'scalar': ISA_Scalar, 'avx2': ISA_AVX2}

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
    
    # ── Spill / reload ──
    
    def emit_spill(self, var, slot):
        I = self.isa
        if I.name == 'scalar':
            self.emit(f"spill_re[{slot}] = {var}_re;")
            self.emit(f"spill_im[{slot}] = {var}_im;")
        else:
            self.emit(f"_mm256_store_pd(&spill_re[{slot} * 4], {var}_re);")
            self.emit(f"_mm256_store_pd(&spill_im[{slot} * 4], {var}_im);")
        self.spill_count += 1
    
    def emit_reload(self, var, slot):
        I = self.isa
        if I.name == 'scalar':
            self.emit(f"{var}_re = spill_re[{slot}];")
            self.emit(f"{var}_im = spill_im[{slot}];")
        else:
            self.emit(f"{var}_re = _mm256_load_pd(&spill_re[{slot} * 4]);")
            self.emit(f"{var}_im = _mm256_load_pd(&spill_im[{slot} * 4]);")
        self.reload_count += 1
    
    def emit_load_input(self, var, n):
        mp = self.isa.macro_prefix
        self.emit(f"{var}_re = {mp}_LD(&in_re[{n} * K + k]);")
        self.emit(f"{var}_im = {mp}_LD(&in_im[{n} * K + k]);")
    
    def emit_store_output(self, var, m):
        mp = self.isa.macro_prefix
        self.emit(f"{mp}_ST(&out_re[{m} * K + k], {var}_re);")
        self.emit(f"{mp}_ST(&out_im[{m} * K + k], {var}_im);")
    
    # ── Arithmetic helpers (ISA-polymorphic) ──
    
    def _add(self, a, b): return self.isa.add(a, b)
    def _sub(self, a, b): return self.isa.sub(a, b)
    def _mul(self, a, b): return self.isa.mul(a, b)
    def _neg(self, a):    return self.isa.xor_sign(a)
    def _fmadd(self, a, b, c): return self.isa.fmadd(a, b, c)
    def _fmsub(self, a, b, c): return self.isa.fmsub(a, b, c)
    
    # ── Radix-4 butterfly ──
    
    def emit_radix4(self, v, direction):
        fwd = (direction == 'fwd')
        a, b, c, d = v[0], v[1], v[2], v[3]
        T = self.isa.vtype
        self.emit(f"{{ {T} t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i;")
        self.emit(f"  t0r = {self._add(f'{a}_re', f'{c}_re')}; t0i = {self._add(f'{a}_im', f'{c}_im')};")
        self.emit(f"  t1r = {self._sub(f'{a}_re', f'{c}_re')}; t1i = {self._sub(f'{a}_im', f'{c}_im')};")
        self.emit(f"  t2r = {self._add(f'{b}_re', f'{d}_re')}; t2i = {self._add(f'{b}_im', f'{d}_im')};")
        self.emit(f"  t3r = {self._sub(f'{b}_re', f'{d}_re')}; t3i = {self._sub(f'{b}_im', f'{d}_im')};")
        self.emit(f"  {a}_re = {self._add('t0r', 't2r')}; {a}_im = {self._add('t0i', 't2i')};")
        self.emit(f"  {c}_re = {self._sub('t0r', 't2r')}; {c}_im = {self._sub('t0i', 't2i')};")
        if fwd:
            self.emit(f"  {b}_re = {self._add('t1r', 't3i')}; {b}_im = {self._sub('t1i', 't3r')};")
            self.emit(f"  {d}_re = {self._sub('t1r', 't3i')}; {d}_im = {self._add('t1i', 't3r')};")
        else:
            self.emit(f"  {b}_re = {self._sub('t1r', 't3i')}; {b}_im = {self._add('t1i', 't3r')};")
            self.emit(f"  {d}_re = {self._add('t1r', 't3i')}; {d}_im = {self._sub('t1i', 't3r')};")
        self.emit(f"}}")
    
    # ── Radix-8 butterfly ──
    
    def emit_radix8(self, v, direction, comment_str=""):
        assert len(v) == 8
        fwd = (direction == 'fwd')
        T = self.isa.vtype
        if comment_str:
            self.comment(f"{comment_str} [{direction}]")
        
        self.emit(f"{{ {T} e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;")
        self.emit(f"  {T} t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i;")
        # Even radix-4
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
        
        # Odd radix-4
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
    
    # ── Radix-16 butterfly (4×4) ──
    
    def emit_radix16(self, v, direction, comment_str=""):
        assert len(v) == 16
        if comment_str:
            self.comment(f"{comment_str} [{direction}]")
        for j in range(4):
            self.emit_radix4([v[j], v[j+4], v[j+8], v[j+12]], direction)
        for j in range(1, 4):
            for k1 in range(1, 4):
                e = (j * k1) % 16
                self.emit_twiddle(v[j + 4*k1], v[j + 4*k1], e, 16, direction)
        for k1 in range(4):
            self.emit_radix4([v[k1*4], v[k1*4+1], v[k1*4+2], v[k1*4+3]], direction)
    
    # ── Butterfly dispatcher ──
    
    def emit_butterfly(self, size, v, direction, comment_str=""):
        if size == 4:
            if comment_str: self.comment(f"{comment_str} [{direction}]")
            self.emit_radix4(v, direction)
        elif size == 8:
            self.emit_radix8(v, direction, comment_str)
        elif size == 16:
            self.emit_radix16(v, direction, comment_str)
        else:
            raise ValueError(f"No butterfly for size {size}")
    
    # ── Output index mapping ──
    
    def subfft_output_var_index(self, k1, N1):
        if N1 <= 8:
            return k1
        elif N1 == 16:
            return (k1 % 4) * 4 + (k1 // 4)
        else:
            raise ValueError(f"Unknown mapping for N1={N1}")
    
    # ── Twiddle emission ──
    
    def emit_twiddle(self, dst, src, e, tN, direction):
        is_special, typ = twiddle_is_trivial(e, tN)
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
            # Generic cmul
            key = (e % tN, tN)
            self.twiddles_needed.add(key)
            label = wN_label(e, tN)
            self.emit(f"{{ const {T} tr = {src}_re;")
            if fwd:
                self.emit(f"  {dst}_re = {self._fmsub(f'{src}_re', f'tw_{label}_re', self._mul(f'{src}_im', f'tw_{label}_im'))};")
                self.emit(f"  {dst}_im = {self._fmadd('tr', f'tw_{label}_im', self._mul(f'{src}_im', f'tw_{label}_re'))}; }}")
            else:
                self.emit(f"  {dst}_re = {self._fmadd(f'{src}_re', f'tw_{label}_re', self._mul(f'{src}_im', f'tw_{label}_im'))};")
                self.emit(f"  {dst}_im = {self._fmsub(f'{src}_im', f'tw_{label}_re', self._mul('tr', f'tw_{label}_im'))}; }}")

# ──────────────────────────────────────────────────────────────────
# Twiddle collection
# ──────────────────────────────────────────────────────────────────

def collect_twiddles():
    tw_set = set()
    N, N1, N2 = 128, 16, 8
    # Inner W₁₆
    for j in range(1, 4):
        for k1 in range(1, 4):
            e = (j * k1) % 16
            is_special, _ = twiddle_is_trivial(e, 16)
            if not is_special:
                tw_set.add((e, 16))
    # Outer W₁₂₈
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
    N, N1, N2 = 128, 16, 8
    I = em.isa
    T = I.vtype
    
    if I.target_attr:
        em.lines.append(f"static {I.target_attr} void")
    else:
        em.lines.append(f"static void")
    em.lines.append(f"radix128_n1_dit_kernel_{direction}_{I.func_suffix}(")
    em.lines.append(f"    const double *RESTRICT in_re, const double *RESTRICT in_im,")
    em.lines.append(f"    double *RESTRICT out_re, double *RESTRICT out_im,")
    em.lines.append(f"    size_t K)")
    em.lines.append(f"{{")
    
    em.indent = 1
    em.spill_count = 0
    em.reload_count = 0
    
    if I.sign_flip_decl:
        em.emit(I.sign_flip_decl)
    em.emit(I.sqrt2_decl)
    em.blank()
    
    # Spill buffer
    if I.name == 'scalar':
        em.emit(f"double spill_re[{N}];")
        em.emit(f"double spill_im[{N}];")
    else:
        em.emit(f"ALIGNAS_32 double spill_re[{N} * 4];  /* {N*32} bytes */")
        em.emit(f"ALIGNAS_32 double spill_im[{N} * 4];")
    em.blank()
    
    # Declare working registers
    for i in range(0, N1, 4):
        chunk = min(4, N1 - i)
        parts = [f"x{i+j}_re, x{i+j}_im" for j in range(chunk)]
        em.emit(f"{T} {', '.join(parts)};")
    em.blank()
    
    # Hoisted twiddle broadcasts
    if tw_set:
        em.comment(f"Hoisted twiddle broadcasts [{direction}]")
        for (e, tN) in sorted(tw_set):
            label = wN_label(e, tN)
            if I.name == 'scalar':
                em.emit(f"const double tw_{label}_re = {label}_re;")
                em.emit(f"const double tw_{label}_im = {label}_im;")
            else:
                em.emit(f"const __m256d tw_{label}_re = _mm256_set1_pd({label}_re);")
                em.emit(f"const __m256d tw_{label}_im = _mm256_set1_pd({label}_im);")
        em.blank()
    
    # k-loop
    em.emit(f"for (size_t k = 0; k < K; k += {I.k_step}) {{")
    em.indent += 1
    
    xvars16 = [f"x{i}" for i in range(N1)]
    xvars8 = [f"x{i}" for i in range(N2)]
    
    # PASS 1: 8 radix-16 sub-FFTs
    em.comment(f"PASS 1: {N2} radix-{N1} sub-FFTs [{direction}]")
    em.blank()
    
    for n2 in range(N2):
        em.comment(f"sub-FFT n₂={n2}")
        for n1 in range(N1):
            em.emit_load_input(f"x{n1}", N2 * n1 + n2)
        em.blank()
        em.emit_butterfly(N1, xvars16, direction, f"radix-{N1} n₂={n2}")
        em.blank()
        for k1 in range(N1):
            x_idx = em.subfft_output_var_index(k1, N1)
            em.emit_spill(f"x{x_idx}", n2 * N1 + k1)
        em.blank()
    
    # PASS 2: 16 radix-8 column combines
    em.comment(f"PASS 2: {N1} radix-{N2} combines [{direction}]")
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
        em.emit_butterfly(N2, xvars8, direction, f"radix-{N2} k₁={k1}")
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
    guard = f"FFT_RADIX128_{isa.name.upper()}_N1_GEN_H"
    
    em.lines.append(f"/**")
    em.lines.append(f" * @file fft_radix128_{isa.name}_n1_gen.h")
    em.lines.append(f" * @brief GENERATED DFT-128 {isa.name.upper()} N1 kernels (fwd + bwd)")
    em.lines.append(f" *")
    em.lines.append(f" * 16×8: 8 radix-16 sub-FFTs + 16 radix-8 combines")
    em.lines.append(f" * Vector width: {isa.width} doubles, k-step: {isa.k_step}")
    em.lines.append(f" * Generated by gen_radix128_n1.py")
    em.lines.append(f" */")
    em.lines.append(f"")
    em.lines.append(f"#ifndef {guard}")
    em.lines.append(f"#define {guard}")
    em.lines.append(f"")
    if isa.name == 'avx2':
        em.lines.append(f"#include <immintrin.h>")
        em.lines.append(f"")
        em.lines.append(f"#ifndef ALIGNAS_32")
        em.lines.append(f"#define ALIGNAS_32 __attribute__((aligned(32)))")
        em.lines.append(f"#endif")
        em.lines.append(f"")
    
    # Twiddle constants grouped by size
    by_tN = {}
    for (e, tN) in sorted(tw_set):
        by_tN.setdefault(tN, []).append(e)
    for tN in sorted(by_tN):
        tguard = f"FFT_W{tN}_TWIDDLES_DEFINED"
        em.lines.append(f"#ifndef {tguard}")
        em.lines.append(f"#define {tguard}")
        for e in sorted(by_tN[tN]):
            wr, wi = wN(e, tN)
            label = wN_label(e, tN)
            em.lines.append(f"static const double {label}_re = {wr:.20e};")
            em.lines.append(f"static const double {label}_im = {wi:.20e};")
        em.lines.append(f"#endif /* {tguard} */")
        em.lines.append(f"")
    
    # Load/store macros
    if isa.name == 'scalar':
        em.lines.append(f"#ifndef {mp}_LD")
        em.lines.append(f"#define {mp}_LD(p) (*(p))")
        em.lines.append(f"#endif")
        em.lines.append(f"#ifndef {mp}_ST")
        em.lines.append(f"#define {mp}_ST(p,v) (*(p) = (v))")
        em.lines.append(f"#endif")
    else:
        em.lines.append(f"#ifndef {mp}_LD")
        em.lines.append(f"#define {mp}_LD(p) _mm256_loadu_pd(p)")
        em.lines.append(f"#endif")
        em.lines.append(f"#ifndef {mp}_ST")
        em.lines.append(f"#define {mp}_ST(p,v) _mm256_storeu_pd((p),(v))")
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
    
    inner = sum(1 for e, tN in tw_set if tN == 16)
    outer = sum(1 for e, tN in tw_set if tN == 128)
    
    print(f"\n=== DFT-128 {isa_name.upper()} ===", file=sys.stderr)
    print(f"  Forward:  {fs} spills + {fr} reloads = {fs+fr} L1 ops", file=sys.stderr)
    print(f"  Backward: {bs} spills + {br} reloads = {bs+br} L1 ops", file=sys.stderr)
    print(f"  Twiddles: {inner} inner (W₁₆) + {outer} outer (W₁₂₈) = {len(tw_set)}", file=sys.stderr)
    print(f"  Lines:    {len(em.lines)}", file=sys.stderr)

if __name__ == '__main__':
    main()
