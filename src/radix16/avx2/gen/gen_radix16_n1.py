#!/usr/bin/env python3
"""
gen_radix16_n1.py — DFT-16 N1 (twiddle-less) codelet generator

4×4 decomposition: 4 radix-4 sub-FFTs + internal W₁₆ + 4 radix-4 combines.
No external twiddles. Spill buffer: 16 slots.

Usage:
  python3 gen_radix16_n1.py scalar > fft_radix16_scalar_n1_gen.h
  python3 gen_radix16_n1.py avx2   > fft_radix16_avx2_n1_gen.h
"""

import math, sys

N, N1, N2 = 16, 4, 4

def wN(e, tN):
    e = e % tN
    a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def twiddle_is_trivial(e, tN):
    e = e % tN
    if e == 0: return True, 'one'
    if (8 * e) % tN == 0:
        o = (8 * e) // tN
        t = ['one','w8_1','neg_j','w8_3','neg_one','neg_w8_1','pos_j','neg_w8_3']
        return True, t[o % 8]
    return False, 'cmul'

def collect_internal_twiddles():
    tw = set()
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2 * k1) % N
            _, typ = twiddle_is_trivial(e, N)
            if typ == 'cmul': tw.add((e, N))
    return tw

# ── ISA configs ──

class Scalar:
    name = 'scalar'
    vtype = 'double'
    k_step = 1
    target = ''
    macro_prefix = 'R16S'
    spill_align = ''
    spill_mul = 1
    sign_flip = None
    sqrt2 = 'const double sqrt2_inv = 0.70710678118654752440;'
    @staticmethod
    def add(a,b): return f"({a}+{b})"
    @staticmethod
    def sub(a,b): return f"({a}-{b})"
    @staticmethod
    def mul(a,b): return f"({a}*{b})"
    @staticmethod
    def neg(a):   return f"(-{a})"
    @staticmethod
    def fmsub(a,b,c): return f"({a}*{b}-{c})"
    @staticmethod
    def fmadd(a,b,c): return f"({a}*{b}+{c})"
    @staticmethod
    def spill_st(ptr,v): return f"*({ptr}) = {v};"
    @staticmethod
    def spill_ld(ptr): return f"*({ptr})"

class AVX2:
    name = 'avx2'
    vtype = '__m256d'
    k_step = 4
    target = '__attribute__((target("avx2,fma")))'
    macro_prefix = 'R16A'
    spill_align = '__attribute__((aligned(32)))'
    spill_mul = 4
    sign_flip = 'const __m256d sign_flip = _mm256_set1_pd(-0.0);'
    sqrt2 = 'const __m256d sqrt2_inv = _mm256_set1_pd(0.70710678118654752440);'
    @staticmethod
    def add(a,b): return f"_mm256_add_pd({a},{b})"
    @staticmethod
    def sub(a,b): return f"_mm256_sub_pd({a},{b})"
    @staticmethod
    def mul(a,b): return f"_mm256_mul_pd({a},{b})"
    @staticmethod
    def neg(a):   return f"_mm256_xor_pd({a},sign_flip)"
    @staticmethod
    def fmsub(a,b,c): return f"_mm256_fmsub_pd({a},{b},{c})"
    @staticmethod
    def fmadd(a,b,c): return f"_mm256_fmadd_pd({a},{b},{c})"
    @staticmethod
    def spill_st(ptr,v): return f"_mm256_store_pd({ptr},{v});"
    @staticmethod
    def spill_ld(ptr): return f"_mm256_load_pd({ptr})"

ISAS = {'scalar': Scalar, 'avx2': AVX2}


class Emitter:
    def __init__(self, isa):
        self.I = isa; self.L = []; self.ind = 1
        self.spill_c = 0; self.reload_c = 0

    def o(s, t=""): s.L.append("    " * s.ind + t)
    def c(s, t): s.o(f"/* {t} */")
    def b(s): s.L.append("")

    def emit_load(s, v, n):
        mp = s.I.macro_prefix
        s.o(f"{v}_re = {mp}_LD(&in_re[{n}*K+k]);")
        s.o(f"{v}_im = {mp}_LD(&in_im[{n}*K+k]);")

    def emit_store(s, v, m):
        mp = s.I.macro_prefix
        s.o(f"{mp}_ST(&out_re[{m}*K+k], {v}_re);")
        s.o(f"{mp}_ST(&out_im[{m}*K+k], {v}_im);")

    def emit_spill(s, v, slot):
        I = s.I; m = I.spill_mul
        if I.name == 'scalar':
            s.o(f"spill_re[{slot}] = {v}_re; spill_im[{slot}] = {v}_im;")
        else:
            s.o(f"_mm256_store_pd(&spill_re[{slot}*{m}], {v}_re);")
            s.o(f"_mm256_store_pd(&spill_im[{slot}*{m}], {v}_im);")
        s.spill_c += 1

    def emit_reload(s, v, slot):
        I = s.I; m = I.spill_mul
        if I.name == 'scalar':
            s.o(f"{v}_re = spill_re[{slot}]; {v}_im = spill_im[{slot}];")
        else:
            s.o(f"{v}_re = _mm256_load_pd(&spill_re[{slot}*{m}]);")
            s.o(f"{v}_im = _mm256_load_pd(&spill_im[{slot}*{m}]);")
        s.reload_c += 1

    def emit_radix4(s, v, d, label=""):
        I = s.I; fwd = (d == 'fwd'); T = I.vtype
        if label: s.c(f"{label} [{d}]")
        a,b,c,dd = v[0],v[1],v[2],v[3]
        s.o(f"{{ {T} t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={I.add(f'{a}_re',f'{c}_re')}; t0i={I.add(f'{a}_im',f'{c}_im')};")
        s.o(f"  t1r={I.sub(f'{a}_re',f'{c}_re')}; t1i={I.sub(f'{a}_im',f'{c}_im')};")
        s.o(f"  t2r={I.add(f'{b}_re',f'{dd}_re')}; t2i={I.add(f'{b}_im',f'{dd}_im')};")
        s.o(f"  t3r={I.sub(f'{b}_re',f'{dd}_re')}; t3i={I.sub(f'{b}_im',f'{dd}_im')};")
        s.o(f"  {a}_re={I.add('t0r','t2r')}; {a}_im={I.add('t0i','t2i')};")
        s.o(f"  {c}_re={I.sub('t0r','t2r')}; {c}_im={I.sub('t0i','t2i')};")
        if fwd:
            s.o(f"  {b}_re={I.add('t1r','t3i')}; {b}_im={I.sub('t1i','t3r')};")
            s.o(f"  {dd}_re={I.sub('t1r','t3i')}; {dd}_im={I.add('t1i','t3r')};")
        else:
            s.o(f"  {b}_re={I.sub('t1r','t3i')}; {b}_im={I.add('t1i','t3r')};")
            s.o(f"  {dd}_re={I.add('t1r','t3i')}; {dd}_im={I.sub('t1i','t3r')};")
        s.o(f"}}")

    def emit_twiddle(s, dst, src, e, tN, d):
        I = s.I; _, typ = twiddle_is_trivial(e, tN); fwd = (d == 'fwd'); T = I.vtype
        if typ == 'one':
            if dst != src: s.o(f"{dst}_re={src}_re; {dst}_im={src}_im;")
        elif typ == 'neg_one':
            s.o(f"{dst}_re={I.neg(f'{src}_re')}; {dst}_im={I.neg(f'{src}_im')};")
        elif typ == 'neg_j':
            if fwd: s.o(f"{{ {T} t={src}_re; {dst}_re={src}_im; {dst}_im={I.neg('t')}; }}")
            else:   s.o(f"{{ {T} t={src}_re; {dst}_re={I.neg(f'{src}_im')}; {dst}_im=t; }}")
        elif typ == 'pos_j':
            if fwd: s.o(f"{{ {T} t={src}_re; {dst}_re={I.neg(f'{src}_im')}; {dst}_im=t; }}")
            else:   s.o(f"{{ {T} t={src}_re; {dst}_re={src}_im; {dst}_im={I.neg('t')}; }}")
        elif typ == 'w8_1':
            s.o(f"{{ {T} tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={I.mul(I.add('tr','ti'),'sqrt2_inv')}; {dst}_im={I.mul(I.sub('ti','tr'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={I.mul(I.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={I.mul(I.add('tr','ti'),'sqrt2_inv')}; }}")
        elif typ == 'w8_3':
            s.o(f"{{ {T} tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={I.mul(I.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={I.neg(I.mul(I.add('tr','ti'),'sqrt2_inv'))}; }}")
            else:   s.o(f"  {dst}_re={I.neg(I.mul(I.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={I.mul(I.sub('tr','ti'),'sqrt2_inv')}; }}")
        elif typ == 'neg_w8_1':
            s.o(f"{{ {T} tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={I.neg(I.mul(I.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={I.mul(I.sub('tr','ti'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={I.mul(I.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={I.neg(I.mul(I.add('tr','ti'),'sqrt2_inv'))}; }}")
        elif typ == 'neg_w8_3':
            s.o(f"{{ {T} tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={I.mul(I.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={I.mul(I.add('tr','ti'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={I.mul(I.add('tr','ti'),'sqrt2_inv')}; {dst}_im={I.mul(I.sub('ti','tr'),'sqrt2_inv')}; }}")
        else:
            label = wN_label(e, tN)
            s.o(f"{{ {T} tr={src}_re;")
            if fwd:
                s.o(f"  {dst}_re={I.fmsub(f'{src}_re',f'tw_{label}_re',I.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={I.fmadd('tr',f'tw_{label}_im',I.mul(f'{src}_im',f'tw_{label}_re'))}; }}")
            else:
                s.o(f"  {dst}_re={I.fmadd(f'{src}_re',f'tw_{label}_re',I.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={I.fmsub(f'{src}_im',f'tw_{label}_re',I.mul('tr',f'tw_{label}_im'))}; }}")


def emit_n1_kernel(em, d, itw_set):
    I = em.I; T = I.vtype
    if I.target:
        em.L.append(f"static {I.target} void")
    else:
        em.L.append(f"static void")
    em.L.append(f"radix16_n1_dit_kernel_{d}_{I.name}(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1; em.spill_c = 0; em.reload_c = 0

    if I.sign_flip: em.o(I.sign_flip)
    em.o(I.sqrt2)
    em.b()

    m = I.spill_mul
    if I.name == 'scalar':
        em.o(f"double spill_re[{N}], spill_im[{N}];")
    else:
        em.o(f"{I.spill_align} double spill_re[{N}*{m}];")
        em.o(f"{I.spill_align} double spill_im[{N}*{m}];")
    em.b()
    em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.b()

    if itw_set:
        em.c(f"Hoisted W₁₆ broadcasts [{d}]")
        for (e, tN) in sorted(itw_set):
            label = wN_label(e, tN)
            if I.name == 'scalar':
                em.o(f"const double tw_{label}_re = {label}_re;")
                em.o(f"const double tw_{label}_im = {label}_im;")
            else:
                em.o(f"const __m256d tw_{label}_re = _mm256_set1_pd({label}_re);")
                em.o(f"const __m256d tw_{label}_im = _mm256_set1_pd({label}_im);")
        em.b()

    xv4 = [f"x{i}" for i in range(4)]

    em.o(f"for (size_t k = 0; k < K; k += {I.k_step}) {{")
    em.ind += 1

    em.c(f"PASS 1: {N2} radix-{N1} sub-FFTs [{d}]")
    em.b()
    for n2 in range(N2):
        em.c(f"sub-FFT n₂={n2}")
        for n1 in range(N1):
            em.emit_load(f"x{n1}", N2 * n1 + n2)
        em.b()
        em.emit_radix4(xv4, d, f"radix-4 n₂={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    em.c(f"PASS 2: {N1} radix-{N2} combines [{d}]")
    em.b()
    for k1 in range(N1):
        em.c(f"column k₁={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, N, d)
            em.b()
        em.emit_radix4(xv4, d, f"radix-4 k₁={k1}")
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2)
        em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}")
    em.L.append("")
    return em.spill_c, em.reload_c


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ISAS:
        print(f"Usage: {sys.argv[0]} <scalar|avx2>", file=sys.stderr)
        sys.exit(1)

    isa_name = sys.argv[1]
    I = ISAS[isa_name]
    itw_set = collect_internal_twiddles()
    em = Emitter(I)
    mp = I.macro_prefix
    guard = f"FFT_RADIX16_{I.name.upper()}_N1_GEN_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix16_{I.name}_n1_gen.h")
    em.L.append(f" * @brief GENERATED DFT-16 {I.name.upper()} N1 kernels (fwd + bwd)")
    em.L.append(f" *")
    em.L.append(f" * 4×4 Cooley-Tukey, k-step={I.k_step}, zero external twiddles.")
    em.L.append(f" * Internal W₁₆: {len(itw_set)} constants.")
    em.L.append(f" * Generated by gen_radix16_n1.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")
    if I.name == 'avx2':
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")

    # Twiddle constants
    by_tN = {}
    for (e, tN) in sorted(itw_set):
        by_tN.setdefault(tN, []).append(e)
    for tN in sorted(by_tN):
        g = f"FFT_W{tN}_TWIDDLES_DEFINED"
        em.L.append(f"#ifndef {g}")
        em.L.append(f"#define {g}")
        for e in sorted(by_tN[tN]):
            wr, wi = wN(e, tN)
            l = wN_label(e, tN)
            em.L.append(f"static const double {l}_re = {wr:.20e};")
            em.L.append(f"static const double {l}_im = {wi:.20e};")
        em.L.append(f"#endif")
        em.L.append(f"")

    # LD/ST macros
    if I.name == 'scalar':
        em.L.append(f"#ifndef {mp}_LD")
        em.L.append(f"#define {mp}_LD(p) (*(p))")
        em.L.append(f"#endif")
        em.L.append(f"#ifndef {mp}_ST")
        em.L.append(f"#define {mp}_ST(p,v) (*(p)=(v))")
        em.L.append(f"#endif")
    else:
        em.L.append(f"#ifndef {mp}_LD")
        em.L.append(f"#define {mp}_LD(p) _mm256_load_pd(p)")
        em.L.append(f"#endif")
        em.L.append(f"#ifndef {mp}_ST")
        em.L.append(f"#define {mp}_ST(p,v) _mm256_store_pd((p),(v))")
        em.L.append(f"#endif")
    em.L.append(f"")

    fwd = emit_n1_kernel(em, 'fwd', itw_set)
    bwd = emit_n1_kernel(em, 'bwd', itw_set)

    em.L.append(f"#endif /* {guard} */")

    print("\n".join(em.L))

    print(f"\n=== DFT-16 {I.name.upper()} N1 ===", file=sys.stderr)
    print(f"  fwd: {fwd[0]}sp+{fwd[1]}rl={fwd[0]+fwd[1]} L1 ops", file=sys.stderr)
    print(f"  bwd: {bwd[0]}sp+{bwd[1]}rl={bwd[0]+bwd[1]} L1 ops", file=sys.stderr)
    print(f"  W₁₆: {len(itw_set)} constants", file=sys.stderr)
    print(f"  Lines: {len(em.L)}", file=sys.stderr)

if __name__ == '__main__':
    main()
