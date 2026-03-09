#!/usr/bin/env python3
"""
gen_r5_native_il.py — DFT-5 native interleaved codelets with log3 twiddles.

FFTW-style: data stays interleaved [re0,im0,re1,im1] throughout.
Butterfly uses addsub/fmaddsub for complex ops — zero shuffle overhead.
Twiddles derived via log3 in interleaved form using VZMUL.

AVX2:   k-step=2 (2 complex per __m256d), uses fmaddsub_pd + addsub_pd
AVX-512: k-step=4 (4 complex per __m512d), uses fmaddsub_pd

Usage: gen_r5_native_il.py <avx2|avx512> <dit|dif>
"""
import sys

# ════════════════════════════════════════
# ISA abstraction — native interleaved ops
# ════════════════════════════════════════

def isa_avx2():
    T = '__m256d'
    C = 2  # complex per vector
    attr = '__attribute__((target("avx2,fma")))'

    def load(v, n):
        return [f'{v} = _mm256_load_pd(&in[2*({n}*K+k)]);']
    def store(v, n):
        return [f'_mm256_store_pd(&out[2*({n}*K+k)], {v});']

    def tw_load(v, idx):
        """Load twiddle from split table → interleaved vector."""
        return [
            f'{{ __m128d _tr = _mm_load_pd(&tw_re[{idx}*K+k]);',
            f'  __m128d _ti = _mm_load_pd(&tw_im[{idx}*K+k]);',
            f'  {v} = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_unpacklo_pd(_tr,_ti)), _mm_unpackhi_pd(_tr,_ti), 1); }}'
        ]

    add  = lambda a,b: f'_mm256_add_pd({a},{b})'
    sub  = lambda a,b: f'_mm256_sub_pd({a},{b})'
    mul  = lambda a,b: f'_mm256_mul_pd({a},{b})'
    fma  = lambda a,b,c: f'_mm256_fmadd_pd({a},{b},{c})'
    fnma = lambda a,b,c: f'_mm256_fnmadd_pd({a},{b},{c})'
    set1 = lambda v: f'_mm256_set1_pd({v})'
    neg  = lambda a: f'_mm256_xor_pd({a},sign_flip)'

    # Native IL primitives
    flip = lambda x: f'_mm256_permute_pd({x},0x5)'
    dupl = lambda x: f'_mm256_movedup_pd({x})'
    duph = lambda x: f'_mm256_permute_pd({x},0xF)'
    # c + j*b: addsub(c, FLIP(b)) → [c_re - b_im, c_im + b_re]
    vfmai = lambda b,c: f'_mm256_addsub_pd({c},{flip(b)})'
    # j*b: addsub(0, FLIP(b)) → [-b_im, b_re]
    vbyi = lambda x: f'_mm256_addsub_pd(_mm256_setzero_pd(),{flip(x)})'
    # tw*x (complex multiply, both interleaved)
    vzmul = lambda tw,x: f'_mm256_fmaddsub_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    # conj(tw)*x
    vzmulj = lambda tw,x: f'_mm256_fmsubadd_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'

    return dict(T=T, C=C, attr=attr, load=load, store=store, tw_load=tw_load,
                add=add, sub=sub, mul=mul, fma=fma, fnma=fnma, set1=set1, neg=neg,
                flip=flip, dupl=dupl, duph=duph,
                vfmai=vfmai, vbyi=vbyi, vzmul=vzmul, vzmulj=vzmulj,
                sign_const='const __m256d sign_flip = _mm256_set1_pd(-0.0);')

def isa_avx512():
    T = '__m512d'
    C = 4
    attr = '__attribute__((target("avx512f,avx512dq,fma")))'

    def load(v, n):
        return [f'{v} = _mm512_load_pd(&in[2*({n}*K+k)]);']
    def store(v, n):
        return [f'_mm512_store_pd(&out[2*({n}*K+k)], {v});']

    def tw_load(v, idx):
        """Load twiddle from split table → interleaved 512-bit vector."""
        return [
            f'{{ __m256d _tr = _mm256_load_pd(&tw_re[{idx}*K+k]);',
            f'  __m256d _ti = _mm256_load_pd(&tw_im[{idx}*K+k]);',
            f'  {v} = _mm512_permutex2var_pd(_mm512_castpd256_pd512(_tr),',
            f'    _mm512_set_epi64(11,3,10,2,9,1,8,0), _mm512_castpd256_pd512(_ti)); }}'
        ]

    add  = lambda a,b: f'_mm512_add_pd({a},{b})'
    sub  = lambda a,b: f'_mm512_sub_pd({a},{b})'
    mul  = lambda a,b: f'_mm512_mul_pd({a},{b})'
    fma  = lambda a,b,c: f'_mm512_fmadd_pd({a},{b},{c})'
    fnma = lambda a,b,c: f'_mm512_fnmadd_pd({a},{b},{c})'
    set1 = lambda v: f'_mm512_set1_pd({v})'
    neg  = lambda a: f'_mm512_sub_pd(_mm512_setzero_pd(),{a})'

    flip = lambda x: f'_mm512_permute_pd({x},0x55)'
    dupl = lambda x: f'_mm512_movedup_pd({x})'
    duph = lambda x: f'_mm512_permute_pd({x},0xFF)'
    # AVX-512 has no standalone addsub, use fmaddsub with 1.0
    # c + j*b = fmaddsub(1.0, c, FLIP(b)) → [1*c_re - flip_re, 1*c_im + flip_im]
    #         = [c_re - b_im, c_im + b_re]
    vfmai = lambda b,c: f'_mm512_fmaddsub_pd(_mm512_set1_pd(1.0),{c},{flip(b)})'
    vbyi = lambda x: f'_mm512_fmaddsub_pd(_mm512_set1_pd(1.0),_mm512_setzero_pd(),{flip(x)})'
    vzmul = lambda tw,x: f'_mm512_fmaddsub_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    vzmulj = lambda tw,x: f'_mm512_fmsubadd_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'

    return dict(T=T, C=C, attr=attr, load=load, store=store, tw_load=tw_load,
                add=add, sub=sub, mul=mul, fma=fma, fnma=fnma, set1=set1, neg=neg,
                flip=flip, dupl=dupl, duph=duph,
                vfmai=vfmai, vbyi=vbyi, vzmul=vzmul, vzmulj=vzmulj,
                sign_const=None)


# ════════════════════════════════════════
# Code emitter
# ════════════════════════════════════════

class Emitter:
    def __init__(self, isa):
        self.isa = isa
        self.L = []
        self.ind = 0
    def o(self, s=''): self.L.append('    ' * self.ind + s)
    def c(self, s): self.o(f'/* {s} */')
    def b(self): self.L.append('')
    def lines(self, ll):
        for l in ll: self.o(l)


def emit_log3_derive(em, isa, fwd):
    """Load W^1 from split table, derive W^2..W^4 in interleaved form."""
    T = isa['T']
    zmul = isa['vzmul']

    em.c('Log3: load W^1 from split table, derive W^2..W^4')
    em.lines(isa['tw_load']('w1', 0))
    em.o(f'const {T} w2 = {zmul("w1","w1")};')
    em.o(f'const {T} w3 = {zmul("w1","w2")};')
    em.o(f'const {T} w4 = {zmul("w1","w3")};')
    em.b()


def emit_butterfly(em, isa, fwd):
    """Native IL DFT-5 butterfly. x0..x4 → stores to out."""
    T = isa['T']
    add, sub, mul = isa['add'], isa['sub'], isa['mul']
    fma, fnma = isa['fma'], isa['fnma']
    vfmai, vbyi = isa['vfmai'], isa['vbyi']

    em.c('DFT-5 butterfly (native interleaved)')
    em.o(f'const {T} s1 = {add("x1","x4")};')
    em.o(f'const {T} s2 = {add("x2","x3")};')
    em.o(f'const {T} d1 = {sub("x1","x4")};')
    em.o(f'const {T} d2 = {sub("x2","x3")};')
    em.b()

    # out[0] = x0 + s1 + s2
    em.o(f'const {T} y0 = {add("x0", add("s1","s2"))};')

    # t0 = x0 - 0.25*(s1+s2)
    em.o(f'const {T} t0 = {fnma("cK3", add("s1","s2"), "x0")};')
    # t1 = cK0 * (s1-s2)
    em.o(f'const {T} t1 = {mul("cK0", sub("s1","s2"))};')
    # p1 = t0+t1, p2 = t0-t1
    em.o(f'const {T} p1 = {add("t0","t1")};')
    em.o(f'const {T} p2 = {sub("t0","t1")};')
    em.b()

    # U = cK1*d1 + cK2*d2 (complex — scalar mul works on interleaved!)
    em.o(f'const {T} U = {fma("cK2","d2", mul("cK1","d1"))};')
    # V = cK1*d2 - cK2*d1
    em.o(f'const {T} V = {fnma("cK2","d1", mul("cK1","d2"))};')
    em.b()

    # fwd: out[1]=p1-j*U, out[4]=p1+j*U, out[2]=p2+j*V, out[3]=p2-j*V
    # bwd: out[1]=p1+j*U, out[4]=p1-j*U, out[2]=p2-j*V, out[3]=p2+j*V
    em.o(f'const {T} jU = {vbyi("U")};')
    em.o(f'const {T} jV = {vbyi("V")};')

    if fwd:
        em.o(f'const {T} y1 = {sub("p1","jU")};')
        em.o(f'const {T} y4 = {add("p1","jU")};')
        em.o(f'const {T} y2 = {add("p2","jV")};')
        em.o(f'const {T} y3 = {sub("p2","jV")};')
    else:
        em.o(f'const {T} y1 = {add("p1","jU")};')
        em.o(f'const {T} y4 = {sub("p1","jU")};')
        em.o(f'const {T} y2 = {sub("p2","jV")};')
        em.o(f'const {T} y3 = {add("p2","jV")};')

    return ['y0','y1','y2','y3','y4']


def gen_dit_tw(isa_cfg, direction):
    """DIT: twiddle inputs → butterfly → store."""
    isa = isa_cfg
    T, C = isa['T'], isa['C']
    fwd = direction == 'fwd'
    zmul = isa['vzmul'] if fwd else isa['vzmulj']
    isa_name = 'avx2' if C == 2 else 'avx512'

    em = Emitter(isa)
    em.L = []
    if isa['attr']: em.L.append(isa['attr'])
    em.L.append(f'static void')
    em.L.append(f'radix5_tw_dit_kernel_{direction}_il_{isa_name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    # Constants
    em.o(f'const {T} cK0 = {isa["set1"]("0.559016994374947424102293417182819058860154590")};')
    em.o(f'const {T} cK1 = {isa["set1"]("0.951056516295153572116439333379382143405698634")};')
    em.o(f'const {T} cK2 = {isa["set1"]("0.587785252292473129168705954639072768597652438")};')
    em.o(f'const {T} cK3 = {isa["set1"]("0.250000000000000000000000000000000000000000000")};')
    if isa.get('sign_const'):
        em.o(isa['sign_const'])
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    # Declare variables
    em.o(f'{T} w1, x0;')

    # Log3 twiddle derivation
    emit_log3_derive(em, isa, fwd)

    # Load x0 (no twiddle), load+twiddle x1..x4
    em.lines(isa['load']('x0', 0))
    for n in range(1, 5):
        em.o(f'{T} r{n};')
        em.lines(isa['load'](f'r{n}', n))
        em.o(f'const {T} x{n} = {zmul(f"w{n}", f"r{n}")};')
    em.b()

    # Butterfly
    yvars = emit_butterfly(em, isa, fwd)
    em.b()

    # Store
    for i, yv in enumerate(yvars):
        em.lines(isa['store'](yv, i))

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


def gen_dif_tw(isa_cfg, direction):
    """DIF: load → butterfly → twiddle outputs → store."""
    isa = isa_cfg
    T, C = isa['T'], isa['C']
    fwd = direction == 'fwd'
    zmul = isa['vzmul'] if fwd else isa['vzmulj']
    isa_name = 'avx2' if C == 2 else 'avx512'

    em = Emitter(isa)
    em.L = []
    if isa['attr']: em.L.append(isa['attr'])
    em.L.append(f'static void')
    em.L.append(f'radix5_tw_dif_kernel_{direction}_il_{isa_name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(f'const {T} cK0 = {isa["set1"]("0.559016994374947424102293417182819058860154590")};')
    em.o(f'const {T} cK1 = {isa["set1"]("0.951056516295153572116439333379382143405698634")};')
    em.o(f'const {T} cK2 = {isa["set1"]("0.587785252292473129168705954639072768597652438")};')
    em.o(f'const {T} cK3 = {isa["set1"]("0.250000000000000000000000000000000000000000000")};')
    if isa.get('sign_const'):
        em.o(isa['sign_const'])
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    # Declare variables
    em.o(f'{T} w1;')

    # Log3 twiddle derivation
    emit_log3_derive(em, isa, fwd)

    # Load all inputs (no twiddle)
    for n in range(5):
        em.o(f'{T} x{n};')
        em.lines(isa['load'](f'x{n}', n))
    em.b()

    # Butterfly → y0..y4
    yvars = emit_butterfly(em, isa, fwd)
    em.b()

    # Twiddle outputs + store
    em.lines(isa['store'](yvars[0], 0))  # y0: no twiddle
    for n in range(1, 5):
        em.o(f'{{ const {T} tw_out = {zmul(f"w{n}", yvars[n])};')
        em.lines(isa['store']('tw_out', n))
        em.o(f'}}')

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


def gen_file(isa_name, mode):
    isa = isa_avx2() if isa_name == 'avx2' else isa_avx512()
    ISA = isa_name.upper()
    guard = f'FFT_RADIX5_{ISA}_IL_{"DIF_TW_" if mode == "dif" else ""}H'
    layout = 'native interleaved (FFTW-style)'

    L = [f'/**',
         f' * @file fft_radix5_{isa_name}_il{"_dif_tw" if mode == "dif" else ""}.h',
         f' * @brief DFT-5 {ISA} — {layout} + log3 twiddles',
         f' * addsub/fmaddsub complex arithmetic, zero shuffle overhead.',
         f' * Generated by gen_r5_native_il.py',
         f' */', f'',
         f'#ifndef {guard}', f'#define {guard}',
         f'#include <stddef.h>', f'#include <immintrin.h>', f'']

    gen = gen_dit_tw if mode == 'dit' else gen_dif_tw
    for d in ('fwd', 'bwd'):
        L.extend(gen(isa, d))
        L.append('')

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('avx2','avx512') or sys.argv[2] not in ('dit','dif'):
        print("Usage: gen_r5_native_il.py <avx2|avx512> <dit|dif>", file=sys.stderr)
        sys.exit(1)
    isa_name, mode = sys.argv[1], sys.argv[2]
    print('\n'.join(gen_file(isa_name, mode)))
