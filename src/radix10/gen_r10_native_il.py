#!/usr/bin/env python3
"""
gen_r10_native_il.py — DFT-10 native interleaved codelets.

FFTW genfft DAG butterfly + log3 twiddle derivation.
4 constants, 33 stack vars — fits entirely in YMM/ZMM registers.
Log3: load W^1, derive W^2..W^9 (8 VZMUL, squaring chain).

Usage: gen_r10_native_il.py <avx2|avx512> <dit|dif>
"""
import sys

R = 10

# ════════════════════════════════════════
# ISA abstraction
# ════════════════════════════════════════

def isa_avx2():
    T = '__m256d'; C = 2
    attr = '__attribute__((target("avx2,fma")))'
    def load(v, n):  return [f'{v} = _mm256_load_pd(&in[2*({n}*K+k)]);']
    def store(v, n): return [f'_mm256_store_pd(&out[2*({n}*K+k)], {v});']
    def tw_load(v, idx):
        return [
            f'{{ __m128d _tr = _mm_load_pd(&tw_re[{idx}*K+k]);',
            f'  __m128d _ti = _mm_load_pd(&tw_im[{idx}*K+k]);',
            f'  {v} = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_unpacklo_pd(_tr,_ti)), _mm_unpackhi_pd(_tr,_ti), 1); }}']
    add  = lambda a,b: f'_mm256_add_pd({a},{b})'
    sub  = lambda a,b: f'_mm256_sub_pd({a},{b})'
    mul  = lambda a,b: f'_mm256_mul_pd({a},{b})'
    fma  = lambda a,b,c: f'_mm256_fmadd_pd({a},{b},{c})'
    fnms = lambda a,b,c: f'_mm256_fnmadd_pd({a},{b},{c})'
    set1 = lambda v: f'_mm256_set1_pd({v})'
    flip = lambda x: f'_mm256_permute_pd({x},0x5)'
    dupl = lambda x: f'_mm256_movedup_pd({x})'
    duph = lambda x: f'_mm256_permute_pd({x},0xF)'
    vfmai  = lambda b,c: f'_mm256_addsub_pd({c},{flip(b)})'
    vfnmsi = lambda b,c: f'_mm256_sub_pd({c},_mm256_addsub_pd(_mm256_setzero_pd(),{flip(b)}))'
    vzmul  = lambda tw,x: f'_mm256_fmaddsub_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    vzmulj = lambda tw,x: f'_mm256_fmsubadd_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    return dict(T=T, C=C, attr=attr, load=load, store=store, tw_load=tw_load,
                add=add, sub=sub, mul=mul, fma=fma, fnms=fnms, set1=set1,
                flip=flip, dupl=dupl, duph=duph,
                vfmai=vfmai, vfnmsi=vfnmsi, vzmul=vzmul, vzmulj=vzmulj)

def isa_avx512():
    T = '__m512d'; C = 4
    attr = '__attribute__((target("avx512f,avx512dq,fma")))'
    def load(v, n):  return [f'{v} = _mm512_load_pd(&in[2*({n}*K+k)]);']
    def store(v, n): return [f'_mm512_store_pd(&out[2*({n}*K+k)], {v});']
    def tw_load(v, idx):
        return [
            f'{{ __m256d _tr = _mm256_load_pd(&tw_re[{idx}*K+k]);',
            f'  __m256d _ti = _mm256_load_pd(&tw_im[{idx}*K+k]);',
            f'  {v} = _mm512_permutex2var_pd(_mm512_castpd256_pd512(_tr),',
            f'    _mm512_set_epi64(11,3,10,2,9,1,8,0), _mm512_castpd256_pd512(_ti)); }}']
    add  = lambda a,b: f'_mm512_add_pd({a},{b})'
    sub  = lambda a,b: f'_mm512_sub_pd({a},{b})'
    mul  = lambda a,b: f'_mm512_mul_pd({a},{b})'
    fma  = lambda a,b,c: f'_mm512_fmadd_pd({a},{b},{c})'
    fnms = lambda a,b,c: f'_mm512_fnmadd_pd({a},{b},{c})'
    set1 = lambda v: f'_mm512_set1_pd({v})'
    flip = lambda x: f'_mm512_permute_pd({x},0x55)'
    dupl = lambda x: f'_mm512_movedup_pd({x})'
    duph = lambda x: f'_mm512_permute_pd({x},0xFF)'
    vfmai  = lambda b,c: f'_mm512_fmaddsub_pd(_mm512_set1_pd(1.0),{c},{flip(b)})'
    vfnmsi = lambda b,c: f'_mm512_sub_pd({c},_mm512_fmaddsub_pd(_mm512_set1_pd(1.0),_mm512_setzero_pd(),{flip(b)}))'
    vzmul  = lambda tw,x: f'_mm512_fmaddsub_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    vzmulj = lambda tw,x: f'_mm512_fmsubadd_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    return dict(T=T, C=C, attr=attr, load=load, store=store, tw_load=tw_load,
                add=add, sub=sub, mul=mul, fma=fma, fnms=fnms, set1=set1,
                flip=flip, dupl=dupl, duph=duph,
                vfmai=vfmai, vfnmsi=vfnmsi, vzmul=vzmul, vzmulj=vzmulj)


class Emitter:
    def __init__(self): self.L=[]; self.ind=0
    def o(self, s=''): self.L.append('    '*self.ind + s)
    def c(self, s): self.o(f'/* {s} */')
    def b(self): self.L.append('')
    def lines(self, ll):
        for l in ll: self.o(l)


def emit_log3_derive(em, isa):
    """Load W^1, derive W^2..W^9 via squaring chain. 1 load + 8 VZMUL."""
    T = isa['T']
    z = isa['vzmul']
    em.c('Log3: load W^1, derive W^2..W^9 (squaring chain, 8 cmuls)')
    em.lines(isa['tw_load']('w1', 0))
    em.o(f'const {T} w2 = {z("w1","w1")};')
    em.o(f'const {T} w3 = {z("w1","w2")};')
    em.o(f'const {T} w4 = {z("w2","w2")};')
    em.o(f'const {T} w5 = {z("w1","w4")};')
    em.o(f'const {T} w6 = {z("w3","w3")};')
    em.o(f'const {T} w7 = {z("w1","w6")};')
    em.o(f'const {T} w8 = {z("w4","w4")};')
    em.o(f'const {T} w9 = {z("w1","w8")};')
    em.b()


def emit_genfft_r10_butterfly(em, isa, fwd):
    """FFTW genfft DAG for DFT-10. Direct transcription.
    Input: x0..x9. Output: stores directly.
    24 adds + 4 muls + 18 FMAs = 46 ops, 4 constants."""
    T = isa['T']
    add, sub, mul = isa['add'], isa['sub'], isa['mul']
    vfma, vfnms = isa['fma'], isa['fnms']
    vfmai, vfnmsi = isa['vfmai'], isa['vfnmsi']

    em.c(f'FFTW genfft DFT-10 DAG [{"fwd" if fwd else "bwd"}]')

    # Stage 1: radix-2 split + pair processing (identical fwd/bwd)
    em.o(f'const {T} T3 = {sub("x0","x5")};')
    em.o(f'const {T} Tr = {add("x0","x5")};')
    em.b()

    em.o(f'const {T} T6 = {sub("x2","x7")};')
    em.o(f'const {T} Ts = {add("x2","x7")};')
    em.o(f'const {T} Tg = {sub("x6","x1")};')
    em.o(f'const {T} Tw = {add("x6","x1")};')
    em.b()

    em.o(f'const {T} T9 = {sub("x8","x3")};')
    em.o(f'const {T} Tt = {add("x8","x3")};')
    em.o(f'const {T} Td = {sub("x4","x9")};')
    em.o(f'const {T} Tv = {add("x4","x9")};')
    em.b()

    # Cross-combines
    em.o(f'const {T} Tm = {sub("T6","T9")};')
    em.o(f'const {T} Tn = {sub("Td","Tg")};')
    em.o(f'const {T} TD = {sub("Ts","Tt")};')
    em.o(f'const {T} TC = {sub("Tv","Tw")};')
    em.o(f'const {T} Tu = {add("Ts","Tt")};')
    em.o(f'const {T} Tx = {add("Tv","Tw")};')
    em.o(f'const {T} Ty = {add("Tu","Tx")};')
    em.o(f'const {T} Ta = {add("T6","T9")};')
    em.o(f'const {T} Th = {add("Td","Tg")};')
    em.o(f'const {T} Ti = {add("Ta","Th")};')
    em.b()

    # Outputs: y5, y0
    em.lines(isa['store'](add("T3","Ti"), 5))
    em.lines(isa['store'](add("Tr","Ty"), 0))
    em.b()

    # Odd group (outputs 1,9,7,3)
    em.o(f'const {T} To = {mul("cK0", vfma("cK1","Tn","Tm"))};')
    em.o(f'const {T} Tq = {mul("cK0", vfnms("cK1","Tm","Tn"))};')
    em.o(f'const {T} Tj = {vfnms("cK2","Ti","T3")};')
    em.o(f'const {T} Tk = {sub("Ta","Th")};')
    em.o(f'const {T} Tl = {vfma("cK3","Tk","Tj")};')
    em.o(f'const {T} Tp = {vfnms("cK3","Tk","Tj")};')

    if fwd:
        em.lines(isa['store'](vfnmsi("To","Tl"), 1))
        em.lines(isa['store'](vfmai("Tq","Tp"), 7))
        em.lines(isa['store'](vfmai("To","Tl"), 9))
        em.lines(isa['store'](vfnmsi("Tq","Tp"), 3))
    else:
        em.lines(isa['store'](vfmai("To","Tl"), 1))
        em.lines(isa['store'](vfnmsi("Tq","Tp"), 7))
        em.lines(isa['store'](vfnmsi("To","Tl"), 9))
        em.lines(isa['store'](vfmai("Tq","Tp"), 3))
    em.b()

    # Even group (outputs 2,8,6,4)
    em.o(f'const {T} TE = {mul("cK0", vfnms("cK1","TD","TC"))};')
    em.o(f'const {T} TG = {mul("cK0", vfma("cK1","TC","TD"))};')
    em.o(f'const {T} Tz = {vfnms("cK2","Ty","Tr")};')
    em.o(f'const {T} TA = {sub("Tu","Tx")};')

    if fwd:
        em.o(f'const {T} TB = {vfnms("cK3","TA","Tz")};')
        em.o(f'const {T} TF = {vfma("cK3","TA","Tz")};')
        em.lines(isa['store'](vfmai("TE","TB"), 2))
        em.lines(isa['store'](vfnmsi("TE","TB"), 8))
        em.lines(isa['store'](vfnmsi("TG","TF"), 6))
        em.lines(isa['store'](vfmai("TG","TF"), 4))
    else:
        em.o(f'const {T} TB = {vfnms("cK3","TA","Tz")};')
        em.o(f'const {T} TF = {vfma("cK3","TA","Tz")};')
        em.lines(isa['store'](vfnmsi("TE","TB"), 2))
        em.lines(isa['store'](vfmai("TE","TB"), 8))
        em.lines(isa['store'](vfmai("TG","TF"), 6))
        em.lines(isa['store'](vfnmsi("TG","TF"), 4))


def gen_dit_tw(isa_cfg, direction):
    isa = isa_cfg
    T, C = isa['T'], isa['C']
    fwd = direction == 'fwd'
    zmul = isa['vzmul'] if fwd else isa['vzmulj']
    isa_name = 'avx2' if C == 2 else 'avx512'

    em = Emitter()
    if isa['attr']: em.L.append(isa['attr'])
    em.L.append(f'static void')
    em.L.append(f'radix10_tw_dit_kernel_{direction}_il_{isa_name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(f'const {T} cK0 = {isa["set1"]("+0.951056516295153572116439333379382143405698634")};')
    em.o(f'const {T} cK1 = {isa["set1"]("+0.618033988749894848204586834365638117720309180")};')
    em.o(f'const {T} cK2 = {isa["set1"]("+0.250000000000000000000000000000000000000000000")};')
    em.o(f'const {T} cK3 = {isa["set1"]("+0.559016994374947424102293417182819058860154590")};')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.o(f'{T} w1;')
    emit_log3_derive(em, isa)

    # Load x0 (no twiddle)
    em.o(f'{T} x0;')
    em.lines(isa['load']('x0', 0))
    # Load + twiddle x1..x9
    tw = ['w1','w2','w3','w4','w5','w6','w7','w8','w9']
    for n in range(1, 10):
        em.o(f'{T} r{n};')
        em.lines(isa['load'](f'r{n}', n))
        em.o(f'const {T} x{n} = {zmul(tw[n-1], f"r{n}")};')
    em.b()

    emit_genfft_r10_butterfly(em, isa, fwd)

    em.ind -= 1
    em.o('}')
    em.L.append('}'); em.L.append('')
    return em.L


def gen_dif_tw(isa_cfg, direction):
    isa = isa_cfg
    T, C = isa['T'], isa['C']
    fwd = direction == 'fwd'
    zmul = isa['vzmul'] if fwd else isa['vzmulj']
    isa_name = 'avx2' if C == 2 else 'avx512'

    em = Emitter()
    if isa['attr']: em.L.append(isa['attr'])
    em.L.append(f'static void')
    em.L.append(f'radix10_tw_dif_kernel_{direction}_il_{isa_name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(f'const {T} cK0 = {isa["set1"]("+0.951056516295153572116439333379382143405698634")};')
    em.o(f'const {T} cK1 = {isa["set1"]("+0.618033988749894848204586834365638117720309180")};')
    em.o(f'const {T} cK2 = {isa["set1"]("+0.250000000000000000000000000000000000000000000")};')
    em.o(f'const {T} cK3 = {isa["set1"]("+0.559016994374947424102293417182819058860154590")};')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.o(f'{T} w1;')
    emit_log3_derive(em, isa)

    # Load all (no twiddle on input for DIF)
    for n in range(10):
        em.o(f'{T} x{n};')
        em.lines(isa['load'](f'x{n}', n))
    em.b()

    # DIF: butterfly first, then need to capture outputs for twiddle
    # The genfft DAG stores directly — for DIF we need to intercept
    # Easier approach: use a temp buffer, butterfly into it, twiddle, store
    # But that's wasteful. Instead, emit DIF-specific butterfly that returns values.

    # Actually for DIF twiddled: butterfly → twiddle outputs → store
    # The genfft DAG does butterfly → store. We need butterfly → twiddle → store.
    # Capture the 10 output values before storing.

    # Rewrite: compute butterfly into y0..y9, then twiddle+store
    em.c(f'FFTW genfft DFT-10 DAG [{"fwd" if fwd else "bwd"}]')
    em.o(f'const {T} T3 = {isa["sub"]("x0","x5")};')
    em.o(f'const {T} Tr = {isa["add"]("x0","x5")};')
    em.b()
    em.o(f'const {T} T6 = {isa["sub"]("x2","x7")};')
    em.o(f'const {T} Ts = {isa["add"]("x2","x7")};')
    em.o(f'const {T} Tg = {isa["sub"]("x6","x1")};')
    em.o(f'const {T} Tw = {isa["add"]("x6","x1")};')
    em.b()
    em.o(f'const {T} T9 = {isa["sub"]("x8","x3")};')
    em.o(f'const {T} Tt = {isa["add"]("x8","x3")};')
    em.o(f'const {T} Td = {isa["sub"]("x4","x9")};')
    em.o(f'const {T} Tv = {isa["add"]("x4","x9")};')
    em.b()
    em.o(f'const {T} Tm = {isa["sub"]("T6","T9")};')
    em.o(f'const {T} Tn = {isa["sub"]("Td","Tg")};')
    em.o(f'const {T} TD = {isa["sub"]("Ts","Tt")};')
    em.o(f'const {T} TC = {isa["sub"]("Tv","Tw")};')
    em.o(f'const {T} Tu = {isa["add"]("Ts","Tt")};')
    em.o(f'const {T} Tx = {isa["add"]("Tv","Tw")};')
    em.o(f'const {T} Ty = {isa["add"]("Tu","Tx")};')
    em.o(f'const {T} Ta = {isa["add"]("T6","T9")};')
    em.o(f'const {T} Th = {isa["add"]("Td","Tg")};')
    em.o(f'const {T} Ti = {isa["add"]("Ta","Th")};')
    em.b()

    # Compute output values
    em.o(f'const {T} y0 = {isa["add"]("Tr","Ty")};')
    em.o(f'const {T} y5 = {isa["add"]("T3","Ti")};')
    em.b()

    em.o(f'const {T} To = {isa["mul"]("cK0", isa["fma"]("cK1","Tn","Tm"))};')
    em.o(f'const {T} Tq = {isa["mul"]("cK0", isa["fnms"]("cK1","Tm","Tn"))};')
    em.o(f'const {T} Tj = {isa["fnms"]("cK2","Ti","T3")};')
    em.o(f'const {T} Tk = {isa["sub"]("Ta","Th")};')
    em.o(f'const {T} Tl = {isa["fma"]("cK3","Tk","Tj")};')
    em.o(f'const {T} Tp = {isa["fnms"]("cK3","Tk","Tj")};')

    if fwd:
        em.o(f'const {T} y1 = {isa["vfnmsi"]("To","Tl")};')
        em.o(f'const {T} y7 = {isa["vfmai"]("Tq","Tp")};')
        em.o(f'const {T} y9 = {isa["vfmai"]("To","Tl")};')
        em.o(f'const {T} y3 = {isa["vfnmsi"]("Tq","Tp")};')
    else:
        em.o(f'const {T} y1 = {isa["vfmai"]("To","Tl")};')
        em.o(f'const {T} y7 = {isa["vfnmsi"]("Tq","Tp")};')
        em.o(f'const {T} y9 = {isa["vfnmsi"]("To","Tl")};')
        em.o(f'const {T} y3 = {isa["vfmai"]("Tq","Tp")};')
    em.b()

    em.o(f'const {T} TE = {isa["mul"]("cK0", isa["fnms"]("cK1","TD","TC"))};')
    em.o(f'const {T} TG = {isa["mul"]("cK0", isa["fma"]("cK1","TC","TD"))};')
    em.o(f'const {T} Tz = {isa["fnms"]("cK2","Ty","Tr")};')
    em.o(f'const {T} TA = {isa["sub"]("Tu","Tx")};')

    if fwd:
        em.o(f'const {T} TB = {isa["fnms"]("cK3","TA","Tz")};')
        em.o(f'const {T} TF = {isa["fma"]("cK3","TA","Tz")};')
        em.o(f'const {T} y2 = {isa["vfmai"]("TE","TB")};')
        em.o(f'const {T} y8 = {isa["vfnmsi"]("TE","TB")};')
        em.o(f'const {T} y6 = {isa["vfnmsi"]("TG","TF")};')
        em.o(f'const {T} y4 = {isa["vfmai"]("TG","TF")};')
    else:
        em.o(f'const {T} TB = {isa["fnms"]("cK3","TA","Tz")};')
        em.o(f'const {T} TF = {isa["fma"]("cK3","TA","Tz")};')
        em.o(f'const {T} y2 = {isa["vfnmsi"]("TE","TB")};')
        em.o(f'const {T} y8 = {isa["vfmai"]("TE","TB")};')
        em.o(f'const {T} y6 = {isa["vfmai"]("TG","TF")};')
        em.o(f'const {T} y4 = {isa["vfnmsi"]("TG","TF")};')
    em.b()

    # Twiddle outputs + store
    tw = ['w1','w2','w3','w4','w5','w6','w7','w8','w9']
    em.lines(isa['store']('y0', 0))
    for n in range(1, 10):
        em.o(f'{{ const {T} tw_out = {zmul(tw[n-1], f"y{n}")};')
        em.lines(isa['store']('tw_out', n))
        em.o(f'}}')

    em.ind -= 1
    em.o('}')
    em.L.append('}'); em.L.append('')
    return em.L


def gen_file(isa_name, mode):
    isa = isa_avx2() if isa_name == 'avx2' else isa_avx512()
    ISA = isa_name.upper()
    suffix = '_dif_tw' if mode == 'dif' else ''
    guard = f'FFT_RADIX10_{ISA}_IL{"_DIF_TW" if mode == "dif" else ""}_H'

    L = [f'/**',
         f' * @file fft_radix10_{isa_name}_il{suffix}.h',
         f' * @brief DFT-10 {ISA} — native interleaved + genfft DAG + log3',
         f' * Generated by gen_r10_native_il.py',
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
        print("Usage: gen_r10_native_il.py <avx2|avx512> <dit|dif>", file=sys.stderr)
        sys.exit(1)
    print('\n'.join(gen_file(sys.argv[1], sys.argv[2])))
