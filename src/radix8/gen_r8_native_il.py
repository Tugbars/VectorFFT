#!/usr/bin/env python3
"""
gen_r8_native_il.py — DFT-8 native interleaved codelets.

FFTW genfft DAG butterfly + log3 twiddle derivation.
Native IL: addsub/fmaddsub complex arithmetic, zero shuffle overhead.
Log3: load W^1,W^2,W^4, derive W^3,W^5,W^6,W^7 via VZMUL (4 cmuls).

Usage: gen_r8_native_il.py <avx2|avx512> <dit|dif>
"""
import sys

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


def emit_log3_derive(em, isa, fwd):
    """Load W^1,W^2,W^4 from split table, derive W^3,W^5,W^6,W^7.
    W^3=W^1*W^2, W^5=W^1*W^4, W^6=W^2*W^4, W^7=W^3*W^4. 4 cmuls."""
    T = isa['T']
    zmul = isa['vzmul']
    em.c('Log3: load W^1,W^2,W^4, derive W^3,W^5,W^6,W^7 (4 cmuls)')
    em.lines(isa['tw_load']('w1', 0))
    em.lines(isa['tw_load']('w2', 1))
    em.lines(isa['tw_load']('w4', 3))
    em.o(f'const {T} w3 = {zmul("w1","w2")};')
    em.o(f'const {T} w5 = {zmul("w1","w4")};')
    em.o(f'const {T} w6 = {zmul("w2","w4")};')
    em.o(f'const {T} w7 = {zmul("w3","w4")};')
    em.b()


def emit_genfft_r8_butterfly(em, isa, fwd):
    """FFTW genfft DAG for DFT-8. Direct transcription from n1fv_8/n1bv_8.
    16 adds + 10 FMAs, 1 constant (√2/2). Uses VFMAI/VFNMSI for ×j."""
    T = isa['T']
    add, sub = isa['add'], isa['sub']
    vfma, vfnms = isa['fma'], isa['fnms']
    vfmai, vfnmsi = isa['vfmai'], isa['vfnmsi']

    em.c(f'FFTW genfft DFT-8 DAG [{"fwd" if fwd else "bwd"}]')

    # Stage 1: butterfly pairs (identical for fwd and bwd)
    em.o(f'const {T} T3 = {sub("x0","x4")};')
    em.o(f'const {T} Tj = {add("x0","x4")};')
    em.o(f'const {T} Te = {sub("x2","x6")};')
    em.o(f'const {T} Tk = {add("x2","x6")};')
    em.b()

    em.o(f'const {T} T6 = {sub("x1","x5")};')
    em.o(f'const {T} T9 = {sub("x7","x3")};')
    em.o(f'const {T} Ta = {add("T6","T9")};')
    em.o(f'const {T} Tn = {add("x7","x3")};')
    if fwd:
        em.o(f'const {T} Tf = {sub("T9","T6")};')
    else:
        em.o(f'const {T} Tf = {sub("T6","T9")};')
    em.o(f'const {T} Tm = {add("x1","x5")};')
    em.b()

    if fwd:
        # Block 1: out[1], out[7]
        em.o(f'const {T} Tb = {vfma("cW8","Ta","T3")};')
        em.o(f'const {T} Tg = {vfnms("cW8","Tf","Te")};')
        em.o(f'const {T} y1 = {vfnmsi("Tg","Tb")};')
        em.o(f'const {T} y7 = {vfmai("Tg","Tb")};')
        em.b()

        # Block 2: out[6], out[2]
        em.o(f'const {T} Tp = {sub("Tj","Tk")};')
        em.o(f'const {T} Tq = {sub("Tn","Tm")};')
        em.o(f'const {T} y6 = {vfnmsi("Tq","Tp")};')
        em.o(f'const {T} y2 = {vfmai("Tq","Tp")};')
        em.b()

        # Block 3: out[5], out[3]
        em.o(f'const {T} Th = {vfnms("cW8","Ta","T3")};')
        em.o(f'const {T} Ti = {vfma("cW8","Tf","Te")};')
        em.o(f'const {T} y5 = {vfnmsi("Ti","Th")};')
        em.o(f'const {T} y3 = {vfmai("Ti","Th")};')
        em.b()

        # Block 4: out[4], out[0]
        em.o(f'const {T} Tl = {add("Tj","Tk")};')
        em.o(f'const {T} To = {add("Tm","Tn")};')
        em.o(f'const {T} y4 = {sub("Tl","To")};')
        em.o(f'const {T} y0 = {add("Tl","To")};')
    else:
        # Block 1: out[3], out[5]
        em.o(f'const {T} Tb = {vfnms("cW8","Ta","T3")};')
        em.o(f'const {T} Tg = {vfnms("cW8","Tf","Te")};')
        em.o(f'const {T} y3 = {vfnmsi("Tg","Tb")};')
        em.o(f'const {T} y5 = {vfmai("Tg","Tb")};')
        em.b()

        # Block 2: out[4], out[0]
        em.o(f'const {T} Tp = {add("Tj","Tk")};')
        em.o(f'const {T} Tq = {add("Tm","Tn")};')
        em.o(f'const {T} y4 = {sub("Tp","Tq")};')
        em.o(f'const {T} y0 = {add("Tp","Tq")};')
        em.b()

        # Block 3: out[1], out[7]
        em.o(f'const {T} Th = {vfma("cW8","Ta","T3")};')
        em.o(f'const {T} Ti = {vfma("cW8","Tf","Te")};')
        em.o(f'const {T} y1 = {vfmai("Ti","Th")};')
        em.o(f'const {T} y7 = {vfnmsi("Ti","Th")};')
        em.b()

        # Block 4: out[6], out[2]
        em.o(f'const {T} Tl = {sub("Tj","Tk")};')
        em.o(f'const {T} To = {sub("Tm","Tn")};')
        em.o(f'const {T} y6 = {vfnmsi("To","Tl")};')
        em.o(f'const {T} y2 = {vfmai("To","Tl")};')

    return ['y0','y1','y2','y3','y4','y5','y6','y7']


def gen_dit_tw(isa_cfg, direction):
    isa = isa_cfg
    T, C = isa['T'], isa['C']
    fwd = direction == 'fwd'
    zmul = isa['vzmul'] if fwd else isa['vzmulj']
    isa_name = 'avx2' if C == 2 else 'avx512'

    em = Emitter()
    if isa['attr']: em.L.append(isa['attr'])
    em.L.append(f'static void')
    em.L.append(f'radix8_tw_dit_kernel_{direction}_il_{isa_name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(f'const {T} cW8 = {isa["set1"]("+0.707106781186547524400844362104849039284835938")};')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.o(f'{T} w1, w2, w4;')
    emit_log3_derive(em, isa, fwd)

    # Load x0 (no twiddle)
    em.o(f'{T} x0;')
    em.lines(isa['load']('x0', 0))
    # Load + twiddle x1..x7
    tw = ['w1','w2','w3','w4','w5','w6','w7']
    for n in range(1, 8):
        em.o(f'{T} r{n};')
        em.lines(isa['load'](f'r{n}', n))
        em.o(f'const {T} x{n} = {zmul(tw[n-1], f"r{n}")};')
    em.b()

    yvars = emit_genfft_r8_butterfly(em, isa, fwd)
    em.b()
    for i, yv in enumerate(yvars):
        em.lines(isa['store'](yv, i))

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
    em.L.append(f'radix8_tw_dif_kernel_{direction}_il_{isa_name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(f'const {T} cW8 = {isa["set1"]("+0.707106781186547524400844362104849039284835938")};')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.o(f'{T} w1, w2, w4;')
    emit_log3_derive(em, isa, fwd)

    for n in range(8):
        em.o(f'{T} x{n};')
        em.lines(isa['load'](f'x{n}', n))
    em.b()

    yvars = emit_genfft_r8_butterfly(em, isa, fwd)
    em.b()

    # y0: no twiddle
    em.lines(isa['store'](yvars[0], 0))
    tw = ['w1','w2','w3','w4','w5','w6','w7']
    for n in range(1, 8):
        em.o(f'{{ const {T} tw_out = {zmul(tw[n-1], yvars[n])};')
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
    guard = f'FFT_RADIX8_{ISA}_IL{"_DIF_TW" if mode == "dif" else ""}_H'

    L = [f'/**',
         f' * @file fft_radix8_{isa_name}_il{suffix}.h',
         f' * @brief DFT-8 {ISA} — native interleaved + genfft DAG + log3',
         f' * Generated by gen_r8_native_il.py',
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
        print("Usage: gen_r8_native_il.py <avx2|avx512> <dit|dif>", file=sys.stderr)
        sys.exit(1)
    print('\n'.join(gen_file(sys.argv[1], sys.argv[2])))
