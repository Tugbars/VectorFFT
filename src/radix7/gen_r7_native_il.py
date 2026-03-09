#!/usr/bin/env python3
"""
gen_r7_native_il.py — DFT-7 native interleaved codelets.

FFTW genfft DAG butterfly + log3 twiddle derivation.
Native IL: addsub/fmaddsub complex arithmetic, zero shuffle overhead.
Log3: load W^1, derive W^2..W^6 via VZMUL (5 cmuls, depth-3 tree).

Usage: gen_r7_native_il.py <avx2|avx512> <dit|dif>
"""
import sys

# ════════════════════════════════════════
# ISA abstraction (reuse from R=5)
# ════════════════════════════════════════

def isa_avx2():
    T = '__m256d'
    C = 2
    attr = '__attribute__((target("avx2,fma")))'
    def load(v, n):
        return [f'{v} = _mm256_load_pd(&in[2*({n}*K+k)]);']
    def store(v, n):
        return [f'_mm256_store_pd(&out[2*({n}*K+k)], {v});']
    def tw_load(v, idx):
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
    flip = lambda x: f'_mm256_permute_pd({x},0x5)'
    dupl = lambda x: f'_mm256_movedup_pd({x})'
    duph = lambda x: f'_mm256_permute_pd({x},0xF)'
    vfmai  = lambda b,c: f'_mm256_addsub_pd({c},{flip(b)})'
    vfnmsi = lambda b,c: f'_mm256_sub_pd({c},_mm256_addsub_pd(_mm256_setzero_pd(),{flip(b)}))'
    vzmul  = lambda tw,x: f'_mm256_fmaddsub_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    vzmulj = lambda tw,x: f'_mm256_fmsubadd_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    return dict(T=T, C=C, attr=attr, load=load, store=store, tw_load=tw_load,
                add=add, sub=sub, mul=mul, fma=fma, fnma=fnma, set1=set1,
                flip=flip, dupl=dupl, duph=duph,
                vfmai=vfmai, vfnmsi=vfnmsi, vzmul=vzmul, vzmulj=vzmulj)

def isa_avx512():
    T = '__m512d'
    C = 4
    attr = '__attribute__((target("avx512f,avx512dq,fma")))'
    def load(v, n):
        return [f'{v} = _mm512_load_pd(&in[2*({n}*K+k)]);']
    def store(v, n):
        return [f'_mm512_store_pd(&out[2*({n}*K+k)], {v});']
    def tw_load(v, idx):
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
    flip = lambda x: f'_mm512_permute_pd({x},0x55)'
    dupl = lambda x: f'_mm512_movedup_pd({x})'
    duph = lambda x: f'_mm512_permute_pd({x},0xFF)'
    vfmai  = lambda b,c: f'_mm512_fmaddsub_pd(_mm512_set1_pd(1.0),{c},{flip(b)})'
    vfnmsi = lambda b,c: f'_mm512_sub_pd({c},_mm512_fmaddsub_pd(_mm512_set1_pd(1.0),_mm512_setzero_pd(),{flip(b)}))'
    vzmul  = lambda tw,x: f'_mm512_fmaddsub_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    vzmulj = lambda tw,x: f'_mm512_fmsubadd_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    return dict(T=T, C=C, attr=attr, load=load, store=store, tw_load=tw_load,
                add=add, sub=sub, mul=mul, fma=fma, fnma=fnma, set1=set1,
                flip=flip, dupl=dupl, duph=duph,
                vfmai=vfmai, vfnmsi=vfnmsi, vzmul=vzmul, vzmulj=vzmulj)


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
    """Load W^1, derive W^2..W^6 in interleaved form.
    Tree: W^2=W^1*W^1, W^3=W^1*W^2, W^4=W^2*W^2, W^5=W^2*W^3, W^6=W^3*W^3
    5 cmuls, depth 3."""
    T = isa['T']
    zmul = isa['vzmul']
    em.c('Log3: load W^1, derive W^2..W^6 (5 cmuls, depth-3 tree)')
    em.lines(isa['tw_load']('w1', 0))
    em.o(f'const {T} w2 = {zmul("w1","w1")};')
    em.o(f'const {T} w3 = {zmul("w1","w2")};')
    em.o(f'const {T} w4 = {zmul("w2","w2")};')
    em.o(f'const {T} w5 = {zmul("w2","w3")};')
    em.o(f'const {T} w6 = {zmul("w3","w3")};')
    em.b()


def emit_genfft_r7_butterfly(em, isa, fwd):
    """FFTW genfft DAG for DFT-7. Direct transcription from FFTW n1fv_7/n1bv_7.
    Input: x0..x6. Output: y0..y6.
    9 adds + 3 muls + 21 FMAs = 33 ops, 6 constants."""
    T = isa['T']
    add, sub, mul = isa['add'], isa['sub'], isa['mul']
    fma, fnma = isa['fma'], isa['fnma']
    vfmai, vfnmsi = isa['vfmai'], isa['vfnmsi']

    em.c(f'FFTW genfft DFT-7 DAG [{"fwd" if fwd else "bwd"}]')

    # Sums (same for fwd and bwd)
    em.o(f'const {T} T4 = {add("x1","x6")};')
    em.o(f'const {T} Ta = {add("x3","x4")};')
    em.o(f'const {T} T7 = {add("x2","x5")};')

    if fwd:
        # fwd: diffs = x[7-n] - x[n]
        em.o(f'const {T} Te = {sub("x6","x1")};')
        em.o(f'const {T} Tf = {sub("x4","x3")};')
        em.o(f'const {T} Tg = {sub("x5","x2")};')
        em.b()

        em.o(f'const {T} Tb = {fnma("cK0","T4","Ta")};')  # Ta - cK0*T4
        em.o(f'const {T} Th = {fma("cK1","Tg","Tf")};')    # cK1*Tg + Tf
        em.o(f'const {T} Tr = {fnma("cK1","Te","Tg")};')   # Tg - cK1*Te
        em.o(f'const {T} To = {fnma("cK0","Ta","T7")};')   # T7 - cK0*Ta
        em.o(f'const {T} Tm = {fma("cK1","Tf","Te")};')    # cK1*Tf + Te
        em.o(f'const {T} Tj = {fnma("cK0","T7","T4")};')   # T4 - cK0*T7
        em.b()

        em.o(f'const {T} y0 = {add("x0", add("T4", add("T7","Ta")))};')
        em.b()

        # Pair 1: out[4], out[3]
        em.o(f'const {T} Ts = {mul("cK4", fnma("cK3","Tr","Tf"))};')  # cK4*(Tf - cK3*Tr)
        em.o(f'const {T} Tp = {fnma("cK2","To","T4")};')              # T4 - cK2*To
        em.o(f'const {T} Tq = {fnma("cK5","Tp","x0")};')
        em.o(f'const {T} y4 = {vfnmsi("Ts","Tq")};')
        em.o(f'const {T} y3 = {vfmai("Ts","Tq")};')
        em.b()

        # Pair 2: out[5], out[2]
        em.o(f'const {T} Ti = {mul("cK4", fnma("cK3","Th","Te"))};')  # cK4*(Te - cK3*Th)
        em.o(f'const {T} Tc = {fnma("cK2","Tb","T7")};')              # T7 - cK2*Tb
        em.o(f'const {T} Td = {fnma("cK5","Tc","x0")};')
        em.o(f'const {T} y5 = {vfnmsi("Ti","Td")};')
        em.o(f'const {T} y2 = {vfmai("Ti","Td")};')
        em.b()

        # Pair 3: out[6], out[1]
        em.o(f'const {T} Tn = {mul("cK4", fma("cK3","Tm","Tg"))};')  # cK4*(cK3*Tm + Tg)
        em.o(f'const {T} Tk = {fnma("cK2","Tj","Ta")};')              # Ta - cK2*Tj
        em.o(f'const {T} Tl = {fnma("cK5","Tk","x0")};')
        em.o(f'const {T} y6 = {vfnmsi("Tn","Tl")};')
        em.o(f'const {T} y1 = {vfmai("Tn","Tl")};')

    else:
        # bwd: diffs reversed from fwd (note variable name shuffle!)
        em.o(f'const {T} Tg = {sub("x1","x6")};')  # note: Tg here, was Te in fwd
        em.o(f'const {T} Te = {sub("x3","x4")};')   # note: Te here, was Tf in fwd
        em.o(f'const {T} Tf = {sub("x2","x5")};')   # note: Tf here, was Tg in fwd
        em.b()

        em.o(f'const {T} Tb = {fnma("cK0","Ta","T7")};')  # T7 - cK0*Ta
        em.o(f'const {T} Th = {fnma("cK1","Tg","Tf")};')  # Tf - cK1*Tg
        em.o(f'const {T} Tr = {fma("cK1","Te","Tg")};')    # cK1*Te + Tg
        em.o(f'const {T} To = {fnma("cK0","T7","T4")};')   # T4 - cK0*T7
        em.o(f'const {T} Tm = {fma("cK1","Tf","Te")};')    # cK1*Tf + Te
        em.o(f'const {T} Tj = {fnma("cK0","T4","Ta")};')   # Ta - cK0*T4
        em.b()

        em.o(f'const {T} y0 = {add("x0", add("T4", add("T7","Ta")))};')
        em.b()

        # Pair 1: out[1], out[6]
        em.o(f'const {T} Ts = {mul("cK4", fma("cK3","Tr","Tf"))};')   # cK4*(cK3*Tr + Tf)
        em.o(f'const {T} Tp = {fnma("cK2","To","Ta")};')               # Ta - cK2*To
        em.o(f'const {T} Tq = {fnma("cK5","Tp","x0")};')
        em.o(f'const {T} y1 = {vfmai("Ts","Tq")};')
        em.o(f'const {T} y6 = {vfnmsi("Ts","Tq")};')
        em.b()

        # Pair 2: out[3], out[4]
        em.o(f'const {T} Ti = {mul("cK4", fnma("cK3","Th","Te"))};')  # cK4*(Te - cK3*Th)
        em.o(f'const {T} Tc = {fnma("cK2","Tb","T4")};')               # T4 - cK2*Tb
        em.o(f'const {T} Td = {fnma("cK5","Tc","x0")};')
        em.o(f'const {T} y3 = {vfmai("Ti","Td")};')
        em.o(f'const {T} y4 = {vfnmsi("Ti","Td")};')
        em.b()

        # Pair 3: out[2], out[5]
        em.o(f'const {T} Tn = {mul("cK4", fnma("cK3","Tm","Tg"))};')  # cK4*(Tg - cK3*Tm)
        em.o(f'const {T} Tk = {fnma("cK2","Tj","T7")};')               # T7 - cK2*Tj
        em.o(f'const {T} Tl = {fnma("cK5","Tk","x0")};')
        em.o(f'const {T} y2 = {vfmai("Tn","Tl")};')
        em.o(f'const {T} y5 = {vfnmsi("Tn","Tl")};')

    return ['y0','y1','y2','y3','y4','y5','y6']


def gen_dit_tw(isa_cfg, direction):
    isa = isa_cfg
    T, C = isa['T'], isa['C']
    fwd = direction == 'fwd'
    zmul = isa['vzmul'] if fwd else isa['vzmulj']
    isa_name = 'avx2' if C == 2 else 'avx512'

    em = Emitter(isa)
    em.L = []
    if isa['attr']: em.L.append(isa['attr'])
    em.L.append(f'static void')
    em.L.append(f'radix7_tw_dit_kernel_{direction}_il_{isa_name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    # Constants (from FFTW genfft)
    em.o(f'const {T} cK0 = {isa["set1"]("+0.356895867892209443894399510021300583399127187")};')
    em.o(f'const {T} cK1 = {isa["set1"]("+0.554958132087371191422194871006410481067288862")};')
    em.o(f'const {T} cK2 = {isa["set1"]("+0.692021471630095869627814897002069140197260599")};')
    em.o(f'const {T} cK3 = {isa["set1"]("+0.801937735804838252472204639014890102331838324")};')
    em.o(f'const {T} cK4 = {isa["set1"]("+0.974927912181823607018131682993931217232785801")};')
    em.o(f'const {T} cK5 = {isa["set1"]("+0.900968867902419126236102319507445051165919162")};')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.o(f'{T} w1;')
    emit_log3_derive(em, isa, fwd)

    # Load x0 (no twiddle)
    em.o(f'{T} x0;')
    em.lines(isa['load']('x0', 0))

    # Load + twiddle x1..x6
    tw_names = ['w1','w2','w3','w4','w5','w6']
    for n in range(1, 7):
        em.o(f'{T} r{n};')
        em.lines(isa['load'](f'r{n}', n))
        em.o(f'const {T} x{n} = {zmul(tw_names[n-1], f"r{n}")};')
    em.b()

    yvars = emit_genfft_r7_butterfly(em, isa, fwd)
    em.b()

    for i, yv in enumerate(yvars):
        em.lines(isa['store'](yv, i))

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


def gen_dif_tw(isa_cfg, direction):
    isa = isa_cfg
    T, C = isa['T'], isa['C']
    fwd = direction == 'fwd'
    zmul = isa['vzmul'] if fwd else isa['vzmulj']
    isa_name = 'avx2' if C == 2 else 'avx512'

    em = Emitter(isa)
    em.L = []
    if isa['attr']: em.L.append(isa['attr'])
    em.L.append(f'static void')
    em.L.append(f'radix7_tw_dif_kernel_{direction}_il_{isa_name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(f'const {T} cK0 = {isa["set1"]("+0.356895867892209443894399510021300583399127187")};')
    em.o(f'const {T} cK1 = {isa["set1"]("+0.554958132087371191422194871006410481067288862")};')
    em.o(f'const {T} cK2 = {isa["set1"]("+0.692021471630095869627814897002069140197260599")};')
    em.o(f'const {T} cK3 = {isa["set1"]("+0.801937735804838252472204639014890102331838324")};')
    em.o(f'const {T} cK4 = {isa["set1"]("+0.974927912181823607018131682993931217232785801")};')
    em.o(f'const {T} cK5 = {isa["set1"]("+0.900968867902419126236102319507445051165919162")};')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.o(f'{T} w1;')
    emit_log3_derive(em, isa, fwd)

    # Load all inputs (no twiddle)
    for n in range(7):
        em.o(f'{T} x{n};')
        em.lines(isa['load'](f'x{n}', n))
    em.b()

    yvars = emit_genfft_r7_butterfly(em, isa, fwd)
    em.b()

    # Twiddle outputs + store
    em.lines(isa['store'](yvars[0], 0))  # y0: no twiddle
    tw_names = ['w1','w2','w3','w4','w5','w6']
    for n in range(1, 7):
        em.o(f'{{ const {T} tw_out = {zmul(tw_names[n-1], yvars[n])};')
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
    guard = f'FFT_RADIX7_{ISA}_IL_{"DIF_TW_" if mode == "dif" else ""}H'

    L = [f'/**',
         f' * @file fft_radix7_{isa_name}_il{"_dif_tw" if mode == "dif" else ""}.h',
         f' * @brief DFT-7 {ISA} — native interleaved + FFTW genfft DAG + log3',
         f' * addsub/fmaddsub arithmetic, zero shuffle overhead.',
         f' * Generated by gen_r7_native_il.py',
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
        print("Usage: gen_r7_native_il.py <avx2|avx512> <dit|dif>", file=sys.stderr)
        sys.exit(1)
    print('\n'.join(gen_file(sys.argv[1], sys.argv[2])))
