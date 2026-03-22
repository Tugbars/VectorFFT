#!/usr/bin/env python3
"""
gen_r20_il_avx2.py — DFT-20 AVX2 native interleaved CT.

Native IL: each YMM = 2 interleaved complex [re0,im0,re1,im1]
k-step = 2, ×j via addsub, complex multiply via fmaddsub/fmsubadd.

16 YMM registers — no pairing possible (DFT-5 peak = 11 YMM).
Sequential passes, zero compiler spills if register budget correct.

4×5 CT: Pass1=5×DFT-4, internal W₂₀, Pass2=4×DFT-5.
N1 + DIT + DIF, fwd + bwd = 6 functions.

Usage: python3 gen_r20_il_avx2.py
"""
import math

R, N1, N2 = 20, 4, 5

T = '__m256d'
C = 2  # complex per YMM (IL: k-step=2)
W = 4  # doubles per YMM
ATTR = '__attribute__((target("avx2,fma")))'


def w20(e):
    e = e % 20
    a = 2.0 * math.pi * e / 20.0
    return (math.cos(a), -math.sin(a))


class Emitter:
    def __init__(self): self.L=[]; self.ind=0
    def o(self, s=''): self.L.append('    '*self.ind + s)
    def c(self, s): self.o(f'/* {s} */')
    def b(self): self.L.append('')


# AVX2 IL primitives
def flip(x):
    return f'_mm256_permute_pd({x},0x5)'

def vzmul(tw, x):
    return f'_mm256_fmaddsub_pd({x},_mm256_movedup_pd({tw}),_mm256_mul_pd({flip(x)},_mm256_permute_pd({tw},0xF)))'

def vzmulj(tw, x):
    return f'_mm256_fmsubadd_pd({x},_mm256_movedup_pd({tw}),_mm256_mul_pd({flip(x)},_mm256_permute_pd({tw},0xF)))'

def vbyi(x):
    """×j: addsub(zero, flip(x)) = [-im,re,-im,re]"""
    return f'_mm256_addsub_pd(_mm256_setzero_pd(),{flip(x)})'

def vnbyi(x):
    """×(-j): sub(zero, addsub(zero, flip(x))) — or addsub then negate odds"""
    return f'_mm256_sub_pd(_mm256_setzero_pd(),_mm256_addsub_pd(_mm256_setzero_pd(),{flip(x)}))'


def emit_tw_load(em, var, idx):
    """Direct load from pre-interleaved tw_il table. Zero shuffle overhead."""
    em.o(f'{var} = _mm256_load_pd(&tw_il[({idx}*K+k)*2]);')


def emit_itw_constants(em):
    exps = set()
    for k1 in range(1, N1):
        for n2 in range(1, N2):
            e = (k1 * n2) % R
            if e != 0:
                exps.add(e)
    em.c('Internal W₂₀ pre-interleaved constants')
    for e in sorted(exps):
        wr, wi = w20(e)
        em.o(f'const {T} iw{e} = _mm256_set_pd({wi:.20e},{wr:.20e},{wi:.20e},{wr:.20e});')
    em.b()


def classify_itw(e):
    e = e % 20
    if e == 0:  return 'one'
    if e == 5:  return 'neg_j'
    if e == 10: return 'neg_one'
    if e == 15: return 'pos_j'
    return 'general'


def emit_itw_apply(em, x, e, fwd):
    e = e % 20
    cls = classify_itw(e)
    if cls == 'one': return
    if cls == 'neg_j':
        em.o(f'{x} = {vnbyi(x) if fwd else vbyi(x)};')
        return
    if cls == 'neg_one':
        em.o(f'{x} = _mm256_sub_pd(_mm256_setzero_pd(), {x});')
        return
    if cls == 'pos_j':
        em.o(f'{x} = {vbyi(x) if fwd else vnbyi(x)};')
        return
    if fwd:
        em.o(f'{x} = {vzmul(f"iw{e}", x)};')
    else:
        em.o(f'{x} = {vzmulj(f"iw{e}", x)};')


def emit_il_dft4(em, n2, has_tw, fwd, tw_mode=None):
    """Single IL DFT-4 for column n2. Peak 8 YMM."""
    inputs = [n2, n2+5, n2+10, n2+15]
    em.c(f'IL DFT-4: n2={n2}')
    em.o(f'{{')
    for j in range(4):
        em.o(f'{T} x{j}=_mm256_load_pd(&in[2*({inputs[j]}*K+k)]);')
    if has_tw:
        for j in range(4):
            m = inputs[j]
            if m > 0:
                twv = f'_tw{j}'
                em.o(f'{{ {T} {twv};')
                emit_tw_load(em, twv, m-1)
                em.o(f'  x{j} = {vzmul(twv, f"x{j}") if fwd else vzmulj(twv, f"x{j}")}; }}')
    em.o(f'{T} s=_mm256_add_pd(x0,x2), d=_mm256_sub_pd(x0,x2);')
    em.o(f'{T} t=_mm256_add_pd(x1,x3), u=_mm256_sub_pd(x1,x3);')
    nju = vnbyi('u') if fwd else vbyi('u')
    for k1, expr in [(0,'_mm256_add_pd(s,t)'), (1,f'_mm256_add_pd(d,{nju})'),
                      (2,'_mm256_sub_pd(s,t)'), (3,f'_mm256_sub_pd(d,{nju})')]:
        slot = k1*5+n2
        em.o(f'_mm256_store_pd(&sp[{slot}*{W}], {expr});')
    em.o(f'}}')
    em.b()


def emit_il_internal_twiddles(em, fwd):
    em.c('Internal W₂₀ twiddles')
    for k1 in range(1, N1):
        for n2 in range(1, N2):
            e = (k1*n2)%R
            if e == 0: continue
            slot = k1*5+n2
            x = f'_itw{slot}'
            em.o(f'{{ {T} {x}=_mm256_load_pd(&sp[{slot}*{W}]);')
            emit_itw_apply(em, x, e, fwd)
            em.o(f'  _mm256_store_pd(&sp[{slot}*{W}],{x}); }}')
    em.b()


def emit_il_dft5(em, k1, out_name, out_stride, fwd, dif_tw=False, tw_mode=None):
    """Single IL DFT-5 for row k1. Peak 11 YMM."""
    em.c(f'IL DFT-5: k1={k1}')
    em.o(f'{{')
    for n2 in range(5):
        slot = k1*5+n2
        em.o(f'{T} x{n2}=_mm256_load_pd(&sp[{slot}*{W}]);')

    cA = '_mm256_set1_pd(0.55901699437494742)'
    cB = '_mm256_set1_pd(0.95105651629515357)'
    cC = '_mm256_set1_pd(0.58778525229247313)'
    cD = '_mm256_set1_pd(0.25)'

    em.o(f'{T} s1=_mm256_add_pd(x1,x4), d1=_mm256_sub_pd(x1,x4);')
    em.o(f'{T} s2=_mm256_add_pd(x2,x3), d2=_mm256_sub_pd(x2,x3);')
    em.o(f'{T} ss=_mm256_add_pd(s1,s2);')
    em.o(f'{T} y0=_mm256_add_pd(x0,ss);')
    em.o(f'{T} t0=_mm256_fnmadd_pd({cD},ss,x0);')
    em.o(f'{T} t1=_mm256_mul_pd({cA},_mm256_sub_pd(s1,s2));')
    em.o(f'{T} p1=_mm256_add_pd(t0,t1), p2=_mm256_sub_pd(t0,t1);')
    em.o(f'{T} U=_mm256_fmadd_pd({cC},d2,_mm256_mul_pd({cB},d1));')
    em.o(f'{T} V=_mm256_fnmadd_pd({cC},d1,_mm256_mul_pd({cB},d2));')

    jU = vbyi('U')
    jV = vbyi('V')
    if fwd:
        outs = [(0,'y0'),(1,f'_mm256_sub_pd(p1,{jU})'),(2,f'_mm256_add_pd(p2,{jV})'),
                (3,f'_mm256_sub_pd(p2,{jV})'),(4,f'_mm256_add_pd(p1,{jU})')]
    else:
        outs = [(0,'y0'),(1,f'_mm256_add_pd(p1,{jU})'),(2,f'_mm256_sub_pd(p2,{jV})'),
                (3,f'_mm256_add_pd(p2,{jV})'),(4,f'_mm256_sub_pd(p1,{jU})')]

    for k2, expr in outs:
        m = k1 + N1*k2
        em.o(f'{{ {T} y={expr};')
        if dif_tw and m > 0:
            twv = f'_otw{k2}'
            em.o(f'  {{ {T} {twv};')
            emit_tw_load(em, twv, m-1)
            em.o(f'    y = {vzmul(twv, "y") if fwd else vzmulj(twv, "y")}; }}')
        em.o(f'  _mm256_store_pd(&{out_name}[2*({m}*{out_stride}+k)], y); }}')

    em.o(f'}}')
    em.b()


def gen_kernel(mode, tw_type, direction):
    """Generate one kernel. mode='n1'|'dit'|'dif', tw_type=None|'strided'."""
    fwd = direction == 'fwd'
    has_tw = mode in ('dit', 'dif')
    dif_tw = mode == 'dif'
    em = Emitter()
    name_parts = ['radix20_ct']
    if mode == 'n1': name_parts.append('n1')
    elif mode == 'dit': name_parts.append('tw_dit')
    else: name_parts.append('tw_dif')
    name_parts.extend([f'kernel_{direction}', 'il_avx2'])
    fname = '_'.join(name_parts)

    em.L.append(ATTR)
    em.L.append(f'static void {fname}(')
    em.L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    if has_tw:
        em.L.append(f'    const double * __restrict__ tw_il,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1
    emit_itw_constants(em)
    em.o(f'__attribute__((aligned(32))) double sp[{R*W}];')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    # Pass 1: 5× DFT-4
    tw_on_input = has_tw and not dif_tw  # DIT: tw on input, DIF: tw on output
    for n2 in range(N2):
        emit_il_dft4(em, n2, has_tw=tw_on_input, fwd=fwd)

    emit_il_internal_twiddles(em, fwd)

    # Pass 2: 4× DFT-5
    for k1 in range(N1):
        emit_il_dft5(em, k1, 'out', 'K', fwd, dif_tw=dif_tw)

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_file():
    guard = 'FFT_RADIX20_AVX2_IL_CT_H'
    L = ['/**',
         ' * @file fft_radix20_avx2_il_ct.h',
         ' * @brief DFT-20 AVX2 native IL — 4×5 CT',
         ' * Each YMM = 2 interleaved complex. k-step=2.',
         ' * N1 + DIT + DIF, fwd + bwd = 6 functions.',
         ' * Generated by gen_r20_il_avx2.py',
         ' */', '',
         f'#ifndef {guard}', f'#define {guard}',
         '#include <stddef.h>', '#include <immintrin.h>', '']

    for mode in ('n1', 'dit', 'dif'):
        for d in ('fwd', 'bwd'):
            L.extend(gen_kernel(mode, 'strided', d))

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    lines = gen_file()
    print('\n'.join(lines))
    import sys
    print(f'/* {len(lines)} lines */', file=sys.stderr)
