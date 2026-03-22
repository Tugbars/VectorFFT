#!/usr/bin/env python3
"""
gen_r20_il_avx512.py — DFT-20 AVX-512 native interleaved CT with ILP pairing.

Native IL: each ZMM = 4 interleaved complex [re0,im0,re1,im1,...,re3,im3]
k-step = 4, ×j via fmaddsub, complex multiply via vzmul/vzmulj.

4×5 CT with ILP pairing (32 ZMM):
  DFT-4: 4 ZMM data + ~4 temps = 8 ZMM/sub → pair 3 = 24 ZMM, 8 free
  DFT-5: 5 ZMM data + ~6 temps = 11 ZMM/sub → pair 2 = 22 ZMM, 10 free

External twiddles: direct load from pre-interleaved tw_il table.
Internal W₂₀: pre-interleaved broadcast constants.

Usage: python3 gen_r20_il_avx512.py
"""
import math

R, N1, N2 = 20, 4, 5

T = '__m512d'
C = 4  # complex per ZMM (IL: k-step=4)
W = 8  # doubles per ZMM
ATTR = '__attribute__((target("avx512f,avx512dq,fma")))'


def w20(e):
    e = e % 20
    a = 2.0 * math.pi * e / 20.0
    return (math.cos(a), -math.sin(a))


class Emitter:
    def __init__(self): self.L=[]; self.ind=0
    def o(self, s=''): self.L.append('    '*self.ind + s)
    def c(self, s): self.o(f'/* {s} */')
    def b(self): self.L.append('')


# IL primitives
def flip(x):
    return f'_mm512_permute_pd({x},0x55)'

def vzmul(tw, x):
    """IL complex multiply: tw * x (forward twiddle)."""
    return f'_mm512_fmaddsub_pd({x},_mm512_movedup_pd({tw}),_mm512_mul_pd({flip(x)},_mm512_permute_pd({tw},0xFF)))'

def vzmulj(tw, x):
    """IL conjugate complex multiply: conj(tw) * x (backward twiddle)."""
    return f'_mm512_fmsubadd_pd({x},_mm512_movedup_pd({tw}),_mm512_mul_pd({flip(x)},_mm512_permute_pd({tw},0xFF)))'

def vbyi(x):
    """Multiply by j (IL): [re,im] → [-im,re]."""
    return f'_mm512_fmaddsub_pd(_mm512_set1_pd(1.0),_mm512_setzero_pd(),{flip(x)})'

def vnbyi(x):
    """Multiply by -j (IL): [re,im] → [im,-re]."""
    return f'_mm512_fmsubadd_pd(_mm512_set1_pd(1.0),_mm512_setzero_pd(),{flip(x)})'


# ════════════════════════════════════════
# Native IL twiddle load — direct from pre-interleaved table
# ════════════════════════════════════════

def emit_tw_load(em, var, idx):
    """Direct load from pre-interleaved tw_il table. Zero shuffle overhead."""
    em.o(f'{var} = _mm512_load_pd(&tw_il[({idx}*K+k)*2]);')


# ════════════════════════════════════════
# Internal W₂₀ as pre-interleaved constants
# ════════════════════════════════════════

def emit_itw_constants(em):
    """Emit pre-interleaved W₂₀ constants for the DFT-5 cross-terms."""
    em.c('Internal W₂₀ pre-interleaved constants')
    exps = set()
    for k1 in range(1, N1):
        for n2 in range(1, N2):
            e = (k1 * n2) % R
            if e != 0:
                exps.add(e)
    for e in sorted(exps):
        wr, wi = w20(e)
        em.o(f'const {T} iw{e} = _mm512_set_pd({wi:.20e},{wr:.20e},{wi:.20e},{wr:.20e},{wi:.20e},{wr:.20e},{wi:.20e},{wr:.20e});')
    em.b()


def emit_itw_apply(em, x, e, fwd):
    """Apply internal W₂₀^e to IL variable x in-place."""
    e = e % 20
    if e == 0:
        return
    if e == 5:
        if fwd:
            em.o(f'{x} = {vnbyi(x)};')  # ×(-j) in fwd = W₂₀^5
        else:
            em.o(f'{x} = {vbyi(x)};')
        return
    if e == 10:
        em.o(f'{x} = _mm512_sub_pd(_mm512_setzero_pd(), {x});')  # ×(-1)
        return
    if e == 15:
        if fwd:
            em.o(f'{x} = {vbyi(x)};')  # ×j in fwd = W₂₀^15
        else:
            em.o(f'{x} = {vnbyi(x)};')
        return
    # General: vzmul/vzmulj with pre-interleaved constant
    if fwd:
        em.o(f'{x} = {vzmul(f"iw{e}", x)};')
    else:
        em.o(f'{x} = {vzmulj(f"iw{e}", x)};')


# ════════════════════════════════════════
# ILP-paired IL DFT-4 (3 at a time!)
# ════════════════════════════════════════

def emit_il_dft4_triple(em, n2_a, n2_b, n2_c, has_tw, fwd):
    """Three IL DFT-4s interleaved. 3×8 = 24 ZMM, 8 free.
    IL DFT-4: 4 data + 4 temps = 8 ZMM per sub-FFT."""
    em.c(f'IL DFT-4 TRIPLE: n2={n2_a},{n2_b},{n2_c} (24 ZMM, 8 free)')
    em.o(f'{{')

    prefixes = ['a', 'b', 'c']
    n2s = [n2_a, n2_b, n2_c]

    # Load (interleaved across triples)
    for j in range(4):
        for p, n2 in zip(prefixes, n2s):
            m = n2 + 5 * j
            em.o(f'{T} {p}{j}=_mm512_load_pd(&in[2*({m}*K+k)]);')

    # External twiddle
    if has_tw:
        for j in range(4):
            for p, n2 in zip(prefixes, n2s):
                m = n2 + 5 * j
                if m > 0:
                    twvar = f'_{p}tw{j}'
                    em.o(f'{{ {T} {twvar};')
                    emit_tw_load(em, twvar, m - 1)
                    if fwd:
                        em.o(f'  {p}{j} = {vzmul(twvar, f"{p}{j}")}; }}')
                    else:
                        em.o(f'  {p}{j} = {vzmulj(twvar, f"{p}{j}")}; }}')

    # DFT-4 butterfly: s=x0+x2, d=x0-x2, t=x1+x3, u=x1-x3
    for p in prefixes:
        em.o(f'{T} {p}s=_mm512_add_pd({p}0,{p}2), {p}d=_mm512_sub_pd({p}0,{p}2);')
        em.o(f'{T} {p}t=_mm512_add_pd({p}1,{p}3), {p}u=_mm512_sub_pd({p}1,{p}3);')

    # out[0]=s+t, out[1]=d+j*u, out[2]=s-t, out[3]=d-j*u
    for p, n2 in zip(prefixes, n2s):
        s, d, t, u = f'{p}s', f'{p}d', f'{p}t', f'{p}u'
        # Forward: X[1]=d+(-j)*u, X[3]=d-(-j)*u (vnbyi gives -j*x)
        # Backward: X[1]=d+j*u, X[3]=d-j*u (vbyi gives j*x)
        nju = vnbyi(u) if fwd else vbyi(u)
        em.o(f'_mm512_store_pd(&sp[{(0*5+n2)}*{W}], _mm512_add_pd({s},{t}));')
        em.o(f'_mm512_store_pd(&sp[{(1*5+n2)}*{W}], _mm512_add_pd({d},{nju}));')
        em.o(f'_mm512_store_pd(&sp[{(2*5+n2)}*{W}], _mm512_sub_pd({s},{t}));')
        em.o(f'_mm512_store_pd(&sp[{(3*5+n2)}*{W}], _mm512_sub_pd({d},{nju}));')

    em.o(f'}}')
    em.b()


def emit_il_dft4_pair(em, n2_a, n2_b, has_tw, fwd):
    """Two IL DFT-4s. 16 ZMM, 16 free."""
    em.c(f'IL DFT-4 PAIR: n2={n2_a},{n2_b}')
    em.o(f'{{')
    prefixes = ['a', 'b']
    n2s = [n2_a, n2_b]
    for j in range(4):
        for p, n2 in zip(prefixes, n2s):
            m = n2 + 5 * j
            em.o(f'{T} {p}{j}=_mm512_load_pd(&in[2*({m}*K+k)]);')
    if has_tw:
        for j in range(4):
            for p, n2 in zip(prefixes, n2s):
                m = n2 + 5 * j
                if m > 0:
                    twvar = f'_{p}tw{j}'
                    em.o(f'{{ {T} {twvar};')
                    emit_tw_load(em, twvar, m - 1)
                    if fwd:
                        em.o(f'  {p}{j} = {vzmul(twvar, f"{p}{j}")}; }}')
                    else:
                        em.o(f'  {p}{j} = {vzmulj(twvar, f"{p}{j}")}; }}')
    for p in prefixes:
        em.o(f'{T} {p}s=_mm512_add_pd({p}0,{p}2), {p}d=_mm512_sub_pd({p}0,{p}2);')
        em.o(f'{T} {p}t=_mm512_add_pd({p}1,{p}3), {p}u=_mm512_sub_pd({p}1,{p}3);')
    for p, n2 in zip(prefixes, n2s):
        s, d, t, u = f'{p}s', f'{p}d', f'{p}t', f'{p}u'
        nju = vnbyi(u) if fwd else vbyi(u)
        em.o(f'_mm512_store_pd(&sp[{(0*5+n2)}*{W}], _mm512_add_pd({s},{t}));')
        em.o(f'_mm512_store_pd(&sp[{(1*5+n2)}*{W}], _mm512_add_pd({d},{nju}));')
        em.o(f'_mm512_store_pd(&sp[{(2*5+n2)}*{W}], _mm512_sub_pd({s},{t}));')
        em.o(f'_mm512_store_pd(&sp[{(3*5+n2)}*{W}], _mm512_sub_pd({d},{nju}));')
    em.o(f'}}')
    em.b()


# ════════════════════════════════════════
# ILP-paired IL DFT-5
# ════════════════════════════════════════

def emit_il_dft5_paired(em, k1_a, k1_b, out_name, out_stride, fwd, dif_tw=False):
    """Two IL DFT-5 columns interleaved. 22 ZMM, 10 free."""
    em.c(f'IL DFT-5 PAIR: k1={k1_a},{k1_b}')
    em.o(f'{{')

    prefixes = ['a', 'b']
    k1s = [k1_a, k1_b]

    # Load from spill
    for n2 in range(5):
        for p, k1 in zip(prefixes, k1s):
            slot = k1 * 5 + n2
            em.o(f'{T} {p}{n2}=_mm512_load_pd(&sp[{slot}*{W}]);')

    # DFT-5 Rader: s1=x1+x4, s2=x2+x3, d1=x1-x4, d2=x2-x3
    for p in prefixes:
        em.o(f'{T} {p}s1=_mm512_add_pd({p}1,{p}4), {p}d1=_mm512_sub_pd({p}1,{p}4);')
        em.o(f'{T} {p}s2=_mm512_add_pd({p}2,{p}3), {p}d2=_mm512_sub_pd({p}2,{p}3);')
        em.o(f'{T} {p}ss=_mm512_add_pd({p}s1,{p}s2);')

    # Constants from memory — broadcast
    cA = '_mm512_set1_pd(0.55901699437494742)'
    cB = '_mm512_set1_pd(0.95105651629515357)'
    cC = '_mm512_set1_pd(0.58778525229247313)'
    cD = '_mm512_set1_pd(0.25)'

    # y0 = x0 + ss → store immediately
    for p, k1 in zip(prefixes, k1s):
        idx = k1 + N1 * 0
        em.o(f'{T} {p}y0=_mm512_add_pd({p}0,{p}ss);')

    for p in prefixes:
        em.o(f'{T} {p}t0=_mm512_fnmadd_pd({cD},{p}ss,{p}0);')
        em.o(f'{T} {p}t1=_mm512_mul_pd({cA},_mm512_sub_pd({p}s1,{p}s2));')
        em.o(f'{T} {p}p1=_mm512_add_pd({p}t0,{p}t1), {p}p2=_mm512_sub_pd({p}t0,{p}t1);')
        em.o(f'{T} {p}U=_mm512_fmadd_pd({cC},{p}d2,_mm512_mul_pd({cB},{p}d1));')
        em.o(f'{T} {p}V=_mm512_fnmadd_pd({cC},{p}d1,_mm512_mul_pd({cB},{p}d2));')

    # y1=p1-j*U, y4=p1+j*U, y2=p2+j*V, y3=p2-j*V (forward)
    for p, k1 in zip(prefixes, k1s):
        jU = vbyi(f'{p}U')
        jV = vbyi(f'{p}V')
        if fwd:
            outs = [
                (0, f'{p}y0'),
                (1, f'_mm512_sub_pd({p}p1,{jU})'),
                (2, f'_mm512_add_pd({p}p2,{jV})'),
                (3, f'_mm512_sub_pd({p}p2,{jV})'),
                (4, f'_mm512_add_pd({p}p1,{jU})'),
            ]
        else:
            outs = [
                (0, f'{p}y0'),
                (1, f'_mm512_add_pd({p}p1,{jU})'),
                (2, f'_mm512_sub_pd({p}p2,{jV})'),
                (3, f'_mm512_add_pd({p}p2,{jV})'),
                (4, f'_mm512_sub_pd({p}p1,{jU})'),
            ]

        for k2, expr in outs:
            m = k1 + N1 * k2
            yvar = f'{p}o{k2}'
            em.o(f'{T} {yvar}={expr};')
            # DIF: twiddle on output
            if dif_tw and m > 0:
                twvar = f'_{p}otw{k2}'
                em.o(f'{{ {T} {twvar};')
                emit_tw_load(em, twvar, m - 1)
                if fwd:
                    em.o(f'  {yvar} = {vzmul(twvar, yvar)}; }}')
                else:
                    em.o(f'  {yvar} = {vzmulj(twvar, yvar)}; }}')
            em.o(f'_mm512_store_pd(&{out_name}[2*({m}*{out_stride}+k)],{yvar});')

    em.o(f'}}')
    em.b()


# ════════════════════════════════════════
# Internal twiddles (IL)
# ════════════════════════════════════════

def emit_il_internal_twiddles(em, fwd):
    em.c('Internal W₂₀ twiddles (IL)')
    for k1 in range(1, N1):
        for n2 in range(1, N2):
            e = (k1 * n2) % R
            if e == 0:
                continue
            slot = k1 * 5 + n2
            x = f'_itw{slot}'
            em.o(f'{{ {T} {x}=_mm512_load_pd(&sp[{slot}*{W}]);')
            emit_itw_apply(em, x, e, fwd)
            em.o(f'  _mm512_store_pd(&sp[{slot}*{W}],{x}); }}')
    em.b()


# ════════════════════════════════════════
# Full kernel generators
# ════════════════════════════════════════

def gen_il_notw(direction):
    fwd = direction == 'fwd'
    em = Emitter()
    em.L.append(ATTR)
    em.L.append(f'static void radix20_ct_n1_kernel_{direction}_il_avx512(')
    em.L.append(f'    const double * __restrict__ in, double * __restrict__ out, size_t K)')
    em.L.append(f'{{')
    em.ind = 1
    emit_itw_constants(em)
    em.o(f'__attribute__((aligned(64))) double sp[{R*W}];')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.c('Pass 1: 5× IL DFT-4 (triple + pair)')
    emit_il_dft4_triple(em, 0, 1, 2, has_tw=False, fwd=fwd)
    emit_il_dft4_pair(em, 3, 4, has_tw=False, fwd=fwd)

    emit_il_internal_twiddles(em, fwd)

    em.c('Pass 2: 4× IL DFT-5 (2 pairs)')
    emit_il_dft5_paired(em, 0, 1, 'out', 'K', fwd)
    emit_il_dft5_paired(em, 2, 3, 'out', 'K', fwd)

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_il_dit_tw(direction):
    fwd = direction == 'fwd'
    em = Emitter()
    em.L.append(ATTR)
    em.L.append(f'static void radix20_ct_tw_dit_kernel_{direction}_il_avx512(')
    em.L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_il,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1
    emit_itw_constants(em)
    em.o(f'__attribute__((aligned(64))) double sp[{R*W}];')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.c('Pass 1: 5× IL DFT-4 with external twiddles')
    emit_il_dft4_triple(em, 0, 1, 2, has_tw=True, fwd=fwd)
    emit_il_dft4_pair(em, 3, 4, has_tw=True, fwd=fwd)

    emit_il_internal_twiddles(em, fwd)

    em.c('Pass 2: 4× IL DFT-5')
    emit_il_dft5_paired(em, 0, 1, 'out', 'K', fwd)
    emit_il_dft5_paired(em, 2, 3, 'out', 'K', fwd)

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_il_dif_tw(direction):
    fwd = direction == 'fwd'
    em = Emitter()
    em.L.append(ATTR)
    em.L.append(f'static void radix20_ct_tw_dif_kernel_{direction}_il_avx512(')
    em.L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_il,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1
    emit_itw_constants(em)
    em.o(f'__attribute__((aligned(64))) double sp[{R*W}];')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.c('Pass 1: 5× IL DFT-4 — NO external twiddles (DIF)')
    emit_il_dft4_triple(em, 0, 1, 2, has_tw=False, fwd=fwd)
    emit_il_dft4_pair(em, 3, 4, has_tw=False, fwd=fwd)

    emit_il_internal_twiddles(em, fwd)

    em.c('Pass 2: 4× IL DFT-5 → external twiddle on output (DIF)')
    emit_il_dft5_paired(em, 0, 1, 'out', 'K', fwd, dif_tw=True)
    emit_il_dft5_paired(em, 2, 3, 'out', 'K', fwd, dif_tw=True)

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_file():
    guard = 'FFT_RADIX20_AVX512_IL_CT_H'
    L = ['/**',
         ' * @file fft_radix20_avx512_il_ct.h',
         ' * @brief DFT-20 AVX-512 native IL — 4×5 CT with ILP pairing',
         ' * Each ZMM = 4 interleaved complex. k-step=4.',
         ' * DFT-4 triple (24 ZMM), DFT-5 pair (22 ZMM).',
         ' * N1 + DIT + DIF, fwd + bwd.',
         ' * Generated by gen_r20_il_avx512.py',
         ' */', '',
         f'#ifndef {guard}', f'#define {guard}',
         '#include <stddef.h>', '#include <immintrin.h>', '']

    for d in ('fwd', 'bwd'):
        L.extend(gen_il_notw(d))
    for d in ('fwd', 'bwd'):
        L.extend(gen_il_dit_tw(d))
    for d in ('fwd', 'bwd'):
        L.extend(gen_il_dif_tw(d))

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    lines = gen_file()
    print('\n'.join(lines))
    import sys
    print(f'/* {len(lines)} lines */', file=sys.stderr)
