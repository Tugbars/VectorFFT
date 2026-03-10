#!/usr/bin/env python3
"""
gen_r64_avx512_tw.py — DFT-64 AVX-512 split twiddled codelet (8×8 CT + log3).

8×8 Cooley-Tukey with radix-8 butterfly (DFT-4 + W8 combine).
External twiddles: log3 — load W^1, W^8 (2 bases), derive all 63 via 12+42 cmuls.
Internal W64 twiddles: 22 general constants + 1 trivial (×-j) + 2 W8-special.
Spill buffer: 64 re + 64 im values in L1.

AVX-512: k-step=8, 32 ZMM registers.
Output: fft_radix64_avx512_tw.h (DIT fwd + bwd)
"""
import math, sys

T = '__m512d'
C = 8  # doubles per vector
ATTR = '__attribute__((target("avx512f,avx512dq,fma")))'

# ═══════════════════════════════════════
# Helpers
# ═══════════════════════════════════════

def w64(e):
    e = e % 64
    a = 2.0 * math.pi * e / 64
    return (math.cos(a), -math.sin(a))

def cmul_split(em, dst_r, dst_i, ar, ai, br, bi):
    """Emit split complex multiply: dst = a * b."""
    em.o(f'const {T} {dst_r} = _mm512_fmsub_pd({ar},{br},_mm512_mul_pd({ai},{bi}));')
    em.o(f'const {T} {dst_i} = _mm512_fmadd_pd({ar},{bi},_mm512_mul_pd({ai},{br}));')

def cmulj_split(em, dst_r, dst_i, ar, ai, br, bi):
    """Emit split conjugate complex multiply: dst = conj(a) * b."""
    em.o(f'const {T} {dst_r} = _mm512_fmadd_pd({ar},{br},_mm512_mul_pd({ai},{bi}));')
    em.o(f'const {T} {dst_i} = _mm512_fnmadd_pd({ar},{bi},_mm512_mul_pd({ai},{br}));')

def cmul_inplace_split(em, xr, xi, wr, wi, fwd):
    """Emit in-place twiddle: x *= w (fwd) or x *= conj(w) (bwd)."""
    em.o(f'{{ {T} tr = {xr};')
    if fwd:
        em.o(f'  {xr} = _mm512_fmsub_pd({xr},{wr},_mm512_mul_pd({xi},{wi}));')
        em.o(f'  {xi} = _mm512_fmadd_pd(tr,{wi},_mm512_mul_pd({xi},{wr})); }}')
    else:
        em.o(f'  {xr} = _mm512_fmadd_pd({xr},{wr},_mm512_mul_pd({xi},{wi}));')
        em.o(f'  {xi} = _mm512_fnmadd_pd(tr,{wi},_mm512_mul_pd({xi},{wr})); }}')


class Emitter:
    def __init__(self): self.L=[]; self.ind=0
    def o(self, s=''): self.L.append('    '*self.ind + s)
    def c(self, s): self.o(f'/* {s} */')
    def b(self): self.L.append('')
    def lines(self, ll):
        for l in ll: self.o(l)


# ═══════════════════════════════════════
# Radix-8 butterfly (split, inline)
# ═══════════════════════════════════════

def emit_r8_butterfly(em, fwd, xr, xi, yr, yi):
    """Radix-8 DFT butterfly in split format.
    xr/xi[0..7] → yr/yi[0..7]. Uses 'vc' (√2/2) and 'vnc' (-√2/2) from outer scope.
    Wrapped in { } scope to avoid variable name clashes across calls."""
    add = lambda a,b: f'_mm512_add_pd({a},{b})'
    sub = lambda a,b: f'_mm512_sub_pd({a},{b})'
    mul = lambda a,b: f'_mm512_mul_pd({a},{b})'

    em.o('{')
    em.ind += 1

    # DFT-4 evens
    em.o(f'const {T} epr={add(xr[0],xr[4])}, epi={add(xi[0],xi[4])};')
    em.o(f'const {T} eqr={sub(xr[0],xr[4])}, eqi={sub(xi[0],xi[4])};')
    em.o(f'const {T} err={add(xr[2],xr[6])}, eri={add(xi[2],xi[6])};')
    em.o(f'const {T} esr={sub(xr[2],xr[6])}, esi={sub(xi[2],xi[6])};')
    em.o(f'const {T} A0r={add("epr","err")}, A0i={add("epi","eri")};')
    em.o(f'const {T} A2r={sub("epr","err")}, A2i={sub("epi","eri")};')
    if fwd:
        em.o(f'const {T} A1r={add("eqr","esi")}, A1i={sub("eqi","esr")};')
        em.o(f'const {T} A3r={sub("eqr","esi")}, A3i={add("eqi","esr")};')
    else:
        em.o(f'const {T} A1r={sub("eqr","esi")}, A1i={add("eqi","esr")};')
        em.o(f'const {T} A3r={add("eqr","esi")}, A3i={sub("eqi","esr")};')

    # DFT-4 odds
    em.o(f'const {T} opr={add(xr[1],xr[5])}, opi={add(xi[1],xi[5])};')
    em.o(f'const {T} oqr={sub(xr[1],xr[5])}, oqi={sub(xi[1],xi[5])};')
    em.o(f'const {T} orr={add(xr[3],xr[7])}, ori={add(xi[3],xi[7])};')
    em.o(f'const {T} osr={sub(xr[3],xr[7])}, osi={sub(xi[3],xi[7])};')
    em.o(f'const {T} B0r={add("opr","orr")}, B0i={add("opi","ori")};')
    em.o(f'const {T} B2r={sub("opr","orr")}, B2i={sub("opi","ori")};')
    if fwd:
        em.o(f'const {T} B1r={add("oqr","osi")}, B1i={sub("oqi","osr")};')
        em.o(f'const {T} B3r={sub("oqr","osi")}, B3i={add("oqi","osr")};')
    else:
        em.o(f'const {T} B1r={sub("oqr","osi")}, B1i={add("oqi","osr")};')
        em.o(f'const {T} B3r={add("oqr","osi")}, B3i={sub("oqi","osr")};')

    # W8 combine
    em.o(f'{yr[0]}={add("A0r","B0r")}; {yi[0]}={add("A0i","B0i")};')
    em.o(f'{yr[4]}={sub("A0r","B0r")}; {yi[4]}={sub("A0i","B0i")};')

    # y1/y5: W8 combine — different formula for fwd vs bwd
    if fwd:
        em.o(f'{{ const {T} t1r={mul("vc",add("B1r","B1i"))}, t1i={mul("vc",sub("B1i","B1r"))};')
    else:
        em.o(f'{{ const {T} t1r={mul("vc",sub("B1r","B1i"))}, t1i={mul("vc",add("B1r","B1i"))};')
    em.o(f'  {yr[1]}={add("A1r","t1r")}; {yi[1]}={add("A1i","t1i")};')
    em.o(f'  {yr[5]}={sub("A1r","t1r")}; {yi[5]}={sub("A1i","t1i")}; }}')

    # y2/y6: different formula for fwd vs bwd
    if fwd:
        em.o(f'{yr[2]}={add("A2r","B2i")}; {yi[2]}={sub("A2i","B2r")};')
        em.o(f'{yr[6]}={sub("A2r","B2i")}; {yi[6]}={add("A2i","B2r")};')
    else:
        em.o(f'{yr[2]}={sub("A2r","B2i")}; {yi[2]}={add("A2i","B2r")};')
        em.o(f'{yr[6]}={add("A2r","B2i")}; {yi[6]}={sub("A2i","B2r")};')

    # y3/y7: W8³ combine — different formula for fwd vs bwd
    if fwd:
        em.o(f'{{ const {T} t3r={mul("vnc",sub("B3r","B3i"))}, t3i={mul("vnc",add("B3r","B3i"))};')
    else:
        em.o(f'{{ const {T} t3r={mul("vnc",add("B3r","B3i"))}, t3i={mul("vc",sub("B3r","B3i"))};')
    em.o(f'  {yr[3]}={add("A3r","t3r")}; {yi[3]}={add("A3i","t3i")};')
    em.o(f'  {yr[7]}={sub("A3r","t3r")}; {yi[7]}={sub("A3i","t3i")}; }}')

    em.ind -= 1
    em.o('}')


# ═══════════════════════════════════════
# Internal W64 constants
# ═══════════════════════════════════════

def collect_internal_twiddles():
    """Return unique non-trivial W64 exponents for 8×8 pass 2."""
    exps = set()
    for k1 in range(1, 8):
        for n2 in range(1, 8):
            e = (n2 * k1) % 64
            exps.add(e)
    return sorted(exps)

ITW_EXPS = collect_internal_twiddles()

def classify_itw(e):
    """Classify internal twiddle: 'trivial', 'w8', or 'general'."""
    e = e % 64
    if e == 0: return 'one'
    if e == 16: return 'neg_j'  # ×(-j)
    if e == 32: return 'neg_one'  # ×(-1)
    if e == 48: return 'pos_j'  # ×(+j)
    if e == 8 or e == 24: return 'w8'
    return 'general'


def emit_itw_constants(em):
    """Emit internal W64 broadcast constants."""
    em.c('Internal W64 twiddle constants (broadcast)')
    for e in ITW_EXPS:
        cls = classify_itw(e)
        if cls == 'general':
            wr, wi = w64(e)
            em.o(f'const {T} iw{e}r = _mm512_set1_pd({wr:.20e});')
            em.o(f'const {T} iw{e}i = _mm512_set1_pd({wi:.20e});')
    em.b()


def emit_itw_apply(em, xr, xi, e, fwd):
    """Apply internal W64^e twiddle to (xr, xi) in-place."""
    e = e % 64
    cls = classify_itw(e)

    if cls == 'one':
        return  # no-op

    if cls == 'neg_j':
        if fwd:
            em.o(f'{{ {T} tr = {xr}; {xr} = {xi}; {xi} = _mm512_sub_pd(_mm512_setzero_pd(), tr); }}')
        else:
            em.o(f'{{ {T} tr = {xr}; {xr} = _mm512_sub_pd(_mm512_setzero_pd(), {xi}); {xi} = tr; }}')
        return

    if cls == 'neg_one':
        em.o(f'{xr} = _mm512_sub_pd(_mm512_setzero_pd(), {xr});')
        em.o(f'{xi} = _mm512_sub_pd(_mm512_setzero_pd(), {xi});')
        return

    if cls == 'pos_j':
        if fwd:
            em.o(f'{{ {T} tr = {xr}; {xr} = _mm512_sub_pd(_mm512_setzero_pd(), {xi}); {xi} = tr; }}')
        else:
            em.o(f'{{ {T} tr = {xr}; {xr} = {xi}; {xi} = _mm512_sub_pd(_mm512_setzero_pd(), tr); }}')
        return

    if cls == 'w8':
        # W64^8 = W8^1 = (√2/2)(1-j), W64^24 = W8^3 = (-√2/2)(1+j)
        if e == 8:
            if fwd:
                em.o(f'{{ {T} tr = _mm512_mul_pd(vc,_mm512_add_pd({xr},{xi}));')
                em.o(f'  {xi} = _mm512_mul_pd(vc,_mm512_sub_pd({xi},{xr})); {xr} = tr; }}')
            else:
                em.o(f'{{ {T} tr = _mm512_mul_pd(vc,_mm512_sub_pd({xr},{xi}));')
                em.o(f'  {xi} = _mm512_mul_pd(vc,_mm512_add_pd({xr},{xi})); {xr} = tr; }}')
        elif e == 24:
            if fwd:
                em.o(f'{{ {T} tr = _mm512_mul_pd(vnc,_mm512_sub_pd({xr},{xi}));')
                em.o(f'  {xi} = _mm512_mul_pd(vnc,_mm512_add_pd({xr},{xi})); {xr} = tr; }}')
            else:
                em.o(f'{{ {T} tr = _mm512_mul_pd(vnc,_mm512_add_pd({xr},{xi}));')
                em.o(f'  {xi} = _mm512_mul_pd(vnc,_mm512_sub_pd({xi},{xr})); {xr} = tr; }}')
        return

    # General case
    cmul_inplace_split(em, xr, xi, f'iw{e}r', f'iw{e}i', fwd)


# ═══════════════════════════════════════
# Main generator
# ═══════════════════════════════════════

def gen_dit_tw(direction):
    fwd = direction == 'fwd'
    em = Emitter()

    em.L.append(ATTR)
    em.L.append(f'static void')
    em.L.append(f'radix64_tw_flat_dit_kernel_{direction}_avx512(')
    em.L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    # Constants
    em.o(f'const {T} vc = _mm512_set1_pd(0.707106781186547524400844362104849039284835938);')
    em.o(f'const {T} vnc = _mm512_set1_pd(-0.707106781186547524400844362104849039284835938);')
    emit_itw_constants(em)

    # Spill buffer
    em.o(f'__attribute__((aligned(64))) double sp_re[64*{C}], sp_im[64*{C}];')
    em.b()

    # Working variables
    em.o(f'{T} x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    # ── Log3: 2 base loads + 12 derivations ──
    em.c('Log3: load W^1, W^8 from table, derive column and row bases')
    em.o(f'const {T} ew1r = _mm512_load_pd(&tw_re[0*K+k]), ew1i = _mm512_load_pd(&tw_im[0*K+k]);')
    em.o(f'const {T} ew8r = _mm512_load_pd(&tw_re[7*K+k]), ew8i = _mm512_load_pd(&tw_im[7*K+k]);')
    em.b()

    # Column bases: W^2..W^7
    em.c('Column bases: W^2..W^7')
    cmul_split(em, 'ew2r','ew2i', 'ew1r','ew1i', 'ew1r','ew1i')
    cmul_split(em, 'ew3r','ew3i', 'ew1r','ew1i', 'ew2r','ew2i')
    cmul_split(em, 'ew4r','ew4i', 'ew2r','ew2i', 'ew2r','ew2i')
    cmul_split(em, 'ew5r','ew5i', 'ew1r','ew1i', 'ew4r','ew4i')
    cmul_split(em, 'ew6r','ew6i', 'ew2r','ew2i', 'ew4r','ew4i')
    cmul_split(em, 'ew7r','ew7i', 'ew1r','ew1i', 'ew6r','ew6i')
    em.b()

    # Row bases: W^16..W^56
    em.c('Row bases: W^16..W^56')
    cmul_split(em, 'ew16r','ew16i', 'ew8r','ew8i', 'ew8r','ew8i')
    cmul_split(em, 'ew24r','ew24i', 'ew8r','ew8i', 'ew16r','ew16i')
    cmul_split(em, 'ew32r','ew32i', 'ew16r','ew16i', 'ew16r','ew16i')
    cmul_split(em, 'ew40r','ew40i', 'ew8r','ew8i', 'ew32r','ew32i')
    cmul_split(em, 'ew48r','ew48i', 'ew16r','ew16i', 'ew32r','ew32i')
    cmul_split(em, 'ew56r','ew56i', 'ew8r','ew8i', 'ew48r','ew48i')
    em.b()

    row_bases = {0: None, 8: ('ew8r','ew8i'), 16: ('ew16r','ew16i'),
                 24: ('ew24r','ew24i'), 32: ('ew32r','ew32i'),
                 40: ('ew40r','ew40i'), 48: ('ew48r','ew48i'), 56: ('ew56r','ew56i')}
    col_bases = {0: None, 1: ('ew1r','ew1i'), 2: ('ew2r','ew2i'),
                 3: ('ew3r','ew3i'), 4: ('ew4r','ew4i'), 5: ('ew5r','ew5i'),
                 6: ('ew6r','ew6i'), 7: ('ew7r','ew7i')}

    xr = [f'x{i}r' for i in range(8)]
    xi = [f'x{i}i' for i in range(8)]

    # ── PASS 1: 8 row sub-FFTs ──
    for n2 in range(8):
        em.c(f'Sub-FFT n2={n2}: inputs at 8*n1+{n2} for n1=0..7')

        for n1 in range(8):
            n = 8*n1 + n2  # input index
            em.o(f'{xr[n1]} = _mm512_load_pd(&in_re[{n}*K+k]); {xi[n1]} = _mm512_load_pd(&in_im[{n}*K+k]);')

            # Apply external twiddle: W^n = W^col · W^row
            if n == 0:
                pass  # no twiddle
            elif n1 == 0:
                # Pure column twiddle
                cr, ci = col_bases[n2]
                cmul_inplace_split(em, xr[n1], xi[n1], cr, ci, fwd)
            elif n2 == 0:
                # Pure row twiddle
                rr, ri = row_bases[8*n1]
                cmul_inplace_split(em, xr[n1], xi[n1], rr, ri, fwd)
            else:
                # Cross: derive W^n = W^col · W^row, apply in-place
                cr, ci = col_bases[n2]
                rr, ri = row_bases[8*n1]
                cmul_split(em, f'ew{n}r', f'ew{n}i', cr, ci, rr, ri)
                cmul_inplace_split(em, xr[n1], xi[n1], f'ew{n}r', f'ew{n}i', fwd)

        em.b()

        # Radix-8 butterfly (in-place)
        yr = [f'x{i}r' for i in range(8)]
        yi = [f'x{i}i' for i in range(8)]
        emit_r8_butterfly(em, fwd, xr, xi, yr, yi)
        em.b()

        # Spill
        for k1 in range(8):
            slot = n2 * 8 + k1
            em.o(f'_mm512_store_pd(&sp_re[{slot}*{C}], {yr[k1]}); _mm512_store_pd(&sp_im[{slot}*{C}], {yi[k1]});')
        em.b()

    # ── PASS 2: 8 column FFTs ──
    for k1 in range(8):
        em.c(f'Column k1={k1}')

        # Reload
        for n2 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{xr[n2]} = _mm512_load_pd(&sp_re[{slot}*{C}]); {xi[n2]} = _mm512_load_pd(&sp_im[{slot}*{C}]);')

        # Internal W64 twiddles
        if k1 > 0:
            for n2 in range(1, 8):
                e = (n2 * k1) % 64
                emit_itw_apply(em, xr[n2], xi[n2], e, fwd)
        em.b()

        # Radix-8 butterfly
        emit_r8_butterfly(em, fwd, xr, xi, xr, xi)
        em.b()

        # Store — output index m = k1 + 8*k2
        for k2 in range(8):
            m = k1 + 8 * k2
            em.o(f'_mm512_store_pd(&out_re[{m}*K+k], {xr[k2]}); _mm512_store_pd(&out_im[{m}*K+k], {xi[k2]});')
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


def gen_file():
    guard = 'FFT_RADIX64_AVX512_TW_H'
    L = [f'/**',
         f' * @file fft_radix64_avx512_tw.h',
         f' * @brief DFT-64 AVX-512 split — 8×8 CT + log3 twiddles',
         f' * 2 base loads + 12 derivations. Radix-8 butterfly.',
         f' * Internal W64 twiddles as broadcast constants.',
         f' * Generated by gen_r64_avx512_tw.py',
         f' */', f'',
         f'#ifndef {guard}', f'#define {guard}',
         f'#include <stddef.h>', f'#include <immintrin.h>', f'']

    for d in ('fwd', 'bwd'):
        L.extend(gen_dit_tw(d))
        L.append('')

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    print('\n'.join(gen_file()))
