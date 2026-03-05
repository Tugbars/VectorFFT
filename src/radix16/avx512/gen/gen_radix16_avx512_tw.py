#!/usr/bin/env python3
"""
gen_radix16_avx512_tw.py — DFT-16 AVX-512 twiddled codelet (flat twiddles)

4×4 decomposition: 4 radix-4 sub-FFTs + 4 radix-4 column combines.
External twiddles applied to inputs n=1..15 before the butterfly.
k-step=8 (8-wide doubles in ZMM).

Flat twiddle table: tw_re[(n-1)*K + k], n=1..15, size = 15*K per component.
At packed T=8: 15*8*8 = 960 bytes — deep in L1, no ladder needed.

Emits: fwd + bwd kernels with configurable LD/ST macros.
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
    """Internal W₁₆ twiddles from the 4×4 decomposition."""
    tw = set()
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2 * k1) % N
            _, typ = twiddle_is_trivial(e, N)
            if typ == 'cmul': tw.add((e, N))
    return tw


class Emitter:
    def __init__(self):
        self.L = []; self.ind = 1; self.spill_c = 0; self.reload_c = 0

    def o(s, t=""): s.L.append("    " * s.ind + t)
    def c(s, t): s.o(f"/* {t} */")
    def b(s): s.L.append("")

    def emit_load(s, v, n, k_expr="k"):
        s.o(f"{v}_re = LD(&in_re[{n}*K+{k_expr}]);")
        s.o(f"{v}_im = LD(&in_im[{n}*K+{k_expr}]);")

    def emit_store(s, v, m, k_expr="k"):
        s.o(f"ST(&out_re[{m}*K+{k_expr}], {v}_re);")
        s.o(f"ST(&out_im[{m}*K+{k_expr}], {v}_im);")

    def emit_spill(s, v, slot):
        s.o(f"_mm512_store_pd(&spill_re[{slot}*8], {v}_re);")
        s.o(f"_mm512_store_pd(&spill_im[{slot}*8], {v}_im);")
        s.spill_c += 1

    def emit_reload(s, v, slot):
        s.o(f"{v}_re = _mm512_load_pd(&spill_re[{slot}*8]);")
        s.o(f"{v}_im = _mm512_load_pd(&spill_im[{slot}*8]);")
        s.reload_c += 1

    def emit_cmul_tw(s, v, tw_idx, d):
        """Apply external twiddle tw_re[(tw_idx)*K+k] to variable v."""
        fwd = (d == 'fwd')
        s.o(f"{{ const __m512d wr = LD(&tw_re[{tw_idx}*K+k]);")
        s.o(f"  const __m512d wi = LD(&tw_im[{tw_idx}*K+k]);")
        s.o(f"  const __m512d tr = {v}_re;")
        if fwd:
            s.o(f"  {v}_re = _mm512_fmsub_pd({v}_re, wr, _mm512_mul_pd({v}_im, wi));")
            s.o(f"  {v}_im = _mm512_fmadd_pd(tr, wi, _mm512_mul_pd({v}_im, wr)); }}")
        else:
            s.o(f"  {v}_re = _mm512_fmadd_pd({v}_re, wr, _mm512_mul_pd({v}_im, wi));")
            s.o(f"  {v}_im = _mm512_fmsub_pd({v}_im, wr, _mm512_mul_pd(tr, wi)); }}")

    def emit_radix4(s, v, d, label=""):
        fwd = (d == 'fwd')
        if label: s.c(f"{label} [{d}]")
        a, b, c, dd = v[0], v[1], v[2], v[3]
        s.o(f"{{ __m512d t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r=_mm512_add_pd({a}_re,{c}_re); t0i=_mm512_add_pd({a}_im,{c}_im);")
        s.o(f"  t1r=_mm512_sub_pd({a}_re,{c}_re); t1i=_mm512_sub_pd({a}_im,{c}_im);")
        s.o(f"  t2r=_mm512_add_pd({b}_re,{dd}_re); t2i=_mm512_add_pd({b}_im,{dd}_im);")
        s.o(f"  t3r=_mm512_sub_pd({b}_re,{dd}_re); t3i=_mm512_sub_pd({b}_im,{dd}_im);")
        s.o(f"  {a}_re=_mm512_add_pd(t0r,t2r); {a}_im=_mm512_add_pd(t0i,t2i);")
        s.o(f"  {c}_re=_mm512_sub_pd(t0r,t2r); {c}_im=_mm512_sub_pd(t0i,t2i);")
        if fwd:
            s.o(f"  {b}_re=_mm512_add_pd(t1r,t3i); {b}_im=_mm512_sub_pd(t1i,t3r);")
            s.o(f"  {dd}_re=_mm512_sub_pd(t1r,t3i); {dd}_im=_mm512_add_pd(t1i,t3r);")
        else:
            s.o(f"  {b}_re=_mm512_sub_pd(t1r,t3i); {b}_im=_mm512_add_pd(t1i,t3r);")
            s.o(f"  {dd}_re=_mm512_add_pd(t1r,t3i); {dd}_im=_mm512_sub_pd(t1i,t3r);")
        s.o(f"}}")

    def emit_twiddle(s, dst, src, e, tN, d):
        _, typ = twiddle_is_trivial(e, tN)
        fwd = (d == 'fwd')
        if typ == 'one':
            if dst != src: s.o(f"{dst}_re={src}_re; {dst}_im={src}_im;")
        elif typ == 'neg_one':
            s.o(f"{dst}_re=_mm512_xor_pd({src}_re,sign_flip); {dst}_im=_mm512_xor_pd({src}_im,sign_flip);")
        elif typ == 'neg_j':
            if fwd: s.o(f"{{ __m512d t={src}_re; {dst}_re={src}_im; {dst}_im=_mm512_xor_pd(t,sign_flip); }}")
            else:   s.o(f"{{ __m512d t={src}_re; {dst}_re=_mm512_xor_pd({src}_im,sign_flip); {dst}_im=t; }}")
        elif typ == 'pos_j':
            if fwd: s.o(f"{{ __m512d t={src}_re; {dst}_re=_mm512_xor_pd({src}_im,sign_flip); {dst}_im=t; }}")
            else:   s.o(f"{{ __m512d t={src}_re; {dst}_re={src}_im; {dst}_im=_mm512_xor_pd(t,sign_flip); }}")
        elif typ == 'w8_1':
            s.o(f"{{ __m512d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re=_mm512_mul_pd(_mm512_add_pd(tr,ti),sqrt2_inv); {dst}_im=_mm512_mul_pd(_mm512_sub_pd(ti,tr),sqrt2_inv); }}")
            else:   s.o(f"  {dst}_re=_mm512_mul_pd(_mm512_sub_pd(tr,ti),sqrt2_inv); {dst}_im=_mm512_mul_pd(_mm512_add_pd(tr,ti),sqrt2_inv); }}")
        elif typ == 'w8_3':
            s.o(f"{{ __m512d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re=_mm512_mul_pd(_mm512_sub_pd(ti,tr),sqrt2_inv); {dst}_im=_mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr,ti),sqrt2_inv),sign_flip); }}")
            else:   s.o(f"  {dst}_re=_mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr,ti),sqrt2_inv),sign_flip); {dst}_im=_mm512_mul_pd(_mm512_sub_pd(tr,ti),sqrt2_inv); }}")
        elif typ == 'neg_w8_1':
            s.o(f"{{ __m512d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re=_mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr,ti),sqrt2_inv),sign_flip); {dst}_im=_mm512_mul_pd(_mm512_sub_pd(tr,ti),sqrt2_inv); }}")
            else:   s.o(f"  {dst}_re=_mm512_mul_pd(_mm512_sub_pd(ti,tr),sqrt2_inv); {dst}_im=_mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr,ti),sqrt2_inv),sign_flip); }}")
        elif typ == 'neg_w8_3':
            s.o(f"{{ __m512d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re=_mm512_mul_pd(_mm512_sub_pd(tr,ti),sqrt2_inv); {dst}_im=_mm512_mul_pd(_mm512_add_pd(tr,ti),sqrt2_inv); }}")
            else:   s.o(f"  {dst}_re=_mm512_mul_pd(_mm512_add_pd(tr,ti),sqrt2_inv); {dst}_im=_mm512_mul_pd(_mm512_sub_pd(ti,tr),sqrt2_inv); }}")
        else:
            label = wN_label(e, tN)
            s.o(f"{{ __m512d tr={src}_re;")
            if fwd:
                s.o(f"  {dst}_re=_mm512_fmsub_pd({src}_re,tw_{label}_re,_mm512_mul_pd({src}_im,tw_{label}_im));")
                s.o(f"  {dst}_im=_mm512_fmadd_pd(tr,tw_{label}_im,_mm512_mul_pd({src}_im,tw_{label}_re)); }}")
            else:
                s.o(f"  {dst}_re=_mm512_fmadd_pd({src}_re,tw_{label}_re,_mm512_mul_pd({src}_im,tw_{label}_im));")
                s.o(f"  {dst}_im=_mm512_fmsub_pd({src}_im,tw_{label}_re,_mm512_mul_pd(tr,tw_{label}_im)); }}")


def emit_tw_kernel(em, d, itw_set):
    em.L.append(f'static __attribute__((target("avx512f,avx512dq,fma"))) void')
    em.L.append(f"radix16_tw_flat_dit_kernel_{d}_avx512(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1
    em.spill_c = 0; em.reload_c = 0

    em.o("const __m512d sign_flip = _mm512_set1_pd(-0.0);")
    em.o("const __m512d sqrt2_inv = _mm512_set1_pd(0.70710678118654752440);")
    em.b()
    em.o("__attribute__((aligned(64))) double spill_re[16 * 8];")
    em.o("__attribute__((aligned(64))) double spill_im[16 * 8];")
    em.b()
    em.o("__m512d x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.b()

    if itw_set:
        em.c(f"Hoisted internal W₁₆ broadcasts [{d}]")
        for (e, tN) in sorted(itw_set):
            label = wN_label(e, tN)
            em.o(f"const __m512d tw_{label}_re = _mm512_set1_pd({label}_re);")
            em.o(f"const __m512d tw_{label}_im = _mm512_set1_pd({label}_im);")
        em.b()

    xv4 = [f"x{i}" for i in range(4)]

    em.o("for (size_t k = 0; k < K; k += 8) {")
    em.ind += 1

    # PASS 1: 4 radix-4 sub-FFTs with external twiddles
    em.c(f"PASS 1: {N2} radix-{N1} sub-FFTs + external twiddles [{d}]")
    em.b()

    for n2 in range(N2):
        em.c(f"sub-FFT n₂={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n)
            if n > 0:
                em.emit_cmul_tw(f"x{n1}", n - 1, d)  # tw_re[(n-1)*K+k]
        em.b()
        em.emit_radix4(xv4, d, f"radix-4 n₂={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: 4 radix-4 column combines with internal twiddles
    em.c(f"PASS 2: {N1} radix-{N2} combines + internal twiddles [{d}]")
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
    itw_set = collect_internal_twiddles()
    em = Emitter()

    em.L.append("/**")
    em.L.append(" * @file fft_radix16_avx512_tw.h")
    em.L.append(" * @brief DFT-16 AVX-512 twiddled codelet — flat twiddles")
    em.L.append(" *")
    em.L.append(" * 4×4 Cooley-Tukey, k-step=8, 15 external twiddle loads/k-step.")
    em.L.append(f" * Internal W₁₆: {len(itw_set)} broadcast constants.")
    em.L.append(" * Flat tw table: tw_re[(n-1)*K+k], n=1..15, size=15*K per component.")
    em.L.append(" * Generated by gen_radix16_avx512_tw.py")
    em.L.append(" */")
    em.L.append("")
    em.L.append("#ifndef FFT_RADIX16_AVX512_TW_H")
    em.L.append("#define FFT_RADIX16_AVX512_TW_H")
    em.L.append("")
    em.L.append("#include <immintrin.h>")
    em.L.append("")

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
        em.L.append("")

    em.L.append("#ifndef R16L_LD")
    em.L.append('#define R16L_LD(p) _mm512_load_pd(p)')
    em.L.append("#endif")
    em.L.append("#ifndef R16L_ST")
    em.L.append('#define R16L_ST(p,v) _mm512_store_pd((p),(v))')
    em.L.append("#endif")
    em.L.append("#define LD R16L_LD")
    em.L.append("#define ST R16L_ST")
    em.L.append("")

    twf = emit_tw_kernel(em, 'fwd', itw_set)
    twb = emit_tw_kernel(em, 'bwd', itw_set)

    em.L.append("#undef LD")
    em.L.append("#undef ST")
    em.L.append("")
    em.L.append("#endif /* FFT_RADIX16_AVX512_TW_H */")

    print("\n".join(em.L))

    print(f"\n=== DFT-16 AVX-512 TWIDDLED ===", file=sys.stderr)
    print(f"  fwd: {twf[0]}sp+{twf[1]}rl={twf[0]+twf[1]} L1 ops, 15 tw loads/k-step", file=sys.stderr)
    print(f"  bwd: {twb[0]}sp+{twb[1]}rl={twb[0]+twb[1]} L1 ops", file=sys.stderr)
    print(f"  Internal W₁₆: {len(itw_set)} broadcasts", file=sys.stderr)
    print(f"  Lines: {len(em.L)}", file=sys.stderr)

if __name__ == '__main__':
    main()
