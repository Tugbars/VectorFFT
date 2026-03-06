#!/usr/bin/env python3
"""
gen_radix32_avx2_tw_v2.py — Flat-twiddle DFT-32 AVX2 codegen v2

AVX2 port with v2 fused-spill improvement.
  - 4-wide doubles (__m256d), 16 YMM registers
  - k-step = 4, flat twiddles only (no ladder — needs 28 regs)
  - U=1 only (U=2 needs 32+ regs)
  - NFUSE=2: keep 2 results from last sub-FFT in s-regs
    (NFUSE=4 would need 16 ymm at pass-2 entry → 0 free for twiddle ops)
  - 8×4 decomposition, fwd + bwd

Register budget at pass-2 k1=0 (worst case):
  x0..x3 re+im = 8 ymm | s1 re+im = 2 ymm | constants = 2 ymm → 12/16

Flat twiddle table: tw_re[31*K], tw_im[31*K].
"""

import math, sys

N, N1, N2 = 32, 8, 4
NFUSE = 2  # AVX2: 2 results kept live (saves 8 mem ops/k-step)

def wN(e, tN):
    e = e % tN
    a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def twiddle_is_trivial(e, tN):
    e = e % tN
    if e == 0: return True, 'one'
    if (8*e) % tN == 0:
        o = (8*e)//tN
        t = ['one','w8_1','neg_j','w8_3','neg_one','neg_w8_1','pos_j','neg_w8_3']
        return True, t[o%8]
    return False, 'cmul'

def collect_internal_twiddles():
    tw = set()
    for n2 in range(1,N2):
        for k1 in range(1,N1):
            e=(n2*k1)%N
            _,t=twiddle_is_trivial(e,N)
            if t=='cmul': tw.add((e,N))
    return tw


class E:
    """AVX2 emitter — 4-wide doubles, 16 YMM registers."""
    def __init__(self):
        self.L=[]; self.ind=1; self.spill_c=0; self.reload_c=0
    def o(s,t=""): s.L.append("    "*s.ind+t)
    def c(s,t): s.o(f"/* {t} */")
    def b(s): s.L.append("")
    def add(s,a,b): return f"_mm256_add_pd({a},{b})"
    def sub(s,a,b): return f"_mm256_sub_pd({a},{b})"
    def mul(s,a,b): return f"_mm256_mul_pd({a},{b})"
    def neg(s,a):   return f"_mm256_xor_pd({a},sign_flip)"
    def fma(s,a,b,c): return f"_mm256_fmadd_pd({a},{b},{c})"
    def fms(s,a,b,c): return f"_mm256_fmsub_pd({a},{b},{c})"

    def emit_load(s,v,n,k_expr="k"):
        s.o(f"{v}_re = LD(&in_re[{n}*K+{k_expr}]);")
        s.o(f"{v}_im = LD(&in_im[{n}*K+{k_expr}]);")
    def emit_store(s,v,m,k_expr="k"):
        s.o(f"ST(&out_re[{m}*K+{k_expr}],{v}_re);")
        s.o(f"ST(&out_im[{m}*K+{k_expr}],{v}_im);")
    def emit_spill(s,v,slot):
        s.o(f"_mm256_store_pd(&spill_re[{slot}*4],{v}_re);")
        s.o(f"_mm256_store_pd(&spill_im[{slot}*4],{v}_im);")
        s.spill_c+=1
    def emit_reload(s,v,slot):
        s.o(f"{v}_re = _mm256_load_pd(&spill_re[{slot}*4]);")
        s.o(f"{v}_im = _mm256_load_pd(&spill_im[{slot}*4]);")
        s.reload_c+=1

    def emit_cmul_inplace(s, v, wr, wi, d):
        fwd = (d=='fwd')
        s.o(f"{{ __m256d tr={v}_re;")
        if fwd:
            s.o(f"  {v}_re={s.fms(f'{v}_re',wr,s.mul(f'{v}_im',wi))};")
            s.o(f"  {v}_im={s.fma('tr',wi,s.mul(f'{v}_im',wr))}; }}")
        else:
            s.o(f"  {v}_re={s.fma(f'{v}_re',wr,s.mul(f'{v}_im',wi))};")
            s.o(f"  {v}_im={s.fms(f'{v}_im',wr,s.mul('tr',wi))}; }}")

    def emit_radix8(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        s.o(f"{{ __m256d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;")
        s.o(f"  __m256d t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={s.add(f'{v[0]}_re',f'{v[4]}_re')}; t0i={s.add(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[0]}_re',f'{v[4]}_re')}; t1i={s.sub(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t2r={s.add(f'{v[2]}_re',f'{v[6]}_re')}; t2i={s.add(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[2]}_re',f'{v[6]}_re')}; t3i={s.sub(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  e0r={s.add('t0r','t2r')}; e0i={s.add('t0i','t2i')};")
        s.o(f"  e2r={s.sub('t0r','t2r')}; e2i={s.sub('t0i','t2i')};")
        j_add, j_sub = ('add','sub') if fwd else ('sub','add')
        s.o(f"  e1r={getattr(s,j_add)('t1r','t3i')}; e1i={getattr(s,j_sub)('t1i','t3r')};")
        s.o(f"  e3r={getattr(s,j_sub)('t1r','t3i')}; e3i={getattr(s,j_add)('t1i','t3r')};")
        s.o(f"  __m256d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;")
        s.o(f"  t0r={s.add(f'{v[1]}_re',f'{v[5]}_re')}; t0i={s.add(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[1]}_re',f'{v[5]}_re')}; t1i={s.sub(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t2r={s.add(f'{v[3]}_re',f'{v[7]}_re')}; t2i={s.add(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[3]}_re',f'{v[7]}_re')}; t3i={s.sub(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  o0r={s.add('t0r','t2r')}; o0i={s.add('t0i','t2i')};")
        s.o(f"  o2r={s.sub('t0r','t2r')}; o2i={s.sub('t0i','t2i')};")
        s.o(f"  o1r={getattr(s,j_add)('t1r','t3i')}; o1i={getattr(s,j_sub)('t1i','t3r')};")
        s.o(f"  o3r={getattr(s,j_sub)('t1r','t3i')}; o3i={getattr(s,j_add)('t1i','t3r')};")
        if fwd:
            s.o(f"  t0r={s.mul(s.add('o1r','o1i'),'sqrt2_inv')};")
            s.o(f"  t0i={s.mul(s.sub('o1i','o1r'),'sqrt2_inv')};")
            s.o(f"  o1r=t0r; o1i=t0i;")
            s.o(f"  t0r=o2i; t0i={s.neg('o2r')};")
            s.o(f"  o2r=t0r; o2i=t0i;")
            s.o(f"  t0r={s.mul(s.sub('o3i','o3r'),'sqrt2_inv')};")
            s.o(f"  t0i={s.neg(s.mul(s.add('o3r','o3i'),'sqrt2_inv'))};")
            s.o(f"  o3r=t0r; o3i=t0i;")
        else:
            s.o(f"  t0r={s.mul(s.sub('o1r','o1i'),'sqrt2_inv')};")
            s.o(f"  t0i={s.mul(s.add('o1r','o1i'),'sqrt2_inv')};")
            s.o(f"  o1r=t0r; o1i=t0i;")
            s.o(f"  t0r={s.neg('o2i')}; t0i=o2r;")
            s.o(f"  o2r=t0r; o2i=t0i;")
            s.o(f"  t0r={s.neg(s.mul(s.add('o3r','o3i'),'sqrt2_inv'))};")
            s.o(f"  t0i={s.mul(s.sub('o3r','o3i'),'sqrt2_inv')};")
            s.o(f"  o3r=t0r; o3i=t0i;")
        for i,j in [(0,4),(1,5),(2,6),(3,7)]:
            s.o(f"  {v[i]}_re={s.add(f'e{i}r',f'o{i}r')}; {v[i]}_im={s.add(f'e{i}i',f'o{i}i')};")
            s.o(f"  {v[j]}_re={s.sub(f'e{i}r',f'o{i}r')}; {v[j]}_im={s.sub(f'e{i}i',f'o{i}i')};")
        s.o(f"}}")

    def emit_radix4(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        a,b,c,dd=v[0],v[1],v[2],v[3]
        s.o(f"{{ __m256d t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={s.add(f'{a}_re',f'{c}_re')}; t0i={s.add(f'{a}_im',f'{c}_im')};")
        s.o(f"  t1r={s.sub(f'{a}_re',f'{c}_re')}; t1i={s.sub(f'{a}_im',f'{c}_im')};")
        s.o(f"  t2r={s.add(f'{b}_re',f'{dd}_re')}; t2i={s.add(f'{b}_im',f'{dd}_im')};")
        s.o(f"  t3r={s.sub(f'{b}_re',f'{dd}_re')}; t3i={s.sub(f'{b}_im',f'{dd}_im')};")
        s.o(f"  {a}_re={s.add('t0r','t2r')}; {a}_im={s.add('t0i','t2i')};")
        s.o(f"  {c}_re={s.sub('t0r','t2r')}; {c}_im={s.sub('t0i','t2i')};")
        if fwd:
            s.o(f"  {b}_re={s.add('t1r','t3i')}; {b}_im={s.sub('t1i','t3r')};")
            s.o(f"  {dd}_re={s.sub('t1r','t3i')}; {dd}_im={s.add('t1i','t3r')};")
        else:
            s.o(f"  {b}_re={s.sub('t1r','t3i')}; {b}_im={s.add('t1i','t3r')};")
            s.o(f"  {dd}_re={s.add('t1r','t3i')}; {dd}_im={s.sub('t1i','t3r')};")
        s.o(f"}}")

    def emit_twiddle(s,dst,src,e,tN,d):
        _,typ=twiddle_is_trivial(e,tN)
        fwd=(d=='fwd')
        if typ=='one':
            if dst!=src: s.o(f"{dst}_re={src}_re; {dst}_im={src}_im;")
        elif typ=='neg_one':
            s.o(f"{dst}_re={s.neg(f'{src}_re')}; {dst}_im={s.neg(f'{src}_im')};")
        elif typ=='neg_j':
            if fwd: s.o(f"{{ __m256d t={src}_re; {dst}_re={src}_im; {dst}_im={s.neg('t')}; }}")
            else:   s.o(f"{{ __m256d t={src}_re; {dst}_re={s.neg(f'{src}_im')}; {dst}_im=t; }}")
        elif typ=='pos_j':
            if fwd: s.o(f"{{ __m256d t={src}_re; {dst}_re={s.neg(f'{src}_im')}; {dst}_im=t; }}")
            else:   s.o(f"{{ __m256d t={src}_re; {dst}_re={src}_im; {dst}_im={s.neg('t')}; }}")
        elif typ=='w8_1':
            s.o(f"{{ __m256d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.add('tr','ti'),'sqrt2_inv')}; }}")
        elif typ=='w8_3':
            s.o(f"{{ __m256d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; }}")
            else:   s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; }}")
        elif typ=='neg_w8_1':
            s.o(f"{{ __m256d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; }}")
        elif typ=='neg_w8_3':
            s.o(f"{{ __m256d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.add('tr','ti'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; }}")
        else:
            label=wN_label(e,tN)
            s.o(f"{{ __m256d tr={src}_re;")
            if fwd:
                s.o(f"  {dst}_re={s.fms(f'{src}_re',f'tw_{label}_re',s.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={s.fma('tr',f'tw_{label}_im',s.mul(f'{src}_im',f'tw_{label}_re'))}; }}")
            else:
                s.o(f"  {dst}_re={s.fma(f'{src}_re',f'tw_{label}_re',s.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={s.fms(f'{src}_im',f'tw_{label}_re',s.mul('tr',f'tw_{label}_im'))}; }}")


def emit_flat_kernel(em, d, itw_set):
    em.L.append(f'static __attribute__((target("avx2,fma"))) void')
    em.L.append(f"radix32_tw_flat_dit_kernel_{d}_avx2(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1; em.spill_c = 0; em.reload_c = 0

    em.o("const __m256d sign_flip = _mm256_set1_pd(-0.0);")
    em.o("const __m256d sqrt2_inv = _mm256_set1_pd(0.70710678118654752440);")
    em.b()
    em.o("__attribute__((aligned(32))) double spill_re[128];")
    em.o("__attribute__((aligned(32))) double spill_im[128];")
    em.b()
    em.o("__m256d x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o("__m256d x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    # v2: saved registers for fused last sub-FFT (NFUSE=2 on AVX2)
    em.o(f"__m256d s0_re,s0_im,s1_re,s1_im;")
    em.b()

    if itw_set:
        em.c(f"Hoisted internal W32 broadcasts")
        for (e,tN) in sorted(itw_set):
            label=wN_label(e,tN)
            em.o(f"const __m256d tw_{label}_re = _mm256_set1_pd({label}_re);")
            em.o(f"const __m256d tw_{label}_im = _mm256_set1_pd({label}_im);")
        em.b()

    em.o("for (size_t k = 0; k < K; k += 4) {")
    em.ind += 1

    xv8 = [f"x{i}" for i in range(8)]
    xv4 = [f"x{i}" for i in range(4)]
    fwd = (d == 'fwd')
    last_n2 = N2 - 1

    # PASS 1: 4 radix-8 sub-FFTs
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2*n1 + n2
            em.emit_load(f"x{n1}", n)
            if n > 0:
                em.o(f"{{ __m256d wr = _mm256_load_pd(&tw_re[{n-1}*K+k]);")
                em.o(f"  __m256d wi = _mm256_load_pd(&tw_im[{n-1}*K+k]);")
                em.o(f"  __m256d tr = x{n1}_re;")
                if fwd:
                    em.o(f"  x{n1}_re = {em.fms(f'x{n1}_re','wr',em.mul(f'x{n1}_im','wi'))};")
                    em.o(f"  x{n1}_im = {em.fma('tr','wi',em.mul(f'x{n1}_im','wr'))}; }}")
                else:
                    em.o(f"  x{n1}_re = {em.fma(f'x{n1}_re','wr',em.mul(f'x{n1}_im','wi'))};")
                    em.o(f"  x{n1}_im = {em.fms(f'x{n1}_im','wr',em.mul('tr','wi'))}; }}")
        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n2={n2}")
        em.b()

        # v2: last sub-FFT keeps first NFUSE results in s-regs
        if is_last:
            em.c(f"FUSED: save x0..x{NFUSE-1} to s-regs, spill x{NFUSE}..x{N1-1}")
            for k1 in range(NFUSE):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(NFUSE, N1):
                em.emit_spill(f"x{k1}", n2*N1+k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2*N1+k1)
        em.b()

    # PASS 2: 8 radix-4 combines
    em.c(f"PASS 2"); em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")

        # v2: first NFUSE columns get n2=last from saved regs
        if k1 < NFUSE:
            for n2 in range(last_n2):
                em.emit_reload(f"x{n2}", n2*N1+k1)
            em.o(f"x{last_n2}_re = s{k1}_re; x{last_n2}_im = s{k1}_im;")
        else:
            for n2 in range(N2):
                em.emit_reload(f"x{n2}", n2*N1+k1)
        em.b()

        if k1 > 0:
            for n2 in range(1,N2):
                e=(n2*k1)%N
                em.emit_twiddle(f"x{n2}",f"x{n2}",e,N,d)
            em.b()
        em.emit_radix4(xv4,d,f"radix-4 k1={k1}")
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1+N1*k2)
        em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}"); em.L.append("")
    return em.spill_c, em.reload_c


def main():
    itw_set = collect_internal_twiddles()
    em = E()

    em.L.append("/**")
    em.L.append(" * @file fft_radix32_avx2_tw.h")
    em.L.append(" * @brief DFT-32 AVX2 twiddled codelet v2 — flat twiddles, NFUSE=2")
    em.L.append(" *")
    em.L.append(" * Flat: tw_re[31*K], tw_im[31*K]")
    em.L.append(" * k-step = 4 (AVX2, 4-wide doubles)")
    em.L.append(f" * 8x4 decomposition, NFUSE={NFUSE}, U=1 only")
    em.L.append(" * Generated by gen_radix32_avx2_tw_v2.py")
    em.L.append(" */")
    em.L.append("")
    em.L.append("#ifndef FFT_RADIX32_AVX2_TW_H")
    em.L.append("#define FFT_RADIX32_AVX2_TW_H")
    em.L.append("")
    em.L.append("#include <immintrin.h>")
    em.L.append("")

    by_tN = {}
    for (e,tN) in sorted(itw_set): by_tN.setdefault(tN,[]).append(e)
    for tN in sorted(by_tN):
        g = f"FFT_W{tN}_TWIDDLES_DEFINED"
        em.L.append(f"#ifndef {g}"); em.L.append(f"#define {g}")
        for e in sorted(by_tN[tN]):
            wr,wi=wN(e,tN); l=wN_label(e,tN)
            em.L.append(f"static const double {l}_re = {wr:.20e};")
            em.L.append(f"static const double {l}_im = {wi:.20e};")
        em.L.append(f"#endif"); em.L.append("")

    em.L.append("#ifndef R32A_LD"); em.L.append("#define R32A_LD(p) _mm256_loadu_pd(p)"); em.L.append("#endif")
    em.L.append("#ifndef R32A_ST"); em.L.append("#define R32A_ST(p,v) _mm256_storeu_pd((p),(v))"); em.L.append("#endif")
    em.L.append("#define LD R32A_LD"); em.L.append("#define ST R32A_ST"); em.L.append("")

    em.L.append("/* === FLAT U=1 (AVX2, NFUSE=2) === */"); em.L.append("")
    ff = emit_flat_kernel(em, 'fwd', itw_set)
    fb = emit_flat_kernel(em, 'bwd', itw_set)

    em.L.append("#undef LD"); em.L.append("#undef ST"); em.L.append("")
    em.L.append("#endif /* FFT_RADIX32_AVX2_TW_H */")

    print("\n".join(em.L))

    v1_ops = 32+32  # old: 32sp + 32rl
    v2_ops = ff[0]+ff[1]
    print(f"\n=== AVX2 v2 STATS ===", file=sys.stderr)
    print(f"  NFUSE={NFUSE}: {ff[0]}sp+{ff[1]}rl={v2_ops} (was {v1_ops}), saved {v1_ops - v2_ops} mem ops/k-step", file=sys.stderr)
    print(f"  Peak reg pressure at pass-2 k1=0: 10+2*({NFUSE}-1) = {10+2*(NFUSE-1)}/16 ymm", file=sys.stderr)
    print(f"  Lines: {len(em.L)}", file=sys.stderr)

if __name__ == '__main__':
    main()
