#!/usr/bin/env python3
"""
gen_radix16_avx2_dif_tw.py — DFT-16 AVX2 DIF twiddled codelet

Same 4×4 as AVX-512 DIF but __m256d, k-step=4.
External twiddle on OUTPUT (after butterfly).
"""
import math, sys

N, N1, N2 = 16, 4, 4

def wN(e, tN):
    e=e%tN; a=2.0*math.pi*e/tN; return (math.cos(a), -math.sin(a))
def wN_label(e, tN): return f"W{tN}_{e%tN}"
def twiddle_is_trivial(e, tN):
    e=e%tN
    if e==0: return True,'one'
    if (8*e)%tN==0:
        o=(8*e)//tN
        t=['one','w8_1','neg_j','w8_3','neg_one','neg_w8_1','pos_j','neg_w8_3']
        return True,t[o%8]
    return False,'cmul'
def collect_internal_twiddles():
    tw=set()
    for n2 in range(1,N2):
        for k1 in range(1,N1):
            e=(n2*k1)%N; _,typ=twiddle_is_trivial(e,N)
            if typ=='cmul': tw.add((e,N))
    return tw

class E:
    def __init__(self):
        self.L=[]; self.ind=1; self.spill_c=0; self.reload_c=0
    def o(s,t=""): s.L.append("    "*s.ind+t)
    def c(s,t): s.o(f"/* {t} */")
    def b(s): s.L.append("")
    def emit_load(s,v,n):
        s.o(f"{v}_re = _mm256_load_pd(&in_re[{n}*K+k]);")
        s.o(f"{v}_im = _mm256_load_pd(&in_im[{n}*K+k]);")
    def emit_store(s,v,m):
        s.o(f"_mm256_store_pd(&out_re[{m}*K+k], {v}_re);")
        s.o(f"_mm256_store_pd(&out_im[{m}*K+k], {v}_im);")
    def emit_spill(s,v,slot):
        s.o(f"_mm256_store_pd(&spill_re[{slot}*4], {v}_re);")
        s.o(f"_mm256_store_pd(&spill_im[{slot}*4], {v}_im);")
        s.spill_c+=1
    def emit_reload(s,v,slot):
        s.o(f"{v}_re = _mm256_load_pd(&spill_re[{slot}*4]);")
        s.o(f"{v}_im = _mm256_load_pd(&spill_im[{slot}*4]);")
        s.reload_c+=1
    def emit_ext_tw(s,v,tw_idx,d):
        fwd=(d=='fwd')
        s.o(f"{{ __m256d wr=_mm256_load_pd(&tw_re[{tw_idx}*K+k]);")
        s.o(f"  __m256d wi=_mm256_load_pd(&tw_im[{tw_idx}*K+k]);")
        s.o(f"  __m256d tr={v}_re;")
        if fwd:
            s.o(f"  {v}_re=_mm256_fmsub_pd({v}_re,wr,_mm256_mul_pd({v}_im,wi));")
            s.o(f"  {v}_im=_mm256_fmadd_pd(tr,wi,_mm256_mul_pd({v}_im,wr)); }}")
        else:
            s.o(f"  {v}_re=_mm256_fmadd_pd({v}_re,wr,_mm256_mul_pd({v}_im,wi));")
            s.o(f"  {v}_im=_mm256_fmsub_pd({v}_im,wr,_mm256_mul_pd(tr,wi)); }}")
    def emit_radix4(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        a,b,c,dd=v[0],v[1],v[2],v[3]
        s.o(f"{{ __m256d t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r=_mm256_add_pd({a}_re,{c}_re); t0i=_mm256_add_pd({a}_im,{c}_im);")
        s.o(f"  t1r=_mm256_sub_pd({a}_re,{c}_re); t1i=_mm256_sub_pd({a}_im,{c}_im);")
        s.o(f"  t2r=_mm256_add_pd({b}_re,{dd}_re); t2i=_mm256_add_pd({b}_im,{dd}_im);")
        s.o(f"  t3r=_mm256_sub_pd({b}_re,{dd}_re); t3i=_mm256_sub_pd({b}_im,{dd}_im);")
        s.o(f"  {a}_re=_mm256_add_pd(t0r,t2r); {a}_im=_mm256_add_pd(t0i,t2i);")
        s.o(f"  {c}_re=_mm256_sub_pd(t0r,t2r); {c}_im=_mm256_sub_pd(t0i,t2i);")
        if fwd:
            s.o(f"  {b}_re=_mm256_add_pd(t1r,t3i); {b}_im=_mm256_sub_pd(t1i,t3r);")
            s.o(f"  {dd}_re=_mm256_sub_pd(t1r,t3i); {dd}_im=_mm256_add_pd(t1i,t3r);")
        else:
            s.o(f"  {b}_re=_mm256_sub_pd(t1r,t3i); {b}_im=_mm256_add_pd(t1i,t3r);")
            s.o(f"  {dd}_re=_mm256_add_pd(t1r,t3i); {dd}_im=_mm256_sub_pd(t1i,t3r);")
        s.o(f"}}")
    def emit_twiddle(s,dst,src,e,tN,d):
        _,typ=twiddle_is_trivial(e,tN); fwd=(d=='fwd')
        sf="sign_flip"; sq="sqrt2_inv"
        if typ=='one':
            if dst!=src: s.o(f"{dst}_re={src}_re; {dst}_im={src}_im;")
        elif typ=='neg_one':
            s.o(f"{dst}_re=_mm256_xor_pd({src}_re,{sf}); {dst}_im=_mm256_xor_pd({src}_im,{sf});")
        elif typ=='neg_j':
            if fwd: s.o(f"{{ __m256d t={src}_re; {dst}_re={src}_im; {dst}_im=_mm256_xor_pd(t,{sf}); }}")
            else:   s.o(f"{{ __m256d t={src}_re; {dst}_re=_mm256_xor_pd({src}_im,{sf}); {dst}_im=t; }}")
        elif typ=='pos_j':
            if fwd: s.o(f"{{ __m256d t={src}_re; {dst}_re=_mm256_xor_pd({src}_im,{sf}); {dst}_im=t; }}")
            else:   s.o(f"{{ __m256d t={src}_re; {dst}_re={src}_im; {dst}_im=_mm256_xor_pd(t,{sf}); }}")
        elif typ in ('w8_1','w8_3','neg_w8_1','neg_w8_3'):
            s.o(f"{{ __m256d tr={src}_re,ti={src}_im;")
            if typ=='w8_1':
                if fwd: s.o(f"  {dst}_re=_mm256_mul_pd(_mm256_add_pd(tr,ti),{sq}); {dst}_im=_mm256_mul_pd(_mm256_sub_pd(ti,tr),{sq}); }}")
                else:   s.o(f"  {dst}_re=_mm256_mul_pd(_mm256_sub_pd(tr,ti),{sq}); {dst}_im=_mm256_mul_pd(_mm256_add_pd(tr,ti),{sq}); }}")
            elif typ=='w8_3':
                if fwd: s.o(f"  {dst}_re=_mm256_mul_pd(_mm256_sub_pd(ti,tr),{sq}); {dst}_im=_mm256_xor_pd(_mm256_mul_pd(_mm256_add_pd(tr,ti),{sq}),{sf}); }}")
                else:   s.o(f"  {dst}_re=_mm256_xor_pd(_mm256_mul_pd(_mm256_add_pd(tr,ti),{sq}),{sf}); {dst}_im=_mm256_mul_pd(_mm256_sub_pd(tr,ti),{sq}); }}")
            elif typ=='neg_w8_1':
                if fwd: s.o(f"  {dst}_re=_mm256_xor_pd(_mm256_mul_pd(_mm256_add_pd(tr,ti),{sq}),{sf}); {dst}_im=_mm256_mul_pd(_mm256_sub_pd(tr,ti),{sq}); }}")
                else:   s.o(f"  {dst}_re=_mm256_mul_pd(_mm256_sub_pd(ti,tr),{sq}); {dst}_im=_mm256_xor_pd(_mm256_mul_pd(_mm256_add_pd(tr,ti),{sq}),{sf}); }}")
            elif typ=='neg_w8_3':
                if fwd: s.o(f"  {dst}_re=_mm256_mul_pd(_mm256_sub_pd(tr,ti),{sq}); {dst}_im=_mm256_mul_pd(_mm256_add_pd(tr,ti),{sq}); }}")
                else:   s.o(f"  {dst}_re=_mm256_mul_pd(_mm256_add_pd(tr,ti),{sq}); {dst}_im=_mm256_mul_pd(_mm256_sub_pd(ti,tr),{sq}); }}")
        else:
            label=wN_label(e,tN)
            s.o(f"{{ __m256d tr={src}_re;")
            if fwd:
                s.o(f"  {dst}_re=_mm256_fmsub_pd({src}_re,tw_{label}_re,_mm256_mul_pd({src}_im,tw_{label}_im));")
                s.o(f"  {dst}_im=_mm256_fmadd_pd(tr,tw_{label}_im,_mm256_mul_pd({src}_im,tw_{label}_re)); }}")
            else:
                s.o(f"  {dst}_re=_mm256_fmadd_pd({src}_re,tw_{label}_re,_mm256_mul_pd({src}_im,tw_{label}_im));")
                s.o(f"  {dst}_im=_mm256_fmsub_pd({src}_im,tw_{label}_re,_mm256_mul_pd(tr,tw_{label}_im)); }}")


def emit_dif_kernel(em, d, itw_set):
    em.L.append(f'__attribute__((target("avx2,fma")))')
    em.L.append(f"static void")
    em.L.append(f"radix16_tw_flat_dif_kernel_{d}_avx2(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind=1; em.spill_c=0; em.reload_c=0
    em.o("const __m256d sign_flip = _mm256_set1_pd(-0.0);")
    em.o("const __m256d sqrt2_inv = _mm256_set1_pd(0.70710678118654752440);")
    em.b()
    em.o("__attribute__((aligned(32))) double spill_re[64], spill_im[64];")
    em.o("__m256d x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.b()
    if itw_set:
        em.c(f"Hoisted internal W16 [{d}]")
        for (e,tN) in sorted(itw_set):
            label=wN_label(e,tN)
            em.o(f"const __m256d tw_{label}_re = _mm256_set1_pd({label}_re);")
            em.o(f"const __m256d tw_{label}_im = _mm256_set1_pd({label}_im);")
        em.b()
    xv4=[f"x{i}" for i in range(4)]
    em.o("for (size_t k = 0; k < K; k += 4) {")
    em.ind+=1
    # PASS 1: Load (NO ext tw) → radix-4 → spill
    for n2 in range(N2):
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            em.emit_load(f"x{n1}",N2*n1+n2)
        em.b()
        em.emit_radix4(xv4,d,f"radix-4 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}",n2*N1+k1)
        em.b()
    # PASS 2: Reload → W16 → radix-4 → ext tw → store
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}",n2*N1+k1)
        em.b()
        if k1 > 0:
            for n2 in range(1,N2):
                e=(n2*k1)%N
                em.emit_twiddle(f"x{n2}",f"x{n2}",e,N,d)
            em.b()
        em.emit_radix4(xv4,d,f"radix-4 k1={k1}")
        em.b()
        for k2 in range(N2):
            m=k1+N1*k2
            if m > 0:
                em.emit_ext_tw(f"x{k2}",m-1,d)
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}",k1+N1*k2)
        em.b()
    em.ind-=1
    em.o("}")
    em.L.append("}"); em.L.append("")
    return em.spill_c, em.reload_c

def main():
    itw_set=collect_internal_twiddles()
    em=E()
    em.L.append("/**")
    em.L.append(" * @file fft_radix16_avx2_dif_tw.h")
    em.L.append(" * @brief DFT-16 AVX2 DIF twiddled — twiddle AFTER butterfly")
    em.L.append(" * 4x4, k-step=4. Generated by gen_radix16_avx2_dif_tw.py")
    em.L.append(" */")
    em.L.append("")
    em.L.append("#ifndef FFT_RADIX16_AVX2_DIF_TW_H")
    em.L.append("#define FFT_RADIX16_AVX2_DIF_TW_H")
    em.L.append("")
    em.L.append("#include <immintrin.h>")
    em.L.append("")
    by_tN={}
    for (e,tN) in sorted(itw_set): by_tN.setdefault(tN,[]).append(e)
    for tN in sorted(by_tN):
        g=f"FFT_W{tN}_TWIDDLES_DEFINED"
        em.L.append(f"#ifndef {g}"); em.L.append(f"#define {g}")
        for e in sorted(by_tN[tN]):
            wr,wi=wN(e,tN); l=wN_label(e,tN)
            em.L.append(f"static const double {l}_re = {wr:.20e};")
            em.L.append(f"static const double {l}_im = {wi:.20e};")
        em.L.append(f"#endif"); em.L.append("")
    ff=emit_dif_kernel(em,'fwd',itw_set)
    fb=emit_dif_kernel(em,'bwd',itw_set)
    em.L.append("#endif /* FFT_RADIX16_AVX2_DIF_TW_H */")
    print("\n".join(em.L))
    print(f"Lines: {len(em.L)}", file=sys.stderr)

if __name__=='__main__':
    main()
