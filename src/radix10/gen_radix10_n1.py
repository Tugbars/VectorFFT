#!/usr/bin/env python3
"""gen_radix10_n1.py — DFT-10 N1 (notw) codelet, 5×2 CT, all ISAs"""
import math, sys
N,N1,N2 = 10,5,2

def wN(e,tN):
    e=e%tN;a=2.0*math.pi*e/tN;return(math.cos(a),-math.sin(a))
def wN_label(e,tN): return f"W{tN}_{e%tN}"

def collect_itw():
    tw=set()
    for n2 in range(1,N2):
        for k1 in range(1,N1):
            e=(n2*k1)%N
            if e!=0 and e!=5: tw.add((e,N))
    return tw

def emit_tw_consts(L,itw):
    by_tN={}
    for(e,tN)in sorted(itw):by_tN.setdefault(tN,[]).append(e)
    for tN in sorted(by_tN):
        g=f"FFT_W{tN}_TWIDDLES_DEFINED"
        L.append(f"#ifndef {g}");L.append(f"#define {g}")
        for e in sorted(by_tN[tN]):
            wr,wi=wN(e,tN);l=wN_label(e,tN)
            L.append(f"static const double {l}_re = {wr:.20e};")
            L.append(f"static const double {l}_im = {wi:.20e};")
        L.append(f"#endif");L.append("")

# ── Scalar ──

def gen_scalar_kernel(d):
    fwd=(d=='fwd');L=[]
    L.append(f"static inline void radix10_n1_dit_kernel_{d}_scalar(")
    L.append(f"    const double*__restrict__ in_re,const double*__restrict__ in_im,")
    L.append(f"    double*__restrict__ out_re,double*__restrict__ out_im,size_t K)")
    L.append(f"{{")
    L.append(f"    double sp_re[10],sp_im[10];")
    L.append(f"    double x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
    L.append(f"    for(size_t k=0;k<K;k++){{")
    K0="0.559016994374947424102293417182819058860154590"
    K1="0.951056516295153572116439333379382143405698634"
    K2="0.587785252292473129168705954639072768597652438"
    def r5(a,b,c,dd,xe):
        L.append(f"        {{ double s1r={b}_re+{xe}_re,s1i={b}_im+{xe}_im;")
        L.append(f"          double s2r={c}_re+{dd}_re,s2i={c}_im+{dd}_im;")
        L.append(f"          double d1r={b}_re-{xe}_re,d1i={b}_im-{xe}_im;")
        L.append(f"          double d2r={c}_re-{dd}_re,d2i={c}_im-{dd}_im;")
        L.append(f"          double t0r={a}_re-0.25*(s1r+s2r),t0i={a}_im-0.25*(s1i+s2i);")
        L.append(f"          double t1r={K0}*(s1r-s2r),t1i={K0}*(s1i-s2i);")
        L.append(f"          double p1r=t0r+t1r,p1i=t0i+t1i,p2r=t0r-t1r,p2i=t0i-t1i;")
        L.append(f"          double Ti={K1}*d1i+{K2}*d2i,Tu={K1}*d1r+{K2}*d2r;")
        L.append(f"          double Tk={K1}*d2i-{K2}*d1i,Tv={K1}*d2r-{K2}*d1r;")
        L.append(f"          {a}_re={a}_re+s1r+s2r;{a}_im={a}_im+s1i+s2i;")
        if fwd:
            L.append(f"          {b}_re=p1r+Ti;{b}_im=p1i-Tu;{xe}_re=p1r-Ti;{xe}_im=p1i+Tu;")
            L.append(f"          {c}_re=p2r-Tk;{c}_im=p2i+Tv;{dd}_re=p2r+Tk;{dd}_im=p2i-Tv; }}")
        else:
            L.append(f"          {b}_re=p1r-Ti;{b}_im=p1i+Tu;{xe}_re=p1r+Ti;{xe}_im=p1i-Tu;")
            L.append(f"          {c}_re=p2r+Tk;{c}_im=p2i-Tv;{dd}_re=p2r-Tk;{dd}_im=p2i+Tv; }}")
    for n2 in range(N2):
        for n1 in range(N1):
            n=N2*n1+n2
            L.append(f"        x{n1}_re=in_re[{n}*K+k];x{n1}_im=in_im[{n}*K+k];")
        r5("x0","x1","x2","x3","x4")
        for k1 in range(N1):
            L.append(f"        sp_re[{n2*N1+k1}]=x{k1}_re;sp_im[{n2*N1+k1}]=x{k1}_im;")
    for k1 in range(N1):
        for n2 in range(N2):
            L.append(f"        x{n2}_re=sp_re[{n2*N1+k1}];x{n2}_im=sp_im[{n2*N1+k1}];")
        if k1>0:
            e=(1*k1)%N
            if e==5:
                L.append(f"        x1_re=-x1_re;x1_im=-x1_im;")
            else:
                label=wN_label(e,N)
                if fwd:
                    L.append(f"        {{ double tr=x1_re;x1_re=tr*{label}_re-x1_im*{label}_im;x1_im=tr*{label}_im+x1_im*{label}_re; }}")
                else:
                    L.append(f"        {{ double tr=x1_re;x1_re=tr*{label}_re+x1_im*{label}_im;x1_im=x1_im*{label}_re-tr*{label}_im; }}")
        L.append(f"        {{ double tr=x0_re-x1_re,ti=x0_im-x1_im;")
        L.append(f"          x0_re=x0_re+x1_re;x0_im=x0_im+x1_im;x1_re=tr;x1_im=ti; }}")
        for k2 in range(N2):
            m=k1+N1*k2
            L.append(f"        out_re[{m}*K+k]=x{k2}_re;out_im[{m}*K+k]=x{k2}_im;")
    L.append(f"    }}")
    L.append(f"}}");L.append("")
    return L

# ── SIMD ──

def gen_simd_kernel(isa, d):
    fwd=(d=='fwd')
    if isa=='avx512':
        W=8;T='__m512d';P='_mm512';attr='__attribute__((target("avx512f,avx512dq,fma")))';align=64
    else:
        W=4;T='__m256d';P='_mm256';attr='__attribute__((target("avx2,fma")))';align=32
    def add(a,b): return f"{P}_add_pd({a},{b})"
    def sub(a,b): return f"{P}_sub_pd({a},{b})"
    def mul(a,b): return f"{P}_mul_pd({a},{b})"
    def fma_(a,b,c): return f"{P}_fmadd_pd({a},{b},{c})"
    def fms(a,b,c): return f"{P}_fmsub_pd({a},{b},{c})"
    def fnma(a,b,c): return f"{P}_fnmadd_pd({a},{b},{c})"
    def set1(v): return f"{P}_set1_pd({v})"
    def load(p): return f"{P}_load_pd({p})"
    def store(p,v): return f"{P}_store_pd({p},{v})"
    def xor_(a,b): return f"{P}_xor_pd({a},{b})"

    L=[]
    L.append(f"{attr}")
    L.append(f"static void radix10_n1_dit_kernel_{d}_{isa}(")
    L.append(f"    const double*__restrict__ in_re,const double*__restrict__ in_im,")
    L.append(f"    double*__restrict__ out_re,double*__restrict__ out_im,size_t K)")
    L.append(f"{{")
    L.append(f"    __attribute__((aligned({align}))) double sp_re[{10*W}],sp_im[{10*W}];")
    L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
    L.append(f"    const {T} sign_flip={set1('-0.0')};")
    L.append(f"    const {T} c0={set1('0.559016994374947424102293417182819058860154590')};")
    L.append(f"    const {T} c1={set1('0.951056516295153572116439333379382143405698634')};")
    L.append(f"    const {T} c2={set1('0.587785252292473129168705954639072768597652438')};")
    L.append(f"    const {T} q25={set1('0.25')};")
    # Hoisted W10 constants
    itw=collect_itw()
    for(e,tN) in sorted(itw):
        label=wN_label(e,tN)
        L.append(f"    const {T} tw_{label}_re={set1(f'{label}_re')};")
        L.append(f"    const {T} tw_{label}_im={set1(f'{label}_im')};")
    L.append(f"    for(size_t k=0;k<K;k+={W}){{")

    def ld(v,n):
        L.append(f"        {v}_re={load(f'&in_re[{n}*K+k]')};")
        L.append(f"        {v}_im={load(f'&in_im[{n}*K+k]')};")
    def st(v,m):
        L.append(f"        {store(f'&out_re[{m}*K+k]',f'{v}_re')};")
        L.append(f"        {store(f'&out_im[{m}*K+k]',f'{v}_im')};")
    def sp(v,slot):
        L.append(f"        {store(f'&sp_re[{slot}*{W}]',f'{v}_re')};")
        L.append(f"        {store(f'&sp_im[{slot}*{W}]',f'{v}_im')};")
    def rl(v,slot):
        L.append(f"        {v}_re={load(f'&sp_re[{slot}*{W}]')};")
        L.append(f"        {v}_im={load(f'&sp_im[{slot}*{W}]')};")

    def r5(a,b,c,dd,xe):
        L.append(f"        {{ {T} s1r={add(f'{b}_re',f'{xe}_re')},s1i={add(f'{b}_im',f'{xe}_im')};")
        L.append(f"          {T} s2r={add(f'{c}_re',f'{dd}_re')},s2i={add(f'{c}_im',f'{dd}_im')};")
        L.append(f"          {T} d1r={sub(f'{b}_re',f'{xe}_re')},d1i={sub(f'{b}_im',f'{xe}_im')};")
        L.append(f"          {T} d2r={sub(f'{c}_re',f'{dd}_re')},d2i={sub(f'{c}_im',f'{dd}_im')};")
        L.append(f"          {T} t0r={fnma('q25',add('s1r','s2r'),f'{a}_re')},t0i={fnma('q25',add('s1i','s2i'),f'{a}_im')};")
        L.append(f"          {T} t1r={mul('c0',sub('s1r','s2r'))},t1i={mul('c0',sub('s1i','s2i'))};")
        L.append(f"          {T} p1r={add('t0r','t1r')},p1i={add('t0i','t1i')};")
        L.append(f"          {T} p2r={sub('t0r','t1r')},p2i={sub('t0i','t1i')};")
        L.append(f"          {T} Ti={fma_('c1','d1i',mul('c2','d2i'))},Tu={fma_('c1','d1r',mul('c2','d2r'))};")
        L.append(f"          {T} Tk={fnma('c2','d1i',mul('c1','d2i'))},Tv={fnma('c2','d1r',mul('c1','d2r'))};")
        L.append(f"          {a}_re={add(f'{a}_re',add('s1r','s2r'))};{a}_im={add(f'{a}_im',add('s1i','s2i'))};")
        if fwd:
            L.append(f"          {b}_re={add('p1r','Ti')};{b}_im={sub('p1i','Tu')};{xe}_re={sub('p1r','Ti')};{xe}_im={add('p1i','Tu')};")
            L.append(f"          {c}_re={sub('p2r','Tk')};{c}_im={add('p2i','Tv')};{dd}_re={add('p2r','Tk')};{dd}_im={sub('p2i','Tv')}; }}")
        else:
            L.append(f"          {b}_re={sub('p1r','Ti')};{b}_im={add('p1i','Tu')};{xe}_re={add('p1r','Ti')};{xe}_im={sub('p1i','Tu')};")
            L.append(f"          {c}_re={add('p2r','Tk')};{c}_im={sub('p2i','Tv')};{dd}_re={sub('p2r','Tk')};{dd}_im={add('p2i','Tv')}; }}")

    # Pass 1
    for n2 in range(N2):
        for n1 in range(N1):
            ld(f"x{n1}",N2*n1+n2)
        r5("x0","x1","x2","x3","x4")
        for k1 in range(N1):
            sp(f"x{k1}",n2*N1+k1)

    # Pass 2
    for k1 in range(N1):
        for n2 in range(N2):
            rl(f"x{n2}",n2*N1+k1)
        if k1>0:
            e=(1*k1)%N
            if e==5:
                L.append(f"        x1_re={xor_('x1_re','sign_flip')};x1_im={xor_('x1_im','sign_flip')};")
            else:
                label=wN_label(e,N)
                L.append(f"        {{ {T} tr=x1_re;")
                if fwd:
                    L.append(f"          x1_re={fms('x1_re',f'tw_{label}_re',mul('x1_im',f'tw_{label}_im'))};")
                    L.append(f"          x1_im={fma_('tr',f'tw_{label}_im',mul('x1_im',f'tw_{label}_re'))}; }}")
                else:
                    L.append(f"          x1_re={fma_('x1_re',f'tw_{label}_re',mul('x1_im',f'tw_{label}_im'))};")
                    L.append(f"          x1_im={fms('x1_im',f'tw_{label}_re',mul('tr',f'tw_{label}_im'))}; }}")
        # radix-2
        L.append(f"        {{ {T} tr={sub('x0_re','x1_re')},ti={sub('x0_im','x1_im')};")
        L.append(f"          x0_re={add('x0_re','x1_re')};x0_im={add('x0_im','x1_im')};x1_re=tr;x1_im=ti; }}")
        for k2 in range(N2):
            st(f"x{k2}",k1+N1*k2)

    L.append(f"    }}")
    L.append(f"}}");L.append("")
    return L

def main():
    if len(sys.argv)<2:
        print("Usage: gen_radix10_n1.py scalar|avx512|avx2",file=sys.stderr);sys.exit(1)
    isa=sys.argv[1]; itw=collect_itw(); L=[]
    if isa=='scalar':
        L.append("/**\n * @file fft_radix10_scalar_n1.h\n * @brief Scalar N1 DFT-10 (5x2 CT)\n * Generated by gen_radix10_n1.py\n */")
        L.append("#ifndef FFT_RADIX10_SCALAR_N1_H\n#define FFT_RADIX10_SCALAR_N1_H\n#include <stddef.h>"); L.append("")
        emit_tw_consts(L,itw)
        L.extend(gen_scalar_kernel('fwd')); L.extend(gen_scalar_kernel('bwd'))
        L.append("#endif")
    elif isa in ('avx512','avx2'):
        ISA=isa.upper()
        L.append(f"/**\n * @file fft_radix10_{isa}_n1.h\n * @brief {ISA} N1 DFT-10 (5x2 CT)\n * Generated by gen_radix10_n1.py\n */")
        L.append(f"#ifndef FFT_RADIX10_{ISA}_N1_H\n#define FFT_RADIX10_{ISA}_N1_H\n#include <immintrin.h>"); L.append("")
        emit_tw_consts(L,itw)
        L.extend(gen_simd_kernel(isa,'fwd')); L.extend(gen_simd_kernel(isa,'bwd'))
        L.append(f"#endif")
    print("\n".join(L))
    print(f"Lines: {len(L)}",file=sys.stderr)

if __name__=='__main__':
    main()
