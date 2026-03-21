#!/usr/bin/env python3
"""
gen_r5_avx512.py — Definitive DFT-5 AVX-512 codelet generator
Dual: flat (K<=2048), log3 (K>2048). 32 ZMM: all upfront loading.
Usage: python gen_r5_avx512.py [outdir]
"""
import sys, os

T='__m512d'; W=8; P='_mm512'; THRESHOLD=2048
ATTR='__attribute__((target("avx512f,avx512dq,fma")))'

def ld(a):   return f'{P}_load_pd({a})'
def st(a,v): return f'{P}_store_pd({a},{v});'
def add(a,b):return f'{P}_add_pd({a},{b})'
def sub(a,b):return f'{P}_sub_pd({a},{b})'
def mul(a,b):return f'{P}_mul_pd({a},{b})'
def fma_(a,b,c):return f'{P}_fmadd_pd({a},{b},{c})'
def fms(a,b,c):return f'{P}_fmsub_pd({a},{b},{c})'
def fnma(a,b,c):return f'{P}_fnmadd_pd({a},{b},{c})'
def set1(v): return f'{P}_set1_pd({v})'
def cmul_re(ar,ai,br,bi): return fms(ar,br,mul(ai,bi))
def cmul_im(ar,ai,br,bi): return fma_(ar,bi,mul(ai,br))
def cmulc_re(ar,ai,br,bi): return fma_(ar,br,mul(ai,bi))
def cmulc_im(ar,ai,br,bi): return fms(ai,br,mul(ar,bi))

def il_load(L,v,idx):
    L.append(f'        {{ {T} _lo={ld(f"&in[2*({idx}*K+k)]")};')
    L.append(f'          {T} _hi={ld(f"&in[2*({idx}*K+k+4)]")};')
    L.append(f'          const __m512i _idx=_mm512_set_epi64(7,5,3,1,6,4,2,0);')
    L.append(f'          {v}r=_mm512_permutexvar_pd(_idx,_mm512_unpacklo_pd(_lo,_hi));')
    L.append(f'          {v}i=_mm512_permutexvar_pd(_idx,_mm512_unpackhi_pd(_lo,_hi)); }}')

def il_store(L,vr,vi,idx):
    L.append(f'        {{ const __m512i _idx=_mm512_set_epi64(7,3,6,2,5,1,4,0);')
    L.append(f'          {T} _rp=_mm512_permutexvar_pd(_idx,{vr});')
    L.append(f'          {T} _ip=_mm512_permutexvar_pd(_idx,{vi});')
    L.append(f'          {st(f"&out[2*({idx}*K+k)]",f"{P}_unpacklo_pd(_rp,_ip)")}')
    L.append(f'          {st(f"&out[2*({idx}*K+k+4)]",f"{P}_unpackhi_pd(_rp,_ip)")} }}')

def emit_constants(L):
    for n,v in [('cA','0.559016994374947424102293417182819058860154590'),('cB','0.951056516295153572116439333379382143405698634'),('cC','0.587785252292473129168705954639072768597652438'),('cD','0.250000000000000000000000000000000000000000000')]:
        L.append(f'    const {T} {n}={set1(v)};')

def emit_butterfly(L,fwd,sf):
    L.append(f'        const {T} s1r={add("x1r","x4r")},s1i={add("x1i","x4i")};')
    L.append(f'        const {T} s2r={add("x2r","x3r")},s2i={add("x2i","x3i")};')
    L.append(f'        const {T} d1r={sub("x1r","x4r")},d1i={sub("x1i","x4i")};')
    L.append(f'        const {T} d2r={sub("x2r","x3r")},d2i={sub("x2i","x3i")};')
    sf(L,0,add("x0r",add("s1r","s2r")),add("x0i",add("s1i","s2i")))
    L.append(f'        const {T} t0r={fnma("cD",add("s1r","s2r"),"x0r")},t0i={fnma("cD",add("s1i","s2i"),"x0i")};')
    L.append(f'        const {T} t1r={mul("cA",sub("s1r","s2r"))},t1i={mul("cA",sub("s1i","s2i"))};')
    L.append(f'        const {T} p1r={add("t0r","t1r")},p1i={add("t0i","t1i")};')
    L.append(f'        const {T} p2r={sub("t0r","t1r")},p2i={sub("t0i","t1i")};')
    L.append(f'        const {T} qar={fma_("cC","d2i",mul("cB","d1i"))},qai={fma_("cC","d2r",mul("cB","d1r"))};')
    L.append(f'        const {T} qbr={fnma("cC","d1i",mul("cB","d2i"))},qbi={fnma("cC","d1r",mul("cB","d2r"))};')
    if fwd:
        sf(L,1,add("p1r","qar"),sub("p1i","qai"));sf(L,4,sub("p1r","qar"),add("p1i","qai"))
        sf(L,2,sub("p2r","qbr"),add("p2i","qbi"));sf(L,3,add("p2r","qbr"),sub("p2i","qbi"))
    else:
        sf(L,1,sub("p1r","qar"),add("p1i","qai"));sf(L,4,add("p1r","qar"),sub("p1i","qai"))
        sf(L,2,add("p2r","qbr"),sub("p2i","qbi"));sf(L,3,sub("p2r","qbr"),add("p2i","qbi"))

def load_inputs(L,il):
    if il:
        L.append(f'        {T} x0r,x0i;')
        il_load(L,'x0',0)
    else:
        L.append(f'        const {T} x0r={ld("&in_re[0*K+k]")},x0i={ld("&in_im[0*K+k]")};')
    for n in range(1,5):
        if il:
            L.append(f'        {T} _r{n}r,_r{n}i;')
            il_load(L,f'_r{n}',n)
        else:
            L.append(f'        const {T} _r{n}r={ld(f"&in_re[{n}*K+k]")},_r{n}i={ld(f"&in_im[{n}*K+k]")};')

def emit_tw_flat(L):
    """AVX-512: load ALL 4 twiddles upfront (32 ZMM, no pressure)."""
    for n in range(1,5):
        L.append(f'        const {T} _w{n}r={ld(f"&tw_re[{n-1}*K+k]")},_w{n}i={ld(f"&tw_im[{n-1}*K+k]")};')

def emit_tw_log3(L):
    L.append(f'        const {T} _w1r={ld("&tw_re[0*K+k]")},_w1i={ld("&tw_im[0*K+k]")};')
    L.append(f'        const {T} _w2r={cmul_re("_w1r","_w1i","_w1r","_w1i")};')
    L.append(f'        const {T} _w2i={cmul_im("_w1r","_w1i","_w1r","_w1i")};')
    L.append(f'        const {T} _w3r={cmul_re("_w1r","_w1i","_w2r","_w2i")};')
    L.append(f'        const {T} _w3i={cmul_im("_w1r","_w1i","_w2r","_w2i")};')
    L.append(f'        const {T} _w4r={cmul_re("_w1r","_w1i","_w3r","_w3i")};')
    L.append(f'        const {T} _w4i={cmul_im("_w1r","_w1i","_w3r","_w3i")};')

def emit_tw_apply(L,fwd):
    cm_re,cm_im=(cmul_re,cmul_im) if fwd else (cmulc_re,cmulc_im)
    for n in range(1,5):
        L.append(f'        const {T} x{n}r={cm_re(f"_r{n}r",f"_r{n}i",f"_w{n}r",f"_w{n}i")};')
        L.append(f'        const {T} x{n}i={cm_im(f"_r{n}r",f"_r{n}i",f"_w{n}r",f"_w{n}i")};')

def gen_notw(d,il=False):
    fwd=d=='fwd';ilt='_il' if il else ''
    L=[ATTR,f'static inline void',f'radix5_notw_dit_kernel_{d}{ilt}_avx512(']
    if il:
        L.append(f'    const double*__restrict__ in,double*__restrict__ out,')
    else:
        L.append(f'    const double*__restrict__ in_re,const double*__restrict__ in_im,')
        L.append(f'    double*__restrict__ out_re,double*__restrict__ out_im,')
    L+=[f'    size_t K)',f'{{'];emit_constants(L);L.append(f'    for(size_t k=0;k<K;k+={W}){{')
    if il:
        for n in range(5):
            L.append(f'        {T} x{n}r,x{n}i;')
            il_load(L,f'x{n}',n)
    else:
        for n in range(5):
            L.append(f'        const {T} x{n}r={ld(f"&in_re[{n}*K+k]")},x{n}i={ld(f"&in_im[{n}*K+k]")};')
    sf_il=lambda L,i,vr,vi:il_store(L,vr,vi,i)
    sf_sp=lambda L,i,vr,vi:L.append(f'        {st(f"&out_re[{i}*K+k]",vr)} {st(f"&out_im[{i}*K+k]",vi)}')
    emit_butterfly(L,fwd,sf_il if il else sf_sp);L+=[f'    }}',f'}}'];return L

def gen_dit_tw_int(d,il,mode):
    fwd=d=='fwd';ilt='_il' if il else ''
    L=[ATTR,f'static inline void',f'radix5_tw_dit_kernel_{d}{ilt}_{mode}_avx512(']
    if il:L.append(f'    const double*__restrict__ in,double*__restrict__ out,')
    else:L.append(f'    const double*__restrict__ in_re,const double*__restrict__ in_im,');L.append(f'    double*__restrict__ out_re,double*__restrict__ out_im,')
    L+=[f'    const double*__restrict__ tw_re,const double*__restrict__ tw_im,',f'    size_t K)',f'{{']
    emit_constants(L);L.append(f'    for(size_t k=0;k<K;k+={W}){{')
    if mode=='flat':emit_tw_flat(L)
    else:emit_tw_log3(L)
    load_inputs(L,il);emit_tw_apply(L,fwd)
    sf_il=lambda L,i,vr,vi:il_store(L,vr,vi,i)
    sf_sp=lambda L,i,vr,vi:L.append(f'        {st(f"&out_re[{i}*K+k]",vr)} {st(f"&out_im[{i}*K+k]",vi)}')
    emit_butterfly(L,fwd,sf_il if il else sf_sp);L+=[f'    }}',f'}}'];return L

def gen_dit_tw(d,il=False):
    ilt='_il' if il else ''
    lf=gen_dit_tw_int(d,il,'flat');ll=gen_dit_tw_int(d,il,'log3')
    L=[ATTR,f'static inline void',f'radix5_tw_dit_kernel_{d}{ilt}_avx512(']
    if il:L.append(f'    const double*__restrict__ in,double*__restrict__ out,')
    else:L.append(f'    const double*__restrict__ in_re,const double*__restrict__ in_im,');L.append(f'    double*__restrict__ out_re,double*__restrict__ out_im,')
    L+=[f'    const double*__restrict__ tw_re,const double*__restrict__ tw_im,',f'    size_t K)',f'{{']
    L.append(f'    if(K<={THRESHOLD}){{')
    if il:L.append(f'        radix5_tw_dit_kernel_{d}{ilt}_flat_avx512(in,out,tw_re,tw_im,K);')
    else:L.append(f'        radix5_tw_dit_kernel_{d}{ilt}_flat_avx512(in_re,in_im,out_re,out_im,tw_re,tw_im,K);')
    L.append(f'    }}else{{')
    if il:L.append(f'        radix5_tw_dit_kernel_{d}{ilt}_log3_avx512(in,out,tw_re,tw_im,K);')
    else:L.append(f'        radix5_tw_dit_kernel_{d}{ilt}_log3_avx512(in_re,in_im,out_re,out_im,tw_re,tw_im,K);')
    L+=[f'    }}',f'}}'];return lf+['']+ll+['']+L

def gen_dif_tw_int(d,il,mode):
    fwd=d=='fwd';cm_re,cm_im=(cmul_re,cmul_im) if fwd else (cmulc_re,cmulc_im)
    ilt='_il' if il else ''
    L=[ATTR,f'static inline void',f'radix5_tw_dif_kernel_{d}{ilt}_{mode}_avx512(']
    if il:L.append(f'    const double*__restrict__ in,double*__restrict__ out,')
    else:L.append(f'    const double*__restrict__ in_re,const double*__restrict__ in_im,');L.append(f'    double*__restrict__ out_re,double*__restrict__ out_im,')
    L+=[f'    const double*__restrict__ tw_re,const double*__restrict__ tw_im,',f'    size_t K)',f'{{']
    emit_constants(L);L.append(f'    for(size_t k=0;k<K;k+={W}){{')
    if il:
        for n in range(5):
            L.append(f'        {T} x{n}r,x{n}i;')
            il_load(L,f'x{n}',n)
    else:
        for n in range(5):
            L.append(f'        const {T} x{n}r={ld(f"&in_re[{n}*K+k]")},x{n}i={ld(f"&in_im[{n}*K+k]")};')
    def stmp(L,i,vr,vi):L.append(f'        const {T} Y{i}r={vr},Y{i}i={vi};')
    emit_butterfly(L,fwd,stmp)
    if il:il_store(L,'Y0r','Y0i',0)
    else:L.append(f'        {st("&out_re[0*K+k]","Y0r")} {st("&out_im[0*K+k]","Y0i")}')
    if mode=='flat':
        for n in range(1,5):
            L.append(f'        {{ const {T} _wr={ld(f"&tw_re[{n-1}*K+k]")},_wi={ld(f"&tw_im[{n-1}*K+k]")};')
            L.append(f'          const {T} _or={cm_re(f"Y{n}r",f"Y{n}i","_wr","_wi")};')
            L.append(f'          const {T} _oi={cm_im(f"Y{n}r",f"Y{n}i","_wr","_wi")};')
            if il:il_store(L,'_or','_oi',n)
            else:L.append(f'          {st(f"&out_re[{n}*K+k]","_or")} {st(f"&out_im[{n}*K+k]","_oi")}')
            L.append(f'        }}')
    else:
        emit_tw_log3(L)
        for n in range(1,5):
            L.append(f'        {{ const {T} _or={cm_re(f"Y{n}r",f"Y{n}i",f"_w{n}r",f"_w{n}i")};')
            L.append(f'          const {T} _oi={cm_im(f"Y{n}r",f"Y{n}i",f"_w{n}r",f"_w{n}i")};')
            if il:il_store(L,'_or','_oi',n)
            else:L.append(f'          {st(f"&out_re[{n}*K+k]","_or")} {st(f"&out_im[{n}*K+k]","_oi")}')
            L.append(f'        }}')
    L+=[f'    }}',f'}}'];return L

def gen_dif_tw(d,il=False):
    ilt='_il' if il else ''
    lf=gen_dif_tw_int(d,il,'flat');ll=gen_dif_tw_int(d,il,'log3')
    L=[ATTR,f'static inline void',f'radix5_tw_dif_kernel_{d}{ilt}_avx512(']
    if il:L.append(f'    const double*__restrict__ in,double*__restrict__ out,')
    else:L.append(f'    const double*__restrict__ in_re,const double*__restrict__ in_im,');L.append(f'    double*__restrict__ out_re,double*__restrict__ out_im,')
    L+=[f'    const double*__restrict__ tw_re,const double*__restrict__ tw_im,',f'    size_t K)',f'{{']
    L.append(f'    if(K<={THRESHOLD}){{')
    if il:L.append(f'        radix5_tw_dif_kernel_{d}{ilt}_flat_avx512(in,out,tw_re,tw_im,K);')
    else:L.append(f'        radix5_tw_dif_kernel_{d}{ilt}_flat_avx512(in_re,in_im,out_re,out_im,tw_re,tw_im,K);')
    L.append(f'    }}else{{')
    if il:L.append(f'        radix5_tw_dif_kernel_{d}{ilt}_log3_avx512(in,out,tw_re,tw_im,K);')
    else:L.append(f'        radix5_tw_dif_kernel_{d}{ilt}_log3_avx512(in_re,in_im,out_re,out_im,tw_re,tw_im,K);')
    L+=[f'    }}',f'}}'];return lf+['']+ll+['']+L

HEADER="""/**
 * @file {fname}
 * @brief DFT-5 AVX-512 — {desc}
 * Dual: flat (K<={threshold}) / log3 (K>{threshold}).
 * 32 ZMM: all upfront loading. Generated by gen_r5_avx512.py
 */
#ifndef {guard}
#define {guard}
#if defined(__AVX512F__)||defined(__AVX512F)
#ifndef R5_FLAT_THRESHOLD_AVX512
#define R5_FLAT_THRESHOLD_AVX512 {threshold}
#endif
#include <stddef.h>
#include <immintrin.h>
"""
FOOTER="#endif /* __AVX512F__ */\n#endif /* {guard} */\n"

def write_file(path,fname,desc,guard,fns):
    lines=[HEADER.format(fname=fname,desc=desc,guard=guard,threshold=THRESHOLD)]
    for fl in fns:lines.append('\n'.join(fl));lines.append('')
    lines.append(FOOTER.format(guard=guard))
    with open(path,'w') as f:f.write('\n'.join(lines)+'\n')

def gen_all(outdir='.'):
    os.makedirs(outdir,exist_ok=True);files=[]
    for fname,desc,guard,fns in [
        ('fft_radix5_avx512.h','notw+DIT tw split','FFT_RADIX5_AVX512_H',
         [gen_notw('fwd'),gen_notw('bwd'),gen_dit_tw('fwd'),gen_dit_tw('bwd')]),
        ('fft_radix5_avx512_dif_tw.h','DIF tw split','FFT_RADIX5_AVX512_DIF_TW_H',
         [gen_dif_tw('fwd'),gen_dif_tw('bwd')]),
        ('fft_radix5_avx512_il.h','notw+DIT tw IL','FFT_RADIX5_AVX512_IL_H',
         [gen_notw('fwd',il=True),gen_notw('bwd',il=True),gen_dit_tw('fwd',il=True),gen_dit_tw('bwd',il=True)]),
        ('fft_radix5_avx512_il_dif_tw.h','DIF tw IL','FFT_RADIX5_AVX512_IL_DIF_TW_H',
         [gen_dif_tw('fwd',il=True),gen_dif_tw('bwd',il=True)]),
    ]:
        write_file(os.path.join(outdir,fname),fname,desc,guard,fns);files.append(fname)
    return files

if __name__=='__main__':
    outdir=sys.argv[1] if len(sys.argv)>1 else '.'
    for f in gen_all(outdir):print(f'  Generated: {f}')
