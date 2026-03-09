#!/usr/bin/env python3
"""
gen_r5_log3.py — DFT-5 codelets with 1-base twiddle derivation.
Supports split re/im and interleaved (IL) layout.
Usage: gen_r5_log3.py <scalar|avx2|avx512> <dit|dif> [il]
"""
import sys

def isa_config(isa):
    if isa == 'scalar':
        return dict(T='double', W=1, attr='',
                    ld=lambda p: f'*({p})', st=lambda p,v: f'*({p}) = {v};')
    elif isa == 'avx2':
        return dict(T='__m256d', W=4,
                    attr='__attribute__((target("avx2,fma")))',
                    ld=lambda p: f'_mm256_load_pd({p})',
                    st=lambda p,v: f'_mm256_store_pd({p},{v});')
    else:
        return dict(T='__m512d', W=8,
                    attr='__attribute__((target("avx512f,avx512dq,fma")))',
                    ld=lambda p: f'_mm512_load_pd({p})',
                    st=lambda p,v: f'_mm512_store_pd({p},{v});')

def ops(isa):
    if isa == 'scalar':
        return dict(add=lambda a,b: f'({a}+{b})', sub=lambda a,b: f'({a}-{b})',
            mul=lambda a,b: f'({a}*{b})', fma=lambda a,b,c: f'({a}*{b}+{c})',
            fms=lambda a,b,c: f'({a}*{b}-{c})', fnma=lambda a,b,c: f'({c}-{a}*{b})',
            set1=lambda v: v)
    P = '_mm256' if isa == 'avx2' else '_mm512'
    return dict(add=lambda a,b: f'{P}_add_pd({a},{b})', sub=lambda a,b: f'{P}_sub_pd({a},{b})',
        mul=lambda a,b: f'{P}_mul_pd({a},{b})', fma=lambda a,b,c: f'{P}_fmadd_pd({a},{b},{c})',
        fms=lambda a,b,c: f'{P}_fmsub_pd({a},{b},{c})', fnma=lambda a,b,c: f'{P}_fnmadd_pd({a},{b},{c})',
        set1=lambda v: f'{P}_set1_pd({v})')

def cmul_re(o, ar,ai,br,bi): return o['fms'](ar,br, o['mul'](ai,bi))
def cmul_im(o, ar,ai,br,bi): return o['fma'](ar,bi, o['mul'](ai,br))
def cmulc_re(o, ar,ai,br,bi): return o['fma'](ar,br, o['mul'](ai,bi))
def cmulc_im(o, ar,ai,br,bi): return o['fnma'](ar,bi, o['mul'](ai,br))

def il_load(L, cfg, isa, v, idx):
    T = cfg['T']
    if isa == 'scalar':
        L.append(f'        const {T} {v}r = in[2*({idx}*K+k)], {v}i = in[2*({idx}*K+k)+1];')
    elif isa == 'avx2':
        L.append(f'        {{ __m256d _lo = _mm256_load_pd(&in[2*({idx}*K+k)]);')
        L.append(f'          __m256d _hi = _mm256_load_pd(&in[2*({idx}*K+k+2)]);')
        L.append(f'          {v}r = _mm256_permute4x64_pd(_mm256_shuffle_pd(_lo,_hi,0x0), 0xD8);')
        L.append(f'          {v}i = _mm256_permute4x64_pd(_mm256_shuffle_pd(_lo,_hi,0xF), 0xD8); }}')
    else:
        L.append(f'        {{ __m512d _lo = _mm512_load_pd(&in[2*({idx}*K+k)]);')
        L.append(f'          __m512d _hi = _mm512_load_pd(&in[2*({idx}*K+k+4)]);')
        L.append(f'          {v}r = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpacklo_pd(_lo,_hi));')
        L.append(f'          {v}i = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpackhi_pd(_lo,_hi)); }}')

def il_store(L, cfg, isa, vr, vi, idx):
    if isa == 'scalar':
        L.append(f'        out[2*({idx}*K+k)] = {vr}; out[2*({idx}*K+k)+1] = {vi};')
    elif isa == 'avx2':
        L.append(f'        {{ __m256d _rp = _mm256_permute4x64_pd({vr}, 0xD8);')
        L.append(f'          __m256d _ip = _mm256_permute4x64_pd({vi}, 0xD8);')
        L.append(f'          _mm256_store_pd(&out[2*({idx}*K+k)], _mm256_shuffle_pd(_rp,_ip,0x0));')
        L.append(f'          _mm256_store_pd(&out[2*({idx}*K+k+2)], _mm256_shuffle_pd(_rp,_ip,0xF)); }}')
    else:
        L.append(f'        {{ __m512d _rp = _mm512_permutexvar_pd(_mm512_set_epi64(7,3,6,2,5,1,4,0), {vr});')
        L.append(f'          __m512d _ip = _mm512_permutexvar_pd(_mm512_set_epi64(7,3,6,2,5,1,4,0), {vi});')
        L.append(f'          _mm512_store_pd(&out[2*({idx}*K+k)], _mm512_unpacklo_pd(_rp,_ip));')
        L.append(f'          _mm512_store_pd(&out[2*({idx}*K+k+4)], _mm512_unpackhi_pd(_rp,_ip)); }}')

def gen_derive(L, o, cfg):
    T = cfg['T']
    L.append(f'        const {T} w1r = {cfg["ld"]("&tw_re[0*K+k]")}, w1i = {cfg["ld"]("&tw_im[0*K+k]")};')
    L.append(f'        const {T} w2r = {cmul_re(o,"w1r","w1i","w1r","w1i")};')
    L.append(f'        const {T} w2i = {cmul_im(o,"w1r","w1i","w1r","w1i")};')
    L.append(f'        const {T} w3r = {cmul_re(o,"w1r","w1i","w2r","w2i")};')
    L.append(f'        const {T} w3i = {cmul_im(o,"w1r","w1i","w2r","w2i")};')
    L.append(f'        const {T} w4r = {cmul_re(o,"w1r","w1i","w3r","w3i")};')
    L.append(f'        const {T} w4i = {cmul_im(o,"w1r","w1i","w3r","w3i")};')

def emit_butterfly(L, o, cfg, fwd, prefix='x', store=True, il=False, isa='scalar'):
    T = cfg['T']
    add, sub, mul, fma, fnma = o['add'], o['sub'], o['mul'], o['fma'], o['fnma']
    p = prefix
    def ST(idx, vr, vi):
        if il: il_store(L, cfg, isa, vr, vi, idx)
        else: L.append(f'        {cfg["st"](f"&out_re[{idx}*K+k]", vr)} {cfg["st"](f"&out_im[{idx}*K+k]", vi)}')

    L.append(f'        const {T} s1r = {add(f"{p}1r",f"{p}4r")}, s1i = {add(f"{p}1i",f"{p}4i")};')
    L.append(f'        const {T} s2r = {add(f"{p}2r",f"{p}3r")}, s2i = {add(f"{p}2i",f"{p}3i")};')
    L.append(f'        const {T} d1r = {sub(f"{p}1r",f"{p}4r")}, d1i = {sub(f"{p}1i",f"{p}4i")};')
    L.append(f'        const {T} d2r = {sub(f"{p}2r",f"{p}3r")}, d2i = {sub(f"{p}2i",f"{p}3i")};')
    if store: ST(0, add(f"{p}0r",add("s1r","s2r")), add(f"{p}0i",add("s1i","s2i")))
    else: L.append(f'        const {T} Y0r = {add(f"{p}0r",add("s1r","s2r"))}, Y0i = {add(f"{p}0i",add("s1i","s2i"))};')
    L.append(f'        const {T} t0r = {fnma("cK3",add("s1r","s2r"),f"{p}0r")}, t0i = {fnma("cK3",add("s1i","s2i"),f"{p}0i")};')
    L.append(f'        const {T} t1r = {mul("cK0",sub("s1r","s2r"))}, t1i = {mul("cK0",sub("s1i","s2i"))};')
    L.append(f'        const {T} p1r = {add("t0r","t1r")}, p1i = {add("t0i","t1i")};')
    L.append(f'        const {T} p2r = {sub("t0r","t1r")}, p2i = {sub("t0i","t1i")};')
    L.append(f'        const {T} Ti = {fma("cK2","d2i",mul("cK1","d1i"))}, Tu = {fma("cK2","d2r",mul("cK1","d1r"))};')
    L.append(f'        const {T} Tk = {fnma("cK2","d1i",mul("cK1","d2i"))}, Tv = {fnma("cK2","d1r",mul("cK1","d2r"))};')
    if fwd: pairs=[('1','+Ti','-Tu'),('4','-Ti','+Tu'),('2','-Tk','+Tv'),('3','+Tk','-Tv')]
    else:   pairs=[('1','-Ti','+Tu'),('4','+Ti','-Tu'),('2','+Tk','-Tv'),('3','-Tk','+Tv')]
    for idx,re_op,im_op in pairs:
        rs,rv = re_op[0],re_op[1:]; is_,iv = im_op[0],im_op[1:]
        rb = 'p1r' if idx in ('1','4') else 'p2r'; ib = 'p1i' if idx in ('1','4') else 'p2i'
        re = add(rb,rv) if rs=='+' else sub(rb,rv); ie = add(ib,iv) if is_=='+' else sub(ib,iv)
        if store: ST(idx, re, ie)
        else: L.append(f'        const {T} Y{idx}r = {re}, Y{idx}i = {ie};')

def emit_constants(L, o, cfg):
    T = cfg['T']
    L.append(f'    const {T} cK0 = {o["set1"]("0.559016994374947424102293417182819058860154590")};')
    L.append(f'    const {T} cK1 = {o["set1"]("0.951056516295153572116439333379382143405698634")};')
    L.append(f'    const {T} cK2 = {o["set1"]("0.587785252292473129168705954639072768597652438")};')
    L.append(f'    const {T} cK3 = {o["set1"]("0.250000000000000000000000000000000000000000000")};')

def gen_notw(isa, direction, il=False):
    cfg, o = isa_config(isa), ops(isa)
    T, W, fwd = cfg['T'], cfg['W'], direction == 'fwd'
    il_tag = '_il' if il else ''
    L = []
    if cfg['attr']: L.append(cfg['attr'])
    L.append(f'static inline void')
    L.append(f'radix5_notw_dit_kernel_{direction}{il_tag}_{isa}(')
    if il: L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L.append(f'    size_t K)'); L.append(f'{{')
    emit_constants(L, o, cfg)
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    if il:
        for n in range(5):
            L.append(f'        {T} x{n}r, x{n}i;')
            il_load(L, cfg, isa, f'x{n}', n)
    else:
        for n in range(5):
            L.append(f'        const {T} x{n}r = {cfg["ld"](f"&in_re[{n}*K+k]")}, x{n}i = {cfg["ld"](f"&in_im[{n}*K+k]")};')
    emit_butterfly(L, o, cfg, fwd, 'x', True, il=il, isa=isa)
    L.append(f'    }}'); L.append(f'}}'); return L

def gen_dit_tw(isa, direction, il=False):
    cfg, o = isa_config(isa), ops(isa)
    T, W, fwd = cfg['T'], cfg['W'], direction == 'fwd'
    cm = (cmul_re,cmul_im) if fwd else (cmulc_re,cmulc_im)
    il_tag = '_il' if il else ''
    L = []
    if cfg['attr']: L.append(cfg['attr'])
    L.append(f'static inline void')
    L.append(f'radix5_tw_dit_kernel_{direction}{il_tag}_{isa}(')
    if il: L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    L.append(f'    size_t K)'); L.append(f'{{')
    emit_constants(L, o, cfg)
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    gen_derive(L, o, cfg)
    if il:
        L.append(f'        {T} x0r, x0i;'); il_load(L, cfg, isa, 'x0', 0)
        for n in range(1, 5):
            L.append(f'        {T} r{n}r, r{n}i;'); il_load(L, cfg, isa, f'r{n}', n)
            L.append(f'        const {T} x{n}r = {cm[0](o,f"r{n}r",f"r{n}i",f"w{n}r",f"w{n}i")}, x{n}i = {cm[1](o,f"r{n}r",f"r{n}i",f"w{n}r",f"w{n}i")};')
    else:
        L.append(f'        const {T} x0r = {cfg["ld"]("&in_re[0*K+k]")}, x0i = {cfg["ld"]("&in_im[0*K+k]")};')
        for n in range(1, 5):
            L.append(f'        const {T} r{n}r = {cfg["ld"](f"&in_re[{n}*K+k]")}, r{n}i = {cfg["ld"](f"&in_im[{n}*K+k]")};')
            L.append(f'        const {T} x{n}r = {cm[0](o,f"r{n}r",f"r{n}i",f"w{n}r",f"w{n}i")}, x{n}i = {cm[1](o,f"r{n}r",f"r{n}i",f"w{n}r",f"w{n}i")};')
    emit_butterfly(L, o, cfg, fwd, 'x', True, il=il, isa=isa)
    L.append(f'    }}'); L.append(f'}}'); return L

def gen_dif_tw(isa, direction, il=False):
    cfg, o = isa_config(isa), ops(isa)
    T, W, fwd = cfg['T'], cfg['W'], direction == 'fwd'
    cm = (cmul_re,cmul_im) if fwd else (cmulc_re,cmulc_im)
    il_tag = '_il' if il else ''
    L = []
    if cfg['attr']: L.append(cfg['attr'])
    L.append(f'static inline void')
    L.append(f'radix5_tw_dif_kernel_{direction}{il_tag}_{isa}(')
    if il: L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    L.append(f'    size_t K)'); L.append(f'{{')
    emit_constants(L, o, cfg)
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    gen_derive(L, o, cfg)
    if il:
        for n in range(5):
            L.append(f'        {T} x{n}r, x{n}i;'); il_load(L, cfg, isa, f'x{n}', n)
    else:
        for n in range(5):
            L.append(f'        const {T} x{n}r = {cfg["ld"](f"&in_re[{n}*K+k]")}, x{n}i = {cfg["ld"](f"&in_im[{n}*K+k]")};')
    emit_butterfly(L, o, cfg, fwd, 'x', False)
    if il:
        il_store(L, cfg, isa, 'Y0r', 'Y0i', 0)
        for n in range(1, 5):
            L.append(f'        {{ const {T} tr = {cm[0](o,f"Y{n}r",f"Y{n}i",f"w{n}r",f"w{n}i")};')
            L.append(f'          const {T} ti = {cm[1](o,f"Y{n}r",f"Y{n}i",f"w{n}r",f"w{n}i")};')
            il_store(L, cfg, isa, 'tr', 'ti', n); L.append(f'        }}')
    else:
        L.append(f'        {cfg["st"]("&out_re[0*K+k]","Y0r")} {cfg["st"]("&out_im[0*K+k]","Y0i")}')
        for n in range(1, 5):
            L.append(f'        {{ const {T} tr = {cm[0](o,f"Y{n}r",f"Y{n}i",f"w{n}r",f"w{n}i")};')
            L.append(f'          const {T} ti = {cm[1](o,f"Y{n}r",f"Y{n}i",f"w{n}r",f"w{n}i")};')
            L.append(f'          {cfg["st"](f"&out_re[{n}*K+k]","tr")} {cfg["st"](f"&out_im[{n}*K+k]","ti")} }}')
    L.append(f'    }}'); L.append(f'}}'); return L

def gen_dit_file(isa, il=False):
    ISA = isa.upper(); il_tag = '_il' if il else ''
    guard = f'FFT_RADIX5_{ISA}{il_tag.upper()}_H'
    layout = 'interleaved' if il else 'split re/im'
    L = [f'/**', f' * @file fft_radix5_{isa}{il_tag}.h',
         f' * @brief DFT-5 {ISA} — {layout} + log3', f' * Generated by gen_r5_log3.py',
         f' */', f'', f'#ifndef {guard}', f'#define {guard}', f'#include <stddef.h>']
    if isa != 'scalar': L.append(f'#include <immintrin.h>')
    L.append('')
    for d in ('fwd','bwd'): L.extend(gen_notw(isa,d,il=il)); L.append('')
    for d in ('fwd','bwd'): L.extend(gen_dit_tw(isa,d,il=il)); L.append('')
    L.append(f'#endif /* {guard} */'); return L

def gen_dif_file(isa, il=False):
    ISA = isa.upper(); il_tag = '_il' if il else ''
    guard = f'FFT_RADIX5_{ISA}{il_tag.upper()}_DIF_TW_H'
    layout = 'interleaved' if il else 'split re/im'
    L = [f'/**', f' * @file fft_radix5_{isa}{il_tag}_dif_tw.h',
         f' * @brief DFT-5 {ISA} DIF — {layout} + log3', f' * Generated by gen_r5_log3.py',
         f' */', f'', f'#ifndef {guard}', f'#define {guard}', f'#include <stddef.h>']
    if isa != 'scalar': L.append(f'#include <immintrin.h>')
    L.append('')
    for d in ('fwd','bwd'): L.extend(gen_dif_tw(isa,d,il=il)); L.append('')
    L.append(f'#endif /* {guard} */'); return L

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: gen_r5_log3.py <scalar|avx2|avx512> <dit|dif> [il]", file=sys.stderr)
        sys.exit(1)
    isa, mode = sys.argv[1], sys.argv[2]
    il = len(sys.argv) > 3 and sys.argv[3] == 'il'
    lines = gen_dit_file(isa, il=il) if mode == 'dit' else gen_dif_file(isa, il=il)
    print('\n'.join(lines))
