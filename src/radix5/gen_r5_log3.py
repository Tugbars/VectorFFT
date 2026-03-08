#!/usr/bin/env python3
"""
gen_r5_log3.py — DFT-5 codelets with 1-base twiddle derivation.
Load W^1 only, derive W^2=W^1², W^3=W^1·W^2, W^4=W^1·W^3.
Saves 6 strided loads per k-step, costs 3 cmuls.
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
        return dict(
            add=lambda a,b: f'({a}+{b})', sub=lambda a,b: f'({a}-{b})',
            mul=lambda a,b: f'({a}*{b})',
            fma=lambda a,b,c: f'({a}*{b}+{c})', fms=lambda a,b,c: f'({a}*{b}-{c})',
            fnma=lambda a,b,c: f'({c}-{a}*{b})',
            set1=lambda v: v)
    P = '_mm256' if isa == 'avx2' else '_mm512'
    return dict(
        add=lambda a,b: f'{P}_add_pd({a},{b})',
        sub=lambda a,b: f'{P}_sub_pd({a},{b})',
        mul=lambda a,b: f'{P}_mul_pd({a},{b})',
        fma=lambda a,b,c: f'{P}_fmadd_pd({a},{b},{c})',
        fms=lambda a,b,c: f'{P}_fmsub_pd({a},{b},{c})',
        fnma=lambda a,b,c: f'{P}_fnmadd_pd({a},{b},{c})',
        set1=lambda v: f'{P}_set1_pd({v})')

def cmul_re(o, ar,ai,br,bi): return o['fms'](ar,br, o['mul'](ai,bi))
def cmul_im(o, ar,ai,br,bi): return o['fma'](ar,bi, o['mul'](ai,br))
def cmulc_re(o, ar,ai,br,bi): return o['fma'](ar,br, o['mul'](ai,bi))
def cmulc_im(o, ar,ai,br,bi): return o['fnma'](ar,bi, o['mul'](ai,br))

def gen_derive(L, o, cfg):
    T = cfg['T']
    L.append(f'        /* Load 1 base: W^1 */')
    L.append(f'        const {T} w1r = {cfg["ld"]("&tw_re[0*K+k]")}, w1i = {cfg["ld"]("&tw_im[0*K+k]")};')
    L.append(f'        /* W^2 = W^1 × W^1 */')
    L.append(f'        const {T} w2r = {cmul_re(o,"w1r","w1i","w1r","w1i")};')
    L.append(f'        const {T} w2i = {cmul_im(o,"w1r","w1i","w1r","w1i")};')
    L.append(f'        /* W^3 = W^1 × W^2 */')
    L.append(f'        const {T} w3r = {cmul_re(o,"w1r","w1i","w2r","w2i")};')
    L.append(f'        const {T} w3i = {cmul_im(o,"w1r","w1i","w2r","w2i")};')
    L.append(f'        /* W^4 = W^1 × W^3 */')
    L.append(f'        const {T} w4r = {cmul_re(o,"w1r","w1i","w3r","w3i")};')
    L.append(f'        const {T} w4i = {cmul_im(o,"w1r","w1i","w3r","w3i")};')

def emit_butterfly(L, o, cfg, fwd, prefix='x', store=True):
    """DFT-5 butterfly on {prefix}0..{prefix}4.
    If store=True, emit stores to out. If store=False, emit into Y0..Y4 vars."""
    T = cfg['T']
    add, sub, mul = o['add'], o['sub'], o['mul']
    fma, fnma = o['fma'], o['fnma']
    p = prefix

    L.append(f'        const {T} s1r = {add(f"{p}1r",f"{p}4r")}, s1i = {add(f"{p}1i",f"{p}4i")};')
    L.append(f'        const {T} s2r = {add(f"{p}2r",f"{p}3r")}, s2i = {add(f"{p}2i",f"{p}3i")};')
    L.append(f'        const {T} d1r = {sub(f"{p}1r",f"{p}4r")}, d1i = {sub(f"{p}1i",f"{p}4i")};')
    L.append(f'        const {T} d2r = {sub(f"{p}2r",f"{p}3r")}, d2i = {sub(f"{p}2i",f"{p}3i")};')

    if store:
        L.append(f'        {cfg["st"](f"&out_re[0*K+k]", add(f"{p}0r", add("s1r","s2r")))}')
        L.append(f'        {cfg["st"](f"&out_im[0*K+k]", add(f"{p}0i", add("s1i","s2i")))}')
    else:
        L.append(f'        const {T} Y0r = {add(f"{p}0r", add("s1r","s2r"))}, Y0i = {add(f"{p}0i", add("s1i","s2i"))};')

    L.append(f'        const {T} t0r = {fnma("cK3", add("s1r","s2r"), f"{p}0r")}, t0i = {fnma("cK3", add("s1i","s2i"), f"{p}0i")};')
    L.append(f'        const {T} t1r = {mul("cK0", sub("s1r","s2r"))}, t1i = {mul("cK0", sub("s1i","s2i"))};')
    L.append(f'        const {T} p1r = {add("t0r","t1r")}, p1i = {add("t0i","t1i")};')
    L.append(f'        const {T} p2r = {sub("t0r","t1r")}, p2i = {sub("t0i","t1i")};')
    L.append(f'        const {T} Ti = {fma("cK2","d2i", mul("cK1","d1i"))}, Tu = {fma("cK2","d2r", mul("cK1","d1r"))};')
    L.append(f'        const {T} Tk = {fnma("cK2","d1i", mul("cK1","d2i"))}, Tv = {fnma("cK2","d1r", mul("cK1","d2r"))};')

    if fwd:
        pairs = [('1','+Ti','-Tu'), ('4','-Ti','+Tu'), ('2','-Tk','+Tv'), ('3','+Tk','-Tv')]
    else:
        pairs = [('1','-Ti','+Tu'), ('4','+Ti','-Tu'), ('2','+Tk','-Tv'), ('3','-Tk','+Tv')]

    for idx, re_op, im_op in pairs:
        re_sign, re_var = re_op[0], re_op[1:]
        im_sign, im_var = im_op[0], im_op[1:]
        re_base = 'p1r' if idx in ('1','4') else 'p2r'
        im_base = 'p1i' if idx in ('1','4') else 'p2i'
        if re_sign == '+':
            re_expr = add(re_base, re_var)
        else:
            re_expr = sub(re_base, re_var)
        if im_sign == '+':
            im_expr = add(im_base, im_var)
        else:
            im_expr = sub(im_base, im_var)
        if store:
            L.append(f'        {cfg["st"](f"&out_re[{idx}*K+k]", re_expr)} {cfg["st"](f"&out_im[{idx}*K+k]", im_expr)}')
        else:
            L.append(f'        const {T} Y{idx}r = {re_expr}, Y{idx}i = {im_expr};')

def emit_constants(L, o, cfg):
    T = cfg['T']
    L.append(f'    const {T} cK0 = {o["set1"]("0.559016994374947424102293417182819058860154590")};')
    L.append(f'    const {T} cK1 = {o["set1"]("0.951056516295153572116439333379382143405698634")};')
    L.append(f'    const {T} cK2 = {o["set1"]("0.587785252292473129168705954639072768597652438")};')
    L.append(f'    const {T} cK3 = {o["set1"]("0.250000000000000000000000000000000000000000000")};')

def gen_notw(isa, direction):
    cfg, o = isa_config(isa), ops(isa)
    T, W, fwd = cfg['T'], cfg['W'], direction == 'fwd'
    L = []
    if cfg['attr']: L.append(cfg['attr'])
    L.append(f'static inline void')
    L.append(f'radix5_notw_dit_kernel_{direction}_{isa}(')
    L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L.append(f'    size_t K)')
    L.append(f'{{')
    emit_constants(L, o, cfg)
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    for n in range(5):
        L.append(f'        const {T} x{n}r = {cfg["ld"](f"&in_re[{n}*K+k]")}, x{n}i = {cfg["ld"](f"&in_im[{n}*K+k]")};')
    emit_butterfly(L, o, cfg, fwd, 'x', True)
    L.append(f'    }}')
    L.append(f'}}')
    return L

def gen_dit_tw(isa, direction):
    cfg, o = isa_config(isa), ops(isa)
    T, W, fwd = cfg['T'], cfg['W'], direction == 'fwd'
    cm = (cmul_re, cmul_im) if fwd else (cmulc_re, cmulc_im)
    L = []
    if cfg['attr']: L.append(cfg['attr'])
    L.append(f'static inline void')
    L.append(f'radix5_tw_dit_kernel_{direction}_{isa}(')
    L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    L.append(f'    size_t K)')
    L.append(f'{{')
    emit_constants(L, o, cfg)
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    gen_derive(L, o, cfg)
    L.append(f'        const {T} x0r = {cfg["ld"]("&in_re[0*K+k]")}, x0i = {cfg["ld"]("&in_im[0*K+k]")};')
    for n in range(1, 5):
        L.append(f'        const {T} r{n}r = {cfg["ld"](f"&in_re[{n}*K+k]")}, r{n}i = {cfg["ld"](f"&in_im[{n}*K+k]")};')
        L.append(f'        const {T} x{n}r = {cm[0](o,f"r{n}r",f"r{n}i",f"w{n}r",f"w{n}i")}, x{n}i = {cm[1](o,f"r{n}r",f"r{n}i",f"w{n}r",f"w{n}i")};')
    emit_butterfly(L, o, cfg, fwd, 'x', True)
    L.append(f'    }}')
    L.append(f'}}')
    return L

def gen_dif_tw(isa, direction):
    cfg, o = isa_config(isa), ops(isa)
    T, W, fwd = cfg['T'], cfg['W'], direction == 'fwd'
    cm = (cmul_re, cmul_im) if fwd else (cmulc_re, cmulc_im)
    L = []
    if cfg['attr']: L.append(cfg['attr'])
    L.append(f'static inline void')
    L.append(f'radix5_tw_dif_kernel_{direction}_{isa}(')
    L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    L.append(f'    size_t K)')
    L.append(f'{{')
    emit_constants(L, o, cfg)
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    gen_derive(L, o, cfg)
    for n in range(5):
        L.append(f'        const {T} x{n}r = {cfg["ld"](f"&in_re[{n}*K+k]")}, x{n}i = {cfg["ld"](f"&in_im[{n}*K+k]")};')
    emit_butterfly(L, o, cfg, fwd, 'x', False)  # → Y0..Y4
    L.append(f'        {cfg["st"]("&out_re[0*K+k]","Y0r")} {cfg["st"]("&out_im[0*K+k]","Y0i")}')
    for n in range(1, 5):
        L.append(f'        {{ const {T} tr = {cm[0](o,f"Y{n}r",f"Y{n}i",f"w{n}r",f"w{n}i")};')
        L.append(f'          const {T} ti = {cm[1](o,f"Y{n}r",f"Y{n}i",f"w{n}r",f"w{n}i")};')
        L.append(f'          {cfg["st"](f"&out_re[{n}*K+k]","tr")} {cfg["st"](f"&out_im[{n}*K+k]","ti")} }}')
    L.append(f'    }}')
    L.append(f'}}')
    return L

def gen_dit_file(isa):
    ISA = isa.upper()
    guard = f'FFT_RADIX5_{ISA}_H'
    L = [f'/**', f' * @file fft_radix5_{isa}.h',
         f' * @brief DFT-5 {ISA} codelets (notw + DIT tw with 1-base log3)',
         f' * Load W^1 only, derive W^2..W^4 in registers.',
         f' * Generated by gen_r5_log3.py', f' */', f'',
         f'#ifndef {guard}', f'#define {guard}', f'#include <stddef.h>']
    if isa != 'scalar': L.append(f'#include <immintrin.h>')
    L.append('')
    for d in ('fwd','bwd'):
        L.extend(gen_notw(isa, d)); L.append('')
    for d in ('fwd','bwd'):
        L.extend(gen_dit_tw(isa, d)); L.append('')
    L.append(f'#endif /* {guard} */')
    return L

def gen_dif_file(isa):
    ISA = isa.upper()
    guard = f'FFT_RADIX5_{ISA}_DIF_TW_H'
    L = [f'/**', f' * @file fft_radix5_{isa}_dif_tw.h',
         f' * @brief DFT-5 {ISA} DIF tw codelets with 1-base log3',
         f' * Generated by gen_r5_log3.py', f' */', f'',
         f'#ifndef {guard}', f'#define {guard}', f'#include <stddef.h>']
    if isa != 'scalar': L.append(f'#include <immintrin.h>')
    L.append('')
    for d in ('fwd','bwd'):
        L.extend(gen_dif_tw(isa, d)); L.append('')
    L.append(f'#endif /* {guard} */')
    return L

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: gen_r5_log3.py <scalar|avx2|avx512> <dit|dif>", file=sys.stderr)
        sys.exit(1)
    isa, mode = sys.argv[1], sys.argv[2]
    lines = gen_dit_file(isa) if mode == 'dit' else gen_dif_file(isa)
    print('\n'.join(lines))
