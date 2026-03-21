#!/usr/bin/env python3
"""
gen_r5_scalar.py — Definitive DFT-5 scalar codelet generator
Flat twiddles. Produces 4 files matching dispatch.h.
Usage: python gen_r5_scalar.py [outdir]
"""
import sys, os

T = 'double'
W = 1

def ld(addr):   return f'*({addr})'
def st(addr,v): return f'*({addr}) = {v};'
def add(a,b):   return f'({a}+{b})'
def sub(a,b):   return f'({a}-{b})'
def mul(a,b):   return f'({a}*{b})'
def fma(a,b,c): return f'({a}*{b}+{c})'
def fms(a,b,c): return f'({a}*{b}-{c})'
def fnma(a,b,c):return f'({c}-{a}*{b})'
def set1(v):    return v

def cmul_re(ar,ai,br,bi): return fms(ar,br, mul(ai,bi))
def cmul_im(ar,ai,br,bi): return fma(ar,bi, mul(ai,br))
def cmulc_re(ar,ai,br,bi): return fma(ar,br, mul(ai,bi))
def cmulc_im(ar,ai,br,bi): return fms(ai,br, mul(ar,bi))

def il_load(L, v, idx):
    L.append(f'        const {T} {v}r = in[2*({idx}*K+k)], {v}i = in[2*({idx}*K+k)+1];')

def il_store(L, vr, vi, idx):
    L.append(f'        out[2*({idx}*K+k)] = {vr}; out[2*({idx}*K+k)+1] = {vi};')

CONSTS = {
    'cA': '0.559016994374947424102293417182819058860154590',
    'cB': '0.951056516295153572116439333379382143405698634',
    'cC': '0.587785252292473129168705954639072768597652438',
    'cD': '0.250000000000000000000000000000000000000000000',
}

def emit_constants(L):
    for name, val in CONSTS.items():
        L.append(f'    const {T} {name} = {val};')

def emit_butterfly(L, fwd, store_fn):
    L.append(f'        const {T} s1r={add("x1r","x4r")},s1i={add("x1i","x4i")};')
    L.append(f'        const {T} s2r={add("x2r","x3r")},s2i={add("x2i","x3i")};')
    L.append(f'        const {T} d1r={sub("x1r","x4r")},d1i={sub("x1i","x4i")};')
    L.append(f'        const {T} d2r={sub("x2r","x3r")},d2i={sub("x2i","x3i")};')
    store_fn(L, 0, add("x0r",add("s1r","s2r")), add("x0i",add("s1i","s2i")))
    L.append(f'        const {T} t0r={fnma("cD",add("s1r","s2r"),"x0r")},t0i={fnma("cD",add("s1i","s2i"),"x0i")};')
    L.append(f'        const {T} t1r={mul("cA",sub("s1r","s2r"))},t1i={mul("cA",sub("s1i","s2i"))};')
    L.append(f'        const {T} p1r={add("t0r","t1r")},p1i={add("t0i","t1i")};')
    L.append(f'        const {T} p2r={sub("t0r","t1r")},p2i={sub("t0i","t1i")};')
    L.append(f'        const {T} qar={fma("cC","d2i",mul("cB","d1i"))},qai={fma("cC","d2r",mul("cB","d1r"))};')
    L.append(f'        const {T} qbr={fnma("cC","d1i",mul("cB","d2i"))},qbi={fnma("cC","d1r",mul("cB","d2r"))};')
    if fwd:
        store_fn(L,1,add("p1r","qar"),sub("p1i","qai"))
        store_fn(L,4,sub("p1r","qar"),add("p1i","qai"))
        store_fn(L,2,sub("p2r","qbr"),add("p2i","qbi"))
        store_fn(L,3,add("p2r","qbr"),sub("p2i","qbi"))
    else:
        store_fn(L,1,sub("p1r","qar"),add("p1i","qai"))
        store_fn(L,4,add("p1r","qar"),sub("p1i","qai"))
        store_fn(L,2,add("p2r","qbr"),sub("p2i","qbi"))
        store_fn(L,3,sub("p2r","qbr"),add("p2i","qbi"))

def gen_notw(direction, il=False):
    fwd = direction == 'fwd'
    il_tag = '_il' if il else ''
    L = [f'static inline void', f'radix5_notw_dit_kernel_{direction}{il_tag}_scalar(']
    if il: L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L += [f'    size_t K)', f'{{']
    emit_constants(L)
    L.append(f'    for (size_t k = 0; k < K; k++) {{')
    if il:
        for n in range(5): il_load(L, f'x{n}', n)
    else:
        for n in range(5):
            L.append(f'        const {T} x{n}r = {ld(f"&in_re[{n}*K+k]")}, x{n}i = {ld(f"&in_im[{n}*K+k]")};')
    if il:
        emit_butterfly(L, fwd, lambda L,i,vr,vi: il_store(L,vr,vi,i))
    else:
        emit_butterfly(L, fwd, lambda L,i,vr,vi: L.append(f'        {st(f"&out_re[{i}*K+k]",vr)} {st(f"&out_im[{i}*K+k]",vi)}'))
    L += [f'    }}', f'}}']
    return L

def gen_dit_tw(direction, il=False):
    fwd = direction == 'fwd'
    cm_re, cm_im = (cmul_re,cmul_im) if fwd else (cmulc_re,cmulc_im)
    il_tag = '_il' if il else ''
    L = [f'static inline void', f'radix5_tw_dit_kernel_{direction}{il_tag}_scalar(']
    if il: L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L += [f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,',
          f'    size_t K)', f'{{']
    emit_constants(L)
    L.append(f'    for (size_t k = 0; k < K; k++) {{')
    if il: il_load(L, 'x0', 0)
    else: L.append(f'        const {T} x0r = {ld("&in_re[0*K+k]")}, x0i = {ld("&in_im[0*K+k]")};')
    for n in range(1,5):
        if il: il_load(L, f'_r{n}', n)
        else: L.append(f'        const {T} _r{n}r = {ld(f"&in_re[{n}*K+k]")}, _r{n}i = {ld(f"&in_im[{n}*K+k]")};')
        L.append(f'        const {T} _w{n}r = {ld(f"&tw_re[{n-1}*K+k]")}, _w{n}i = {ld(f"&tw_im[{n-1}*K+k]")};')
        L.append(f'        const {T} x{n}r = {cm_re(f"_r{n}r",f"_r{n}i",f"_w{n}r",f"_w{n}i")};')
        L.append(f'        const {T} x{n}i = {cm_im(f"_r{n}r",f"_r{n}i",f"_w{n}r",f"_w{n}i")};')
    if il:
        emit_butterfly(L, fwd, lambda L,i,vr,vi: il_store(L,vr,vi,i))
    else:
        emit_butterfly(L, fwd, lambda L,i,vr,vi: L.append(f'        {st(f"&out_re[{i}*K+k]",vr)} {st(f"&out_im[{i}*K+k]",vi)}'))
    L += [f'    }}', f'}}']
    return L

def gen_dif_tw(direction, il=False):
    fwd = direction == 'fwd'
    cm_re, cm_im = (cmul_re,cmul_im) if fwd else (cmulc_re,cmulc_im)
    il_tag = '_il' if il else ''
    L = [f'static inline void', f'radix5_tw_dif_kernel_{direction}{il_tag}_scalar(']
    if il: L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L += [f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,',
          f'    size_t K)', f'{{']
    emit_constants(L)
    L.append(f'    for (size_t k = 0; k < K; k++) {{')
    if il:
        for n in range(5): il_load(L, f'x{n}', n)
    else:
        for n in range(5):
            L.append(f'        const {T} x{n}r = {ld(f"&in_re[{n}*K+k]")}, x{n}i = {ld(f"&in_im[{n}*K+k]")};')
    def store_tmp(L,i,vr,vi): L.append(f'        const {T} Y{i}r = {vr}, Y{i}i = {vi};')
    emit_butterfly(L, fwd, store_tmp)
    if il: il_store(L, 'Y0r', 'Y0i', 0)
    else: L.append(f'        {st("&out_re[0*K+k]","Y0r")} {st("&out_im[0*K+k]","Y0i")}')
    for n in range(1,5):
        L.append(f'        {{ const {T} _wr = {ld(f"&tw_re[{n-1}*K+k]")}, _wi = {ld(f"&tw_im[{n-1}*K+k]")};')
        L.append(f'          const {T} _or = {cm_re(f"Y{n}r",f"Y{n}i","_wr","_wi")};')
        L.append(f'          const {T} _oi = {cm_im(f"Y{n}r",f"Y{n}i","_wr","_wi")};')
        if il: il_store(L, '_or', '_oi', n)
        else: L.append(f'          {st(f"&out_re[{n}*K+k]","_or")} {st(f"&out_im[{n}*K+k]","_oi")}')
        L.append(f'        }}')
    L += [f'    }}', f'}}']
    return L

HEADER = """/**
 * @file {fname}
 * @brief DFT-5 scalar — {desc}
 * Flat twiddles. Generated by gen_r5_scalar.py
 */

#ifndef {guard}
#define {guard}
#include <stddef.h>
"""

def write_file(path, fname, desc, guard, functions):
    lines = [HEADER.format(fname=fname, desc=desc, guard=guard)]
    for fn_lines in functions:
        lines.append('\n'.join(fn_lines)); lines.append('')
    lines.append(f'#endif /* {guard} */')
    with open(path, 'w') as f: f.write('\n'.join(lines)+'\n')

def gen_all(outdir='.'):
    os.makedirs(outdir, exist_ok=True)
    files = []
    for fname,desc,guard,fns in [
        ('fft_radix5_scalar.h','notw+DIT tw split','FFT_RADIX5_SCALAR_H',
         [gen_notw('fwd'),gen_notw('bwd'),gen_dit_tw('fwd'),gen_dit_tw('bwd')]),
        ('fft_radix5_scalar_dif_tw.h','DIF tw split','FFT_RADIX5_SCALAR_DIF_TW_H',
         [gen_dif_tw('fwd'),gen_dif_tw('bwd')]),
        ('fft_radix5_scalar_il.h','notw+DIT tw IL','FFT_RADIX5_SCALAR_IL_H',
         [gen_notw('fwd',il=True),gen_notw('bwd',il=True),gen_dit_tw('fwd',il=True),gen_dit_tw('bwd',il=True)]),
        ('fft_radix5_scalar_il_dif_tw.h','DIF tw IL','FFT_RADIX5_SCALAR_IL_DIF_TW_H',
         [gen_dif_tw('fwd',il=True),gen_dif_tw('bwd',il=True)]),
    ]:
        write_file(os.path.join(outdir,fname),fname,desc,guard,fns)
        files.append(fname)
    return files

if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    for f in gen_all(outdir): print(f'  Generated: {f}')
