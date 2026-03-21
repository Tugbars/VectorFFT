#!/usr/bin/env python3
"""
gen_r5_avx2.py — Definitive DFT-5 AVX2 codelet generator

Dual twiddle strategy with K-threshold dispatch:
  K <= R5_FLAT_THRESHOLD (default 2048): flat — 4 independent loads
  K >  R5_FLAT_THRESHOLD:               log3 — 1 load, derive W^2..W^4

Load-apply-free scheduling keeps peak at ~14 YMM (no spills) for both paths.

Produces 4 files matching dispatch.h:
  fft_radix5_avx2.h           — notw + tw DIT (split)
  fft_radix5_avx2_dif_tw.h    — tw DIF (split)
  fft_radix5_avx2_il.h        — notw + tw DIT (IL)
  fft_radix5_avx2_il_dif_tw.h — tw DIF (IL)

Usage: python gen_r5_avx2.py [outdir]
"""
import sys, os

T    = '__m256d'
W    = 4
ATTR = '__attribute__((target("avx2,fma")))'
P    = '_mm256'
THRESHOLD = 2048

def ld(addr):   return f'{P}_load_pd({addr})'
def st(addr,v): return f'{P}_store_pd({addr},{v});'
def add(a,b):   return f'{P}_add_pd({a},{b})'
def sub(a,b):   return f'{P}_sub_pd({a},{b})'
def mul(a,b):   return f'{P}_mul_pd({a},{b})'
def fma_(a,b,c):return f'{P}_fmadd_pd({a},{b},{c})'
def fms(a,b,c): return f'{P}_fmsub_pd({a},{b},{c})'
def fnma(a,b,c):return f'{P}_fnmadd_pd({a},{b},{c})'
def set1(v):    return f'{P}_set1_pd({v})'

def cmul_re(ar,ai,br,bi): return fms(ar,br, mul(ai,bi))
def cmul_im(ar,ai,br,bi): return fma_(ar,bi, mul(ai,br))
def cmulc_re(ar,ai,br,bi): return fma_(ar,br, mul(ai,bi))
def cmulc_im(ar,ai,br,bi): return fms(ai,br, mul(ar,bi))

# ─── IL load/store ───

def il_load(L, v, idx):
    L.append(f'        {{ {T} _lo = {ld(f"&in[2*({idx}*K+k)]")};')
    L.append(f'          {T} _hi = {ld(f"&in[2*({idx}*K+k+2)]")};')
    L.append(f'          {v}r = {P}_permute4x64_pd({P}_shuffle_pd(_lo,_hi,0x0), 0xD8);')
    L.append(f'          {v}i = {P}_permute4x64_pd({P}_shuffle_pd(_lo,_hi,0xF), 0xD8); }}')

def il_store(L, vr, vi, idx):
    L.append(f'        {{ {T} _rp = {P}_permute4x64_pd({vr}, 0xD8);')
    L.append(f'          {T} _ip = {P}_permute4x64_pd({vi}, 0xD8);')
    L.append(f'          {st(f"&out[2*({idx}*K+k)]", f"{P}_shuffle_pd(_rp,_ip,0x0)")}')
    L.append(f'          {st(f"&out[2*({idx}*K+k+2)]", f"{P}_shuffle_pd(_rp,_ip,0xF)")} }}')

# ─── Constants ───

def emit_constants(L):
    L.append(f'    const {T} cA = {set1("0.559016994374947424102293417182819058860154590")};')
    L.append(f'    const {T} cB = {set1("0.951056516295153572116439333379382143405698634")};')
    L.append(f'    const {T} cC = {set1("0.587785252292473129168705954639072768597652438")};')
    L.append(f'    const {T} cD = {set1("0.250000000000000000000000000000000000000000000")};')

# ─── Butterfly ───

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
    L.append(f'        const {T} qar={fma_("cC","d2i",mul("cB","d1i"))},qai={fma_("cC","d2r",mul("cB","d1r"))};')
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

# ─── Twiddle load helpers ───

def emit_tw_flat(L, il):
    """Flat: 4 independent loads, load-apply-free scheduling."""
    if il:
        L.append(f'        {T} x0r, x0i;')
        il_load(L, 'x0', 0)
    else:
        L.append(f'        const {T} x0r = {ld("&in_re[0*K+k]")}, x0i = {ld("&in_im[0*K+k]")};')
    for n in range(1, 5):
        if il:
            L.append(f'        {T} _r{n}r, _r{n}i;')
            il_load(L, f'_r{n}', n)
        else:
            L.append(f'        const {T} _r{n}r = {ld(f"&in_re[{n}*K+k]")}, _r{n}i = {ld(f"&in_im[{n}*K+k]")};')
        L.append(f'        const {T} _w{n}r = {ld(f"&tw_re[{n-1}*K+k]")}, _w{n}i = {ld(f"&tw_im[{n-1}*K+k]")};')
    return True  # tw vars are _w1r.._w4r

def emit_tw_log3(L, il):
    """Log3: 1 load, derive W^2=(W^1)², W^3=W^1×W^2, W^4=W^1×W^3."""
    L.append(f'        const {T} _w1r = {ld("&tw_re[0*K+k]")}, _w1i = {ld("&tw_im[0*K+k]")};')
    L.append(f'        const {T} _w2r = {cmul_re("_w1r","_w1i","_w1r","_w1i")};')
    L.append(f'        const {T} _w2i = {cmul_im("_w1r","_w1i","_w1r","_w1i")};')
    L.append(f'        const {T} _w3r = {cmul_re("_w1r","_w1i","_w2r","_w2i")};')
    L.append(f'        const {T} _w3i = {cmul_im("_w1r","_w1i","_w2r","_w2i")};')
    L.append(f'        const {T} _w4r = {cmul_re("_w1r","_w1i","_w3r","_w3i")};')
    L.append(f'        const {T} _w4i = {cmul_im("_w1r","_w1i","_w3r","_w3i")};')
    if il:
        L.append(f'        {T} x0r, x0i;')
        il_load(L, 'x0', 0)
    else:
        L.append(f'        const {T} x0r = {ld("&in_re[0*K+k]")}, x0i = {ld("&in_im[0*K+k]")};')
    for n in range(1, 5):
        if il:
            L.append(f'        {T} _r{n}r, _r{n}i;')
            il_load(L, f'_r{n}', n)
        else:
            L.append(f'        const {T} _r{n}r = {ld(f"&in_re[{n}*K+k]")}, _r{n}i = {ld(f"&in_im[{n}*K+k]")};')

def emit_tw_apply(L, fwd):
    """Apply twiddles _w1.._w4 to raw _r1.._r4 → x1..x4."""
    cm_re, cm_im = (cmul_re, cmul_im) if fwd else (cmulc_re, cmulc_im)
    for n in range(1, 5):
        L.append(f'        const {T} x{n}r = {cm_re(f"_r{n}r",f"_r{n}i",f"_w{n}r",f"_w{n}i")};')
        L.append(f'        const {T} x{n}i = {cm_im(f"_r{n}r",f"_r{n}i",f"_w{n}r",f"_w{n}i")};')

# ═══════════════════════════════════════════════════════════════
# NOTW (unchanged — no twiddles)
# ═══════════════════════════════════════════════════════════════

def gen_notw(direction, il=False):
    fwd = direction == 'fwd'
    il_tag = '_il' if il else ''
    L = [ATTR, f'static inline void',
         f'radix5_notw_dit_kernel_{direction}{il_tag}_avx2(']
    if il: L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L += [f'    size_t K)', f'{{']
    emit_constants(L)
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    if il:
        for n in range(5):
            L.append(f'        {T} x{n}r, x{n}i;')
            il_load(L, f'x{n}', n)
    else:
        for n in range(5):
            L.append(f'        const {T} x{n}r = {ld(f"&in_re[{n}*K+k]")}, x{n}i = {ld(f"&in_im[{n}*K+k]")};')
    if il:
        emit_butterfly(L, fwd, lambda L,i,vr,vi: il_store(L,vr,vi,i))
    else:
        emit_butterfly(L, fwd, lambda L,i,vr,vi: L.append(f'        {st(f"&out_re[{i}*K+k]",vr)} {st(f"&out_im[{i}*K+k]",vi)}'))
    L += [f'    }}', f'}}']
    return L

# ═══════════════════════════════════════════════════════════════
# DIT TW — dual mode: flat + log3 internals, K-threshold wrapper
# ═══════════════════════════════════════════════════════════════

def gen_dit_tw_internal(direction, il, mode):
    """Generate a _flat or _log3 static helper."""
    fwd = direction == 'fwd'
    il_tag = '_il' if il else ''
    L = [ATTR, f'static inline void',
         f'radix5_tw_dit_kernel_{direction}{il_tag}_{mode}_avx2(']
    if il: L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L += [f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,',
          f'    size_t K)', f'{{']
    emit_constants(L)
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    if mode == 'flat':
        emit_tw_flat(L, il)
    else:
        emit_tw_log3(L, il)
    emit_tw_apply(L, fwd)
    if il:
        emit_butterfly(L, fwd, lambda L,i,vr,vi: il_store(L,vr,vi,i))
    else:
        emit_butterfly(L, fwd, lambda L,i,vr,vi: L.append(f'        {st(f"&out_re[{i}*K+k]",vr)} {st(f"&out_im[{i}*K+k]",vi)}'))
    L += [f'    }}', f'}}']
    return L

def gen_dit_tw(direction, il=False):
    """Generate public wrapper that dispatches flat vs log3 on K."""
    fwd = direction == 'fwd'
    il_tag = '_il' if il else ''
    # First emit both internal variants
    lines_flat = gen_dit_tw_internal(direction, il, 'flat')
    lines_log3 = gen_dit_tw_internal(direction, il, 'log3')
    # Then the wrapper
    L = [ATTR, f'static inline void',
         f'radix5_tw_dit_kernel_{direction}{il_tag}_avx2(']
    if il: L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L += [f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,',
          f'    size_t K)', f'{{']
    L.append(f'    if (K <= {THRESHOLD}) {{')
    if il:
        L.append(f'        radix5_tw_dit_kernel_{direction}{il_tag}_flat_avx2(in, out, tw_re, tw_im, K);')
    else:
        L.append(f'        radix5_tw_dit_kernel_{direction}{il_tag}_flat_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);')
    L.append(f'    }} else {{')
    if il:
        L.append(f'        radix5_tw_dit_kernel_{direction}{il_tag}_log3_avx2(in, out, tw_re, tw_im, K);')
    else:
        L.append(f'        radix5_tw_dit_kernel_{direction}{il_tag}_log3_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);')
    L += [f'    }}', f'}}']
    return lines_flat + [''] + lines_log3 + [''] + L

# ═══════════════════════════════════════════════════════════════
# DIF TW — same dual dispatch
# ═══════════════════════════════════════════════════════════════

def gen_dif_tw_internal(direction, il, mode):
    fwd = direction == 'fwd'
    cm_re, cm_im = (cmul_re, cmul_im) if fwd else (cmulc_re, cmulc_im)
    il_tag = '_il' if il else ''
    L = [ATTR, f'static inline void',
         f'radix5_tw_dif_kernel_{direction}{il_tag}_{mode}_avx2(']
    if il: L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L += [f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,',
          f'    size_t K)', f'{{']
    emit_constants(L)
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    # Load inputs (no twiddle yet — DIF)
    if il:
        for n in range(5):
            L.append(f'        {T} x{n}r, x{n}i;')
            il_load(L, f'x{n}', n)
    else:
        for n in range(5):
            L.append(f'        const {T} x{n}r = {ld(f"&in_re[{n}*K+k]")}, x{n}i = {ld(f"&in_im[{n}*K+k]")};')
    # Butterfly → Y temporaries
    def store_tmp(L,i,vr,vi): L.append(f'        const {T} Y{i}r = {vr}, Y{i}i = {vi};')
    emit_butterfly(L, fwd, store_tmp)
    # Store Y0 (no twiddle)
    if il: il_store(L, 'Y0r', 'Y0i', 0)
    else: L.append(f'        {st("&out_re[0*K+k]","Y0r")} {st("&out_im[0*K+k]","Y0i")}')
    # Twiddle Y1..Y4
    if mode == 'flat':
        for n in range(1, 5):
            L.append(f'        {{ const {T} _wr = {ld(f"&tw_re[{n-1}*K+k]")}, _wi = {ld(f"&tw_im[{n-1}*K+k]")};')
            L.append(f'          const {T} _or = {cm_re(f"Y{n}r",f"Y{n}i","_wr","_wi")};')
            L.append(f'          const {T} _oi = {cm_im(f"Y{n}r",f"Y{n}i","_wr","_wi")};')
            if il: il_store(L, '_or', '_oi', n)
            else: L.append(f'          {st(f"&out_re[{n}*K+k]","_or")} {st(f"&out_im[{n}*K+k]","_oi")}')
            L.append(f'        }}')
    else:  # log3
        L.append(f'        const {T} _w1r = {ld("&tw_re[0*K+k]")}, _w1i = {ld("&tw_im[0*K+k]")};')
        L.append(f'        const {T} _w2r = {cmul_re("_w1r","_w1i","_w1r","_w1i")};')
        L.append(f'        const {T} _w2i = {cmul_im("_w1r","_w1i","_w1r","_w1i")};')
        L.append(f'        const {T} _w3r = {cmul_re("_w1r","_w1i","_w2r","_w2i")};')
        L.append(f'        const {T} _w3i = {cmul_im("_w1r","_w1i","_w2r","_w2i")};')
        L.append(f'        const {T} _w4r = {cmul_re("_w1r","_w1i","_w3r","_w3i")};')
        L.append(f'        const {T} _w4i = {cmul_im("_w1r","_w1i","_w3r","_w3i")};')
        for n in range(1, 5):
            L.append(f'        {{ const {T} _or = {cm_re(f"Y{n}r",f"Y{n}i",f"_w{n}r",f"_w{n}i")};')
            L.append(f'          const {T} _oi = {cm_im(f"Y{n}r",f"Y{n}i",f"_w{n}r",f"_w{n}i")};')
            if il: il_store(L, '_or', '_oi', n)
            else: L.append(f'          {st(f"&out_re[{n}*K+k]","_or")} {st(f"&out_im[{n}*K+k]","_oi")}')
            L.append(f'        }}')
    L += [f'    }}', f'}}']
    return L

def gen_dif_tw(direction, il=False):
    fwd = direction == 'fwd'
    il_tag = '_il' if il else ''
    lines_flat = gen_dif_tw_internal(direction, il, 'flat')
    lines_log3 = gen_dif_tw_internal(direction, il, 'log3')
    L = [ATTR, f'static inline void',
         f'radix5_tw_dif_kernel_{direction}{il_tag}_avx2(']
    if il: L.append(f'    const double * __restrict__ in, double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L += [f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,',
          f'    size_t K)', f'{{']
    L.append(f'    if (K <= {THRESHOLD}) {{')
    if il:
        L.append(f'        radix5_tw_dif_kernel_{direction}{il_tag}_flat_avx2(in, out, tw_re, tw_im, K);')
    else:
        L.append(f'        radix5_tw_dif_kernel_{direction}{il_tag}_flat_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);')
    L.append(f'    }} else {{')
    if il:
        L.append(f'        radix5_tw_dif_kernel_{direction}{il_tag}_log3_avx2(in, out, tw_re, tw_im, K);')
    else:
        L.append(f'        radix5_tw_dif_kernel_{direction}{il_tag}_log3_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);')
    L += [f'    }}', f'}}']
    return lines_flat + [''] + lines_log3 + [''] + L

# ═══════════════════════════════════════════════════════════════
# File generators
# ═══════════════════════════════════════════════════════════════

HEADER = """/**
 * @file {fname}
 * @brief DFT-5 AVX2 — {desc}
 *
 * Dual twiddle: flat (K<={threshold}) / log3 (K>{threshold}).
 * Load-apply-free scheduling, ~14 YMM peak, zero spills.
 * Generated by gen_r5_avx2.py
 */

#ifndef {guard}
#define {guard}

#ifndef R5_FLAT_THRESHOLD_AVX2
#define R5_FLAT_THRESHOLD_AVX2 {threshold}
#endif

#include <stddef.h>
#include <immintrin.h>
"""

def write_file(path, fname, desc, guard, functions):
    lines = [HEADER.format(fname=fname, desc=desc, guard=guard, threshold=THRESHOLD)]
    for fn_lines in functions:
        lines.append('\n'.join(fn_lines)); lines.append('')
    lines.append(f'#endif /* {guard} */')
    with open(path, 'w') as f: f.write('\n'.join(lines)+'\n')

def gen_all(outdir='.'):
    os.makedirs(outdir, exist_ok=True)
    files = []
    for fname,desc,guard,fns in [
        ('fft_radix5_avx2.h','notw+DIT tw split','FFT_RADIX5_AVX2_H',
         [gen_notw('fwd'),gen_notw('bwd'),gen_dit_tw('fwd'),gen_dit_tw('bwd')]),
        ('fft_radix5_avx2_dif_tw.h','DIF tw split','FFT_RADIX5_AVX2_DIF_TW_H',
         [gen_dif_tw('fwd'),gen_dif_tw('bwd')]),
        ('fft_radix5_avx2_il.h','notw+DIT tw IL','FFT_RADIX5_AVX2_IL_H',
         [gen_notw('fwd',il=True),gen_notw('bwd',il=True),gen_dit_tw('fwd',il=True),gen_dit_tw('bwd',il=True)]),
        ('fft_radix5_avx2_il_dif_tw.h','DIF tw IL','FFT_RADIX5_AVX2_IL_DIF_TW_H',
         [gen_dif_tw('fwd',il=True),gen_dif_tw('bwd',il=True)]),
    ]:
        write_file(os.path.join(outdir,fname),fname,desc,guard,fns)
        files.append(fname)
    return files

if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    for f in gen_all(outdir): print(f'  Generated: {f}')
