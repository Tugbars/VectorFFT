#!/usr/bin/env python3
"""
gen_radix2.py — DFT-2 codelets for VectorFFT

Algorithm: y[0] = x[0] + x[1], y[1] = x[0] - x[1]
Identical for fwd and bwd (DFT-2 is its own inverse up to scaling).

Zero internal twiddles, zero spill, zero constants.
The simplest possible codelet — just add/sub.

Variants:
  notw:    y[0] = x[0]+x[1], y[1] = x[0]-x[1]
  dit_tw:  x[1] *= W^k first, then add/sub
  dif_tw:  add/sub first, then y[1] *= W^k

Usage:
  python3 gen_radix2.py avx2   > fft_radix2_avx2.h
  python3 gen_radix2.py avx512 > fft_radix2_avx512.h
  python3 gen_radix2.py scalar > fft_radix2_scalar.h
"""

import sys

ISA = {
    'scalar': {
        'T': 'double', 'VL': 1,
        'target': '',
        'ld': lambda p: f'(*({p}))',
        'st': lambda p,v: f'*({p}) = {v}',
        'add': lambda a,b: f'({a})+({b})',
        'sub': lambda a,b: f'({a})-({b})',
        'mul': lambda a,b: f'({a})*({b})',
        'fma': lambda a,b,c: f'({a})*({b})+({c})',
        'fms': lambda a,b,c: f'({a})*({b})-({c})',
        'ld_macro': 'R2S_LD', 'st_macro': 'R2S_ST',
    },
    'avx2': {
        'T': '__m256d', 'VL': 4,
        'target': '__attribute__((target("avx2,fma")))',
        'ld': lambda p: f'LD({p})',
        'st': lambda p,v: f'ST({p},{v})',
        'add': lambda a,b: f'_mm256_add_pd({a},{b})',
        'sub': lambda a,b: f'_mm256_sub_pd({a},{b})',
        'mul': lambda a,b: f'_mm256_mul_pd({a},{b})',
        'fma': lambda a,b,c: f'_mm256_fmadd_pd({a},{b},{c})',
        'fms': lambda a,b,c: f'_mm256_fmsub_pd({a},{b},{c})',
        'ld_macro': 'R2A_LD', 'st_macro': 'R2A_ST',
        'ld_raw': '_mm256_loadu_pd', 'st_raw': '_mm256_storeu_pd',
    },
    'avx512': {
        'T': '__m512d', 'VL': 8,
        'target': '__attribute__((target("avx512f,fma")))',
        'ld': lambda p: f'LD({p})',
        'st': lambda p,v: f'ST({p},{v})',
        'add': lambda a,b: f'_mm512_add_pd({a},{b})',
        'sub': lambda a,b: f'_mm512_sub_pd({a},{b})',
        'mul': lambda a,b: f'_mm512_mul_pd({a},{b})',
        'fma': lambda a,b,c: f'_mm512_fmadd_pd({a},{b},{c})',
        'fms': lambda a,b,c: f'_mm512_fmsub_pd({a},{b},{c})',
        'ld_macro': 'R2Z_LD', 'st_macro': 'R2Z_ST',
        'ld_raw': '_mm512_loadu_pd', 'st_raw': '_mm512_storeu_pd',
    },
}


def gen_notw(isa_name, direction):
    """Notw: y[0]=x[0]+x[1], y[1]=x[0]-x[1]. Same for fwd/bwd."""
    I = ISA[isa_name]
    T, VL = I['T'], I['VL']
    lines = []

    if isa_name == 'scalar':
        lines.append(f'    for (size_t k = 0; k < K; k++) {{')
        lines.append(f'        {T} ar = in_re[k], ai = in_im[k];')
        lines.append(f'        {T} br = in_re[K+k], bi = in_im[K+k];')
        lines.append(f'        out_re[k]   = {I["add"]("ar","br")};')
        lines.append(f'        out_im[k]   = {I["add"]("ai","bi")};')
        lines.append(f'        out_re[K+k] = {I["sub"]("ar","br")};')
        lines.append(f'        out_im[K+k] = {I["sub"]("ai","bi")};')
        lines.append(f'    }}')
    else:
        lines.append(f'    for (size_t k = 0; k < K; k += {VL}) {{')
        lines.append(f'        {T} ar = {I["ld"]("&in_re[k]")}, ai = {I["ld"]("&in_im[k]")};')
        lines.append(f'        {T} br = {I["ld"]("&in_re[K+k]")}, bi = {I["ld"]("&in_im[K+k]")};')
        lines.append(f'        {I["st"]("&out_re[k]", I["add"]("ar","br"))};')
        lines.append(f'        {I["st"]("&out_im[k]", I["add"]("ai","bi"))};')
        lines.append(f'        {I["st"]("&out_re[K+k]", I["sub"]("ar","br"))};')
        lines.append(f'        {I["st"]("&out_im[K+k]", I["sub"]("ai","bi"))};')
        lines.append(f'    }}')

    return '\n'.join(lines)


def gen_dit_tw(isa_name, direction):
    """DIT tw: x[1] *= W^k, then y[0]=x[0]+x[1], y[1]=x[0]-x[1]."""
    I = ISA[isa_name]
    T, VL = I['T'], I['VL']
    fwd = (direction == 'fwd')
    lines = []

    if isa_name == 'scalar':
        lines.append(f'    for (size_t k = 0; k < K; k++) {{')
        lines.append(f'        {T} ar = in_re[k], ai = in_im[k];')
        lines.append(f'        {T} br = in_re[K+k], bi = in_im[K+k];')
        lines.append(f'        {T} wr = tw_re[k], wi = tw_im[k];')
        if fwd:
            lines.append(f'        {T} tr = br*wr - bi*wi;')
            lines.append(f'        {T} ti = br*wi + bi*wr;')
        else:
            lines.append(f'        {T} tr = br*wr + bi*wi;')
            lines.append(f'        {T} ti = bi*wr - br*wi;')
        lines.append(f'        out_re[k]   = ar + tr;')
        lines.append(f'        out_im[k]   = ai + ti;')
        lines.append(f'        out_re[K+k] = ar - tr;')
        lines.append(f'        out_im[K+k] = ai - ti;')
        lines.append(f'    }}')
    else:
        lines.append(f'    for (size_t k = 0; k < K; k += {VL}) {{')
        lines.append(f'        {T} ar = {I["ld"]("&in_re[k]")}, ai = {I["ld"]("&in_im[k]")};')
        lines.append(f'        {T} br = {I["ld"]("&in_re[K+k]")}, bi = {I["ld"]("&in_im[K+k]")};')
        lines.append(f'        {T} wr = {I["ld"]("&tw_re[k]")}, wi = {I["ld"]("&tw_im[k]")};')
        if fwd:
            lines.append(f'        {T} tr = {I["fms"]("br","wr",I["mul"]("bi","wi"))};')
            lines.append(f'        {T} ti = {I["fma"]("br","wi",I["mul"]("bi","wr"))};')
        else:
            lines.append(f'        {T} tr = {I["fma"]("br","wr",I["mul"]("bi","wi"))};')
            lines.append(f'        {T} ti = {I["fms"]("bi","wr",I["mul"]("br","wi"))};')
        lines.append(f'        {I["st"]("&out_re[k]", I["add"]("ar","tr"))};')
        lines.append(f'        {I["st"]("&out_im[k]", I["add"]("ai","ti"))};')
        lines.append(f'        {I["st"]("&out_re[K+k]", I["sub"]("ar","tr"))};')
        lines.append(f'        {I["st"]("&out_im[K+k]", I["sub"]("ai","ti"))};')
        lines.append(f'    }}')

    return '\n'.join(lines)


def gen_dif_tw(isa_name, direction):
    """DIF tw: y[0]=x[0]+x[1], y[1]=(x[0]-x[1])*W^k."""
    I = ISA[isa_name]
    T, VL = I['T'], I['VL']
    fwd = (direction == 'fwd')
    lines = []

    if isa_name == 'scalar':
        lines.append(f'    for (size_t k = 0; k < K; k++) {{')
        lines.append(f'        {T} ar = in_re[k], ai = in_im[k];')
        lines.append(f'        {T} br = in_re[K+k], bi = in_im[K+k];')
        lines.append(f'        {T} sr = ar + br, si = ai + bi;')
        lines.append(f'        {T} dr = ar - br, di = ai - bi;')
        lines.append(f'        {T} wr = tw_re[k], wi = tw_im[k];')
        lines.append(f'        out_re[k]   = sr;')
        lines.append(f'        out_im[k]   = si;')
        if fwd:
            lines.append(f'        out_re[K+k] = dr*wr - di*wi;')
            lines.append(f'        out_im[K+k] = dr*wi + di*wr;')
        else:
            lines.append(f'        out_re[K+k] = dr*wr + di*wi;')
            lines.append(f'        out_im[K+k] = di*wr - dr*wi;')
        lines.append(f'    }}')
    else:
        lines.append(f'    for (size_t k = 0; k < K; k += {VL}) {{')
        lines.append(f'        {T} ar = {I["ld"]("&in_re[k]")}, ai = {I["ld"]("&in_im[k]")};')
        lines.append(f'        {T} br = {I["ld"]("&in_re[K+k]")}, bi = {I["ld"]("&in_im[K+k]")};')
        lines.append(f'        {T} sr = {I["add"]("ar","br")}, si = {I["add"]("ai","bi")};')
        lines.append(f'        {T} dr = {I["sub"]("ar","br")}, di = {I["sub"]("ai","bi")};')
        lines.append(f'        {T} wr = {I["ld"]("&tw_re[k]")}, wi = {I["ld"]("&tw_im[k]")};')
        lines.append(f'        {I["st"]("&out_re[k]", "sr")};')
        lines.append(f'        {I["st"]("&out_im[k]", "si")};')
        if fwd:
            lines.append(f'        {I["st"]("&out_re[K+k]", I["fms"]("dr","wr",I["mul"]("di","wi")))};')
            lines.append(f'        {I["st"]("&out_im[K+k]", I["fma"]("dr","wi",I["mul"]("di","wr")))};')
        else:
            lines.append(f'        {I["st"]("&out_re[K+k]", I["fma"]("dr","wr",I["mul"]("di","wi")))};')
            lines.append(f'        {I["st"]("&out_im[K+k]", I["fms"]("di","wr",I["mul"]("dr","wi")))};')
        lines.append(f'    }}')

    return '\n'.join(lines)


def gen_file(isa_name):
    I = ISA[isa_name]
    T, VL = I['T'], I['VL']
    is_scalar = (isa_name == 'scalar')
    guard = f'FFT_RADIX2_{isa_name.upper()}_H'

    parts = []
    parts.append(f'/**')
    parts.append(f' * @file fft_radix2_{isa_name}.h')
    parts.append(f' * @brief DFT-2 {isa_name.upper()} codelets (notw + DIT tw + DIF tw)')
    parts.append(f' *')
    parts.append(f' * y[0] = x[0]+x[1], y[1] = x[0]-x[1] (identical fwd/bwd)')
    parts.append(f' * Zero constants, zero spill, k-step={VL}')
    parts.append(f' *')
    parts.append(f' * ── Operation counts per k-step ──')
    parts.append(f' *   notw:   2 add + 2 sub = 4 arith, 4 ld + 4 st = 8 mem')
    parts.append(f' *   dit_tw: 2 add + 2 sub + 2 mul + 2 fma = 8 arith, 6 ld + 4 st = 10 mem')
    parts.append(f' *   dif_tw: 2 add + 2 sub + 2 mul + 2 fma = 8 arith, 6 ld + 4 st = 10 mem')
    parts.append(f' *')
    parts.append(f' * Generated by gen_radix2.py — do not edit.')
    parts.append(f' */')
    parts.append(f'')
    parts.append(f'#ifndef {guard}')
    parts.append(f'#define {guard}')
    parts.append(f'')

    if is_scalar:
        parts.append(f'#include <stddef.h>')
    else:
        parts.append(f'#include <immintrin.h>')
        parts.append(f'#include <stddef.h>')
    parts.append(f'')

    # LD/ST macros (SIMD only)
    if not is_scalar:
        lm, sm = I['ld_macro'], I['st_macro']
        parts.append(f'#ifndef {lm}')
        parts.append(f'#define {lm}(p) {I["ld_raw"]}(p)')
        parts.append(f'#endif')
        parts.append(f'#ifndef {sm}')
        parts.append(f'#define {sm}(p,v) {I["st_raw"]}((p),(v))')
        parts.append(f'#endif')
        parts.append(f'#define LD {lm}')
        parts.append(f'#define ST {sm}')
        parts.append(f'')

    # Generate all kernel variants
    generators = [
        ('notw', gen_notw, False),
        ('tw_dit', gen_dit_tw, True),
        ('tw_dif', gen_dif_tw, True),
    ]

    for kind, gen_fn, has_tw in generators:
        for direction in ['fwd', 'bwd']:
            tw_params = ''
            if has_tw:
                tw_params = ',\n    const double * __restrict__ tw_re, const double * __restrict__ tw_im'

            target = I['target']
            if target:
                parts.append(f'{target}')
            parts.append(f'static inline void')
            parts.append(f'radix2_{kind}_kernel_{direction}_{isa_name}(')
            parts.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
            parts.append(f'    double * __restrict__ out_re, double * __restrict__ out_im{tw_params},')
            parts.append(f'    size_t K)')
            parts.append(f'{{')
            parts.append(gen_fn(isa_name, direction))
            parts.append(f'}}')
            parts.append(f'')

    # Cleanup
    if not is_scalar:
        parts.append(f'#undef LD')
        parts.append(f'#undef ST')
        parts.append(f'')

    parts.append(f'#endif /* {guard} */')
    return '\n'.join(parts)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ISA:
        print(f'Usage: {sys.argv[0]} avx512|avx2|scalar', file=sys.stderr)
        sys.exit(1)
    print(gen_file(sys.argv[1]))
