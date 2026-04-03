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


# ═══════════════════════════════════════════════════════════════
# STRIDE-VECTOR (sv) CODELETS
#
# No k-loop. Processes exactly VL k-values per call.
# Elements at stride vs (typically VL=4 for AVX2).
# Data:    in_re[n * vs + 0..VL-1]
# Twiddle: tw_re[(n-1) * vs + 0..VL-1]
#
# The executor calls this K/VL times, advancing all pointers by VL.
# SIMD only — scalar has no vectorization benefit.
# ═══════════════════════════════════════════════════════════════

def gen_n1sv(isa_name, direction):
    """n1sv: no twiddle, stride-vector. y[0]=x[0]+x[1], y[1]=x[0]-x[1]."""
    I = ISA[isa_name]
    T = I['T']
    lines = []
    lines.append(f'    {T} ar = {I["ld"]("&in_re[0]")},  ai = {I["ld"]("&in_im[0]")};')
    lines.append(f'    {T} br = {I["ld"]("&in_re[vs]")}, bi = {I["ld"]("&in_im[vs]")};')
    lines.append(f'    {I["st"]("&out_re[0]",  I["add"]("ar","br"))};')
    lines.append(f'    {I["st"]("&out_im[0]",  I["add"]("ai","bi"))};')
    lines.append(f'    {I["st"]("&out_re[vs]", I["sub"]("ar","br"))};')
    lines.append(f'    {I["st"]("&out_im[vs]", I["sub"]("ai","bi"))};')
    return '\n'.join(lines)


def gen_t1sv_dit(isa_name, direction):
    """t1sv DIT: twiddle input[1], then butterfly. Stride-vector."""
    I = ISA[isa_name]
    T = I['T']
    fwd = (direction == 'fwd')
    lines = []
    lines.append(f'    {T} ar = {I["ld"]("&in_re[0]")},  ai = {I["ld"]("&in_im[0]")};')
    lines.append(f'    {T} br = {I["ld"]("&in_re[vs]")}, bi = {I["ld"]("&in_im[vs]")};')
    lines.append(f'    {T} wr = {I["ld"]("&tw_re[0]")},  wi = {I["ld"]("&tw_im[0]")};')
    if fwd:
        lines.append(f'    {T} tr = {I["fms"]("br","wr",I["mul"]("bi","wi"))};')
        lines.append(f'    {T} ti = {I["fma"]("br","wi",I["mul"]("bi","wr"))};')
    else:
        lines.append(f'    {T} tr = {I["fma"]("br","wr",I["mul"]("bi","wi"))};')
        lines.append(f'    {T} ti = {I["fms"]("bi","wr",I["mul"]("br","wi"))};')
    lines.append(f'    {I["st"]("&out_re[0]",  I["add"]("ar","tr"))};')
    lines.append(f'    {I["st"]("&out_im[0]",  I["add"]("ai","ti"))};')
    lines.append(f'    {I["st"]("&out_re[vs]", I["sub"]("ar","tr"))};')
    lines.append(f'    {I["st"]("&out_im[vs]", I["sub"]("ai","ti"))};')
    return '\n'.join(lines)


def gen_t1sv_dif(isa_name, direction):
    """t1sv DIF: butterfly first, then twiddle output[1]. Stride-vector."""
    I = ISA[isa_name]
    T = I['T']
    fwd = (direction == 'fwd')
    lines = []
    lines.append(f'    {T} ar = {I["ld"]("&in_re[0]")},  ai = {I["ld"]("&in_im[0]")};')
    lines.append(f'    {T} br = {I["ld"]("&in_re[vs]")}, bi = {I["ld"]("&in_im[vs]")};')
    lines.append(f'    {T} sr = {I["add"]("ar","br")}, si = {I["add"]("ai","bi")};')
    lines.append(f'    {T} dr = {I["sub"]("ar","br")}, di = {I["sub"]("ai","bi")};')
    lines.append(f'    {T} wr = {I["ld"]("&tw_re[0]")},  wi = {I["ld"]("&tw_im[0]")};')
    lines.append(f'    {I["st"]("&out_re[0]", "sr")};')
    lines.append(f'    {I["st"]("&out_im[0]", "si")};')
    if fwd:
        lines.append(f'    {I["st"]("&out_re[vs]", I["fms"]("dr","wr",I["mul"]("di","wi")))};')
        lines.append(f'    {I["st"]("&out_im[vs]", I["fma"]("dr","wi",I["mul"]("di","wr")))};')
    else:
        lines.append(f'    {I["st"]("&out_re[vs]", I["fma"]("dr","wr",I["mul"]("di","wi")))};')
        lines.append(f'    {I["st"]("&out_im[vs]", I["fms"]("di","wr",I["mul"]("dr","wi")))};')
    return '\n'.join(lines)


def gen_file(isa_name):
    I = ISA[isa_name]
    T, VL = I['T'], I['VL']
    is_scalar = (isa_name == 'scalar')
    guard = f'FFT_RADIX2_{isa_name.upper()}_H'

    parts = []
    parts.append(f'/**')
    parts.append(f' * @file fft_radix2_{isa_name}.h')
    parts.append(f' * @brief DFT-2 {isa_name.upper()} codelets')
    parts.append(f' *')
    parts.append(f' * t2 codelets: loop over K, elements at stride K')
    parts.append(f' *   notw/tw_dit/tw_dif (K parameter = stride = loop bound)')
    if not is_scalar:
        parts.append(f' * sv codelets: no loop, elements at stride vs (SIMD-width)')
        parts.append(f' *   n1sv/t1sv_dit/t1sv_dif (vs parameter = element stride)')
        parts.append(f' *   Executor calls K/VL times, advancing pointers by VL={VL}.')
    parts.append(f' * CT codelets: n1 (stride is/os), t1_dit, t1_dif, n1_ovs')
    parts.append(f' *')
    parts.append(f' * Generated by gen_radix2.py')
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

    # ── t2 codelets (current: loop over K, stride K) ──
    t2_generators = [
        ('notw', gen_notw, False),
        ('tw_dit', gen_dit_tw, True),
        ('tw_dif', gen_dif_tw, True),
    ]

    parts.append(f'/* === t2 codelets: loop over K, elements at stride K === */')
    parts.append(f'')

    for kind, gen_fn, has_tw in t2_generators:
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

    # ── sv codelets (new: no loop, elements at stride vs) ──
    if not is_scalar:
        sv_generators = [
            ('n1sv', gen_n1sv, False),
            ('t1sv_dit', gen_t1sv_dit, True),
            ('t1sv_dif', gen_t1sv_dif, True),
        ]

        parts.append(f'/* === sv codelets: no loop, elements at stride vs === */')
        parts.append(f'/* Executor calls K/VL times, advancing base pointers by VL={VL}. */')
        parts.append(f'')

        for kind, gen_fn, has_tw in sv_generators:
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
                parts.append(f'    size_t vs)')
                parts.append(f'{{')
                parts.append(gen_fn(isa_name, direction))
                parts.append(f'}}')
                parts.append(f'')

    # ── CT codelets: n1, t1_dit, t1_dif, n1_ovs ──
    parts.append(f'/* === CT codelets: stride-based n1 + t1 for executor === */')
    parts.append(f'')

    for direction in ['fwd', 'bwd']:
        fwd = (direction == 'fwd')

        # ── Scalar t1 DIT ──
        if is_scalar:
            if fwd:
                tw_r = 'r1r*wr - r1i*wi'; tw_i = 'r1r*wi + r1i*wr'
            else:
                tw_r = 'r1r*wr + r1i*wi'; tw_i = 'r1i*wr - r1r*wi'
            parts.append(f'''static inline void
radix2_t1_dit_{direction}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t mb, size_t me, size_t ms)
{{
    for (size_t m = mb; m < me; m++) {{
        const double x0r = rio_re[m*ms + 0*ios], x0i = rio_im[m*ms + 0*ios];
        const double r1r = rio_re[m*ms + 1*ios], r1i = rio_im[m*ms + 1*ios];
        const double wr = W_re[0*me + m], wi = W_im[0*me + m];
        const double x1r = {tw_r}, x1i = {tw_i};
        rio_re[m*ms + 0*ios] = x0r + x1r; rio_im[m*ms + 0*ios] = x0i + x1i;
        rio_re[m*ms + 1*ios] = x0r - x1r; rio_im[m*ms + 1*ios] = x0i - x1i;
    }}
}}''')
            parts.append(f'')

            # ── Scalar t1 DIF ──
            if fwd:
                tw_r = 'dr*wr - di*wi'; tw_i = 'dr*wi + di*wr'
            else:
                tw_r = 'dr*wr + di*wi'; tw_i = 'di*wr - dr*wi'
            parts.append(f'''static inline void
radix2_t1_dif_{direction}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t mb, size_t me, size_t ms)
{{
    for (size_t m = mb; m < me; m++) {{
        const double x0r = rio_re[m*ms + 0*ios], x0i = rio_im[m*ms + 0*ios];
        const double x1r = rio_re[m*ms + 1*ios], x1i = rio_im[m*ms + 1*ios];
        const double wr = W_re[0*me + m], wi = W_im[0*me + m];
        const double dr = x0r - x1r, di = x0i - x1i;
        rio_re[m*ms + 0*ios] = x0r + x1r; rio_im[m*ms + 0*ios] = x0i + x1i;
        rio_re[m*ms + 1*ios] = {tw_r}; rio_im[m*ms + 1*ios] = {tw_i};
    }}
}}''')
            parts.append(f'')

        # ── SIMD: n1, n1_ovs, t1_dit, t1_dif ──
        if not is_scalar:
            p = {'avx2': '_mm256', 'avx512': '_mm512'}[isa_name]
            ADD = f'{p}_add_pd'
            SUB = f'{p}_sub_pd'
            MUL = f'{p}_mul_pd'
            FNMA = f'{p}_fnmadd_pd'
            FMA = f'{p}_fmadd_pd'
            FMS = f'{p}_fmsub_pd'

            if fwd:
                def cmul_v(xr, xi, wr, wi):
                    return (f'{FMS}({xr},{wr},{MUL}({xi},{wi}))',
                            f'{FMA}({xr},{wi},{MUL}({xi},{wr}))')
            else:
                def cmul_v(xr, xi, wr, wi):
                    return (f'{FMA}({xi},{wi},{MUL}({xr},{wr}))',
                            f'{FNMA}({xr},{wi},{MUL}({xi},{wr}))')

            target = I['target']

            # ── SIMD n1: reads at stride is, writes at stride os ──
            parts.append(f'''{target}
static inline void
radix2_n1_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t is, size_t os, size_t vl)
{{
    for (size_t k = 0; k < vl; k += {VL}) {{
        const {T} ar = LD(&in_re[0*is+k]), ai = LD(&in_im[0*is+k]);
        const {T} br = LD(&in_re[1*is+k]), bi = LD(&in_im[1*is+k]);
        ST(&out_re[0*os+k], {ADD}(ar, br)); ST(&out_im[0*os+k], {ADD}(ai, bi));
        ST(&out_re[1*os+k], {SUB}(ar, br)); ST(&out_im[1*os+k], {SUB}(ai, bi));
    }}
}}''')
            parts.append(f'')

            # ── SIMD n1_ovs: 2 bins = all scatter (no 4x4 groups) ──
            parts.append(f'''{target}
static inline void
radix2_n1_ovs_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t is, size_t os, size_t vl, size_t ovs)
{{
    for (size_t k = 0; k < vl; k += {VL}) {{
        const {T} ar = LD(&in_re[0*is+k]), ai = LD(&in_im[0*is+k]);
        const {T} br = LD(&in_re[1*is+k]), bi = LD(&in_im[1*is+k]);
        const {T} y0r = {ADD}(ar, br), y0i = {ADD}(ai, bi);
        const {T} y1r = {SUB}(ar, br), y1i = {SUB}(ai, bi);''')
            if isa_name == 'avx2':
                parts.append(f'''        /* 2 bins: extract + scatter */
        {{ __m128d lo0=_mm256_castpd256_pd128(y0r), hi0=_mm256_extractf128_pd(y0r,1);
          _mm_storel_pd(&out_re[(k+0)*ovs+os*0], lo0);
          _mm_storeh_pd(&out_re[(k+1)*ovs+os*0], lo0);
          _mm_storel_pd(&out_re[(k+2)*ovs+os*0], hi0);
          _mm_storeh_pd(&out_re[(k+3)*ovs+os*0], hi0); }}
        {{ __m128d lo0=_mm256_castpd256_pd128(y0i), hi0=_mm256_extractf128_pd(y0i,1);
          _mm_storel_pd(&out_im[(k+0)*ovs+os*0], lo0);
          _mm_storeh_pd(&out_im[(k+1)*ovs+os*0], lo0);
          _mm_storel_pd(&out_im[(k+2)*ovs+os*0], hi0);
          _mm_storeh_pd(&out_im[(k+3)*ovs+os*0], hi0); }}
        {{ __m128d lo1=_mm256_castpd256_pd128(y1r), hi1=_mm256_extractf128_pd(y1r,1);
          _mm_storel_pd(&out_re[(k+0)*ovs+os*1], lo1);
          _mm_storeh_pd(&out_re[(k+1)*ovs+os*1], lo1);
          _mm_storel_pd(&out_re[(k+2)*ovs+os*1], hi1);
          _mm_storeh_pd(&out_re[(k+3)*ovs+os*1], hi1); }}
        {{ __m128d lo1=_mm256_castpd256_pd128(y1i), hi1=_mm256_extractf128_pd(y1i,1);
          _mm_storel_pd(&out_im[(k+0)*ovs+os*1], lo1);
          _mm_storeh_pd(&out_im[(k+1)*ovs+os*1], lo1);
          _mm_storel_pd(&out_im[(k+2)*ovs+os*1], hi1);
          _mm_storeh_pd(&out_im[(k+3)*ovs+os*1], hi1); }}''')
            else:  # avx512
                for b in range(2):
                    yv = f'y{b}'
                    for comp, arr in [('r', 'out_re'), ('i', 'out_im')]:
                        for j in range(VL):
                            parts.append(f'        {arr}[(k+{j})*ovs+os*{b}] = (({T}){yv}{comp})[{j}];')
            parts.append(f'    }}')
            parts.append(f'}}')
            parts.append(f'')

            # ── SIMD t1 DIT: twiddle x1, then butterfly ──
            vc1r, vc1i = cmul_v('r1r', 'r1i', 'wr', 'wi')
            parts.append(f'''{target}
static inline void
radix2_t1_dit_{direction}_{isa_name}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{{
    for (size_t m = 0; m < me; m += {VL}) {{
        const {T} x0r = LD(&rio_re[m + 0*ios]), x0i = LD(&rio_im[m + 0*ios]);
        const {T} r1r = LD(&rio_re[m + 1*ios]), r1i = LD(&rio_im[m + 1*ios]);
        const {T} wr = LD(&W_re[0*me+m]), wi = LD(&W_im[0*me+m]);
        const {T} x1r = {vc1r}, x1i = {vc1i};
        ST(&rio_re[m + 0*ios], {ADD}(x0r, x1r)); ST(&rio_im[m + 0*ios], {ADD}(x0i, x1i));
        ST(&rio_re[m + 1*ios], {SUB}(x0r, x1r)); ST(&rio_im[m + 1*ios], {SUB}(x0i, x1i));
    }}
}}''')
            parts.append(f'')

            # ── SIMD t1 DIF: butterfly, then twiddle y1 ──
            vd1r, vd1i = cmul_v('dr', 'di', 'wr', 'wi')
            parts.append(f'''{target}
static inline void
radix2_t1_dif_{direction}_{isa_name}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{{
    for (size_t m = 0; m < me; m += {VL}) {{
        const {T} x0r = LD(&rio_re[m + 0*ios]), x0i = LD(&rio_im[m + 0*ios]);
        const {T} x1r = LD(&rio_re[m + 1*ios]), x1i = LD(&rio_im[m + 1*ios]);
        const {T} wr = LD(&W_re[0*me+m]), wi = LD(&W_im[0*me+m]);
        const {T} dr = {SUB}(x0r, x1r), di = {SUB}(x0i, x1i);
        ST(&rio_re[m + 0*ios], {ADD}(x0r, x1r)); ST(&rio_im[m + 0*ios], {ADD}(x0i, x1i));
        ST(&rio_re[m + 1*ios], {vd1r}); ST(&rio_im[m + 1*ios], {vd1i});
    }}
}}''')
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
