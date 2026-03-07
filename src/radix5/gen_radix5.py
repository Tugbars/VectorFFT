#!/usr/bin/env python3
"""
gen_radix5.py — DFT-5 standalone codelets for VectorFFT

DFT-5: 2 conjugate pairs, 4 constants, ~32 ops.
Cross-term sign pattern (forward, verified against FFTW):
  Pair (1,4):  X[1]=p1+Ti-Tu*j,  X[4]=p1-Ti+Tu*j
  Pair (2,3):  X[2]=p2-Tk+Tv*j,  X[3]=p2+Tk-Tv*j
  Where Ti=K1*d1i+K2*d2i, Tk=K1*d2i-K2*d1i, Tu=K1*d1r+K2*d2r, Tv=K1*d2r-K2*d1r

Twiddle table: 4*K doubles per component (W^1..W^4).
Usage: python3 gen_radix5.py avx512|avx2|scalar
"""
import sys

ISA = {
    'avx512': {'VL':8,'V':'__m512d','p':'_mm512',
        'target':'__attribute__((target("avx512f,fma")))',
        'guard_begin':'#if defined(__AVX512F__) || defined(__AVX512F)',
        'guard_end':'#endif /* __AVX512F__ */'},
    'avx2': {'VL':4,'V':'__m256d','p':'_mm256',
        'target':'__attribute__((target("avx2,fma")))',
        'guard_begin':'#ifdef __AVX2__',
        'guard_end':'#endif /* __AVX2__ */'},
    'scalar': {'VL':1,'V':'double','p':'',
        'target':'',
        'guard_begin':'',
        'guard_end':''},
}

def gen_scalar(direction, tw):
    fwd = direction == 'fwd'
    lines = []
    lines.append('    const double K0 = 0.559016994374947424102293417182819058860154590;')
    lines.append('    const double K1 = 0.951056516295153572116439333379382143405698634;')
    lines.append('    const double K2 = 0.587785252292473129168705954639072768597652438;')
    lines.append('    for (size_t k = 0; k < K; k++) {')

    if tw:
        lines.append('        const double x0r = in_re[0*K+k], x0i = in_im[0*K+k];')
        for n in range(1, 5):
            lines.append(f'        double r{n}r = in_re[{n}*K+k], r{n}i = in_im[{n}*K+k];')
        for n in range(1, 5):
            lines.append(f'        const double w{n}r = tw_re[{n-1}*K+k], w{n}i = tw_im[{n-1}*K+k];')
        if fwd:
            cm = lambda vr,vi,wr,wi: (f'{vr}*{wr} - {vi}*{wi}', f'{vr}*{wi} + {vi}*{wr}')
        else:
            cm = lambda vr,vi,wr,wi: (f'{vr}*{wr} + {vi}*{wi}', f'-{vr}*{wi} + {vi}*{wr}')
        for n in range(1, 5):
            cr, ci = cm(f'r{n}r',f'r{n}i',f'w{n}r',f'w{n}i')
            lines.append(f'        const double x{n}r = {cr}, x{n}i = {ci};')
    else:
        for n in range(5):
            lines.append(f'        const double x{n}r = in_re[{n}*K+k], x{n}i = in_im[{n}*K+k];')

    lines.append('        const double s1r = x1r + x4r, s1i = x1i + x4i;')
    lines.append('        const double s2r = x2r + x3r, s2i = x2i + x3i;')
    lines.append('        const double d1r = x1r - x4r, d1i = x1i - x4i;')
    lines.append('        const double d2r = x2r - x3r, d2i = x2i - x3i;')
    lines.append('        out_re[0*K+k] = x0r + s1r + s2r;')
    lines.append('        out_im[0*K+k] = x0i + s1i + s2i;')
    lines.append('        const double t0r = x0r - 0.25*(s1r + s2r);')
    lines.append('        const double t0i = x0i - 0.25*(s1i + s2i);')
    lines.append('        const double t1r = K0*(s1r - s2r);')
    lines.append('        const double t1i = K0*(s1i - s2i);')
    lines.append('        const double p1r = t0r + t1r, p1i = t0i + t1i;')
    lines.append('        const double p2r = t0r - t1r, p2i = t0i - t1i;')

    # Cross terms — forward sign convention
    # Ti, Tu for pair (1,4); Tk, Tv for pair (2,3)
    if fwd:
        lines.append('        const double Ti = K1*d1i + K2*d2i;')
        lines.append('        const double Tu = K1*d1r + K2*d2r;')
        lines.append('        const double Tk = K1*d2i - K2*d1i;')
        lines.append('        const double Tv = K1*d2r - K2*d1r;')
        lines.append('        out_re[1*K+k] = p1r + Ti; out_im[1*K+k] = p1i - Tu;')
        lines.append('        out_re[4*K+k] = p1r - Ti; out_im[4*K+k] = p1i + Tu;')
        lines.append('        out_re[2*K+k] = p2r - Tk; out_im[2*K+k] = p2i + Tv;')
        lines.append('        out_re[3*K+k] = p2r + Tk; out_im[3*K+k] = p2i - Tv;')
    else:
        # Backward: negate exponent → swap signs on all cross terms
        lines.append('        const double Ti = K1*d1i + K2*d2i;')
        lines.append('        const double Tu = K1*d1r + K2*d2r;')
        lines.append('        const double Tk = K1*d2i - K2*d1i;')
        lines.append('        const double Tv = K1*d2r - K2*d1r;')
        lines.append('        out_re[1*K+k] = p1r - Ti; out_im[1*K+k] = p1i + Tu;')
        lines.append('        out_re[4*K+k] = p1r + Ti; out_im[4*K+k] = p1i - Tu;')
        lines.append('        out_re[2*K+k] = p2r + Tk; out_im[2*K+k] = p2i - Tv;')
        lines.append('        out_re[3*K+k] = p2r - Tk; out_im[3*K+k] = p2i + Tv;')
    lines.append('    }')
    return '\n'.join(lines)

def gen_simd(isa_name, direction, tw):
    I = ISA[isa_name]
    V, p, VL = I['V'], I['p'], I['VL']
    ADD, SUB, MUL = f'{p}_add_pd', f'{p}_sub_pd', f'{p}_mul_pd'
    FMA, FNMA = f'{p}_fmadd_pd', f'{p}_fnmadd_pd'
    SET1 = f'{p}_set1_pd'
    fwd = direction == 'fwd'

    lines = []
    lines.append(f'    const {V} cK0 = {SET1}(0.559016994374947424102293417182819058860154590);')
    lines.append(f'    const {V} cK1 = {SET1}(0.951056516295153572116439333379382143405698634);')
    lines.append(f'    const {V} cK2 = {SET1}(0.587785252292473129168705954639072768597652438);')
    lines.append(f'    const {V} cK3 = {SET1}(0.250000000000000000000000000000000000000000000);')
    lines.append(f'    for (size_t k = 0; k < K; k += {VL}) {{')

    if tw:
        lines.append(f'        const {V} x0r = R5_LD(&in_re[0*K+k]), x0i = R5_LD(&in_im[0*K+k]);')
        for n in range(1, 5):
            lines.append(f'        const {V} r{n}r = R5_LD(&in_re[{n}*K+k]), r{n}i = R5_LD(&in_im[{n}*K+k]);')
        for n in range(1, 5):
            lines.append(f'        const {V} w{n}r = R5_LD(&tw_re[{n-1}*K+k]), w{n}i = R5_LD(&tw_im[{n-1}*K+k]);')
        for n in range(1, 5):
            if fwd:
                cr = f'{FNMA}(r{n}i, w{n}i, {MUL}(r{n}r, w{n}r))'
                ci = f'{FMA}(r{n}r, w{n}i, {MUL}(r{n}i, w{n}r))'
            else:
                cr = f'{FMA}(r{n}i, w{n}i, {MUL}(r{n}r, w{n}r))'
                ci = f'{FNMA}(r{n}r, w{n}i, {MUL}(r{n}i, w{n}r))'
            lines.append(f'        const {V} x{n}r = {cr}, x{n}i = {ci};')
    else:
        for n in range(5):
            lines.append(f'        const {V} x{n}r = R5_LD(&in_re[{n}*K+k]), x{n}i = R5_LD(&in_im[{n}*K+k]);')

    lines.append(f'        const {V} s1r = {ADD}(x1r, x4r), s1i = {ADD}(x1i, x4i);')
    lines.append(f'        const {V} s2r = {ADD}(x2r, x3r), s2i = {ADD}(x2i, x3i);')
    lines.append(f'        const {V} d1r = {SUB}(x1r, x4r), d1i = {SUB}(x1i, x4i);')
    lines.append(f'        const {V} d2r = {SUB}(x2r, x3r), d2i = {SUB}(x2i, x3i);')
    lines.append(f'        R5_ST(&out_re[0*K+k], {ADD}(x0r, {ADD}(s1r, s2r)));')
    lines.append(f'        R5_ST(&out_im[0*K+k], {ADD}(x0i, {ADD}(s1i, s2i)));')
    lines.append(f'        const {V} t0r = {FNMA}(cK3, {ADD}(s1r, s2r), x0r);')
    lines.append(f'        const {V} t0i = {FNMA}(cK3, {ADD}(s1i, s2i), x0i);')
    lines.append(f'        const {V} t1r = {MUL}(cK0, {SUB}(s1r, s2r));')
    lines.append(f'        const {V} t1i = {MUL}(cK0, {SUB}(s1i, s2i));')
    lines.append(f'        const {V} p1r = {ADD}(t0r, t1r), p1i = {ADD}(t0i, t1i);')
    lines.append(f'        const {V} p2r = {SUB}(t0r, t1r), p2i = {SUB}(t0i, t1i);')

    # Cross terms: Ti=K1*d1i+K2*d2i, Tu=K1*d1r+K2*d2r, Tk=K1*d2i-K2*d1i, Tv=K1*d2r-K2*d1r
    lines.append(f'        const {V} Ti = {FMA}(cK2, d2i, {MUL}(cK1, d1i));')
    lines.append(f'        const {V} Tu = {FMA}(cK2, d2r, {MUL}(cK1, d1r));')
    lines.append(f'        const {V} Tk = {FNMA}(cK2, d1i, {MUL}(cK1, d2i));')
    lines.append(f'        const {V} Tv = {FNMA}(cK2, d1r, {MUL}(cK1, d2r));')

    if fwd:
        lines.append(f'        R5_ST(&out_re[1*K+k], {ADD}(p1r, Ti)); R5_ST(&out_im[1*K+k], {SUB}(p1i, Tu));')
        lines.append(f'        R5_ST(&out_re[4*K+k], {SUB}(p1r, Ti)); R5_ST(&out_im[4*K+k], {ADD}(p1i, Tu));')
        lines.append(f'        R5_ST(&out_re[2*K+k], {SUB}(p2r, Tk)); R5_ST(&out_im[2*K+k], {ADD}(p2i, Tv));')
        lines.append(f'        R5_ST(&out_re[3*K+k], {ADD}(p2r, Tk)); R5_ST(&out_im[3*K+k], {SUB}(p2i, Tv));')
    else:
        lines.append(f'        R5_ST(&out_re[1*K+k], {SUB}(p1r, Ti)); R5_ST(&out_im[1*K+k], {ADD}(p1i, Tu));')
        lines.append(f'        R5_ST(&out_re[4*K+k], {ADD}(p1r, Ti)); R5_ST(&out_im[4*K+k], {SUB}(p1i, Tu));')
        lines.append(f'        R5_ST(&out_re[2*K+k], {ADD}(p2r, Tk)); R5_ST(&out_im[2*K+k], {SUB}(p2i, Tv));')
        lines.append(f'        R5_ST(&out_re[3*K+k], {SUB}(p2r, Tk)); R5_ST(&out_im[3*K+k], {ADD}(p2i, Tv));')
    lines.append('    }')
    return '\n'.join(lines)

def gen_file(isa_name):
    I = ISA[isa_name]
    guard = f'FFT_RADIX5_{isa_name.upper()}_H'
    is_scalar = isa_name == 'scalar'
    parts = []
    parts.append(f'''/**
 * @file fft_radix5_{isa_name}.h
 * @brief DFT-5 {isa_name.upper()} codelets (notw + twiddled)
 *
 * DFT-5: 2 conjugate pairs, 4 constants, ~32 ops.
 * Twiddle table: 4*K doubles per component (W^1..W^4).
 *
 * Generated by gen_radix5.py — do not edit.
 */

#ifndef {guard}
#define {guard}

{I['guard_begin']}

#include <stddef.h>''')
    if not is_scalar:
        parts.append('#include <immintrin.h>')
        parts.append(f'\n#define R5_LD(p) {I["p"]}_loadu_pd(p)')
        parts.append(f'#define R5_ST(p,v) {I["p"]}_storeu_pd((p),(v))')

    for direction in ['fwd', 'bwd']:
        parts.append(f'''
{I['target']}
static inline void
radix5_notw_dit_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{{''')
        parts.append(gen_scalar(direction, False) if is_scalar else gen_simd(isa_name, direction, False))
        parts.append('}')
        parts.append(f'''
{I['target']}
static inline void
radix5_tw_dit_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{{''')
        parts.append(gen_scalar(direction, True) if is_scalar else gen_simd(isa_name, direction, True))
        parts.append('}')

    if not is_scalar:
        parts.append('\n#undef R5_LD\n#undef R5_ST')
    parts.append(f'\n{I["guard_end"]}')
    parts.append(f'#endif /* {guard} */')
    return '\n'.join(parts)

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ISA:
        print(f'Usage: {sys.argv[0]} avx512|avx2|scalar', file=sys.stderr); sys.exit(1)
    print(gen_file(sys.argv[1]))
