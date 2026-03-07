#!/usr/bin/env python3
"""
gen_radix4.py — DFT-4 standalone codelets for VectorFFT

DFT-4 = 2×DFT-2 + ×(-j) combine. 16 adds, 0 muls, 0 constants.
Forward/backward differ only in the ×j sign on the odd-diff term.

Twiddle table: 3*K doubles per component (W^1, W^2, W^3).

Usage: python3 gen_radix4.py avx512|avx2|scalar
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

def gen_scalar_notw(direction):
    # Forward: X[1].re = t1_re + t3_im,  X[1].im = t1_im - t3_re
    #          X[3].re = t1_re - t3_im,  X[3].im = t1_im + t3_re
    # Backward: swap X[1] and X[3] roles (j → -j)
    fwd = direction == 'fwd'
    if fwd:
        r1 = 't1r + t3i'; i1 = 't1i - t3r'
        r3 = 't1r - t3i'; i3 = 't1i + t3r'
    else:
        r1 = 't1r - t3i'; i1 = 't1i + t3r'
        r3 = 't1r + t3i'; i3 = 't1i - t3r'
    return f"""    for (size_t k = 0; k < K; k++) {{
        /* Even: x[0] ± x[2] */
        const double t0r = in_re[0*K+k] + in_re[2*K+k];
        const double t0i = in_im[0*K+k] + in_im[2*K+k];
        const double t1r = in_re[0*K+k] - in_re[2*K+k];
        const double t1i = in_im[0*K+k] - in_im[2*K+k];
        /* Odd: x[1] ± x[3] */
        const double t2r = in_re[1*K+k] + in_re[3*K+k];
        const double t2i = in_im[1*K+k] + in_im[3*K+k];
        const double t3r = in_re[1*K+k] - in_re[3*K+k];
        const double t3i = in_im[1*K+k] - in_im[3*K+k];
        /* Combine */
        out_re[0*K+k] = t0r + t2r;
        out_im[0*K+k] = t0i + t2i;
        out_re[2*K+k] = t0r - t2r;
        out_im[2*K+k] = t0i - t2i;
        out_re[1*K+k] = {r1};
        out_im[1*K+k] = {i1};
        out_re[3*K+k] = {r3};
        out_im[3*K+k] = {i3};
    }}"""

def gen_scalar_tw(direction):
    fwd = direction == 'fwd'
    # cmul fwd: xr*wr - xi*wi, xr*wi + xi*wr
    # cmul bwd: xr*wr + xi*wi, -xr*wi + xi*wr
    if fwd:
        cm = lambda vr,vi,wr,wi: (f'{vr}*{wr} - {vi}*{wi}', f'{vr}*{wi} + {vi}*{wr}')
        r1 = 't1r + t3i'; i1 = 't1i - t3r'
        r3 = 't1r - t3i'; i3 = 't1i + t3r'
    else:
        cm = lambda vr,vi,wr,wi: (f'{vr}*{wr} + {vi}*{wi}', f'-{vr}*{wi} + {vi}*{wr}')
        r1 = 't1r - t3i'; i1 = 't1i + t3r'
        r3 = 't1r + t3i'; i3 = 't1i - t3r'
    c1r, c1i = cm('r1r','r1i','w1r','w1i')
    c2r, c2i = cm('r2r','r2i','w2r','w2i')
    c3r, c3i = cm('r3r','r3i','w3r','w3i')
    return f"""    for (size_t k = 0; k < K; k++) {{
        const double x0r = in_re[0*K+k], x0i = in_im[0*K+k];
        const double r1r = in_re[1*K+k], r1i = in_im[1*K+k];
        const double r2r = in_re[2*K+k], r2i = in_im[2*K+k];
        const double r3r = in_re[3*K+k], r3i = in_im[3*K+k];
        const double w1r = tw_re[0*K+k], w1i = tw_im[0*K+k];
        const double w2r = tw_re[1*K+k], w2i = tw_im[1*K+k];
        const double w3r = tw_re[2*K+k], w3i = tw_im[2*K+k];
        const double x1r = {c1r}, x1i = {c1i};
        const double x2r = {c2r}, x2i = {c2i};
        const double x3r = {c3r}, x3i = {c3i};
        const double t0r = x0r + x2r, t0i = x0i + x2i;
        const double t1r = x0r - x2r, t1i = x0i - x2i;
        const double t2r = x1r + x3r, t2i = x1i + x3i;
        const double t3r = x1r - x3r, t3i = x1i - x3i;
        out_re[0*K+k] = t0r + t2r; out_im[0*K+k] = t0i + t2i;
        out_re[2*K+k] = t0r - t2r; out_im[2*K+k] = t0i - t2i;
        out_re[1*K+k] = {r1}; out_im[1*K+k] = {i1};
        out_re[3*K+k] = {r3}; out_im[3*K+k] = {i3};
    }}"""

def gen_simd_notw(isa_name, direction):
    I = ISA[isa_name]
    V, p, VL = I['V'], I['p'], I['VL']
    ADD, SUB = f'{p}_add_pd', f'{p}_sub_pd'
    fwd = direction == 'fwd'
    # Forward: X[1].re = t1r + t3i,  X[1].im = t1i - t3r
    # Backward: swap
    if fwd:
        r1, i1 = f'{ADD}(t1r, t3i)', f'{SUB}(t1i, t3r)'
        r3, i3 = f'{SUB}(t1r, t3i)', f'{ADD}(t1i, t3r)'
    else:
        r1, i1 = f'{SUB}(t1r, t3i)', f'{ADD}(t1i, t3r)'
        r3, i3 = f'{ADD}(t1r, t3i)', f'{SUB}(t1i, t3r)'

    return f"""    for (size_t k = 0; k < K; k += {VL}) {{
        const {V} x0r = R4_LD(&in_re[0*K+k]), x0i = R4_LD(&in_im[0*K+k]);
        const {V} x1r = R4_LD(&in_re[1*K+k]), x1i = R4_LD(&in_im[1*K+k]);
        const {V} x2r = R4_LD(&in_re[2*K+k]), x2i = R4_LD(&in_im[2*K+k]);
        const {V} x3r = R4_LD(&in_re[3*K+k]), x3i = R4_LD(&in_im[3*K+k]);
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&out_re[0*K+k], {ADD}(t0r, t2r)); R4_ST(&out_im[0*K+k], {ADD}(t0i, t2i));
        R4_ST(&out_re[2*K+k], {SUB}(t0r, t2r)); R4_ST(&out_im[2*K+k], {SUB}(t0i, t2i));
        R4_ST(&out_re[1*K+k], {r1}); R4_ST(&out_im[1*K+k], {i1});
        R4_ST(&out_re[3*K+k], {r3}); R4_ST(&out_im[3*K+k], {i3});
    }}"""

def gen_simd_tw(isa_name, direction):
    I = ISA[isa_name]
    V, p, VL = I['V'], I['p'], I['VL']
    ADD, SUB = f'{p}_add_pd', f'{p}_sub_pd'
    MUL = f'{p}_mul_pd'
    FNMA = f'{p}_fnmadd_pd'  # -a*b + c
    FMA  = f'{p}_fmadd_pd'   #  a*b + c
    fwd = direction == 'fwd'
    # fwd cmul: re = fnmadd(xi, wi, mul(xr, wr)), im = fmadd(xr, wi, mul(xi, wr))
    # bwd cmul: re = fmadd(xi, wi, mul(xr, wr)),  im = fnmadd(xr, wi, mul(xi, wr))
    if fwd:
        def cmul(xr,xi,wr,wi): return (f'{FNMA}({xi},{wi},{MUL}({xr},{wr}))',
                                        f'{FMA}({xr},{wi},{MUL}({xi},{wr}))')
        r1, i1 = lambda: f'{ADD}(t1r, t3i)', lambda: f'{SUB}(t1i, t3r)'
        r3, i3 = lambda: f'{SUB}(t1r, t3i)', lambda: f'{ADD}(t1i, t3r)'
    else:
        def cmul(xr,xi,wr,wi): return (f'{FMA}({xi},{wi},{MUL}({xr},{wr}))',
                                        f'{FNMA}({xr},{wi},{MUL}({xi},{wr}))')
        r1, i1 = lambda: f'{SUB}(t1r, t3i)', lambda: f'{ADD}(t1i, t3r)'
        r3, i3 = lambda: f'{ADD}(t1r, t3i)', lambda: f'{SUB}(t1i, t3r)'

    c1r, c1i = cmul('r1r','r1i','w1r','w1i')
    c2r, c2i = cmul('r2r','r2i','w2r','w2i')
    c3r, c3i = cmul('r3r','r3i','w3r','w3i')

    return f"""    for (size_t k = 0; k < K; k += {VL}) {{
        const {V} x0r = R4_LD(&in_re[0*K+k]), x0i = R4_LD(&in_im[0*K+k]);
        const {V} r1r = R4_LD(&in_re[1*K+k]), r1i = R4_LD(&in_im[1*K+k]);
        const {V} r2r = R4_LD(&in_re[2*K+k]), r2i = R4_LD(&in_im[2*K+k]);
        const {V} r3r = R4_LD(&in_re[3*K+k]), r3i = R4_LD(&in_im[3*K+k]);
        const {V} w1r = R4_LD(&tw_re[0*K+k]), w1i = R4_LD(&tw_im[0*K+k]);
        const {V} w2r = R4_LD(&tw_re[1*K+k]), w2i = R4_LD(&tw_im[1*K+k]);
        const {V} w3r = R4_LD(&tw_re[2*K+k]), w3i = R4_LD(&tw_im[2*K+k]);
        const {V} x1r = {c1r}, x1i = {c1i};
        const {V} x2r = {c2r}, x2i = {c2i};
        const {V} x3r = {c3r}, x3i = {c3i};
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&out_re[0*K+k], {ADD}(t0r, t2r)); R4_ST(&out_im[0*K+k], {ADD}(t0i, t2i));
        R4_ST(&out_re[2*K+k], {SUB}(t0r, t2r)); R4_ST(&out_im[2*K+k], {SUB}(t0i, t2i));
        R4_ST(&out_re[1*K+k], {r1()}); R4_ST(&out_im[1*K+k], {i1()});
        R4_ST(&out_re[3*K+k], {r3()}); R4_ST(&out_im[3*K+k], {i3()});
    }}"""

def gen_file(isa_name):
    I = ISA[isa_name]
    guard = f'FFT_RADIX4_{isa_name.upper()}_H'
    is_scalar = isa_name == 'scalar'

    parts = []
    parts.append(f'''/**
 * @file fft_radix4_{isa_name}.h
 * @brief DFT-4 {isa_name.upper()} codelets (notw + twiddled)
 *
 * DFT-4 = 2xDFT-2 + x(-j) combine. 16 adds, 0 muls, 0 constants.
 * Forward/backward differ only in the xj sign on the odd-diff term.
 * Twiddle table: 3*K doubles per component (W^1, W^2, W^3).
 *
 * Generated by gen_radix4.py — do not edit.
 */

#ifndef {guard}
#define {guard}

{I['guard_begin']}

#include <stddef.h>''')

    if not is_scalar:
        parts.append('#include <immintrin.h>')
        parts.append('')
        parts.append(f'#define R4_LD(p) {I["p"]}_loadu_pd(p)')
        parts.append(f'#define R4_ST(p,v) {I["p"]}_storeu_pd((p),(v))')

    for direction in ['fwd', 'bwd']:
        # notw
        parts.append(f'''
{I['target']}
static inline void
radix4_notw_dit_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{{''')
        if is_scalar:
            parts.append(gen_scalar_notw(direction))
        else:
            parts.append(gen_simd_notw(isa_name, direction))
        parts.append('}')

        # tw
        parts.append(f'''
{I['target']}
static inline void
radix4_tw_dit_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{{''')
        if is_scalar:
            parts.append(gen_scalar_tw(direction))
        else:
            parts.append(gen_simd_tw(isa_name, direction))
        parts.append('}')

    if not is_scalar:
        parts.append('\n#undef R4_LD\n#undef R4_ST')

    parts.append(f'\n{I["guard_end"]}')
    parts.append(f'#endif /* {guard} */')
    return '\n'.join(parts)

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ISA:
        print(f'Usage: {sys.argv[0]} avx512|avx2|scalar', file=sys.stderr)
        sys.exit(1)
    print(gen_file(sys.argv[1]))
