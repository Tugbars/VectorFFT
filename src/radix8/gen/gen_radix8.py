#!/usr/bin/env python3
"""
gen_radix8.py — DFT-8 codelets for VectorFFT (v2, correct formulas)

Algorithm: DFT-8 = DFT-4(evens) + W8-combine(odds)

DFT-4 forward on {a0,a1,a2,a3}:
  p = a0+a2, q = a0-a2, r = a1+a3, s = a1-a3
  A[0] = p+r
  A[1] = (qr+si) + i*(qi-sr)     [×(-i) on s]
  A[2] = p-r
  A[3] = (qr-si) + i*(qi+sr)     [×(+i) on s]

W8 combine (forward, c = √2/2):
  W8^0 = 1,  W8^1 = c-ci,  W8^2 = -i,  W8^3 = -c-ci
  W8^k * B[k] applied, then Y[k] = A[k] + tw, Y[k+4] = A[k] - tw

Only 1 constant: c = √2/2 = 0.707106781186547524...
"""

import sys, math

C = math.sqrt(2.0) / 2.0

def fmt(val):
    return f'{val:.48f}'.rstrip('0')

ISA = {
    'avx512': {'VL':8,'V':'__m512d','p':'_mm512',
        'target':'__attribute__((target("avx512f,fma")))',
        'guard':'__AVX512F__',
        'ld_macro':'#ifndef R8_512_LD\n#define R8_512_LD(p) _mm512_loadu_pd(p)\n#define R8_512_ST(p,v) _mm512_storeu_pd((p),(v))\n#endif\n#define LD R8_512_LD\n#define ST R8_512_ST'},
    'avx2': {'VL':4,'V':'__m256d','p':'_mm256',
        'target':'__attribute__((target("avx2,fma")))',
        'guard':'__AVX2__',
        'ld_macro':'#ifndef R8_256_LD\n#define R8_256_LD(p) _mm256_loadu_pd(p)\n#define R8_256_ST(p,v) _mm256_storeu_pd((p),(v))\n#endif\n#define LD R8_256_LD\n#define ST R8_256_ST'},
    'scalar': {'VL':1,'V':'double','p':'',
        'target':'','guard':None,'ld_macro':''},
}


def gen_scalar_body(direction, twiddled):
    """Generate the scalar loop body for DFT-8."""
    fwd = direction == 'fwd'
    lines = []
    lines.append(f'    const double c = {fmt(C)};')
    lines.append('')
    lines.append('    for (size_t k = 0; k < K; k++) {')

    # Load
    if twiddled:
        lines.append('        const double x0r = in_re[k], x0i = in_im[k];')
        for n in range(1, 8):
            lines.append(f'        const double r{n}r = in_re[{n}*K+k], r{n}i = in_im[{n}*K+k];')
            lines.append(f'        const double tw{n}r = tw_re[{n-1}*K+k], tw{n}i = tw_im[{n-1}*K+k];')
            if fwd:
                lines.append(f'        const double x{n}r = r{n}r*tw{n}r - r{n}i*tw{n}i;')
                lines.append(f'        const double x{n}i = r{n}r*tw{n}i + r{n}i*tw{n}r;')
            else:
                lines.append(f'        const double x{n}r = r{n}r*tw{n}r + r{n}i*tw{n}i;')
                lines.append(f'        const double x{n}i = -r{n}r*tw{n}i + r{n}i*tw{n}r;')
    else:
        for n in range(8):
            lines.append(f'        const double x{n}r = in_re[{n}*K+k], x{n}i = in_im[{n}*K+k];')

    lines.append('')
    lines.append('        /* DFT-4 of evens: x0, x2, x4, x6 */')
    lines.append('        const double ep_r = x0r+x4r, eq_r = x0r-x4r, er_r = x2r+x6r, es_r = x2r-x6r;')
    lines.append('        const double ep_i = x0i+x4i, eq_i = x0i-x4i, er_i = x2i+x6i, es_i = x2i-x6i;')

    lines.append('        const double A0r = ep_r+er_r, A0i = ep_i+er_i;')
    lines.append('        const double A2r = ep_r-er_r, A2i = ep_i-er_i;')
    if fwd:
        lines.append('        const double A1r = eq_r+es_i, A1i = eq_i-es_r;')
        lines.append('        const double A3r = eq_r-es_i, A3i = eq_i+es_r;')
    else:
        lines.append('        const double A1r = eq_r-es_i, A1i = eq_i+es_r;')
        lines.append('        const double A3r = eq_r+es_i, A3i = eq_i-es_r;')

    lines.append('')
    lines.append('        /* DFT-4 of odds: x1, x3, x5, x7 */')
    lines.append('        const double op_r = x1r+x5r, oq_r = x1r-x5r, or_r = x3r+x7r, os_r = x3r-x7r;')
    lines.append('        const double op_i = x1i+x5i, oq_i = x1i-x5i, or_i = x3i+x7i, os_i = x3i-x7i;')

    lines.append('        const double B0r = op_r+or_r, B0i = op_i+or_i;')
    lines.append('        const double B2r = op_r-or_r, B2i = op_i-or_i;')
    if fwd:
        lines.append('        const double B1r = oq_r+os_i, B1i = oq_i-os_r;')
        lines.append('        const double B3r = oq_r-os_i, B3i = oq_i+os_r;')
    else:
        lines.append('        const double B1r = oq_r-os_i, B1i = oq_i+os_r;')
        lines.append('        const double B3r = oq_r+os_i, B3i = oq_i-os_r;')

    lines.append('')
    lines.append('        /* W8 combine */')
    # k=0: tw = B[0]
    lines.append('        out_re[0*K+k] = A0r + B0r; out_im[0*K+k] = A0i + B0i;')
    lines.append('        out_re[4*K+k] = A0r - B0r; out_im[4*K+k] = A0i - B0i;')

    # k=1: tw = W8^1 * B[1]
    # fwd: W8^1 = c-ci → (c-ci)(Br+Bi*i) = c(Br+Bi) + i*c(Bi-Br)
    # bwd: W8^1 = c+ci → (c+ci)(Br+Bi*i) = c(Br-Bi) + i*c(Br+Bi)
    if fwd:
        lines.append('        const double w1r = c*(B1r+B1i), w1i = c*(B1i-B1r);')
    else:
        lines.append('        const double w1r = c*(B1r-B1i), w1i = c*(B1r+B1i);')
    lines.append('        out_re[1*K+k] = A1r + w1r; out_im[1*K+k] = A1i + w1i;')
    lines.append('        out_re[5*K+k] = A1r - w1r; out_im[5*K+k] = A1i - w1i;')

    # k=2: tw = W8^2 * B[2]
    # fwd: W8^2 = -i → -i*(Br+Bi*i) = Bi - i*Br
    # bwd: W8^2 = +i → i*(Br+Bi*i) = -Bi + i*Br
    if fwd:
        lines.append('        out_re[2*K+k] = A2r + B2i; out_im[2*K+k] = A2i - B2r;')
        lines.append('        out_re[6*K+k] = A2r - B2i; out_im[6*K+k] = A2i + B2r;')
    else:
        lines.append('        out_re[2*K+k] = A2r - B2i; out_im[2*K+k] = A2i + B2r;')
        lines.append('        out_re[6*K+k] = A2r + B2i; out_im[6*K+k] = A2i - B2r;')

    # k=3: tw = W8^3 * B[3]
    # fwd: W8^3 = -c-ci → (-c-ci)(Br+Bi*i) = -c(Br-Bi) + i*(-c(Br+Bi))
    # bwd: W8^3 = -c+ci → (-c+ci)(Br+Bi*i) = -c(Br+Bi) + i*c(Br-Bi)
    #                                         Actually: (-c+ci)(Br+Bi*i) = -cBr + cBi*i^2 + ci*Br + ci*Bi*i
    #                                         = -cBr - cBi + i*(cBr + ... )
    # Let me redo: (-c+ci)(a+bi) = -ca + cbi + cai + cb*i^2 = -ca - cb + i(cb + ca - ... )
    # = -ca - cb + i(ca - cb)  ... no.
    # (-c+ci)(a+bi) = -ca + (-c)(bi) + (ci)(a) + (ci)(bi)
    #               = -ca - cbi + cai + cbi^2
    #               = -ca - cb + i(-cb + ca)
    #               = -c(a+b) + ic(a-b)
    if fwd:
        lines.append('        const double w3r = -c*(B3r-B3i), w3i = -c*(B3r+B3i);')
    else:
        lines.append('        const double w3r = -c*(B3r+B3i), w3i = c*(B3r-B3i);')
    lines.append('        out_re[3*K+k] = A3r + w3r; out_im[3*K+k] = A3i + w3i;')
    lines.append('        out_re[7*K+k] = A3r - w3r; out_im[7*K+k] = A3i - w3i;')

    lines.append('    }')
    return '\n'.join(lines)


def gen_simd_body(isa_name, direction, twiddled):
    I = ISA[isa_name]
    V, p = I['V'], I['p']
    add = f'{p}_add_pd'
    sub = f'{p}_sub_pd'
    mul = f'{p}_mul_pd'
    fma = f'{p}_fmadd_pd'
    fnma = f'{p}_fnmadd_pd'
    set1 = f'{p}_set1_pd'
    fwd = direction == 'fwd'

    lines = []
    lines.append(f'    const {V} vc = {set1}({fmt(C)});')
    lines.append(f'    const {V} vnc = {set1}({fmt(-C)});')
    lines.append('')
    lines.append('    for (size_t k = 0; k < K; k += VL) {')

    # Load + optional twiddle multiply
    if twiddled:
        lines.append(f'        const {V} x0r = LD(&in_re[k]), x0i = LD(&in_im[k]);')
        for n in range(1, 8):
            lines.append(f'        const {V} r{n}r = LD(&in_re[{n}*K+k]), r{n}i = LD(&in_im[{n}*K+k]);')
            lines.append(f'        const {V} tw{n}r = LD(&tw_re[{n-1}*K+k]), tw{n}i = LD(&tw_im[{n-1}*K+k]);')
            if fwd:
                lines.append(f'        const {V} x{n}r = {fnma}(r{n}i,tw{n}i,{mul}(r{n}r,tw{n}r));')
                lines.append(f'        const {V} x{n}i = {fma}(r{n}r,tw{n}i,{mul}(r{n}i,tw{n}r));')
            else:
                lines.append(f'        const {V} x{n}r = {fma}(r{n}i,tw{n}i,{mul}(r{n}r,tw{n}r));')
                lines.append(f'        const {V} x{n}i = {fnma}(r{n}r,tw{n}i,{mul}(r{n}i,tw{n}r));')
    else:
        for n in range(8):
            lines.append(f'        const {V} x{n}r = LD(&in_re[{n}*K+k]), x{n}i = LD(&in_im[{n}*K+k]);')

    lines.append('')
    lines.append('        /* DFT-4 of evens: x0, x2, x4, x6 */')
    lines.append(f'        const {V} epr = {add}(x0r,x4r), eqr = {sub}(x0r,x4r);')
    lines.append(f'        const {V} epi = {add}(x0i,x4i), eqi = {sub}(x0i,x4i);')
    lines.append(f'        const {V} err = {add}(x2r,x6r), esr = {sub}(x2r,x6r);')
    lines.append(f'        const {V} eri = {add}(x2i,x6i), esi = {sub}(x2i,x6i);')
    lines.append(f'        const {V} A0r = {add}(epr,err), A0i = {add}(epi,eri);')
    lines.append(f'        const {V} A2r = {sub}(epr,err), A2i = {sub}(epi,eri);')
    if fwd:
        lines.append(f'        const {V} A1r = {add}(eqr,esi), A1i = {sub}(eqi,esr);')
        lines.append(f'        const {V} A3r = {sub}(eqr,esi), A3i = {add}(eqi,esr);')
    else:
        lines.append(f'        const {V} A1r = {sub}(eqr,esi), A1i = {add}(eqi,esr);')
        lines.append(f'        const {V} A3r = {add}(eqr,esi), A3i = {sub}(eqi,esr);')

    lines.append('')
    lines.append('        /* DFT-4 of odds: x1, x3, x5, x7 */')
    lines.append(f'        const {V} opr = {add}(x1r,x5r), oqr = {sub}(x1r,x5r);')
    lines.append(f'        const {V} opi = {add}(x1i,x5i), oqi = {sub}(x1i,x5i);')
    lines.append(f'        const {V} orr = {add}(x3r,x7r), osr = {sub}(x3r,x7r);')
    lines.append(f'        const {V} ori = {add}(x3i,x7i), osi = {sub}(x3i,x7i);')
    lines.append(f'        const {V} B0r = {add}(opr,orr), B0i = {add}(opi,ori);')
    lines.append(f'        const {V} B2r = {sub}(opr,orr), B2i = {sub}(opi,ori);')
    if fwd:
        lines.append(f'        const {V} B1r = {add}(oqr,osi), B1i = {sub}(oqi,osr);')
        lines.append(f'        const {V} B3r = {sub}(oqr,osi), B3i = {add}(oqi,osr);')
    else:
        lines.append(f'        const {V} B1r = {sub}(oqr,osi), B1i = {add}(oqi,osr);')
        lines.append(f'        const {V} B3r = {add}(oqr,osi), B3i = {sub}(oqi,osr);')

    lines.append('')
    lines.append('        /* W8 combine and output */')
    # k=0
    lines.append(f'        ST(&out_re[0*K+k], {add}(A0r,B0r)); ST(&out_im[0*K+k], {add}(A0i,B0i));')
    lines.append(f'        ST(&out_re[4*K+k], {sub}(A0r,B0r)); ST(&out_im[4*K+k], {sub}(A0i,B0i));')

    # k=1: fwd W=c-ci → tw = c*(B1r+B1i) + ic*(B1i-B1r)
    #       bwd W=c+ci → tw = c*(B1r-B1i) + ic*(B1r+B1i)
    if fwd:
        lines.append(f'        const {V} t1r = {mul}(vc, {add}(B1r,B1i));')
        lines.append(f'        const {V} t1i = {mul}(vc, {sub}(B1i,B1r));')
    else:
        lines.append(f'        const {V} t1r = {mul}(vc, {sub}(B1r,B1i));')
        lines.append(f'        const {V} t1i = {mul}(vc, {add}(B1r,B1i));')
    lines.append(f'        ST(&out_re[1*K+k], {add}(A1r,t1r)); ST(&out_im[1*K+k], {add}(A1i,t1i));')
    lines.append(f'        ST(&out_re[5*K+k], {sub}(A1r,t1r)); ST(&out_im[5*K+k], {sub}(A1i,t1i));')

    # k=2: fwd → tw = B2i - iB2r
    if fwd:
        lines.append(f'        ST(&out_re[2*K+k], {add}(A2r,B2i)); ST(&out_im[2*K+k], {sub}(A2i,B2r));')
        lines.append(f'        ST(&out_re[6*K+k], {sub}(A2r,B2i)); ST(&out_im[6*K+k], {add}(A2i,B2r));')
    else:
        lines.append(f'        ST(&out_re[2*K+k], {sub}(A2r,B2i)); ST(&out_im[2*K+k], {add}(A2i,B2r));')
        lines.append(f'        ST(&out_re[6*K+k], {add}(A2r,B2i)); ST(&out_im[6*K+k], {sub}(A2i,B2r));')

    # k=3: fwd W=-c-ci → tw = -c(Br-Bi) - ic(Br+Bi)
    #       bwd W=-c+ci → tw = -c(Br+Bi) + ic(Br-Bi)
    if fwd:
        lines.append(f'        const {V} t3r = {mul}(vnc, {sub}(B3r,B3i));')
        lines.append(f'        const {V} t3i = {mul}(vnc, {add}(B3r,B3i));')
    else:
        lines.append(f'        const {V} t3r = {mul}(vnc, {add}(B3r,B3i));')
        lines.append(f'        const {V} t3i = {mul}(vc, {sub}(B3r,B3i));')
    lines.append(f'        ST(&out_re[3*K+k], {add}(A3r,t3r)); ST(&out_im[3*K+k], {add}(A3i,t3i));')
    lines.append(f'        ST(&out_re[7*K+k], {sub}(A3r,t3r)); ST(&out_im[7*K+k], {sub}(A3i,t3i));')

    lines.append('    }')
    return '\n'.join(lines)


def gen_file(isa_name):
    I = ISA[isa_name]
    guard = f'FFT_RADIX8_{isa_name.upper()}_H'
    is_scalar = isa_name == 'scalar'

    parts = []
    parts.append(f'''/**
 * @file fft_radix8_{isa_name}.h
 * @brief DFT-8 {isa_name.upper()} codelets (notw + twiddled)
 *
 * DFT-8 = 2×DFT-4 + W8 combine. Only 1 constant: √2/2.
 * ~52 adds + 4 muls (scalar) per DFT-8.
 *
 * Twiddle table: tw_re[(n-1)*K+k] n=1..7 → 7*K doubles per component.
 *
 * Generated by gen_radix8.py — do not edit.
 */

#ifndef {guard}
#define {guard}

#include <stddef.h>''')

    if not is_scalar:
        parts.append('#include <immintrin.h>')
        parts.append('')
        parts.append(I['ld_macro'])
        parts.append(f'#define VL {I["VL"]}')

    for tw in [False, True]:
        for direction in ['fwd', 'bwd']:
            kind = 'tw' if tw else 'notw'
            tw_params = (',\n    const double * __restrict__ tw_re, const double * __restrict__ tw_im'
                        if tw else '')

            parts.append(f'''
{I["target"]}
static inline void
radix8_{kind}_dit_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im{tw_params},
    size_t K)
{{''')
            if is_scalar:
                parts.append(gen_scalar_body(direction, tw))
            else:
                parts.append(gen_simd_body(isa_name, direction, tw))
            parts.append('}')

    if not is_scalar:
        parts.append('\n#undef LD\n#undef ST\n#undef VL')

    parts.append(f'\n#endif /* {guard} */')
    return '\n'.join(parts)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ISA:
        print(f'Usage: {sys.argv[0]} avx512|avx2|scalar', file=sys.stderr)
        sys.exit(1)
    print(gen_file(sys.argv[1]))
