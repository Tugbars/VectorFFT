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

import sys, math, re

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


def gen_scalar_dif_body(direction):
    """DIF: butterfly then post-twiddle outputs 1..7."""
    fwd = direction == 'fwd'
    lines = []
    lines.append(f'    const double c = {fmt(C)};')
    lines.append('')
    lines.append('    for (size_t k = 0; k < K; k++) {')
    for n in range(8):
        lines.append(f'        const double x{n}r = in_re[{n}*K+k], x{n}i = in_im[{n}*K+k];')

    # Same butterfly as notw
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

    # W8 combine into y0..y7 (same as notw)
    lines.append('        double y0r = A0r+B0r, y0i = A0i+B0i;')
    lines.append('        double y4r = A0r-B0r, y4i = A0i-B0i;')
    if fwd:
        lines.append(f'        double y1r = A1r+c*(B1r+B1i), y1i = A1i+c*(B1i-B1r);')
        lines.append(f'        double y5r = A1r-c*(B1r+B1i), y5i = A1i-c*(B1i-B1r);')
        lines.append('        double y2r = A2r+B2i, y2i = A2i-B2r;')
        lines.append('        double y6r = A2r-B2i, y6i = A2i+B2r;')
        lines.append(f'        double y3r = A3r-c*(B3r-B3i), y3i = A3i-c*(B3r+B3i);')
        lines.append(f'        double y7r = A3r+c*(B3r-B3i), y7i = A3i+c*(B3r+B3i);')
    else:
        lines.append(f'        double y1r = A1r+c*(B1r-B1i), y1i = A1i+c*(B1r+B1i);')
        lines.append(f'        double y5r = A1r-c*(B1r-B1i), y5i = A1i-c*(B1r+B1i);')
        lines.append('        double y2r = A2r-B2i, y2i = A2i+B2r;')
        lines.append('        double y6r = A2r+B2i, y6i = A2i-B2r;')
        lines.append(f'        double y3r = A3r-c*(B3r+B3i), y3i = A3i+c*(B3r-B3i);')
        lines.append(f'        double y7r = A3r+c*(B3r+B3i), y7i = A3i-c*(B3r-B3i);')

    # Store y0 (no twiddle), post-twiddle y1..y7
    lines.append('        out_re[0*K+k] = y0r; out_im[0*K+k] = y0i;')
    for n in range(1, 8):
        lines.append(f'        {{ double wr = tw_re[{n-1}*K+k], wi = tw_im[{n-1}*K+k], tr = y{n}r;')
        if fwd:
            lines.append(f'          out_re[{n}*K+k] = tr*wr - y{n}i*wi; out_im[{n}*K+k] = tr*wi + y{n}i*wr; }}')
        else:
            lines.append(f'          out_re[{n}*K+k] = tr*wr + y{n}i*wi; out_im[{n}*K+k] = y{n}i*wr - tr*wi; }}')

    lines.append('    }')
    return '\n'.join(lines)


def gen_simd_dif_body(isa_name, direction):
    """DIF SIMD: butterfly then post-twiddle outputs 1..7."""
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

    # Load all 8 (no pre-twiddle)
    for n in range(8):
        lines.append(f'        const {V} x{n}r = LD(&in_re[{n}*K+k]), x{n}i = LD(&in_im[{n}*K+k]);')

    # Same butterfly as notw
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

    # W8 combine into y vars
    lines.append(f'        const {V} y0r = {add}(A0r,B0r), y0i = {add}(A0i,B0i);')
    lines.append(f'        const {V} y4r = {sub}(A0r,B0r), y4i = {sub}(A0i,B0i);')
    if fwd:
        lines.append(f'        const {V} t1r = {mul}(vc, {add}(B1r,B1i)), t1i = {mul}(vc, {sub}(B1i,B1r));')
    else:
        lines.append(f'        const {V} t1r = {mul}(vc, {sub}(B1r,B1i)), t1i = {mul}(vc, {add}(B1r,B1i));')
    lines.append(f'        const {V} y1r = {add}(A1r,t1r), y1i = {add}(A1i,t1i);')
    lines.append(f'        const {V} y5r = {sub}(A1r,t1r), y5i = {sub}(A1i,t1i);')
    if fwd:
        lines.append(f'        const {V} y2r = {add}(A2r,B2i), y2i = {sub}(A2i,B2r);')
        lines.append(f'        const {V} y6r = {sub}(A2r,B2i), y6i = {add}(A2i,B2r);')
    else:
        lines.append(f'        const {V} y2r = {sub}(A2r,B2i), y2i = {add}(A2i,B2r);')
        lines.append(f'        const {V} y6r = {add}(A2r,B2i), y6i = {sub}(A2i,B2r);')
    if fwd:
        lines.append(f'        const {V} t3r = {mul}(vnc, {sub}(B3r,B3i)), t3i = {mul}(vnc, {add}(B3r,B3i));')
    else:
        lines.append(f'        const {V} t3r = {mul}(vnc, {add}(B3r,B3i)), t3i = {mul}(vc, {sub}(B3r,B3i));')
    lines.append(f'        const {V} y3r = {add}(A3r,t3r), y3i = {add}(A3i,t3i);')
    lines.append(f'        const {V} y7r = {sub}(A3r,t3r), y7i = {sub}(A3i,t3i);')

    # Store y0 (no twiddle), post-twiddle y1..y7
    lines.append(f'        ST(&out_re[0*K+k], y0r); ST(&out_im[0*K+k], y0i);')
    for n in range(1, 8):
        lines.append(f'        {{ {V} wr = LD(&tw_re[{n-1}*K+k]), wi = LD(&tw_im[{n-1}*K+k]), tr = y{n}r;')
        if fwd:
            lines.append(f'          ST(&out_re[{n}*K+k], {fnma}(y{n}i,wi,{mul}(tr,wr)));')
            lines.append(f'          ST(&out_im[{n}*K+k], {fma}(tr,wi,{mul}(y{n}i,wr))); }}')
        else:
            lines.append(f'          ST(&out_re[{n}*K+k], {fma}(y{n}i,wi,{mul}(tr,wr)));')
            lines.append(f'          ST(&out_im[{n}*K+k], {fnma}(tr,wi,{mul}(y{n}i,wr))); }}')

    lines.append('    }')
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# LOG3 TWIDDLE VARIANTS
#
# Bases: W^1, W^2, W^4 (3 loads from base_tw table, indices 0-2)
# Derived: W^3=W^1*W^2, W^5=W^1*W^4, W^6=W^2*W^4, W^7=W^3*W^4
# 4 cmuls = 16 FMA-port ops, saves 8 loads
# Table: 3*K*16 bytes vs flat 7*K*16 bytes (2.3x smaller)
# ═══════════════════════════════════════════════════════════════

def _emit_cmul(V, fma, fnma, mul, dst_r, dst_i, ar, ai, br, bi, fwd):
    """Return lines for dst = a*b (fwd) or a*conj(b) (bwd)."""
    if fwd:
        return [
            f'        const {V} {dst_r} = {fnma}({ai},{bi},{mul}({ar},{br}));',
            f'        const {V} {dst_i} = {fma}({ar},{bi},{mul}({ai},{br}));',
        ]
    else:
        return [
            f'        const {V} {dst_r} = {fma}({ai},{bi},{mul}({ar},{br}));',
            f'        const {V} {dst_i} = {fnma}({ar},{bi},{mul}({ai},{br}));',
        ]

def _emit_cmul_inplace(V, fma, fnma, mul, v_r, v_i, wr, wi, fwd):
    """Return lines for v *= (wr+j*wi) (fwd) or v *= conj(wr+j*wi) (bwd)."""
    lines = [f'        {{ const {V} tr = {v_r};']
    if fwd:
        lines.append(f'          {v_r} = {fnma}({v_i},{wi},{mul}(tr,{wr}));')
        lines.append(f'          {v_i} = {fma}(tr,{wi},{mul}({v_i},{wr})); }}')
    else:
        lines.append(f'          {v_r} = {fma}({v_i},{wi},{mul}(tr,{wr}));')
        lines.append(f'          {v_i} = {fnma}(tr,{wi},{mul}({v_i},{wr})); }}')
    return lines


def gen_simd_log3_dit_body(isa_name, direction):
    """DIT log3: load 3 bases, derive 4, apply to inputs, then butterfly."""
    I = ISA[isa_name]
    V, p = I['V'], I['p']
    add, sub, mul = f'{p}_add_pd', f'{p}_sub_pd', f'{p}_mul_pd'
    fma, fnma = f'{p}_fmadd_pd', f'{p}_fnmadd_pd'
    set1 = f'{p}_set1_pd'
    ld = f'{p}_load_pd' if 'load_pd' not in I.get('ld_macro','') else 'LD'
    fwd = direction == 'fwd'

    lines = []
    lines.append(f'    const {V} vc = {set1}({fmt(C)});')
    lines.append(f'    const {V} vnc = {set1}({fmt(-C)});')
    lines.append('')
    lines.append('    for (size_t k = 0; k < K; k += VL) {')

    # Load 3 base twiddles
    lines.append(f'        /* Log3: load W^1, W^2, W^4 bases */')
    lines.append(f'        const {V} b1r = LD(&base_tw_re[0*K+k]), b1i = LD(&base_tw_im[0*K+k]);')
    lines.append(f'        const {V} b2r = LD(&base_tw_re[1*K+k]), b2i = LD(&base_tw_im[1*K+k]);')
    lines.append(f'        const {V} b4r = LD(&base_tw_re[2*K+k]), b4i = LD(&base_tw_im[2*K+k]);')
    lines.append('')

    # Derive W^3 = W^1 * W^2
    lines.append(f'        /* Derive W^3 = W^1 * W^2 */')
    lines.extend(_emit_cmul(V, fma, fnma, mul, 'w3r', 'w3i', 'b1r', 'b1i', 'b2r', 'b2i', True))
    # Derive W^5 = W^1 * W^4
    lines.append(f'        /* Derive W^5 = W^1 * W^4 */')
    lines.extend(_emit_cmul(V, fma, fnma, mul, 'w5r', 'w5i', 'b1r', 'b1i', 'b4r', 'b4i', True))
    # Derive W^6 = W^2 * W^4
    lines.append(f'        /* Derive W^6 = W^2 * W^4 */')
    lines.extend(_emit_cmul(V, fma, fnma, mul, 'w6r', 'w6i', 'b2r', 'b2i', 'b4r', 'b4i', True))
    # Derive W^7 = W^3 * W^4
    lines.append(f'        /* Derive W^7 = W^3 * W^4 */')
    lines.extend(_emit_cmul(V, fma, fnma, mul, 'w7r', 'w7i', 'w3r', 'w3i', 'b4r', 'b4i', True))
    lines.append('')

    # Load x0 (no twiddle), x1..x7 with derived twiddles
    lines.append(f'        const {V} x0r = LD(&in_re[k]), x0i = LD(&in_im[k]);')
    tw_map = {1:('b1r','b1i'), 2:('b2r','b2i'), 3:('w3r','w3i'),
              4:('b4r','b4i'), 5:('w5r','w5i'), 6:('w6r','w6i'), 7:('w7r','w7i')}
    for n in range(1, 8):
        wr, wi = tw_map[n]
        lines.append(f'        {V} x{n}r = LD(&in_re[{n}*K+k]), x{n}i = LD(&in_im[{n}*K+k]);')
        lines.extend(_emit_cmul_inplace(V, fma, fnma, mul, f'x{n}r', f'x{n}i', wr, wi, fwd))
    lines.append('')

    # Same butterfly as notw (reuse from gen_simd_body with twiddled=False logic)
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
    lines.append(f'        ST(&out_re[0*K+k], {add}(A0r,B0r)); ST(&out_im[0*K+k], {add}(A0i,B0i));')
    lines.append(f'        ST(&out_re[4*K+k], {sub}(A0r,B0r)); ST(&out_im[4*K+k], {sub}(A0i,B0i));')
    if fwd:
        lines.append(f'        const {V} t1r = {mul}(vc, {add}(B1r,B1i));')
        lines.append(f'        const {V} t1i = {mul}(vc, {sub}(B1i,B1r));')
    else:
        lines.append(f'        const {V} t1r = {mul}(vc, {sub}(B1r,B1i));')
        lines.append(f'        const {V} t1i = {mul}(vc, {add}(B1r,B1i));')
    lines.append(f'        ST(&out_re[1*K+k], {add}(A1r,t1r)); ST(&out_im[1*K+k], {add}(A1i,t1i));')
    lines.append(f'        ST(&out_re[5*K+k], {sub}(A1r,t1r)); ST(&out_im[5*K+k], {sub}(A1i,t1i));')
    if fwd:
        lines.append(f'        ST(&out_re[2*K+k], {add}(A2r,B2i)); ST(&out_im[2*K+k], {sub}(A2i,B2r));')
        lines.append(f'        ST(&out_re[6*K+k], {sub}(A2r,B2i)); ST(&out_im[6*K+k], {add}(A2i,B2r));')
    else:
        lines.append(f'        ST(&out_re[2*K+k], {sub}(A2r,B2i)); ST(&out_im[2*K+k], {add}(A2i,B2r));')
        lines.append(f'        ST(&out_re[6*K+k], {add}(A2r,B2i)); ST(&out_im[6*K+k], {sub}(A2i,B2r));')
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


def gen_simd_log3_dif_body(isa_name, direction):
    """DIF log3: butterfly first, then derive twiddles and apply to outputs."""
    I = ISA[isa_name]
    V, p = I['V'], I['p']
    add, sub, mul = f'{p}_add_pd', f'{p}_sub_pd', f'{p}_mul_pd'
    fma, fnma = f'{p}_fmadd_pd', f'{p}_fnmadd_pd'
    set1 = f'{p}_set1_pd'
    fwd = direction == 'fwd'

    lines = []
    lines.append(f'    const {V} vc = {set1}({fmt(C)});')
    lines.append(f'    const {V} vnc = {set1}({fmt(-C)});')
    lines.append('')
    lines.append('    for (size_t k = 0; k < K; k += VL) {')

    # Load 3 base twiddles
    lines.append(f'        /* Log3: load W^1, W^2, W^4 bases */')
    lines.append(f'        const {V} b1r = LD(&base_tw_re[0*K+k]), b1i = LD(&base_tw_im[0*K+k]);')
    lines.append(f'        const {V} b2r = LD(&base_tw_re[1*K+k]), b2i = LD(&base_tw_im[1*K+k]);')
    lines.append(f'        const {V} b4r = LD(&base_tw_re[2*K+k]), b4i = LD(&base_tw_im[2*K+k]);')
    lines.append('')

    # Derive all 4 — needed for post-twiddle
    lines.append(f'        /* Derive W^3,5,6,7 */')
    lines.extend(_emit_cmul(V, fma, fnma, mul, 'w3r', 'w3i', 'b1r', 'b1i', 'b2r', 'b2i', True))
    lines.extend(_emit_cmul(V, fma, fnma, mul, 'w5r', 'w5i', 'b1r', 'b1i', 'b4r', 'b4i', True))
    lines.extend(_emit_cmul(V, fma, fnma, mul, 'w6r', 'w6i', 'b2r', 'b2i', 'b4r', 'b4i', True))
    lines.extend(_emit_cmul(V, fma, fnma, mul, 'w7r', 'w7i', 'w3r', 'w3i', 'b4r', 'b4i', True))
    lines.append('')

    # Load all 8 (no pre-twiddle)
    for n in range(8):
        lines.append(f'        const {V} x{n}r = LD(&in_re[{n}*K+k]), x{n}i = LD(&in_im[{n}*K+k]);')

    # Same butterfly as notw
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

    # W8 combine into y vars
    lines.append(f'        {V} y0r = {add}(A0r,B0r), y0i = {add}(A0i,B0i);')
    lines.append(f'        {V} y4r = {sub}(A0r,B0r), y4i = {sub}(A0i,B0i);')
    if fwd:
        lines.append(f'        const {V} wt1r = {mul}(vc, {add}(B1r,B1i)), wt1i = {mul}(vc, {sub}(B1i,B1r));')
    else:
        lines.append(f'        const {V} wt1r = {mul}(vc, {sub}(B1r,B1i)), wt1i = {mul}(vc, {add}(B1r,B1i));')
    lines.append(f'        {V} y1r = {add}(A1r,wt1r), y1i = {add}(A1i,wt1i);')
    lines.append(f'        {V} y5r = {sub}(A1r,wt1r), y5i = {sub}(A1i,wt1i);')
    if fwd:
        lines.append(f'        {V} y2r = {add}(A2r,B2i), y2i = {sub}(A2i,B2r);')
        lines.append(f'        {V} y6r = {sub}(A2r,B2i), y6i = {add}(A2i,B2r);')
    else:
        lines.append(f'        {V} y2r = {sub}(A2r,B2i), y2i = {add}(A2i,B2r);')
        lines.append(f'        {V} y6r = {add}(A2r,B2i), y6i = {sub}(A2i,B2r);')
    if fwd:
        lines.append(f'        const {V} wt3r = {mul}(vnc, {sub}(B3r,B3i)), wt3i = {mul}(vnc, {add}(B3r,B3i));')
    else:
        lines.append(f'        const {V} wt3r = {mul}(vnc, {add}(B3r,B3i)), wt3i = {mul}(vc, {sub}(B3r,B3i));')
    lines.append(f'        {V} y3r = {add}(A3r,wt3r), y3i = {add}(A3i,wt3i);')
    lines.append(f'        {V} y7r = {sub}(A3r,wt3r), y7i = {sub}(A3i,wt3i);')
    lines.append('')

    # Post-twiddle y1..y7, store
    lines.append(f'        /* DIF: post-twiddle outputs 1..7 */')
    lines.append(f'        ST(&out_re[0*K+k], y0r); ST(&out_im[0*K+k], y0i);')
    tw_map = {1:('b1r','b1i'), 2:('b2r','b2i'), 3:('w3r','w3i'),
              4:('b4r','b4i'), 5:('w5r','w5i'), 6:('w6r','w6i'), 7:('w7r','w7i')}
    for n in range(1, 8):
        wr, wi = tw_map[n]
        lines.extend(_emit_cmul_inplace(V, fma, fnma, mul, f'y{n}r', f'y{n}i', wr, wi, fwd))
        lines.append(f'        ST(&out_re[{n}*K+k], y{n}r); ST(&out_im[{n}*K+k], y{n}i);')

    lines.append('    }')
    return '\n'.join(lines)


def gen_stats_header(isa_name):
    is_scalar = isa_name == 'scalar'
    if is_scalar:
        return """ *
 * ── Operation counts per k-step ──
 *
 *   kernel                 add   sub   mul   neg | arith flops |  ld  st  mem
 *   ──────────────────── ───── ───── ───── ───── + ───── ───── + ─── ─── ────
 *   notw fwd/bwd             24    24     4     0 |    52    52 |  16  16   32
 *   dit_tw fwd/bwd           24    24    32     0 |    80    80 |  30  16   46
 *   dif_tw fwd/bwd           24    24    32     0 |    80    80 |  30  16   46"""
    else:
        return f""" *
 * ── Operation counts per k-step ──
 *
 *   kernel                 add   sub   mul   neg  fnma   fma | arith flops |  ld  st  mem
 *   ──────────────────── ───── ───── ───── ───── ───── ───── + ───── ───── + ─── ─── ────
 *   notw fwd/bwd             24    24     4     0     0     0 |    52    52 |  16  16   32
 *   dit_tw fwd/bwd           24    24    11     0     7     7 |    73    87 |  30  16   46
 *   dif_tw fwd/bwd           24    24    11     0     7     7 |    73    87 |  30  16   46"""


def _t2_to_sv(body):
    """Transform a t2 codelet body (loop over K, stride K) to sv (no loop, stride vs).
    
    Mechanical text replacement:
    1. Remove 'for (size_t k = 0; k < K; k += VL) {' line
    2. Remove matching closing '    }'
    3. Replace 'n*K+k' with 'n*vs' in all addressing
    4. Replace '[k]' with '[0]' (element 0 at base pointer)
    5. Dedent body by 4 spaces (was inside for-loop)
    """
    lines = body.split('\n')
    out = []
    depth = 0
    for line in lines:
        stripped = line.strip()
        # Skip the for-loop line
        if stripped.startswith('for (size_t k') and 'k < K' in stripped:
            depth += 1
            continue
        # Skip the matching closing brace
        if depth > 0 and stripped == '}':
            depth -= 1
            continue
        # Transform addressing: n*K+k -> n*vs (digits before *K)
        line = re.sub(r'(\d+)\*K\+k', r'\1*vs', line)
        # Transform [k] -> [0] for element 0 (appears in notw: in_re[k])
        line = re.sub(r'\[k\]', '[0]', line)
        # Dedent by one level (4 spaces)
        if line.startswith('        '):
            line = line[4:]
        out.append(line)
    return '\n'.join(out)


def gen_file(isa_name):
    I = ISA[isa_name]
    guard = f'FFT_RADIX8_{isa_name.upper()}_H'
    is_scalar = isa_name == 'scalar'

    parts = []
    parts.append(f'''/**
 * @file fft_radix8_{isa_name}.h
 * @brief DFT-8 {isa_name.upper()} codelets (notw + DIT tw + DIF tw)
 *
 * DFT-8 = 2×DFT-4 + W8 combine. Only 1 constant: √2/2.
 * Twiddle table: tw_re[(n-1)*K+k] n=1..7 → 7*K doubles per component.
{gen_stats_header(isa_name)}
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

    # DIF tw (fused: butterfly + post-twiddle outputs 1..7)
    for direction in ['fwd', 'bwd']:
        parts.append(f'''
{I["target"]}
static inline void
radix8_tw_dif_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{{''')
        if is_scalar:
            parts.append(gen_scalar_dif_body(direction))
        else:
            parts.append(gen_simd_dif_body(isa_name, direction))
        parts.append('}')

    # Log3 twiddle variants (SIMD only — scalar has no cache pressure benefit)
    if not is_scalar:
        # DIT log3
        for direction in ['fwd', 'bwd']:
            parts.append(f'''
{I["target"]}
static inline void
radix8_tw_log3_dit_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ base_tw_re, const double * __restrict__ base_tw_im,
    size_t K)
{{''')
            parts.append(gen_simd_log3_dit_body(isa_name, direction))
            parts.append('}')

        # DIF log3
        for direction in ['fwd', 'bwd']:
            parts.append(f'''
{I["target"]}
static inline void
radix8_tw_log3_dif_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ base_tw_re, const double * __restrict__ base_tw_im,
    size_t K)
{{''')
            parts.append(gen_simd_log3_dif_body(isa_name, direction))
            parts.append('}')

    if not is_scalar:
        parts.append('\n#undef LD\n#undef ST\n#undef VL')

    # ── sv codelets (SIMD only): no loop, elements at stride vs ──
    if not is_scalar:
        parts.append(f'\n/* === sv codelets: no loop, elements at stride vs === */')
        parts.append(f'/* Executor calls K/{I["VL"]} times, advancing base pointers by {I["VL"]}. */')
        parts.append('')
        parts.append(I['ld_macro'])
        parts.append(f'#define VL {I["VL"]}')

        for tw in [False, True]:
            for direction in ['fwd', 'bwd']:
                kind = 'n1sv' if not tw else 't1sv_dit'
                tw_params = (',\n    const double * __restrict__ tw_re, const double * __restrict__ tw_im'
                            if tw else '')
                # Generate t2 body and transform to sv
                body = gen_simd_body(isa_name, direction, tw)
                sv_body = _t2_to_sv(body)
                parts.append(f'''
{I["target"]}
static inline void
radix8_{kind}_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im{tw_params},
    size_t vs)
{{''')
                parts.append(sv_body)
                parts.append('}')

        # t1sv DIF
        for direction in ['fwd', 'bwd']:
            body = gen_simd_dif_body(isa_name, direction)
            sv_body = _t2_to_sv(body)
            parts.append(f'''
{I["target"]}
static inline void
radix8_t1sv_dif_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t vs)
{{''')
            parts.append(sv_body)
            parts.append('}')

        parts.append('\n#undef LD\n#undef ST\n#undef VL')

    parts.append(f'\n#endif /* {guard} */')
    return '\n'.join(parts)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ISA:
        print(f'Usage: {sys.argv[0]} avx512|avx2|scalar', file=sys.stderr)
        sys.exit(1)
    print(gen_file(sys.argv[1]))
