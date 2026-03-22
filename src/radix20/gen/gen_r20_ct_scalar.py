#!/usr/bin/env python3
"""
gen_r20_ct_scalar.py — DFT-20 scalar fallback CT codelet generator.

4×5 Cooley-Tukey, plain C doubles. k-step=1.
N1 + DIT + DIF, fwd + bwd = 6 functions.

Usage: python3 gen_r20_ct_scalar.py
"""
import math

R, N1, N2 = 20, 4, 5

C_A = "0.55901699437494742"
C_B = "0.95105651629515357"
C_C = "0.58778525229247313"
C_D = "0.25"

def w20(e):
    e = e % 20
    a = 2.0 * math.pi * e / 20.0
    return (math.cos(a), -math.sin(a))


class Emitter:
    def __init__(self): self.L=[]; self.ind=0
    def o(self, s=''): self.L.append('    '*self.ind + s)
    def c(self, s): self.o(f'/* {s} */')
    def b(self): self.L.append('')


def emit_dft4(em, n2, has_tw, dif_tw):
    """Scalar DFT-4 for column n2."""
    inputs = [n2, n2+5, n2+10, n2+15]
    em.o(f'{{ double r0r=in_re[{inputs[0]}*K+k], r0i=in_im[{inputs[0]}*K+k];')
    em.o(f'  double r1r=in_re[{inputs[1]}*K+k], r1i=in_im[{inputs[1]}*K+k];')
    em.o(f'  double r2r=in_re[{inputs[2]}*K+k], r2i=in_im[{inputs[2]}*K+k];')
    em.o(f'  double r3r=in_re[{inputs[3]}*K+k], r3i=in_im[{inputs[3]}*K+k];')

    if has_tw and not dif_tw:
        for j, m in enumerate(inputs):
            if m > 0:
                em.o(f'  {{ double wr=tw_re[{m-1}*K+k], wi=tw_im[{m-1}*K+k], t=r{j}r;')
                em.o(f'    r{j}r = t*wr - r{j}i*wi; r{j}i = t*wi + r{j}i*wr; }}')

    # s=r0+r2, d=r0-r2, t=r1+r3, u=r1-r3
    em.o(f'  double sr=r0r+r2r, si=r0i+r2i, dr=r0r-r2r, di=r0i-r2i;')
    em.o(f'  double tr=r1r+r3r, ti=r1i+r3i, ur=r1r-r3r, ui=r1i-r3i;')
    # fwd: [0]=s+t, [1]=d-j*u=(d+ui,di-ur), [2]=s-t, [3]=d+j*u=(d-ui,di+ur)
    # (forward ×(-j): (ui, -ur))
    for k1, (rexpr, iexpr) in enumerate([
        ('sr+tr', 'si+ti'),
        ('dr+ui', 'di-ur'),
        ('sr-tr', 'si-ti'),
        ('dr-ui', 'di+ur'),
    ]):
        slot = k1 * 5 + n2
        em.o(f'  sp_re[{slot}] = {rexpr}; sp_im[{slot}] = {iexpr};')
    em.o(f'}}')


def emit_internal_twiddles(em):
    for k1 in range(1, N1):
        for n2 in range(1, N2):
            e = (k1 * n2) % R
            if e == 0:
                continue
            wr, wi = w20(e)
            slot = k1 * 5 + n2
            em.o(f'{{ double t=sp_re[{slot}], wr={wr:.20e}, wi={wi:.20e};')
            em.o(f'  sp_re[{slot}] = t*wr - sp_im[{slot}]*wi;')
            em.o(f'  sp_im[{slot}] = t*wi + sp_im[{slot}]*wr; }}')


def emit_dft5(em, k1, dif_tw):
    """Scalar DFT-5 for row k1."""
    em.o(f'{{')
    for n2 in range(5):
        slot = k1 * 5 + n2
        em.o(f'double x{n2}r=sp_re[{slot}], x{n2}i=sp_im[{slot}];')

    em.o(f'{{ double s1r=x1r+x4r, s1i=x1i+x4i, d1r=x1r-x4r, d1i=x1i-x4i;')
    em.o(f'  double s2r=x2r+x3r, s2i=x2i+x3i, d2r=x2r-x3r, d2i=x2i-x3i;')
    em.o(f'  double ssr=s1r+s2r, ssi=s1i+s2i;')
    em.o(f'  double t0r=x0r-{C_D}*ssr, t0i=x0i-{C_D}*ssi;')
    em.o(f'  double t1r={C_A}*(s1r-s2r), t1i={C_A}*(s1i-s2i);')
    em.o(f'  double p1r=t0r+t1r, p1i=t0i+t1i, p2r=t0r-t1r, p2i=t0i-t1i;')
    em.o(f'  double Ur={C_B}*d1r+{C_C}*d2r, Ui={C_B}*d1i+{C_C}*d2i;')
    em.o(f'  double Vr={C_B}*d2r-{C_C}*d1r, Vi={C_B}*d2i-{C_C}*d1i;')

    # fwd: y1=p1-j*U=(p1r+Ui,p1i-Ur), y4=p1+j*U, y2=p2+j*V, y3=p2-j*V
    outputs = [
        (0, 'x0r+ssr', 'x0i+ssi'),
        (1, 'p1r+Ui',  'p1i-Ur'),
        (2, 'p2r-Vi',  'p2i+Vr'),
        (3, 'p2r+Vi',  'p2i-Vr'),
        (4, 'p1r-Ui',  'p1i+Ur'),
    ]

    for k2, rexpr, iexpr in outputs:
        m = k1 + N1 * k2
        if dif_tw and m > 0:
            em.o(f'  {{ double yr={rexpr}, yi={iexpr};')
            em.o(f'    double wr=tw_re[{m-1}*K+k], wi=tw_im[{m-1}*K+k], t=yr;')
            em.o(f'    out_re[{m}*K+k] = t*wr - yi*wi;')
            em.o(f'    out_im[{m}*K+k] = t*wi + yi*wr; }}')
        else:
            em.o(f'  out_re[{m}*K+k] = {rexpr}; out_im[{m}*K+k] = {iexpr};')
    em.o(f'}}')
    em.o(f'}}')


def gen_kernel(mode, direction):
    fwd = direction == 'fwd'
    has_tw = mode in ('dit', 'dif')
    dif_tw = mode == 'dif'
    em = Emitter()

    tag = {'n1': 'n1', 'dit': 'tw_dit', 'dif': 'tw_dif'}[mode]
    fname = f'radix20_ct_{tag}_kernel_{direction}_scalar'

    em.o(f'static void {fname}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    if has_tw:
        em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'double sp_re[20], sp_im[20];')
    em.b()

    if not fwd:
        # Backward: swap re<->im on everything and call forward
        em.o(f'/* Backward: pointer-swap trick */')
        args = 'in_im, in_re, out_im, out_re, '
        if has_tw:
            args += 'tw_re, tw_im, '
        args += 'K'
        fwd_fname = f'radix20_ct_{tag}_kernel_fwd_scalar'
        em.o(f'{fwd_fname}({args});')
        em.ind -= 1
        em.o(f'}}')
        em.b()
        return em.L

    em.o(f'for (size_t k = 0; k < K; k++) {{')
    em.ind += 1

    # Pass 1: 5× DFT-4
    for n2 in range(N2):
        emit_dft4(em, n2, has_tw, dif_tw)
    em.b()

    # Internal twiddles
    emit_internal_twiddles(em)
    em.b()

    # Pass 2: 4× DFT-5
    for k1 in range(N1):
        emit_dft5(em, k1, dif_tw)
    em.b()

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_file():
    guard = 'FFT_RADIX20_SCALAR_CT_H'
    L = ['/**',
         ' * @file fft_radix20_scalar_ct.h',
         ' * @brief DFT-20 scalar fallback — 4×5 CT',
         ' * N1 + DIT + DIF, fwd + bwd = 6 functions.',
         ' * Generated by gen_r20_ct_scalar.py',
         ' */', '',
         f'#ifndef {guard}', f'#define {guard}',
         '#include <stddef.h>', '']

    for mode in ('n1', 'dit', 'dif'):
        for d in ('fwd', 'bwd'):
            L.extend(gen_kernel(mode, d))

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    lines = gen_file()
    print('\n'.join(lines))
    import sys
    print(f'/* {len(lines)} lines */', file=sys.stderr)
